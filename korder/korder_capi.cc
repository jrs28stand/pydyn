/*
 * This file is part of pydyn.
 *
 * It is adapted from mex/sources/k_order_perturbation/k_order_perturbation.cc in Dynare 7.2
 * (Copyright (C) 2008-2026 Dynare Team, https://www.dynare.org), modified in 2026, and is meant
 * to be compiled together with Dynare's libkorder sources.
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see
 * the GNU General Public License (file LICENSE) for details.
 */
/* C interface to Dynare's k-order perturbation library (libkorder), for use from Python.
   Replaces the MEX gateway mex/sources/k_order_perturbation/k_order_perturbation.cc:
   model derivatives are passed in precomputed instead of being obtained by calling MATLAB. */
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "approximation.hh"
#include "dynamic_abstract_class.hh"
#include "dynare_exception.hh"
#include "exception.hh"
#include "k_ord_dynare.hh"
#include "kord_exception.hh"
#include "SylvException.hh"
#include "thread_pool.hh"
#include "tl_exception.hh"

namespace
{
struct SparseDeriv
{
  int nnz;
  std::vector<int> rows;  // nnz, 0-based equation index
  std::vector<int> cols;  // nnz*order, row-major, 0-based Dynare column index
  std::vector<double> vals;
};

class DynamicModelPrecomputed : public DynamicModelAC
{
  std::vector<SparseDeriv> derivs;  // derivs[o-1] is the order-o derivative

public:
  DynamicModelPrecomputed(int order_arg, std::vector<SparseDeriv> derivs_arg) :
      DynamicModelAC {order_arg, nullptr, nullptr, nullptr, {}}, derivs {std::move(derivs_arg)}
  {
  }
  void
  eval([[maybe_unused]] const Vector& y, [[maybe_unused]] const Vector& x,
       [[maybe_unused]] const Vector& params, [[maybe_unused]] const Vector& ySteady,
       Vector& residual, const std::map<int, int>& dynToDynpp,
       TensorContainer<FSSparseTensor>& derivatives) noexcept(false) override
  {
    residual.zeros();
    for (int o {1}; o <= order; o++)
      {
        const SparseDeriv& d {derivs[o - 1]};
        IntSequence s(o, 0);
        auto tensor = std::make_unique<FSSparseTensor>(o, dynToDynpp.size(), residual.length());
        for (int k {0}; k < d.nnz; k++)
          {
            for (int i {0}; i < o; i++)
              s[i] = dynToDynpp.at(d.cols[k * o + i]);
            s.sort();
            tensor->insert(s, d.rows[k], d.vals[k]);
          }
        derivatives.insert(std::move(tensor));
      }
  }
};

struct Result
{
  std::map<std::string, std::vector<double>> data;
  std::map<std::string, std::pair<int, int>> dims;
  std::string error;
};

void
store(Result& r, const std::string& name, int nrows, int ncols, const double* base)
{
  r.dims[name] = {nrows, ncols};
  r.data[name].assign(base, base + static_cast<size_t>(nrows) * ncols);
}

void
store_deriv(Result& r, const Symmetry& sym, const FGSContainer& derivs, const std::string& name)
{
  auto x = derivs.get(sym).unfold();
  store(r, name, x->nrows(), x->ncols(), x->getData().base());
}
}

extern "C"
{
  /* Returns an opaque handle (never null). Check pydyn_korder_error() before reading results.
     cols[] for order o holds nnz[o-1]*o entries; rows/cols/vals are concatenated over orders. */
  void*
  pydyn_korder_solve(int order, int nstat, int npred, int nboth, int nfwrd, int nexo, int npar,
                     const double* ys, const double* sigma_e, const double* params,
                     const int* dr_order, const int* nnz, const int* rows, const int* cols,
                     const double* vals, double qz_criterium, int pruning, int num_threads)
  {
    auto res = new Result;
    try
      {
        const int nendo {nstat + npred + nboth + nfwrd};
        std::vector<SparseDeriv> derivs;
        for (int o {1}; o <= order; o++)
          {
            SparseDeriv d;
            d.nnz = nnz[o - 1];
            d.rows.assign(rows, rows + d.nnz);
            d.cols.assign(cols, cols + static_cast<size_t>(d.nnz) * o);
            d.vals.assign(vals, vals + d.nnz);
            rows += d.nnz;
            cols += static_cast<size_t>(d.nnz) * o;
            vals += d.nnz;
            derivs.push_back(std::move(d));
          }
        Vector ySteady(nendo), modParams(std::max(npar, 1));
        std::copy_n(ys, nendo, ySteady.base());
        modParams.zeros();
        std::copy_n(params, npar, modParams.base());
        TwoDMatrix vCov(nexo, nexo);
        std::copy_n(sigma_e, nexo * nexo, vCov.getData().base());
        std::vector<int> drOrder(dr_order, dr_order + nendo);
        std::vector<std::string> endoNames, exoNames;
        for (int i {0}; i < nendo; i++)
          endoNames.push_back("y" + std::to_string(i));
        for (int i {0}; i < nexo; i++)
          exoNames.push_back("u" + std::to_string(i));

        Journal journal;
        TLStatic::init(order, nstat + 2 * npred + 3 * nboth + 2 * nfwrd + nexo);
        thread_pool::initialize(std::max(num_threads, 1));
        const int nSteps {0};
        KordpDynare dynare(endoNames, exoNames, nexo, npar, ySteady, vCov, modParams, nstat, npred,
                           nfwrd, nboth, nSteps, order, journal,
                           std::make_unique<DynamicModelPrecomputed>(order, std::move(derivs)),
                           drOrder);
        Approximation app(dynare, journal, nSteps, false, pruning != 0, qz_criterium);
        app.walkStochSteady();

        const FoldDecisionRule& fdr = app.getFoldDecisionRule();
        for (int i {0}; i <= order; i++)
          {
            const FFSTensor& t = fdr.get(Symmetry {i});
            store(*res, "g_" + std::to_string(i), t.nrows(), t.ncols(), t.getData().base());
          }
        if (pruning)
          {
            const UnfoldDecisionRule& udr = app.getUnfoldDecisionRulePruning();
            for (int i {0}; i <= order; i++)
              {
                const UFSTensor& t = udr.get(Symmetry {i});
                store(*res, "pruning_g_" + std::to_string(i), t.nrows(), t.ncols(),
                      t.getData().base());
              }
          }
        const FGSContainer& d = app.get_rule_ders();
        store_deriv(*res, Symmetry {1, 0, 0, 0}, d, "gy");
        store_deriv(*res, Symmetry {0, 1, 0, 0}, d, "gu");
        if (order >= 2)
          {
            store_deriv(*res, Symmetry {2, 0, 0, 0}, d, "gyy");
            store_deriv(*res, Symmetry {0, 2, 0, 0}, d, "guu");
            store_deriv(*res, Symmetry {1, 1, 0, 0}, d, "gyu");
            store_deriv(*res, Symmetry {0, 0, 0, 2}, d, "gss");
          }
        if (order >= 3)
          {
            store_deriv(*res, Symmetry {3, 0, 0, 0}, d, "gyyy");
            store_deriv(*res, Symmetry {0, 3, 0, 0}, d, "guuu");
            store_deriv(*res, Symmetry {2, 1, 0, 0}, d, "gyyu");
            store_deriv(*res, Symmetry {1, 2, 0, 0}, d, "gyuu");
            store_deriv(*res, Symmetry {1, 0, 0, 2}, d, "gyss");
            store_deriv(*res, Symmetry {0, 1, 0, 2}, d, "guss");
          }
      }
    catch (const KordException& e)
      {
        res->error = "Kord exception: " + e.get_message();
      }
    catch (const TLException& e)
      {
        res->error = "tensor library exception";
      }
    catch (SylvException& e)
      {
        res->error = "Sylvester solver exception";
      }
    catch (const DynareException& e)
      {
        res->error = "KordDynare exception: " + e.message();
      }
    catch (const ogu::Exception& e)
      {
        res->error = "general exception: " + e.message();
      }
    catch (const std::exception& e)
      {
        res->error = std::string {"C++ exception: "} + e.what();
      }
    catch (...)
      {
        res->error = "unknown C++ exception";
      }
    if (res->error.empty() && res->data.empty())
      res->error = "no output";
    return res;
  }

  const char*
  pydyn_korder_error(void* h)
  {
    return static_cast<Result*>(h)->error.c_str();
  }

  int
  pydyn_korder_dims(void* h, const char* name, int* nrows, int* ncols)
  {
    auto r = static_cast<Result*>(h);
    auto it = r->dims.find(name);
    if (it == r->dims.end())
      return 0;
    *nrows = it->second.first;
    *ncols = it->second.second;
    return 1;
  }

  void
  pydyn_korder_copy(void* h, const char* name, double* out)
  {
    auto r = static_cast<Result*>(h);
    const auto& v = r->data.at(name);
    std::copy(v.begin(), v.end(), out);
  }

  void
  pydyn_korder_free(void* h)
  {
    delete static_cast<Result*>(h);
  }
}
