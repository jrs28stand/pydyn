/* Minimal stand-in for the MATLAB/Octave MEX header, so Dynare's k-order library can be
   built as an ordinary shared library. None of the mx* functions are reachable from the
   pydyn entry point; they abort if called. */
#ifndef PYDYN_STUB_MEX_H
#define PYDYN_STUB_MEX_H
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct mxArray_tag;
typedef struct mxArray_tag mxArray;
typedef size_t mwSize;
typedef size_t mwIndex;
typedef int32_t int32_T;
typedef enum { mxREAL = 0, mxCOMPLEX = 1 } mxComplexity;
typedef enum { mxDOUBLE_CLASS = 6, mxINT32_CLASS = 12 } mxClassID;

[[noreturn]] inline void pydyn_no_mex(const char* f)
{
  std::fprintf(stderr, "pydyn: MEX function %s called outside MATLAB/Octave\n", f);
  std::abort();
}
inline int mexPrintf(const char* fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  int r = std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  return r;
}
inline int mexAtExit(void (*f)()) { return std::atexit(f); }
[[noreturn]] inline void mexErrMsgTxt(const char* s) { std::fprintf(stderr, "%s\n", s); std::abort(); }
inline double* mxGetDoubles(const mxArray*) { pydyn_no_mex("mxGetDoubles"); }
inline int32_T* mxGetInt32s(const mxArray*) { pydyn_no_mex("mxGetInt32s"); }
inline size_t mxGetM(const mxArray*) { pydyn_no_mex("mxGetM"); }
inline size_t mxGetN(const mxArray*) { pydyn_no_mex("mxGetN"); }
inline size_t mxGetNumberOfElements(const mxArray*) { pydyn_no_mex("mxGetNumberOfElements"); }
inline bool mxIsDouble(const mxArray*) { pydyn_no_mex("mxIsDouble"); }
inline bool mxIsComplex(const mxArray*) { pydyn_no_mex("mxIsComplex"); }
inline bool mxIsSparse(const mxArray*) { pydyn_no_mex("mxIsSparse"); }
inline mxArray* mxCreateDoubleScalar(double) { pydyn_no_mex("mxCreateDoubleScalar"); }
inline mxArray* mxCreateDoubleMatrix(mwSize, mwSize, mxComplexity) { pydyn_no_mex("mxCreateDoubleMatrix"); }
inline mxArray* mxCreateNumericMatrix(mwSize, mwSize, mxClassID, mxComplexity) { pydyn_no_mex("mxCreateNumericMatrix"); }
inline mxArray* mxCreateString(const char*) { pydyn_no_mex("mxCreateString"); }
inline mxArray* mxCreateStructArray(mwSize, const mwSize*, int, const char**) { pydyn_no_mex("mxCreateStructArray"); }
inline void mxSetFieldByNumber(mxArray*, mwIndex, int, mxArray*) { pydyn_no_mex("mxSetFieldByNumber"); }
#endif
