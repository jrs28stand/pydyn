import json, collections, sys
import numpy as np
R = [json.loads(l) for l in open('soak/approx/results/E.jsonl')]
meths = ('lcp', 'regime', 'hybrid')
tab = collections.defaultdict(lambda: collections.defaultdict(collections.Counter)); tt = collections.defaultdict(lambda: collections.defaultdict(list))
mx = collections.defaultdict(lambda: collections.defaultdict(float)); big = collections.defaultdict(list)
for r in R:
    for v in r['v']:
        o = v['occbin']
        for m in meths:
            b = v[m]; c = tab[r['model']][m]
            if o['lp'] is not None and b['lp'] is not None:
                d = abs(o['lp'] - b['lp']); scale = max(1.0, abs(o['lp']))
                c['pairs'] += 1; tt[r['model']][m].append(b['t'])
                tt[r['model']]['occ_' + m].append(o['t'])
                if d > 1e-9 * scale and d > 1e-9:
                    c['diff'] += 1
                    if d > 0.12: c['>0.12'] += 1
                    if d > 0.4: c['>0.4'] += 1
                    mx[r['model']][m] = max(mx[r['model']][m], d)
                    if m == 'lcp' and d > 0.12: big[r['model']].append((r['i'], v['th'], d))
            elif o['lp'] is None and b['lp'] is None: c['both_fail'] += 1
            elif o['lp'] is None: c['only_method_finite'] += 1
            else: c['only_occbin_finite'] += 1
names = sorted(tab)
print("model | method | pairs | diff>1e-9 | >0.12 | >0.4 | max | occ-only-finite | method-only-finite | both fail | median t method / occbin (s)")
for n in names:
    for m in meths:
        c = tab[n][m]; p = max(c['pairs'], 1)
        print(f"{n} | {m} | {c['pairs']} | {c['diff']} ({100*c['diff']/p:.1f}%) | {c['>0.12']} | {c['>0.4']} | {mx[n][m]:.3g} | {c['only_occbin_finite']} | {c['only_method_finite']} | {c['both_fail']} | "
              f"{np.median(tt[n][m]) if tt[n][m] else float('nan'):.3f} / {np.median(tt[n]['occ_'+m]) if tt[n]['occ_'+m] else float('nan'):.3f}")
