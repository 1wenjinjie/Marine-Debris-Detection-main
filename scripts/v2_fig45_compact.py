"""Read-only lossless compaction of fig45_20261015 for transport (<2 MB read limit).
Fig4: PR curve as integers (system, rank, ctp). recall=ctp/n_gt, precision=ctp/rank.
Kept ranks: first, every TP rank, the rank just before each TP, last. Between kept
ranks recall is constant and precision decreases monotonically, so the plotted
polyline is identical. Every kept row is checked against fig4_pr.csv.
Fig5: rgb_uint8_* and labels_* only; rgb_uint8 is re-verified against the shared
stretch applied to rgb_reflectance_* here. SHA-256 of each array is recorded.
Writes only a NEW directory; refuses an existing one.
"""
import base64, csv, hashlib, json, zlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'runs_v2/_tmp_eval/fig45_20261015'
OUT = ROOT/'runs_v2/_tmp_eval/fig45_20261015_compact'

def main():
    if OUT.exists(): raise SystemExit('Refusing existing output directory.')
    OUT.mkdir(parents=True)
    with (SRC/'fig4_pr.csv').open(encoding='utf-8', newline='') as f: rows = list(csv.DictReader(f))
    by = {}
    for r in rows: by.setdefault(r['system'], []).append(r)
    out = []; checks = {'fig4_pr_sha256': hashlib.sha256((SRC/'fig4_pr.csv').read_bytes()).hexdigest(), 'systems': {}}
    for name, rr in by.items():
        n = int(rr[0]['n_gt']); rank = np.array([int(r['rank']) for r in rr])
        rec = np.array([float(r['recall']) for r in rr]); pre = np.array([float(r['precision']) for r in rr])
        assert np.array_equal(rank, np.arange(1, len(rr)+1))
        ctp = np.rint(rec*n).astype(int)
        assert np.allclose(ctp/n, rec, atol=1e-12, rtol=0) and np.allclose(ctp/rank, pre, atol=1e-12, rtol=0)
        tp = np.flatnonzero(np.diff(np.r_[0, ctp]) > 0)
        keep = sorted(set([0, len(rr)-1]) | set(tp.tolist()) | set((tp-1)[tp > 0].tolist()))
        for k in keep: out.append({'system': name, 'rank': int(rank[k]), 'ctp': int(ctp[k]), 'n_gt': n})
        checks['systems'][name] = {'n_rows_full': len(rr), 'n_kept': len(keep), 'final_ctp': int(ctp[-1])}
    with (OUT/'fig4_pr_vertices.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    man = json.loads((SRC/'fig45_manifest.json').read_text(encoding='utf-8'))
    st = man['stretch']; lo = np.array(st['lo']); hi = np.array(st['hi'])
    a = np.load(SRC/'fig5_arrays.npz', allow_pickle=False); lines = []; checks['arrays'] = {}
    for p in man['fig5_panels']:
        i = p['panel']; rgb = a[f'rgb_uint8_{i}']
        chk = np.uint8(np.rint(np.clip((a[f'rgb_reflectance_{i}']-lo)/(hi-lo), 0, 1)*255))
        assert np.array_equal(rgb, chk), 'rgb_uint8 differs from shared stretch'
        for nm in (f'rgb_uint8_{i}', f'labels_{i}'):
            x = np.ascontiguousarray(a[nm]); checks['arrays'][nm] = hashlib.sha256(x.tobytes()).hexdigest()
            b = base64.b64encode(zlib.compress(x.tobytes(), 9)).decode('ascii')
            lines.append(f'ARRAY {nm} {x.dtype} {",".join(map(str, x.shape))} {len(b)}')
            lines.extend(b[j:j+3000] for j in range(0, len(b), 3000)); lines.append('END')
    checks['stretch_verified_from_reflectance'] = True
    (OUT/'fig5_uint8_transport.txt').write_text('\n'.join(lines)+'\n', encoding='ascii')
    (OUT/'compact_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print('Complete:', OUT, {k: v['n_kept'] for k, v in checks['systems'].items()}, flush=True)

if __name__ == '__main__': main()
