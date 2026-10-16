"""CPU-only, read-only source extraction for MPB Figs 4/5. No model inference.

FIXED SELECTION BEFORE IMAGE VIEWING (do not change after first extraction):
C6 seed 0, IoU .5, per-fold top-nGT from v2_error_analysis.py.
(a) highest-score TP; (b) first patch ID among missed 4-7-pixel GT,
    ties by component ID; (c-f) highest-score FP in other_debris_pixels,
    algae_organic, ship, foam_waves_wakes. Score ties: patch ID then index.
An absent category stays empty. The top other_debris_pixels example is NOT
silently replaced if its debris is not subthreshold; that conflict is logged.
Six 64x64 real MARIDA B4/B3/B2 crops share per-band 2-98 percentile stretch.
Only writes to a NEW directory under runs_v2/_tmp_eval; existing outputs refused.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, json, sys, zlib
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'scripts/mpb')]
import cv_pipeline as cv
import audit_toolkit as at
import v2_error_analysis as ea

RULES = __doc__
EA = ROOT/'runs_v2/_tmp_eval/error_analysis'
RAW = ROOT/'data/raw/MARIDA/patches'
FDI = ROOT/'runs_v2/_tmp_eval/fdi_baseline'
STRATA = ('area_4-7', 'area_8-15', 'area_ge16')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_csv(p, data):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def encode(name, a, lines):
    a = np.ascontiguousarray(a)
    b = base64.b64encode(zlib.compress(a.tobytes(),9)).decode('ascii')
    lines.append(f'ARRAY {name} {a.dtype} {",".join(map(str,a.shape))} {len(b)}')
    lines.extend(b[i:i+3000] for i in range(0,len(b),3000)); lines.append('END')

def debris_component_sizes(cl, box):
    """All 4-connected class-1 component sizes touching the selected FP box."""
    x0,y0 = np.floor(np.maximum(0,box[:2]/2)).astype(int)
    x1,y1 = np.ceil(np.minimum(256,box[2:]/2)).astype(int)
    x0,y0 = min(x0,255), min(y0,255)
    x1,y1 = max(x1,x0+1), max(y1,y0+1)
    todo = set(map(tuple,np.argwhere(cl[y0:y1,x0:x1] == 1)+[y0,x0]))
    sizes=[]
    while todo:
        seed=min(todo); seen={seed}; stack=[seed]
        while stack:
            y,x=stack.pop()
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<256 and 0<=xx<256 and cl[yy,xx]==1 and (yy,xx) not in seen:
                    seen.add((yy,xx)); stack.append((yy,xx))
        sizes.append(len(seen)); todo-=seen
    return sorted(sizes)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='runs_v2/_tmp_eval/fig45_20261015')
    args=ap.parse_args(); out=(ROOT/args.out).resolve()
    out.relative_to((ROOT/'runs_v2/_tmp_eval').resolve())
    if out.exists(): raise SystemExit('Refusing existing output directory; selection cannot be replaced.')
    out.mkdir(parents=True)
    manifest={'selection_rule':RULES, 'inputs':{}, 'checks':{}, 'warnings':[]}
    def record(p):
        manifest['inputs'][p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (Path(__file__),ROOT/'scripts/mpb/cv_pipeline.py',ROOT/'scripts/mpb/audit_toolkit.py',
              ROOT/'scripts/v2_error_analysis.py',cv.AUDIT_DIR/'marine_debris_components.csv',
              cv.AUDIT_DIR/'patch_inventory.csv', ROOT/'runs_cv/cv_stat_results.json',
              FDI/'fdi_oof.json',FDI/'summary.json',FDI/'cv9_fold_assignment_actual.csv',
              EA/'miss_strata.csv',EA/'miss_strata_over_seeds.csv',
              EA/'operating_point_by_fold.csv',EA/'fp_pixel_category.csv'): record(p)
    print('[1/5] Loading existing validated OOF only; no inference or training.',flush=True)
    byrun,_=cv.load_validated_oof((0,1,2),smoke=False)
    for k in range(9):
        for ch in ('c6','c7'):
            for s in range(3): record(cv.OOF_DIR/f'fold{k}_{ch}_s{s}_oof.json')
    scene_fold={s:k for k,ss in enumerate(cv.FOLDS) for s in ss}
    actual={r['scene']:int(r['fold']) for r in rows(FDI/'cv9_fold_assignment_actual.csv')}
    assert actual==scene_fold, 'Actual fold assignment differs from cv_pipeline'
    gt={e['stem']:np.asarray(e['ground_truth'],float).reshape(-1,4)
        for k in range(9) for e in byrun[(k,'c6',0)]}
    scene={e['stem']:e['scene'] for k in range(9) for e in byrun[(k,'c6',0)]}
    stems=sorted(gt); assert len(stems)==1381 and sum(map(len,gt.values()))==212
    systems={}
    for ch in ('c6','c7'):
        for s in range(3):
            per={}
            for k in range(9):
                for e in byrun[(k,ch,s)]:
                    p=np.asarray(e['predictions'],float).reshape(-1,6)
                    per[e['stem']]=p[p[:,5]==0]
            systems[f'{ch}_s{s}']=per
    per={}
    for e in json.loads((FDI/'fdi_oof.json').read_text(encoding='utf-8')):
        st=e['stem']; assert st not in per and e['scene']==scene[st]
        g=np.asarray(e['ground_truth_xyxy'],float).reshape(-1,4)
        assert g.shape==gt[st].shape and np.allclose(g,gt[st],atol=1e-3)
        p=np.asarray(e['predictions_xyxy_conf_cls'],float).reshape(-1,6)
        per[st]=p[p[:,5]==0]
    assert sorted(per)==stems
    systems['fdi']=per
    ref=json.loads((ROOT/'runs_cv/cv_stat_results.json').read_text(encoding='utf-8'))
    fref=json.loads((FDI/'summary.json').read_text(encoding='utf-8'))
    opref={(r['system'],int(r['fold'])):r for r in rows(EA/'operating_point_by_fold.csv')}
    state={}; curves=[]; ops=[]; checks={}
    print('[2/5] Exporting exact ranked PR points and fold-specific operating points.',flush=True)
    for name,per in systems.items():
        rr={}
        for st in stems:
            p=per[st]; tp,gp=ea.greedy_match(p,gt[st],.5)
            assert np.array_equal(tp,cv.match_patch_tp(p,gt[st]))
            rr[st]={'p':p,'tp':tp,'gp':gp,'sel':np.zeros(len(p),bool)}
        owner_all=[]; ns=nt=0
        for k in range(9):
            sk=[st for st in stems if scene_fold[scene[st]]==k]
            own=[(st,j) for st in sk for j in range(len(per[st]))]
            n=sum(len(gt[st]) for st in sk)
            order=np.argsort([-per[st][j,4] for st,j in own],kind='stable')[:n]
            for ix in order:
                st,j=own[ix]; rr[st]['sel'][j]=True
            tp=int(sum(rr[own[ix][0]]['tp'][own[ix][1]] for ix in order))
            r=opref[name,k]
            assert (len(own),len(order),tp)==(int(r['n_predictions_fold']),int(r['n_selected']),int(r['n_tp_iou50']))
            ns+=len(order); nt+=tp; owner_all+=own
        # Scene-ledger order matches the AP implementation, including equal-score ties.
        entries=([e for k in range(9) for e in byrun[(k,name[:2],int(name[-1]))]]
                 if name!='fdi' else [{'scene':scene[st],'predictions':per[st], 'ground_truth':gt[st]} for st in stems])
        led=cv.entries_to_scene_ledger(entries); sl=sorted(led)
        sc,t,n=at._pool(led,sl); order=np.argsort(-sc,kind='stable')
        scores=sc[order]; ctp=np.cumsum(t[order]); rec=ctp/n
        pre=ctp/np.arange(1,len(scores)+1); env=np.maximum.accumulate(pre[::-1])[::-1]
        for j in range(len(scores)):
            curves.append({'system':name,'rank':j+1,'score':float(scores[j]),'recall':float(rec[j]),
                           'precision':float(pre[j]),'precision_envelope':float(env[j]),'n_gt':n})
        value=at.pooled_ap(led,sl)
        expected=fref['fdi_results']['iou_0.50']['pooled_AP'] if name=='fdi' else ref['channels'][name[:2]]['per_seed_AP'][int(name[-1])]
        assert np.isclose(value,expected,atol=1e-4,rtol=0), (name,value,expected)
        checks[name]={'AP50':value,'reference_AP50':expected,'n_selected':ns,'n_tp':nt}
        ops.append({'system':name,'n_gt':212,'n_selected':ns,'n_tp':nt,'recall':nt/212,
                    'precision':nt/ns,'definition':'pool of per-fold top-nGT sets; not a global-score threshold'})
        state[name]=rr
    write_csv(out/'fig4_pr.csv',curves); write_csv(out/'fig4_operating_points.csv',ops)
    for fn in ('miss_strata.csv','miss_strata_over_seeds.csv','operating_point_by_fold.csv'):
        (out/fn).write_bytes((EA/fn).read_bytes())
    rr=state['c6_s0']; components={}
    for r in rows(cv.AUDIT_DIR/'marine_debris_components.csv'):
        if int(r['original_class'])==1 and int(r['area_pixels'])>=4:
            b=ea.padded_box_512(*(int(r[c]) for c in ('x_min','y_min','x_max','y_max')))
            components.setdefault(r['stem'],[]).append((np.array(b),r))
    candidates={'TP':[],'small_miss':[],'other_debris_pixels':[],
                'algae_organic':[],'ship':[],'foam_waves_wakes':[]}
    for st in stems:
        p=rr[st]['p']; g=gt[st]
        for gi,b in enumerate(g):
            comp=[r for bb,r in components[st] if np.allclose(bb,b,atol=1e-3)]
            assert len(comp)==1
            r=comp[0]; j=int(rr[st]['gp'][gi]); hit=j>=0 and rr[st]['sel'][j]
            if 4<=int(r['area_pixels'])<=7 and not hit:
                candidates['small_miss'].append({'stem':st,'gt_index':gi,'component_id':r['component_id'],
                    'area_pixels':int(r['area_pixels']),'box_512':b.tolist(),'score':None})
        ids=np.flatnonzero(rr[st]['sel'])
        if not len(ids): continue
        clpath=RAW/scene[st]/f'{st}_cl.tif'
        with rasterio.open(clpath) as src: cl=src.read(1).astype(np.uint8)
        for j in ids:
            d={'stem':st,'pred_index':int(j),'box_512':p[j,:4].tolist(),'score':float(p[j,4])}
            if rr[st]['tp'][j]:
                d['gt_index']=int(np.flatnonzero(rr[st]['gp']==j)[0]); candidates['TP'].append(d)
            else:
                mi=float(ea.pairwise_iou(p[j:j+1,:4],g).max()) if len(g) else 0.
                cat,_=ea.pixel_category(ea.box_pixels(cl,p[j,:4]),mi)
                if cat in candidates:
                    if cat=='other_debris_pixels':
                        sizes=debris_component_sizes(cl,p[j,:4]); d['touching_debris_component_areas']=sizes
                        d['all_debris_subthreshold']=bool(sizes) and max(sizes)<4
                    candidates[cat].append(d)
    # Cross-check FP candidate counts against the existing error analysis.
    for r in rows(EA/'fp_pixel_category.csv'):
        if r['system']=='c6_s0' and r['fp_def']=='top_ngt' and r['category'] in candidates:
            assert len(candidates[r['category']])==int(r['n_fp'])
    chosen=[]
    for cat,cc in candidates.items():
        cc.sort(key=(lambda d:(d['stem'],d['component_id'])) if cat=='small_miss'
                else (lambda d:(-d['score'],d['stem'],d['pred_index'])))
        chosen.append({'category':cat,'n_candidates':len(cc), 'selected':cc[0] if cc else None})
    manifest['fig5_selection']=chosen; manifest['checks']=checks
    if chosen[2]['selected'] and not chosen[2]['selected']['all_debris_subthreshold']:
        manifest['warnings'].append('Panel c highest-score other-debris FP is not exclusively subthreshold. Do not relabel or replace it silently.')
    # Freeze the selection BEFORE reading radiometric RGB data.
    (out/'fig5_selection_frozen.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('[3/5] Selection frozen. Reading true-colour raw TIFF crops.',flush=True)
    crops=[]; arrays={}; metadata=[]
    for i,c in enumerate(chosen):
        d=c['selected']
        if d is None: continue
        st=d['stem']; path=RAW/scene[st]/f'{st}.tif'; record(path)
        clpath=RAW/scene[st]/f'{st}_cl.tif'; record(clpath)
        b=np.array(d['box_512'])/2; cx,cy=(b[:2]+b[2:])/2
        x0=int(np.clip(np.floor(cx-32),0,192)); y0=int(np.clip(np.floor(cy-32),0,192))
        with rasterio.open(path) as src:
            assert src.shape==(256,256)
            assert np.allclose(src.res,(10,10)), 'Scale bar requires 10-m pixels'
            rgb=np.stack([src.read(j)[y0:y0+64,x0:x0+64] for j in (4,3,2)],axis=-1).astype(np.float32)
            assert np.isfinite(rgb).all(), 'Non-finite RGB pixels: do not synthesize replacements'
            crs=str(src.crs); affine=list(src.transform)
        crops.append(rgb); arrays[f'rgb_reflectance_{i}']=rgb
        with rasterio.open(clpath) as src: arrays[f'labels_{i}']=src.read(1)[y0:y0+64,x0:x0+64].astype(np.uint8)
        gi=d.get('gt_index'); gt_boxes=[{'index':j,'box_256':(bb/2).tolist()} for j,bb in enumerate(gt[st])]
        pred_boxes=[{'index':int(j),'box_256':(rr[st]['p'][j,:4]/2).tolist(),
                     'tp':bool(rr[st]['tp'][j]),'score':float(rr[st]['p'][j,4])}
                    for j in np.flatnonzero(rr[st]['sel'])]
        info=at.parse_marida_name(st)
        metadata.append({'panel':i,'category':c['category'],'stem':st,'scene':scene[st],
            'date':str(info['date']),'tile':info['tile'],'fold':scene_fold[scene[st]],
            'score':d['score'],'target_box_256':b.tolist(),'gt_index':gi,'pred_index':d.get('pred_index'),
            'window_xywh':[x0,y0,64,64], 'gt_boxes':gt_boxes,'selected_predictions':pred_boxes,
            'crs':crs,'source_affine':affine,'metres_per_pixel':10})
    assert crops, 'No real imagery selected'
    vals=np.concatenate([x.reshape(-1,3) for x in crops]); lo,hi=np.percentile(vals,[2,98],axis=0)
    assert np.all(hi>lo)
    for m in metadata:
        i=m['panel']; arrays[f'rgb_uint8_{i}']=np.uint8(np.rint(np.clip((arrays[f'rgb_reflectance_{i}']-lo)/(hi-lo),0,1)*255))
    manifest['fig5_panels']=metadata
    manifest['stretch']={'bands':['B4','B3','B2'],'percentiles':[2,98], 'lo':lo.tolist(),'hi':hi.tolist(),
                         'shared_across_all_six':True, 'unit':'Rayleigh-corrected reflectance'}
    lines=[]
    for name,a in arrays.items(): encode(name,a,lines)
    print('[4/5] Writing lossless transport, raw-reflectance cache and source hashes.',flush=True)
    (out/'fig5_extract.txt').write_text('\n'.join(lines)+'\n',encoding='ascii')
    np.savez_compressed(out/'fig5_arrays.npz',**arrays)
    (out/'fig45_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('[5/5] Complete:',out,flush=True)
    print('Warnings:',manifest['warnings'],flush=True)

if __name__=='__main__': main()
