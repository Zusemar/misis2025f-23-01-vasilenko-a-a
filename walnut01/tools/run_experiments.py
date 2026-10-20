"""Select parameters on development sections, freeze, then score control sections."""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import subprocess

import numpy as np
from PIL import Image
from evaluation import confusion,scores,boundary_f1,json_finite

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
PROTOCOL=json.loads((ROOT/'configs/protocol.json').read_text())

def load_mask(path):
    a=np.asarray(Image.open(path))
    if a.shape!=(501,501) or a.dtype!=np.uint8:
        raise ValueError(f'Invalid mask geometry/type: {path}')
    return a

def read_reference(code):
    return load_mask(ROOT/'annotations/masks'/f'walnut01.{code}.png')

def write_csv(path,records):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows({k:('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v)
                          for k,v in row.items()} for row in records)

def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(binary,inputs,output,config):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'request.json';path.write_text(json.dumps(config,indent=2)+'\n')
    process=subprocess.run([str(binary),str(inputs),str(output),'--config',str(path)],capture_output=True,text=True)
    if process.returncode:
        raise RuntimeError(process.stdout+process.stderr)

def search_grid():
    for method in ['threshold','regions']:
        cores=[2,3,4] if method=='regions' else [3]
        for factor,core in itertools.product([0.8,1.0,1.2],cores):
            yield dict(method=method,filter='none',air_factor=factor,core_radius=core)
            for sigma in [0.7,1.2]:
                yield dict(method=method,filter='gaussian',sigma=sigma,air_factor=factor,core_radius=core)
            for radius,eps in itertools.product([2,4],[0.0003,0.001,0.003]):
                yield dict(method=method,filter='guided',radius=radius,epsilon=eps,air_factor=factor,core_radius=core)

def assess(prediction_dir,codes):
    mat=np.zeros((3,3),np.int64)
    for code in codes:
        mat+=confusion(read_reference(code),load_mask(prediction_dir/'masks'/f'walnut01.{code}.png'))
    return scores(mat)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,default=ROOT/'build/walnut_segment')
    parser.add_argument('--reuse-selection',action='store_true')
    args=parser.parse_args()
    binary=args.binary.resolve()
    if not binary.is_file():raise FileNotFoundError(binary)
    RESULTS.mkdir(exist_ok=True)
    assert not set(PROTOCOL['development'])&set(PROTOCOL['control'])
    assert len(set(PROTOCOL['development']+PROTOCOL['control']))==12
    label_hashes={c:hash_file(ROOT/'annotations/masks'/f'walnut01.{c}.png') for c in PROTOCOL['development']+PROTOCOL['control']}
    (RESULTS/'protocol.json').write_text(json.dumps({**PROTOCOL,'label_sha256':label_hashes},indent=2)+'\n')
    search_path=RESULTS/'search_predictions'
    development_inputs=search_path/'development_inputs'
    development_inputs.mkdir(parents=True,exist_ok=True)
    for c in PROTOCOL['development']:
        dest=development_inputs/f'walnut01.{c}.tif'
        if not dest.exists():dest.symlink_to(ROOT/'data/images'/dest.name)
    candidates=list(search_grid())
    if not args.reuse_selection:
        # Save the complete search space before running the first candidate.
        (RESULTS/'search_space.json').write_text(json.dumps(candidates,indent=2)+'\n')
        best={};trials=[]
        for i,config in enumerate(candidates):
            output=search_path/'current'
            run(binary,development_inputs,output,config)
            quality=assess(output,PROTOCOL['development'])
            key=config['filter']+'_'+config['method']
            value=quality['foreground_miou']
            row={'trial':i,'variant':key,**config,'development_foreground_miou':value}
            trials.append(row)
            if key not in best or value>best[key]['development_foreground_miou']:
                best[key]={'config':config,'development_foreground_miou':value,'trial':i}
            if (i+1)%9==0:
                print(f'Development search: {i+1}/{len(candidates)} candidates',flush=True)
        (RESULTS/'search_trials.json').write_text(json.dumps(json_finite(trials),indent=2)+'\n')
        # Freeze all variant configurations before accessing control references.
        (RESULTS/'selection.json').write_text(json.dumps(best,indent=2)+'\n')
        for key,selected in best.items():
            (ROOT/'configs'/f'{key}.json').write_text(json.dumps(selected['config'],indent=2)+'\n')
    else:
        best=json.loads((RESULTS/'selection.json').read_text())
    rows=[];summary={}
    # Primary runs use independently selected development parameters per variant.
    variants={key:{'config':item['config'],'group':'selected_per_variant'} for key,item in best.items()}
    # The following ablation freezes segmentation settings and changes only the
    # filter. It is distinct from independently optimizing each comparison.
    anchor=dict(best['guided_regions']['config'])
    for filt in ['none','gaussian']:
        variants[f'ablation_{filt}_regions']={'config':{**anchor,'filter':filt,'sigma':1.0},'group':'fixed_guided_settings'}
    for key,item in variants.items():
        output=RESULTS/key
        run(binary,ROOT/'data/images',output,item['config'])
        summary[key]={'config':item['config'],'group':item['group'],'subsets':{}}
        for subset,codes in [('development',PROTOCOL['development']),('control',PROTOCOL['control']),('all',PROTOCOL['development']+PROTOCOL['control'])]:
            matrix=np.zeros((3,3),np.int64);stable_matrix=np.zeros((3,3),np.int64)
            boundaries={k:[] for k in range(3)}
            for code in codes:
                gt=read_reference(code)
                pred=load_mask(output/'masks'/f'walnut01.{code}.png')
                uncertainty=load_mask(ROOT/'annotations/uncertainty'/f'walnut01.{code}.png')
                m=confusion(gt,pred)
                stable=confusion(gt,pred,uncertainty==0)
                matrix+=m; stable_matrix+=stable
                s=scores(m)
                bf=[boundary_f1(gt,pred,k,PROTOCOL['boundary_tolerance_pixels']) for k in range(3)]
                for k in range(3):boundaries[k].append(bf[k])
                if subset!='all':
                    rows.append({'variant':key,'subset':subset,'image':code,
                        'iou_air':s['iou'][0],'iou_shell':s['iou'][1],'iou_kernel':s['iou'][2],
                        'dice_shell':s['dice'][1],'dice_kernel':s['dice'][2],
                        'boundary_f1_shell':bf[1],'boundary_f1_kernel':bf[2],
                        'foreground_miou':s['foreground_miou'],'accuracy':s['accuracy']})
            s=scores(matrix);s['confusion']=matrix
            s['stable_interior']=scores(stable_matrix)
            s['boundary_f1']=[float(np.nanmean(boundaries[k])) for k in range(3)]
            s['count']=len(codes)
            summary[key]['subsets'][subset]=s
        print(key,'control foreground mIoU:',summary[key]['subsets']['control']['foreground_miou'],flush=True)
    # Choosing the recommended configuration uses development scores only.
    winner=max(best,key=lambda k:best[k]['development_foreground_miou'])
    summary['_recommended']={'variant':winner,'selected_by':'development only','config':best[winner]['config']}
    (ROOT/'configs/recommended.json').write_text(json.dumps(best[winner]['config'],indent=2)+'\n')
    (RESULTS/'metrics.json').write_text(json.dumps(json_finite(summary),ensure_ascii=False,indent=2)+'\n')
    write_csv(RESULTS/'per_image_metrics.csv',rows)
    table=[]
    for key,item in summary.items():
        if key.startswith('_'):continue
        for subset,s in item['subsets'].items():
            table.append({'variant':key,'comparison_group':item['group'],'subset':subset,
                'images':s['count'],'iou_air':s['iou'][0],'iou_shell':s['iou'][1],'iou_kernel':s['iou'][2],
                'dice_shell':s['dice'][1],'dice_kernel':s['dice'][2],
                'foreground_miou':s['foreground_miou'],'miou':s['miou'],
                'boundary_f1_shell':s['boundary_f1'][1],'boundary_f1_kernel':s['boundary_f1'][2],
                'stable_interior_foreground_miou':s['stable_interior']['foreground_miou']})
    write_csv(RESULTS/'summary.csv',table)
    sources={str(p.relative_to(ROOT)):hash_file(p) for p in sorted((ROOT/'src').glob('*'))}
    (RESULTS/'reproducibility.json').write_text(json.dumps({'binary_sha256':hash_file(binary),
        'source_sha256':sources,'protocol_sha256':hash_file(ROOT/'configs/protocol.json'),
        'annotation_mask_sha256':label_hashes,'selection_sha256':hash_file(RESULTS/'selection.json')},indent=2)+'\n')
    assert label_hashes=={c:hash_file(ROOT/'annotations/masks'/f'walnut01.{c}.png') for c in label_hashes}
    print('Recommended by development scores:',winner,flush=True)

if __name__=='__main__':main()
