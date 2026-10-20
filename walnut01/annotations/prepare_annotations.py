"""Rebuild the agent-assisted annotation draft and its review images.

The image-specific seed selections and envelopes in annotation_spec.json are
annotation input. This utility is not an automatic inference/evaluation method.
"""
from pathlib import Path
import hashlib
import json
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'annotation_spec.json').read_text())
COLORS = np.array([[17,24,39], [242,173,72], [56,189,248]], dtype=np.uint8)

def font(size):
    for p in ['/System/Library/Fonts/Supplemental/Arial.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              'C:/Windows/Fonts/arial.ttf']:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def display_image(a):
    # One fixed display window for all sections, separate from segmentation data.
    return Image.fromarray(np.uint8(np.clip(a / 0.067, 0, 1) * 255)).convert('RGB')

def annotate(a, record):
    if record.get('empty'):
        return np.zeros(a.shape, dtype=np.uint8)
    pars = SPEC['assistance']
    high = a >= pars['component_level']
    _, components, stats, _ = cv2.connectedComponentsWithStats(high.astype('uint8'), 8)
    core = np.zeros(a.shape, dtype=bool)
    for x,y in record['kernel_seeds']:
        ident = int(components[y,x])
        if ident == 0:
            raise ValueError(f'Seed {(x,y)} missed a tissue component')
        core |= components == ident
    for polygon in record['kernel_envelopes']:
        envelope = np.zeros(a.shape, dtype='uint8')
        cv2.fillPoly(envelope, [np.array(polygon, dtype=np.int32)], 1)
        core |= high & envelope.astype(bool) & (a < 0.052)
    # Trim narrow septa attached to selected kernel profiles. The operation is
    # conservative (2 pixels); every changed boundary is in the uncertainty map.
    opened = cv2.morphologyEx(core.astype('uint8'), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
    dist_to_opened = ndi.distance_transform_edt(~opened)
    core &= dist_to_opened <= 1.5
    material = a >= pars['material_level']
    # Remove isolated one/two-pixel speckles; retain real fine septa.
    _, cc, cs, _ = cv2.connectedComponentsWithStats(material.astype('uint8'), 8)
    material &= cs[:,cv2.CC_STAT_AREA][cc] >= 3
    # The largest high-intensity component spans the visible shell. Its loose
    # bounding box removes isolated support/reconstruction speckles far outside.
    largest = 1 + int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    x,y,w,h,_ = stats[largest]
    support = np.zeros(a.shape,bool)
    support[max(0,y-8):min(a.shape[0],y+h+8),max(0,x-8):min(a.shape[1],x+w+8)] = True
    material &= support
    high_labels = np.where(core, 2, 1).astype('uint8')
    distance, nearest = ndi.distance_transform_edt(~high, return_indices=True)
    label = np.zeros(a.shape, dtype='uint8')
    label[material] = 1
    nearby = material & (distance <= pars['max_edge_extension_pixels'])
    label[nearby] = high_labels[nearest[0][nearby], nearest[1][nearby]]
    # Image-specific negative strokes, placed during visual review to distinguish
    # thin septa from adjacent kernel. They do not paint absent tissue into air.
    for stroke in record.get('shell_strokes',[]):
        paint=np.zeros(a.shape,dtype='uint8')
        cv2.polylines(paint,[np.array(stroke['points'],dtype='int32')],False,1,stroke['width'])
        label[paint.astype(bool) & material] = 1
    return label

def legend(draw, x, y, size=18):
    for k, (name, dx) in enumerate([('Воздух / фон',0),('Скорлупа и перегородки',175),('Ядро',460)]):
        draw.rounded_rectangle((x+dx,y,x+dx+18,y+18),radius=4,fill=tuple(COLORS[k]))
        draw.text((x+dx+27,y-1),name,font=font(size),fill='#26364b')

def pair_image(a, labels, code):
    canvas=Image.new('RGB',(1066,614),'#f6f8fc')
    d=ImageDraw.Draw(canvas)
    d.text((24,16),f'walnut01.{code}',font=font(24),fill='#14243a')
    d.text((24,54),'Исходный снимок',font=font(19),fill='#536176')
    d.text((541,54),'Разметка',font=font(19),fill='#536176')
    canvas.paste(display_image(a),(24,85))
    canvas.paste(Image.fromarray(COLORS[labels]),(541,85))
    legend(d,24,592,15)
    if not np.any(labels):
        d.text((730,54),'Орех отсутствует',font=font(16),fill='#536176')
    return canvas

def main():
    for sub in ['masks','uncertainty','previews','overlays']:
        (HERE/sub).mkdir(exist_ok=True)
    manifest={'version':SPEC['version'],'status':SPEC['status'],
              'source_commit':'5e7d4a2d3b07244be972243acca1a6f6a8ccc2b9',
              'labels':SPEC['classes'],'images':[]}
    pairs=[]
    for code,record in SPEC['images'].items():
        source=ROOT/'data/images'/f'walnut01.{code}.tif'
        a=np.asarray(Image.open(source))
        assert a.shape==(501,501) and a.dtype==np.float32
        labels=annotate(a,record)
        assert labels.dtype==np.uint8 and set(np.unique(labels)) <= {0,1,2}
        Image.fromarray(labels).save(HERE/'masks'/f'walnut01.{code}.png')
        # This uncertainty layer is separate from the complete 0/1/2 labelmap.
        boundary=np.zeros(a.shape,bool)
        for k in [0,1,2]:
            region=labels==k
            boundary |= ndi.binary_dilation(region) & ~ndi.binary_erosion(region,border_value=1)
        uncertain=ndi.binary_dilation(boundary,iterations=1)
        Image.fromarray(uncertain.astype('uint8')*255).save(HERE/'uncertainty'/f'walnut01.{code}.png')
        base=np.array(display_image(a)).astype(float)
        over=base.copy()
        tissue=labels>0
        over[tissue]=base[tissue]*0.40+COLORS[labels[tissue]]*0.60
        Image.fromarray(over.astype('uint8')).save(HERE/'overlays'/f'walnut01.{code}.png')
        pair=pair_image(a,labels,code)
        pair.save(HERE/'previews'/f'walnut01.{code}_pair.png')
        pairs.append(pair)
        counts={str(k):int(np.sum(labels==k)) for k in range(3)}
        manifest['images'].append({'id':code,'file':source.name,
            'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'mask':f'masks/walnut01.{code}.png','counts':counts,
            'empty':record.get('empty',False),'review_note':record.get('note','')})
        print(code,counts)
    for group in range(3):
        sheet=Image.new('RGB',(2164,1260),'#e4eaf2')
        for j in range(4):
            sheet.paste(pairs[group*4+j],(8+(j%2)*1078,8+(j//2)*626))
        sheet.save(HERE/'previews'/f'overview_{group+1:02d}.png')
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(HERE/'walnut01_annotation_v01.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in [HERE/'annotation_spec.json',HERE/'manifest.json',HERE/'README.md']:
            if p.exists(): archive.write(p,p.name)
        for sub in ['masks','uncertainty','previews','overlays']:
            for p in sorted((HERE/sub).glob('*.png')):
                archive.write(p,str(p.relative_to(HERE)))

if __name__=='__main__':
    main()
