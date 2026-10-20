"""Make inspectable figures from saved predictions; no inference or fitting."""
from pathlib import Path
import csv
import json
import os
import tempfile

import numpy as np
from PIL import Image,ImageDraw,ImageFont
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'walnut-matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'
VIS=RES/'visuals'
COLORS=np.array([[17,24,39],[242,173,72],[56,189,248]],dtype=np.uint8)
ERROR_COLORS={'false_tissue':(251,75,88),'missed_tissue':(71,138,255),
              'shell_as_kernel':(225,80,186),'kernel_as_shell':(255,199,87)}
NAMES={'none_threshold':'Пороги, без фильтра','gaussian_threshold':'Пороги, Gaussian',
       'guided_threshold':'Пороги, guided','none_regions':'Области, без фильтра',
       'gaussian_regions':'Области, Gaussian','guided_regions':'Области, guided'}

def font(size):
    for p in ['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arial.ttf']:
        if Path(p).exists():return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def load(path):return np.asarray(Image.open(path))
def gray(a):return Image.fromarray(np.uint8(np.clip(a/0.067,0,1)*255)).convert('RGB')
def color(a):return Image.fromarray(COLORS[a])
def pct(v):return f'{v*100:.2f}%' if v is not None else 'нет класса'

def mosaic(panels,columns,title,subtitle='',tile=501):
    rows=(len(panels)+columns-1)//columns
    header=92
    width=columns*(tile+22)+22
    canvas=Image.new('RGB',(width,header+rows*(tile+63)+20),'#f5f7fb')
    d=ImageDraw.Draw(canvas)
    d.text((22,16),title,font=font(27),fill='#17283e')
    d.text((22,55),subtitle,font=font(18),fill='#58677a')
    for i,(label,im) in enumerate(panels):
        x=22+i%columns*(tile+22);y=header+i//columns*(tile+63)
        d.text((x,y),label,font=font(18),fill='#25364d')
        canvas.paste(im.resize((tile,tile),Image.Resampling.NEAREST),(x,y+33))
    return canvas

def errors(a,gt,pred):
    base=np.array(gray(a))*0.34
    matched=(gt==pred)&(gt>0)
    base[matched]=np.array(gray(a))[matched]*0.7
    conditions=[(gt==0)&(pred>0),(gt>0)&(pred==0),(gt==1)&(pred==2),(gt==2)&(pred==1)]
    for cond,co in zip(conditions,ERROR_COLORS.values()):base[cond]=co
    return Image.fromarray(base.astype('uint8'))

def stage_panels(code,metrics):
    selected=metrics['_recommended']['variant']
    a=load(ROOT/'data/images'/f'walnut01.{code}.tif')
    guided=load(RES/'guided_regions/filtered'/f'walnut01.{code}.tif')
    gauss=load(RES/'gaussian_regions/filtered'/f'walnut01.{code}.tif')
    baseline=load(RES/'none_threshold/masks'/f'walnut01.{code}.png')
    pred=load(RES/selected/'masks'/f'walnut01.{code}.png')
    gt=load(ROOT/'annotations/masks'/f'walnut01.{code}.png')
    material=load(RES/selected/'intermediates'/f'walnut01.{code}_material.png')
    core=load(RES/selected/'intermediates'/f'walnut01.{code}_cores.png')
    markers=load(RES/selected/'intermediates'/f'walnut01.{code}_markers.png')
    panels=[('1. Исходный TIFF',gray(a)),('2. Guided filter',gray(guided)),
            ('3. Gaussian для сравнения',gray(gauss)),('4. Базовые пороги',color(baseline)),
            ('5. Маска тканей',Image.fromarray(material).convert('RGB')),
            ('6. Центральные части областей',Image.fromarray(core).convert('RGB')),
            ('7. Маркеры (тёмное = неизвестно)',color(markers)),
            ('8. Итог выбранного метода',color(pred)),('9. Разметка v0.1',color(gt))]
    # For the recommended variant, the selected filter is none; panels 2 and 3
    # are comparison branches, not fictitious mandatory steps of that variant.
    img=mosaic(panels,3,f'Промежуточные результаты: walnut01.{code}',
        'Фильтры — варианты сравнения; выбран: '+NAMES[selected]+'.')
    img.save(VIS/f'{code}_stages.png')

def main():
    VIS.mkdir(parents=True,exist_ok=True)
    m=json.loads((RES/'metrics.json').read_text())
    recommended=m['_recommended']['variant']
    protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    rows=list(csv.DictReader((RES/'per_image_metrics.csv').open()))
    for code in protocol['development']+protocol['control']:
        a=load(ROOT/'data/images'/f'walnut01.{code}.tif')
        gt=load(ROOT/'annotations/masks'/f'walnut01.{code}.png')
        pred=load(RES/recommended/'masks'/f'walnut01.{code}.png')
        row=next(x for x in rows if x['variant']==recommended and x['image']==code)
        parts=[('Исходный снимок',gray(a)),('Разметка v0.1',color(gt)),
               ('Автоматический результат',color(pred)),('Карта ошибок',errors(a,gt,pred))]
        shell=float(row['iou_shell']) if row['iou_shell'] else None
        kernel=float(row['iou_kernel']) if row['iou_kernel'] else None
        image=mosaic(parts,4,f'walnut01.{code}: результат и ошибки',
            f'IoU скорлупы: {pct(shell)}; IoU ядра: {pct(kernel)}. Часть: '+('контрольная' if code in protocol['control'] else 'настроечная'))
        extra=Image.new('RGB',(image.width,image.height+44),'#f5f7fb');extra.paste(image)
        d=ImageDraw.Draw(extra)
        labels=['Лишняя ткань','Пропущенная ткань','Скорлупа → ядро','Ядро → скорлупа']
        for i,(label,co) in enumerate(zip(labels,ERROR_COLORS.values())):
            x=22+i*380;y=image.height+4
            d.rectangle((x,y,x+17,y+17),fill=co)
            d.text((x+25,y-1),label,font=font(16),fill='#25364d')
        extra.save(VIS/f'{code}_comparison.png')
        stage_panels(code,m)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    variants=list(NAMES)
    x=np.arange(len(variants));width=0.36
    fig,ax=plt.subplots(figsize=(12,5.8),layout='constrained')
    for j,(cls,label,co) in enumerate([(1,'Скорлупа и перегородки','#F2AD48'),(2,'Ядро','#38BDF8')]):
        values=[m[k]['subsets']['control']['iou'][cls]*100 for k in variants]
        bars=ax.bar(x+(j-.5)*width,values,width,color=co,label=label)
        ax.bar_label(bars,fmt='%.1f',padding=4,fontsize=9)
    ax.set_xticks(x,[NAMES[k].replace(', ','\n') for k in variants])
    ax.set_ylim(0,106);ax.set_ylabel('IoU, %');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    ax.set_title('8 контрольных сечений: согласие с разметкой v0.1',loc='left',pad=18,fontweight='bold')
    ax.legend(loc='lower right');ax.spines[['top','right']].set_visible(False)
    fig.savefig(VIS/'method_comparison.png',dpi=180);plt.close(fig)
    cm=np.array(m[recommended]['subsets']['control']['confusion'])
    normalized=cm/cm.sum(axis=1,keepdims=True)*100
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    im=ax.imshow(normalized,vmin=0,vmax=100,cmap='Blues')
    for y in range(3):
        for x in range(3):
            ax.text(x,y,f'{normalized[y,x]:.2f}%\n{cm[y,x]:,}',ha='center',va='center',color='white' if normalized[y,x]>55 else '#203044')
    names=['Воздух / фон','Скорлупа','Ядро']
    ax.set_xticks(range(3),names);ax.set_yticks(range(3),names)
    ax.set_xlabel('Предсказанный класс');ax.set_ylabel('Класс разметки')
    ax.set_title('Контроль: матрица ошибок',pad=16)
    fig.colorbar(im,ax=ax,label='% внутри класса разметки')
    fig.savefig(VIS/'confusion_matrix.png',dpi=180);plt.close(fig)
    # A standalone HTML gallery is portable when extracted with its image folder.
    html=['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>Грецкий орех: результаты этапа 2</title><style>body{font:17px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#17283e;background:#f5f7fb}img{width:100%;height:auto;border-radius:12px;margin:10px 0 28px}a{color:#245f92}h1{font-size:30px}p{line-height:1.5}</style>',
          '<h1>Грецкий орех: сегментация и промежуточные результаты</h1><p>Три класса: воздух/фон, скорлупа и перегородки, ядро. Цветные маски сравниваются с полуавтоматической разметкой v0.1.</p>',
          '<p>Данные одного ореха; результат характеризует согласие с этой разметкой. Фильтры на иллюстрациях — отдельные варианты сравнения.</p>',
          '<img src="visuals/method_comparison.png" alt="Сравнение методов">']
    for code in sorted(protocol['development']+protocol['control']):
        html += [f'<h2>walnut01.{code}</h2>',f'<img src="visuals/{code}_comparison.png" alt="Исходник, разметка, результат и ошибки">',
                 f'<details><summary>Промежуточные результаты</summary><img src="visuals/{code}_stages.png" alt="Варианты фильтрации и маски"></details>']
    (RES/'gallery.html').write_text('\n'.join(html)+'</html>\n')
    print('Created comparisons and intermediate-stage figures for 12 sections.')

if __name__=='__main__':main()
