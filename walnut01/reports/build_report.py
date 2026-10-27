"""Build the editable project report and print-sized figures from frozen results.

Requirements: python-docx, Pillow, numpy. PDF is exported from the DOCX so both
deliverables have the same contents and pagination. No model is fitted here.
"""
from pathlib import Path
from copy import deepcopy
import csv
import json
import math
import hashlib
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIG=HERE/'figures'
OUT=HERE/'walnut_project_report.docx'
CODE_COMMIT='67e83ccc96e7eeddc1f27dbc0c364e621f5f3b7c'
REPO='https://github.com/Zusemar/misis2025f-23-01-vasilenko-a-a'
CODE_URL=f'{REPO}/tree/{CODE_COMMIT}/walnut01'
SOURCE_URL='https://github.com/dvpsun/misis2025f-23-00-polevoi-d-v/tree/5e7d4a2d3b07244be972243acca1a6f6a8ccc2b9/data/walnut01'
COLORS=np.array([[17,24,39],[242,173,72],[56,189,248]],dtype=np.uint8)
METRICS=json.loads((ROOT/'results/metrics.json').read_text())
MANIFEST=json.loads((ROOT/'annotations/manifest.json').read_text())
PROTOCOL=json.loads((ROOT/'configs/protocol.json').read_text())
SELECTED=METRICS['_recommended']['variant']
PER_IMAGE=list(csv.DictReader((ROOT/'results/per_image_metrics.csv').open()))
PAGINATION=json.loads((HERE/'pagination.json').read_text()) if (HERE/'pagination.json').exists() else {}

def font(size,bold=False):
    options=['/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf']
    for p in options:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def load(path):return np.asarray(Image.open(path))
def raw(code):return load(ROOT/'data/images'/f'walnut01.{code}.tif')
def gt(code):return load(ROOT/'annotations/masks'/f'walnut01.{code}.png')
def pred(code):return load(ROOT/'results'/SELECTED/'masks'/f'walnut01.{code}.png')
def gray(a):return Image.fromarray(np.uint8(np.clip(a/0.067,0,1)*255)).convert('RGB')
def mask(a):return Image.fromarray(COLORS[a])
def pct(v,d=2):return '-' if v is None else f'{100*v:.{d}f}'.replace('.',',')

def panels(items,cols,name):
    tile=501;pad=20;line=46
    rows=math.ceil(len(items)/cols)
    im=Image.new('RGB',(pad+cols*(tile+pad),pad+rows*(tile+line+pad)),'white')
    draw=ImageDraw.Draw(im)
    for i,(text,image) in enumerate(items):
        x=pad+i%cols*(tile+pad);y=pad+i//cols*(tile+line+pad)
        draw.text((x,y+2),text,font=font(29),fill='black')
        im.paste(image.resize((tile,tile),Image.Resampling.NEAREST),(x,y+line))
    im.save(FIG/name)
    return FIG/name

def make_figures():
    FIG.mkdir(exist_ok=True)
    for code in sorted(PROTOCOL['development']+PROTOCOL['control']):
        panels([('Исходное сечение',gray(raw(code))),('Разметка классов',mask(gt(code)))],2,f'annotation_{code}.png')
    for code in ['0230','2230']:
        a=raw(code);g=gt(code);p=pred(code)
        err=np.array(gray(a)).astype(float)*0.35
        correct=(g==p)&(g>0);err[correct]=np.array(gray(a))[correct]*0.7
        for cond,co in [((g==0)&(p>0),(251,75,88)),((g>0)&(p==0),(71,138,255)),
                        ((g==1)&(p==2),(225,80,186)),((g==2)&(p==1),(255,199,87))]:
            err[cond]=co
        panels([('Исходное сечение',gray(a)),('Разметка v0.1',mask(g)),
                ('Автоматический результат',mask(p)),('Карта ошибок',Image.fromarray(err.astype('uint8')))],2,f'case_{code}.png')
    code='0130'
    items=[]
    filters=[('Без фильтра',raw(code)),
             ('Gaussian',load(ROOT/'results/gaussian_regions/filtered'/f'walnut01.{code}.tif')),
             ('Guided filter',load(ROOT/'results/guided_regions/filtered'/f'walnut01.{code}.tif'))]
    for name,a in filters:items.append((name,gray(a)))
    for name,a in filters:items.append(('Фрагмент '+name.lower(),gray(a).crop((175,105,280,210))))
    panels(items,3,'filters.png')
    code='2230'
    parts=ROOT/'results'/SELECTED/'intermediates'
    panels([('Исходное сечение',gray(raw(code))),
            ('Маска тканей',Image.open(parts/f'walnut01.{code}_material.png').convert('RGB')),
            ('Центральные области',Image.open(parts/f'walnut01.{code}_cores.png').convert('RGB')),
            ('Автоматические маркеры',mask(load(parts/f'walnut01.{code}_markers.png')))],2,'markers.png')
    im=Image.new('RGB',(1800,730),'white');draw=ImageDraw.Draw(im)
    boxes=[((30,250,330,460),'TIFF float32\n501 × 501'),
           ((400,250,760,460),'Без фильтра\nGaussian\nGuided filter'),
           ((830,250,1160,460),'Гистограмма\nМаска тканей'),
           ((1240,50,1760,260),'Пороги яркости\nБазовый метод'),
           ((1240,450,1760,680),'Области и маркеры\nРаспространение классов')]
    for box,text in boxes:
        draw.rounded_rectangle(box,radius=16,fill='#F4F6F8',outline='#2D3F50',width=3)
        cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        draw.multiline_text((cx,cy),text,font=font(35),fill='black',anchor='mm',align='center',spacing=14)
    def arrow(points):
        draw.line(points,fill='#2D3F50',width=5)
        x,y=points[-1];x0,y0=points[-2];angle=math.atan2(y-y0,x-x0)
        draw.polygon([(x,y),(x-19*math.cos(angle-.5),y-19*math.sin(angle-.5)),
                      (x-19*math.cos(angle+.5),y-19*math.sin(angle+.5))],fill='#2D3F50')
    arrow([(330,355),(400,355)]);arrow([(760,355),(830,355)])
    arrow([(1160,355),(1200,355),(1200,155),(1240,155)])
    arrow([(1160,355),(1200,355),(1200,565),(1240,565)])
    draw.text((1250,320),'Обе ветки дают маску 0 1 2',font=font(31),fill='black')
    im.save(FIG/'pipeline.png')

def xml(tag,**attrs):
    element=OxmlElement(tag)
    for key,value in attrs.items():element.set(qn(key),str(value))
    return element

def mr(text):
    element=xml('m:r');element.append(xml('m:rPr'))
    t=xml('m:t');t.text=text;element.append(t);return element
def expr(value):return mr(value) if isinstance(value,str) else deepcopy(value)
def mrow(values):
    e=xml('m:e')
    for value in values:e.append(expr(value))
    return e
def sub(base,index):
    e=xml('m:sSub');e.append(mrow([base]));s=xml('m:sub');s.append(expr(index));e.append(s);return e
def sup(base,index):
    e=xml('m:sSup');e.append(mrow([base]));s=xml('m:sup');s.append(expr(index));e.append(s);return e
def frac(numerator,denominator):
    e=xml('m:f');n=xml('m:num');d=xml('m:den')
    for value in numerator:n.append(expr(value))
    for value in denominator:d.append(expr(value))
    e.append(n);e.append(d);return e
def bar(base):
    e=xml('m:acc');pr=xml('m:accPr');pr.append(xml('m:chr',**{'m:val':'̅'}));e.append(pr);e.append(mrow([base]));return e

class Report:
    def __init__(self):
        self.doc=Document();self.count=0;self.figures=0
        s=self.doc.sections[0]
        s.page_width=Cm(21);s.page_height=Cm(29.7)
        s.top_margin=Cm(1.8);s.bottom_margin=Cm(1.8)
        s.left_margin=Cm(2.2);s.right_margin=Cm(1.8)
        s.header_distance=Cm(0.7);s.footer_distance=Cm(0.7)
        s.different_first_page_header_footer=True
        styles=self.doc.styles
        for st in styles:
            if not hasattr(st,'font'):continue
            st.font.name='Times New Roman';st.font.size=Pt(14);st.font.color.rgb=RGBColor(0,0,0)
            st._element.get_or_add_rPr().append(xml('w:rFonts',**{'w:ascii':'Times New Roman','w:hAnsi':'Times New Roman','w:eastAsia':'Times New Roman','w:cs':'Times New Roman'}))
        normal=styles['Normal'];normal.font.size=Pt(14)
        normal.paragraph_format.line_spacing=1.08
        normal.paragraph_format.space_after=Pt(5)
        normal.paragraph_format.first_line_indent=Cm(0.7)
        normal.paragraph_format.widow_control=True
        for name,size in [('Heading 1',14),('Heading 2',14),('Heading 3',14)]:
            style=styles[name];style.font.size=Pt(size);style.font.bold=True
            style.paragraph_format.first_line_indent=Cm(0)
            style.paragraph_format.space_before=Pt(8);style.paragraph_format.space_after=Pt(8)
            style.paragraph_format.keep_with_next=True
        styles['Title'].font.size=Pt(14);styles['Title'].font.bold=True
        styles['Title'].paragraph_format.first_line_indent=Cm(0)
        # The bundled Word template carries a blue border on Title. Remove
        # inherited decorative paragraph borders for this academic report.
        for node in list(styles.element.xpath('.//w:pBdr')):
            node.getparent().remove(node)
        styles['Caption'].font.size=Pt(14)
        styles['Caption'].paragraph_format.first_line_indent=Cm(0)
        styles['Caption'].paragraph_format.space_after=Pt(8)
        styles['Caption'].paragraph_format.line_spacing=1.05
        styles['Caption'].font.italic=False
        # Minimal page furniture: only the page number is retained.
        s.header.paragraphs[0].text=''
        footer=s.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
        footer.paragraph_format.first_line_indent=Cm(0)
        run=footer.add_run();run._r.append(xml('w:fldChar',**{'w:fldCharType':'begin'}))
        instr=xml('w:instrText');instr.text=' PAGE ';run._r.append(instr)
        run._r.append(xml('w:fldChar',**{'w:fldCharType':'end'}));run.font.size=Pt(14)
        settings=self.doc.settings.element
        settings.append(xml('w:updateFields',**{'w:val':'true'}))
        self.doc.core_properties.title='Сегментация частей грецкого ореха на КТ сечениях'
        self.doc.core_properties.subject='Учебный проект по компьютерному зрению'
        self.doc.core_properties.author='Василенко А. А.'
        self.doc.core_properties.keywords='КТ, грецкий орех, сегментация, guided filter, OpenCV'

    def page(self,title=None):
        new_page=self.count>0
        self.count+=1
        if title:
            heading=self.doc.add_heading(title,level=1)
            heading.paragraph_format.page_break_before=new_page
        elif new_page:self.doc.add_page_break()

    def p(self,text='',bold=False,align=None):
        p=self.doc.add_paragraph();p.alignment=align or WD_ALIGN_PARAGRAPH.JUSTIFY
        p.add_run(text).bold=bold
        return p

    def h(self,text):return self.doc.add_heading(text,level=2)

    def link(self,text,url):
        p=self.doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(0)
        h=xml('w:hyperlink',**{'r:id':p.part.relate_to(url,RT.HYPERLINK,is_external=True)})
        r=xml('w:r');pr=xml('w:rPr');pr.append(xml('w:color',**{'w:val':'000000'}));pr.append(xml('w:u',**{'w:val':'single'}))
        r.append(pr);t=xml('w:t');t.text=text;r.append(t);h.append(r);p._p.append(h)
        return p

    def picture(self,path,width,caption):
        self.figures+=1
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_after=Pt(3)
        p.paragraph_format.keep_with_next=True
        shape=p.add_run().add_picture(str(path),width=Cm(width))
        shape._inline.docPr.set('descr',caption);shape._inline.docPr.set('title',caption)
        c=self.doc.add_paragraph(f'Рисунок {self.figures} - {caption}',style='Caption')
        c.alignment=WD_ALIGN_PARAGRAPH.CENTER
        return c

    def table(self,headers,rows,widths,caption=None,size=14):
        if caption:
            p=self.doc.add_paragraph(caption,style='Caption');p.paragraph_format.keep_with_next=True
        t=self.doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for i,w in enumerate(widths):t.columns[i].width=Cm(w)
        for i,text in enumerate(headers):t.rows[0].cells[i].text=str(text)
        t.rows[0]._tr.get_or_add_trPr().append(xml('w:tblHeader',**{'w:val':'true'}))
        for values in rows:
            cells=t.add_row().cells
            for cell,value in zip(cells,values):cell.text=str(value)
        for rownum,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(xml('w:cantSplit'))
            for i,cell in enumerate(row.cells):
                cell.width=Cm(widths[i]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr();borders=xml('w:tcBorders')
                for side in ['top','left','bottom','right']:
                    borders.append(xml('w:'+side,**{'w:val':'single','w:sz':'5','w:color':'000000'}))
                tcpr.append(borders)
                margins=xml('w:tcMar')
                for side,value in [('top',65),('bottom',65),('left',85),('right',85)]:
                    margins.append(xml('w:'+side,**{'w:w':value,'w:type':'dxa'}))
                tcpr.append(margins)
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_after=Pt(0)
                    p.paragraph_format.line_spacing=1.05
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(14);run.bold=rownum==0
        self.doc.add_paragraph().paragraph_format.space_after=Pt(1)
        return t

    def equation(self,*parts):
        p=self.doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_before=Pt(3)
        p.paragraph_format.space_after=Pt(7)
        mp=xml('m:oMathPara');m=xml('m:oMath')
        for part in parts:m.append(expr(part))
        mp.append(m);p._p.append(mp)

    def code(self,text):
        for line in text.strip().splitlines():
            p=self.doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(0)
            p.paragraph_format.space_after=Pt(1);p.paragraph_format.line_spacing=1.0
            run=p.add_run(line);run.font.name='Times New Roman';run.font.size=Pt(14)

def build():
    snapshot=json.loads((HERE/'frozen_inputs.json').read_text())
    for name,digest in snapshot['inputs_sha256'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Report inputs changed; review the text before rebuilding: '+name)
    expected=[FIG/f'annotation_{c}.png' for c in PROTOCOL['development']+PROTOCOL['control']]
    expected += [FIG/f'{name}.png' for name in ['case_0230','case_2230','filters','markers','pipeline']]
    if not all(p.exists() for p in expected):make_figures()
    r=Report()
    # 1 Cover. Details absent from the request (group/semester) are not invented.
    r.page()
    for text in ['Национальный исследовательский технологический','университет «МИСИС»']:
        p=r.p(text,align=WD_ALIGN_PARAGRAPH.CENTER);p.paragraph_format.first_line_indent=Cm(0)
    r.p().paragraph_format.space_after=Pt(48)
    p=r.doc.add_paragraph('Сегментация частей\nгрецкого ореха\nна КТ сечениях',style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(20)
    p=r.p('Отчёт по учебному проекту',align=WD_ALIGN_PARAGRAPH.CENTER);p.paragraph_format.first_line_indent=Cm(0)
    p=r.p('Дисциплина «Компьютерное зрение»',align=WD_ALIGN_PARAGRAPH.CENTER);p.paragraph_format.first_line_indent=Cm(0)
    r.p().paragraph_format.space_after=Pt(36)
    p=r.p('Выполнил Василенко А. А.',align=WD_ALIGN_PARAGRAPH.RIGHT);p.paragraph_format.first_line_indent=Cm(0)
    p=r.p('27 октября 2026 года',align=WD_ALIGN_PARAGRAPH.RIGHT);p.paragraph_format.first_line_indent=Cm(0)
    r.p().paragraph_format.space_after=Pt(30)
    r.link('Репозиторий проекта',REPO)
    r.link('Код экспериментов и результаты',CODE_URL)
    r.p('Версия кода 67e83cc. Разметка v0.1.',align=WD_ALIGN_PARAGRAPH.CENTER)
    r.p('Москва 2026',align=WD_ALIGN_PARAGRAPH.CENTER)

    # 2 Purpose and navigation.
    r.page('Об отчёте')
    r.p('В проекте решена задача выделения воздуха, скорлупы с видимыми перегородками и ядра на готовых КТ сечениях грецкого ореха. Подготовлена разметка 12 изображений, реализована автоматическая обработка на C++20 и OpenCV, проведено сравнение двух способов сегментации и трёх вариантов предобработки.')
    r.p('Метод по областям без сглаживания получил средний IoU скорлупы и ядра 95,84% на восьми контрольных сечениях. Базовый пороговый метод получил 66,30%. Guided filter проверен как способ сглаживания с сохранением границ; в этих экспериментах он не улучшил основной вариант. Цифры описывают согласие с полуавтоматической разметкой одного ореха.')
    r.h('Содержание')
    toc=[('1 Постановка задачи и данные',3),('2 Разметка частей ореха',4),('3 Схема обработки и базовый метод',5),
         ('4 Протокол эксперимента',9),('5 Результаты сравнения',11),
         ('6 Архитектура программы',15),('6 3 Параметры и оценка',16),
         ('6 5 Воспроизведение',17),('6 6 Проверки и измерения',18),
         ('7 Выводы и ограничения',19),('Источники',20),
         ('Приложение А Размеченные сечения',18),('Приложение Б Качество по сечениям',24)]
    for title,page in toc:
        p=r.doc.add_paragraph();p.paragraph_format.first_line_indent=Cm(0);p.paragraph_format.space_after=Pt(3)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(16.8),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
        p.add_run(title+'\t'+str(PAGINATION.get(title,page)))

    # 3 Input dataset.
    r.page('1 Постановка задачи и данные')
    r.p('Для каждого пикселя исходного сечения требуется определить один из трёх классов. Выходом служит одноканальная маска того же размера со значениями 0, 1 и 2. Цветное изображение используется для просмотра; числовая маска сохраняется отдельно, без JPEG сжатия.')
    r.p('Опорная работа Der Sarkissian и соавторов описывает КТ данные 42 орехов и реконструкции объёмов 501³ [1]. В этом проекте используется учебная выборка из 12 отдельных сечений одного ореха [2]. Термин ground truth в статье относится к качественной реконструкции, а не к готовой разметке частей ореха.')
    r.p('Все файлы имеют размер 501 × 501, один канал и тип float32. Минимум яркости равен нулю. TIFF читаются без изменения глубины данных; старые JPEG и PNG с искажённым контрастом заменены исходниками. Совпадение файлов с источником проверено контрольными суммами.')
    rows=[]
    for code in sorted(PROTOCOL['development']+PROTOCOL['control']):
        rows.append([f'walnut01.{code}.tif',f'{raw(code).max():.5f}'.replace('.',','),'Настройка' if code in PROTOCOL['development'] else 'Контроль'])
    r.table(['Файл','Максимум яркости','Часть выборки'],rows,[7.2,4.1,5.7],'Таблица 1 - Состав учебной выборки')
    r.p('На сечениях 1430 и 2430 ткань ореха не различима. Держатель и текстура реконструкции относятся к фону. Файлы не объединяются в вымышленный последовательный 3D объём: их пространственная геометрия в учебной выборке не задана.')

    # 4 Annotation.
    r.page('2 Разметка частей ореха')
    r.table(['Код','Класс и цвет','Правило отнесения'],[
        ['0','Воздух и фон\nТёмно-синий','Наружный фон, внутренние пустоты, держатель и фоновые артефакты'],
        ['1','Скорлупа\nОранжевый','Скорлупа и различимые внутренние стенки и перегородки'],
        ['2','Ядро\nГолубой','Ткань ядра, в том числе небольшие изолированные профили']],
        [1.5,4.3,11.2],'Таблица 2 - Принятые классы и цвета разметки')
    r.p('Эти цвета используются на цветных масках разметки и итоговой сегментации. В исходных КТ изображениях показаны уровни серого. На наложениях цвет нанесён поверх исходника; карта ошибок имеет отдельную легенду, приведённую при разборе результатов.')
    r.p('Перегородки включены в класс скорлупы по принятому правилу сегментации. Это не утверждение об их гистологическом составе. На границе тканей яркости смешиваются, поэтому часть пикселей допускает неоднозначное толкование.')
    r.p('Разметка v0.1 подготовлена агентом полуавтоматически. По исходникам выбраны области ядра, огибающие полигоны и исправляющие штрихи перегородок; пиксельные границы уточнены по интенсивности. Агент просмотрел наложения и исправил заметные ошибки. Независимая экспертная проверка не выполнялась.')
    r.page('2 1 Подготовка и проверка разметки')
    r.picture(FIG/'annotation_0130.png',16.3,'Пример исходного сечения и разметки v0.1')
    r.p('Маски 0/1/2 сохраняют исходные координаты. Отдельно записана полоса около границ классов, требующая осторожной интерпретации. Это эвристическая полоса, а не вероятность ошибки. В основных метриках учитываются все пиксели.')
    r.p('3D Slicer предоставляет инструменты просмотра и редактирования сегментов [7]. В выполненных экспериментах разметка подготовлена программно и проверена изображениями наложений; обработка в 3D Slicer не выполнялась.')

    # 5 Overall method and threshold baseline.
    r.page('3 Схема обработки и базовый метод')
    r.p('Сравниваются две схемы, каждая может работать без фильтра, после Gaussian или после guided filter. На входе проверяются тип float32, число каналов и конечность неотрицательных значений. Нормированная яркость получается делением исходной на 0,067. Геометрия сечения остаётся прежней.')
    r.picture(FIG/'pipeline.png',17,'Сравниваемые ветки автоматической обработки')
    r.p('Два порога рассчитываются по гистограмме с 256 интервалами как максимум межклассовой дисперсии трёх групп. Это обобщение критерия Отцу [4]. Нижний порог выделяет материал, второй разделяет средние и высокие яркости. Чтобы расчёт не уходил в шумовой диапазон, применяются ограничения из таблицы 3.')
    r.p('В базовом методе пиксели ниже порога материала относятся к фону, средний диапазон - к ядру, верхний - к скорлупе. Преимущество такого решения - простота. Ограничение состоит в том, что яркости тонких стенок, перегородок и ядра могут перекрываться.')
    r.p('Маска тканей очищается только от изолированных компонент размером менее трёх пикселей. Реальные пустоты внутри ореха автоматически не заполняются. Если максимальная яркость исходника ниже 0,012, программа выдаёт фоновую маску.')
    r.p('Для учебной выборки одного ореха выбраны классические методы с небольшим числом параметров и явными промежуточными масками. Их поведение можно проверить по каждому сечению и каждой ошибке.')

    # 6 Guided filter.
    r.page('3 1 Фильтрация с сохранением границ')
    r.p('Guided filter реализован по работе He, Sun и Tang [3]. В режиме self-guided направляющим является само исходное изображение. В локальном окне вычисляются среднее μ и дисперсия σ², затем коэффициенты линейной модели. Усреднение коэффициентов даёт выход фильтра.')
    sk=sup(sub('σ','k'),'2')
    r.equation(sub('a','k'),' = ',frac([sk],[sk,' + ε']))
    r.equation(sub('b','k'),' = ',sub('μ','k'),' - ',sub('a','k'),sub('μ','k'))
    r.equation(sub('q','i'),' = ',sub(bar('a'),'i'),sub('I','i'),' + ',sub(bar('b'),'i'))
    r.p('При выраженном перепаде яркости коэффициент a приближается к единице, поэтому граница сглаживается слабее, чем однородная область. Реализация использует boxFilter OpenCV [6] с отражением по границам. Параметр ε относится к нормированной шкале яркости; радиус r задаётся в пикселях.')
    r.picture(FIG/'filters.png',17,'Сравнение предобработки и увеличенного фрагмента сечения 0130')
    r.p('Для варианта по областям на настроечных данных выбраны r = 2 пикселя и ε = 0,0003. Для Gaussian выбран σ = 0,7 пикселя. Нижний ряд показывает один и тот же фрагмент исходника; изображения отображаются в общем окне яркости [0; 0,067].')

    # 7 Markers.
    r.page('3 2 Автоматические маркеры')
    r.p('После выделения тканей эрозия радиусом два пикселя оставляет центральные части областей. Для каждой связной центральной области рассчитываются медианная яркость и среднее локальное СКО в окне 7 × 7. Текстура измеряется на исходном нормированном изображении, одинаково для всех вариантов фильтрации.')
    r.picture(FIG/'markers.png',15.4,'Получение центральных областей и автоматических маркеров на сечении 2230')
    r.p('На нижней правой панели оранжевый цвет обозначает маркеры скорлупы, голубой - ядра. Тёмные пиксели на этой панели ещё не классифицированы. Они не означают заранее установленный класс воздуха. Все маркеры получены из входного TIFF без точек и полигонов разметки.')

    # 8 Propagation and active settings.
    r.page('3 3 Распространение классов')
    r.p('Класс центральной области определяется её признаками. Медианная нормированная яркость ядра должна быть не ниже 0,40 и ниже порога скорлупы, а среднее локальное СКО - не выше 0,10. Остальные центральные области служат маркерами скорлупы. Области маркеров меньше четырёх пикселей не используются.')
    r.p('Распространение выполняется по кратчайшим путям Дейкстры [5] на 8-связном графе внутри маски тканей. При пересечении скачка яркости стоимость возрастает. Для соседних пикселей p и q используется стоимость')
    r.equation('c(p,q) = l(p,q) · (1 + λ |I(p) - I(q)|)')
    r.p('Здесь l равно 1 для прямого шага и √2 для диагонального, λ = 10. Все маркеры одновременно начинают распространение. Внешний фон исключён из графа. Несвязанные тонкие фрагменты без центральной области относятся к перегородкам.')
    r.table(['Параметр','Значение','Назначение'],[
        ['Рабочая шкала','I / 0,067','Нормирование яркости'],
        ['Минимум порога ткани','0,012','Нижняя граница в исходной шкале'],
        ['Порог пустого сечения','максимум < 0,012','Отсечение фоновых сечений'],
        ['Порог скорлупы','0,042 ... 0,053','Ограничение порога из гистограммы'],
        ['Множитель нижнего порога','1,0','Основной выбранный вариант'],
        ['Радиус центральной области','2 пикселя','Эрозия маски тканей'],
        ['Минимальная площадь маркера','4 пикселя','Отсечение малых центральных областей'],
        ['Окно локального СКО','7 × 7','Измерение текстуры'],
        ['Вес скачка яркости','10','Стоимость перехода в графе']],
        [6.3,4.2,6.5],'Таблица 3 - Активные параметры основного варианта')
    r.p('Тонкая перегородка может касаться ядра, оставаясь другим классом. Поэтому на её протяжённых участках задаются автоматические маркеры скорлупы: локальная полутолщина не более r + 0,5 пикселя и расстояние до центральной области больше r + 2 пикселей. Это препятствует распространению ядра вдоль всей перегородки.')

    # 9 Experiment protocol.
    r.page('4 Протокол эксперимента')
    r.p('Разделение сечений зафиксировано до подбора параметров. Выборка содержит один орех, поэтому термин «контроль» означает исключение этих файлов из настройки, но не независимость объектов: сечения могут пересекать одни и те же структуры.')
    r.table(['Часть','Сечения','Назначение'],[
        ['Настроечная','0130, 1230, 1430, 2130','Подбор параметров и выбор варианта'],
        ['Контрольная','0230, 0330, 0430, 1130, 1330, 2230, 2330, 2430','Проверка фиксированных параметров']],
        [3.5,7.2,6.3],'Таблица 4 - Разделение данных')
    r.p('Проверено 108 конфигураций: 27 для порогового метода и 81 для метода по областям. Каждый из шести вариантов отдельно настраивался по среднему IoU скорлупы и ядра на настроечной части. Сетка содержит множитель порога {0,8; 1,0; 1,2}, радиусы центральных областей {2; 3; 4}, σ Gaussian {0,7; 1,2}, радиусы guided filter {2; 4} и ε {0,0003; 0,001; 0,003}.')
    r.p('Сегментация получает только исходный TIFF и конфигурацию. Отдельный инструмент оценки загружает числовые маски разметки. Имена файлов, точки, огибающие и исправляющие штрихи разметки в алгоритм C++ не включены. Для проверки TIFF переименован; результат остался тем же.')
    r.p('При первом техническом прогоне искусственный тест выявил, что класс ядра может распространяться вдоль тонкой перегородки. Реализация исправлена по этому искусственному примеру. Уже выбранные параметры после просмотра контрольных результатов не перенастраивались. Итоговые цифры относятся к повторной проверке после исправления; первый прогон сохранён отдельно.')
    r.p('Помимо независимо настроенных вариантов выполнена абляция: параметры сегментации взяты у guided варианта и зафиксированы, изменяется только фильтр. Такое сравнение позволяет отделить влияние предобработки от изменения остальных параметров.')
    r.p('Контрольные суммы разметки до и после эксперимента совпадают. Полные списки конфигураций, результаты каждого испытания, матрицы ошибок и параметры запусков сохранены в репозитории [8].')

    # 10 Metrics.
    r.page('4 1 Метрики качества')
    r.p('Для каждого класса c рассчитываются истинно положительные TP, ложноположительные FP и ложноотрицательные FN пиксели. IoU измеряет отношение пересечения масок к их объединению, Dice - удвоенное пересечение к сумме площадей.')
    r.equation(sub('IoU','c'),' = ',frac([sub('TP','c')],[sub('TP','c'),' + ',sub('FP','c'),' + ',sub('FN','c')]))
    r.equation(sub('Dice','c'),' = ',frac(['2',sub('TP','c')],['2',sub('TP','c'),' + ',sub('FP','c'),' + ',sub('FN','c')]))
    r.p('Основной критерий - среднее IoU скорлупы и ядра. Для каждой части выборки сначала суммируются матрицы ошибок всех сечений, затем рассчитываются IoU двух классов и их среднее. Это не среднее процентов по отдельным изображениям.')
    r.equation('mIoU тканей = ',frac([sub('IoU','скорлупа'),' + ',sub('IoU','ядро')],['2']))
    r.p('Если класс отсутствует и в разметке, и в предсказании, его IoU не определён. Он записывается как null в JSON и пустая ячейка в CSV, не добавляя условные 100%. Если метод ошибочно предсказал ткань на пустом сечении, соответствующие ложные пиксели учитываются.')
    r.p('Для проверки границ применяется F1 с допуском один пиксель в окрестности 3 × 3. Точность границ показывает долю предсказанных граничных пикселей рядом с эталонной границей, полнота - долю эталонных рядом с предсказанной.')
    r.equation('F1 границ = ',frac(['2PR'],['P + R']))
    r.p('F1 границ усредняется по сечениям с определённой метрикой. Основные IoU и Dice учитывают все пиксели. Дополнительно измеряется согласие вне полосы неопределённости разметки, но эти более высокие значения не заменяют основные результаты.')
    r.p('Общая точность по пикселям приводится в машинных результатах, однако не служит главным критерием: большая площадь фона может скрывать ошибки различения скорлупы и ядра.')

    # 11 Main results.
    r.page('5 Результаты сравнения')
    keys=['none_threshold','gaussian_threshold','guided_threshold','none_regions','gaussian_regions','guided_regions']
    names=['Без фильтра и пороги','Gaussian и пороги','Guided и пороги','Без фильтра и области','Gaussian и области','Guided и области']
    rows=[]
    for name,key in zip(names,keys):
        s=METRICS[key]['subsets']['control'];rows.append([name,pct(s['iou'][1]),pct(s['iou'][2]),pct(s['foreground_miou'])])
    r.table(['Вариант','IoU скорлупы %','IoU ядра %','Средний IoU тканей %'],rows,[6.8,3.4,3.1,3.7],'Таблица 5 - Согласие с разметкой на восьми контрольных сечениях')
    r.picture(ROOT/'results/visuals/method_comparison.png',17,'Сравнение двух способов сегментации и трёх вариантов предобработки')
    r.p('Метод по областям без фильтра получил 95,84% против 66,30% у базовых порогов. Прирост составил 29,54 процентного пункта. Основной эффект связан с учётом областей и перегородок, а не со сглаживанием.')
    r.p('Вариант по областям с guided filter получил 95,76%. Разница с основным вариантом - около 0,09 процентного пункта. На этой небольшой выборке такой результат не даёт основания объявлять преимущество фильтра или статистически значимое различие.')
    r.p('При фиксированных параметрах сегментации абляция дала 95,84% без фильтра, 95,76% с guided filter и 80,36% с Gaussian при σ = 1. Последняя цифра характеризует именно эту замену фильтра; отдельно настроенный Gaussian вариант получил 92,70%.')

    # 12 Class details.
    r.page('5 1 Классы и границы')
    s=METRICS[SELECTED]['subsets']['control']
    r.table(['Класс','IoU %','Dice %','Precision %','Recall %'],
        [[name,pct(s['iou'][i]),pct(s['dice'][i]),pct(s['precision'][i]),pct(s['recall'][i])]
         for i,name in enumerate(['Воздух и фон','Скорлупа','Ядро'])],
        [4.7,3.0,3.0,3.15,3.15],'Таблица 6 - Метрики основного варианта на контрольной части')
    r.picture(ROOT/'results/visuals/confusion_matrix.png',11.0,'Матрица ошибок основного варианта на контрольных сечениях')
    r.p('Строки соответствуют разметке, столбцы - автоматическому результату. Проценты нормированы внутри каждой строки; под ними указано число пикселей. Фон выделяется почти полностью, а основные содержательные ошибки связаны с границами тканей и обменом между скорлупой и ядром.')
    r.p('Средний F1 границ при допуске один пиксель равен 98,06% для скорлупы и 95,27% для ядра. Dice этих классов составляет 97,43% и 98,32% соответственно. Вне полосы неопределённости средний IoU тканей равен 98,23%; это дополнительный результат, рассчитанный на более простой части пикселей.')

    # 13 Good control case.
    r.page('5 2 Пример успешной сегментации')
    r.p('Сечение 0230 относится к контрольной части. IoU скорлупы составляет 96,96%, ядра - 98,97%. Метод сохраняет воздушные пустоты и основную структуру внутренних перегородок.')
    r.picture(FIG/'case_0230.png',15.4,'Исходник разметка автоматический результат и ошибки для сечения 0230')
    r.p('На карте ошибок красный обозначает лишнюю ткань, синий - пропущенную. Малиновый показывает замену скорлупы ядром, жёлтый - обратную замену. Большинство различий в этом примере сосредоточено около границ.')

    # 14 Difficult control case.
    r.page('5 3 Разбор трудного сечения')
    r.p('Сечение 2230 - самое трудное в контрольной части: IoU скорлупы 86,31%, ядра 91,66%. Широкая внутренняя область, отнесённая в разметке к перегородке, автоматически определяется как ядро.')
    r.picture(FIG/'case_2230.png',15.4,'Ошибка различения широкой перегородки и ядра на контрольном сечении 2230')
    r.p('Различие выделено малиновым. Яркость и текстура этой области близки к выбранным признакам ядра. Такой случай требует проверки правила разметки и дополнительных признаков; усиление сглаживания само по себе не разрешает семантическую неоднозначность.')

    # 15 Implementation and runnable commands.
    r.page('6 Архитектура программы')
    r.p('Программа разделена на вычислительное ядро C++ и консольную оболочку. Подбор параметров, расчёт метрик и иллюстрации вынесены в Python инструменты. Такое разделение позволяет запускать один и тот же алгоритм в эксперименте и при обработке нового TIFF, а также проверять вычисления отдельно от файловых операций.')
    r.h('6 1 Вычислительное ядро и консольная оболочка')
    r.p('CMake собирает segmentation.cpp в библиотеку walnut_core. Её интерфейс объявлен в segmentation.hpp: структуры Parameters и Result, функции фильтрации, расчёта порогов и segment. Основной вызов segment(image, parameters) получает изображение и параметры в памяти. Ему не передаются имя файла, каталог разметки или координаты размеченных областей.')
    r.p('Файл main.cpp отвечает за аргументы запуска, поиск TIFF, загрузку и сохранение результатов. Файлы в папке сортируются перед обработкой. Консольная программа вызывает библиотеку для каждого изображения, поэтому тесты могут обращаться к тому же ядру без создания файлов и запуска отдельного процесса.')
    r.h('6 2 Контракты входа и результата')
    r.p('Для изображения принят строгий контракт: один канал float32, конечные неотрицательные значения. Неподходящие данные вызывают исключение; main перехватывает его, выводит сообщение и завершает запуск с ненулевым кодом. Это позволяет обнаружить ошибочную глубину TIFF до получения маски в неверной шкале яркости.')
    r.p('Структура Result возвращает итоговые метки uint8 со значениями 0, 1 и 2, обработанное изображение float32 в исходных единицах и маски промежуточных этапов. В ней также хранятся пороги, признаки областей и время вычисления. Цвета вводятся только при визуализации: числовые классы не зависят от палитры.')

    r.page('6 3 Параметры и оценка')
    r.p('Настройки собраны в структуре Parameters и читаются из JSON через OpenCV FileStorage. Консольные параметры --filter и --method могут переопределить соответствующие поля; после этого конфигурация проверяется повторно. Фактически использованные значения сохраняются в parameters.json вместе с версией OpenCV и числом потоков. Это позволяет установить, с какими настройками была получена маска.')
    r.p('Выбор фильтра и способа сегментации выполняется внутри общего вычислительного пути. Ветки none, gaussian и guided используют одну шкалу интенсивности, а методы threshold и regions возвращают одинаковый формат Result. Для небольшого набора вариантов достаточно функций и структур; отдельная иерархия классов для каждого фильтра не вводилась.')
    r.p('Guided filter реализован через boxFilter из стандартного imgproc. Для сборки нужны только core, imgproc и imgcodecs, без модуля ximgproc. Это уменьшает число внешних компонентов и сохраняет возможность проверить формулы фильтра непосредственно в исходниках. Обратная сторона решения - необходимость собственных проверок численной реализации.')
    r.h('6 4 Разделение сегментации и оценки')
    r.p('Инструмент run_experiments.py запускает собранную C++ программу отдельным процессом. Он читает протокол разделения, перебирает конфигурации на настроечной части и сохраняет selection.json. Контрольные маски используются отдельным шагом для оценки выбранных вариантов. Разметка доступна инструменту оценки, но не входит в интерфейс segment.')
    r.p('Для проверки причин ошибок сохраняются material, cores и markers, а также таблицы diagnostics.csv и regions.csv. Это реальные данные из C++ вычислений. Галерея строится из них и итоговых масок, поэтому позволяет отличить ошибку порога от ошибки маркера или распространения класса. Дополнительные файлы увеличивают объём результатов, но делают каждый этап проверяемым.')

    r.page('6 5 Воспроизведение')
    r.p('Основная программа написана на C++20 с OpenCV. Она не обращается к разметке. Python инструменты отдельно выполняют подбор параметров, оценку и подготовку иллюстраций. В репозитории сохранены конфигурации всех сравниваемых вариантов и промежуточные числовые маски.')
    r.table(['Материал','Назначение'],[
        ['walnut01/src/','Фильтры, пороги, автоматические маркеры, распространение классов и CLI'],
        ['walnut01/configs/','Протокол разделения и параметры выбранных вариантов'],
        ['walnut01/annotations/','Разметка v0.1 и описание её происхождения'],
        ['walnut01/results/','Маски, метрики, промежуточные этапы и галерея'],
        ['walnut01/tests/','Проверки обработки и расчёта метрик']],
        [6.2,10.8],'Таблица 7 - Организация материалов')
    r.p('Команды выполняются из корня репозитория. В качестве входа можно указать один TIFF или папку. Версия результатов отчёта соответствует коммиту 67e83cc [8].')
    r.code('''cmake -S walnut01 -B walnut01/build -DCMAKE_BUILD_TYPE=Release
cmake --build walnut01/build --parallel
ctest --test-dir walnut01/build --output-on-failure
walnut01/build/walnut_segment walnut01/data/images \\
  walnut01/results/my_run \\
  --config walnut01/configs/recommended.json
python walnut01/tools/run_experiments.py --reuse-selection
python walnut01/tools/visualize_results.py''')
    r.page('6 6 Проверки и измерения')
    r.p('В сборке CMake выделены три цели: библиотека walnut_core, программа walnut_segment и тестовый исполняемый файл walnut_tests. Последний подключает ту же библиотеку и запускается через CTest. Разделение целей позволяет проверить фильтр и семантику сегментации без повторной реализации алгоритма в тестах.')
    r.p('Проверенная среда: macOS, AppleClang 17, OpenCV 4.13.0. Вычисления OpenCV выполняются в одном потоке. Среднее время обработки одного сечения в сохранённом запуске основного варианта - 16,38 мс; чтение и запись файлов в эту величину не входят. Это наблюдение одного запуска, а не универсальный замер производительности.')
    r.p('Проверены сохранение константы и резкой границы фильтром, фоновые изображения, классы на искусственном сечении и тонкая перегородка. Неверный тип входа и NaN отвергаются. Четыре Python теста проверяют метрики и особые случаи отсутствующих классов. Проверка переименованного TIFF подтвердила независимость результата от имени файла.')
    r.p('При консольном запуске OpenCV ограничен одним потоком, а список входных файлов отсортирован. Это фиксирует порядок обработки и условия сохранённого измерения времени. Параметры, версия библиотеки и контрольные суммы разметки сохраняются отдельно от изображений; повторный запуск можно сопоставить с исходным экспериментом.')

    # 16 Conclusions.
    r.page('7 Выводы и ограничения')
    r.p('Подготовлена разметка 12 сечений, реализована сегментация на три класса и проведено воспроизводимое сравнение вариантов. Метод по областям без сглаживания получил средний IoU тканей 95,84% на восьми контрольных сечениях. Пороговый метод получил 66,30%.')
    r.p('Guided filter реализован и исследован как способ сглаживания с сохранением границ. Для метода по областям его результат составил 95,76%. Основным выбран вариант без фильтра; ветка с guided filter сохранена в проекте для воспроизведения сравнения.')
    r.p('Сечения 1430 и 2430 корректно определены как фоновые. На 0430 не возник ложный класс ядра. Наиболее выраженная оставшаяся ошибка - широкая перегородка на 2230, отнесённая алгоритмом к ядру.')
    r.h('Границы применимости')
    r.p('Все изображения относятся к одному ореху. Контрольные сечения исключены из подбора параметров, но могут содержать пересекающиеся с настроечной частью структуры. Результат не измеряет переносимость на новый орех.')
    r.p('Разметка полуавтоматическая, подготовлена и визуально проверена агентом. Её исходные яркостные операции создают зависимость от признаков, похожих на признаки оцениваемого метода. Высокий IoU поэтому следует понимать как согласие с этой разметкой, а не как независимую экспертную точность.')
    r.p('Параметры интенсивности и радиусы привязаны к шкале и разрешению учебных TIFF. Перед обработкой другого набора необходимо проверить яркость, толщину стенок и размер различимых фрагментов. Учебные файлы не используются для реконструкции недостающих 3D слоёв.')
    r.h('Дальнейшая проверка')
    r.p('Следующий содержательный шаг - независимый просмотр спорных перегородок человеком и проверка на другом орехе при заранее зафиксированных параметрах. При изменении разметки её следует сохранить новой версией и пересчитать все варианты одинаково, чтобы сравнение оставалось проверяемым.')
    r.p('Итоговые материалы включают исходные TIFF, числовые маски, конфигурации, таблицы метрик, карты ошибок и изображения промежуточных стадий для каждого сечения. Они доступны вместе с кодом [8].')

    # 17 References.
    r.page('Источники')
    sources=[
        ('1 Der Sarkissian H., Lucka F., van Eijnatten M. et al. A cone-beam X-ray computed tomography data collection designed for machine learning. Scientific Data. 2019. Vol. 6. Article 215.', 'https://doi.org/10.1038/s41597-019-0235-y'),
        ('2 Учебная выборка walnut01 в репозитории Полевого Д. В. Зафиксированная версия 5e7d4a2d3b07244be972243acca1a6f6a8ccc2b9.',SOURCE_URL),
        ('3 He K., Sun J., Tang X. Guided Image Filtering. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2013. Vol. 35, no. 6. P. 1397-1409.','https://doi.org/10.1109/TPAMI.2012.213'),
        ('4 Otsu N. A Threshold Selection Method from Gray-Level Histograms. IEEE Transactions on Systems, Man, and Cybernetics. 1979. Vol. 9, no. 1. P. 62-66.','https://doi.org/10.1109/TSMC.1979.4310076'),
        ('5 Dijkstra E. W. A note on two problems in connexion with graphs. Numerische Mathematik. 1959. Vol. 1. P. 269-271.','https://doi.org/10.1007/BF01386390'),
        ('6 OpenCV. Документация по фильтрации изображений и преобразованиям масок.','https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html'),
        ('7 3D Slicer. Документация модуля Segment Editor.','https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html'),
        ('8 Василенко А. А. Материалы проекта и код экспериментов. Коммит 67e83ccc96e7eeddc1f27dbc0c364e621f5f3b7c.',CODE_URL)]
    for text,url in sources:
        p=r.p(text);p.paragraph_format.first_line_indent=Cm(0)
        # Compact, meaningful hyperlink labels avoid wrapping long commit URLs.
        label=url if len(url)<90 else ('Учебные данные' if 'dvpsun' in url else 'Код и результаты зафиксированной версии')
        r.link(label,url)
    r.p('Дата обращения к электронным источникам 20 октября 2026 года.')

    # 18-23 All annotation pairs, two per page.
    codes=sorted(PROTOCOL['development']+PROTOCOL['control'])
    for index in range(0,12,2):
        r.page('Приложение А Размеченные сечения' if index==0 else 'Приложение А Продолжение')
        if index==0:r.p('Слева показан исходный TIFF в общем окне яркости, справа - разметка v0.1. Оранжевый цвет обозначает скорлупу и перегородки, голубой - ядро, тёмный - воздух и фон.')
        for code in codes[index:index+2]:
            r.picture(FIG/f'annotation_{code}.png',16.0,f'Сечение walnut01.{code} и соответствующая разметка')
            if code in ['1430','2430']:
                p=r.p('Ткань ореха не различима. Всё сечение отнесено к фону.');p.paragraph_format.space_after=Pt(2)
            elif code=='0430':
                p=r.p('Скорлупа и перегородки присутствуют; уверенно различимого ядра нет.');p.paragraph_format.space_after=Pt(2)

    # 24 Per-image table and links to every intermediate image.
    r.page('Приложение Б Качество по сечениям')
    r.p('Результаты основного варианта по каждому сечению приведены отдельно. Они дополняют суммарные показатели контрольной части, но не заменяют их. Прочерк означает отсутствие класса и в разметке, и в результате.')
    rows=[]
    for code in codes:
        row=next(x for x in PER_IMAGE if x['variant']==SELECTED and x['image']==code)
        rows.append([code,'Настройка' if row['subset']=='development' else 'Контроль',
                     pct(float(row['iou_shell'])) if row['iou_shell'] else '-',
                     pct(float(row['iou_kernel'])) if row['iou_kernel'] else '-'])
    r.table(['Сечение','Часть','IoU скорлупы %','IoU ядра %'],rows,[3.0,4.0,5.0,5.0],'Таблица 8 - Качество на отдельных сечениях')
    r.p('Полный набор иллюстраций содержит для каждого из 12 изображений исходник, варианты фильтрации, базовую сегментацию, маску тканей, центральные области, автоматические маркеры, итог и карту ошибок. Фильтры показаны как сравниваемые ветки; основной вариант работает без сглаживания.')
    r.link('Все изображения промежуточных этапов',f'{REPO}/tree/{CODE_COMMIT}/walnut01/results/visuals')
    r.link('Числовые результаты по каждому сечению',f'{REPO}/blob/{CODE_COMMIT}/walnut01/results/per_image_metrics.csv')
    r.link('Сводные показатели и матрицы ошибок',f'{REPO}/blob/{CODE_COMMIT}/walnut01/results/metrics.json')
    r.p('Для автономного просмотра можно открыть results/gallery.html вместе с папкой results/visuals. Конфигурация recommended.json соответствует основному варианту; guided_regions.json воспроизводит вариант с фильтром, сохраняющим границы.')
    # Enforce the requested typography and unshaded tables, including inherited
    # Word template styles. The embedded photographs and plots are preserved.
    for parent in [r.doc.styles.element,r.doc.element]:
        for node in list(parent.xpath('.//w:shd')):node.getparent().remove(node)
        for node in parent.xpath('.//w:color'):
            node.attrib.clear();node.set(qn('w:val'),'000000')
        for node in parent.xpath('.//w:rFonts'):
            node.attrib.clear()
            for key in ['ascii','hAnsi','eastAsia','cs']:node.set(qn('w:'+key),'Times New Roman')
        for node in parent.xpath('.//w:sz | .//w:szCs'):node.set(qn('w:val'),'28')
    for math_run in r.doc.element.xpath('.//m:r'):
        pr=xml('w:rPr');pr.append(xml('w:rFonts',**{'w:ascii':'Times New Roman','w:hAnsi':'Times New Roman'}));pr.append(xml('w:sz',**{'w:val':'28'}));math_run.insert(1,pr)
    for setting in r.doc.settings.element.xpath('.//m:mathFont'):setting.set(qn('m:val'),'Times New Roman')
    r.doc.save(OUT)
    print(f'DOCX: {OUT}; planned pages: {r.count}; figures: {r.figures}')

def update_pagination(pdf_path):
    """Synchronize the contents with an exported PDF before the final render."""
    from pypdf import PdfReader
    titles=[p.text for p in Document(OUT).paragraphs if p.style.name=='Heading 1']
    found={}
    for number,page in enumerate(PdfReader(pdf_path).pages,1):
        lines=page.extract_text().splitlines()
        if lines and lines[0].strip()==str(number):lines=lines[1:]
        text=' '.join(' '.join(lines).split())
        for title in titles:
            if text.startswith(title):found.setdefault(title,number)
    required=['1 Постановка задачи и данные','2 Разметка частей ореха',
              '3 Схема обработки и базовый метод','4 Протокол эксперимента',
              '5 Результаты сравнения','6 Архитектура программы',
              '6 3 Параметры и оценка','6 5 Воспроизведение','6 6 Проверки и измерения',
              '7 Выводы и ограничения','Источники','Приложение А Размеченные сечения',
              'Приложение Б Качество по сечениям']
    missing=set(required)-found.keys()
    if missing:raise RuntimeError('Section headings not found in the rendered PDF: '+str(missing))
    PAGINATION.update(found)
    (HERE/'pagination.json').write_text(json.dumps(PAGINATION,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pagination-from',type=Path,help='Refresh contents page numbers from an exported PDF')
    args=parser.parse_args()
    if args.pagination_from:update_pagination(args.pagination_from)
    build()
