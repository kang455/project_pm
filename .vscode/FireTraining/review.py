import json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
images=sorted((ROOT/'dataset_v3/images').glob('*/*_fire.jpg'))
selected=images[::max(1,len(images)//12)][:12]
sheet=Image.new('RGB',(640*3,390*4),(30,30,30));draw=ImageDraw.Draw(sheet)
for n,p in enumerate(selected):
 im=Image.open(p);d=ImageDraw.Draw(im)
 label=ROOT/'dataset_v3/labels'/p.parent.name/(p.stem+'.txt')
 for line in label.read_text().splitlines():
  _,x,y,w,h=map(float,line.split());d.rectangle(((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height),outline='red',width=2)
 px=(n%3)*640;py=(n//3)*390;sheet.paste(im,(px,py));draw.text((px+5,py+363),p.parent.name+'/'+p.name,fill='white')
sheet.save(ROOT/'label_review.jpg')
out=ROOT/'external_negatives';out.mkdir(exist_ok=True)
for name in ['?ㅽ겕由곗꺑 2026-10-05 153909.png','?ㅽ겕由곗꺑 2026-10-05 153852.png']:
 p=Path('C:/Users/a0105/OneDrive/?ъ쭊/?ㅽ겕由곗꺑')/name
 if p.exists():Image.open(p).crop((0,594,812,739)).convert('RGB').save(out/(p.stem+'.jpg'))
summary={}
for split in ['train','val','test']:
 files=list((ROOT/'dataset_v3/images'/split).glob('*.jpg'))
 positives=sum(bool((ROOT/'dataset_v3/labels'/split/(p.stem+'.txt')).read_text().strip()) for p in files)
 summary[split]=dict(images=len(files),positive=positives,negative=len(files)-positives)
(ROOT/'dataset_summary.json').write_text(json.dumps(summary,indent=2));print(summary)

