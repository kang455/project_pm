import json, random, argparse
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='dataset');args=parser.parse_args()
base=ROOT/args.dataset
summary={}
tiles=[]
for model in ('helmet','nohelmet','forklift'):
    summary[model]={}
    for split in ('train','val','test'):
        files=sorted((base/model/'images'/split).glob('*.jpg'))
        counts={}
        positives=0
        for p in files:
            rows=(base/model/'labels'/split/(p.stem+'.txt')).read_text().splitlines()
            positives+=bool(rows)
            for row in rows:
                cid=int(row.split()[0]);counts[cid]=counts.get(cid,0)+1
        summary[model][split]={'images':len(files),'positive_images':positives,'boxes':counts}
        random.Random(10527).shuffle(files)
        for p in files[:4]:
            im=Image.open(p).convert('RGB');draw=ImageDraw.Draw(im)
            for row in (base/model/'labels'/split/(p.stem+'.txt')).read_text().splitlines():
                cid,x,y,w,h=map(float,row.split());x*=im.width;y*=im.height;w*=im.width;h*=im.height
                draw.rectangle((x-w/2,y-h/2,x+w/2,y+h/2),outline='red' if int(cid)==0 else 'lime',width=2)
                draw.text((x-w/2,y-h/2),str(int(cid)),fill='white')
            im=im.resize((320,180));ImageDraw.Draw(im).text((4,4),model+' '+split+' '+p.stem,fill='white');tiles.append(im)
canvas=Image.new('RGB',(1280,180*9))
for i,tile in enumerate(tiles):canvas.paste(tile,((i%4)*320,(i//4)*180))
canvas.save(ROOT/('label_review.jpg' if args.dataset=='dataset' else args.dataset+'_label_review.jpg'))
(ROOT/(args.dataset+'_summary.json')).write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
