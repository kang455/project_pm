import json,argparse
from pathlib import Path
from PIL import Image,ImageDraw
from evaluate import iou
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('model');args=parser.parse_args()
cid={'helmet':1,'nohelmet':0,'forklift':2}[args.model]
rows=json.loads((ROOT/(args.model+'_holdout_predictions.json')).read_text());tiles=[]
for row in rows:
    truth=[b for c,b in row['labels'] if c==cid];used=set();extra=[];unique=[]
    boxes=sorted([(s,b) for c,s,b in row['predictions'] if c==cid and s>=.25],reverse=True)
    for score,box in boxes:
        if any(iou(box,b)>.4 for b in unique):continue
        unique.append(box)
        best=max([(iou(box,gt),i) for i,gt in enumerate(truth) if i not in used],default=(0,-1))
        if best[0]>=.3:used.add(best[1])
        else:extra.append((score,box))
    if not extra:continue
    im=Image.open(ROOT/'holdout'/args.model/'images'/row['split']/row['image']).convert('RGB');d=ImageDraw.Draw(im)
    for box in truth:d.rectangle(box,outline='lime',width=2)
    for score,box in extra:d.rectangle(box,outline='red',width=2);d.text((box[0],box[1]),str(round(score,2)),fill='red')
    d.text((5,5),row['image'],fill='white');tiles.append(im.resize((320,180)))
canvas=Image.new('RGB',(1280,180*((len(tiles)+3)//4)))
for i,im in enumerate(tiles):canvas.paste(im,((i%4)*320,(i//4)*180))
canvas.save(ROOT/(args.model+'_holdout_errors.jpg'));print('Unmatched boxes in images',len(tiles))
