import json
from pathlib import Path
from PIL import Image,ImageDraw
from evaluate import iou
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'helmet_predictions.json').read_text())['test'];tiles=[]
for row in rows:
    truth=[b for c,b in row['labels'] if c==1];used=set();extra=[]
    boxes=sorted([(s,b) for c,s,b in row['predictions'] if c==1 and s>=.25],reverse=True)
    for score,box in boxes:
        best=max([(iou(box,gt),i) for i,gt in enumerate(truth) if i not in used],default=(0,-1))
        if best[0]>=.3:used.add(best[1])
        else:extra.append((score,box))
    if not extra:continue
    im=Image.open(ROOT/'dataset/helmet/images/test'/row['image']).convert('RGB');d=ImageDraw.Draw(im)
    for box in truth:d.rectangle(box,outline='lime',width=2)
    for score,box in extra:d.rectangle(box,outline='red',width=2);d.text((box[0],box[1]),str(round(score,2)),fill='red')
    d.text((5,5),row['image'],fill='white');tiles.append(im.resize((640,360)))
canvas=Image.new('RGB',(1280,360*((len(tiles)+1)//2)))
for i,im in enumerate(tiles):canvas.paste(im,((i%2)*640,(i//2)*360))
canvas.save(ROOT/'helmet_errors.jpg')
print('Images with unmatched helmet boxes',len(tiles))
