import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for model in ('helmet','nohelmet','forklift'):
    print('TRAINING '+model,flush=True)
    with (ROOT/(model+'_train.log')).open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'train.py'),model],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('FINISHED '+model,flush=True)
with (ROOT/'evaluate.log').open('w') as log:
    subprocess.run([sys.executable,str(ROOT/'evaluate.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('EVALUATION COMPLETE',flush=True)
