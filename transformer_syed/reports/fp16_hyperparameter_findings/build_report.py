"""Build a standalone vector PDF from frozen study artifacts; standard library only."""
from pathlib import Path
import json, statistics, textwrap, hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'minGPT/experiments/flash_attention/results/fp16_hyperparameter_search/runs/20261111_fp16_matched'
TASKS = ['basketball_fp16', 'spam_fp16']
data = {}
inputs = {}
for task in TASKS:
    p = ROOT / task
    def read(name):
        f = p / name
        inputs[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
        return json.loads(f.read_text())
    screens = sum([read(f'screen-{i}.json') for i in range(3)], [])
    assert len(screens) == 90
    data[task] = dict(selection=read('selection.json'), confirm=read('confirm.json'),
                      spread=read('TUNING_SPREAD.json'), tests=read('lora-test.json'))
    data[task]['medians'] = {key: {str(v): statistics.median(r['validation_nll'] for r in screens if r['config'][key] == v)
        for v in sorted({r['config'][key] for r in screens})} for key in ['rank', 'lr', 'targets']}

# Minimal PDF drawing primitives. Standard Type 1 fonts keep the artifact portable.
W, H = 595, 842
pages = []
ops = []
NAVY = (0.08, .15, .25)
TEAL = (.02, .45, .48)
GRAY = (.32, .38, .43)
def esc(s):
    return str(s).replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
def text(x, y, s, size=10, bold=False, color=NAVY):
    s.encode('ascii')
    ops.append(f'{color[0]} {color[1]} {color[2]} rg BT /{"F2" if bold else "F1"} {size} Tf 1 0 0 1 {x} {H-y} Tm ({esc(s)}) Tj ET')
def line(x, y, xx, yy, color=(.8,.84,.87)):
    ops.append(f'{color[0]} {color[1]} {color[2]} RG 0.7 w {x} {H-y} m {xx} {H-yy} l S')
def rect(x, y, w, h, color):
    ops.append(f'{color[0]} {color[1]} {color[2]} rg {x} {H-y-h} {w} {h} re f')
def para(s, y, width=92, size=10, color=NAVY):
    for row in textwrap.wrap(s, width=width):
        text(44, y, row, size, color=color)
        y += 15
    return y + 9
def heading(s, y):
    text(44, y, s, 14, True)
    return y + 25
def table(headers, rows, y, widths):
    x = 44
    rect(44,y-15,sum(widths),25,NAVY)
    for h,w in zip(headers,widths):
        text(x+8,y+1,h,9,True,(1,1,1)); x+=w
    y+=27
    for i,row in enumerate(rows):
        if i%2==0: rect(44,y-14,sum(widths),25,(.94,.96,.97))
        x=44
        for cell,w in zip(row,widths):
            text(x+8,y+1,str(cell),9); x+=w
        y+=25
    return y+15
def start(n, title, subtitle):
    global ops
    ops=[]
    rect(0,0,W,9,TEAL)
    text(44,40,'TRANSFORMER ACCELERATION MQP',9,True,TEAL)
    text(44,78,title,23,True)
    text(44,102,subtitle,10,color=GRAY)
    line(44,119,551,119)
    line(44,793,551,793)
    text(44,813,'FP16 matched study | 20261111_fp16_matched | 12 November 2026',8,color=GRAY)
    text(529,813,str(n),9,color=GRAY)
def finish(): pages.append('\n'.join(ops).encode('ascii'))

start(1,'FP16 hyperparameter findings','Measured trends, selected configurations, and practical interpretation')
y=heading('What stood out',153)
y=para('The preferred settings changed with task and training budget. A short screening run favored a high learning rate, while the confirmed finalists used lower rates. Basketball favored a small batch; spam favored a large batch and early checkpoint selection.',y)
y=heading('Selected FP16 configurations',y+10)
b=data[TASKS[0]]['selection']['winner']; s=data[TASKS[1]]['selection']['winner']
rows=[]
for label,key in [('LoRA rank','rank'),('Alpha / rank','scale'),('Learning rate','lr'),('Adapter targets','targets'),('Dropout','dropout'),('Weight decay','weight_decay'),('Effective batch','batch'),('Microbatch','microbatch'),('Sequence length','length')]:
    rows.append([label,str(b[key]),str(s[key])])
y=table(['Setting','Basketball','SMS spam'],rows,y,[205,150,152])
y=para('Both winners used rank 32, alpha/rank = 2, and adapters on all projections. They were selected by mean validation negative log-likelihood (NLL; lower is better) across seeds 42, 43, and 44.',y)
y=heading('Study context',y+9)
y=para('Each task completed 126 training trials. Screening covered 90 core configurations at 1,200 examples; later stages used 9,600 examples and explored regularization, batch/accumulation, and length. This is a finite staged search, so these settings are the best observed candidates, not global optima.',y)
y=para('FP16 denotes mixed precision: projection/attention computation and KV caches use FP16; master weights, optimizer state, and stable reductions retain FP32. Measurements used NVIDIA A30 GPUs.',y,color=GRAY)
finish()

start(2,'What the screening revealed','Balanced 90-configuration sweep | 1,200 examples | seed 42')
y=heading('Rank helped basketball more consistently',151)
y=para('Basketball median validation NLL decreased with increasing rank, with little additional median improvement from rank 16 to 32. Spam had its best screening median at rank 8, even though rank 32 won after later tuning.',y)
# Two vector bar charts share geometry, but each has its own explicitly labeled scale.
for idx,task in enumerate(TASKS):
    x0=56+idx*260; top=242; bottom=398
    vals=data[task]['medians']['rank']; maxv=.22 if idx==0 else .12
    text(x0,top-12,['Basketball','SMS spam'][idx],12,True)
    text(x0,top+4,f'Median validation NLL | axis 0 to {maxv}',8,color=GRAY)
    line(x0,bottom,x0+220,bottom)
    for j,(rank,val) in enumerate(vals.items()):
        x=x0+5+j*43; h=val/maxv*120
        rect(x,bottom-h,28,h,TEAL if idx==0 else (.22,.40,.66))
        text(x-1,bottom-h-7,f'{val:.3f}',8)
        text(x+6,bottom+17,rank,9)
    text(x0,bottom+35,'LoRA rank',9,color=GRAY)
y=heading('Learning rate and adapter placement mattered',476)
rows=[]
for lr in ['0.0001','0.0003','0.001']:
    rows.append([lr,*[f"{data[t]['medians']['lr'][lr]:.4f}" for t in TASKS]])
y=table(['Learning rate','Basketball NLL','Spam NLL'],rows,y,[205,150,152])
y=para('At the screening budget, 1e-3 had the lowest median NLL on both tasks. The final selected rates were lower: 3e-4 for basketball and 1e-4 for spam. Early rankings therefore did not identify the final winners reliably.',y)
a=data[TASKS[0]]['medians']['targets']; z=data[TASKS[1]]['medians']['targets']
y=para(f"All-projection adapters also had lower screening medians than attention-only adapters: {a['all']:.4f} vs {a['attention']:.4f} for basketball, and {z['all']:.4f} vs {z['attention']:.4f} for spam.",y)
y=para('These are medians over the other balanced screening settings, not isolated causal effects. Task losses have different meanings and should not be compared directly across tasks.',y,color=GRAY)
finish()

start(3,'Budget, batch, and reliability','Confirmation seeds and checkpoint histories explain the headline results')
y=heading('Batch size pulled in opposite directions',151)
y=para('Basketball selected effective batch 4, while spam selected batch 32. With 9,600 examples, these imply 2,400 versus 300 optimizer updates. Batch comparisons therefore mix gradient aggregation with the number of updates at a fixed example budget.',y)
y=heading('Basketball: strong results, substantial seed variability',y+5)
conf=data[TASKS[0]]['confirm']; win=b
r=[r for r in conf if r['config']==win]
y=table(['Seed','Best validation NLL','Selected step'],[[v['seed'],f"{v['validation_nll']:.6f}",v['selected_step']] for v in r],y,[100,235,172])
y=para('The winner averaged 0.00821 validation NLL (sample standard deviation 0.01082). The seed-42 value of 0.00062 is unusually strong and should be reported alongside the other seeds. Held-out primary correctness was 100%, 90%, and 100%, respectively.',y)
y=heading('Spam: checkpoint selection mattered',y+5)
r=[r for r in data[TASKS[1]]['confirm'] if r['config']==s]
y=table(['Seed','Best NLL','Final NLL','Best step / total'],[[v['seed'],f"{v['validation_nll']:.4f}",f"{v['validation'][-1]['nll']:.4f}",f"{v['selected_step']} / {v['steps']}"] for v in r],y,[65,120,120,202])
y=para('All three spam runs worsened after their best checkpoint. The selected checkpoint occurred at 25% or 50% of the training run. This is evidence of overfitting in these runs and supports selecting checkpoints using validation loss.',y)
y=heading('How large was the tuning spread?',y+5)
y=para('At 1,200 examples, worst/best observed validation NLL was 16.3x for basketball and 11.7x for spam. At 9,600 examples it was 105x and 2.34x. The 105x basketball ratio is driven partly by a near-zero best loss; these are loss ratios, not accuracy or speed gains.',y)
finish()

start(4,'Interpretation and evidence','What these experiments support, and where the conclusions stop')
y=heading('Implications for this project',151)
for title,body in [
 ('Use the screening stage to narrow the search.', 'Recheck candidates at the intended training budget. The highest screening learning rate was not the final selected rate on either task.'),
 ('Tune for each task.', 'The selected batch size and regularization differed sharply: basketball used dropout 0.1 with zero weight decay; spam used zero dropout with weight decay 0.1.'),
 ('Report confirmation seeds and checkpoint selection.', 'Basketball variability and spam overfitting would be hidden by reporting only the best run or only the final checkpoint.')]:
    text(44,y,title,11,True);y+=20;y=para(body,y)
y=heading('Limits of the evidence',y+5)
y=para('Only the top three finalists were confirmed across three seeds; screening summaries use seed 42. Later stages are selected subsets, not a full factorial search. Rank 32 is the upper tested boundary, so its selection does not establish an optimal rank beyond the tested grid.',y)
y=para('Existing held-out splits were reused. Validation tuning spread is not evidence of fresh unseen-data generalization. The spam length sweep retains the 96-token content cap; selecting a larger sequence length would not by itself establish a benefit from longer message content.',y)
y=para('These findings concern adaptation quality. They do not establish a quality-preserving serving speedup: only 126/1,440 basketball and 144/1,440 spam FP16 serving cells passed the original strict logit/token parity gate.',y)
y=heading('Source artifacts and reproducibility',y+5)
y=para('Source run: experiments/flash_attention/results/fp16_hyperparameter_search/ runs/20261111_fp16_matched, under transformer_syed/minGPT. Per-task files: screen-0/1/2.json, selection.json, confirm.json, TUNING_SPREAD.json, and lora-test.json.',y,width=90,size=9)
y=para('The accompanying build_report.py recomputes screening medians and reads winning configurations and seed histories directly from the study artifacts. evidence.json records the values and input SHA-256 hashes used for this report. No experiments or selection rules were changed.',y,width=90,size=9)
finish()

# Serialize a PDF 1.4 file with explicit cross-reference offsets.
objects=[b'<< /Type /Catalog /Pages 2 0 R >>',None,
 b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
 b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>']
page_ids=[]
for stream in pages:
    pid=len(objects)+1; sid=pid+1;page_ids.append(pid)
    objects.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {sid} 0 R >>'.encode())
    objects.append(f'<< /Length {len(stream)} >>\nstream\n'.encode()+stream+b'\nendstream')
objects[1]=f'<< /Type /Pages /Kids [{" ".join(f"{i} 0 R" for i in page_ids)}] /Count {len(pages)} >>'.encode()
pdf=bytearray(b'%PDF-1.4\n');offsets=[0]
for i,obj in enumerate(objects,1):
    offsets.append(len(pdf));pdf.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
startxref=len(pdf)
pdf.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode())
for off in offsets[1:]:pdf.extend(f'{off:010d} 00000 n \n'.encode())
pdf.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n'.encode())
out=HERE/'FP16_HYPERPARAMETER_FINDINGS.pdf';out.write_bytes(pdf)
# Compact evidence excludes long raw trial histories and generated predictions.
evidence={'study':ROOT.name,'input_sha256':inputs,'tasks':{t:{'winner':data[t]['selection']['winner'],'ranked':data[t]['selection']['ranked'],'screening_medians':data[t]['medians']} for t in TASKS}}
(HERE/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(f'Created {out} ({len(pages)} pages, {len(pdf):,} bytes)')
