"""Prepend reproducible LoRA diagrams and measured charts to the meeting report."""
from pathlib import Path
import json
import shutil
import statistics as stats
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from pypdf import PdfReader, PdfWriter

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'results'
INK='#173650';FULL='#bc682c';LORA='#247c9f';BASE='#8696a4'


def page(title,subtitle):
 f=plt.figure(figsize=(8.27,11.69),facecolor='white')
 f.text(.075,.95,title,size=20,weight='bold',color=INK)
 f.text(.075,.92,subtitle,size=10,color='#536473')
 return f


def para(f,y,text):
 f.text(.075,y,'\n'.join(textwrap.wrap(text,98)),va='top',size=10,linespacing=1.45,color=INK)


def style(ax):
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 ax.tick_params(labelsize=9)


def box(ax,x,y,w,h,label,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',facecolor=color,edgecolor='white'))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',size=10,color=INK)


def arrow(ax,a,b):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=INK,lw=1.5))


def main():
 selection=json.loads((OUT.parent/'selection.json').read_text())
 runs={m:[json.loads((ROOT/selection['methods'][m]['runs'][str(seed)]/'report.json').read_text()) for seed in [42,43,44]] for m in ['full','lora']}
 control=json.loads((OUT.parent/'familiar-control.json').read_text())
 for l,f in zip(runs['lora'],runs['full']):
  assert l['selected_changes']['frozen_changed_elements']==0
  for key in ['initial_base_sha256','sample_order_sha256','data_sha256','processed_training_tokens']:
   assert l[key]==f[key],key
 figs=[]
 f=page('How LoRA changes training','Same pretrained starting weights and the same new training data')
 para(f,.865,'Both approaches learn from our basketball examples. Full fine-tuning lets every original model parameter change. LoRA freezes the original parameters and learns small added matrices. The diagram shows one adapted linear projection; the experiment applies adapters throughout the transformer.')
 ax=f.add_axes([.075,.36,.85,.42]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 ax.text(.02,.95,'FULL FINE-TUNING',weight='bold',color=FULL,size=12)
 ax.text(.54,.95,'LORA FINE-TUNING',weight='bold',color=LORA,size=12)
 box(ax,.08,.75,.25,.10,'Input x','#e8eef3');box(ax,.08,.44,.25,.18,'Original W\nTRAINABLE','#f3d6c0');box(ax,.08,.14,.25,.10,'Output y','#e8eef3')
 arrow(ax,(.205,.75),(.205,.62));arrow(ax,(.205,.44),(.205,.24))
 box(ax,.65,.75,.22,.10,'Input x','#e8eef3')
 box(ax,.53,.44,.20,.18,'Original W\nFROZEN','#dbe2e8')
 box(ax,.80,.53,.15,.10,'A: down\nto rank 4','#9bd0e0');box(ax,.80,.34,.15,.10,'B: back\nto output','#9bd0e0')
 box(ax,.65,.14,.22,.10,'Add outputs','#e8eef3')
 arrow(ax,(.70,.75),(.63,.62));arrow(ax,(.83,.75),(.875,.63));arrow(ax,(.875,.53),(.875,.44));arrow(ax,(.63,.44),(.70,.24));arrow(ax,(.875,.34),(.83,.24))
 ax.text(.76,.04,'y = Wx + (alpha / rank) BAx',ha='center',size=10,color=INK)
 ax.text(.205,.04,'y = Wx',ha='center',size=10,color=INK)
 para(f,.335,'Only the blue A and B matrices train in the LoRA model. Rank = 4; alpha = 8. Biases are omitted from this conceptual diagram. LoRA adapters were applied to QKV, attention-output, and both MLP projections in all 12 transformer blocks.')
 f.text(.075,.225,'163,037,184',size=24,weight='bold',color=FULL)
 f.text(.54,.225,'589,824',size=24,weight='bold',color=LORA)
 f.text(.075,.195,'trainable in full fine-tuning',size=10,color=INK)
 f.text(.54,.195,'trainable with LoRA',size=10,color=INK)
 para(f,.13,'The untouched pretrained model is a third reference: it receives no task training. The two trained models use identical examples, sample order, batch size, and step count. This isolates parameter-efficient adaptation rather than comparing training against no training.')
 figs.append(('lora-mechanism',f))

 f=page('Total, trainable, and actually changed','These are different quantities; all counts refer to individual scalar values')
 ax=f.add_axes([.12,.47,.80,.34]);positions=[0,1,2]
 for name,offset,color,label in [('full',-.18,FULL,'Full fine-tuning'),('lora',.18,LORA,'LoRA')]:
  r=runs[name][0];values=[r['parameters']['total'],r['parameters']['trainable'],r['selected_changes']['changed_elements']]
  bars=ax.bar([p+offset for p in positions],[v/1e6 for v in values],.34,color=color,label=label)
  ax.bar_label(bars,labels=[f'{v/1e6:.3f}M' for v in values],padding=5,size=9)
 ax.set(xticks=positions,xticklabels=['Total stored','Allowed to train','Actually changed\n(seed 42)'],ylabel='Millions of parameters',ylim=(0,195));style(ax)
 ax.legend(frameon=False,loc='upper right',bbox_to_anchor=(1,1.16),ncol=2,fontsize=10)
 para(f,.40,'Total: every model parameter, including frozen weights. Trainable: parameters allowed to receive optimizer updates. Actually changed: individual values differing from their pre-training snapshots at the selected checkpoint. Being trainable does not guarantee a value changed.')
 f.text(.075,.265,'276.4x fewer',size=25,weight='bold',color=LORA)
 f.text(.075,.235,'trainable parameters with LoRA',size=11,color=INK)
 f.text(.59,.265,'0',size=25,weight='bold',color=LORA)
 f.text(.59,.235,'frozen base values changed',size=11,color=INK)
 para(f,.18,'LoRA makes 0.3605% of its total parameters trainable. Every LoRA base-weight audit passed for all three seeds. Full fine-tuning changed 123,851,224 values in seed 42; LoRA changed all 589,824 adapter values. The chart uses a zero baseline; small adapter bars are labeled explicitly.')
 figs.append(('lora-parameters',f))

 f=page('Training resources: what is saved?','NVIDIA A30 | FP32 | 1,200 steps | batch 8 | sequence length 128')
 metrics=[('Parameter storage','parameters',None,'MiB'),('Peak allocated GPU memory','training_peak_allocated_bytes',None,'MiB'),('Training-step time','training_step_seconds',None,'Seconds')]
 for i,(title,key,_,unit) in enumerate(metrics):
  ax=f.add_axes([.12,.69-i*.24,.79,.145])
  vals=[];errors=[]
  for m in ['full','lora']:
   xs=[(r['parameters']['parameter_bytes'] if key=='parameters' else r[key])/(1 if unit=='Seconds' else 2**20) for r in runs[m]]
   vals.append(stats.mean(xs));errors.append(stats.stdev(xs))
  bars=ax.bar([0,1],vals,.5,color=[FULL,LORA],yerr=errors,capsize=4)
  ax.bar_label(bars,labels=[f'{v:,.2f}' for v in vals],padding=5,size=10)
  ax.set(xticks=[0,1],xticklabels=['Full fine-tuning','LoRA'],ylabel=unit,ylim=(0,max(vals)*1.3));style(ax)
  ax.set_title(title,loc='left',color=INK,size=12,weight='bold')
 para(f,.15,'LoRA adds 2.25 MiB of FP32 weights, but reduces measured peak allocated training memory by 33.8% and training-step time by 29.8%. Runtime memory also includes gradients, optimizer states, activations, and temporary allocations. Time excludes transfers, validation, audits, checkpointing, and testing. Bars show three-seed means; error bars show sample standard deviation.')
 figs.append(('lora-resources',f))

 f=page('Quality alongside parameter efficiency','Correct primary answers | means and sample standard deviations over three seeds')
 for i,diag in enumerate([False,True]):
  ax=f.add_axes([.12,.63-i*.29,.79,.19])
  base=control['base']['metrics']['primary_correct'] if diag else runs['lora'][0]['base']['metrics']['primary_correct']
  vals=[base*100];errs=[0]
  for m in ['full','lora']:
   xs=([v['metrics']['primary_correct']*100 for v in control['methods'][m].values()] if diag else [r[m]['metrics']['primary_correct']*100 for r in runs[m]])
   vals.append(stats.mean(xs));errs.append(stats.stdev(xs))
  bars=ax.bar([0,1,2],vals,.55,color=[BASE,FULL,LORA],yerr=errs,capsize=4)
  ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=5,size=10)
  ax.set(xticks=[0,1,2],xticklabels=['Pretrained','Full fine-tuning','LoRA'],ylabel='Correct answers (%)',ylim=(0,105));style(ax)
  ax.set_title('Exploratory diagnostic: familiar team names' if diag else 'Original test: unseen team names and match scores',loc='left',size=12,color=INK,weight='bold')
 para(f,.275,'Original test: 120 questions, with test team names and score combinations absent from training; none of its full reference answers occur in training. LoRA generalized better to unfamiliar names in this experiment. Full fine-tuning often substituted training-team names.')
 para(f,.18,f"The familiar-name diagnostic was added after inspecting failures. Only names changed; scores and checkpoints stayed fixed. {control['reference_answers_seen_in_training']} of 120 reference answers overlap training. Its similar accuracy is encouraging but does not establish statistical equivalence.")
 para(f,.085,'These are synthetic, format-sensitive question-answering scores, not broad conversational quality. Both methods scored around 50% on comeback questions. Error bars are seed-to-seed variation, not confidence intervals.')
 figs.append(('lora-quality',f))

 with PdfPages(OUT/'LORA_VISUALS.pdf') as pdf:
  for i,(name,f) in enumerate(figs,1):
   f.text(.075,.02,'Source: saved matched experiment reports | Conceptual diagrams labeled separately',size=8,color='#555')
   f.text(.92,.02,f'V{i}',size=8,color='#555')
   f.savefig(OUT/f'{name}.png',dpi=140);f.savefig(OUT/f'{name}.svg');pdf.savefig(f);plt.close(f)
 details=OUT/'MEETING_REPORT_DETAILS.pdf'
 if not details.exists():shutil.copy2(OUT/'MEETING_REPORT.pdf',details)
 writer=PdfWriter()
 for path in [OUT/'LORA_VISUALS.pdf',details]:writer.append(str(path))
 temp=OUT/'MEETING_REPORT.updated.pdf'
 with temp.open('wb') as dest:writer.write(dest)
 writer.close();temp.replace(OUT/'MEETING_REPORT.pdf')
 assert len(PdfReader(OUT/'MEETING_REPORT.pdf').pages)==4+len(PdfReader(details).pages)
 print('Updated',OUT/'MEETING_REPORT.pdf')


if __name__=='__main__':main()
