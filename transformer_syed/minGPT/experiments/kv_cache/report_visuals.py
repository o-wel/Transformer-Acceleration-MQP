"""Vector diagrams and plots derived from the recorded KV-cache experiment."""
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK='#173650'
OFF='#bc682c'
ON='#247c9f'
HISTORY='#d5e9ed'


def page(title, subtitle):
    fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
    fig.text(.075,.95,title,size=20,weight='bold',color=INK)
    fig.text(.075,.92,subtitle,size=10,color='#536473')
    return fig


def paragraph(fig,y,text):
    fig.text(.075,y,'\n'.join(textwrap.wrap(text,98)),size=10,va='top',linespacing=1.45,color=INK)


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def box(ax,x,y,w,h,label,color,fontsize=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',
                              facecolor=color,edgecolor='white'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fontsize,color=INK)


def mechanism():
    fig=page('What KV caching changes','Same weights and tested outputs; less repeated computation')
    paragraph(fig,.875,'A three-token prompt is followed by two decoding steps. Orange and blue boxes are processed now; pale boxes represent reused K/V. Both paths still attend to the preceding history.')
    ax=fig.add_axes([.075,.48,.85,.31]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.94,'WITHOUT CACHE',color=OFF,weight='bold',fontsize=12)
    ax.text(.54,.94,'WITH CACHE',color=ON,weight='bold',fontsize=12)
    labels=['Prompt pass','Next step','Following step']
    for row,count in enumerate([3,4,5]):
        y=.72-row*.27
        for origin,cached in [(0,False),(.52,True)]:
            ax.text(origin,y+.10,labels[row],fontsize=9,color=INK)
            for token in range(count):
                reuse=cached and row>0 and token<count-1
                fill=HISTORY if reuse else ('#f4dac6' if not cached else '#9bd0e0')
                box(ax,origin+token*.082,y-.04,.07,.105,str(token+1),fill)
            ax.text(origin,y-.105,('Reuse saved K/V; process newest token' if cached and row else 'Process all shown tokens'),
                    fontsize=8,color='#536473')
    paragraph(fig,.455,'Within each attention layer, the new token supplies a query. That query attends to saved keys and values plus the new token\'s own key and value. The updated K/V are kept for the next step.')
    ax=fig.add_axes([.075,.18,.85,.19]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.02,.37,.21,.30,'New token\nQ, K, V','#9bd0e0')
    box(ax,.35,.68,.28,.23,'Saved history: K, V',HISTORY)
    box(ax,.35,.15,.28,.30,'Attention over\nhistory + new token','#e8eef3')
    box(ax,.77,.15,.20,.30,'Layer output\nto next layer','#e8eef3')
    for a,b in [((.23,.47),(.35,.32)),((.49,.68),(.49,.45)),((.63,.30),(.77,.30))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=INK,lw=1.6))
    ax.annotate('Append new K/V',xy=(.35,.80),xytext=(.025,.86),fontsize=8,color=INK,
                arrowprops=dict(arrowstyle='->',color=INK,connectionstyle='arc3,rad=-.15'))
    paragraph(fig,.135,'Each of the 12 layers has its own cache. Caching adds no learned parameters. At the 1,024-token context limit, this implementation rebuilds the cropped window to preserve minGPT\'s learned-position behavior; incremental speed benefits then stop.')
    return fig


def speed(report, precision='FP32', fp16_report=None):
    reports=[(report,precision)]
    if fp16_report is not None:
        reports.append((fp16_report,'FP16'))
    fig=page('Where caching speeds up generation',
             ' | '.join(f'{label} | NVIDIA A30 | 64 generated tokens per sequence | 5 measured repeats'
                        for _,label in reports))
    positions=list(range(4));labels=['32 tokens\nbatch 1','128 tokens\nbatch 1','512 tokens\nbatch 1','128 tokens\nbatch 4']
    columns=len(reports)
    for column,(current_report,current_precision) in enumerate(reports):
        left=.08+column*(.92/columns)
        width=.84/columns
        for index,name in enumerate(['pretrained','lora']):
            ax=fig.add_axes([left,.65-index*.27,width,.19])
            rows=current_report['models'][name]['performance']
            for mode,offset,color,label in [('uncached',-.18,OFF,'No cache'),('cached',.18,ON,'KV cache')]:
                values=[r['summary'][mode]['wall_ms']['mean'] for r in rows]
                errors=[r['summary'][mode]['wall_ms']['stdev'] for r in rows]
                bars=ax.bar([p+offset for p in positions],values,.34,color=color,label=label,yerr=errors,capsize=3)
                ax.bar_label(bars,labels=[f'{v:.0f}' for v in values],fontsize=7,padding=3)
            ax.set(xticks=positions,xticklabels=labels,ylabel='Request time (ms)',ylim=(0,1700))
            ax.set_title(f'{current_precision}: ' + ('Pretrained GPT-2' if name=='pretrained' else 'Trained LoRA'),
                         loc='left',fontsize=10,weight='bold',color=INK)
            style(ax)
            if index==0 and column==0:
                ax.legend(frameon=False,fontsize=8,loc='lower right',bbox_to_anchor=(1,1.015),ncol=2)
        ax=fig.add_axes([left,.14,width,.15])
        for name,offset,color,label in [('pretrained',-.16,INK,'Pretrained'),('lora',.16,ON,'LoRA')]:
            values=[r['speedup'] for r in current_report['models'][name]['performance']]
            bars=ax.bar([p+offset for p in positions],values,.29,color=color,label=label)
            ax.bar_label(bars,labels=[f'{v:.2f}x' for v in values],fontsize=8,padding=3)
        ax.axhline(1,color=OFF,ls='--',lw=1.3)
        ax.set(xticks=positions,xticklabels=labels,ylabel='Speedup',ylim=(0,3.8))
        ax.set_title(f'{current_precision}: speedup',loc='left',fontsize=10,color=INK)
        style(ax)
        if column==0:ax.legend(frameon=False,fontsize=8,loc='upper left')
    paragraph(fig,.077,'Longer prompts and batch four benefit most; short requests are near break-even. Error bars show one standard deviation. Timings exclude model loading and tokenization.')
    return fig


def memory(report, precision='FP32', fp16_report=None):
    reports=[(report,precision)]
    if fp16_report is not None:
        reports.append((fp16_report,'FP16'))
    fig=page('The speed–memory trade-off','Cache storage is temporary inference state, not additional model weights')
    labels=['32 / B1','128 / B1','512 / B1','128 / B4'];positions=list(range(4))
    columns=len(reports)
    for column,(current_report,current_precision) in enumerate(reports):
        left=.08+column*(.92/columns)
        width=.84/columns
        for index,name in enumerate(['pretrained','lora']):
            ax=fig.add_axes([left,.65-index*.26,width,.18])
            rows=current_report['models'][name]['performance']
            for mode,offset,color,label in [('uncached',-.18,OFF,'No cache'),('cached',.18,ON,'KV cache')]:
                values=[r['summary'][mode]['peak_allocated_bytes']['mean']/2**20 for r in rows]
                bars=ax.bar([p+offset for p in positions],values,.34,color=color,label=label)
                ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],fontsize=7,padding=3)
            ax.set(xticks=positions,xticklabels=labels,ylabel='Peak allocated GPU MiB',ylim=(0,940))
            ax.set_title(f'{current_precision}: ' + ('Pretrained GPT-2' if name=='pretrained' else 'Trained LoRA'),
                         loc='left',fontsize=10,weight='bold',color=INK)
            style(ax)
            if index==0 and column==0:
                ax.legend(frameon=False,fontsize=8,loc='lower right',bbox_to_anchor=(1,1.015),ncol=2)
    paragraph(fig,.34,'X-axis: prompt tokens / batch size. Memory axes start at zero. Peak allocation includes weights, masks, caches, and live temporaries. It is measured runtime memory, not simply the sum of parameter and cache storage.')
    for column,(current_report,current_precision) in enumerate(reports):
        left=.08+column*(.92/columns)
        width=.84/columns
        ax=fig.add_axes([left,.16,width,.10])
        rows=current_report['models']['pretrained']['performance']
        values=[r['summary']['cached']['cache_bytes']['mean']/2**20 for r in rows]
        bars=ax.bar(positions,values,.5,color=ON)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],fontsize=8,padding=3)
        ax.set(xticks=positions,xticklabels=labels,ylabel='Cache MiB',ylim=(0,68))
        ax.set_title(f'{current_precision}: logical K/V storage',loc='left',fontsize=10,color=INK)
        style(ax)
    cache_sizes=[]
    for _,current_precision in reports:
        bytes_per_value=2 if current_precision == 'FP16' else 4
        cache_sizes.append(f'{current_precision}: {72*bytes_per_value/4:.0f} KiB')
    paragraph(fig,.105,f'The cache requires {"; ".join(cache_sizes)} per stored token per sequence. After predicting 64 output tokens, it holds prompt length + 63 tokens. Cache storage grows with sequence length and batch size; it is already included in the peak-memory measurements above.')
    return fig


def visual_pages(report, precision='FP32', fp16_report=None):
    return [('mechanism',mechanism()),('generation-speed',speed(report, precision, fp16_report)),
            ('gpu-memory',memory(report, precision, fp16_report))]
