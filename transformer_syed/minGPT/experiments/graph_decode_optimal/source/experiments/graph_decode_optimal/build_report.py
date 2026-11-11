"""Build matching Markdown, standalone HTML and PDF from measured JSON evidence."""
import base64
import csv
import hashlib
import html
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
STUDY=ROOT/'experiments/paper_hyperparameters/runs/study-20261105-v2'
BLUE='#14566c';TEAL='#138a86';INK='#16313e';LIGHT='#edf4f6'


def figures(summary,selection):
    out=HERE/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,3.2));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3.2)
    boxes=[(.05,1.25,1.65,'GPU token\nand position'),(2.0,1.25,1.65,'Embedding\nand layers'),
        (3.95,1.0,2.0,'LoRA expansion\n+ KV append\n+ streaming attention'),
        (6.25,1.25,1.6,'Remaining\nGPT operations'),(8.15,1.25,1.8,'Greedy selection\nand state update')]
    for x,y,w,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,1.0,boxstyle='round,pad=.04',facecolor=LIGHT,edgecolor=BLUE,lw=1.4))
        ax.text(x+w/2,y+.5,label,ha='center',va='center',fontsize=10,color=INK)
    for i in range(4):
        x,y,w,_=boxes[i];nx,ny,nw,_=boxes[i+1]
        ax.annotate('',xy=(nx-.07,ny+.5),xytext=(x+w+.05,y+.5),arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.4))
    ax.annotate('',xy=(.8,1.17),xytext=(9.0,1.15),arrowprops=dict(arrowstyle='->',connectionstyle='bar,fraction=-.13',color=TEAL,lw=1.6))
    ax.text(5,.25,'One CPU graph replay per new token; changing state stays on the GPU',ha='center',color=TEAL,fontsize=11)
    ax.text(5,2.85,'Prefill: forced PyTorch FlashAttention builds the adapted cache once per request',ha='center',color=INK,fontsize=11)
    fig.tight_layout();fig.savefig(out/'pipeline.png',dpi=200,bbox_inches='tight');fig.savefig(out/'pipeline.svg',bbox_inches='tight');plt.close(fig)
    scores=np.array([[next(x['score_ms'] for x in selection['candidates'] if x['bn']==b and x['warps']==w)
        for w in (1,2,4,8)] for b in (32,64,128,256)],dtype=float)
    fig,ax=plt.subplots(figsize=(7.8,3.0));im=ax.imshow(scores,cmap='YlGnBu_r',aspect='auto')
    ax.set_xticks(range(4),['1','2','4','8']);ax.set_yticks(range(4),['32','64','128','256'])
    ax.set_xlabel('Warps per program');ax.set_ylabel('Cached tokens per tile')
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{scores[i,j]:.2f}',ha='center',va='center',fontsize=10,
            color='white' if scores[i,j]<np.nanmedian(scores) else INK)
    selected=selection['selected'];i=(32,64,128,256).index(selected['bn']);j=(1,2,4,8).index(selected['warps'])
    ax.add_patch(Rectangle((j-.47,i-.47),.94,.94,fill=False,edgecolor='#eb6f3f',linewidth=3))
    ax.set_title('Discovery: geometric mean request latency (ms); lower is better')
    fig.colorbar(im,ax=ax,label='ms');fig.tight_layout();fig.savefig(out/'runtime-search.png',dpi=200);fig.savefig(out/'runtime-search.svg');plt.close(fig)
    keys=['primary','beyond_previous_fusion','graph_replay','lora_fusion','runtime_tuning','versus_merged','merged_versus_original']
    labels=['Original / selected graph','Previous fusion / selected graph','Eager / selected graph',
        'Ordinary LoRA graph / fused graph','Default graph tiles / selected tiles','Merged graph / unmerged graph','Original / merged graph']
    fig,ax=plt.subplots(figsize=(9,3.8))
    for i,key in enumerate(keys):
        x=summary['contrasts'][key]['wall_ms'];v=x['ratio'];lo,hi=x['ci95']
        ax.errorbar(v,6-i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color=BLUE if v>=1 else '#b6562a')
    ax.set_yticks(range(7),labels[::-1]);ax.axvline(1,color='gray',linestyle='--',lw=1)
    ax.set_xscale('log');ax.set_xlabel('Paired latency ratio; higher favors the denominator')
    ax.set_title('Independent confirmation: five model-seed/allocation units, eight fixed workloads')
    ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(out/'speedups.png',dpi=200);fig.savefig(out/'speedups.svg');plt.close(fig)


def document(summary,selection):
    blocks=[]
    def add(t,**kw):blocks.append(dict(type=t,**kw))
    def p(s):add('p',text=s)
    def h(s):add('h',text=s)
    def table(headers,rows):add('table',headers=headers,rows=rows)
    def page():add('page')
    def ratio(k):
        x=summary['contrasts'][k]['wall_ms'];return f"{x['ratio']:.3f}x",f"{x['ci95'][0]:.3f}–{x['ci95'][1]:.3f}x"
    primary,ci=ratio('primary');previous,pci=ratio('beyond_previous_fusion');sel=selection['selected']
    inherited=json.loads((STUDY/'selection.json').read_text());q=summary['quality']
    add('title',text='Graph-Compatible Joint Decoding for minGPT')
    add('subtitle',text='Methodology, validation-selected hyperparameters, speedups and answer quality | November 10, 2026')
    p(f'This report evaluates a reusable CUDA graph runtime that combines an unmerged LoRA adapter, a full KV cache, and fused streaming attention. '
      f'Using the previously selected rank-2 model setting and a newly selected decoder launch configuration, the runtime achieved {primary} speedup '
      f'over the same trained model using dynamic-cache forced FlashAttention (95% interval {ci}). It achieved {previous} over the preceding Python fusion runtime ({pci}).')
    p('The primary comparison holds model weights fixed. Graph replay reduces repeated CPU launch overhead; fusion removes a smaller intermediate-computation boundary. '
      'The report uses “best tested under the declared objective,” not global optimality. The earlier 4.474x result used a different rank-4 checkpoint and is kept separate.')
    table(['Selected component','Setting','Evidence'],[
        ['LoRA rank / alpha','2 / 8 (scale = 4)','Inherited validation-based 45-setting study'],
        ['Training learning rate','0.001, constant','Inherited selected training configuration'],
        ['Cache / prefill','1,024 positions / one full prefill','Full-context cache; no truncation or eviction'],
        ['Decode history tile',str(sel['bn']),'New 16-configuration runtime search'],
        ['Decode program warps',str(sel['warps']),'Selected jointly with tile size'],
        ['Serving precision','Native FP16; TF32 disabled','Identical policy in every new runtime comparison']])
    add('image',file='pipeline.png',width=510)
    p(f"Fresh exact-reference accuracy averaged {q['original']['mean_exact_accuracy']:.2%} for the original runtime and "
      f"{q['selected_graph']['mean_exact_accuracy']:.2%} for the selected graph runtime. These results use five trained seeds and 360 fresh questions from 120 match groups. "
      'Better checkpoint quality relative to the old exploratory model is not an accuracy gain caused by graph replay.')
    page();h('1. What the implementation changes')
    table(['Architecture / adaptation','Setting'],[
        ['Blocks / heads / embedding / head width','12 / 12 / 768 / 64'],
        ['Vocabulary / learned context','50,257 / 1,024'],
        ['LoRA targets / trainable parameters','QKV, attention output and both MLP projections in all blocks; 294,912 parameters across 48 adapted projections']])
    p('Prefill processes the prompt with forced PyTorch FlashAttention, then copies each layer’s adapted keys and values into a fixed-capacity buffer. '
      'The first generated token comes from the final prompt logit. Subsequent tokens use one captured GPU decoding step per token.')
    p('The original Python path concatenates new K/V tensors onto the history and launches many small operations for every token. '
      'The graph-compatible path retains stable addresses for cache buffers, current token, current learned position, output buffer and write index. '
      'Those tensor values change on the GPU while their shapes and addresses remain compatible with replay.')
    table(['Stage','Operation','Why it helps'],[
        ['Static KV storage','Write new adapted K/V into the next slot','Avoid repeatedly copying the entire history'],
        ['Device-resident position','Attention loads the live valid length','Growing context does not require a new graph'],
        ['Joint QKV fusion','Expand LoRA, append K/V and attend in one kernel','Remove expanded-update tensors and separate launches'],
        ['Streaming attention','Read valid K/V tiles with online softmax','Avoid an attention-probability matrix'],
        ['Full-step graph replay','Replay embeddings, layers, sampler and state updates','Consolidate repeated CPU submission work']])
    p('For one projection, y = W0 x + (alpha/r) B(Ax). The base projection and z = Ax remain outside the fused attention kernel. '
      'Inside it, each head forms q = q0 + (alpha/r) Bq z, k = k0 + (alpha/r) Bk z and v = v0 + (alpha/r) Bv z. '
      'It writes adapted K/V and consumes current values immediately. The three other LoRA projections per block remain ordinary operations inside the graph. [1]')
    p('Online softmax maintains a running maximum, normalization sum and weighted value sum as it visits cache tiles. '
      'Historical K/V remain fully adapter-specific; they are not approximated by a different adapter’s or base model’s cache. '
      'The valid history is read according to the device position; unused capacity is not treated as real context. [2]')
    p('Example: after a 128-token prompt, prefill predicts the first answer token and leaves 128 cached inputs. '
      'The first replay processes that token at position 128 and predicts the second; the cache then contains 129 inputs. '
      'The next replay advances the position to 130 and the cache to 130 inputs.')
    p('A 128-token answer uses one prefill and 127 one-token replays. A graph records operations and dependencies, not a fixed answer. '
      'Different prompts still produce different outputs. The decode kernel is an experimental FlashAttention-style implementation, '
      'not the official FA1 or FA2 decode implementation. Official PyTorch FlashAttention handles prefill. Graph replay and its stable-address requirements follow established CUDA/PyTorch mechanisms. [3,4]')
    page();h('2. How the hyperparameters were selected')
    p('The inherited study searched 45 training settings: rank {2,4,8,16,32}, alpha/rank {1,2,4}, and learning rate {0.0001,0.0003,0.001}. '
      'Every setting used seeds 11,22,33, yielding 135 discovery training runs. Each run received 2,400 successful updates, batch 8, sequence length 128, '
      'AdamW with zero weight decay, clipping at norm 1 and zero dropout. Validation occurred every 300 updates; the lowest validation NLL selected a checkpoint.')
    p('The operational objective selected the lowest mean warm training-step latency among settings whose mean validation NLL was within 0.01 of the quality-best setting. '
      'It selected rank 2 / alpha 8 / LR 0.001. The validation-quality-best setting was rank 32 / alpha 64 / LR 0.0003. '
      'These are different objectives. Five new seeds (101,202,303,404,505) were independently trained after selection; their completed checkpoint metadata and hashes were verified here. '
      'Learning rate affects the trained checkpoint; it is not an inference-time launch parameter. Rank and alpha also determine the stored adapter shape and serving scale.')
    p('The inherited cache sweep selected chunk 1024, which means one full prefill for these workloads. Its FA1-derived prototype selected query/key tiles 64/128 and four warps. '
      'That query tile does not transfer to a one-query decode kernel. This report therefore inherited the trained adapter/cache choice but ran a new decoder-specific search.')
    p('The new search crossed history tiles {32,64,128,256} with warps {1,2,4,8}. Three fresh A30 allocations used discovery checkpoints 11/22/33. '
      'Each measured batch {1,4}, prompt {32,896}, output 32, with two warmups and five shuffled paired blocks. '
      'Every configuration had to pass the declared numerical gates in all allocations. Selection minimized equally weighted geometric mean per-workload median wall latency.')
    add('image',file='runtime-search.png',width=480)
    default=next(x for x in selection['candidates'] if x['bn']==128 and x['warps']==4)
    p(f"The selected configuration was tile {sel['bn']} / {sel['warps']} warps, with discovery score {sel['score_ms']:.3f} ms. "
      f"The original 128/4 control scored {default['score_ms']:.3f} ms. All 16 configurations passed. "
      'selection.json was hashed before confirmation; fresh test answers did not select any setting. '
      'The finite grid has boundaries: a winner at tile 256 and/or eight warps does not prove that a larger untested setting would be worse. Separating discovery and confirmation addresses selection-bias concerns. [5]')
    page();h('3. Controlled confirmation methodology')
    table(['Design element','Specification'],[
        ['Hardware','Five new A30 allocations; physical GPU UUIDs retained'],
        ['Model units','Five independently trained selected-setting seeds; one allocation per seed'],
        ['Held-out shapes','Batch {1,4} × prompt {128,512} × output {32,128}: eight shapes'],
        ['Repetitions','Three warmup blocks excluded; ten measured paired blocks per shape'],
        ['Pairing','Same fresh synthetic prompt across modes; shuffled order within block'],
        ['Sampling / context','Fixed output count; greedy softmax/topk; prompt + output ≤ 1024'],
        ['Precision','Same native-FP16 weights; TF32 off; eval/inference mode'],
        ['Timers','Synchronized host wall time; CUDA-event time also recorded'],
        ['Included costs','Prefill, initial cache copy/reset, decode and output assembly'],
        ['Excluded costs','Model load, graph construction/capture, compilation, host IO and mode installation'],
        ['Memory controls','Same resident model/control set throughout paired comparisons'],
        ['Uncertainty','5,000 paired bootstrap draws over seed/allocation units and blocks; shapes fixed']])
    p('Primary estimand: exp(mean(log(original latency / selected-graph latency))), equally weighting the eight shapes and five units. '
      'The five units combine training-seed and allocation effects; one allocation per seed cannot disentangle those variance components. '
      'Workload shapes are a fixed design rather than independently sampled tasks. Secondary intervals are descriptive and not adjusted for multiple comparisons.')
    table(['Control','Purpose'],[
        ['Original dynamic-cache forced Flash','Matched trained-model deployment reference'],
        ['Previous Python joint fusion','Progress beyond the first fusion implementation'],
        ['Eager fixed-buffer fused runtime','Isolate graph replay on identical runtime mathematics'],
        ['Ordinary-LoRA graph, selected tiles','Isolate LoRA fusion after graph replay'],
        ['Fused graph, original 128/4 tiles','Isolate the new launch-parameter tuning'],
        ['Fused graph, selected tiles','Candidate'],
        ['Merged-LoRA graph, selected tiles','Strong fixed-adapter serving control']])
    p('No isolated speedups are multiplied. All ratios come from directly measured matched modes. '
      'Synthetic timing prompts control shape and length; fresh task prompts separately assess answer quality. '
      'No prior rank-4 timing samples or old test answers enter these estimates.')
    page();h('4. Measured speedups')
    labels=[('primary','Original / selected graph'),('beyond_previous_fusion','Previous fusion / selected graph'),
        ('graph_replay','Eager / selected graph'),('lora_fusion','Ordinary-LoRA graph / fused graph'),
        ('runtime_tuning','Original-tile graph / selected-tile graph'),('versus_merged','Merged graph / unmerged graph'),
        ('merged_versus_original','Original / merged graph')]
    table(['Comparison','Speedup','95% interval'],[[label,*ratio(key)] for key,label in labels])
    add('image',file='speedups.png',width=480)
    table(['Batch / prompt / output','Original ms','Selected graph ms','Merged graph ms'],[
        [' / '.join(map(str,w['shape'])),f"{w['modes']['original']['median_ms']:.2f}",
         f"{w['modes']['selected_graph']['median_ms']:.2f}",f"{w['modes']['merged_graph']['median_ms']:.2f}"] for w in summary['workloads']])
    tune=summary['contrasts']['runtime_tuning']['wall_ms']
    p(f"The launch-tuning contrast is {tune['ratio']:.4f}x (95% interval {tune['ci95'][0]:.4f}–{tune['ci95'][1]:.4f}x). " +
      ('Its interval supports a positive transfer of the discovery choice.' if tune['ci95'][0]>1 else
       'Its interval does not establish a positive tuning benefit; report the selection as best on discovery rather than a confirmed improvement over 128/4.'))
    regressions=[(i,v) for i,v in enumerate(tune['by_shape']) if v<1]
    if regressions:
        index,worst=min(regressions,key=lambda x:x[1]);shape=summary['workloads'][index]['shape']
        p(f"The selected launch setting was slower than 128/4 on {len(regressions)} of eight held-out shapes. "
          f"The worst tuning ratio was {worst:.4f}x at batch {shape[0]}, prompt {shape[1]}, output {shape[2]}. "
          'It is a weighted workload-family choice, not a per-shape optimum; all mode timings are in TIMINGS.csv.')
    page();h('5. Answer quality')
    p('A new generator seed produced 120 fresh fictional matches and 360 winner/halftime/comeback questions before runtime search. '
      'Score quadruples exclude the original sports splits, completed tuning-study splits and both preceding fresh graph/fusion sets. '
      'The task uses familiar training team names and established templates; this is within-template generalization, not a real-world language benchmark.')
    p('The same 360 questions were evaluated under all five trained model seeds: 1,800 paired model/question cases per mode, but only 120 unique match groups. '
      'Generation used equal-token-length batches of up to four, identically grouped across modes, with 64 greedy tokens. '
      'Scoring trims the first EOS and subsequent User turn. Raw token sequences are retained separately from trimmed answer agreement. '
      'Answer-token NLL includes EOS and uses cached teacher forcing. NLL differences must be ≤0.01 for every seed.')
    table(['Runtime','Mean exact accuracy','Crossed 95% interval','Mean NLL'],[
        [name,f"{q[key]['mean_exact_accuracy']:.2%}",
         'Degenerate bootstrap; see bound below' if q[key].get('all_observed_correct') else f"{q[key]['ci95'][0]:.2%}–{q[key]['ci95'][1]:.2%}",
         f"{q[key]['mean_nll']:.6f}"]
        for key,name in [('original','Original'),('selected_graph','Selected graph'),('merged_graph','Merged graph')]])
    table(['Training seed','Original exact','Graph exact','Merged exact'],[
        [str(q['original']['by_seed'][i]['seed'])]+[f"{q[k]['by_seed'][i]['reference_exact']:.2%}" for k in ('original','selected_graph','merged_graph')]
        for i in range(5)])
    for name,label in [('selected_graph','Selected graph'),('merged_graph','Merged graph')]:
        x=q['comparisons'][name]
        p(f"{label}: {x['exact_answer_agreement']}/1,800 trimmed answers agreed with original; {x['raw_sequence_agreement']}/1,800 full raw sequences agreed. "
          f"Mean exact-accuracy difference {100*x['mean_accuracy_difference']:+.3f} percentage points; crossed 95% interval "
          f"{100*x['difference_ci95'][0]:+.3f} to {100*x['difference_ci95'][1]:+.3f} points. "
          f"Maximum per-seed absolute NLL difference {x['max_seed_absolute_nll_difference']:.6g}; all NLL gates passed: {x['all_nll_gates_passed']}.")
    if q['selected_graph'].get('all_observed_correct'):
        p(f"All observed questions were correct for the selected graph under all five fixed models. Rather than treating a 100%–100% "
          f"bootstrap interval as certainty, the one-sided 95% upper bound on a fresh match having any error is "
          f"{q['selected_graph']['any_error_match_probability_upper95']:.2%}. This bound is conditional on the fixed tested models/generator; "
          'it does not cover unseen training seeds, task domains or model families.')
    p('Quality intervals jointly resample seeds and match groups, retaining related questions. Zero observed differences do not prove population equality. '
      'Accuracy differences from the old rank-4 model reflect training/checkpoint selection, not graph replay.')
    page();h('6. Setup, memory and practical deployment')
    table(['Batch','Selected setup ms','Incremental runner resident MiB'],[
        [str(b),f"{summary['setup'][str(b)]['selected_graph']['total_ms']:.2f}",
         f"{summary['setup'][str(b)]['selected_graph']['incremental_resident_mib']:.2f}"] for b in (1,4)])
    p('Observed setup includes allocation, warmup and graph capture with a populated on-disk compilation cache. '
      'It excludes common model loading and is not a clean-machine cold-start measurement. '
      'The comparison’s total resident allocation contains both model variants and all control runners; it is not a single-runner deployment footprint. '
      'Persistent graph/cache memory and incremental request peaks are reported separately.')
    table(['Batch / prompt / output','1 request ms','10 requests ms/req','100 requests ms/req','Break-even'],[
        [' / '.join(str(x[k]) for k in ('batch','prompt','new_tokens')),f"{x['request_1_ms']:.2f}",
         f"{x['request_10_ms']:.2f}",f"{x['request_100_ms']:.2f}",str(x['break_even_requests'])] for x in summary['amortization'] if x['new_tokens']==128])
    p('Amortized latency is (observed setup + N × median warm request latency) / N. Break-even compares this setup cost with the matched original runtime. '
      'The table shows representative 128-token generations; AMORTIZATION.csv contains all eight shapes. '
      'Real model loading, uncached compilation, tokenization and service scheduling add deployment costs not estimated by this formula.')
    p('A runner is tied to one model/adapter, batch size, rank, scale, precision and device. Prompt lengths may vary without recapture; each prompt resets its cache. '
      'The prototype has no continuous batching, concurrent requests, mixed adapters, arbitrary sampling or context rollover. '
      'Adapter replacement requires a new runner and fresh cache. Reserving a full cache costs more memory on short requests. '
      'The asymptotic attention cost remains proportional to valid history length; the improvement is scheduling and data movement, not skipped context or layers.')
    h('Numerical checks')
    table(['Numerical runtime','Cases','Max logit error','Argmax differences'],[
        [n,str(x['cases']),f"{x['max_error']:.6g}",str(x['argmax_disagreements'])] for n,x in summary['checks'].items()])
    p('Logit gates use atol=0.1 and rtol=0.02, so a maximum absolute error above 0.1 can still pass for sufficiently large reference logits. '
      'Eager and graph execution of the same selected fused kernel were checked for bitwise-equal logits. '
      'First/last cache positions, stale-prompt isolation and rejection of overflow/batch mismatch passed. Model parameter hashes were unchanged by inference.')
    page();h('7. Scope, prior art and reproducibility')
    table(['Evidence source','What it establishes','What it does not establish'],[
        ['Inherited 45-setting study','Operational training/cache settings under a declared quality/cost objective','Global optimality or decoder-kernel optimality'],
        ['Earlier rank-4 graph experiment','4.474x repository improvement for that model/workload family','Performance of the new rank-2 model'],
        ['New 16-setting runtime search','Best discovery launch choice among the tested configurations','Best possible tile/warp policy on every workload'],
        ['New matched five-seed confirmation','Runtime/quality behavior with selected checkpoints and fresh data','Cross-model/task/GPU serving superiority']])
    p('CUDA graph replay, graph-compatible LoRA execution, online-softmax attention and adapter/cache management have prior art. '
      'The narrow contribution is a reproducible implementation/evaluation of current-token LoRA expansion, full-cache append and streaming attention inside a graph-compatible GPT-2 decode step. '
      'This report does not establish publication novelty. Production fused-KV FlashAttention, production multi-adapter graph engines, additional models/ranks and GPU families remain missing external-validity comparisons. [6,7]')
    p(f"The five confirmation allocations used {len(summary['physical_gpu_uuids'])} physical A30 GPUs. GPU clocks were not fixed, and comprehensive co-tenant isolation was not imposed. "
      'Paired shuffled modes reduce timing-state effects but do not remove them. One allocation per seed means seed and allocation uncertainty cannot be separated. '
      'The fresh quality set is synthetic and has only 120 independent match groups. Native-FP16 serving differs from the inherited study’s FP32-weight/autocast policy.')
    p(f"New search plus confirmation/quality allocation time: {summary['gpu_seconds']/60:.2f} GPU-minutes across eight jobs. "
      'This includes loading, compilation, capture and validation. No new training was performed. The inherited study’s 135 discovery runs used '
      '324,000 successful updates; its selected/control confirmation used another 24,000 updates. Those training costs are separate from deployment speedups.')
    table(['Reproducibility artifact','Contents'],[
        ['PROTOCOL.md / manifest.json','Predeclared design, inherited selection hash and frozen fresh-data hash'],
        ['selection.json / selection.sha256','All runtime candidates and discovery-only winner'],
        ['SUMMARY.json / TIMINGS.csv','Paired estimands, intervals and per-workload timings'],
        ['AMORTIZATION.csv / ACCOUNTING.txt','Setup assumptions and Slurm allocation costs'],
        ['results/*.json and logs','Checkpoint/source hashes, raw outputs, timings, quality and failure evidence'],
        ['source/ and ARTIFACT_SHA256.json','Archived executable sources and content hashes']])
    p('Reproduction requires the existing cached GPT-2 base/tokenizer and trained adapter checkpoints whose hashes are recorded. '
      'The report package retains sources and raw evidence but does not duplicate large pretrained weights. '
      'Prior production model files and earlier study outputs are preserved.')
    page();h('References and runtime-search appendix')
    refs=[('Hu et al.','LoRA: Low-Rank Adaptation of Large Language Models','https://arxiv.org/abs/2106.09685'),
        ('Dao et al.','FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness','https://arxiv.org/abs/2205.14135'),
        ('PyTorch','CUDA semantics: CUDA graphs','https://docs.pytorch.org/docs/stable/notes/cuda'),
        ('NVIDIA','CUDA Programming Guide: CUDA Graphs','https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html'),
        ('Cawley and Talbot','On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation','https://jmlr.org/papers/v11/cawley10a.html'),
        ('Jeon et al.','LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents','https://arxiv.org/abs/2602.01053'),
        ('Tran and Yu','EpiLoRA: Fusing Many LoRA Adapters into Low-Precision GEMMs for Serving and RL','https://arxiv.org/abs/2611.03899')]
    for i,(author,title,url) in enumerate(refs,1):p(f'[{i}] {author}. {title}. {url}')
    h('Complete tested decoder grid')
    table(['History tile','Warps','Numerical gates','Discovery score ms'],[
        [str(x['bn']),str(x['warps']),'Passed' if x['valid'] else 'Failed',f"{x['score_ms']:.3f}" if x['valid'] else 'Excluded']
        for x in selection['candidates']])
    return blocks


def render(blocks):
    font=Path(matplotlib.get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('DejaVu',str(font/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuBold',str(font/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVuBold',italic='DejaVu',boldItalic='DejaVuBold')
    body=ParagraphStyle('body',fontName='DejaVu',fontSize=9,leading=13.3,textColor=colors.HexColor(INK),spaceAfter=8,allowWidows=0,allowOrphans=0)
    title=ParagraphStyle('title',parent=body,fontName='DejaVuBold',fontSize=24,leading=29,spaceAfter=12)
    subtitle=ParagraphStyle('subtitle',parent=body,fontSize=10,leading=15,textColor=colors.HexColor(BLUE),spaceAfter=13)
    heading=ParagraphStyle('heading',parent=body,fontName='DejaVuBold',fontSize=15,leading=20,spaceAfter=12,keepWithNext=True)
    cell=ParagraphStyle('cell',parent=body,fontSize=7.4,leading=10.5,spaceAfter=0)
    header=ParagraphStyle('header',parent=cell,fontName='DejaVuBold',textColor=colors.white)
    story=[];md=[];web=[]
    for block in blocks:
        kind=block['type']
        if kind in ('title','subtitle','h','p'):
            text=block['text'];style={'title':title,'subtitle':subtitle,'h':heading,'p':body}[kind]
            escaped=html.escape(text)
            pdftext=re.sub(r'https?://[^\s]+',lambda m:f'<link href="{m.group(0)}" color="#14566c">{m.group(0)}</link>',escaped)
            story.append(Paragraph(pdftext,style))
            mdtext=re.sub(r'https?://[^\s]+',lambda m:'[Source]('+m.group(0)+')',text)
            md.append(('# ' if kind=='title' else '## ' if kind=='h' else '')+mdtext+'\n')
            tag={'title':'h1','subtitle':'div','h':'h2','p':'p'}[kind]
            webtext=re.sub(r'https?://[^\s]+',lambda m:f'<a href="{m.group(0)}">{m.group(0)}</a>',escaped)
            web.append(f'<{tag}>'+webtext+f'</{tag}>')
        elif kind=='page':story.append(PageBreak());web.append('<div class="pagebreak"></div>')
        elif kind=='image':
            path=HERE/'figures'/block['file'];from PIL import Image as PILImage
            with PILImage.open(path) as im:w,h=im.size
            width=min(block['width'],A4[0]-84);story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,9))
            md.append(f"![Research figure](figures/{block['file']})\n")
            data=base64.b64encode(path.read_bytes()).decode();web.append(f'<img src="data:image/png;base64,{data}" alt="Research figure">')
        elif kind=='table':
            rows=[block['headers']]+block['rows'];n=len(rows[0]);width=(A4[0]-84)/n
            data=[[Paragraph(html.escape(str(v)),header if i==0 else cell) for v in row] for i,row in enumerate(rows)]
            table=Table(data,colWidths=[width]*n,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(BLUE)),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor(LIGHT)]),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor(BLUE))]))
            story.extend([table,Spacer(1,10)])
            md.append('| '+' | '.join(block['headers'])+' |\n| '+' | '.join('---' for _ in range(n))+' |')
            md.extend('| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in block['rows']);md.append('')
            web.append('<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in rows[0])+'</tr></thead><tbody>'+''.join(
                '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows[1:])+'</tbody></table>')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c5d5dc'));canvas.line(42,36,A4[0]-42,36)
        canvas.setFont('DejaVu',7);canvas.setFillColor(colors.HexColor('#58707c'))
        canvas.drawString(42,23,'minGPT / MQP — best-tested hyperparameter report — 10 November 2026')
        canvas.drawRightString(A4[0]-42,23,str(doc.page));canvas.restoreState()
    pdf=HERE/'MINGPT_OPTIMIZATION_REPORT.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=39,bottomMargin=47,
        title='Graph-Compatible Joint Decoding for minGPT',author='MQP experiment report',subject='Methods, best-tested hyperparameters and matched speedups')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (HERE/'REPORT.md').write_text('\n'.join(md)+'\n')
    css='body{font:16px/1.6 system-ui,sans-serif;color:#16313e;max-width:1120px;margin:40px auto;padding:0 24px}h1{font-size:38px;line-height:1.2}h2{color:#14566c;margin-top:44px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th{background:#14566c;color:white;text-align:left}td,th{padding:10px;vertical-align:top}tr:nth-child(even){background:#edf4f6}img{max-width:100%;height:auto}.pagebreak{margin-top:42px}@media print{.pagebreak{break-before:page}body{font-size:10pt;max-width:none}}'
    (HERE/'REPORT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>minGPT Optimization Report</title><style>'+css+'</style></head><body>'+''.join(web)+'</body></html>')


def main():
    summary=json.loads((HERE/'SUMMARY.json').read_text());selection=json.loads((HERE/'selection.json').read_text())
    figures(summary,selection);render(document(summary,selection));print(HERE/'MINGPT_OPTIMIZATION_REPORT.pdf')


if __name__=='__main__':main()
