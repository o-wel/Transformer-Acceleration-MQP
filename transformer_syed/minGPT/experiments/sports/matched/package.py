"""Shareable evidence package; full-model checkpoints remain on the cluster."""
import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[3]
folder=root/'experiments/sports/matched'
selection=json.loads((folder/'selection.json').read_text())
assert selection.get('complete')
files=[p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip']]
files += [root/p for p in ['LICENSE','sports_experiment.py','sports_demo.py','lora_experiment.py','mingpt/__init__.py','mingpt/model.py','mingpt/utils.py','mingpt/measurement.py','mingpt/trainer.py','tests/test_lora.py','tests/test_measurement.py','tests/test_sports.py','tests/test_matched.py','experiments/sports/README.md','experiments/sports/METHODS.md','experiments/sports/data.py','experiments/sports/scoring.py','experiments/sports/build_report.py']]
files += list((root/'experiments/sports').glob('*.jsonl'))+[root/'experiments/sports/manifest.json']
for info in selection['methods'].values():
    paths={c['path'] for c in info['candidates']}|set(info['runs'].values())
    for path in paths: files += list((root/path).glob('*.json'))
files.append(root/selection['methods']['lora']['runs']['42']/'adapter.pt')
archive=root/'experiments/MATCHED_PROFESSOR_PACKAGE.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(set(files)):
        z.write(path,path.relative_to(root))
    z.writestr('START_HERE.txt','Corrected full fine-tuning versus LoRA comparison. Open experiments/sports/matched/results/PROFESSOR_REPORT.pdf and RESULTS.html. See experiments/sports/matched/PRESENTATION.md for the walkthrough. All raw reports, audits, synthetic data, source code, and the selected seed-42 LoRA adapter are included. Large full-model checkpoints are NOT bundled; they remain in the cluster paths in selection.json. Run sports_demo.py --matched --interactive in the original minGPT workspace. Local execution also requires pretrained GPT-2 weights/tokenizer and the documented Python environment.\n')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(archive,archive.stat().st_size,'bytes')
