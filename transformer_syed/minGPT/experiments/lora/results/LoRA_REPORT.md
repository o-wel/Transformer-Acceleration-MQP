# LoRA versus full fine-tuning

## 1
Research question: Can LoRA adapt the pretrained model to the held-out questions while updating fewer parameters than full fine-tuning?

## 2
Data and evaluation are taken directly from the supplied report files. This report does not retrain either model. LoRA report: C:\Users\owenv\Documents\Projects\MQP\MQP-Repo\transformer_syed\minGPT\out\lora-sports-fp16\report.json. Full-training report: C:\Users\owenv\Documents\Projects\MQP\MQP-Repo\transformer_syed\minGPT\out\full-sports-fp16\report.json.

## 3
Protocol: LoRA recorded FP16; LoRA steps 300; full-training steps 10. LoRA configuration: batch 4, length 64, rank 4, alpha 8.

## 4
Held-out answer NLL: pretrained base 2.9955 -> LoRA 0.0042; full fine-tuning 2.0696.

## 5
Expected-phrase accuracy: base 0.0% -> LoRA 100.0%. The supplied generic reports contain 12 fixed comparison questions.

## 6
LoRA model: 163,627,008 total parameters; 589,824 trainable (0.3605%); 163,037,184 frozen. Full fine-tuning trainable parameters: 163,037,184.

## 7
Storage: LoRA parameters 624.19 MiB; full parameters 621.94 MiB. Peak allocated training memory: LoRA 1278.82 MiB, full 3197.82 MiB.

## 8
Interpretation: this is a synthetic, format-sensitive task report. It does not establish broad conversational quality or quality equivalence between differently configured runs.
