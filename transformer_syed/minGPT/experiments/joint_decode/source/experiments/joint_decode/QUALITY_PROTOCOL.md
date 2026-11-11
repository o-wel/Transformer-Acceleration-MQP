# Supplementary fresh task check, fixed before outputs

This check is added after exploratory timing but before any fresh task output.
It does not select or change the kernel, weights or dispatch. It is separate
from the timing protocol and tests the original unmerged dynamic-Flash path
against the joint fused path using exactly the same native-FP16 model.

Generate 60 new fictional matches with seed 20261107, original training team
names, halftime scores 25..55 and final scores 65..105, with distinct scores
within each pair. Exclude all score quadruples in the original sports splits
and the completed paper study's fresh validation/test/counterfactual splits.
Use the existing deterministic question/answer generator for winner, halftime
and comeback: 180 questions, 60 independent match groups. These measure fresh
within-template/task behavior, not new-language or real-world generalization.
Save and hash the exact inputs before the GPU job.

Fixed 64-token greedy generation in both modes; trim the first EOS and next
User turn before scoring. Preserve raw generated IDs, answer texts and
reference/field scores. Report exact output agreement and normalized exact
reference accuracy. Compute answer-token NLL including EOS by first caching
the prompt and then teacher-forcing each answer token. Report absolute NLL
difference; the declared runtime equivalence gate is <=0.01 nats/token.

Use the existing scoring.py without modification. Cluster paired accuracy
differences by match, with 5000 bootstrap draws and fixed seed 20261107.
With only one adapter training seed and 60 matches, this is supplementary
evidence. Do not claim population-wide noninferiority from a zero-width
bootstrap when every pair agrees. For zero match-level decoded-answer
discordance, also report the one-sided 95% binomial upper bound
1 - 0.05^(1/60) for the probability of a match with any disagreement; this
avoids interpreting zero observed differences as proof of equality.

The fresh accuracy assessment is descriptive: no post-result margin selection,
no early stopping and no retuning. Any NLL failure or material answer regression
must limit the speed claim to an unvalidated implementation. Preserve source,
checkpoint/data hashes, GPU/Slurm metadata and before/after model hashes.
