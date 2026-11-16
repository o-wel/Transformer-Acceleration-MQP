# Optimization matrix: spam / fp16
Status: COMPLETE

Design: 27 off/best/poor combinations plus four jointly selected attention/cache cases.
Best and poor are observed finite-grid choices. Training configurations were fixed from prior validation.
Poor LoRA was selected by worst seed-42 full-budget validation loss, then retrained on three seeds.
Off LoRA means the selected full-fine-tuning control; this is not an unadapted pretrained model.
Attention/cache best and poor are calibrated independently per checkpoint and workload; joint cases use paired settings.
Calibration uses a distinct prompt stream and three timings; confirmation uses seven timings and three training seeds.
Strict parity is checked against each checkpoint on eager/no-cache, not between different trained checkpoints.
Failures remain diagnostic. Complete artifact coverage does not mean every optimization passed correctness.
Task quality uses natural held-out examples with canonical serving choices (batch 1, prompt 128, output 32).
SMS is a one-step label task; its cache benefit is assessed only by synthetic serving workloads.
Timing SD within an allocation and variation across model seeds are not independent hardware replications.
Original data splits are reused; full/LoRA inherited tuning search budgets are unequal.

Coverage: {"actual": {"calibration": 2592, "confirmation": 2232, "quality": 93, "training": 9}, "confirmation_status": {"ok": 648, "parity_failed": 1584}, "expected": {"calibration": 2592, "confirmation": 2232, "quality": 93, "training": 9}, "repetitions_ok": true, "status": "COMPLETE"}

## Frozen training configurations
off: {"batch": 8, "dropout": 0.0, "length": 512, "lr": 0.0001, "method": "full", "microbatch": 8, "rank": 32, "scale": 2.0, "targets": "all", "weight_decay": 0.1}
best: {"batch": 32, "dropout": 0.0, "length": 128, "lr": 0.0001, "method": "lora", "microbatch": 8, "rank": 32, "scale": 2.0, "targets": "all", "weight_decay": 0.1}
poor: {"batch": 8, "dropout": 0.0, "length": 128, "lr": 0.0003, "method": "lora", "microbatch": 8, "rank": 16, "scale": 4.0, "targets": "all", "weight_decay": 0.0}

## Task quality by state (mean across available seeds)

LoRA / Attention / Cache | seeds | metrics
best / best / best | 3/3 | accuracy=0.9866, balanced_accuracy=0.95991, spam_precision=0.96293, spam_recall=0.92473, spam_f1=0.9432, label_nll=0.040925
best / best / off | 3/3 | accuracy=0.9866, balanced_accuracy=0.95991, spam_precision=0.96293, spam_recall=0.92473, spam_f1=0.9432, label_nll=0.040925
best / best / poor | 3/3 | accuracy=0.9866, balanced_accuracy=0.95991, spam_precision=0.96293, spam_recall=0.92473, spam_f1=0.9432, label_nll=0.040925
best / joint_best / joint_best | 3/3 | accuracy=0.98703, balanced_accuracy=0.9617, spam_precision=0.96302, spam_recall=0.92832, spam_f1=0.94517, label_nll=0.04096
best / joint_poor / joint_poor | 3/3 | accuracy=0.98703, balanced_accuracy=0.96015, spam_precision=0.9664, spam_recall=0.92473, spam_f1=0.94493, label_nll=0.040953
best / off / best | 3/3 | accuracy=0.98746, balanced_accuracy=0.96195, spam_precision=0.96649, spam_recall=0.92832, spam_f1=0.94689, label_nll=0.040919
best / off / off | 3/3 | accuracy=0.98746, balanced_accuracy=0.96195, spam_precision=0.96649, spam_recall=0.92832, spam_f1=0.94689, label_nll=0.040919
best / off / poor | 3/3 | accuracy=0.98746, balanced_accuracy=0.96195, spam_precision=0.96649, spam_recall=0.92832, spam_f1=0.94689, label_nll=0.040919
best / poor / best | 3/3 | accuracy=0.9866, balanced_accuracy=0.95991, spam_precision=0.96293, spam_recall=0.92473, spam_f1=0.9432, label_nll=0.040925
best / poor / off | 3/3 | accuracy=0.9866, balanced_accuracy=0.95991, spam_precision=0.96293, spam_recall=0.92473, spam_f1=0.9432, label_nll=0.040925
best / poor / poor | 3/3 | accuracy=0.9866, balanced_accuracy=0.95991, spam_precision=0.96293, spam_recall=0.92473, spam_f1=0.9432, label_nll=0.040925
off / best / best | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043204
off / best / off | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043204
off / best / poor | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043204
off / off / best | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043266
off / off / off | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043266
off / off / poor | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043266
off / poor / best | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043213
off / poor / off | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043213
off / poor / poor | 3/3 | accuracy=0.99135, balanced_accuracy=0.97344, spam_precision=0.9779, spam_recall=0.94982, spam_f1=0.96365, label_nll=0.043213
poor / best / best | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.05677
poor / best / off | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.05677
poor / best / poor | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.05677
poor / joint_best / joint_best | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.05677
poor / joint_poor / joint_poor | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.05678
poor / off / best | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.056799
poor / off / off | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.056799
poor / off / poor | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.056799
poor / poor / best | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.056799
poor / poor / off | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.056799
poor / poor / poor | 3/3 | accuracy=0.98789, balanced_accuracy=0.97611, spam_precision=0.94352, spam_recall=0.96057, spam_f1=0.95071, label_nll=0.056799

## Confirmation by workload

Latency and peak memory are medians across seed medians. Speed ratio uses the all-off control.
Each row reports passing seeds; partial passing coverage cannot support a blanket speed claim.

Batch 1, prompt 32, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 402.211 | 819.439 | -
best / best / off | 0/3 | 422.464 | 776.559 | -
best / best / poor | 0/3 | 393.829 | 780.408 | -
best / joint_best / joint_best | 0/3 | 392.742 | 780.408 | -
best / joint_poor / joint_poor | 0/3 | 401.770 | 778.404 | -
best / off / best | 3/3 | 417.503 | 819.439 | 0.623
best / off / off | 3/3 | 438.880 | 776.559 | 0.591
best / off / poor | 3/3 | 421.046 | 780.408 | 0.618
best / poor / best | 0/3 | 400.339 | 819.439 | -
best / poor / off | 0/3 | 421.192 | 776.559 | -
best / poor / poor | 0/3 | 392.397 | 780.408 | -
off / best / best | 0/3 | 225.671 | 761.533 | -
off / best / off | 0/3 | 236.836 | 757.684 | -
off / best / poor | 0/3 | 236.861 | 799.939 | -
off / off / best | 3/3 | 256.223 | 761.533 | 1.014
off / off / off | 3/3 | 259.567 | 757.684 | 1.000
off / off / poor | 3/3 | 255.381 | 799.939 | 1.018
off / poor / best | 0/3 | 226.310 | 761.533 | -
off / poor / off | 0/3 | 237.245 | 757.684 | -
off / poor / poor | 0/3 | 236.082 | 799.939 | -
poor / best / best | 0/3 | 403.160 | 769.404 | -
poor / best / off | 0/3 | 421.987 | 767.559 | -
poor / best / poor | 0/3 | 395.762 | 771.408 | -
poor / joint_best / joint_best | 0/3 | 395.797 | 771.408 | -
poor / joint_poor / joint_poor | 0/3 | 407.078 | 810.439 | -
poor / off / best | 3/3 | 416.647 | 769.404 | 0.624
poor / off / off | 3/3 | 440.157 | 767.559 | 0.590
poor / off / poor | 3/3 | 425.308 | 771.408 | 0.610
poor / poor / best | 0/3 | 404.421 | 769.404 | -
poor / poor / off | 0/3 | 420.670 | 767.559 | -
poor / poor / poor | 0/3 | 395.354 | 771.408 | -

Batch 1, prompt 32, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 798.297 | 779.530 | -
best / best / off | 0/3 | 839.424 | 776.841 | -
best / best / poor | 0/3 | 783.080 | 782.659 | -
best / joint_best / joint_best | 0/3 | 788.827 | 782.659 | -
best / joint_poor / joint_poor | 0/3 | 798.990 | 779.530 | -
best / off / best | 3/3 | 827.908 | 779.530 | 0.623
best / off / off | 3/3 | 875.063 | 776.841 | 0.587
best / off / poor | 3/3 | 838.386 | 782.659 | 0.613
best / poor / best | 0/3 | 798.415 | 779.530 | -
best / poor / off | 0/3 | 842.809 | 776.841 | -
best / poor / poor | 0/3 | 787.967 | 782.659 | -
off / best / best | 0/3 | 470.897 | 799.940 | -
off / best / off | 0/3 | 468.190 | 757.966 | -
off / best / poor | 0/3 | 448.099 | 763.784 | -
off / off / best | 3/3 | 504.758 | 799.940 | 1.019
off / off / off | 3/3 | 514.143 | 757.966 | 1.000
off / off / poor | 3/3 | 508.432 | 763.784 | 1.011
off / poor / best | 0/3 | 470.988 | 799.940 | -
off / poor / off | 0/3 | 469.070 | 757.966 | -
off / poor / poor | 0/3 | 446.044 | 763.784 | -
poor / best / best | 0/3 | 800.395 | 770.530 | -
poor / best / off | 0/3 | 837.452 | 767.841 | -
poor / best / poor | 0/3 | 783.778 | 773.659 | -
poor / joint_best / joint_best | 0/3 | 780.900 | 773.659 | -
poor / joint_poor / joint_poor | 0/3 | 806.190 | 810.440 | -
poor / off / best | 3/3 | 828.676 | 770.530 | 0.618
poor / off / off | 3/3 | 878.866 | 767.841 | 0.587
poor / off / poor | 3/3 | 839.329 | 773.659 | 0.611
poor / poor / best | 0/3 | 799.868 | 770.530 | -
poor / poor / off | 0/3 | 837.095 | 767.841 | -
poor / poor / poor | 0/3 | 785.868 | 773.659 | -

Batch 1, prompt 128, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 405.111 | 820.285 | -
best / best / off | 0/3 | 421.676 | 777.406 | -
best / best / poor | 0/3 | 398.039 | 787.301 | -
best / joint_best / joint_best | 0/3 | 396.151 | 787.301 | -
best / joint_poor / joint_poor | 0/3 | 410.158 | 782.625 | -
best / off / best | 3/3 | 420.855 | 820.285 | 0.597
best / off / off | 3/3 | 439.414 | 777.406 | 0.571
best / off / poor | 3/3 | 425.202 | 787.301 | 0.591
best / poor / best | 0/3 | 404.996 | 820.285 | -
best / poor / off | 0/3 | 420.879 | 777.406 | -
best / poor / poor | 0/3 | 395.986 | 787.301 | -
off / best / best | 0/3 | 237.413 | 800.785 | -
off / best / off | 0/3 | 230.935 | 758.531 | -
off / best / poor | 0/3 | 227.837 | 768.426 | -
off / off / best | 3/3 | 255.184 | 800.785 | 0.982
off / off / off | 3/3 | 250.654 | 758.531 | 1.000
off / off / poor | 3/3 | 258.673 | 768.426 | 0.967
off / poor / best | 0/3 | 237.604 | 800.785 | -
off / poor / off | 0/3 | 230.910 | 758.531 | -
off / poor / poor | 0/3 | 227.806 | 768.426 | -
poor / best / best | 0/3 | 402.892 | 811.285 | -
poor / best / off | 0/3 | 418.890 | 768.406 | -
poor / best / poor | 0/3 | 394.607 | 778.301 | -
poor / joint_best / joint_best | 0/3 | 397.824 | 778.301 | -
poor / joint_poor / joint_poor | 0/3 | 410.151 | 773.625 | -
poor / off / best | 3/3 | 420.959 | 811.285 | 0.597
poor / off / off | 3/3 | 437.825 | 768.406 | 0.574
poor / off / poor | 3/3 | 424.976 | 778.301 | 0.591
poor / poor / best | 0/3 | 406.021 | 811.285 | -
poor / poor / off | 0/3 | 417.709 | 768.406 | -
poor / poor / poor | 0/3 | 395.325 | 778.301 | -

Batch 1, prompt 128, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 801.847 | 820.285 | -
best / best / off | 0/3 | 834.351 | 777.687 | -
best / best / poor | 0/3 | 784.380 | 789.411 | -
best / joint_best / joint_best | 0/3 | 784.631 | 789.411 | -
best / joint_poor / joint_poor | 0/3 | 808.509 | 783.750 | -
best / off / best | 3/3 | 829.525 | 820.285 | 0.601
best / off / off | 3/3 | 874.543 | 777.687 | 0.571
best / off / poor | 3/3 | 844.743 | 789.411 | 0.592
best / poor / best | 0/3 | 801.776 | 820.285 | -
best / poor / off | 0/3 | 840.925 | 777.687 | -
best / poor / poor | 0/3 | 785.283 | 789.411 | -
off / best / best | 0/3 | 471.371 | 800.785 | -
off / best / off | 0/3 | 456.034 | 758.812 | -
off / best / poor | 0/3 | 473.693 | 764.875 | -
off / off / best | 3/3 | 506.448 | 800.785 | 0.988
off / off / off | 3/3 | 498.608 | 758.812 | 1.000
off / off / poor | 3/3 | 506.046 | 764.875 | 0.985
off / poor / best | 0/3 | 469.798 | 800.785 | -
off / poor / off | 0/3 | 459.674 | 758.812 | -
off / poor / poor | 0/3 | 473.738 | 764.875 | -
poor / best / best | 0/3 | 803.796 | 811.285 | -
poor / best / off | 0/3 | 834.154 | 768.687 | -
poor / best / poor | 0/3 | 788.514 | 780.411 | -
poor / joint_best / joint_best | 0/3 | 782.376 | 780.411 | -
poor / joint_poor / joint_poor | 0/3 | 814.308 | 774.750 | -
poor / off / best | 3/3 | 830.059 | 811.285 | 0.600
poor / off / off | 3/3 | 870.620 | 768.687 | 0.573
poor / off / poor | 3/3 | 835.397 | 780.411 | 0.597
poor / poor / best | 0/3 | 801.314 | 811.285 | -
poor / poor / off | 0/3 | 836.251 | 768.687 | -
poor / poor / poor | 0/3 | 785.143 | 780.411 | -

Batch 1, prompt 512, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 405.819 | 826.669 | -
best / best / off | 0/3 | 422.388 | 781.770 | -
best / best / poor | 0/3 | 398.733 | 821.060 | -
best / joint_best / joint_best | 0/3 | 401.318 | 821.060 | -
best / joint_poor / joint_poor | 0/3 | 407.760 | 800.759 | -
best / off / best | 3/3 | 420.850 | 826.669 | 0.621
best / off / off | 3/3 | 442.990 | 781.770 | 0.591
best / off / poor | 3/3 | 425.595 | 821.810 | 0.611
best / poor / best | 0/3 | 404.061 | 826.669 | -
best / poor / off | 0/3 | 423.382 | 781.770 | -
best / poor / poor | 0/3 | 397.867 | 821.060 | -
off / best / best | 0/3 | 238.615 | 781.884 | -
off / best / off | 0/3 | 250.972 | 762.895 | -
off / best / poor | 0/3 | 230.203 | 802.185 | -
off / off / best | 3/3 | 256.036 | 781.884 | 1.021
off / off / off | 3/3 | 260.800 | 762.895 | 1.000
off / off / poor | 3/3 | 259.848 | 802.935 | 1.004
off / poor / best | 0/3 | 238.910 | 781.884 | -
off / poor / off | 0/3 | 328.778 | 762.895 | -
off / poor / poor | 0/3 | 231.027 | 802.185 | -
poor / best / best | 0/3 | 403.970 | 817.669 | -
poor / best / off | 0/3 | 423.989 | 772.770 | -
poor / best / poor | 0/3 | 400.895 | 812.060 | -
poor / joint_best / joint_best | 0/3 | 398.410 | 812.060 | -
poor / joint_poor / joint_poor | 0/3 | 408.619 | 791.759 | -
poor / off / best | 3/3 | 420.155 | 817.669 | 0.621
poor / off / off | 3/3 | 446.363 | 772.770 | 0.584
poor / off / poor | 3/3 | 426.112 | 812.810 | 0.612
poor / poor / best | 0/3 | 406.164 | 817.669 | -
poor / poor / off | 0/3 | 422.431 | 772.770 | -
poor / poor / poor | 0/3 | 400.820 | 812.060 | -

Batch 1, prompt 512, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 809.148 | 801.884 | -
best / best / off | 0/3 | 838.901 | 781.770 | -
best / best / poor | 0/3 | 786.494 | 821.060 | -
best / joint_best / joint_best | 0/3 | 791.074 | 821.060 | -
best / joint_poor / joint_poor | 0/3 | 809.829 | 801.884 | -
best / off / best | 3/3 | 839.285 | 801.884 | 0.616
best / off / off | 3/3 | 886.610 | 781.770 | 0.583
best / off / poor | 3/3 | 846.726 | 821.810 | 0.609
best / poor / best | 0/3 | 806.116 | 801.884 | -
best / poor / off | 0/3 | 839.336 | 781.770 | -
best / poor / poor | 0/3 | 786.766 | 821.060 | -
off / best / best | 0/3 | 470.430 | 803.544 | -
off / best / off | 0/3 | 510.581 | 762.895 | -
off / best / poor | 0/3 | 453.767 | 802.185 | -
off / off / best | 3/3 | 506.134 | 803.544 | 1.021
off / off / off | 3/3 | 516.960 | 762.895 | 1.000
off / off / poor | 3/3 | 512.042 | 802.935 | 1.010
off / poor / best | 0/3 | 471.242 | 803.544 | -
off / poor / off | 0/3 | 677.221 | 762.895 | -
off / poor / poor | 0/3 | 453.443 | 802.185 | -
poor / best / best | 0/3 | 799.761 | 817.669 | -
poor / best / off | 0/3 | 842.952 | 772.770 | -
poor / best / poor | 0/3 | 792.452 | 812.060 | -
poor / joint_best / joint_best | 0/3 | 790.697 | 812.060 | -
poor / joint_poor / joint_poor | 0/3 | 809.960 | 792.884 | -
poor / off / best | 3/3 | 835.208 | 817.669 | 0.619
poor / off / off | 3/3 | 883.558 | 772.770 | 0.586
poor / off / poor | 3/3 | 838.390 | 812.810 | 0.617
poor / poor / best | 0/3 | 804.924 | 817.669 | -
poor / poor / off | 0/3 | 839.949 | 772.770 | -
poor / poor / poor | 0/3 | 791.208 | 812.060 | -

Batch 1, prompt 896, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 395.414 | 865.727 | -
best / best / off | 0/3 | 528.364 | 784.179 | -
best / best / poor | 0/3 | 405.480 | 827.053 | -
best / joint_best / joint_best | 0/3 | 394.703 | 865.727 | -
best / joint_poor / joint_poor | 0/3 | 406.445 | 825.024 | -
best / off / best | 3/3 | 425.313 | 865.733 | 1.101
best / off / off | 3/3 | 544.806 | 784.179 | 0.860
best / off / poor | 3/3 | 422.286 | 837.143 | 1.104
best / poor / best | 0/3 | 392.490 | 865.727 | -
best / poor / off | 0/3 | 760.401 | 784.179 | -
best / poor / poor | 0/3 | 402.398 | 827.053 | -
off / best / best | 0/3 | 237.047 | 807.547 | -
off / best / off | 0/3 | 448.571 | 765.963 | -
off / best / poor | 0/3 | 226.262 | 847.413 | -
off / off / best | 3/3 | 255.713 | 814.375 | 1.834
off / off / off | 3/3 | 468.630 | 765.963 | 1.000
off / off / poor | 3/3 | 257.739 | 846.537 | 1.820
off / poor / best | 0/3 | 236.431 | 807.547 | -
off / poor / off | 0/3 | 680.830 | 765.963 | -
off / poor / poor | 0/3 | 224.711 | 847.413 | -
poor / best / best | 0/3 | 406.105 | 816.024 | -
poor / best / off | 0/3 | 526.678 | 775.179 | -
poor / best / poor | 0/3 | 405.200 | 818.053 | -
poor / joint_best / joint_best | 0/3 | 395.027 | 856.727 | -
poor / joint_poor / joint_poor | 0/3 | 401.883 | 816.024 | -
poor / off / best | 3/3 | 422.878 | 825.665 | 1.108
poor / off / off | 3/3 | 543.967 | 775.179 | 0.861
poor / off / poor | 3/3 | 422.677 | 828.143 | 1.109
poor / poor / best | 0/3 | 402.685 | 816.024 | -
poor / poor / off | 0/3 | 759.401 | 775.179 | -
poor / poor / poor | 0/3 | 404.397 | 818.053 | -

Batch 1, prompt 896, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 803.477 | 827.053 | -
best / best / off | 0/3 | 1072.397 | 784.460 | -
best / best / poor | 0/3 | 804.417 | 825.446 | -
best / joint_best / joint_best | 0/3 | 779.688 | 865.727 | -
best / joint_poor / joint_poor | 0/3 | 804.186 | 825.446 | -
best / off / best | 3/3 | 829.729 | 837.143 | 1.145
best / off / off | 3/3 | 1107.205 | 784.460 | 0.867
best / off / poor | 3/3 | 837.419 | 835.071 | 1.142
best / poor / best | 0/3 | 804.226 | 827.053 | -
best / poor / off | 0/3 | 1554.119 | 784.460 | -
best / poor / poor | 0/3 | 804.429 | 825.446 | -
off / best / best | 0/3 | 469.550 | 807.734 | -
off / best / off | 0/3 | 912.912 | 766.151 | -
off / best / poor | 0/3 | 446.482 | 847.413 | -
off / off / best | 3/3 | 506.448 | 814.422 | 1.895
off / off / off | 3/3 | 959.654 | 766.151 | 1.000
off / off / poor | 3/3 | 507.134 | 846.537 | 1.892
off / poor / best | 0/3 | 467.764 | 807.734 | -
off / poor / off | 0/3 | 1394.250 | 766.151 | -
off / poor / poor | 0/3 | 446.258 | 847.413 | -
poor / best / best | 0/3 | 803.214 | 818.053 | -
poor / best / off | 0/3 | 1069.588 | 775.460 | -
poor / best / poor | 0/3 | 781.252 | 856.727 | -
poor / joint_best / joint_best | 0/3 | 776.552 | 856.727 | -
poor / joint_poor / joint_poor | 0/3 | 799.044 | 816.446 | -
poor / off / best | 3/3 | 838.779 | 828.143 | 1.144
poor / off / off | 3/3 | 1105.074 | 775.460 | 0.868
poor / off / poor | 3/3 | 832.051 | 856.733 | 1.144
poor / poor / best | 0/3 | 805.173 | 818.053 | -
poor / poor / off | 0/3 | 1551.501 | 775.460 | -
poor / poor / poor | 0/3 | 776.389 | 856.727 | -

Batch 4, prompt 32, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 434.554 | 922.333 | -
best / best / off | 0/3 | 423.282 | 778.537 | -
best / best / poor | 0/3 | 425.378 | 794.480 | -
best / joint_best / joint_best | 0/3 | 422.059 | 794.480 | -
best / joint_poor / joint_poor | 0/3 | 433.676 | 922.333 | -
best / off / best | 3/3 | 448.069 | 922.333 | 0.592
best / off / off | 3/3 | 449.638 | 778.537 | 0.593
best / off / poor | 3/3 | 453.542 | 794.480 | 0.586
best / poor / best | 0/3 | 435.062 | 922.333 | -
best / poor / off | 0/3 | 424.080 | 778.537 | -
best / poor / poor | 0/3 | 425.751 | 794.480 | -
off / best / best | 0/3 | 251.117 | 902.458 | -
off / best / off | 0/3 | 234.242 | 759.662 | -
off / best / poor | 0/3 | 255.619 | 767.318 | -
off / off / best | 3/3 | 271.413 | 902.458 | 0.974
off / off / off | 3/3 | 263.687 | 759.662 | 1.000
off / off / poor | 3/3 | 269.414 | 767.318 | 0.974
off / poor / best | 0/3 | 251.201 | 902.458 | -
off / poor / off | 0/3 | 234.244 | 759.662 | -
off / poor / poor | 0/3 | 255.584 | 767.318 | -
poor / best / best | 0/3 | 427.176 | 785.480 | -
poor / best / off | 0/3 | 423.373 | 769.537 | -
poor / best / poor | 0/3 | 435.810 | 913.333 | -
poor / joint_best / joint_best | 0/3 | 428.436 | 785.480 | -
poor / joint_poor / joint_poor | 0/3 | 436.793 | 913.333 | -
poor / off / best | 3/3 | 454.255 | 785.480 | 0.586
poor / off / off | 3/3 | 450.477 | 769.537 | 0.590
poor / off / poor | 3/3 | 451.516 | 913.333 | 0.588
poor / poor / best | 0/3 | 427.438 | 785.480 | -
poor / poor / off | 0/3 | 422.589 | 769.537 | -
poor / poor / poor | 0/3 | 437.975 | 913.333 | -

Batch 4, prompt 32, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 861.803 | 922.334 | -
best / best / off | 0/3 | 843.980 | 780.906 | -
best / best / poor | 0/3 | 847.860 | 803.482 | -
best / joint_best / joint_best | 0/3 | 848.340 | 803.482 | -
best / joint_poor / joint_poor | 0/3 | 865.012 | 790.694 | -
best / off / best | 3/3 | 894.356 | 922.334 | 0.597
best / off / off | 3/3 | 903.303 | 780.906 | 0.595
best / off / poor | 3/3 | 902.406 | 803.482 | 0.592
best / poor / best | 0/3 | 861.992 | 922.334 | -
best / poor / off | 0/3 | 846.459 | 780.906 | -
best / poor / poor | 0/3 | 847.373 | 803.482 | -
off / best / best | 0/3 | 498.931 | 902.459 | -
off / best / off | 0/3 | 473.748 | 762.031 | -
off / best / poor | 0/3 | 485.427 | 784.607 | -
off / off / best | 3/3 | 539.462 | 902.459 | 0.985
off / off / off | 3/3 | 532.790 | 762.031 | 1.000
off / off / poor | 3/3 | 544.460 | 784.607 | 0.977
off / poor / best | 0/3 | 501.248 | 902.459 | -
off / poor / off | 0/3 | 472.306 | 762.031 | -
off / poor / poor | 0/3 | 484.530 | 784.607 | -
poor / best / best | 0/3 | 866.618 | 781.694 | -
poor / best / off | 0/3 | 844.439 | 771.906 | -
poor / best / poor | 0/3 | 851.661 | 794.482 | -
poor / joint_best / joint_best | 0/3 | 850.219 | 794.482 | -
poor / joint_poor / joint_poor | 0/3 | 869.146 | 781.694 | -
poor / off / best | 3/3 | 897.979 | 781.694 | 0.593
poor / off / off | 3/3 | 900.571 | 771.906 | 0.592
poor / off / poor | 3/3 | 902.181 | 794.482 | 0.593
poor / poor / best | 0/3 | 863.960 | 781.694 | -
poor / poor / off | 0/3 | 844.587 | 771.906 | -
poor / poor / poor | 0/3 | 848.876 | 794.482 | -

Batch 4, prompt 128, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 434.746 | 924.621 | -
best / best / off | 0/3 | 429.565 | 781.351 | -
best / best / poor | 0/3 | 429.942 | 822.051 | -
best / joint_best / joint_best | 0/3 | 426.194 | 822.051 | -
best / joint_poor / joint_poor | 0/3 | 436.118 | 802.980 | -
best / off / best | 3/3 | 450.112 | 924.621 | 0.606
best / off / off | 3/3 | 460.581 | 781.351 | 0.593
best / off / poor | 3/3 | 455.840 | 822.051 | 0.600
best / poor / best | 0/3 | 433.996 | 924.621 | -
best / poor / off | 0/3 | 433.706 | 781.351 | -
best / poor / poor | 0/3 | 429.846 | 822.051 | -
off / best / best | 0/3 | 253.157 | 905.996 | -
off / best / off | 0/3 | 242.871 | 762.476 | -
off / best / poor | 0/3 | 249.025 | 803.176 | -
off / off / best | 3/3 | 268.699 | 905.996 | 1.008
off / off / off | 3/3 | 270.965 | 762.476 | 1.000
off / off / poor | 3/3 | 276.854 | 803.176 | 0.979
off / poor / best | 0/3 | 252.845 | 905.996 | -
off / poor / off | 0/3 | 242.203 | 762.476 | -
off / poor / poor | 0/3 | 247.527 | 803.176 | -
poor / best / best | 0/3 | 436.230 | 793.980 | -
poor / best / off | 0/3 | 433.013 | 772.351 | -
poor / best / poor | 0/3 | 428.535 | 813.051 | -
poor / joint_best / joint_best | 0/3 | 429.296 | 813.051 | -
poor / joint_poor / joint_poor | 0/3 | 435.481 | 915.621 | -
poor / off / best | 3/3 | 452.335 | 793.980 | 0.603
poor / off / off | 3/3 | 462.622 | 772.351 | 0.587
poor / off / poor | 3/3 | 457.118 | 813.051 | 0.597
poor / poor / best | 0/3 | 435.239 | 793.980 | -
poor / poor / off | 0/3 | 433.257 | 772.351 | -
poor / poor / poor | 0/3 | 429.830 | 813.051 | -

Batch 4, prompt 128, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 862.794 | 818.337 | -
best / best / off | 0/3 | 848.326 | 781.923 | -
best / best / poor | 0/3 | 843.704 | 845.055 | -
best / joint_best / joint_best | 0/3 | 847.934 | 845.055 | -
best / joint_poor / joint_poor | 0/3 | 864.513 | 924.622 | -
best / off / best | 3/3 | 888.759 | 818.337 | 0.614
best / off / off | 3/3 | 905.883 | 781.923 | 0.603
best / off / poor | 3/3 | 901.887 | 845.055 | 0.605
best / poor / best | 0/3 | 855.367 | 818.337 | -
best / poor / off | 0/3 | 858.754 | 781.923 | -
best / poor / poor | 0/3 | 850.416 | 845.055 | -
off / best / best | 0/3 | 504.760 | 800.349 | -
off / best / off | 0/3 | 482.863 | 763.048 | -
off / best / poor | 0/3 | 487.169 | 824.163 | -
off / off / best | 3/3 | 535.751 | 800.349 | 1.014
off / off / off | 3/3 | 546.002 | 763.048 | 1.000
off / off / poor | 3/3 | 543.276 | 824.163 | 1.005
off / poor / best | 0/3 | 504.793 | 800.349 | -
off / poor / off | 0/3 | 486.186 | 763.048 | -
off / poor / poor | 0/3 | 487.592 | 824.163 | -
poor / best / best | 0/3 | 866.085 | 809.337 | -
poor / best / off | 0/3 | 864.043 | 772.923 | -
poor / best / poor | 0/3 | 851.105 | 836.055 | -
poor / joint_best / joint_best | 0/3 | 850.394 | 836.055 | -
poor / joint_poor / joint_poor | 0/3 | 865.313 | 915.622 | -
poor / off / best | 3/3 | 896.937 | 809.337 | 0.613
poor / off / off | 3/3 | 924.429 | 772.923 | 0.595
poor / off / poor | 3/3 | 904.001 | 836.055 | 0.610
poor / poor / best | 0/3 | 865.299 | 809.337 | -
poor / poor / off | 0/3 | 864.544 | 772.923 | -
poor / poor / poor | 0/3 | 849.897 | 836.055 | -

Batch 4, prompt 512, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 435.464 | 947.883 | -
best / best / off | 0/3 | 743.282 | 810.825 | -
best / best / poor | 0/3 | 427.992 | 958.074 | -
best / joint_best / joint_best | 0/3 | 424.250 | 958.074 | -
best / joint_poor / joint_poor | 0/3 | 435.121 | 881.039 | -
best / off / best | 3/3 | 450.145 | 974.883 | 1.815
best / off / off | 3/3 | 962.627 | 822.722 | 0.849
best / off / poor | 3/3 | 453.409 | 958.074 | 1.802
best / poor / best | 0/3 | 435.640 | 947.883 | -
best / poor / off | 0/3 | 820.531 | 810.825 | -
best / poor / poor | 0/3 | 429.418 | 958.074 | -
off / best / best | 0/3 | 256.092 | 928.383 | -
off / best / off | 0/3 | 598.291 | 791.313 | -
off / best / poor | 0/3 | 245.832 | 938.580 | -
off / off / best | 3/3 | 272.678 | 955.008 | 2.996
off / off / off | 3/3 | 816.397 | 801.583 | 1.000
off / off / poor | 3/3 | 273.763 | 938.580 | 2.985
off / poor / best | 0/3 | 254.752 | 928.383 | -
off / poor / off | 0/3 | 675.464 | 791.313 | -
off / poor / poor | 0/3 | 246.103 | 938.580 | -
poor / best / best | 0/3 | 434.112 | 938.883 | -
poor / best / off | 0/3 | 741.161 | 801.825 | -
poor / best / poor | 0/3 | 427.735 | 949.074 | -
poor / joint_best / joint_best | 0/3 | 427.690 | 949.074 | -
poor / joint_poor / joint_poor | 0/3 | 435.469 | 872.039 | -
poor / off / best | 3/3 | 450.658 | 965.883 | 1.813
poor / off / off | 3/3 | 960.804 | 813.722 | 0.850
poor / off / poor | 3/3 | 453.829 | 949.074 | 1.800
poor / poor / best | 0/3 | 435.540 | 938.883 | -
poor / poor / off | 0/3 | 818.799 | 801.825 | -
poor / poor / poor | 0/3 | 427.631 | 949.074 | -

Batch 4, prompt 512, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 866.903 | 884.743 | -
best / best / off | 0/3 | 1524.732 | 815.729 | -
best / best / poor | 0/3 | 842.953 | 958.075 | -
best / joint_best / joint_best | 0/3 | 848.000 | 958.075 | -
best / joint_poor / joint_poor | 0/3 | 867.696 | 947.884 | -
best / off / best | 3/3 | 887.313 | 912.636 | 1.918
best / off / off | 3/3 | 1994.245 | 832.833 | 0.852
best / off / poor | 3/3 | 898.524 | 958.075 | 1.894
best / poor / best | 0/3 | 867.576 | 884.743 | -
best / poor / off | 0/3 | 1701.167 | 815.729 | -
best / poor / poor | 0/3 | 848.638 | 958.075 | -
off / best / best | 0/3 | 503.117 | 928.384 | -
off / best / off | 0/3 | 1231.182 | 796.592 | -
off / best / poor | 0/3 | 486.641 | 938.581 | -
off / off / best | 3/3 | 540.048 | 955.009 | 3.152
off / off / off | 3/3 | 1701.006 | 813.287 | 1.000
off / off / poor | 3/3 | 541.048 | 938.581 | 3.146
off / poor / best | 0/3 | 507.434 | 928.384 | -
off / poor / off | 0/3 | 1398.518 | 796.592 | -
off / poor / poor | 0/3 | 487.680 | 938.581 | -
poor / best / best | 0/3 | 867.397 | 938.884 | -
poor / best / off | 0/3 | 1521.039 | 806.729 | -
poor / best / poor | 0/3 | 846.374 | 949.075 | -
poor / joint_best / joint_best | 0/3 | 848.949 | 949.075 | -
poor / joint_poor / joint_poor | 0/3 | 865.868 | 875.743 | -
poor / off / best | 3/3 | 894.470 | 965.884 | 1.903
poor / off / off | 3/3 | 1989.653 | 823.833 | 0.853
poor / off / poor | 3/3 | 898.276 | 949.075 | 1.895
poor / poor / best | 0/3 | 867.193 | 938.884 | -
poor / poor / off | 0/3 | 1691.721 | 806.729 | -
poor / poor / poor | 0/3 | 847.244 | 949.075 | -

Batch 4, prompt 896, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 444.470 | 1026.659 | -
best / best / off | 0/3 | 1320.309 | 885.899 | -
best / best / poor | 0/3 | 438.931 | 1095.604 | -
best / joint_best / joint_best | 0/3 | 440.363 | 1095.604 | -
best / joint_poor / joint_poor | 0/3 | 453.068 | 1026.659 | -
best / off / best | 3/3 | 471.863 | 1197.659 | 3.576
best / off / off | 3/3 | 1939.311 | 1001.710 | 0.873
best / off / poor | 3/3 | 479.265 | 1158.784 | 3.537
best / poor / best | 0/3 | 453.371 | 1026.659 | -
best / poor / off | 0/3 | 1585.199 | 885.899 | -
best / poor / poor | 0/3 | 446.624 | 1095.604 | -
off / best / best | 0/3 | 263.394 | 992.667 | -
off / best / off | 0/3 | 1074.129 | 867.410 | -
off / best / poor | 0/3 | 257.135 | 1075.484 | -
off / off / best | 3/3 | 294.694 | 1165.280 | 5.728
off / off / off | 3/3 | 1692.969 | 983.494 | 1.000
off / off / poor | 3/3 | 298.381 | 1139.909 | 5.674
off / poor / best | 0/3 | 268.926 | 992.667 | -
off / poor / off | 0/3 | 1336.026 | 867.410 | -
off / poor / poor | 0/3 | 264.693 | 1075.484 | -
poor / best / best | 0/3 | 443.892 | 1004.917 | -
poor / best / off | 0/3 | 1315.895 | 876.899 | -
poor / best / poor | 0/3 | 437.576 | 1086.604 | -
poor / joint_best / joint_best | 0/3 | 436.977 | 1086.604 | -
poor / joint_poor / joint_poor | 0/3 | 451.895 | 1004.917 | -
poor / off / best | 3/3 | 471.285 | 1175.280 | 3.592
poor / off / off | 3/3 | 1934.773 | 992.710 | 0.875
poor / off / poor | 3/3 | 475.221 | 1149.784 | 3.562
poor / poor / best | 0/3 | 451.963 | 1004.917 | -
poor / poor / off | 0/3 | 1576.973 | 876.899 | -
poor / poor / poor | 0/3 | 445.284 | 1086.604 | -

Batch 4, prompt 896, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 875.811 | 1020.590 | -
best / best / off | 0/3 | 2698.085 | 893.100 | -
best / best / poor | 0/3 | 858.351 | 1095.604 | -
best / joint_best / joint_best | 0/3 | 860.793 | 1095.604 | -
best / joint_poor / joint_poor | 0/3 | 881.935 | 1026.660 | -
best / off / best | 3/3 | 917.683 | 1190.436 | 3.786
best / off / off | 3/3 | 3964.858 | 1022.078 | 0.876
best / off / poor | 3/3 | 921.758 | 1158.785 | 3.769
best / poor / best | 0/3 | 883.116 | 1020.590 | -
best / poor / off | 0/3 | 3253.488 | 893.100 | -
best / poor / poor | 0/3 | 866.602 | 1095.604 | -
off / best / best | 0/3 | 515.722 | 1006.160 | -
off / best / off | 0/3 | 2203.081 | 874.791 | -
off / best / poor | 0/3 | 498.491 | 1075.485 | -
off / off / best | 3/3 | 564.512 | 1179.410 | 6.154
off / off / off | 3/3 | 3474.005 | 1003.768 | 1.000
off / off / poor | 3/3 | 567.858 | 1139.910 | 6.118
off / poor / best | 0/3 | 524.135 | 1006.160 | -
off / poor / off | 0/3 | 2750.832 | 874.791 | -
off / poor / poor | 0/3 | 504.581 | 1075.485 | -
poor / best / best | 0/3 | 874.417 | 1017.660 | -
poor / best / off | 0/3 | 2686.179 | 884.100 | -
poor / best / poor | 0/3 | 856.610 | 1086.604 | -
poor / joint_best / joint_best | 0/3 | 850.024 | 1086.604 | -
poor / joint_poor / joint_poor | 0/3 | 872.628 | 1017.660 | -
poor / off / best | 3/3 | 908.173 | 1188.660 | 3.825
poor / off / off | 3/3 | 3959.903 | 1013.078 | 0.877
poor / off / poor | 3/3 | 911.863 | 1149.785 | 3.775
poor / poor / best | 0/3 | 882.263 | 1017.660 | -
poor / poor / off | 0/3 | 3242.778 | 884.100 | -
poor / poor / poor | 0/3 | 863.256 | 1086.604 | -

Batch 8, prompt 32, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 430.527 | 796.578 | -
best / best / off | 0/3 | 422.124 | 781.971 | -
best / best / poor | 0/3 | 426.863 | 813.242 | -
best / joint_best / joint_best | 0/3 | 426.319 | 813.242 | -
best / joint_poor / joint_poor | 0/3 | 430.536 | 1066.859 | -
best / off / best | 3/3 | 445.351 | 796.578 | 0.602
best / off / off | 3/3 | 451.189 | 781.971 | 0.597
best / off / poor | 3/3 | 453.710 | 813.242 | 0.591
best / poor / best | 0/3 | 430.892 | 796.578 | -
best / poor / off | 0/3 | 421.812 | 781.971 | -
best / poor / poor | 0/3 | 426.042 | 813.242 | -
off / best / best | 0/3 | 254.627 | 1047.984 | -
off / best / off | 0/3 | 242.663 | 763.096 | -
off / best / poor | 0/3 | 248.878 | 794.367 | -
off / off / best | 3/3 | 273.564 | 1047.984 | 0.997
off / off / off | 3/3 | 272.052 | 763.096 | 1.000
off / off / poor | 3/3 | 277.431 | 794.367 | 0.977
off / poor / best | 0/3 | 255.026 | 1047.984 | -
off / poor / off | 0/3 | 242.216 | 763.096 | -
off / poor / poor | 0/3 | 249.115 | 794.367 | -
poor / best / best | 0/3 | 434.718 | 1057.859 | -
poor / best / off | 0/3 | 424.613 | 772.971 | -
poor / best / poor | 0/3 | 431.733 | 804.242 | -
poor / joint_best / joint_best | 0/3 | 430.664 | 804.242 | -
poor / joint_poor / joint_poor | 0/3 | 434.691 | 1057.859 | -
poor / off / best | 3/3 | 450.189 | 1057.859 | 0.605
poor / off / off | 3/3 | 454.148 | 772.971 | 0.599
poor / off / poor | 3/3 | 459.641 | 804.242 | 0.592
poor / poor / best | 0/3 | 433.483 | 1057.859 | -
poor / poor / off | 0/3 | 426.769 | 772.971 | -
poor / poor / poor | 0/3 | 431.738 | 804.242 | -

Batch 8, prompt 32, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 856.419 | 1066.861 | -
best / best / off | 0/3 | 845.215 | 782.772 | -
best / best / poor | 0/3 | 844.169 | 845.020 | -
best / joint_best / joint_best | 0/3 | 844.903 | 845.020 | -
best / joint_poor / joint_poor | 0/3 | 856.292 | 1066.861 | -
best / off / best | 3/3 | 885.358 | 1066.861 | 0.613
best / off / off | 3/3 | 899.988 | 782.772 | 0.602
best / off / poor | 3/3 | 898.848 | 845.020 | 0.602
best / poor / best | 0/3 | 858.456 | 1066.861 | -
best / poor / off | 0/3 | 843.761 | 782.772 | -
best / poor / poor | 0/3 | 844.579 | 845.020 | -
off / best / best | 0/3 | 508.283 | 1047.986 | -
off / best / off | 0/3 | 485.357 | 763.897 | -
off / best / poor | 0/3 | 493.888 | 824.254 | -
off / off / best | 3/3 | 542.479 | 1047.986 | 1.008
off / off / off | 3/3 | 547.039 | 763.897 | 1.000
off / off / poor | 3/3 | 552.362 | 824.254 | 0.990
off / poor / best | 0/3 | 508.852 | 1047.986 | -
off / poor / off | 0/3 | 485.083 | 763.897 | -
off / poor / poor | 0/3 | 493.922 | 824.254 | -
poor / best / best | 0/3 | 863.670 | 808.201 | -
poor / best / off | 0/3 | 856.975 | 773.772 | -
poor / best / poor | 0/3 | 848.315 | 836.020 | -
poor / joint_best / joint_best | 0/3 | 847.960 | 836.020 | -
poor / joint_poor / joint_poor | 0/3 | 866.439 | 1057.861 | -
poor / off / best | 3/3 | 883.961 | 808.201 | 0.616
poor / off / off | 3/3 | 909.606 | 773.772 | 0.601
poor / off / poor | 3/3 | 905.002 | 836.020 | 0.604
poor / poor / best | 0/3 | 860.548 | 808.201 | -
poor / poor / off | 0/3 | 850.388 | 773.772 | -
poor / poor / poor | 0/3 | 850.890 | 836.020 | -

Batch 8, prompt 128, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 433.348 | 1072.152 | -
best / best / off | 0/3 | 430.801 | 785.974 | -
best / best / poor | 0/3 | 421.817 | 875.922 | -
best / joint_best / joint_best | 0/3 | 421.850 | 875.922 | -
best / joint_poor / joint_poor | 0/3 | 432.355 | 831.867 | -
best / off / best | 3/3 | 447.434 | 1072.152 | 0.797
best / off / off | 3/3 | 461.943 | 785.974 | 0.774
best / off / poor | 3/3 | 448.839 | 875.922 | 0.795
best / poor / best | 0/3 | 430.706 | 1072.152 | -
best / poor / off | 0/3 | 434.840 | 785.974 | -
best / poor / poor | 0/3 | 421.993 | 875.922 | -
off / best / best | 0/3 | 256.003 | 1053.277 | -
off / best / off | 0/3 | 319.621 | 767.099 | -
off / best / poor | 0/3 | 246.284 | 856.309 | -
off / off / best | 3/3 | 276.160 | 1053.277 | 1.283
off / off / off | 3/3 | 355.679 | 767.099 | 1.000
off / off / poor | 3/3 | 276.121 | 856.309 | 1.288
off / poor / best | 0/3 | 256.042 | 1053.277 | -
off / poor / off | 0/3 | 329.926 | 767.099 | -
off / poor / poor | 0/3 | 245.895 | 856.309 | -
poor / best / best | 0/3 | 426.901 | 866.922 | -
poor / best / off | 0/3 | 429.001 | 776.974 | -
poor / best / poor | 0/3 | 437.477 | 1063.152 | -
poor / joint_best / joint_best | 0/3 | 426.692 | 866.922 | -
poor / joint_poor / joint_poor | 0/3 | 431.488 | 1063.152 | -
poor / off / best | 3/3 | 449.486 | 866.922 | 0.791
poor / off / off | 3/3 | 459.575 | 776.974 | 0.774
poor / off / poor | 3/3 | 451.551 | 1063.152 | 0.788
poor / poor / best | 0/3 | 424.999 | 866.922 | -
poor / poor / off | 0/3 | 432.085 | 776.974 | -
poor / poor / poor | 0/3 | 435.861 | 1063.152 | -

Batch 8, prompt 128, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 835.402 | 891.451 | -
best / best / off | 0/3 | 914.358 | 787.571 | -
best / best / poor | 0/3 | 858.906 | 1072.154 | -
best / joint_best / joint_best | 0/3 | 835.514 | 891.451 | -
best / joint_poor / joint_poor | 0/3 | 857.670 | 1072.154 | -
best / off / best | 3/3 | 889.715 | 891.451 | 0.885
best / off / off | 3/3 | 994.048 | 787.571 | 0.792
best / off / poor | 3/3 | 888.878 | 1072.154 | 0.886
best / poor / best | 0/3 | 836.920 | 891.451 | -
best / poor / off | 0/3 | 933.259 | 787.571 | -
best / poor / poor | 0/3 | 857.742 | 1072.154 | -
off / best / best | 0/3 | 508.805 | 1053.279 | -
off / best / off | 0/3 | 702.078 | 768.696 | -
off / best / poor | 0/3 | 508.653 | 827.139 | -
off / off / best | 3/3 | 540.813 | 1053.279 | 1.458
off / off / off | 3/3 | 784.330 | 768.696 | 1.000
off / off / poor | 3/3 | 534.631 | 827.139 | 1.468
off / poor / best | 0/3 | 502.632 | 1053.279 | -
off / poor / off | 0/3 | 723.687 | 768.696 | -
off / poor / poor | 0/3 | 505.934 | 827.139 | -
poor / best / best | 0/3 | 861.667 | 1063.154 | -
poor / best / off | 0/3 | 912.150 | 778.571 | -
poor / best / poor | 0/3 | 838.495 | 882.451 | -
poor / joint_best / joint_best | 0/3 | 842.896 | 882.451 | -
poor / joint_poor / joint_poor | 0/3 | 861.993 | 1063.154 | -
poor / off / best | 3/3 | 885.551 | 1063.154 | 0.886
poor / off / off | 3/3 | 990.032 | 778.571 | 0.793
poor / off / poor | 3/3 | 894.559 | 882.451 | 0.877
poor / poor / best | 0/3 | 866.626 | 1063.154 | -
poor / poor / off | 0/3 | 929.129 | 778.571 | -
poor / poor / poor | 0/3 | 841.289 | 882.451 | -

Batch 8, prompt 512, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 444.047 | 1191.298 | -
best / best / off | 0/3 | 1353.932 | 915.087 | -
best / best / poor | 0/3 | 437.076 | 1139.442 | -
best / joint_best / joint_best | 0/3 | 437.383 | 1139.442 | -
best / joint_poor / joint_poor | 0/3 | 447.401 | 1055.541 | -
best / off / best | 3/3 | 463.920 | 1244.298 | 3.321
best / off / off | 3/3 | 1817.888 | 938.631 | 0.848
best / off / poor | 3/3 | 471.589 | 1139.061 | 3.278
best / poor / best | 0/3 | 445.658 | 1191.298 | -
best / poor / off | 0/3 | 1476.205 | 915.087 | -
best / poor / poor | 0/3 | 441.013 | 1139.442 | -
off / best / best | 0/3 | 262.669 | 1036.939 | -
off / best / off | 0/3 | 1076.705 | 896.212 | -
off / best / poor | 0/3 | 254.137 | 1119.579 | -
off / off / best | 3/3 | 285.831 | 1091.052 | 5.391
off / off / off | 3/3 | 1540.647 | 919.119 | 1.000
off / off / poor | 3/3 | 291.645 | 1119.197 | 5.292
off / poor / best | 0/3 | 267.611 | 1036.939 | -
off / poor / off | 0/3 | 1207.709 | 896.212 | -
off / poor / poor | 0/3 | 258.494 | 1119.579 | -
poor / best / best | 0/3 | 448.227 | 1182.298 | -
poor / best / off | 0/3 | 1349.727 | 906.087 | -
poor / best / poor | 0/3 | 442.417 | 1130.442 | -
poor / joint_best / joint_best | 0/3 | 441.260 | 1130.442 | -
poor / joint_poor / joint_poor | 0/3 | 451.072 | 1046.541 | -
poor / off / best | 3/3 | 468.478 | 1235.298 | 3.286
poor / off / off | 3/3 | 1814.140 | 929.631 | 0.850
poor / off / poor | 3/3 | 474.741 | 1130.061 | 3.245
poor / poor / best | 0/3 | 446.968 | 1182.298 | -
poor / poor / off | 0/3 | 1472.927 | 906.087 | -
poor / poor / poor | 0/3 | 445.348 | 1130.442 | -

Batch 8, prompt 512, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 870.857 | 1064.281 | -
best / best / off | 0/3 | 2786.772 | 927.560 | -
best / best / poor | 0/3 | 869.507 | 1191.300 | -
best / joint_best / joint_best | 0/3 | 853.213 | 1139.444 | -
best / joint_poor / joint_poor | 0/3 | 873.851 | 1064.281 | -
best / off / best | 3/3 | 906.390 | 1118.281 | 3.533
best / off / off | 3/3 | 3773.655 | 962.857 | 0.850
best / off / poor | 3/3 | 905.615 | 1244.300 | 3.539
best / poor / best | 0/3 | 874.464 | 1064.281 | -
best / poor / off | 0/3 | 3063.530 | 927.560 | -
best / poor / poor | 0/3 | 872.275 | 1191.300 | -
off / best / best | 0/3 | 509.620 | 1045.667 | -
off / best / off | 0/3 | 2217.297 | 908.685 | -
off / best / poor | 0/3 | 491.547 | 1119.581 | -
off / off / best | 3/3 | 554.868 | 1100.500 | 5.782
off / off / off | 3/3 | 3201.855 | 943.721 | 1.000
off / off / poor | 3/3 | 560.182 | 1119.199 | 5.765
off / poor / best | 0/3 | 513.763 | 1045.667 | -
off / poor / off | 0/3 | 2495.294 | 908.685 | -
off / poor / poor | 0/3 | 494.044 | 1119.581 | -
poor / best / best | 0/3 | 872.588 | 1055.281 | -
poor / best / off | 0/3 | 2780.025 | 918.560 | -
poor / best / poor | 0/3 | 859.963 | 1130.444 | -
poor / joint_best / joint_best | 0/3 | 861.194 | 1130.444 | -
poor / joint_poor / joint_poor | 0/3 | 879.447 | 1055.281 | -
poor / off / best | 3/3 | 913.907 | 1109.281 | 3.496
poor / off / off | 3/3 | 3766.954 | 953.857 | 0.852
poor / off / poor | 3/3 | 918.157 | 1130.062 | 3.487
poor / poor / best | 0/3 | 884.649 | 1055.281 | -
poor / poor / off | 0/3 | 3057.373 | 918.560 | -
poor / poor / poor | 0/3 | 861.019 | 1130.444 | -

Batch 8, prompt 896, output 32
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 471.219 | 1313.262 | -
best / best / off | 0/3 | 2373.495 | 1065.994 | -
best / best / poor | 0/3 | 467.083 | 1431.723 | -
best / joint_best / joint_best | 0/3 | 468.499 | 1431.723 | -
best / joint_poor / joint_poor | 0/3 | 485.152 | 1340.223 | -
best / off / best | 3/3 | 514.772 | 1662.262 | 6.260
best / off / off | 3/3 | 3681.282 | 1298.499 | 0.874
best / off / poor | 3/3 | 521.526 | 1612.223 | 6.165
best / poor / best | 0/3 | 484.318 | 1313.262 | -
best / poor / off | 0/3 | 2789.861 | 1065.994 | -
best / poor / poor | 0/3 | 478.152 | 1431.723 | -
off / best / best | 0/3 | 284.676 | 1322.598 | -
off / best / off | 0/3 | 1917.985 | 1047.504 | -
off / best / poor | 0/3 | 281.975 | 1413.098 | -
off / off / best | 3/3 | 329.865 | 1669.098 | 9.748
off / off / off | 3/3 | 3217.574 | 1280.283 | 1.000
off / off / poor | 3/3 | 337.013 | 1593.598 | 9.555
off / poor / best | 0/3 | 296.962 | 1322.598 | -
off / poor / off | 0/3 | 2323.880 | 1047.504 | -
off / poor / poor | 0/3 | 292.787 | 1413.098 | -
poor / best / best | 0/3 | 473.977 | 1304.262 | -
poor / best / off | 0/3 | 2368.110 | 1056.994 | -
poor / best / poor | 0/3 | 472.509 | 1422.723 | -
poor / joint_best / joint_best | 0/3 | 471.963 | 1422.723 | -
poor / joint_poor / joint_poor | 0/3 | 491.331 | 1331.223 | -
poor / off / best | 3/3 | 520.945 | 1653.262 | 6.172
poor / off / off | 3/3 | 3675.793 | 1289.499 | 0.876
poor / off / poor | 3/3 | 525.941 | 1603.223 | 6.118
poor / poor / best | 0/3 | 489.255 | 1304.262 | -
poor / poor / off | 0/3 | 2782.954 | 1056.994 | -
poor / poor / poor | 0/3 | 484.696 | 1422.723 | -

Batch 8, prompt 896, output 64
LoRA / Attention / Cache | pass/seeds | ms | peak MiB | paired speed ratio
best / best / best | 0/3 | 902.237 | 1322.467 | -
best / best / off | 0/3 | 4835.909 | 1077.287 | -
best / best / poor | 0/3 | 878.503 | 1431.725 | -
best / joint_best / joint_best | 0/3 | 879.280 | 1431.725 | -
best / joint_poor / joint_poor | 0/3 | 911.568 | 1340.225 | -
best / off / best | 3/3 | 968.938 | 1671.467 | 6.825
best / off / off | 3/3 | 7561.937 | 1335.815 | 0.875
best / off / poor | 3/3 | 961.519 | 1612.225 | 6.869
best / poor / best | 0/3 | 910.605 | 1322.467 | -
best / poor / off | 0/3 | 5704.523 | 1077.287 | -
best / poor / poor | 0/3 | 891.196 | 1431.725 | -
off / best / best | 0/3 | 534.769 | 1322.600 | -
off / best / off | 0/3 | 3915.073 | 1059.739 | -
off / best / poor | 0/3 | 520.560 | 1413.100 | -
off / off / best | 3/3 | 598.962 | 1669.100 | 11.068
off / off / off | 3/3 | 6612.577 | 1318.267 | 1.000
off / off / poor | 3/3 | 605.808 | 1593.600 | 10.942
off / poor / best | 0/3 | 546.795 | 1322.600 | -
off / poor / off | 0/3 | 4766.519 | 1059.739 | -
off / poor / poor | 0/3 | 532.304 | 1413.100 | -
poor / best / best | 0/3 | 903.176 | 1313.467 | -
poor / best / off | 0/3 | 4825.908 | 1068.287 | -
poor / best / poor | 0/3 | 886.522 | 1422.725 | -
poor / joint_best / joint_best | 0/3 | 891.309 | 1422.725 | -
poor / joint_poor / joint_poor | 0/3 | 917.526 | 1313.467 | -
poor / off / best | 3/3 | 961.532 | 1662.467 | 6.877
poor / off / off | 3/3 | 7541.528 | 1326.815 | 0.876
poor / off / poor | 3/3 | 972.130 | 1603.225 | 6.794
poor / poor / best | 0/3 | 923.530 | 1313.467 | -
poor / poor / off | 0/3 | 5692.049 | 1068.287 | -
poor / poor / poor | 0/3 | 902.429 | 1422.725 | -

## Source and provenance
Study: 20261112_full_matrix
Source and dataset hashes: ../SNAPSHOT_SHA256.json; frozen plans include prior selection/trial hashes.
Raw timings, numerical errors and dtype audits: calibration-*.json and confirmation-*.json.
Natural-task outputs: quality-*.json. Training and checkpoint hashes: training.json.
No test score is used to select configurations. Diagnostic results are retained.
