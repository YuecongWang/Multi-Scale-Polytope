# Multi-Scale-Polytope
Inspired by convex  poisoning, Multi-Scale-Polytope is a novel concept that integrates the geometric properties of convex  with multi-scale decomposition and fusion mechanisms to achieve both global stability and local adaptability, breaking through the limitations of traditional single-scale strategies.


How to use BP VS MSP?
----
python visualize_msp_features_bp_fixed_v3.py ^
--gpu 0 ^
--analysis-net ResNet50 ^
--model-resume-path checkpoint-ln ^
--analysis-chk-name "cifar10-ckpt-%s-4800to0-dp0.250-droplayer0.000-seed1226.t7" ^
--analysis-dp 0.25 ^
--target-label 6 ^
--target-index 0 ^
--train-data-path "datasets/CIFAR10_TRAIN_Split.pth" ^
--bp-poison-path "attack-results/bp/poisons/poison.pth" ^
--msp-poison-path "attack-results/msp/poisons/poison.pth" ^
--multi-scale-layers 3 ^
--out-dir "attack-results/feature-analysis"
"""
