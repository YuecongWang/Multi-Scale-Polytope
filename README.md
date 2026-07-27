# Multi-Scale-Polytope
Inspired by convex  poisoning, Multi-Scale-Polytope is a novel concept that integrates the geometric properties of convex  with multi-scale decomposition and fusion mechanisms to achieve both global stability and local adaptability, breaking through the limitations of traditional single-scale strategies.


#How to use BP-MSP Test?
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
The Result is :


1.One Target:
<img width="2970" height="2210" alt="figure_g_pca_distance_layer4_pooled" src="https://github.com/user-attachments/assets/7ba3d816-e822-4042-b8d4-c60a25e2e360" />
<img width="2662" height="2066" alt="figure_g_pca_layer4_pooled" src="https://github.com/user-attachments/assets/19a690da-20d3-4951-8845-32a1db9129bc" />
<img width="2662" height="2066" alt="figure_h_pca_concat" src="https://github.com/user-attachments/assets/afc474fb-a8aa-4f6d-ba6f-bfc7c63b0b88" />
<img width="2970" height="2209" alt="figure_h_pca_distance_concat" src="https://github.com/user-attachments/assets/2ba990bf-c976-4964-a845-851f67a1514e" />



2.All Targets:
<img width="2670" height="2060" alt="global_pca_target_centroids_layer4_pooled" src="https://github.com/user-attachments/assets/1acc3d29-4ebe-4b83-9dc5-82132f2bc1a0" />
<img width="2670" height="2059" alt="global_pca_target_centroids_concat" src="https://github.com/user-attachments/assets/19dc2ea7-e5ca-4846-ae12-b1f5cb2f8bf1" />
