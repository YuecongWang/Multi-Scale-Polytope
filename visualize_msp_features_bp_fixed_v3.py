#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_msp_features_bp.py

本脚本完全沿用 craft_poisons_transfer.py 的数据和模型读取方式：
1. 使用 utils.load_pretrained_net() 加载项目中的训练模型；
2. 使用 utils.fetch_target() 从 CIFAR-10_TRAIN_Split.pth 取得同一个目标；
3. 使用与攻击代码一致的 CIFAR-10 mean/std；
4. 直接读取 poison.pth / poison_XXXXX.pth 中的 state_dict['poison']；
5. 使用项目模型原生接口：
       net(x=..., penu=True)
       net(x=..., multi_layer=True)
       net(x=..., block=True)
   不再使用 torchvision 模型和手工 forward hook；
6. 直接计算 BP 论文公式 (2)：
       0.5 * ||phi(x_t)-mean_j phi(x_p^j)||_2^2 / ||phi(x_t)||_2^2

适用于 Python 3.7 和 Windows CMD。

示例
----
python visualize_msp_features_bp.py ^
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

from __future__ import print_function

from multiprocessing import freeze_support
import argparse
import csv
import math
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from utils import load_pretrained_net, fetch_target
from trainer import least_squares_simplex


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BP/MSP features using the original BP project pipeline."
    )

    parser.add_argument("--gpu", default="0", type=str)

    # 与 craft_poisons_transfer.py 相同的模型加载参数
    parser.add_argument("--analysis-net", default="ResNet50", type=str)
    parser.add_argument("--model-resume-path", default="model-chks", type=str)
    parser.add_argument(
        "--analysis-chk-name",
        default="ckpt-%s-4800.t7",
        type=str,
        help="与 load_pretrained_net 使用的 checkpoint 名称模板一致。",
    )
    parser.add_argument(
        "--analysis-dp",
        default=0.0,
        type=float,
        help="传给 load_pretrained_net(..., test_dp=...) 的 dropout。",
    )

    # 与攻击代码相同的目标读取参数
    parser.add_argument("--target-label", default=6, type=int)
    parser.add_argument("--target-index", default=0, type=int)
    parser.add_argument("--num-per-class", default=50, type=int)
    parser.add_argument("--subset", default="others", type=str)
    parser.add_argument(
        "--train-data-path",
        default="datasets/CIFAR10_TRAIN_Split.pth",
        type=str,
    )

    # 直接读取攻击脚本保存的 poison checkpoint
    parser.add_argument("--bp-poison-path", default="", type=str)
    parser.add_argument("--msp-poison-path", default="", type=str)

    parser.add_argument(
        "--multi-scale-layers",
        default=3,
        type=int,
        help="仅控制 block=True 返回多少个 layer4 内部特征；不影响 multi_layer=True。",
    )
    parser.add_argument(
        "--num-feature-maps",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--num-filter-kernels",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--out-dir",
        default="feature_visualization",
        type=str,
    )
    parser.add_argument(
        "--seed",
        default=1226,
        type=int,
        help="固定诊断前向传播的随机种子，减小 dropout 随机性。",
    )
    parser.add_argument(
        "--tol",
        default=1e-6,
        type=float,
        help="CP 单纯形系数优化的停止阈值，与 trainer.py 保持一致。",
    )
    return parser.parse_args()


def unwrap_model(net):
    return net.module if hasattr(net, "module") else net


def set_multiscale_layers(net, multi_scale_layers):
    module = unwrap_model(net)
    if hasattr(module, "middle_feat_num"):
        module.middle_feat_num = multi_scale_layers
        print("Set middle_feat_num =", multi_scale_layers)
    else:
        print(
            "[Warning] {} has no middle_feat_num attribute.".format(
                module.__class__.__name__
            )
        )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_poison_batch(poison_path, device):
    """
    读取 craft_poisons_transfer.py 保存的：
        {'poison': poison_tuple_list, 'idx': ..., ...}

    poison_tuple_list 中每项通常为 (poison_tensor, poison_label)。
    """
    if not poison_path:
        return None

    if not os.path.isfile(poison_path):
        raise FileNotFoundError("Poison checkpoint not found: " + poison_path)

    state = torch.load(poison_path, map_location="cpu")

    if not isinstance(state, dict) or "poison" not in state:
        raise RuntimeError(
            "{} does not contain state['poison'].".format(poison_path)
        )

    poison_items = state["poison"]
    poison_tensors = []

    for item in poison_items:
        if isinstance(item, (tuple, list)):
            tensor = item[0]
        else:
            tensor = item

        if not torch.is_tensor(tensor):
            raise TypeError(
                "Unexpected poison item type: {}".format(type(tensor))
            )

        # 兼容 [1,C,H,W] 和 [C,H,W]
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor[0]
        if tensor.dim() != 3:
            raise ValueError(
                "Expected poison tensor [C,H,W], got {}".format(
                    tuple(tensor.size())
                )
            )

        poison_tensors.append(tensor.detach().float())

    if not poison_tensors:
        raise RuntimeError("No poison tensor found in " + poison_path)

    batch = torch.stack(poison_tensors, dim=0).to(device)
    print(
        "Loaded {} poisons from {}: {}".format(
            len(poison_tensors), poison_path, tuple(batch.size())
        )
    )
    return batch


def target_tensor_to_pil(target):
    """
    fetch_target 返回的是已经 Normalize 的 Tensor。
    先用攻击代码相同 mean/std 反归一化，再转成 PIL。
    """
    x = target.detach().cpu()
    if x.dim() == 4:
        x = x[0]

    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    x = torch.clamp(x * std + mean, 0.0, 1.0)
    return TF.to_pil_image(x)


def pil_to_project_tensor(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return transform(image.convert("RGB"))


def make_target_variants(target):
    image = target_tensor_to_pil(target)

    variants_pil = OrderedDict([
        ("original", image),
        ("rotate_-10", TF.rotate(image, angle=-10)),
        ("rotate_+10", TF.rotate(image, angle=10)),
        (
            "scale_0.90",
            TF.affine(
                image,
                angle=0,
                translate=[0, 0],
                scale=0.90,
                shear=0,
            ),
        ),
        (
            "scale_1.10",
            TF.affine(
                image,
                angle=0,
                translate=[0, 0],
                scale=1.10,
                shear=0,
            ),
        ),
        ("horizontal_flip", TF.hflip(image)),
    ])

    variants_tensor = OrderedDict()
    for name, pil_image in variants_pil.items():
        variants_tensor[name] = pil_to_project_tensor(pil_image).unsqueeze(0)

    return variants_pil, variants_tensor


def save_target_grid(variants_pil, output_path):
    names = list(variants_pil.keys())
    columns = 3
    rows = int(math.ceil(float(len(names)) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(9, 3 * rows))
    axes = np.asarray(axes).reshape(-1)

    for index, ax in enumerate(axes):
        ax.axis("off")
        if index < len(names):
            name = names[index]
            ax.imshow(variants_pil[name])
            ax.set_title(name)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_project_features(net, inputs):
    """
    严格对应当前 ResNet.return_layers()：

        multi_layer=True 返回一个 list，而不是已经 CONCAT 的 Tensor：
        [layer2_flat, layer3_flat, layer4_pooled]

    因此这里显式执行 torch.cat，得到与 MSP 应使用的联合表示。
    注意：第三项 layer4_pooled 与 penu=True 的输出相同，所以主评价中不再
    重复加入 penultimate。
    """
    with torch.no_grad():
        multi_features = net(x=inputs, multi_layer=True)

    if not isinstance(multi_features, (list, tuple)):
        raise TypeError(
            "Expected multi_layer=True to return list/tuple, got {}. "
            "Please check ResNet.return_layers().".format(
                type(multi_features)
            )
        )

    if len(multi_features) != 3:
        raise RuntimeError(
            "Expected return_layers() to return 3 features "
            "[layer2, layer3, pooled layer4], got {}.".format(
                len(multi_features)
            )
        )

    layer2_feature = multi_features[0].detach()
    layer3_feature = multi_features[1].detach()
    layer4_pooled_feature = multi_features[2].detach()

    concat_feature = torch.cat(
        [
            flatten_feature(layer2_feature),
            flatten_feature(layer3_feature),
            flatten_feature(layer4_pooled_feature),
        ],
        dim=1,
    ).detach()

    # 不再单独调用 penu=True 做数值一致性检查。
    # 当 test_dp > 0 时，模型即使在 eval() 下也会主动执行随机 dropout；
    # 两次独立前向传播使用不同掩码，数值自然不同。几何分析统一采用
    # 本次 multi_layer=True 前向得到的 layer4_pooled，避免随机掩码错配。

    result = OrderedDict()
    result["layer2_flat"] = layer2_feature
    result["layer3_flat"] = layer3_feature
    result["layer4_pooled"] = layer4_pooled_feature
    result["concat"] = concat_feature
    return result


def extract_block_features(net, inputs):
    """
    block=True 对应 get_block_feats()，仅用于空间特征图可视化。

    它与 multi_layer=True 不是同一组输出：
    - multi_layer=True: layer2、layer3、池化后的 layer4；
    - block=True: layer4 内部若干 block 输出 + 池化后的最终特征。

    middle_feat_num 只影响这里，不影响 multi_layer=True。
    """
    with torch.no_grad():
        block_features = net(x=inputs, block=True)

    if not isinstance(block_features, (list, tuple)):
        raise TypeError(
            "Expected block=True to return list/tuple, got {}.".format(
                type(block_features)
            )
        )

    result = OrderedDict()
    for index, feature in enumerate(block_features):
        result["late_block_{}".format(index + 1)] = feature.detach()
    return result


def flatten_feature(feature):
    return feature.reshape(feature.size(0), -1)


def bp_equation_2(target_feature, poison_feature, eps=1e-12):
    """
    BP 论文公式 (2) 在单个模型/单个表示上的值：

        0.5 * ||phi(x_t) - 1/k sum_j phi(x_p^j)||_2^2
              / ||phi(x_t)||_2^2
    """
    target_vector = flatten_feature(target_feature)
    poison_matrix = flatten_feature(poison_feature)

    if target_vector.size(0) != 1:
        raise ValueError(
            "Target batch size must be 1, got {}".format(
                target_vector.size(0)
            )
        )

    poison_center = poison_matrix.mean(dim=0, keepdim=True)
    numerator = (target_vector - poison_center).pow(2).sum()
    denominator = target_vector.pow(2).sum() + eps
    return float((0.5 * numerator / denominator).item())


def cp_equation_1(target_feature, poison_feature, tol=1e-6, eps=1e-12):
    """
    与 trainer.get_CP_loss() 一致：先在概率单纯形上优化凸组合系数，
    再计算 CP 公式 (1) 的归一化重构损失。

    返回
    ----
    loss:
        0.5 * ||phi(x_t)-sum_j c_j phi(x_p^j)||^2 / ||phi(x_t)||^2
    coeffs:
        优化后的凸组合系数。
    entropy:
        系数熵。均匀系数时最大；系数越偏斜，熵越低。
    """
    target_vector = flatten_feature(target_feature)
    poison_matrix = flatten_feature(poison_feature)

    if target_vector.size(0) != 1:
        raise ValueError(
            "Target batch size must be 1, got {}".format(
                target_vector.size(0)
            )
        )

    poison_num = poison_matrix.size(0)
    init_coeffs = (
        torch.ones(poison_num, 1, device=poison_matrix.device) /
        float(poison_num)
    )

    coeffs = least_squares_simplex(
        A=poison_matrix.t().detach(),
        b=target_vector.t().detach(),
        x_init=init_coeffs,
        tol=tol,
        device=str(poison_matrix.device),
    )

    reconstruction = torch.sum(
        coeffs.view(-1, 1) * poison_matrix,
        dim=0,
        keepdim=True,
    )
    residual = target_vector - reconstruction
    denominator = target_vector.pow(2).sum() + eps
    loss = 0.5 * residual.pow(2).sum() / denominator

    coeff_flat = coeffs.view(-1)
    entropy = -torch.sum(
        coeff_flat * torch.log2(coeff_flat + eps)
    )

    return (
        float(loss.item()),
        coeff_flat.detach().cpu().numpy(),
        float(entropy.item()),
    )


def cosine_feature_drift(original_feature, transformed_feature, eps=1e-12):
    a = flatten_feature(original_feature)
    b = flatten_feature(transformed_feature)

    a = a / (a.norm(p=2, dim=1, keepdim=True) + eps)
    b = b / (b.norm(p=2, dim=1, keepdim=True) + eps)
    similarity = torch.sum(a * b, dim=1)
    return float((1.0 - similarity.mean()).item())


def extract_all_target_features(net, variant_tensors, device, seed):
    output = OrderedDict()
    for index, (name, tensor) in enumerate(variant_tensors.items()):
        # 每个输入使用固定种子，便于复现实验
        set_seed(seed + index)
        output[name] = extract_project_features(net, tensor.to(device))
    return output


def compute_drift_table(target_features):
    representations = list(next(iter(target_features.values())).keys())
    original = target_features["original"]
    matrix = []

    for transform_name, features in target_features.items():
        row = []
        for representation in representations:
            row.append(
                cosine_feature_drift(
                    original[representation],
                    features[representation],
                )
            )
        matrix.append(row)

    return list(target_features.keys()), representations, np.asarray(matrix)


def compute_geometry_loss_table(
    target_features,
    poison_features_by_method,
    tol,
):
    """
    同时计算两种几何指标：

    1. bp_eq2_uniform:
       BP 公式 (2)，固定系数 1/k，即目标到投毒质心的残差。

    2. cp_eq1_optimized:
       与当前 trainer.get_CP_loss() 一致，先优化单纯形系数，再计算
       CP 公式 (1) 的残差。

    这样可以判断 poison 文件究竟更符合 CP 几何关系还是 BP 质心关系。
    """
    rows = []

    for method_name, poison_features in poison_features_by_method.items():
        original_metrics = {}

        for representation in poison_features.keys():
            bp_loss = bp_equation_2(
                target_features["original"][representation],
                poison_features[representation],
            )
            cp_loss, cp_coeffs, cp_entropy = cp_equation_1(
                target_features["original"][representation],
                poison_features[representation],
                tol=tol,
            )
            original_metrics[representation] = {
                "bp": bp_loss,
                "cp": cp_loss,
            }

        for transform_name, features in target_features.items():
            for representation in poison_features.keys():
                bp_loss = bp_equation_2(
                    features[representation],
                    poison_features[representation],
                )
                cp_loss, cp_coeffs, cp_entropy = cp_equation_1(
                    features[representation],
                    poison_features[representation],
                    tol=tol,
                )

                bp_original = original_metrics[representation]["bp"]
                cp_original = original_metrics[representation]["cp"]

                rows.append({
                    "method": method_name,
                    "transform": transform_name,
                    "representation": representation,
                    "bp_eq2_uniform": bp_loss,
                    "bp_delta_from_original": bp_loss - bp_original,
                    "bp_ratio_to_original":
                        (bp_loss + 1e-12) / (bp_original + 1e-12),
                    "cp_eq1_optimized": cp_loss,
                    "cp_delta_from_original": cp_loss - cp_original,
                    "cp_ratio_to_original":
                        (cp_loss + 1e-12) / (cp_original + 1e-12),
                    "cp_coefficient_entropy": cp_entropy,
                    "cp_coefficients":
                        " ".join(
                            ["{:.8f}".format(value) for value in cp_coeffs]
                        ),
                })

    return rows

def long_rows_to_matrix(rows, method_name, value_key):
    selected = [
        row for row in rows
        if row["method"] == method_name
    ]

    transform_names = []
    representation_names = []

    for row in selected:
        if row["transform"] not in transform_names:
            transform_names.append(row["transform"])
        if row["representation"] not in representation_names:
            representation_names.append(row["representation"])

    matrix = np.zeros(
        (len(transform_names), len(representation_names)),
        dtype=np.float64,
    )

    for row in selected:
        i = transform_names.index(row["transform"])
        j = representation_names.index(row["representation"])
        matrix[i, j] = row[value_key]

    return transform_names, representation_names, matrix


def save_matrix_csv(matrix, row_names, column_names, output_path):
    with open(str(output_path), "w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["sample"] + list(column_names))
        for name, values in zip(row_names, matrix):
            writer.writerow(
                [name] + ["{:.10f}".format(value) for value in values]
            )


def save_long_csv(rows, output_path):
    fieldnames = [
        "method",
        "transform",
        "representation",
        "bp_eq2_uniform",
        "bp_delta_from_original",
        "bp_ratio_to_original",
        "cp_eq1_optimized",
        "cp_delta_from_original",
        "cp_ratio_to_original",
        "cp_coefficient_entropy",
        "cp_coefficients",
    ]

    with open(str(output_path), "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def plot_heatmap(matrix, row_names, column_names, title, output_path):
    fig, ax = plt.subplots(
        figsize=(
            max(8, 1.55 * len(column_names)),
            max(4.5, 0.72 * len(row_names)),
        )
    )
    image = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(column_names)))
    ax.set_xticklabels(column_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                "{:.3f}".format(matrix[row_index, column_index]),
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def activation_to_maps(activation):
    if activation.dim() != 4:
        return None
    return activation[0].detach().cpu().numpy()


def visualize_feature_maps(feature_dict, output_dir, prefix, max_channels):
    """
    只对 block=True 返回的 4D 特征图进行可视化。
    layer2_flat/layer3_flat/layer4_pooled/concat 均是二维向量，因此不画成空间特征图。
    """
    for representation, activation in feature_dict.items():
        maps = activation_to_maps(activation)
        if maps is None:
            continue

        scores = np.mean(np.abs(maps), axis=(1, 2))
        selected = np.argsort(scores)[::-1][:max_channels]

        columns = int(math.ceil(math.sqrt(len(selected))))
        rows = int(math.ceil(float(len(selected)) / columns))

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(1.7 * columns, 1.7 * rows),
        )
        axes = np.asarray(axes).reshape(-1)

        for index, ax in enumerate(axes):
            ax.axis("off")
            if index < len(selected):
                channel = int(selected[index])
                feature_map = maps[channel]
                minimum = feature_map.min()
                maximum = feature_map.max()
                feature_map = (
                    (feature_map - minimum) /
                    (maximum - minimum + 1e-12)
                )
                ax.imshow(feature_map, cmap="gray")
                ax.set_title("ch {}".format(channel), fontsize=7)

        fig.suptitle("{} - {}".format(prefix, representation))
        fig.tight_layout()
        fig.savefig(
            str(
                output_dir /
                "{}_{}_feature_maps.png".format(prefix, representation)
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def visualize_first_conv_filters(net, output_path, max_kernels):
    module = unwrap_model(net)
    first_conv = None
    first_conv_name = None

    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            first_conv_name = name
            break

    if first_conv is None:
        print("[Warning] No Conv2d layer found.")
        return

    weights = first_conv.weight.detach().cpu().numpy()
    count = min(max_kernels, weights.shape[0])
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(float(count) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(1.35 * columns, 1.35 * rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for index, ax in enumerate(axes):
        ax.axis("off")
        if index >= count:
            continue

        kernel = weights[index]
        if kernel.shape[0] >= 3:
            image = np.transpose(kernel[:3], (1, 2, 0))
        else:
            image = kernel[0]

        minimum = image.min()
        maximum = image.max()
        image = (image - minimum) / (maximum - minimum + 1e-12)
        ax.imshow(image, interpolation="nearest")
        ax.set_title(str(index), fontsize=7)

    fig.suptitle("First convolution kernels: {}".format(first_conv_name))
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def normalized_vector(feature):
    vector = flatten_feature(feature)
    return vector / (vector.norm(p=2, dim=1, keepdim=True) + 1e-12)


def pca_2d(matrix):
    """
    返回二维投影和前两个主成分的解释方差比例。
    """
    matrix = matrix.astype(np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)

    _, singular_values, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    points = np.dot(centered, vt[:2].T)

    explained_variance = singular_values ** 2
    total_variance = explained_variance.sum()
    if total_variance <= 1e-20:
        explained_ratio = np.zeros_like(explained_variance)
    else:
        explained_ratio = explained_variance / total_variance

    return points, explained_ratio[:2]


def convex_hull_2d(points):
    """
    Andrew monotonic-chain 二维凸包，不依赖 scipy。
    返回按边界顺序排列的二维点。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 2:
        return points

    unique_points = sorted(set(
        (float(point[0]), float(point[1]))
        for point in points
    ))

    if len(unique_points) <= 2:
        return np.asarray(unique_points)

    def cross(origin, a, b):
        return (
            (a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    lower = []
    for point in unique_points:
        while (
            len(lower) >= 2
            and cross(lower[-2], lower[-1], point) <= 0
        ):
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique_points):
        while (
            len(upper) >= 2
            and cross(upper[-2], upper[-1], point) <= 0
        ):
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1])


def _feature_matrix_for_distance(feature):
    """
    展平但不逐样本归一化。

    后续统一除以原始目标的范数。这样：
        d_full = ||target - poison_centroid|| / ||target||
    且：
        BP Eq.(2) = 0.5 * d_full^2

    因而图中标注的高维距离与 BP 公式严格一致。
    """
    return flatten_feature(feature).detach().cpu().numpy().astype(
        np.float64
    )


def make_pca_plot(
    target_features,
    poison_features_by_method,
    representation,
    output_path,
    summary_rows=None,
):
    """
    更直观的 PCA 几何图：

    1. 只给原始目标、两组毒样本及其质心添加主要标记；
    2. 用虚线直接连接原始目标与 BP/MSP 质心；
    3. 在线段旁同时标注：
         d_2D   : PCA 投影中的二维距离；
         d_full : 原始高维空间中的归一化距离；
    4. 绘制 BP/MSP 毒样本的二维凸包；
    5. 坐标轴显示 PC1/PC2 的解释方差比例。

    判断方法：
        两条连线中，d_full 越小越好。
    d_2D 仅用于视觉辅助，正式结论以 d_full 或 BP Eq.(2) 为准。
    """
    if "original" not in target_features:
        raise KeyError("target_features must contain 'original'.")

    original_raw = _feature_matrix_for_distance(
        target_features["original"][representation]
    )
    if original_raw.shape[0] != 1:
        raise ValueError("Original target batch size must be 1.")

    target_norm = float(np.linalg.norm(original_raw[0]))
    if target_norm <= 1e-12:
        target_norm = 1.0

    vectors = []
    metadata = []

    # 所有向量统一除以原始目标范数，而不是逐样本归一化。
    # 这样可保持 BP Eq.(2) 所使用的质心与距离关系。
    for transform_name, features in target_features.items():
        raw = _feature_matrix_for_distance(
            features[representation]
        )[0]
        vectors.append(raw / target_norm)
        metadata.append({
            "kind": "target",
            "name": transform_name,
        })

    method_info = OrderedDict()

    for method_name, poison_features in poison_features_by_method.items():
        poison_raw = _feature_matrix_for_distance(
            poison_features[representation]
        )
        poison_scaled = poison_raw / target_norm
        centroid_scaled = poison_scaled.mean(axis=0)

        poison_indices = []
        for poison_index, vector in enumerate(poison_scaled):
            poison_indices.append(len(vectors))
            vectors.append(vector)
            metadata.append({
                "kind": "poison",
                "method": method_name,
                "name": "poison{}".format(poison_index + 1),
            })

        centroid_index = len(vectors)
        vectors.append(centroid_scaled)
        metadata.append({
            "kind": "centroid",
            "method": method_name,
            "name": method_name + " centroid",
        })

        full_distance = float(
            np.linalg.norm(
                original_raw[0] / target_norm - centroid_scaled
            )
        )
        bp_eq2_value = 0.5 * full_distance ** 2

        method_info[method_name] = {
            "poison_indices": poison_indices,
            "centroid_index": centroid_index,
            "full_distance": full_distance,
            "bp_eq2": bp_eq2_value,
        }

    matrix = np.stack(vectors, axis=0)
    points, explained_ratio = pca_2d(matrix)

    original_index = None
    transformed_target_indices = []

    for index, item in enumerate(metadata):
        if item["kind"] == "target":
            if item["name"] == "original":
                original_index = index
            else:
                transformed_target_indices.append(index)

    if original_index is None:
        raise RuntimeError("Original target was not added to PCA.")

    # 使用 matplotlib 默认颜色循环，避免手工指定固定颜色。
    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key().get("color", [])
    if not default_colors:
        default_colors = [None]

    method_names = list(method_info.keys())
    method_colors = {}
    for method_index, method_name in enumerate(method_names):
        method_colors[method_name] = default_colors[
            (method_index + 1) % len(default_colors)
        ]

    target_color = default_colors[0]

    fig, ax = plt.subplots(figsize=(10, 7.5))

    # 变换目标：弱化显示，不逐个写文字，减少拥挤。
    if transformed_target_indices:
        ax.scatter(
            points[transformed_target_indices, 0],
            points[transformed_target_indices, 1],
            marker="o",
            s=42,
            facecolors="none",
            edgecolors=target_color,
            alpha=0.55,
            linewidths=1.2,
            label="Transformed targets",
        )

    # 原始目标：最醒目的星形点。
    target_point = points[original_index]
    ax.scatter(
        [target_point[0]],
        [target_point[1]],
        marker="*",
        s=280,
        color=target_color,
        edgecolors="black",
        linewidths=0.8,
        zorder=8,
        label="Original target",
    )
    ax.annotate(
        "Target",
        (target_point[0], target_point[1]),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    marker_map = {
        "BP": "^",
        "MSP": "s",
    }
    centroid_marker_map = {
        "BP": "X",
        "MSP": "P",
    }

    for method_index, method_name in enumerate(method_names):
        info = method_info[method_name]
        color = method_colors[method_name]
        poison_indices = info["poison_indices"]
        centroid_index = info["centroid_index"]

        poison_points = points[poison_indices]
        centroid_point = points[centroid_index]

        marker = marker_map.get(
            method_name,
            ["^", "s", "D", "v"][method_index % 4],
        )
        centroid_marker = centroid_marker_map.get(
            method_name,
            ["X", "P", "D", "*"][method_index % 4],
        )

        # 二维凸包只作视觉辅助。
        hull = convex_hull_2d(poison_points)
        if len(hull) >= 3:
            ax.fill(
                hull[:, 0],
                hull[:, 1],
                color=color,
                alpha=0.08,
                zorder=1,
            )
            closed_hull = np.vstack([hull, hull[0]])
            ax.plot(
                closed_hull[:, 0],
                closed_hull[:, 1],
                color=color,
                alpha=0.55,
                linewidth=1.2,
                zorder=2,
            )

        ax.scatter(
            poison_points[:, 0],
            poison_points[:, 1],
            marker=marker,
            s=72,
            color=color,
            alpha=0.75,
            label="{} poisons".format(method_name),
            zorder=4,
        )

        ax.scatter(
            [centroid_point[0]],
            [centroid_point[1]],
            marker=centroid_marker,
            s=230,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            label="{} centroid".format(method_name),
            zorder=7,
        )

        # 目标到质心的二维距离。
        projected_distance = float(
            np.linalg.norm(target_point - centroid_point)
        )
        info["projected_distance"] = projected_distance

        # 直接画目标到质心的箭头。
        ax.annotate(
            "",
            xy=(centroid_point[0], centroid_point[1]),
            xytext=(target_point[0], target_point[1]),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 2.2,
                "linestyle": "--",
                "color": color,
                "alpha": 0.9,
            },
            zorder=6,
        )

        # 在线段中点附近标注二维/高维距离。
        midpoint = 0.5 * (target_point + centroid_point)
        label_text = (
            "{}\n"
            r"$d_{{2D}}={:.3f}$"
            "\n"
            r"$d_{{full}}={:.3f}$"
        ).format(
            method_name,
            projected_distance,
            info["full_distance"],
        )

        vertical_offset = 12 if method_index % 2 == 0 else -42
        ax.annotate(
            label_text,
            (midpoint[0], midpoint[1]),
            xytext=(7, vertical_offset),
            textcoords="offset points",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.88,
            },
            zorder=9,
        )

        ax.annotate(
            "{} centroid".format(method_name),
            (centroid_point[0], centroid_point[1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

        if summary_rows is not None:
            summary_rows.append({
                "representation": representation,
                "method": method_name,
                "pca_2d_distance": projected_distance,
                "full_normalized_distance": info["full_distance"],
                "bp_eq2": info["bp_eq2"],
                "pc1_explained_ratio": float(
                    explained_ratio[0]
                    if len(explained_ratio) > 0 else 0.0
                ),
                "pc2_explained_ratio": float(
                    explained_ratio[1]
                    if len(explained_ratio) > 1 else 0.0
                ),
            })

    explained_pc1 = (
        100.0 * explained_ratio[0]
        if len(explained_ratio) > 0 else 0.0
    )
    explained_pc2 = (
        100.0 * explained_ratio[1]
        if len(explained_ratio) > 1 else 0.0
    )

    ax.set_title(
        "Target-to-centroid distances in {} representation\n"
        "Shorter full-dimensional distance indicates better alignment".format(
            representation
        )
    )
    ax.set_xlabel("PC1 ({:.1f}% variance)".format(explained_pc1))
    ax.set_ylabel("PC2 ({:.1f}% variance)".format(explained_pc2))
    ax.grid(True, alpha=0.25)

    # 去除重复 legend 项。
    handles, labels = ax.get_legend_handles_labels()
    unique = OrderedDict()
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    ax.legend(
        unique.values(),
        unique.keys(),
        loc="best",
        frameon=True,
        fontsize=9,
    )

    ax.text(
        0.01,
        0.01,
        (
            r"$d_{full}=\|\phi(x_t)-\mu_p\|_2/\|\phi(x_t)\|_2$"
            "\n"
            r"BP Eq. (2) $=\frac{1}{2}d_{full}^{2}$"
            "\nPCA distance is visualization only."
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.84,
        },
    )

    fig.tight_layout()
    fig.savefig(
        str(output_path),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_pca_distance_summary(rows, output_path):
    fieldnames = [
        "representation",
        "method",
        "pca_2d_distance",
        "full_normalized_distance",
        "bp_eq2",
        "pc1_explained_ratio",
        "pc2_explained_ratio",
    ]

    with open(str(output_path), "w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    if not args.bp_poison_path and not args.msp_poison_path:
        raise ValueError(
            "Provide --bp-poison-path and/or --msp-poison-path."
        )

    if args.multi_scale_layers < 0:
        raise ValueError("--multi-scale-layers must be non-negative.")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is unavailable; using CPU.")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 与 craft_poisons_transfer.py 完全相同的加载方式
    net = load_pretrained_net(
        args.analysis_net,
        args.analysis_chk_name,
        model_chk_path=args.model_resume_path,
        test_dp=args.analysis_dp,
    )
    set_multiscale_layers(net, args.multi_scale_layers)
    net.eval()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    target = fetch_target(
        args.target_label,
        args.target_index,
        args.num_per_class,
        subset=args.subset,
        path=args.train_data_path,
        transforms=transform_test,
    ).to(device)

    print("Target tensor:", tuple(target.size()))

    variants_pil, variants_tensor = make_target_variants(target)
    save_target_grid(
        variants_pil,
        output_dir / "figure_a_target_transformations.png",
    )

    target_features = extract_all_target_features(
        net,
        variants_tensor,
        device,
        args.seed,
    )

    print("Representations:")
    for name, feature in target_features["original"].items():
        print("  {}: {}".format(name, tuple(feature.size())))

    poison_batches = OrderedDict()

    if args.bp_poison_path:
        poison_batches["BP"] = load_poison_batch(
            args.bp_poison_path,
            device,
        )

    if args.msp_poison_path:
        poison_batches["MSP"] = load_poison_batch(
            args.msp_poison_path,
            device,
        )

    poison_features_by_method = OrderedDict()
    for index, (method_name, poison_batch) in enumerate(
        poison_batches.items()
    ):
        set_seed(args.seed + 100 + index)
        poison_features_by_method[method_name] = extract_project_features(
            net,
            poison_batch,
        )

    # 1. 普通目标特征漂移：只说明变换敏感性
    drift_rows, drift_columns, drift_matrix = compute_drift_table(
        target_features
    )
    save_matrix_csv(
        drift_matrix,
        drift_rows,
        drift_columns,
        output_dir / "target_feature_drift.csv",
    )
    plot_heatmap(
        drift_matrix,
        drift_rows,
        drift_columns,
        "Cosine drift relative to the original target",
        output_dir / "figure_b_target_feature_drift.png",
    )

    # 2. 同时评价：
    #    - BP 公式 (2)：固定 1/k 质心；
    #    - 当前 trainer.py 实际使用的 CP 公式 (1)：优化单纯形系数。
    geometry_rows = compute_geometry_loss_table(
        target_features,
        poison_features_by_method,
        args.tol,
    )
    save_long_csv(
        geometry_rows,
        output_dir / "geometry_loss_results.csv",
    )

    for method_name in poison_features_by_method.keys():
        row_names, column_names, bp_matrix = long_rows_to_matrix(
            geometry_rows,
            method_name,
            "bp_eq2_uniform",
        )
        plot_heatmap(
            bp_matrix,
            row_names,
            column_names,
            "{} poisons: BP Eq. (2), uniform centroid".format(
                method_name
            ),
            output_dir /
            "figure_c_{}_bp_eq2_uniform.png".format(
                method_name.lower()
            ),
        )

        row_names, column_names, bp_ratio = long_rows_to_matrix(
            geometry_rows,
            method_name,
            "bp_ratio_to_original",
        )
        plot_heatmap(
            bp_ratio,
            row_names,
            column_names,
            "{} poisons: BP Eq. (2) ratio to original".format(
                method_name
            ),
            output_dir /
            "figure_d_{}_bp_eq2_ratio.png".format(
                method_name.lower()
            ),
        )

        row_names, column_names, cp_matrix = long_rows_to_matrix(
            geometry_rows,
            method_name,
            "cp_eq1_optimized",
        )
        plot_heatmap(
            cp_matrix,
            row_names,
            column_names,
            "{} poisons: CP Eq. (1), optimized coefficients".format(
                method_name
            ),
            output_dir /
            "figure_e_{}_cp_eq1_optimized.png".format(
                method_name.lower()
            ),
        )

        row_names, column_names, entropy_matrix = long_rows_to_matrix(
            geometry_rows,
            method_name,
            "cp_coefficient_entropy",
        )
        plot_heatmap(
            entropy_matrix,
            row_names,
            column_names,
            "{} poisons: optimized coefficient entropy".format(
                method_name
            ),
            output_dir /
            "figure_f_{}_coefficient_entropy.png".format(
                method_name.lower()
            ),
        )

    # 3. 空间特征图：单独使用 block=True。
    #    这些图只用于展示激活，不参与上面的公平几何指标计算。
    original_block_features = extract_block_features(
        net,
        variants_tensor["original"].to(device),
    )
    rotated_block_features = extract_block_features(
        net,
        variants_tensor["rotate_+10"].to(device),
    )
    visualize_feature_maps(
        original_block_features,
        output_dir,
        "target_original",
        args.num_feature_maps,
    )
    visualize_feature_maps(
        rotated_block_features,
        output_dir,
        "target_rotate_+10",
        args.num_feature_maps,
    )

    # 4. 第一层卷积核
    visualize_first_conv_filters(
        net,
        output_dir / "first_conv_kernels.png",
        args.num_filter_kernels,
    )

    # 5. PCA 距离图：直接连接原始目标与 BP/MSP 质心，
    #    同时标注二维投影距离和高维归一化距离。
    pca_distance_rows = []
    make_pca_plot(
        target_features,
        poison_features_by_method,
        "layer4_pooled",
        output_dir / "figure_g_pca_distance_layer4_pooled.png",
        summary_rows=pca_distance_rows,
    )
    make_pca_plot(
        target_features,
        poison_features_by_method,
        "concat",
        output_dir / "figure_h_pca_distance_concat.png",
        summary_rows=pca_distance_rows,
    )
    save_pca_distance_summary(
        pca_distance_rows,
        output_dir / "pca_distance_summary.csv",
    )

    print("Finished. Results are in:", str(output_dir.resolve()))
    print("Recommended paper figures:")
    print("  figure_b_target_feature_drift.png")
    print("  figure_c_bp_bp_eq2_uniform.png")
    print("  figure_c_msp_bp_eq2_uniform.png")
    print("  figure_e_bp_cp_eq1_optimized.png")
    print("  figure_g_pca_distance_layer4_pooled.png")
    print("  figure_h_pca_distance_concat.png")
    print("Numerical results:")
    print("  geometry_loss_results.csv")
    print("  pca_distance_summary.csv")


if __name__ == "__main__":
    freeze_support()
    main()
