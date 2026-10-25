# MSP 三层特征提取

每个 *_msp.py 是独立模型文件，保留原有模型类、构造函数、参数和分类接口。
新增方法不引入参数，原模型 state_dict 可按原方式加载。

用法：
```python
from models.resnet_msp import ResNet18
import torch
model = ResNet18().eval()
x = torch.randn(1, 3, 32, 32)
pooled = model.penultimate1(x)  # [1, 896]
flattened = model.penultimate2(x)  # [1, 57344]
logits = model(x)  # 原分类接口
```

- penultimate1：各层使用 F.avg_pool2d(feature, feature.shape[-2:]) 全局平均池化，再 flatten(1)，最后沿 dim=1 拼接。
- penultimate2：三层全部直接 flatten(1)，不做任何池化，再沿 dim=1 拼接。
- 输出保留 batch 维，形状为 [B, n1+n2+n3]；B=1 时即 [1, n]。
- forward(x)、forward(x, penu=True)、penultimate(x) 保留原行为。需要三层特征时直接调用新增方法。
- 没有 detach/no_grad，支持输入梯度反传；提取模式与训练/测试随机性仍由原模型控制。
- 三层是完整 stage 的输出，而非末尾三个卷积算子。
- DenseNet 使用 dense2/3/4 的原始输出；dense4 在原最终 bn/relu 之前。
- MobileNetV2 使用最后三个 cfg stage (96/160/320)，不包含末尾 1280 通道 conv2 投影。
- 新方法的池化核随特征图尺寸变化；原分类路径仍保留原上传代码中的固定池化核。

## 选层
- resnet: layer2, layer3, layer4
- senet: layer2, layer3, layer4
- dpn: layer2, layer3, layer4
- resnext: layer1, layer2, layer3
- densenet: dense2, dense3, dense4 (before final bn/relu)
- googlenet: e4, a5, b5
- mobilenetv2: last three cfg stages: 96, 160, 320 channels

验证：所有文件通过 Python 语法编译；当前运行环境未安装 PyTorch，未进行真实张量运行测试。
