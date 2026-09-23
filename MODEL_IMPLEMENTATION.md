# 模型实现与调优说明

## 1. 模型总体思路

本文实现的是一种**监督增强的 Transformer-Isolation Forest 融合异常检测模型**。模型不是单纯依赖传统异常检测算法，也不是只使用深度学习分类器，而是将两类方法结合起来：

- 使用 Transformer 学习输入数据的高层特征表示；
- 使用监督分类任务增强 Transformer 对正常样本和异常样本的区分能力；
- 使用 Isolation Forest 对 Transformer 提取的特征进行异常评分；
- 将 Transformer 分类概率与 Isolation Forest 异常分数融合，得到最终异常分数；
- 使用验证集选择最佳阈值，再在测试集上评估模型效果。

整体流程如下：

```text
原始数据
→ 数据清洗与特征工程
→ 标准化 / 归一化
→ Transformer 编码器
→ 高层特征表示
→ Transformer 分类概率
→ Isolation Forest 异常分数
→ 分数融合
→ 阈值判别
→ 输出正常 / 异常
```

## 2. 使用的数据集

### 2.1 UCI Localization 跌倒检测数据

使用 UCI Localization Data for Person Activity 数据集，将 `falling` 作为异常类别，其余行为作为正常类别。

预处理方式：

- 按 `(sequence, tag_id)` 分组，避免不同人员、不同传感器数据混合；
- 每组内部按时间排序；
- 使用滑动窗口切分时序数据；
- 窗口长度为 `50`，步长为 `25`；
- 只要窗口内出现 `falling`，该窗口就标记为异常；
- 构造 x、y、z、合运动幅值、倾斜角、局部均值、局部标准差等特征；
- 使用 Z-Score 标准化。

### 2.2 CIC-IDS2017 网络入侵检测数据

使用 CIC-IDS2017 的 `Friday-WorkingHours-Afternoon-DDos` 机器学习子集，将 `DDoS` 作为异常类别，`BENIGN` 作为正常类别。

预处理方式：

- 替换 `inf` 和 `-inf` 为缺失值；
- 使用中位数填充缺失值；
- 删除高度相关特征，相关系数阈值为 `0.9`；
- 使用 Random Forest 计算特征重要性；
- 支持 `top7` 和累计重要性 `95%` 两种特征选择模式；
- 使用 Min-Max 归一化；
- 将表格特征转换为 Transformer 可处理的序列形式。

## 3. 核心算法

## 3.1 Transformer 编码器

Transformer 用作特征提取器。它通过自注意力机制学习不同时间步或不同特征之间的关联关系。

在跌倒检测任务中，输入形状为：

```text
[batch_size, window_size, feature_dim]
```

在网络入侵检测任务中，输入形状为：

```text
[batch_size, selected_feature_count, 1]
```

模型结构包括：

- 输入线性映射层；
- 位置编码；
- Transformer Encoder；
- 平均池化层；
- 重构输出层；
- 二分类输出层。

Transformer 同时承担两个任务：

1. **重构任务**

   通过重构输入数据，使模型学习稳定的数据表示。

2. **分类任务**

   通过正常 / 异常标签训练分类头，使特征空间更容易区分异常样本。

综合损失函数为：

```text
L = L_reconstruction + λ * L_classification
```

其中：

- `L_reconstruction` 使用 MSE 损失；
- `L_classification` 使用带类别权重的 BCEWithLogitsLoss；
- `λ` 为监督损失权重，默认值为 `0.5`。

类别权重用于缓解异常样本和正常样本比例不均衡的问题。

## 3.2 Isolation Forest

Isolation Forest 用于对 Transformer 提取出的高层特征进行异常评分。

基本思想是：

- 异常样本通常更容易被随机切分隔离；
- 如果一个样本在孤立树中的平均路径长度较短，则更可能是异常；
- 多棵孤立树共同计算异常分数，提高稳定性。

在本文实现中，Isolation Forest 只使用训练集中的正常样本进行训练。这一点很关键，因为异常检测任务中如果把大量异常样本也喂给 Isolation Forest，模型可能会把异常模式当成正常分布，导致异常分数失效。

主要参数：

```text
n_estimators = 200
max_samples = 0.8
contamination = 0.1
random_state = 42
```

快速实验模式下会降低树数量以加快运行。

## 3.3 分数融合

最终模型不是单独使用 Transformer，也不是单独使用 Isolation Forest，而是融合两者分数。

融合公式为：

```text
S_final = α * S_transformer + (1 - α) * S_iforest
```

其中：

- `S_transformer` 表示 Transformer 输出的异常分类概率；
- `S_iforest` 表示 Isolation Forest 输出的归一化异常分数；
- `S_final` 表示最终异常分数；
- `α` 为融合权重。

当前实验中使用：

```text
α = 0.65
```

也就是说，最终结果更偏向 Transformer 的监督判别能力，同时保留 Isolation Forest 对特征空间异常程度的评分能力。

## 4. 对比方法与消融实验

为了证明模型有效性，实验中设计了多个基线和消融模型。

| 方法 | 作用 |
|---|---|
| Z-Score | 传统统计异常检测方法 |
| One-Class SVM | 经典单类异常检测算法 |
| Isolation Forest | 原始孤立森林模型 |
| Transformer-only | 只使用 Transformer 重构 / 分类分数 |
| AutoEncoder + IF | 使用 AutoEncoder 替代 Transformer |
| Transformer + IF | 无监督 Transformer 特征 + Isolation Forest |
| Supervised Transformer + IF Fusion | 本文优化后的融合模型 |

消融实验主要验证：

- 去掉 Transformer 后，Isolation Forest 难以处理复杂特征关系；
- 去掉 Isolation Forest 后，模型缺少基于特征空间隔离程度的异常评分；
- 使用 AutoEncoder 替代 Transformer 后，长距离依赖和复杂特征交互建模能力下降；
- 加入监督分类目标和分数融合后，模型性能明显提升。

## 5. 阈值选择策略

异常检测不能随便固定阈值，否则很容易出现误报或漏报。

本文采用验证集自动选择阈值：

1. 训练集划分为开发集和验证集；
2. 模型在开发集上训练；
3. 在验证集上计算异常分数；
4. 绘制 PR 曲线；
5. 选择 F1 分数最高的阈值；
6. 使用该阈值在测试集上评估最终结果。

这样可以避免直接在测试集上调阈值造成数据泄漏。

## 6. 调优策略

### 6.1 数据层面调优

跌倒检测数据：

- 按 `(sequence, tag_id)` 分组，避免不同传感器数据混合；
- 将窗口标签从“多数投票”改为“只要出现 falling 即异常”，更符合跌倒检测任务；
- 使用类别权重缓解异常样本不足问题。

网络入侵数据：

- 删除高度相关特征，降低冗余；
- 使用 Random Forest 做特征选择；
- 使用 `top7` 模式降低输入维度，提高训练速度；
- 使用正常样本训练 Isolation Forest，避免异常样本污染异常评分器。

### 6.2 模型层面调优

Transformer 参数：

```text
hidden_dim = 128
n_heads = 4
n_layers = 2
dim_feedforward = 128
dropout = 0.1
learning_rate = 1e-3
```

快速实验模式：

```text
hidden_dim = 64
n_layers = 1
epochs = 4
iforest_estimators = 80
```

完整实验可使用更多 epoch 和更大的模型。

### 6.3 损失函数调优

为了同时保持特征稳定性和分类能力，模型使用联合损失：

```text
L = MSE_reconstruction + 0.5 * BCE_classification
```

其中 BCE 分类损失使用类别权重：

```text
pos_weight = normal_count / anomaly_count
```

这样可以减少少数类异常样本被模型忽略的问题。

### 6.4 分数融合调优

实验中发现，单独使用 Isolation Forest 或单独使用 Transformer 都不够稳定。因此加入融合分数：

```text
S_final = 0.65 * S_transformer + 0.35 * S_iforest
```

该设置让模型主要依赖 Transformer 的监督判别能力，同时利用 Isolation Forest 补充异常空间评分。

## 7. 实验结果概述

优化后的监督融合模型在两个数据集上均取得较好效果。

| 数据集 | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| CIC-IDS2017 DDoS | 0.9979 | 0.9988 | 0.9976 | 0.9982 | 0.9992 |
| UCI Localization Fall | 0.9193 | 0.6069 | 0.7447 | 0.6688 | 0.9259 |

结果说明：

- 在网络入侵检测任务中，模型对 DDoS 攻击具有很强识别能力；
- 在跌倒检测任务中，模型相比传统无监督方法明显提升；
- 消融实验说明监督 Transformer 表征学习和 Isolation Forest 分数融合是性能提升的关键。

## 8. 方法总结

本文模型的核心优势在于：

- Transformer 能够学习复杂特征交互；
- 监督分类目标增强异常与正常样本的可分性；
- Isolation Forest 提供基于特征空间隔离程度的异常评分；
- 分数融合提高模型稳定性；
- 验证集阈值选择避免测试集泄漏。

需要注意的是，本文最终优化模型属于**监督增强异常检测模型**，不是完全无监督模型。因此在论文中应避免写成“无监督异常检测方法”，更准确的表述是：

> 本文提出一种监督增强的 Transformer-Isolation Forest 融合异常检测模型，通过 Transformer 学习判别式高层特征，并结合 Isolation Forest 的异常评分能力，实现对网络入侵和跌倒行为的有效检测。
