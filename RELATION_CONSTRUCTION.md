# Relations 构造方式详解

## 概述
`relations` 是 VBD/FlowMatching 模型中用于建模**所有实体之间相对几何关系**的关键输入，维度为 `[N+M+TL, N+M+TL, 3]`，其中：
- N = agents 数量
- M = polylines 数量  
- TL = traffic lights 数量
- 3 = (local_x, local_y, theta_diff)

---

## 1. 构造过程 (`calculate_relations`)

### 输入
- `agents`: [N, T, >=3] - 车辆历史轨迹，取最后一帧 `agents[:, -1, :3]` → (x, y, yaw)
- `polylines`: [M, W, >=3] - 道路折线，取首点 `polylines[:, 0, :3]` → (x, y, yaw)
- `traffic_lights`: [TL, >=2] - 信号灯位置 (x, y)，补零得到 (x, y, 0)

### 步骤 1: 合并所有实体
```python
all_elements = np.concatenate([
    agents[:, -1, :3],           # [N, 3]
    polylines[:, 0, :3],         # [M, 3]
    traffic_lights_padded        # [TL, 3]
], axis=0)  # → [N+M+TL, 3]
```

### 步骤 2: 计算全局位置差
```python
pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]
# → [N+M+TL, N+M+TL, 2]
```
这是一个**成对差矩阵**：`pos_diff[i, j] = position[i] - position[j]`

### 步骤 3: 转换到局部坐标系
对于每个实体 i，将其他实体 j 的位置转换到 i 的**局部坐标系**（以 i 的朝向为 x 轴）：

```python
cos_theta = np.cos(all_elements[:, 2])[:, None]  # [N+M+TL, 1]
sin_theta = np.sin(all_elements[:, 2])[:, None]

# 旋转矩阵应用
local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
```

**物理含义**：
- `local_pos_x[i, j]`：从实体 i 看，实体 j 在前方（正值）还是后方（负值）
- `local_pos_y[i, j]`：从实体 i 看，实体 j 在左侧（正值）还是右侧（负值）

### 步骤 4: 计算朝向差
```python
theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])
# θ_i - θ_j，归一化到 [-π, π]
```

**特殊处理**：
- 交通灯没有朝向 → 涉及 traffic_light 的 `theta_diff` 置 0

### 步骤 5: 对角线处理
对角线 (i, i) 表示自身与自身的关系，设为小常数 `ε=0.01` 避免除零：
```python
diag_mask = np.eye(n, dtype=bool)
local_pos_x[diag_mask] = 0.01
local_pos_y[diag_mask] = 0.01
theta_diff[diag_mask] = 0.01
```

### 步骤 6: 无效实体遮蔽
如果实体坐标为 (0, 0)（padding 的无效实体），关系置 0：
```python
zero_mask = (all_elements[:, 0][:, None] == 0) | (all_elements[:, 0][None, :] == 0)
relations = np.where(zero_mask[..., None], 0.0, relations)
```

### 输出
```python
relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)
# → [N+M+TL, N+M+TL, 3]
```

---

## 2. Relations 的使用 (`FourierEmbedding`)

在 Encoder 中，`relations` 经过 `FourierEmbedding` 转换为高维特征：

### Fourier 编码
```python
# relations: [B, N+M+TL, N+M+TL, 3]
x = relations.unsqueeze(-1) * freqs.weight * 2π  # 频率嵌入
x = [cos(x), sin(x), relations]                   # 三角函数编码
# → [B, N+M+TL, N+M+TL, num_freq_bands*2+1]
```

### MLP 处理
对每个维度（local_x, local_y, theta）独立通过 MLP 再求和：
```python
for i in [0, 1, 2]:  # 对应 local_x, local_y, theta_diff
    x_i = mlps[i](x[:, :, :, i])  # → [B, N+M+TL, N+M+TL, 256]
encoded_relations = sum(x_0, x_1, x_2)  # → [B, N+M+TL, N+M+TL, 256]
```

### 最终输出
`encoded_relations` 维度：`[B, N+M+TL, N+M+TL, 256]`，用于 Transformer 的**相对位置编码**。

---

## 3. 在 Transformer 中的应用

### 3.1 QCMHA (Quadratic Complexity Multi-Head Attention)
在 `SelfTransformer` 的注意力机制中：
```python
# query: [B, N+M+TL, 256]
# relations: [B, N+M+TL, N+M+TL, 256]

q = query.reshape(B, N, heads, head_dim)
rel_pos = relations.reshape(B, N, N, heads, head_dim)

# 注意力分数 = Q·K^T + Q·R^T (相对位置增强)
dot_score = matmul(q, k) + matmul(q, rel_pos)
```

**关键作用**：将相对几何关系注入注意力计算，使模型能感知"谁在谁的左前方"等空间关系。

### 3.2 CrossTransformer
在解码器中，`relations` 用于 query 与 context 的交叉注意力：
```python
# key + relations 作为增强的 key
key_enhanced = key + relations[:, i]  # 为第 i 个 agent 加上其与所有实体的关系
value = key_enhanced
attention_output = cross_attention(query, key_enhanced, value)
```

---

## 4. 数据流示意图

```
输入场景
├── agents (N车辆)         → (x, y, yaw) 最后一帧
├── polylines (M道路)      → (x, y, yaw) 首点
└── traffic_lights (TL)    → (x, y, 0)

                ↓ concatenate

        all_elements [N+M+TL, 3]

                ↓ calculate_relations

        relations [N+M+TL, N+M+TL, 3]
        ├── local_x: 前后相对位置
        ├── local_y: 左右相对位置
        └── theta_diff: 朝向差

                ↓ FourierEmbedding

    encoded_relations [B, N+M+TL, N+M+TL, 256]

                ↓ Transformer

    融合到注意力机制中，增强空间感知能力
```

---

## 5. 设计优势

1. **显式几何关系**：直接编码相对位置和朝向，比纯坐标更高效
2. **旋转不变性**：使用局部坐标系，对全局旋转具有不变性
3. **统一表示**：车辆、道路、信号灯用同一套关系表示
4. **高效计算**：一次性批量计算所有成对关系
5. **可学习频率**：Fourier Embedding 的频率参数可学习，自适应编码尺度

---

## 6. 代码位置
- 构造函数：[vbd/data/data_utils.py#L123](vbd/data/data_utils.py#L123) `calculate_relations()`
- 编码模块：[vbd/model/modules.py#L354](vbd/model/modules.py#L354) `FourierEmbedding`
- 使用位置：
  - [vbd/model/modules.py#L246](vbd/model/modules.py#L246) `QCMHA.forward()` 
  - [vbd/model/modules.py#L339](vbd/model/modules.py#L339) `SelfTransformer.forward()`
  - [vbd/model/modules.py#L417](vbd/model/modules.py#L417) `CrossTransformer.forward()`

---

## 总结
`relations` 通过**相对几何关系矩阵**实现了场景中所有实体的空间感知，是模型理解"道路在车辆左侧""车辆朝向与道路平行"等复杂空间关系的核心机制。
