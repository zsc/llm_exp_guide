# 第八章：评估与基准测试

构建全面、可靠的评估体系是 LLM 后训练成功的关键。本章深入探讨如何设计自动评估指标、组织人工评估、实施在线实验，以及构建高质量的基准测试集。我们将重点关注评估的可靠性、效率和防止数据泄露等实际挑战。

## 8.1 自动评估指标设计

### 8.1.1 评估指标分类体系

LLM 的自动评估指标可分为四个层次：

```
┌─────────────────────────────────────────────────┐
│                  评估指标体系                      │
├─────────────────────────────────────────────────┤
│                                                   │
│  1. 表面指标 (Surface Metrics)                    │
│     ├── BLEU, ROUGE, METEOR                      │
│     └── 困惑度 (Perplexity)                       │
│                                                   │
│  2. 语义指标 (Semantic Metrics)                   │
│     ├── BERTScore, BLEURT                        │
│     └── 语义相似度 (Cosine Similarity)            │
│                                                   │
│  3. 任务指标 (Task-specific Metrics)              │
│     ├── 准确率、召回率、F1                         │
│     └── 领域特定指标                              │
│                                                   │
│  4. 模型评估 (Model-based Evaluation)             │
│     ├── GPT-4 评分                               │
│     └── 奖励模型评分                              │
│                                                   │
└─────────────────────────────────────────────────┘
```

### 8.1.2 组合指标设计

单一指标往往无法全面反映模型性能，需要设计组合指标：

$$\text{Score}_{\text{composite}} = \sum_{i=1}^{n} w_i \cdot \text{normalize}(m_i)$$

其中 $w_i$ 是权重，$m_i$ 是第 $i$ 个指标，normalize 函数将不同量纲的指标归一化。

**权重设计原则**：
1. **业务导向**：根据实际应用场景调整权重
2. **动态调整**：随模型能力提升调整权重分配
3. **敏感性分析**：验证权重变化对最终排序的影响

### 8.1.3 指标可靠性验证

评估指标本身需要验证其可靠性：

```python
# 伪代码：指标可靠性验证框架
def validate_metric(metric, test_cases):
    # 1. 一致性检验
    consistency_score = check_consistency(metric, test_cases)
    
    # 2. 区分度检验
    discrimination_score = check_discrimination(metric, test_cases)
    
    # 3. 与人工评分相关性
    human_correlation = compute_correlation(metric, human_scores)
    
    # 4. 稳定性检验（多次运行）
    stability_score = check_stability(metric, test_cases)
    
    return {
        'consistency': consistency_score,
        'discrimination': discrimination_score,
        'human_correlation': human_correlation,
        'stability': stability_score
    }
```

### 8.1.4 LLM-as-Judge 方法

使用强大的 LLM 作为评判者已成为主流方法：

**优势**：
- 能理解复杂的语义和上下文
- 可以提供详细的评分理由
- 易于扩展到新任务

**挑战与解决方案**：
1. **位置偏差**：随机打乱候选答案顺序
2. **长度偏差**：归一化或使用配对比较
3. **自我偏好**：使用多个不同模型交叉验证
4. **提示敏感性**：测试多个提示模板

**评分提示模板示例**：
```
请评估以下回答的质量，考虑以下维度：
1. 准确性（0-10分）：信息是否正确
2. 相关性（0-10分）：是否回答了问题
3. 完整性（0-10分）：是否涵盖关键点
4. 清晰度（0-10分）：表达是否清楚

问题：{question}
回答：{answer}

请提供每个维度的分数和简要理由。
```

## 8.2 人工评估的组织与偏差控制

### 8.2.1 评估任务设计

人工评估设计的核心要素：

```
┌──────────────────────────────────────┐
│         人工评估任务设计               │
├──────────────────────────────────────┤
│                                      │
│  1. 评估类型选择                      │
│     ├── 绝对评分 (1-5分)             │
│     ├── 配对比较 (A vs B)            │
│     └── 排序任务 (Ranking)           │
│                                      │
│  2. 评估维度定义                      │
│     ├── 单维度 vs 多维度              │
│     └── 整体 vs 细粒度                │
│                                      │
│  3. 标注界面设计                      │
│     ├── 清晰的指令                   │
│     ├── 示例说明                     │
│     └── 实时反馈机制                 │
│                                      │
└──────────────────────────────────────┘
```

### 8.2.2 标注者管理

**标注者选择**：
- **专家标注**：领域专家，质量高但成本高
- **众包标注**：规模大但需要严格质控
- **内部团队**：可控性强，适合敏感数据

**质量控制机制**：
1. **黄金标准题**：插入已知答案的测试题
2. **重复标注**：计算标注者间一致性（Kappa系数）
3. **动态监控**：实时追踪标注质量趋势
4. **培训与反馈**：定期培训和个性化反馈

### 8.2.3 偏差识别与缓解

常见的人工评估偏差：

| 偏差类型 | 描述 | 缓解策略 |
|---------|------|----------|
| 顺序效应 | 第一个或最后一个选项更容易被选中 | 随机化呈现顺序 |
| 锚定效应 | 受第一个评分影响 | 使用配对比较代替绝对评分 |
| 疲劳效应 | 长时间标注导致质量下降 | 限制单次任务量，强制休息 |
| 个人偏好 | 标注者的主观偏好 | 多人标注取平均，异常值检测 |
| 样本选择偏差 | 评估集不代表真实分布 | 分层采样，定期更新评估集 |

### 8.2.4 标注一致性分析

**Fleiss' Kappa 计算**：

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

其中 $P_o$ 是观察到的一致性，$P_e$ 是随机一致性。

**一致性阈值参考**：
- κ > 0.8：几乎完美一致
- 0.6 < κ ≤ 0.8：实质性一致
- 0.4 < κ ≤ 0.6：中等一致
- κ ≤ 0.4：需要改进标注流程

## 8.3 A/B 测试与在线实验

### 8.3.1 实验设计原则

```
   用户流量分配
   ┌─────────────────────────────────┐
   │          总流量                  │
   └────────────┬────────────────────┘
                │
      ┌─────────┴─────────┐
      │                   │
  ┌───▼───┐         ┌────▼────┐
  │控制组  │         │ 实验组   │
  │ 50%   │         │  50%    │
  └───┬───┘         └────┬────┘
      │                   │
      │                   │
  ┌───▼───┐         ┌────▼────┐
  │模型 A  │         │ 模型 B  │
  └───────┘         └─────────┘
```

**关键考虑因素**：
1. **样本量计算**：基于统计功效分析
2. **分流策略**：用户级 vs 会话级
3. **实验时长**：考虑周期性效应
4. **护栏指标**：防止负面影响

### 8.3.2 统计显著性检验

**样本量估算公式**：

$$n = \frac{2(Z_{\alpha/2} + Z_{\beta})^2 \sigma^2}{\delta^2}$$

其中：
- $Z_{\alpha/2}$：显著性水平对应的 Z 值
- $Z_{\beta}$：统计功效对应的 Z 值
- $\sigma$：标准差
- $\delta$：最小可检测效应

### 8.3.3 多臂老虎机优化

动态调整流量分配以最大化收益：

**Thompson Sampling 算法**：
```python
# 伪代码
def thompson_sampling(arms, alpha, beta):
    samples = []
    for i in range(len(arms)):
        # 从 Beta 分布采样
        sample = np.random.beta(alpha[i], beta[i])
        samples.append(sample)
    
    # 选择采样值最大的臂
    return np.argmax(samples)
```

### 8.3.4 实验监控与早停

**监控指标体系**：
1. **核心指标**：直接业务目标（如用户满意度）
2. **护栏指标**：安全边界（如延迟、错误率）
3. **诊断指标**：帮助理解变化原因
4. **领先指标**：预测长期影响

**早停决策框架**：
- 显著负面影响：立即停止
- 显著正面影响：考虑提前结束
- 无显著差异：评估继续的价值

## 8.4 基准测试集的构建原则

### 8.4.1 测试集设计原则

高质量基准测试集的特征：

```
┌────────────────────────────────────────┐
│         基准测试集设计原则                │
├────────────────────────────────────────┤
│                                        │
│  1. 代表性 (Representativeness)        │
│     └── 覆盖真实使用场景                 │
│                                        │
│  2. 多样性 (Diversity)                 │
│     └── 难度梯度、领域覆盖               │
│                                        │
│  3. 区分度 (Discrimination)            │
│     └── 能够区分不同能力水平             │
│                                        │
│  4. 稳定性 (Stability)                 │
│     └── 评分一致、不易过拟合             │
│                                        │
│  5. 可解释性 (Interpretability)        │
│     └── 错误可分析、可归因               │
│                                        │
└────────────────────────────────────────┘
```

### 8.4.2 动态基准构建

静态基准容易被过拟合，需要动态更新：

**动态生成策略**：
1. **对抗样本生成**：基于模型弱点生成
2. **难度自适应**：根据模型能力调整
3. **版本控制**：保持向后兼容性
4. **增量更新**：定期添加新样本

### 8.4.3 多维度评估矩阵

```
评估维度矩阵示例：

        知识  推理  创造  安全  效率
任务1    ✓    ✓    -    ✓    -
任务2    -    ✓    ✓    -    ✓
任务3    ✓    -    -    ✓    ✓
任务4    -    ✓    ✓    ✓    -
任务5    ✓    -    ✓    -    ✓

✓ 表示该任务测试该维度
```

### 8.4.4 标准化评估协议

**评估协议要素**：
1. **输入格式**：统一的提示模板
2. **输出解析**：标准化的答案提取
3. **评分规则**：明确的计分方法
4. **运行环境**：固定的推理参数

## 8.5 评估数据泄露的检测与预防

### 8.5.1 数据泄露的类型与危害

数据泄露会严重影响评估的可靠性：

```
数据泄露分类：

1. 直接泄露 (Direct Leakage)
   └── 测试集直接出现在训练数据中

2. 间接泄露 (Indirect Leakage)
   ├── 相似样本泄露
   └── 答案模式泄露

3. 时间泄露 (Temporal Leakage)
   └── 使用未来数据训练

4. 特征泄露 (Feature Leakage)
   └── 标签信息编码在特征中
```

### 8.5.2 泄露检测方法

**1. N-gram 重叠检测**

```python
def detect_ngram_overlap(train_data, test_data, n=13):
    """
    检测训练集和测试集的 n-gram 重叠
    n=13 是经验值，能有效检测大部分泄露
    """
    train_ngrams = extract_ngrams(train_data, n)
    test_ngrams = extract_ngrams(test_data, n)
    
    overlap = train_ngrams.intersection(test_ngrams)
    overlap_ratio = len(overlap) / len(test_ngrams)
    
    return {
        'overlap_ratio': overlap_ratio,
        'suspicious_samples': find_suspicious_samples(overlap)
    }
```

**2. 嵌入空间相似度检测**

使用句子嵌入检测语义相似的样本：

$$\text{similarity} = \frac{\mathbf{e}_{\text{train}} \cdot \mathbf{e}_{\text{test}}}{||\mathbf{e}_{\text{train}}|| \cdot ||\mathbf{e}_{\text{test}}||}$$

阈值设置：
- similarity > 0.95：高度可疑
- 0.90 < similarity ≤ 0.95：需要人工审核
- similarity ≤ 0.90：基本安全

**3. 模型行为分析**

```python
def analyze_model_behavior(model, test_set):
    """
    通过模型行为检测可能的数据泄露
    """
    metrics = {}
    
    # 1. 困惑度异常检测
    perplexity = compute_perplexity(model, test_set)
    metrics['perplexity_anomaly'] = detect_anomaly(perplexity)
    
    # 2. 置信度分布分析
    confidence = get_confidence_distribution(model, test_set)
    metrics['confidence_skew'] = compute_skewness(confidence)
    
    # 3. 记忆化检测
    memorization = test_memorization(model, test_set)
    metrics['memorization_score'] = memorization
    
    return metrics
```

### 8.5.3 预防策略

**1. 数据隔离机制**

```
┌─────────────────────────────────────┐
│          数据隔离架构                 │
├─────────────────────────────────────┤
│                                     │
│   训练环境          评估环境          │
│   ┌──────┐        ┌──────┐         │
│   │训练集 │        │测试集 │         │
│   └──┬───┘        └──┬───┘         │
│      │               │              │
│      ▼               ▼              │
│   ┌──────┐        ┌──────┐         │
│   │ 模型  │───────>│ 评估  │         │
│   └──────┘        └──────┘         │
│                                     │
│   访问控制：                         │
│   - 训练团队无法访问测试集            │
│   - 评估团队无法修改训练数据          │
│   - 审计日志记录所有访问              │
│                                     │
└─────────────────────────────────────┘
```

**2. 时间戳验证**

确保训练数据的时间早于测试数据：

```python
def validate_temporal_integrity(train_data, test_data):
    train_latest = max(sample.timestamp for sample in train_data)
    test_earliest = min(sample.timestamp for sample in test_data)
    
    if train_latest >= test_earliest:
        raise DataLeakageError("Temporal leakage detected")
```

**3. 哈希指纹追踪**

为每个数据样本生成唯一指纹：

```python
def generate_data_fingerprint(sample):
    # 规范化文本
    normalized = normalize_text(sample)
    # 生成哈希
    fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
    return fingerprint

# 构建指纹数据库
fingerprint_db = {
    'train': set(generate_fingerprint(s) for s in train_data),
    'test': set(generate_fingerprint(s) for s in test_data)
}

# 检查重叠
overlap = fingerprint_db['train'] & fingerprint_db['test']
```

### 8.5.4 持续监控机制

**监控指标仪表板**：

```
实时监控指标：
┌──────────────────────────────────────┐
│  数据泄露监控仪表板                    │
├──────────────────────────────────────┤
│                                      │
│  N-gram 重叠率：     0.3% [正常]      │
│  嵌入相似度峰值：    0.89 [正常]      │
│  困惑度异常值：      2个 [警告]       │
│  新增测试样本：      523个            │
│  最后检查时间：      2小时前          │
│                                      │
│  历史趋势：                          │
│  ┌────────────────────────┐         │
│  │     ╱╲    ╱╲           │         │
│  │    ╱  ╲  ╱  ╲          │         │
│  │   ╱    ╲╱    ╲         │         │
│  │  ╱            ╲        │         │
│  └────────────────────────┘         │
│                                      │
└──────────────────────────────────────┘
```

## 8.6 本章小结

本章系统介绍了 LLM 后训练的评估体系构建，核心要点包括：

**关键概念**：
1. **多层次评估体系**：从表面指标到模型评估的递进式评估
2. **人机结合评估**：自动指标与人工评估的优势互补
3. **在线实验方法**：A/B 测试和多臂老虎机的应用
4. **数据泄露防控**：检测和预防机制的建立

**核心公式**：
- 组合指标：$\text{Score}_{\text{composite}} = \sum_{i=1}^{n} w_i \cdot \text{normalize}(m_i)$
- Kappa 一致性：$\kappa = \frac{P_o - P_e}{1 - P_e}$
- 样本量估算：$n = \frac{2(Z_{\alpha/2} + Z_{\beta})^2 \sigma^2}{\delta^2}$

**实践要点**：
- 评估指标需要验证其自身的可靠性
- 人工评估需要严格的质量控制流程
- 基准测试集应当动态更新以防过拟合
- 数据泄露检测应贯穿整个训练周期

## 8.7 常见陷阱与错误

### ⚠️ 陷阱 1：过度依赖单一指标

**问题**：仅优化 BLEU 或困惑度，导致模型产生不自然的输出。

**解决**：
- 使用多维度评估矩阵
- 结合自动指标和人工评估
- 定期审查指标的业务相关性

### ⚠️ 陷阱 2：评估集分布偏移

**问题**：评估集不能代表实际使用场景，导致线上效果差。

**解决**：
- 定期从线上采样更新评估集
- 监控评估集与线上数据的分布差异
- 使用分层采样确保覆盖各种场景

### ⚠️ 陷阱 3：标注者偏差未控制

**问题**：标注质量不稳定，导致评估结果不可靠。

**解决**：
- 实施标注者培训和认证
- 使用多人标注和一致性检查
- 定期审计标注质量

### ⚠️ 陷阱 4：A/B 测试样本量不足

**问题**：过早得出结论，统计功效不足。

**解决**：
- 事先进行功效分析计算样本量
- 使用序贯测试方法
- 设置最小实验时长

### ⚠️ 陷阱 5：忽视隐性数据泄露

**问题**：只检查直接重复，忽略语义相似的泄露。

**解决**：
- 使用多种检测方法（n-gram、嵌入、行为）
- 建立数据谱系追踪系统
- 实施严格的数据访问控制

### 💡 调试技巧

1. **评估调试检查清单**：
   - [ ] 评估指标与业务目标对齐
   - [ ] 人工评估指南清晰无歧义
   - [ ] A/B 测试流量分配均匀
   - [ ] 数据泄露检测已运行
   - [ ] 评估结果可复现

2. **异常结果排查流程**：
   ```
   评估异常 → 检查数据质量 → 验证评估代码 
            → 分析模型输出 → 审查标注一致性
   ```

3. **性能瓶颈优化**：
   - 批量化评估请求
   - 缓存重复计算结果
   - 并行化独立评估任务

## 8.8 练习题

### 练习 8.1：设计组合评估指标
设计一个用于评估客服对话系统的组合指标，需要同时考虑准确性、响应速度和用户满意度。

**Hint**: 考虑不同指标的量纲和业务重要性。

<details>
<summary>参考答案</summary>

组合指标设计：

$$\text{Score} = 0.4 \times \text{norm}(\text{准确率}) + 0.3 \times \text{norm}(\text{满意度}) + 0.2 \times \text{norm}(\frac{1}{\text{延迟}}) + 0.1 \times \text{norm}(\text{覆盖率})$$

其中归一化函数：
$$\text{norm}(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

权重设计理由：
- 准确率（40%）：核心业务指标，直接影响问题解决率
- 满意度（30%）：用户体验的直接反映
- 响应速度（20%）：影响用户等待体验，使用倒数确保越快越好
- 覆盖率（10%）：能处理的问题类型范围

实施要点：
1. 定期根据业务反馈调整权重
2. 设置各指标的最低阈值，任一指标低于阈值则总分置零
3. 使用滑动窗口计算，避免短期波动影响
</details>

### 练习 8.2：计算标注一致性
三位标注者对 100 个样本进行二分类标注，结果如下：
- 所有人都标注为正例：30 个
- 所有人都标注为负例：40 个
- 两人正例一人负例：20 个
- 一人正例两人负例：10 个

计算 Fleiss' Kappa 值并评估一致性水平。

**Hint**: Fleiss' Kappa 需要先计算观察一致性和期望一致性。

<details>
<summary>参考答案</summary>

计算步骤：

1. 构建一致性矩阵：
   - 完全一致（3人同意）：70个样本
   - 部分一致（2人同意）：30个样本

2. 计算观察一致性 $P_o$：
   $$P_o = \frac{1}{N \times k(k-1)} \sum_{i=1}^{N} \sum_{j=1}^{c} n_{ij}(n_{ij}-1)$$
   
   其中 N=100, k=3, c=2
   
   $$P_o = \frac{70 \times 3 \times 2 + 30 \times 2 \times 1}{100 \times 3 \times 2} = \frac{480}{600} = 0.8$$

3. 计算期望一致性 $P_e$：
   - 正例总数：30×3 + 20×2 + 10×1 = 140
   - 负例总数：40×3 + 20×1 + 10×2 = 160
   - $p_{\text{正}} = 140/300 = 0.467$
   - $p_{\text{负}} = 160/300 = 0.533$
   
   $$P_e = p_{\text{正}}^2 + p_{\text{负}}^2 = 0.467^2 + 0.533^2 = 0.502$$

4. 计算 Kappa：
   $$\kappa = \frac{P_o - P_e}{1 - P_e} = \frac{0.8 - 0.502}{1 - 0.502} = \frac{0.298}{0.498} = 0.598$$

评估：κ = 0.598 表示"中等一致性"，接近"实质性一致"的边界。建议加强标注指南培训。
</details>

### 练习 8.3：A/B 测试样本量计算
计划进行一个 A/B 测试，当前转化率为 5%，希望检测出 0.5% 的提升（到 5.5%），要求显著性水平 α=0.05，统计功效 1-β=0.8。需要多少样本量？

**Hint**: 使用比例检验的样本量公式。

<details>
<summary>参考答案</summary>

使用比例差异的样本量公式：

给定参数：
- $p_1 = 0.05$（控制组转化率）
- $p_2 = 0.055$（实验组期望转化率）
- $\alpha = 0.05$，$Z_{\alpha/2} = 1.96$
- $\beta = 0.2$，$Z_{\beta} = 0.84$

计算步骤：

1. 计算平均比例：
   $$\bar{p} = \frac{p_1 + p_2}{2} = 0.0525$$

2. 计算标准差：
   $$\sigma = \sqrt{2\bar{p}(1-\bar{p})} = \sqrt{2 \times 0.0525 \times 0.9475} = 0.315$$

3. 计算样本量：
   $$n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \times \sigma^2}{(p_2 - p_1)^2}$$
   $$n = \frac{(1.96 + 0.84)^2 \times 0.315^2}{0.005^2} = \frac{7.84 \times 0.099}{0.000025} = 31,046$$

每组需要约 31,000 个样本，总共需要 62,000 个样本。

实践建议：
- 考虑使用序贯测试减少样本需求
- 可以先进行小规模试验评估实际效应大小
- 考虑使用贝叶斯方法获得更灵活的决策框架
</details>

### 练习 8.4：设计数据泄露检测系统
设计一个完整的数据泄露检测系统，包括预防、检测和响应机制。

**Hint**: 考虑技术手段和流程管理两个层面。

<details>
<summary>参考答案</summary>

数据泄露检测系统设计：

**1. 预防层**
- 数据访问控制：基于角色的权限管理
- 数据加密：测试集加密存储，仅评估时解密
- 审计日志：记录所有数据访问操作
- 自动化隔离：CI/CD 流程中自动分离训练和测试数据

**2. 检测层**
- 静态检测：
  - N-gram 重叠分析（n=5,10,15）
  - 语义相似度检测（阈值 0.9）
  - 哈希指纹匹配
- 动态检测：
  - 模型困惑度异常检测
  - 置信度分布分析
  - 记忆化测试

**3. 响应机制**
- 自动告警：检测到泄露立即通知
- 泄露评估：量化泄露程度和影响
- 数据清理：移除或替换泄露样本
- 模型回滚：必要时回滚到未污染版本

**4. 监控指标**
```python
monitoring_metrics = {
    'ngram_overlap': {'threshold': 0.01, 'action': 'warn'},
    'semantic_similarity': {'threshold': 0.9, 'action': 'alert'},
    'perplexity_drop': {'threshold': 0.3, 'action': 'investigate'},
    'exact_match_rate': {'threshold': 0.001, 'action': 'block'}
}
```

**5. 流程集成**
- 训练前检查：自动扫描训练数据
- 训练中监控：定期采样检测
- 评估前验证：确认测试集完整性
- 定期审计：月度数据泄露审计报告
</details>

### 练习 8.5：优化人工评估成本
你有 1000 个样本需要评估，预算只够 2000 次标注。如何设计标注策略以获得最可靠的结果？

**Hint**: 考虑样本选择、标注分配和质量控制的平衡。

<details>
<summary>参考答案</summary>

优化策略设计：

**1. 样本分层（300个样本）**
- 使用聚类将 1000 个样本分成 10 类
- 每类随机选择 30 个代表性样本
- 每个样本标注 3 次（900 次标注）

**2. 重点样本（200个样本）**
- 模型置信度低的 100 个样本
- 业务关键场景 100 个样本
- 每个样本标注 2 次（400 次标注）

**3. 普通采样（300个样本）**
- 剩余样本随机采样 300 个
- 每个样本标注 1 次（300 次标注）

**4. 质量控制（200个样本）**
- 50 个黄金标准题（已知答案）
- 150 个重复题（从已标注中选择）
- 分散在标注任务中（200 次标注）

**5. 动态调整（200次机动）**
- 根据初步结果调整：
  - 一致性低的增加标注
  - 发现新模式时扩大采样

总计：300×3 + 200×2 + 300×1 + 200 + 200 = 2000 次

**效果评估**：
- 核心样本覆盖率：80%（800/1000）
- 平均标注次数：2次（2000/1000）
- 质控比例：10%（200/2000）
</details>

### 练习 8.6：设计多模态评估体系
为一个视觉-语言模型设计评估体系，需要评估图像理解、文本生成和跨模态对齐能力。

**Hint**: 考虑单模态和跨模态的评估指标。

<details>
<summary>参考答案</summary>

多模态评估体系设计：

**1. 图像理解评估**
- 图像分类准确率
- 目标检测 mAP
- 图像描述 CIDEr 分数
- 视觉问答准确率

**2. 文本生成评估**
- 语言流畅度（困惑度）
- 事实准确性（知识问答）
- 逻辑一致性（推理任务）
- 多样性指标（Self-BLEU）

**3. 跨模态对齐评估**
- 图文匹配准确率（检索任务）
- 语义相似度（CLIP Score）
- 细粒度对齐（区域-词汇对应）
- 时序一致性（视频理解）

**4. 综合评估框架**
```python
multimodal_eval = {
    'image_only': {
        'classification': 0.2,
        'detection': 0.2,
        'captioning': 0.6
    },
    'text_only': {
        'fluency': 0.3,
        'accuracy': 0.4,
        'diversity': 0.3
    },
    'cross_modal': {
        'retrieval': 0.3,
        'alignment': 0.4,
        'reasoning': 0.3
    }
}

# 总分计算
total_score = (
    0.3 * weighted_avg(image_scores, image_weights) +
    0.3 * weighted_avg(text_scores, text_weights) +
    0.4 * weighted_avg(cross_scores, cross_weights)
)
```

**5. 特殊考虑**
- 模态缺失鲁棒性：测试单模态输入
- 模态冲突处理：提供矛盾信息
- 计算效率：推理速度和内存占用
- 公平性评估：不同人群、文化的表现

**6. 基准数据集**
- COCO Captions：图像描述
- VQA v2：视觉问答
- Flickr30K：图文检索
- Conceptual Captions：大规模预训练评估
</details>

### 练习 8.7：实现在线实验早停机制
设计一个 A/B 测试的早停决策系统，需要平衡统计严谨性和业务效率。

**Hint**: 考虑序贯测试和业务约束。

<details>
<summary>参考答案</summary>

早停决策系统设计：

**1. 统计早停规则**

```python
class StatisticalStopping:
    def __init__(self, alpha=0.05, beta=0.2):
        self.alpha = alpha  # 显著性水平
        self.beta = beta    # Type II 错误率
        
    def should_stop(self, data_a, data_b, t):
        # 序贯概率比检验 (SPRT)
        llr = self.log_likelihood_ratio(data_a, data_b)
        
        # 计算停止边界
        upper_bound = np.log((1-self.beta)/self.alpha)
        lower_bound = np.log(self.beta/(1-self.alpha))
        
        if llr >= upper_bound:
            return True, 'significant_positive'
        elif llr <= lower_bound:
            return True, 'significant_negative'
        elif t >= self.max_duration:
            return True, 'max_duration'
        return False, 'continue'
```

**2. 业务早停规则**

```python
class BusinessStopping:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        
    def should_stop(self, metrics):
        # 护栏指标检查
        if metrics['error_rate'] > self.thresholds['max_error']:
            return True, 'guardrail_violation'
            
        # 业务指标严重恶化
        if metrics['revenue_drop'] > self.thresholds['max_revenue_drop']:
            return True, 'business_impact'
            
        # 明显改善，可考虑提前全量
        if metrics['improvement'] > self.thresholds['clear_win']:
            return True, 'clear_winner'
            
        return False, 'continue'
```

**3. 综合决策框架**

```python
def make_stopping_decision(experiment):
    # 收集数据
    data = collect_experiment_data(experiment)
    
    # 统计检验
    stat_stop, stat_reason = statistical_check(data)
    
    # 业务检查
    biz_stop, biz_reason = business_check(data)
    
    # 最小样本量检查
    if data['sample_size'] < MIN_SAMPLE_SIZE:
        return 'continue', 'insufficient_data'
    
    # 决策逻辑
    if biz_reason == 'guardrail_violation':
        return 'stop_immediately', biz_reason
    
    if stat_stop and data['duration'] >= MIN_DURATION:
        return 'stop', stat_reason
        
    if biz_stop:
        return 'stop', biz_reason
        
    return 'continue', 'monitoring'
```

**4. 实施要点**
- 设置最小运行时间（如 7 天）避免周期效应
- 使用 Bonferroni 校正处理多重比较
- 保留少量流量继续实验以验证长期效应
- 记录所有早停决策用于事后分析
</details>

### 练习 8.8：检测和量化模型记忆化
设计实验检测 LLM 是否记忆了训练数据，并量化记忆化程度。

**Hint**: 考虑使用探针任务和统计分析。

<details>
<summary>参考答案</summary>

记忆化检测实验设计：

**1. 直接探测法**

```python
def direct_memorization_test(model, known_samples):
    """
    测试模型是否能逐字复现训练样本
    """
    memorization_scores = []
    
    for sample in known_samples:
        # 提供前缀，让模型续写
        prefix = sample[:len(sample)//2]
        generation = model.generate(prefix, max_length=len(sample))
        
        # 计算精确匹配率
        exact_match = (generation == sample)
        char_overlap = calculate_char_overlap(generation, sample)
        
        memorization_scores.append({
            'exact_match': exact_match,
            'char_overlap': char_overlap,
            'perplexity': model.perplexity(sample)
        })
    
    return aggregate_scores(memorization_scores)
```

**2. 隐私攻击测试**

```python
def membership_inference_attack(model, members, non_members):
    """
    通过置信度差异判断样本是否在训练集中
    """
    member_scores = [model.log_likelihood(s) for s in members]
    non_member_scores = [model.log_likelihood(s) for s in non_members]
    
    # 训练分类器区分两组
    classifier = train_classifier(member_scores, non_member_scores)
    auc = compute_auc(classifier)
    
    # AUC > 0.5 表示存在记忆化
    memorization_level = 2 * (auc - 0.5)  # 归一化到 [0, 1]
    
    return memorization_level
```

**3. 知识探针测试**

```python
def knowledge_probe_test(model, factual_data):
    """
    测试模型记忆的事实性知识
    """
    results = []
    
    for fact in factual_data:
        # 测试不同提示下的一致性
        prompts = generate_paraphrases(fact['question'])
        answers = [model.generate(p) for p in prompts]
        
        # 计算答案一致性
        consistency = calculate_consistency(answers)
        accuracy = check_accuracy(answers, fact['answer'])
        
        results.append({
            'consistency': consistency,
            'accuracy': accuracy,
            'confidence': model.confidence(fact['question'])
        })
    
    return analyze_memorization_pattern(results)
```

**4. 量化指标体系**

```python
def compute_memorization_metrics(model, test_suite):
    metrics = {
        # 表面记忆
        'verbatim_memorization': test_exact_reproduction(model),
        
        # 语义记忆
        'semantic_memorization': test_paraphrase_invariance(model),
        
        # 隐私泄露风险
        'privacy_risk': membership_inference_score(model),
        
        # 知识固化程度
        'knowledge_rigidity': test_fact_flexibility(model),
        
        # 分布记忆
        'distribution_memorization': test_style_mimicry(model)
    }
    
    # 综合记忆化分数
    total_score = weighted_average(metrics, weights={
        'verbatim': 0.3,
        'semantic': 0.2,
        'privacy': 0.3,
        'rigidity': 0.1,
        'distribution': 0.1
    })
    
    return metrics, total_score
```

**5. 缓解策略验证**

测试不同训练技术的去记忆化效果：
- Dropout 率对记忆化的影响
- 差分隐私训练的效果
- 数据增强的作用
- 正则化强度的影响

通过 A/B 对比实验量化各策略的效果。
</details>