# 第七章：训练循环与迭代优化

本章深入探讨 LLM 后训练的端到端训练流程设计，重点关注如何构建高效的迭代优化系统。我们将从数据-标注-训练-评估的完整循环开始，逐步深入到主动学习、模型合并、超参数优化以及分布式训练的工程实践。通过本章学习，您将掌握构建可扩展、高效率训练系统的核心方法论。

## 7.1 数据-标注-训练-评估循环设计

后训练的成功很大程度上取决于能否建立高效的迭代循环。与预训练的单次大规模训练不同，后训练需要持续的数据收集、标注、训练和评估，形成一个不断改进的闭环系统。

### 7.1.1 循环架构的基本原理

数据-标注-训练-评估（DLTE）循环是后训练的核心架构模式。其基本流程如下：

```
    ┌─────────────┐
    │   数据收集   │ ← 用户反馈/合成生成
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   数据标注   │ ← 人工/模型辅助
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   模型训练   │ ← SFT/RLHF/DPO
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   模型评估   │ → 指标监控
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  部署/迭代   │
    └─────────────┘
```

**关键设计原则**：

1. **增量式更新**：每次循环不必处理全量数据，而是关注新增和高价值数据
2. **快速迭代**：缩短循环周期，加快反馈速度（目标：日级别迭代）
3. **质量门控**：每个环节设置质量检查点，防止低质量数据污染
4. **可追溯性**：数据来源、标注历史、训练配置完全可追溯

### 7.1.2 数据流管道设计

高效的数据流管道需要解决数据收集、预处理、存储和版本管理等挑战：

**数据收集策略**：

- **用户交互数据**：收集真实用户查询和反馈
- **合成数据生成**：使用更强模型生成训练数据
- **困难案例挖掘**：主动收集模型表现不佳的案例
- **领域数据爬取**：针对特定领域的数据收集

**管道架构设计**：

```
输入源 → 数据验证 → 去重清洗 → 格式标准化 → 数据池
  ↓         ↓          ↓           ↓          ↓
监控      质量报告   清洗日志    schema检查  版本控制
```

**数据版本管理**：

```python
# 数据版本配置示例
data_version = {
    "version": "v2.3.1",
    "base_dataset": "v2.3.0",
    "incremental": {
        "user_feedback": 50000,
        "synthetic": 100000,
        "hard_negatives": 20000
    },
    "filters_applied": [
        "deduplication",
        "quality_threshold_0.8",
        "safety_filter"
    ],
    "split_ratio": {
        "train": 0.9,
        "val": 0.05,
        "test": 0.05
    }
}
```

### 7.1.3 标注系统集成

标注是后训练数据质量的关键环节。现代标注系统需要平衡效率、质量和成本：

**标注模式选择**：

1. **纯人工标注**：
   - 优点：质量高，可处理复杂任务
   - 缺点：成本高，速度慢，一致性难保证
   - 适用：安全相关、高价值任务

2. **模型辅助标注**：
   - 优点：效率高，成本低
   - 缺点：可能传播模型偏见
   - 适用：大规模初筛、简单分类任务

3. **混合标注策略**：
   ```
   原始数据 → 模型预标注 → 人工审核/修正 → 质量采样检查 → 最终标注
                ↓              ↓                ↓
            置信度评分    标注者一致性    随机质检(5-10%)
   ```

**标注规范设计要点**：

- **清晰的任务定义**：避免歧义，提供充分示例
- **层次化标注**：将复杂任务分解为简单子任务
- **动态规范更新**：根据标注过程中的问题持续优化规范
- **标注者培训**：系统化培训和认证机制

### 7.1.4 训练触发机制

确定何时触发新一轮训练是循环设计的关键决策点：

**触发策略类型**：

1. **定时触发**：
   - 固定周期（如每日、每周）
   - 优点：可预测，便于资源规划
   - 缺点：可能浪费计算资源

2. **数据量触发**：
   - 累积足够新数据后触发（如10万条）
   - 优点：确保每次训练有足够增量
   - 缺点：时间不可控

3. **性能触发**：
   - 监控指标下降到阈值时触发
   - 优点：按需训练，针对性强
   - 缺点：需要可靠的在线监控

4. **混合触发策略**：
   ```python
   def should_trigger_training():
       conditions = [
           data_accumulated > min_data_threshold,
           days_since_last_training > max_wait_days,
           performance_degradation > alert_threshold,
           critical_bug_fixes_pending
       ]
       return any(conditions) or all(conditions[:2])
   ```

### 7.1.5 评估反馈路径

评估结果需要有效反馈到循环的各个环节：

**反馈机制设计**：

```
评估结果 ──┬── 数据收集：指导困难样本收集
          ├── 标注优化：调整标注规范
          ├── 训练策略：调整损失权重
          └── 模型选择：决定部署版本
```

**关键指标监控**：

1. **任务指标**：准确率、BLEU、ROUGE等
2. **质量指标**：响应相关性、事实准确性
3. **安全指标**：有害内容率、偏见程度
4. **效率指标**：推理延迟、吞吐量

**自动化决策规则**：

```python
def evaluate_and_decide(model_metrics):
    decisions = {
        "deploy": all([
            model_metrics["accuracy"] > baseline["accuracy"] * 1.02,
            model_metrics["safety_score"] > 0.95,
            model_metrics["latency_p99"] < baseline["latency_p99"] * 1.1
        ]),
        "collect_more_data": model_metrics["accuracy"] < target_accuracy,
        "adjust_training": model_metrics["loss_variance"] > threshold,
        "rollback": model_metrics["safety_score"] < 0.9
    }
    return decisions
```

**持续改进机制**：

- **A/B测试**：新模型与基线模型对比
- **渐进式发布**：逐步扩大新模型的流量比例
- **回滚机制**：性能下降时快速恢复
- **案例分析**：定期分析失败案例，优化数据收集

## 7.2 主动学习与数据选择策略

在后训练中，并非所有数据都具有同等价值。主动学习（Active Learning）帮助我们识别和优先处理最有价值的数据，从而以最小的标注成本获得最大的模型改进。

### 7.2.1 不确定性采样

不确定性采样是主动学习的核心策略，通过选择模型最不确定的样本进行标注：

**不确定性度量方法**：

1. **预测熵（Entropy）**：
   $$H(x) = -\sum_{i=1}^{K} p(y_i|x) \log p(y_i|x)$$
   
   其中 $K$ 是类别数，$p(y_i|x)$ 是模型对类别 $i$ 的预测概率

2. **最小置信度（Least Confidence）**：
   $$LC(x) = 1 - \max_i p(y_i|x)$$

3. **边际采样（Margin Sampling）**：
   $$MS(x) = p(y_1|x) - p(y_2|x)$$
   
   其中 $y_1, y_2$ 是概率最高的两个类别

**LLM 特定的不确定性估计**：

```python
def compute_llm_uncertainty(model, prompt, num_samples=10):
    """通过多次采样估计 LLM 的不确定性"""
    responses = []
    log_probs = []
    
    for _ in range(num_samples):
        response, lp = model.generate(
            prompt, 
            temperature=0.7,
            return_log_probs=True
        )
        responses.append(response)
        log_probs.append(lp)
    
    # 计算响应多样性
    unique_responses = len(set(responses))
    diversity_score = unique_responses / num_samples
    
    # 计算平均对数概率方差
    avg_log_prob = np.mean([np.mean(lp) for lp in log_probs])
    log_prob_variance = np.var([np.mean(lp) for lp in log_probs])
    
    uncertainty = {
        "diversity": diversity_score,
        "avg_confidence": np.exp(avg_log_prob),
        "confidence_variance": log_prob_variance
    }
    
    return uncertainty
```

**实践技巧**：

- **温度调节**：使用不同温度多次采样，高不确定性样本的输出差异更大
- **注意力分析**：分析注意力权重的分散程度，识别模型"犹豫"的位置
- **早期层特征**：利用中间层表示的变化评估不确定性

### 7.2.2 多样性选择

仅选择不确定样本可能导致数据冗余。多样性选择确保覆盖不同的数据分布：

**多样性策略**：

1. **聚类采样**：
   ```python
   def diversity_sampling(embeddings, n_samples, n_clusters=100):
       # 先聚类
       kmeans = KMeans(n_clusters=n_clusters)
       cluster_labels = kmeans.fit_predict(embeddings)
       
       selected_indices = []
       samples_per_cluster = n_samples // n_clusters
       
       for cluster_id in range(n_clusters):
           cluster_indices = np.where(cluster_labels == cluster_id)[0]
           # 从每个簇中选择最接近中心的样本
           cluster_center = kmeans.cluster_centers_[cluster_id]
           distances = np.linalg.norm(
               embeddings[cluster_indices] - cluster_center, 
               axis=1
           )
           selected = cluster_indices[np.argsort(distances)[:samples_per_cluster]]
           selected_indices.extend(selected)
       
       return selected_indices
   ```

2. **最大边际相关性（MMR）**：
   $$MMR = \arg\max_{d_i \in R \setminus S} [\lambda \cdot Sim_1(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} Sim_2(d_i, d_j)]$$
   
   平衡与查询的相关性和与已选样本的差异性

3. **子模函数优化**：
   利用子模函数的递减边际效用特性，贪心选择提供最大信息增益的样本

**实现考虑**：

- **嵌入空间选择**：使用任务相关的嵌入（如最后一层 vs 中间层）
- **增量更新**：维护已选样本的摘要统计，避免重复计算
- **批量选择**：一次选择多个样本，考虑样本间的相互影响

### 7.2.3 困难样本挖掘

困难样本（Hard Examples）是模型容易出错的案例，对改进模型性能至关重要：

**困难样本识别方法**：

1. **损失排序**：
   ```python
   def find_hard_examples(model, dataset, percentile=95):
       losses = []
       for batch in dataset:
           with torch.no_grad():
               loss = model.compute_loss(batch)
               losses.extend(loss.cpu().numpy())
       
       threshold = np.percentile(losses, percentile)
       hard_indices = np.where(losses > threshold)[0]
       return hard_indices
   ```

2. **对抗样本生成**：
   ```python
   def generate_adversarial_prompts(model, base_prompt, epsilon=0.1):
       """生成对抗性提示"""
       embedding = model.encode(base_prompt)
       embedding.requires_grad = True
       
       output = model(embedding)
       loss = compute_target_loss(output)
       loss.backward()
       
       # 对嵌入添加扰动
       perturbation = epsilon * embedding.grad.sign()
       adversarial_embedding = embedding + perturbation
       
       # 解码回文本（近似）
       adversarial_prompt = model.decode(adversarial_embedding)
       return adversarial_prompt
   ```

3. **模型分歧挖掘**：
   比较不同模型或同一模型不同版本的预测差异

**困难样本的类型**：

- **边界案例**：接近决策边界的样本
- **长尾样本**：罕见但重要的案例
- **组合复杂性**：需要多步推理的样本
- **领域偏移**：与训练分布差异较大的样本

### 7.2.4 课程学习设计

课程学习（Curriculum Learning）通过控制样本的学习顺序来提高训练效率：

**课程设计策略**：

1. **难度递增课程**：
   ```python
   class CurriculumScheduler:
       def __init__(self, dataset, difficulty_scores):
           self.dataset = dataset
           self.difficulty_scores = difficulty_scores
           self.current_difficulty = 0.0
       
       def get_batch(self, epoch, max_epoch):
           # 线性增加难度阈值
           difficulty_threshold = (epoch / max_epoch) * 1.0
           
           # 选择难度低于阈值的样本
           valid_indices = np.where(
               self.difficulty_scores <= difficulty_threshold
           )[0]
           
           # 加权采样，优先选择接近阈值的样本
           weights = 1.0 - np.abs(
               self.difficulty_scores[valid_indices] - difficulty_threshold * 0.8
           )
           weights = weights / weights.sum()
           
           batch_indices = np.random.choice(
               valid_indices, 
               size=batch_size, 
               p=weights
           )
           
           return self.dataset[batch_indices]
   ```

2. **自适应课程**：
   根据模型当前性能动态调整课程：
   
   ```python
   def adaptive_curriculum(model, dataset, window_size=1000):
       performance_history = deque(maxlen=window_size)
       
       for batch in dataset:
           loss = model.train_step(batch)
           performance_history.append(loss)
           
           # 如果性能停滞，增加难度
           if len(performance_history) == window_size:
               recent_improvement = abs(
                   np.mean(list(performance_history)[:500]) - 
                   np.mean(list(performance_history)[500:])
               )
               
               if recent_improvement < threshold:
                   # 引入更难的样本
                   increase_difficulty_level()
   ```

3. **反课程学习**：
   某些任务中，先学习困难样本可能更有效

**课程学习的关键考虑**：

- **难度评估**：准确评估样本难度是关键
- **过渡平滑**：避免难度跳跃过大
- **遗忘问题**：定期回顾简单样本，防止灾难性遗忘

### 7.2.5 数据价值评估

量化每个数据点对模型改进的贡献，优化数据投资回报：

**数据价值度量方法**：

1. **留一法影响函数（Leave-One-Out Influence）**：
   $$\mathcal{I}(z) = \nabla_\theta \mathcal{L}(z)^T H^{-1} \nabla_\theta \mathcal{L}(z_{test})$$
   
   其中 $H$ 是 Hessian 矩阵，$z$ 是训练样本，$z_{test}$ 是测试样本

2. **数据 Shapley 值**：
   ```python
   def compute_data_shapley(model, dataset, test_set, n_iterations=100):
       n_samples = len(dataset)
       shapley_values = np.zeros(n_samples)
       
       for _ in range(n_iterations):
           # 随机排列
           perm = np.random.permutation(n_samples)
           
           prev_score = evaluate(model, test_set)
           
           for i, idx in enumerate(perm):
               # 增量训练
               model.update(dataset[idx])
               new_score = evaluate(model, test_set)
               
               # 边际贡献
               shapley_values[idx] += (new_score - prev_score)
               prev_score = new_score
       
       return shapley_values / n_iterations
   ```

3. **梯度相似性**：
   衡量训练样本梯度与验证集梯度的一致性

**数据价值的应用**：

- **数据购买决策**：确定哪些数据值得购买/标注
- **数据清理优先级**：优先清理高价值数据
- **样本权重调整**：根据价值调整训练权重
- **数据归因**：追踪模型能力来源

**实践建议**：

1. **组合策略**：结合不确定性、多样性和困难度
2. **动态调整**：随训练进展调整选择策略
3. **成本考虑**：平衡数据价值和获取成本
4. **在线更新**：实时更新数据价值估计

## 7.3 模型合并与集成学习

模型合并和集成学习技术允许我们组合多个模型的优势，在不增加推理成本的情况下提升性能。这在后训练中特别有价值，因为我们经常需要整合不同任务或领域的专门化模型。

### 7.3.1 参数空间合并技术

直接在参数空间合并模型是最直接的方法，但需要仔细处理以保持模型质量：

**基础合并方法**：

1. **线性插值（LERP）**：
   $$\theta_{merged} = \alpha \cdot \theta_A + (1-\alpha) \cdot \theta_B$$
   
   其中 $\alpha \in [0,1]$ 是插值系数

2. **球面线性插值（SLERP）**：
   ```python
   def slerp(theta_A, theta_B, alpha):
       """球面线性插值，保持参数向量的范数"""
       # 归一化参数向量
       theta_A_norm = theta_A / np.linalg.norm(theta_A)
       theta_B_norm = theta_B / np.linalg.norm(theta_B)
       
       # 计算夹角
       dot_product = np.dot(theta_A_norm, theta_B_norm)
       omega = np.arccos(np.clip(dot_product, -1, 1))
       
       # 球面插值
       if np.abs(omega) < 1e-10:
           return alpha * theta_A + (1 - alpha) * theta_B
       
       sin_omega = np.sin(omega)
       theta_merged = (np.sin((1 - alpha) * omega) / sin_omega) * theta_A + \
                     (np.sin(alpha * omega) / sin_omega) * theta_B
       
       return theta_merged
   ```

3. **加权平均**：
   $$\theta_{merged} = \frac{\sum_{i=1}^{N} w_i \cdot \theta_i}{\sum_{i=1}^{N} w_i}$$

**高级合并技术**：

1. **Fisher 加权合并**：
   使用 Fisher 信息矩阵作为重要性权重：
   
   ```python
   def fisher_weighted_merge(models, dataset):
       """基于 Fisher 信息的参数合并"""
       fisher_matrices = []
       
       for model in models:
           # 计算 Fisher 信息矩阵（对角近似）
           fisher = compute_fisher_diagonal(model, dataset)
           fisher_matrices.append(fisher)
       
       # 归一化 Fisher 权重
       total_fisher = sum(fisher_matrices)
       weights = [f / total_fisher for f in fisher_matrices]
       
       # 加权合并
       merged_params = {}
       for param_name in models[0].state_dict():
           merged_params[param_name] = sum(
               w[param_name] * m.state_dict()[param_name] 
               for w, m in zip(weights, models)
           )
       
       return merged_params
   ```

2. **RegMean（正则化均值）**：
   通过添加正则化项防止合并后的参数偏离过远：
   
   $$\theta_{merged} = \arg\min_\theta \sum_{i=1}^{N} \|\theta - \theta_i\|^2 + \lambda R(\theta)$$

### 7.3.2 任务向量与模型算术

任务向量（Task Vectors）将模型能力表示为参数空间中的向量，实现模型能力的算术运算：

**任务向量定义**：
$$\tau = \theta_{finetuned} - \theta_{pretrained}$$

**模型算术运算**：

1. **能力添加**：
   ```python
   def add_capabilities(base_model, task_vectors, scaling_factors=None):
       """向基础模型添加多个任务能力"""
       if scaling_factors is None:
           scaling_factors = [1.0] * len(task_vectors)
       
       merged_params = base_model.state_dict().copy()
       
       for param_name in merged_params:
           # 累加任务向量
           delta = sum(
               scale * tv[param_name] 
               for scale, tv in zip(scaling_factors, task_vectors)
           )
           merged_params[param_name] += delta
       
       return merged_params
   ```

2. **能力删除**：
   ```python
   def remove_capability(model, task_vector, scale=1.0):
       """从模型中移除特定能力"""
       updated_params = model.state_dict().copy()
       
       for param_name in updated_params:
           updated_params[param_name] -= scale * task_vector[param_name]
       
       return updated_params
   ```

3. **能力组合的冲突检测**：
   ```python
   def detect_task_conflicts(task_vectors):
       """检测任务向量之间的冲突"""
       conflicts = {}
       
       for i, tv1 in enumerate(task_vectors):
           for j, tv2 in enumerate(task_vectors[i+1:], i+1):
               # 计算余弦相似度
               similarity = cosine_similarity(
                   flatten(tv1), flatten(tv2)
               )
               
               # 负相关表示潜在冲突
               if similarity < -0.5:
                   conflicts[(i, j)] = similarity
       
       return conflicts
   ```

**实践考虑**：

- **缩放因子优化**：通过验证集搜索最优缩放系数
- **正交化**：确保任务向量相互正交，减少干扰
- **稀疏化**：只保留重要的参数变化

### 7.3.3 层级合并策略

不同层可能需要不同的合并策略，基于层的功能和重要性：

**层级重要性分析**：

```python
def compute_layer_importance(model, dataset, layer_names):
    """计算各层对任务的重要性"""
    importances = {}
    
    for layer_name in layer_names:
        # 临时移除层
        original_layer = getattr(model, layer_name)
        setattr(model, layer_name, nn.Identity())
        
        # 评估性能下降
        degraded_performance = evaluate(model, dataset)
        
        # 恢复层
        setattr(model, layer_name, original_layer)
        baseline_performance = evaluate(model, dataset)
        
        importance = baseline_performance - degraded_performance
        importances[layer_name] = importance
    
    return importances
```

**分层合并策略**：

1. **底层共享，高层特化**：
   ```python
   def hierarchical_merge(models, split_layer=12):
       """低层参数平均，高层保持特化"""
       merged_model = models[0].copy()
       
       for layer_idx in range(len(merged_model.layers)):
           if layer_idx < split_layer:
               # 低层：平均合并
               merged_model.layers[layer_idx] = average_layers(
                   [m.layers[layer_idx] for m in models]
               )
           else:
               # 高层：选择最佳或保持独立
               best_model_idx = select_best_for_layer(
                   models, layer_idx
               )
               merged_model.layers[layer_idx] = \
                   models[best_model_idx].layers[layer_idx]
       
       return merged_model
   ```

2. **注意力头选择性合并**：
   ```python
   def merge_attention_heads(models, importance_threshold=0.1):
       """根据重要性选择性合并注意力头"""
       merged_attention = []
       
       n_heads = models[0].n_heads
       for head_idx in range(n_heads):
           head_candidates = [
               m.attention.heads[head_idx] for m in models
           ]
           
           # 评估每个头的重要性
           importances = [
               evaluate_head_importance(h) for h in head_candidates
           ]
           
           if max(importances) > importance_threshold:
               # 选择最重要的头
               best_idx = np.argmax(importances)
               merged_attention.append(head_candidates[best_idx])
           else:
               # 低重要性头可以平均
               merged_attention.append(average_heads(head_candidates))
       
       return merged_attention
   ```

### 7.3.4 集成学习方法

当无法直接合并参数时，集成多个模型的预测：

**集成策略**：

1. **加权投票**：
   ```python
   class WeightedEnsemble:
       def __init__(self, models, weights=None):
           self.models = models
           self.weights = weights or [1/len(models)] * len(models)
       
       def predict(self, input_data):
           predictions = []
           for model, weight in zip(self.models, self.weights):
               pred = model.predict(input_data)
               predictions.append(weight * pred)
           
           # 加权平均
           ensemble_pred = sum(predictions)
           
           # 对于分类任务，可能需要重新归一化
           if self.task_type == 'classification':
               ensemble_pred = softmax(ensemble_pred)
           
           return ensemble_pred
   ```

2. **混合专家（MoE）风格**：
   ```python
   class MixtureOfExperts:
       def __init__(self, experts, gating_network):
           self.experts = experts
           self.gating = gating_network
       
       def forward(self, x):
           # 计算门控权重
           gate_weights = self.gating(x)  # [batch_size, n_experts]
           
           # 获取每个专家的输出
           expert_outputs = torch.stack([
               expert(x) for expert in self.experts
           ], dim=1)  # [batch_size, n_experts, output_dim]
           
           # 加权组合
           output = torch.sum(
               gate_weights.unsqueeze(-1) * expert_outputs, 
               dim=1
           )
           
           return output
   ```

3. **级联集成**：
   ```python
   def cascade_ensemble(models, input_data, confidence_threshold=0.8):
       """按置信度级联使用模型"""
       for i, model in enumerate(models):
           prediction, confidence = model.predict_with_confidence(input_data)
           
           if confidence > confidence_threshold or i == len(models) - 1:
               return prediction
       
       # 如果所有模型置信度都低，可以组合预测
       return weighted_ensemble(models, input_data)
   ```

### 7.3.5 合并冲突解决

处理模型合并中的冲突是确保质量的关键：

**冲突类型与解决策略**：

1. **参数符号冲突**：
   ```python
   def resolve_sign_conflicts(param_A, param_B, method='magnitude'):
       """解决参数符号相反的冲突"""
       sign_conflict = (param_A * param_B) < 0
       
       if method == 'magnitude':
           # 选择绝对值较大的
           merged = torch.where(
               torch.abs(param_A) > torch.abs(param_B),
               param_A, param_B
           )
       elif method == 'zero':
           # 冲突位置置零
           merged = torch.where(sign_conflict, 0, (param_A + param_B) / 2)
       elif method == 'interpolate':
           # 根据重要性插值
           importance_A = compute_param_importance(param_A)
           importance_B = compute_param_importance(param_B)
           alpha = importance_A / (importance_A + importance_B)
           merged = alpha * param_A + (1 - alpha) * param_B
       
       return merged
   ```

2. **梯度冲突检测与缓解**：
   ```python
   def gradient_surgery(gradients, threshold=0.0):
       """梯度手术：投影冲突梯度到正交空间"""
       modified_grads = []
       
       for i, g_i in enumerate(gradients):
           g_modified = g_i.clone()
           
           for j, g_j in enumerate(gradients):
               if i != j:
                   # 计算余弦相似度
                   cos_sim = F.cosine_similarity(g_i, g_j, dim=0)
                   
                   if cos_sim < threshold:  # 负相关，存在冲突
                       # 投影到正交空间
                       projection = (g_i @ g_j) / (g_j @ g_j + 1e-8) * g_j
                       g_modified = g_modified - projection
           
           modified_grads.append(g_modified)
       
       return modified_grads
   ```

**合并质量验证**：

```python
def validate_merge_quality(original_models, merged_model, test_sets):
    """验证合并后模型的质量"""
    metrics = {
        'performance_retention': [],
        'capability_preservation': [],
        'emergence_score': 0
    }
    
    for original, test_set in zip(original_models, test_sets):
        # 性能保持率
        original_score = evaluate(original, test_set)
        merged_score = evaluate(merged_model, test_set)
        retention = merged_score / original_score
        metrics['performance_retention'].append(retention)
        
        # 能力保留检查
        capability_tests = generate_capability_tests(original)
        preservation = evaluate_capabilities(merged_model, capability_tests)
        metrics['capability_preservation'].append(preservation)
    
    # 涌现能力（合并后出现的新能力）
    combined_test = combine_test_sets(test_sets)
    emergence = evaluate_emergence(merged_model, combined_test, original_models)
    metrics['emergence_score'] = emergence
    
    return metrics
```

## 7.4 超参数搜索的实用技巧

超参数优化是后训练成功的关键因素之一。与预训练相比，后训练的超参数空间更复杂，需要更精细的搜索策略。

### 7.4.1 搜索空间设计

合理的搜索空间设计可以大幅提高搜索效率：

**关键超参数及其范围**：

```python
hyperparameter_space = {
    # 学习率相关
    "learning_rate": {
        "type": "log_uniform",
        "low": 1e-6,
        "high": 1e-3,
        "description": "峰值学习率"
    },
    "warmup_ratio": {
        "type": "uniform",
        "low": 0.0,
        "high": 0.1,
        "description": "预热步数比例"
    },
    
    # 训练配置
    "batch_size": {
        "type": "choice",
        "values": [8, 16, 32, 64],
        "description": "有效批次大小"
    },
    "gradient_accumulation_steps": {
        "type": "choice",
        "values": [1, 2, 4, 8],
        "description": "梯度累积步数"
    },
    
    # 正则化
    "weight_decay": {
        "type": "log_uniform",
        "low": 1e-4,
        "high": 1e-1
    },
    "dropout": {
        "type": "uniform",
        "low": 0.0,
        "high": 0.3
    },
    
    # RLHF 特定参数
    "kl_coefficient": {
        "type": "log_uniform",
        "low": 0.001,
        "high": 0.1,
        "description": "KL 散度惩罚系数"
    },
    "clip_range": {
        "type": "uniform",
        "low": 0.1,
        "high": 0.3,
        "description": "PPO 裁剪范围"
    }
}
```

**条件依赖关系**：

```python
def conditional_hyperparameters(config):
    """处理超参数间的条件依赖"""
    # 如果使用 LoRA，添加相关参数
    if config.get("use_lora", False):
        config["lora_rank"] = sample_from(4, 64, log=True)
        config["lora_alpha"] = config["lora_rank"] * 2
        config["lora_dropout"] = sample_from(0.0, 0.1)
    
    # 学习率与批次大小的关系
    if config["batch_size"] > 32:
        config["learning_rate"] *= np.sqrt(config["batch_size"] / 32)
    
    # 梯度累积与实际批次大小
    config["effective_batch_size"] = (
        config["batch_size"] * config["gradient_accumulation_steps"]
    )
    
    return config
```

**搜索空间约简技巧**：

1. **分阶段搜索**：
   ```python
   def staged_search():
       # 第一阶段：粗粒度搜索
       stage1_space = {
           "learning_rate": [1e-5, 1e-4, 1e-3],
           "batch_size": [16, 32, 64]
       }
       best_config_stage1 = grid_search(stage1_space)
       
       # 第二阶段：细粒度搜索
       stage2_space = create_refined_space(best_config_stage1)
       best_config = bayesian_optimization(stage2_space)
       
       return best_config
   ```

2. **重要性采样**：
   基于参数重要性分配搜索预算

### 7.4.2 贝叶斯优化

贝叶斯优化通过建立代理模型高效探索超参数空间：

**高斯过程实现**：

```python
class BayesianOptimizer:
    def __init__(self, space, acquisition_func="ei"):
        self.space = space
        self.gp = GaussianProcessRegressor()
        self.acquisition_func = acquisition_func
        self.observations = []
    
    def suggest_next(self):
        if len(self.observations) < 5:
            # 初始随机探索
            return self.random_sample()
        
        # 训练高斯过程
        X = [obs["config"] for obs in self.observations]
        y = [obs["score"] for obs in self.observations]
        self.gp.fit(X, y)
        
        # 优化获取函数
        next_point = self.optimize_acquisition()
        return next_point
    
    def optimize_acquisition(self):
        """优化获取函数找到下一个采样点"""
        if self.acquisition_func == "ei":
            return self.expected_improvement()
        elif self.acquisition_func == "ucb":
            return self.upper_confidence_bound()
        elif self.acquisition_func == "pi":
            return self.probability_improvement()
    
    def expected_improvement(self):
        """期望改进获取函数"""
        best_y = max([obs["score"] for obs in self.observations])
        
        def ei(x):
            mean, std = self.gp.predict(x, return_std=True)
            z = (mean - best_y) / (std + 1e-9)
            ei_value = (mean - best_y) * norm.cdf(z) + std * norm.pdf(z)
            return -ei_value  # 负值用于最小化
        
        # 多起点优化
        best_x = None
        best_ei = float("-inf")
        
        for _ in range(10):
            x0 = self.random_sample()
            result = minimize(ei, x0, bounds=self.space.bounds)
            if -result.fun > best_ei:
                best_ei = -result.fun
                best_x = result.x
        
        return best_x
```

**多保真度优化（Multi-fidelity）**：

```python
def successive_halving(configs, budget, eta=3):
    """Successive Halving 算法"""
    n = len(configs)
    r = budget / n  # 初始资源分配
    
    while n > 1:
        # 训练所有配置 r 个 epoch
        scores = []
        for config in configs:
            score = train_and_evaluate(config, epochs=int(r))
            scores.append(score)
        
        # 保留前 1/eta
        k = max(1, n // eta)
        top_k_indices = np.argsort(scores)[-k:]
        configs = [configs[i] for i in top_k_indices]
        
        # 增加资源
        n = len(configs)
        r = r * eta
    
    return configs[0]
```

### 7.4.3 群体训练策略

同时训练多个模型变体，利用群体智慧：

**Population Based Training (PBT)**：

```python
class PopulationBasedTraining:
    def __init__(self, population_size=16):
        self.population_size = population_size
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            member = {
                "config": self.sample_hyperparameters(),
                "model": None,
                "score": 0,
                "age": 0
            }
            population.append(member)
        return population
    
    def evolve(self, steps=100):
        """进化过程"""
        for step in range(steps):
            # 并行训练一段时间
            self.train_population(epochs=5)
            
            # 评估性能
            self.evaluate_population()
            
            # 执行进化操作
            if step % 10 == 0:
                self.exploit_and_explore()
    
    def exploit_and_explore(self):
        """利用和探索"""
        # 排序种群
        self.population.sort(key=lambda x: x["score"], reverse=True)
        
        # 底部 25% 被顶部 25% 替换
        bottom_quartile = self.population_size // 4
        top_quartile = self.population_size // 4
        
        for i in range(bottom_quartile):
            # 复制高性能成员
            source_idx = i % top_quartile
            self.population[-(i+1)] = self.copy_member(
                self.population[source_idx]
            )
            
            # 扰动超参数（探索）
            self.perturb_hyperparameters(self.population[-(i+1)])
    
    def perturb_hyperparameters(self, member):
        """扰动超参数进行探索"""
        config = member["config"]
        for param, value in config.items():
            if random.random() < 0.2:  # 20% 概率扰动
                if isinstance(value, float):
                    # 乘性扰动
                    factor = random.choice([0.8, 1.2])
                    config[param] = value * factor
                elif isinstance(value, int):
                    # 加性扰动
                    delta = random.choice([-1, 1])
                    config[param] = max(1, value + delta)
```

### 7.4.4 早停与预算分配

智能分配计算资源，避免在差配置上浪费时间：

**自适应早停策略**：

```python
class AdaptiveEarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('-inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, current_score, epoch):
        # 相对改进
        relative_improvement = (
            (current_score - self.best_score) / (abs(self.best_score) + 1e-10)
        )
        
        if relative_improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        # 动态调整 patience
        if epoch < 10:
            adjusted_patience = self.patience * 2  # 早期更宽容
        elif epoch > 50:
            adjusted_patience = self.patience // 2  # 后期更严格
        else:
            adjusted_patience = self.patience
        
        if self.counter >= adjusted_patience:
            self.early_stop = True
            
        return self.early_stop
```

**预算分配算法**：

```python
def hyperband(max_iter=81, eta=3):
    """Hyperband 算法"""
    logeta = lambda x: np.log(x) / np.log(eta)
    s_max = int(logeta(max_iter))
    B = (s_max + 1) * max_iter
    
    results = []
    
    for s in reversed(range(s_max + 1)):
        n = int(np.ceil(B / max_iter / (s + 1) * eta ** s))
        r = max_iter * eta ** (-s)
        
        # Successive halving
        configs = [sample_configuration() for _ in range(n)]
        
        for i in range(s + 1):
            n_configs = n * eta ** (-i)
            n_iterations = r * eta ** i
            
            # 训练和评估
            scores = []
            for config in configs[:int(n_configs)]:
                score = train_and_evaluate(
                    config, 
                    iterations=int(n_iterations)
                )
                scores.append((score, config))
            
            # 选择最佳配置继续
            scores.sort(reverse=True)
            configs = [config for _, config in scores[:int(n_configs / eta)]]
        
        results.extend(scores[:1])  # 保存最佳结果
    
    return max(results, key=lambda x: x[0])
```

### 7.4.5 超参数迁移学习

利用历史任务的超参数知识加速新任务的搜索：

**元学习超参数**：

```python
class HyperparameterMetaLearner:
    def __init__(self):
        self.task_embeddings = {}
        self.hyperparameter_history = []
        self.meta_model = self.build_meta_model()
    
    def build_meta_model(self):
        """构建元学习模型"""
        return RandomForestRegressor(n_estimators=100)
    
    def learn_from_task(self, task_features, best_hyperparams, performance):
        """从任务中学习"""
        self.hyperparameter_history.append({
            "task_features": task_features,
            "hyperparams": best_hyperparams,
            "performance": performance
        })
        
        # 更新元模型
        if len(self.hyperparameter_history) > 10:
            X = [h["task_features"] + list(h["hyperparams"].values()) 
                 for h in self.hyperparameter_history]
            y = [h["performance"] for h in self.hyperparameter_history]
            self.meta_model.fit(X, y)
    
    def suggest_initial_hyperparams(self, new_task_features, n_suggestions=5):
        """为新任务建议初始超参数"""
        if len(self.hyperparameter_history) < 10:
            # 历史数据不足，返回随机配置
            return [sample_random_config() for _ in range(n_suggestions)]
        
        # 找到相似任务
        similar_tasks = self.find_similar_tasks(new_task_features)
        
        suggestions = []
        for task in similar_tasks[:n_suggestions]:
            # 基于相似任务的超参数，添加小扰动
            base_config = task["hyperparams"].copy()
            perturbed_config = self.perturb_config(base_config)
            suggestions.append(perturbed_config)
        
        return suggestions
    
    def find_similar_tasks(self, task_features, k=5):
        """找到最相似的历史任务"""
        distances = []
        for hist in self.hyperparameter_history:
            dist = np.linalg.norm(
                np.array(task_features) - np.array(hist["task_features"])
            )
            distances.append((dist, hist))
        
        distances.sort(key=lambda x: x[0])
        return [hist for _, hist in distances[:k]]
```

## 7.5 分布式训练的工程优化

大规模 LLM 后训练必须依赖分布式系统。合理的分布式策略和工程优化可以显著提升训练效率和稳定性。

### 7.5.1 并行策略选择

根据模型规模和硬件资源选择合适的并行策略：

**并行策略对比**：

| 策略 | 通信开销 | 内存效率 | 适用场景 |
|------|---------|---------|----------|
| 数据并行(DP) | O(模型大小) | 低 | 小模型，大批次 |
| 张量并行(TP) | O(激活大小) | 高 | 单层参数超过GPU内存 |
| 流水线并行(PP) | O(批次大小) | 中 | 深层网络 |
| 3D并行 | 混合 | 最高 | 超大规模模型 |

**混合并行策略实现**：

```python
class HybridParallelConfig:
    def __init__(self, world_size, model_config):
        self.world_size = world_size
        self.model_config = model_config
        self.strategy = self.determine_strategy()
    
    def determine_strategy(self):
        """根据模型和硬件自动确定并行策略"""
        model_params = self.model_config.num_parameters
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        
        # 估算单GPU能否容纳模型
        bytes_per_param = 4  # FP32
        model_memory = model_params * bytes_per_param
        activation_memory = self.estimate_activation_memory()
        
        if model_memory + activation_memory < gpu_memory * 0.8:
            # 纯数据并行
            return {"dp": self.world_size, "tp": 1, "pp": 1}
        
        # 需要模型并行
        if self.model_config.num_layers > 24:
            # 深层网络，使用流水线并行
            pp_size = min(8, self.model_config.num_layers // 12)
            remaining = self.world_size // pp_size
            
            if self.model_config.hidden_size > 8192:
                # 宽模型，添加张量并行
                tp_size = min(8, remaining)
                dp_size = remaining // tp_size
            else:
                tp_size = 1
                dp_size = remaining
        else:
            # 浅层宽模型，主要使用张量并行
            tp_size = min(8, self.world_size)
            dp_size = self.world_size // tp_size
            pp_size = 1
        
        return {"dp": dp_size, "tp": tp_size, "pp": pp_size}
```

**ZeRO 优化策略**：

```python
def configure_zero_optimization(stage=2):
    """配置 ZeRO 优化"""
    zero_config = {
        "stage": stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    }
    
    if stage >= 2:
        # ZeRO-2: 优化器状态分片
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    if stage >= 3:
        # ZeRO-3: 参数分片
        zero_config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
        zero_config["param_persistence_threshold"] = 1e5
    
    return zero_config
```

### 7.5.2 通信优化

减少通信开销是分布式训练的关键优化点：

**梯度压缩**：

```python
class GradientCompressor:
    def __init__(self, compression_ratio=0.01):
        self.compression_ratio = compression_ratio
        self.residuals = {}
    
    def compress(self, tensor, name):
        """Top-k 稀疏化压缩"""
        # 添加残差
        if name in self.residuals:
            tensor = tensor + self.residuals[name]
        
        # 选择 top-k 元素
        numel = tensor.numel()
        k = max(1, int(numel * self.compression_ratio))
        
        values, indices = torch.topk(tensor.abs().view(-1), k)
        mask = torch.zeros_like(tensor.view(-1))
        mask[indices] = 1
        
        compressed = tensor.view(-1) * mask
        
        # 保存残差
        self.residuals[name] = tensor.view(-1) - compressed
        
        # 返回稀疏表示
        return indices, compressed[indices]
    
    def decompress(self, indices, values, shape):
        """解压缩"""
        tensor = torch.zeros(shape).view(-1)
        tensor[indices] = values
        return tensor.view(shape)
```

**通信调度优化**：

```python
class CommunicationScheduler:
    def __init__(self, model):
        self.model = model
        self.comm_groups = self.create_comm_groups()
    
    def schedule_allreduce(self, gradients):
        """优化 AllReduce 调度"""
        # 按大小分组
        small_grads = []
        large_grads = []
        
        for name, grad in gradients.items():
            if grad.numel() < 1e6:
                small_grads.append((name, grad))
            else:
                large_grads.append((name, grad))
        
        # 小梯度合并通信
        if small_grads:
            merged = torch.cat([g.view(-1) for _, g in small_grads])
            handle = dist.all_reduce(merged, async_op=True)
            
        # 大梯度流水线通信
        handles = []
        for name, grad in large_grads:
            handle = dist.all_reduce(grad, async_op=True)
            handles.append(handle)
        
        return handles
```

### 7.5.3 内存管理

精细的内存管理对大模型训练至关重要：

**激活检查点（Activation Checkpointing）**：

```python
class SelectiveCheckpointing:
    """选择性激活检查点"""
    
    def __init__(self, model, checkpoint_ratio=0.5):
        self.model = model
        self.checkpoint_ratio = checkpoint_ratio
        self.setup_checkpointing()
    
    def setup_checkpointing(self):
        """配置哪些层使用检查点"""
        total_layers = len(self.model.layers)
        checkpoint_layers = int(total_layers * self.checkpoint_ratio)
        
        # 选择内存占用大的层
        layer_memories = []
        for i, layer in enumerate(self.model.layers):
            memory = self.estimate_layer_memory(layer)
            layer_memories.append((i, memory))
        
        # 优先检查点内存占用大的层
        layer_memories.sort(key=lambda x: x[1], reverse=True)
        
        for i, _ in layer_memories[:checkpoint_layers]:
            self.model.layers[i].use_checkpoint = True
    
    def estimate_layer_memory(self, layer):
        """估算层的激活内存"""
        # 简化估算
        return sum(p.numel() for p in layer.parameters())
```

**内存池管理**：

```python
class MemoryPool:
    """预分配内存池减少碎片"""
    
    def __init__(self, pool_size=1024**3):  # 1GB
        self.pool = torch.cuda.ByteTensor(pool_size)
        self.allocated = 0
        self.allocations = {}
    
    def allocate(self, size, name):
        """从池中分配内存"""
        if self.allocated + size > len(self.pool):
            raise RuntimeError("Memory pool exhausted")
        
        start = self.allocated
        self.allocated += size
        self.allocations[name] = (start, size)
        
        return self.pool[start:start+size].view(size)
    
    def free(self, name):
        """释放内存（逻辑释放）"""
        if name in self.allocations:
            del self.allocations[name]
            # 实际实现中需要内存整理
```

### 7.5.4 故障恢复机制

构建健壮的故障恢复机制确保训练稳定性：

**检查点管理**：

```python
class RobustCheckpointing:
    def __init__(self, checkpoint_dir, keep_last_n=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.checkpoint_history = []
    
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """保存检查点with原子操作"""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        # 临时文件
        temp_path = f"{self.checkpoint_dir}/temp_ckpt_{epoch}.pt"
        final_path = f"{self.checkpoint_dir}/checkpoint_{epoch}.pt"
        
        # 原子写入
        torch.save(checkpoint, temp_path)
        
        # 验证检查点
        if self.verify_checkpoint(temp_path):
            os.rename(temp_path, final_path)
            self.checkpoint_history.append(final_path)
            
            # 清理旧检查点
            if len(self.checkpoint_history) > self.keep_last_n:
                old_ckpt = self.checkpoint_history.pop(0)
                os.remove(old_ckpt)
        else:
            os.remove(temp_path)
            raise RuntimeError("Checkpoint verification failed")
    
    def verify_checkpoint(self, path):
        """验证检查点完整性"""
        try:
            checkpoint = torch.load(path, map_location="cpu")
            required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
            return all(k in checkpoint for k in required_keys)
        except:
            return False
```

**弹性训练**：

```python
class ElasticTrainer:
    """支持动态节点增减的弹性训练"""
    
    def __init__(self, min_nodes=1, max_nodes=8):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.current_nodes = self.detect_available_nodes()
    
    def handle_node_failure(self, failed_node):
        """处理节点故障"""
        self.current_nodes.remove(failed_node)
        
        if len(self.current_nodes) < self.min_nodes:
            # 等待新节点或终止
            self.wait_for_nodes()
        else:
            # 重新配置并继续
            self.reconfigure_training()
    
    def reconfigure_training(self):
        """重新配置训练"""
        # 重新初始化进程组
        dist.destroy_process_group()
        dist.init_process_group(
            backend="nccl",
            world_size=len(self.current_nodes)
        )
        
        # 调整批次大小保持全局批次不变
        global_batch_size = self.config.global_batch_size
        self.config.local_batch_size = (
            global_batch_size // len(self.current_nodes)
        )
        
        # 重新分配数据
        self.redistribute_data()
```

### 7.5.5 性能调优实践

系统级优化提升训练效率：

**性能分析工具集成**：

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.profiler = None
    
    def profile_iteration(self, iteration):
        """分析单次迭代"""
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # 执行训练步骤
            yield
            
        # 分析结果
        self.analyze_profile(prof, iteration)
    
    def analyze_profile(self, prof, iteration):
        """分析性能瓶颈"""
        # GPU利用率
        cuda_time = sum([
            item.cuda_time_total for item in prof.key_averages()
        ])
        
        # 通信时间
        comm_time = sum([
            item.cuda_time_total for item in prof.key_averages()
            if "nccl" in item.key or "allreduce" in item.key
        ])
        
        # 内存峰值
        memory_peak = torch.cuda.max_memory_allocated()
        
        self.metrics["gpu_utilization"].append(cuda_time)
        self.metrics["comm_overhead"].append(comm_time / cuda_time)
        self.metrics["memory_peak"].append(memory_peak)
        
        # 识别瓶颈
        if comm_time / cuda_time > 0.3:
            print(f"Warning: High communication overhead ({comm_time/cuda_time:.2%})")
```

**自动性能调优**：

```python
class AutoTuner:
    """自动调优训练配置"""
    
    def __init__(self, model, initial_config):
        self.model = model
        self.config = initial_config
        self.performance_history = []
    
    def auto_tune(self, num_trials=10):
        """自动调优"""
        best_throughput = 0
        best_config = self.config.copy()
        
        for trial in range(num_trials):
            # 生成新配置
            trial_config = self.generate_config_variant()
            
            # 测试性能
            throughput = self.benchmark_config(trial_config)
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_config = trial_config
            
            self.performance_history.append({
                "config": trial_config,
                "throughput": throughput
            })
        
        return best_config
    
    def generate_config_variant(self):
        """生成配置变体"""
        config = self.config.copy()
        
        # 调整关键参数
        params_to_tune = [
            ("micro_batch_size", [1, 2, 4, 8]),
            ("gradient_accumulation_steps", [1, 2, 4, 8, 16]),
            ("num_workers", [0, 2, 4, 8]),
            ("pin_memory", [True, False]),
            ("prefetch_factor", [2, 4, 8])
        ]
        
        for param, values in params_to_tune:
            if random.random() < 0.3:  # 30% 概率修改
                config[param] = random.choice(values)
        
        return config
```

## 本章小结

本章系统介绍了 LLM 后训练中的训练循环与迭代优化方法：

**核心要点**：

1. **数据-标注-训练-评估循环**：
   - 建立高效的闭环系统是后训练成功的基础
   - 每个环节的质量控制和反馈机制至关重要
   - 自动化和智能化决策提升迭代效率

2. **主动学习与数据选择**：
   - 不确定性采样、多样性选择、困难样本挖掘相结合
   - 课程学习优化训练顺序，提高收敛速度
   - 数据价值评估指导资源分配

3. **模型合并与集成**：
   - 参数空间合并技术实现零成本集成
   - 任务向量支持模型能力的算术运算
   - 层级策略和冲突解决确保合并质量

4. **超参数优化**：
   - 贝叶斯优化和群体训练提高搜索效率
   - 多保真度方法优化计算资源使用
   - 超参数迁移学习加速新任务适配

5. **分布式训练优化**：
   - 混合并行策略适应不同规模模型
   - 通信和内存优化降低训练成本
   - 故障恢复机制保证训练稳定性

**关键公式回顾**：

- 不确定性度量：$H(x) = -\sum_{i=1}^{K} p(y_i|x) \log p(y_i|x)$
- 任务向量：$\tau = \theta_{finetuned} - \theta_{pretrained}$
- 期望改进：$EI(x) = (\mu(x) - f^*) \Phi(Z) + \sigma(x) \phi(Z)$

## 练习题

### 基础题

1. **循环设计理解**
   设计一个数据-标注-训练-评估循环，要求日产出 1000 个高质量标注样本，描述各环节的关键指标和质量控制点。
   
   <details>
   <summary>提示</summary>
   考虑标注效率、质量检查、自动化程度、反馈延迟等因素。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   循环设计方案：
   - 数据收集：从用户交互日志筛选（2000样本/日），预过滤规则去除明显低质量
   - 标注：混合模式，模型预标注(5000/日) → 人工审核(1500/日) → 质检(100%覆盖)
   - 训练：累积5000样本触发，增量训练2小时
   - 评估：自动评估(准确率>95%) + 人工抽检(5%)
   - 关键指标：标注一致性>0.85，模型改进>2%，端到端延迟<24小时
   </details>

2. **不确定性计算**
   给定模型对三个类别的预测概率为 [0.4, 0.35, 0.25]，计算预测熵、最小置信度和边际采样分数。
   
   <details>
   <summary>提示</summary>
   直接应用本章介绍的三个公式。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   - 预测熵：H = -0.4×log(0.4) - 0.35×log(0.35) - 0.25×log(0.25) ≈ 1.08
   - 最小置信度：LC = 1 - 0.4 = 0.6
   - 边际采样：MS = 0.4 - 0.35 = 0.05
   - 结论：高不确定性样本，适合主动学习
   </details>

3. **模型合并权重**
   两个模型在验证集上的性能分别为 0.85 和 0.90，Fisher 信息矩阵范数比为 2:3，计算最优合并权重。
   
   <details>
   <summary>提示</summary>
   结合性能和 Fisher 信息确定权重。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   综合权重计算：
   - 性能权重：0.85:0.90 = 0.486:0.514
   - Fisher 权重：2:3 = 0.4:0.6
   - 综合权重（平均）：(0.486+0.4)/2 : (0.514+0.6)/2 = 0.443:0.557
   - 归一化：0.443:0.557
   </details>

4. **并行策略选择**
   模型参数 70B，32 个 A100 GPU（80GB），批次大小 512，如何设计 3D 并行策略？
   
   <details>
   <summary>提示</summary>
   计算模型内存需求，考虑激活内存。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   策略设计：
   - 模型内存：70B × 4 bytes = 280GB（FP32）
   - 单GPU无法容纳，需要模型并行
   - 建议配置：TP=4, PP=2, DP=4
   - 每个GPU负责：70B/(4×2) = 8.75B 参数
   - 内存占用：35GB模型 + 20GB激活 < 80GB
   </details>

### 挑战题

5. **主动学习策略设计**
   设计一个结合不确定性、多样性和困难度的综合主动学习策略，给出具体的评分函数和选择算法。
   
   <details>
   <summary>提示</summary>
   考虑三个维度的权重平衡和归一化。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   综合评分函数：
   Score(x) = α·Uncertainty(x) + β·Diversity(x) + γ·Difficulty(x)
   
   其中：
   - Uncertainty(x) = H(x) / log(K)  # 归一化熵
   - Diversity(x) = min_distance(x, selected_set) / max_distance
   - Difficulty(x) = loss(x) / percentile_95_loss
   - α=0.4, β=0.3, γ=0.3（可调）
   
   选择算法：
   1. 计算所有候选样本的综合分数
   2. 贪心选择：每次选最高分，更新已选集合
   3. 动态调整权重：早期重视多样性，后期重视困难度
   </details>

6. **超参数迁移方案**
   设计一个跨任务的超参数迁移学习系统，包括任务相似度计算、历史知识存储和初始化策略。
   
   <details>
   <summary>提示</summary>
   考虑任务特征提取、相似度度量、知识蒸馏。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   系统设计：
   
   1. 任务特征提取：
      - 数据统计：样本数、类别数、文本长度分布
      - 模型特征：架构、参数量、预训练来源
      - 领域特征：任务类型、评估指标
   
   2. 相似度计算：
      - 特征向量余弦相似度
      - 任务嵌入（通过元学习获得）
      - 历史性能相关性
   
   3. 知识迁移：
      - Top-3相似任务的超参数加权平均
      - 添加探索噪声（±20%）
      - 保留任务特定调整空间
   </details>

7. **分布式训练故障恢复**
   设计一个能处理节点故障、网络分区和数据损坏的完整故障恢复系统。
   
   <details>
   <summary>提示</summary>
   考虑检测、隔离、恢复、验证四个阶段。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   故障恢复系统：
   
   1. 故障检测：
      - 心跳监控（5秒超时）
      - 梯度范数异常检测
      - 通信错误率监控
   
   2. 故障隔离：
      - 标记故障节点
      - 重组通信拓扑
      - 数据重分配
   
   3. 状态恢复：
      - 从最近检查点恢复
      - 重放日志恢复中间状态
      - 验证模型参数一致性
   
   4. 弹性调整：
      - 动态调整并行度
      - 重新计算批次大小
      - 更新学习率（根据有效批次）
   </details>

8. **模型合并冲突解决**
   两个模型在相同任务上训练但使用不同数据集，合并时发现30%的参数符号相反，设计解决方案。
   
   <details>
   <summary>提示</summary>
   分析冲突原因，设计多级解决策略。
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   冲突解决方案：
   
   1. 冲突分析：
      - 按层统计冲突比例
      - 识别系统性 vs 随机性冲突
      - 评估参数重要性（梯度、Fisher信息）
   
   2. 分级处理：
      - 关键层（注意力）：基于验证集性能选择
      - 中间层：重要性加权插值
      - 顶层：任务相关性决定
   
   3. 后处理：
      - 微调合并模型（小学习率）
      - 知识蒸馏对齐
      - 验证关键能力保持
   
   4. 预防措施：
      - 训练时添加一致性正则化
      - 使用相同初始化
      - 定期交换梯度信息
   </details>

## 常见陷阱与错误

### 数据循环相关

⚠️ **陷阱1：数据泄露**
- 错误：验证集数据进入训练循环
- 后果：过高估计模型性能
- 解决：严格的数据隔离，版本控制

⚠️ **陷阱2：标注漂移**
- 错误：标注规范随时间变化但未更新历史数据
- 后果：数据不一致，模型学习冲突信号
- 解决：定期审查规范，必要时重新标注

### 主动学习相关

⚠️ **陷阱3：采样偏差**
- 错误：过度关注不确定样本，忽略代表性
- 后果：模型在常见案例上性能下降
- 解决：平衡不确定性和多样性

⚠️ **陷阱4：冷启动问题**
- 错误：初始模型太差，不确定性估计不可靠
- 后果：选择低价值样本
- 解决：初始随机采样建立基线

### 模型合并相关

⚠️ **陷阱5：盲目平均**
- 错误：直接平均所有参数
- 后果：破坏学习的特征表示
- 解决：考虑参数重要性和任务相关性

⚠️ **陷阱6：忽视初始化**
- 错误：合并不同初始化的模型
- 后果：参数空间不对齐
- 解决：使用相同预训练模型作为基础

### 分布式训练相关

⚠️ **陷阱7：通信瓶颈**
- 错误：忽视网络带宽限制
- 后果：GPU利用率低
- 解决：梯度压缩，通信优化

⚠️ **陷阱8：检查点损坏**
- 错误：检查点保存不完整或损坏
- 后果：无法恢复训练
- 解决：原子操作，多副本，验证机制

### 超参数优化相关

⚠️ **陷阱9：过早停止**
- 错误：在学习率预热阶段就停止
- 后果：错过潜在好配置
- 解决：设置最小训练步数

⚠️ **陷阱10：搜索空间过大**
- 错误：同时搜索所有超参数
- 后果：搜索效率低
- 解决：分阶段搜索，固定次要参数