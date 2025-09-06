# 第四章：纯语言任务实验设计

## 章节概览

本章深入探讨纯语言任务的后训练实验设计方法。我们将系统性地介绍如何针对不同类型的文本任务设计实验，包括多轮对话、长文本处理、推理链训练等核心场景。重点关注实验设计的方法论、评估指标选择、数据构造策略以及常见问题的解决方案。通过本章学习，您将掌握设计和执行高质量语言任务实验的完整流程。

## 4.1 多轮对话的意图识别与状态管理

多轮对话是 LLM 后训练中最具挑战性的任务之一。与单轮问答不同，多轮对话需要模型维护复杂的上下文状态、理解隐含意图、处理指代消解，并在保持连贯性的同时适应话题转换。本节将深入探讨如何设计实验来优化这些能力。

### 4.1.1 多轮对话的核心挑战

多轮对话系统面临的主要技术挑战包括：

**1. 上下文依赖性建模**

多轮对话中，用户的每个输入都可能依赖于之前的对话历史。模型需要准确理解这种依赖关系：

```
用户：帮我分析这份报告的数据
助手：[提供分析]
用户：第三部分有什么问题吗？  <- 需要理解"第三部分"指代报告的第三部分
```

实验设计时需要考虑：
- **显式依赖** vs **隐式依赖**：显式依赖通过代词、指示词体现；隐式依赖需要推理
- **依赖距离**：依赖的轮次跨度影响模型的记忆负担
- **依赖类型**：实体指代、事件指代、属性继承等

**2. 意图演化与切换**

用户意图在对话过程中会发生演化或突然切换：

```
轮次 1-3：讨论技术问题（意图：技术咨询）
轮次 4：突然询问天气（意图切换：日常闲聊）
轮次 5：回到技术讨论（意图恢复：技术咨询）
```

关键实验指标：
- 意图切换检测准确率
- 意图恢复后的上下文保持能力
- 多意图并存时的优先级处理

**3. 对话状态的累积效应**

随着对话轮次增加，状态管理的复杂度呈指数增长：

```
状态空间大小 ≈ O(|S|^n)
其中 |S| 是单轮状态空间，n 是对话轮次
```

这导致：
- **状态爆炸**：需要设计高效的状态压缩机制
- **信息衰减**：早期轮次的信息逐渐被遗忘
- **噪声累积**：错误理解在后续轮次中被放大

### 4.1.2 意图识别的实验设计

**实验框架设计**

构建多轮意图识别实验需要系统性的方法：

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│  对话历史   │────>│  特征提取    │────>│ 意图分类  │
│  H_1...H_t  │     │  & 编码器    │     │   模型    │
└─────────────┘     └──────────────┘     └───────────┘
       │                    │                   │
       v                    v                   v
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│ 当前输入 U_t│     │ 上下文融合   │     │ 意图 I_t  │
└─────────────┘     └──────────────┘     └───────────┘
```

**数据构造策略**

1. **意图标注体系设计**
   - 层次化意图树：主意图 -> 子意图 -> 细粒度意图
   - 意图转移矩阵：P(I_t | I_{t-1}, U_t)
   - 复合意图处理：多标签 vs 主导意图

2. **训练数据增强**
   ```python
   # 伪代码：对话数据增强
   def augment_dialogue(dialogue):
       # 1. 意图插入：在对话中插入无关意图
       # 2. 意图省略：删除部分显式意图表达
       # 3. 语言变换：同义改写保持意图不变
       # 4. 顺序打乱：测试鲁棒性
       return augmented_samples
   ```

3. **困难样本挖掘**
   - 意图边界模糊的对话
   - 快速意图切换序列
   - 长距离意图依赖

**评估指标设计**

标准准确率不足以评估多轮意图识别，需要设计专门指标：

1. **轮次感知准确率（Turn-Aware Accuracy）**
   $$TAA = \sum_{t=1}^{T} w_t \cdot \mathbb{1}[pred_t = true_t]$$
   其中 $w_t$ 是轮次权重，可以设计为递增（后期轮次更重要）或递减（早期理解更关键）

2. **意图切换 F1（Intent Switch F1）**
   专门评估模型检测意图切换点的能力

3. **意图一致性得分（Intent Coherence Score）**
   $$ICS = \frac{1}{T-1} \sum_{t=2}^{T} sim(I_t, I_{t-1}) \cdot \mathbb{1}[switch_t = 0]$$

### 4.1.3 对话状态追踪

**状态表示学习**

对话状态不仅包含显式信息（实体、属性），还包含隐式知识（用户偏好、情感状态）：

```
State_t = {
    "entities": {实体及其属性},
    "relations": {实体间关系},
    "user_profile": {推断的用户信息},
    "discourse": {对话结构信息},
    "intent_history": {意图演化轨迹}
}
```

**状态更新机制**

1. **覆盖更新（Overwrite）**
   ```
   if 新信息与旧信息冲突:
       State[key] = new_value
   ```

2. **累积更新（Accumulate）**
   ```
   State[key] = merge(State[key], new_value)
   ```

3. **条件更新（Conditional）**
   ```
   if confidence(new_value) > threshold:
       State[key] = weighted_avg(State[key], new_value)
   ```

**实验设计要点**

1. **状态槽位定义**
   - 预定义槽位：适用于垂直领域
   - 开放槽位：适用于通用对话
   - 混合方案：核心槽位预定义 + 动态扩展

2. **状态初始化策略**
   - 零初始化：从空状态开始
   - 先验初始化：基于用户画像或领域知识
   - 迁移初始化：从相似对话学习初始状态

3. **状态压缩与遗忘**
   ```python
   def compress_state(state, max_size):
       # 信息论方法：保留高信息量状态
       # 注意力方法：保留高注意力权重状态
       # 时间衰减：指数衰减旧状态权重
       return compressed_state
   ```

### 4.1.4 上下文窗口管理策略

**固定窗口 vs 动态窗口**

1. **固定窗口策略**
   - 优点：实现简单，计算开销可控
   - 缺点：可能截断重要信息
   - 适用场景：轮次较少的任务导向对话

2. **动态窗口策略**
   - 基于信息密度调整：信息密集处扩大窗口
   - 基于意图边界调整：意图切换时重置窗口
   - 基于计算资源调整：自适应GPU内存

**窗口内容选择**

不是所有历史信息都同等重要，需要设计选择机制：

```
重要性得分 = α * 时间因子 + β * 相关性因子 + γ * 信息量因子

时间因子 = exp(-λ * (t_current - t_utterance))
相关性因子 = cosine_sim(embedding_current, embedding_historical)
信息量因子 = -log(P(utterance | context))
```

**压缩技术**

1. **摘要压缩**
   ```
   原始对话（10轮） -> 摘要（2-3句） -> 保留关键信息
   ```

2. **关键词提取**
   ```
   每轮提取 top-k 关键词，构建稀疏表示
   ```

3. **层次化压缩**
   ```
   近期轮次：完整保留
   中期轮次：保留主要内容
   远期轮次：仅保留意图和关键实体
   ```

**实验优化策略**

1. **渐进式训练**
   ```python
   for epoch in range(num_epochs):
       if epoch < warmup_epochs:
           max_context_length = min_length
       else:
           max_context_length = min(
               min_length + (epoch - warmup_epochs) * increment,
               max_length
           )
       train_with_context_length(max_context_length)
   ```

2. **课程学习**
   - 从短对话到长对话
   - 从单意图到多意图
   - 从简单依赖到复杂依赖

3. **注意力模式分析**
   定期可视化注意力矩阵，识别：
   - 位置偏差（过度关注近期或远期）
   - 模式坍塌（注意力过度集中）
   - 梯度消失（深层注意力失效）

## 4.2 长文本处理与位置编码优化

长文本处理是 LLM 后训练中的关键技术挑战。随着应用场景从简单问答扩展到文档理解、代码生成、长篇创作等领域，模型需要处理数万甚至数十万 token 的输入。本节探讨如何通过实验设计克服长序列建模的技术瓶颈。

### 4.2.1 长序列建模的技术瓶颈

**计算复杂度问题**

标准 Transformer 的自注意力机制具有 O(n²) 的时间和空间复杂度：

```
内存需求 = batch_size × seq_len² × hidden_dim × 4 bytes
计算需求 = 2 × batch_size × seq_len² × hidden_dim FLOPs

例：seq_len=32K, hidden_dim=4096, batch_size=1
内存 ≈ 16GB (仅注意力矩阵)
```

这导致：
- **内存墙**：GPU内存限制了最大序列长度
- **计算墙**：训练时间随序列长度平方增长
- **梯度消失**：深层网络中的长距离依赖难以学习

**位置编码的局限性**

1. **绝对位置编码的外推失败**
   ```
   训练：position ∈ [0, 2048]
   推理：position = 10000
   结果：性能急剧下降
   ```

2. **相对位置编码的计算开销**
   ```
   相对位置矩阵：O(n² × d_k)
   每个注意力头都需要独立计算
   ```

3. **位置信息的语义混淆**
   ```
   "第1000个token" 在不同文档中含义不同
   需要考虑：段落边界、句子边界、语义单元
   ```

**信息瓶颈与遗忘曲线**

长序列中的信息传递存在瓶颈：

```
信息保留率 ≈ exp(-λ × distance)
其中 λ 是遗忘系数，distance 是 token 距离
```

实验观察：
- 2K tokens：95% 信息保留
- 8K tokens：70% 信息保留  
- 32K tokens：<40% 信息保留

### 4.2.2 位置编码的改进方法

**RoPE（旋转位置编码）优化**

RoPE 通过旋转矩阵编码相对位置，具有良好的外推性：

```python
def rope_encoding(q, k, positions):
    # 基础 RoPE
    theta = 10000.0
    dim = q.shape[-1]
    
    # 位置频率计算
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # 应用旋转
    q_rot = apply_rotation(q, freqs, positions)
    k_rot = apply_rotation(k, freqs, positions)
    
    return q_rot, k_rot
```

**实验优化策略**：

1. **动态 theta 调整**
   ```python
   # 根据序列长度自适应调整
   theta = base_theta * (current_len / base_len) ** 0.5
   ```

2. **NTK-aware 缩放**
   ```python
   # Neural Tangent Kernel 感知的位置缩放
   scale = (max_len / base_len) ** (dim / (dim - 2))
   positions = positions / scale
   ```

3. **位置插值（Position Interpolation）**
   ```python
   # 线性插值压缩位置范围
   compressed_pos = positions * (base_len / current_len)
   ```

**ALiBi（Attention with Linear Biases）**

ALiBi 通过线性偏置实现位置感知，无需显式位置编码：

```python
def alibi_bias(seq_len, num_heads):
    # 为每个注意力头生成不同的斜率
    slopes = 2 ** (-8 * torch.arange(num_heads) / num_heads)
    
    # 构建偏置矩阵
    positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    biases = slopes.unsqueeze(1).unsqueeze(1) * positions.unsqueeze(0)
    
    return biases
```

**实验对比框架**

```python
def compare_position_encodings(model, test_lengths):
    results = {}
    
    for encoding_type in ['rope', 'alibi', 'absolute', 'relative']:
        model.set_position_encoding(encoding_type)
        
        for length in test_lengths:
            # 测试困惑度
            ppl = evaluate_perplexity(model, length)
            
            # 测试长距离依赖
            long_dep_score = test_long_range_dependency(model, length)
            
            # 测试位置敏感任务
            position_acc = test_position_sensitive_tasks(model, length)
            
            results[encoding_type][length] = {
                'perplexity': ppl,
                'long_dependency': long_dep_score,
                'position_accuracy': position_acc
            }
    
    return results
```

### 4.2.3 注意力机制的优化

**稀疏注意力模式**

1. **局部注意力（Local Attention）**
   ```
   每个 token 只关注窗口内的 k 个邻近 token
   复杂度：O(n × k) where k << n
   ```

2. **跨步注意力（Strided Attention）**
   ```
   固定步长采样：attend to every k-th token
   复杂度：O(n²/k)
   ```

3. **Longformer 风格的混合注意力**
   ```
   Global + Local + Dilated patterns
   全局 token：[CLS], [SEP] 等特殊标记
   局部窗口：size = 512
   膨胀窗口：dilation = 2, 4, 8
   ```

**Flash Attention 优化**

Flash Attention 通过 IO 优化显著提升长序列性能：

```python
def flash_attention_experiment(seq_lengths, batch_sizes):
    results = []
    
    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            # 标准注意力
            std_time, std_mem = benchmark_standard_attention(
                batch_size, seq_len
            )
            
            # Flash Attention
            flash_time, flash_mem = benchmark_flash_attention(
                batch_size, seq_len
            )
            
            results.append({
                'seq_len': seq_len,
                'batch_size': batch_size,
                'speedup': std_time / flash_time,
                'memory_saving': std_mem / flash_mem
            })
    
    return results
```

**注意力下采样策略**

```python
class AttentionDownsampling:
    def __init__(self, downsample_ratio=4):
        self.ratio = downsample_ratio
    
    def compute_attention(self, q, k, v):
        # 1. 对 K,V 进行下采样
        k_downsampled = self.downsample(k, self.ratio)
        v_downsampled = self.downsample(v, self.ratio)
        
        # 2. 计算稀疏注意力
        attn_scores = torch.matmul(q, k_downsampled.transpose(-2, -1))
        attn_weights = F.softmax(attn_scores / math.sqrt(d_k), dim=-1)
        
        # 3. 加权聚合
        output = torch.matmul(attn_weights, v_downsampled)
        
        # 4. 上采样恢复
        output = self.upsample(output, self.ratio)
        
        return output
```

### 4.2.4 分块策略与滑动窗口

**分块处理框架**

```python
class ChunkedProcessing:
    def __init__(self, chunk_size=2048, overlap=256):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_long_text(self, text, model):
        chunks = self.create_chunks(text)
        chunk_outputs = []
        
        # 维护跨块状态
        cross_chunk_state = None
        
        for i, chunk in enumerate(chunks):
            # 添加上下文
            if i > 0:
                context = chunks[i-1][-self.overlap:]
                chunk = context + chunk
            
            # 处理当前块
            output, new_state = model.process_with_state(
                chunk, cross_chunk_state
            )
            
            # 更新状态
            cross_chunk_state = new_state
            
            # 去除重叠部分
            if i > 0:
                output = output[self.overlap:]
            
            chunk_outputs.append(output)
        
        return self.merge_outputs(chunk_outputs)
```

**滑动窗口注意力**

```python
def sliding_window_attention(seq_len, window_size, stride):
    """
    生成滑动窗口注意力掩码
    """
    mask = torch.zeros(seq_len, seq_len)
    
    for i in range(seq_len):
        # 局部窗口
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
        
        # 跨步连接（用于长距离依赖）
        for j in range(0, seq_len, stride):
            if abs(i - j) > window_size:
                mask[i, j] = 1
    
    return mask
```

**层次化处理策略**

```
文档级别处理：
┌──────────────────────────────────┐
│          完整文档 (32K)          │
└──────────────────────────────────┘
                ↓
┌─────────┬─────────┬─────────┬─────────┐
│ 段落1   │ 段落2   │ 段落3   │ 段落4   │
│ (8K)    │ (8K)    │ (8K)    │ (8K)    │
└─────────┴─────────┴─────────┴─────────┘
                ↓
┌───┬───┬───┬───┬───┬───┬───┬───┐
│句1│句2│...│句n│句1│句2│...│句m│  细粒度处理
└───┴───┴───┴───┴───┴───┴───┴───┘
```

**实验评估协议**

```python
def evaluate_long_context_methods():
    test_suite = {
        'retrieval': test_needle_in_haystack,  # 长文检索
        'summary': test_summarization,          # 摘要生成
        'qa': test_multi_hop_qa,               # 多跳问答
        'coherence': test_long_form_generation  # 长文生成连贯性
    }
    
    methods = {
        'baseline': StandardAttention(),
        'chunked': ChunkedProcessing(),
        'sliding': SlidingWindowAttention(),
        'sparse': SparseAttention(),
        'hierarchical': HierarchicalProcessing()
    }
    
    results = {}
    for method_name, method in methods.items():
        results[method_name] = {}
        
        for task_name, task_fn in test_suite.items():
            score = task_fn(method)
            results[method_name][task_name] = score
    
    return results
```

**内存优化技巧**

1. **梯度检查点（Gradient Checkpointing）**
   ```python
   # 用计算换内存
   def forward_with_checkpointing(self, x):
       # 不保存中间激活值，反向传播时重新计算
       return checkpoint(self.transformer_block, x)
   ```

2. **混合精度训练**
   ```python
   # FP16/BF16 减少内存占用
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   ```

3. **激活值重计算**
   ```python
   # 选择性保存关键层的激活值
   critical_layers = [0, 12, 24]  # 仅保存这些层
   ```

## 4.3 思维链（CoT）训练策略

思维链（Chain-of-Thought）是提升 LLM 推理能力的关键技术。通过训练模型生成中间推理步骤，CoT 显著改善了复杂推理任务的性能。本节探讨如何设计实验来构建高质量的 CoT 数据集、优化训练策略，以及解决 CoT 训练中的常见问题。

### 4.3.1 CoT 数据的构造方法

**CoT 数据的核心要素**

高质量的 CoT 数据需要满足：

1. **步骤完整性**：覆盖从问题到答案的所有推理步骤
2. **逻辑连贯性**：步骤间有清晰的因果关系
3. **粒度适中**：既不过于冗长，也不跳跃太大
4. **错误可追溯**：便于定位推理错误的位置

**自动化 CoT 生成策略**

```python
class CoTGenerator:
    def __init__(self, base_model, verifier_model):
        self.base_model = base_model
        self.verifier = verifier_model
    
    def generate_cot(self, question, answer):
        # 1. 零样本 CoT 生成
        prompt = f"Question: {question}\nLet's think step by step:"
        reasoning = self.base_model.generate(prompt)
        
        # 2. 验证推理链
        validity_score = self.verifier.check_reasoning(
            question, reasoning, answer
        )
        
        # 3. 迭代优化
        while validity_score < threshold:
            # 添加引导信息
            feedback = self.verifier.get_feedback(reasoning)
            prompt_with_feedback = f"{prompt}\n{feedback}"
            reasoning = self.base_model.generate(prompt_with_feedback)
            validity_score = self.verifier.check_reasoning(
                question, reasoning, answer
            )
        
        return reasoning
```

**多样化 CoT 采样**

```python
def diverse_cot_sampling(question, num_samples=5):
    """
    生成多样化的推理路径
    """
    cot_samples = []
    
    # 温度采样
    for temp in [0.3, 0.5, 0.7, 0.9, 1.1]:
        cot = generate_with_temperature(question, temp)
        cot_samples.append(cot)
    
    # 提示变体
    prompt_templates = [
        "Let's solve this step by step:",
        "Breaking this down:",
        "First, let me understand the problem:",
        "Let's approach this systematically:",
        "Working through this carefully:"
    ]
    
    for template in prompt_templates:
        cot = generate_with_prompt(question, template)
        cot_samples.append(cot)
    
    # 去重和质量筛选
    unique_cots = deduplicate_reasoning_paths(cot_samples)
    high_quality_cots = filter_by_quality(unique_cots)
    
    return high_quality_cots
```

**领域特定 CoT 模板**

```python
# 数学推理模板
MATH_COT_TEMPLATE = """
1. 理解问题：{problem_understanding}
2. 识别已知条件：{given_conditions}
3. 确定求解目标：{target}
4. 选择解法：{method_selection}
5. 执行计算：{calculation_steps}
6. 验证答案：{verification}
"""

# 逻辑推理模板
LOGIC_COT_TEMPLATE = """
1. 前提条件：{premises}
2. 推理规则：{rules}
3. 推导过程：{derivation}
4. 结论：{conclusion}
"""

# 代码推理模板
CODE_COT_TEMPLATE = """
1. 需求分析：{requirements}
2. 算法设计：{algorithm}
3. 复杂度分析：{complexity}
4. 边界条件：{edge_cases}
5. 实现：{implementation}
"""
```

### 4.3.2 推理步骤的质量控制

**步骤粒度控制**

```python
def optimize_step_granularity(cot_chain):
    """
    优化推理步骤的粒度
    """
    optimized_chain = []
    
    for i, step in enumerate(cot_chain):
        # 检测过于复杂的步骤
        if is_complex_step(step):
            # 分解为子步骤
            sub_steps = decompose_step(step)
            optimized_chain.extend(sub_steps)
        
        # 检测冗余步骤
        elif is_redundant_step(step, optimized_chain):
            continue  # 跳过冗余
        
        # 合并过于细碎的步骤
        elif is_trivial_step(step) and i > 0:
            # 与前一步合并
            optimized_chain[-1] = merge_steps(optimized_chain[-1], step)
        
        else:
            optimized_chain.append(step)
    
    return optimized_chain
```

**逻辑一致性检查**

```python
class LogicConsistencyChecker:
    def __init__(self):
        self.rules = self.load_logic_rules()
    
    def check_consistency(self, cot_chain):
        """
        检查推理链的逻辑一致性
        """
        issues = []
        
        # 1. 前后矛盾检测
        for i in range(len(cot_chain) - 1):
            if self.contradicts(cot_chain[i], cot_chain[i+1]):
                issues.append(f"Step {i} contradicts step {i+1}")
        
        # 2. 循环推理检测
        if self.has_circular_reasoning(cot_chain):
            issues.append("Circular reasoning detected")
        
        # 3. 跳跃推理检测
        for i in range(len(cot_chain) - 1):
            if not self.is_valid_transition(cot_chain[i], cot_chain[i+1]):
                issues.append(f"Invalid transition from step {i} to {i+1}")
        
        # 4. 前提完备性检查
        missing_premises = self.check_premises(cot_chain)
        if missing_premises:
            issues.append(f"Missing premises: {missing_premises}")
        
        return issues
```

**数值准确性验证**

```python
def verify_numerical_accuracy(cot_chain):
    """
    验证推理链中的数值计算
    """
    numerical_errors = []
    
    for step in cot_chain:
        # 提取数学表达式
        expressions = extract_math_expressions(step)
        
        for expr in expressions:
            try:
                # 符号计算验证
                symbolic_result = sympy.simplify(expr)
                
                # 数值计算验证
                numerical_result = eval_safely(expr)
                
                # 检查一致性
                if not math.isclose(symbolic_result, numerical_result, rel_tol=1e-9):
                    numerical_errors.append({
                        'step': step,
                        'expression': expr,
                        'symbolic': symbolic_result,
                        'numerical': numerical_result
                    })
            except Exception as e:
                numerical_errors.append({
                    'step': step,
                    'expression': expr,
                    'error': str(e)
                })
    
    return numerical_errors
```

### 4.3.3 自洽性训练

**Self-Consistency 实现**

```python
class SelfConsistencyTraining:
    def __init__(self, model, num_paths=5):
        self.model = model
        self.num_paths = num_paths
    
    def train_step(self, question, true_answer):
        # 1. 生成多条推理路径
        reasoning_paths = []
        for _ in range(self.num_paths):
            path = self.model.generate_cot(question)
            reasoning_paths.append(path)
        
        # 2. 投票得出答案
        answers = [extract_answer(path) for path in reasoning_paths]
        majority_answer = self.majority_vote(answers)
        
        # 3. 选择一致的路径作为正例
        consistent_paths = [
            path for path, ans in zip(reasoning_paths, answers)
            if ans == majority_answer
        ]
        
        # 4. 构造训练样本
        if majority_answer == true_answer:
            # 正确答案：所有一致路径都是正例
            positive_samples = consistent_paths
            negative_samples = []
        else:
            # 错误答案：需要生成修正路径
            positive_samples = [self.generate_correction(q, true_answer)]
            negative_samples = consistent_paths
        
        # 5. 对比学习
        loss = self.contrastive_loss(
            positive_samples, negative_samples
        )
        
        return loss
```

**路径多样性奖励**

```python
def diversity_reward(reasoning_paths):
    """
    计算推理路径的多样性奖励
    """
    n = len(reasoning_paths)
    diversity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # 计算路径相似度
            similarity = path_similarity(
                reasoning_paths[i], 
                reasoning_paths[j]
            )
            diversity_matrix[i][j] = 1 - similarity
            diversity_matrix[j][i] = diversity_matrix[i][j]
    
    # 多样性得分：平均两两差异
    diversity_score = np.mean(diversity_matrix)
    
    return diversity_score

def path_similarity(path1, path2):
    """
    计算两条推理路径的相似度
    """
    # 1. 结构相似度（步骤数量、顺序）
    struct_sim = structural_similarity(path1, path2)
    
    # 2. 语义相似度（内容相似性）
    semantic_sim = semantic_similarity(path1, path2)
    
    # 3. 方法相似度（使用的推理方法）
    method_sim = method_similarity(path1, path2)
    
    return 0.3 * struct_sim + 0.5 * semantic_sim + 0.2 * method_sim
```

### 4.3.4 错误传播与纠正机制

**错误传播分析**

```python
class ErrorPropagationAnalyzer:
    def __init__(self):
        self.error_types = {
            'calculation': '计算错误',
            'logic': '逻辑错误',
            'assumption': '假设错误',
            'interpretation': '理解错误'
        }
    
    def trace_error_propagation(self, cot_chain, final_error):
        """
        追踪错误在推理链中的传播
        """
        error_trace = []
        
        # 反向追踪
        for i in range(len(cot_chain) - 1, -1, -1):
            step = cot_chain[i]
            
            # 检测该步骤是否包含错误
            step_errors = self.detect_step_errors(step)
            
            if step_errors:
                error_trace.append({
                    'step_index': i,
                    'step_content': step,
                    'errors': step_errors,
                    'impact': self.assess_impact(step_errors, final_error)
                })
                
                # 判断是否为根本原因
                if self.is_root_cause(step_errors, cot_chain[:i]):
                    error_trace[-1]['is_root_cause'] = True
                    break
        
        return error_trace
```

**自动纠错机制**

```python
class AutoCorrector:
    def __init__(self, model, verifier):
        self.model = model
        self.verifier = verifier
    
    def correct_reasoning_chain(self, cot_chain, question, expected_answer):
        """
        自动纠正推理链中的错误
        """
        corrected_chain = []
        error_detected = False
        
        for i, step in enumerate(cot_chain):
            # 验证当前步骤
            step_valid, error_info = self.verifier.verify_step(
                step, 
                context=corrected_chain,
                question=question
            )
            
            if step_valid:
                corrected_chain.append(step)
            else:
                error_detected = True
                
                # 生成修正步骤
                correction_prompt = self.build_correction_prompt(
                    step, error_info, corrected_chain, question
                )
                
                corrected_step = self.model.generate(correction_prompt)
                
                # 验证修正
                if self.verifier.verify_step(corrected_step, corrected_chain, question)[0]:
                    corrected_chain.append(corrected_step)
                else:
                    # 如果修正失败，重新生成整个后续链
                    remaining_chain = self.regenerate_from_step(
                        i, corrected_chain, question, expected_answer
                    )
                    corrected_chain.extend(remaining_chain)
                    break
        
        return corrected_chain, error_detected
```

**错误模式学习**

```python
class ErrorPatternLearning:
    def __init__(self):
        self.error_patterns = defaultdict(list)
    
    def learn_from_errors(self, error_examples):
        """
        从错误案例中学习模式
        """
        for example in error_examples:
            # 提取错误特征
            features = self.extract_error_features(example)
            
            # 聚类相似错误
            cluster_id = self.cluster_error(features)
            
            self.error_patterns[cluster_id].append({
                'features': features,
                'correction': example['correction'],
                'explanation': example['explanation']
            })
    
    def predict_error_likelihood(self, step, context):
        """
        预测步骤出错的可能性
        """
        features = self.extract_step_features(step, context)
        
        max_similarity = 0
        most_likely_error = None
        
        for cluster_id, patterns in self.error_patterns.items():
            for pattern in patterns:
                similarity = self.feature_similarity(
                    features, pattern['features']
                )
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_likely_error = pattern
        
        return max_similarity, most_likely_error
```

## 4.4 领域适应与持续学习

### 4.4.1 领域数据的选择与配比

### 4.4.2 灾难性遗忘的缓解

### 4.4.3 增量学习策略

### 4.4.4 知识蒸馏与正则化

## 4.5 幻觉检测与缓解

### 4.5.1 幻觉的分类与成因

### 4.5.2 检测方法与评估指标

### 4.5.3 训练时的缓解策略

### 4.5.4 推理时的干预技术

## 本章小结

## 练习题

## 常见陷阱与错误（Gotchas）