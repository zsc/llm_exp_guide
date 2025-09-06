# 第一章：后训练基础理论

## 引言

后训练（Post-training）是将预训练好的大语言模型转化为实用AI助手的关键步骤。不同于预训练阶段追求的通用语言建模能力，后训练专注于让模型学会遵循人类指令、保持安全输出、展现有用性。本章将深入探讨后训练的理论基础，包括核心方法论、目标函数设计、以及实践中的关键权衡。

**学习目标**：
- 理解后训练与预训练的本质区别
- 掌握SFT、RLHF、DPO等主流方法的数学原理
- 认识对齐税（Alignment Tax）及其影响
- 学会分析和处理分布偏移问题

## 1.1 后训练的定义与动机

### 1.1.1 从语言模型到AI助手

预训练模型通过在海量文本上学习，获得了强大的语言理解和生成能力。然而，原始的语言模型存在几个关键问题：

1. **目标不匹配**：预训练优化的是下一个token预测准确率，而非完成用户任务
2. **行为不可控**：可能生成有害、偏见或虚假内容  
3. **交互能力差**：缺乏对话管理、指令理解等能力

后训练通过引入人类偏好和价值观，将"预测下一个token"的模型转化为"完成人类任务"的助手。这个转化过程本质上是一个**分布对齐**问题：

$$P_{pretrain}(y|x) \rightarrow P_{aligned}(y|x) \approx P_{human}(y|x)$$

### 1.1.2 后训练的目标层次

后训练的目标可以分解为多个层次，每个层次都有其独特的挑战：

```
L4: 价值对齐 (Value Alignment)
    ├── 伦理原则遵循
    ├── 文化敏感性
    └── 长期影响考虑
    
L3: 任务能力 (Task Capability)  
    ├── 指令理解与执行
    ├── 多步推理
    └── 知识运用
    
L2: 交互质量 (Interaction Quality)
    ├── 对话连贯性
    ├── 角色一致性
    └── 语气适应性
    
L1: 基础安全 (Basic Safety)
    ├── 有害内容过滤
    ├── 个人信息保护
    └── 事实准确性
```

每个层次的优化往往存在冲突。例如，过度强调L1的安全性可能损害L3的任务能力，这就是后训练中的根本性权衡。

### 1.1.3 后训练的核心挑战

后训练面临的挑战不仅是技术性的，更是系统性的：

```
预训练分布 P_pretrain(x)
     ↓
   后训练（多目标优化）
     ↓
目标分布 P_aligned(x)

核心挑战：
1. 保留原有能力（避免灾难性遗忘）
   - 知识保持率 > 90%
   - 推理能力保持率 > 85%
   
2. 学习新行为（指令跟随）
   - 指令遵循率 > 95%
   - 格式一致性 > 90%
   
3. 维持输出多样性（避免模式坍塌）
   - 响应熵 H(Y|X) > 阈值
   - 创造性任务多样性保持
   
4. 分布泛化（OOD Generalization）
   - 训练分布 ≠ 部署分布
   - 需要鲁棒性机制
```

### 1.1.4 后训练的数学形式化

从优化角度看，后训练可以形式化为约束优化问题：

$$\begin{aligned}
\max_{\theta} &\quad \mathbb{E}_{x \sim \mathcal{D}_{deploy}}[\text{Utility}(x, \pi_\theta)] \\
\text{s.t.} &\quad \text{Safety}(\pi_\theta) \geq \tau_{safe} \\
&\quad D_{KL}(\pi_\theta || \pi_{pretrain}) \leq \epsilon \\
&\quad \text{Diversity}(\pi_\theta) \geq \tau_{div}
\end{aligned}$$

其中：
- Utility 衡量模型的有用性
- Safety 确保输出安全性
- KL约束防止能力退化
- Diversity 保持输出多样性

### 1.1.5 预训练vs后训练的本质区别

| 维度 | 预训练 | 后训练 |
|------|--------|--------|
| **优化目标** | 最大似然 $P(x)$ | 最大效用 $U(x,y)$ |
| **数据规模** | TB级别 | GB级别 |
| **数据质量** | 数量>质量 | 质量>数量 |
| **学习范式** | 无监督/自监督 | 监督/强化学习 |
| **计算需求** | 数千GPU天 | 数十GPU天 |
| **更新频率** | 一次性 | 持续迭代 |
| **评估标准** | 困惑度 | 人类偏好 |

## 1.2 后训练方法体系

### 1.2.1 监督微调（SFT）

监督微调是最直接的后训练方法，通过高质量的（指令，响应）对来训练模型。虽然概念简单，但SFT的实施细节决定了后续所有方法的基础质量。

**损失函数**：
$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{SFT}} \left[ \sum_{t=1}^{|y|} \log p_\theta(y_t | x, y_{<t}) \right]$$

其中：
- $x$ 是输入指令
- $y$ 是目标响应  
- $\mathcal{D}_{SFT}$ 是监督数据集

**数据构造策略**：

1. **人工编写**：成本高但质量最佳
   - 典型规模：5K-50K样本
   - 成本：$5-50/样本
   
2. **模型生成+人工筛选**：平衡成本和质量
   - 生成10x候选，人工选择最佳
   - 成本降低80%，质量保持90%
   
3. **自举方法（Self-Instruct）**：
   ```python
   def self_instruct(seed_tasks, model, n_iter):
       tasks = seed_tasks
       for i in range(n_iter):
           # 生成新指令
           new_instructions = model.generate_instructions(tasks)
           # 生成响应
           responses = model.generate_responses(new_instructions)
           # 质量过滤
           filtered = quality_filter(new_instructions, responses)
           tasks.extend(filtered)
       return tasks
   ```

**关键超参数选择**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 ~ 5e-6 | 比预训练低10x |
| Batch Size | 32-128 | 取决于序列长度 |
| Epochs | 1-3 | 避免过拟合 |
| Warmup Steps | 100-500 | 平滑初始训练 |
| Weight Decay | 0.01 | 轻微正则化 |
| Gradient Clipping | 1.0 | 防止梯度爆炸 |

**SFT的隐含陷阱**：
- **格式过拟合**：模型记住特定格式而非学会任务
- **多样性丧失**：响应变得模板化
- **长度偏见**：倾向生成训练数据的平均长度

### 1.2.2 人类反馈强化学习（RLHF）

RLHF通过强化学习优化人类偏好，是目前最成功的对齐方法。其核心创新在于将人类偏好转化为可优化的奖励信号。

**完整RLHF流程**：

```
Step 1: 收集偏好数据
├── 对同一指令生成多个响应
├── 人工标注偏好排序
└── 构建对比数据集

Step 2: 训练奖励模型
├── 使用Bradley-Terry模型
├── 学习隐含奖励函数
└── 验证奖励一致性

Step 3: PPO优化
├── 初始化：π_θ = π_SFT
├── 采样：生成响应
├── 评分：计算奖励
├── 更新：PPO梯度步
└── 约束：KL惩罚
```

**1. 奖励模型训练**：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}_{pref}} \left[ \log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l)) \right]$$

**奖励模型架构选择**：
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model
        # 关键：使用独立的value head
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask)
        # 使用最后一个token的表示
        rewards = self.value_head(outputs.last_hidden_state[:, -1, :])
        return rewards
```

**2. PPO策略优化**：

PPO的核心是通过clip机制限制策略更新幅度：

$$\mathcal{L}_{PPO}^{clip} = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$

**PPO超参数经验值**：
- clip范围 $\epsilon$: 0.2
- 价值函数系数: 0.5
- 熵奖励系数: 0.01
- KL惩罚 $\beta$: 动态调整
  ```python
  if kl > target_kl * 1.5:
      β *= 1.5  # 增强约束
  elif kl < target_kl * 0.5:
      β *= 0.5  # 放松约束
  ```

**3. KL散度约束的重要性**：

KL约束防止策略偏离过远，保持模型能力：

$$D_{KL}(\pi_\theta || \pi_{ref}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]$$

实践中的KL预算分配：
- 初期（探索）：KL budget = 10-20
- 中期（优化）：KL budget = 5-10  
- 后期（收敛）：KL budget = 1-5

### 1.2.3 直接偏好优化（DPO）

DPO通过重参数化技巧，将RLHF的RL问题转化为监督学习问题，大幅简化了训练流程。

**理论推导**：

从RLHF的优化目标出发：
$$\max_{\pi_\theta} \mathbb{E}_{x,y \sim \pi_\theta} [r(x,y)] - \beta D_{KL}(\pi_\theta || \pi_{ref})$$

最优策略的闭式解为：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta}r(x,y)\right)$$

代入Bradley-Terry模型，得到DPO损失：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**DPO实施细节**：

```python
def compute_dpo_loss(model, ref_model, batch, beta=0.1):
    # 计算策略模型的log概率
    policy_chosen_logps = model.log_prob(batch.chosen)
    policy_reject_logps = model.log_prob(batch.rejected)
    
    # 计算参考模型的log概率
    with torch.no_grad():
        ref_chosen_logps = ref_model.log_prob(batch.chosen)
        ref_reject_logps = ref_model.log_prob(batch.rejected)
    
    # 计算log比率
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    reject_rewards = beta * (policy_reject_logps - ref_reject_logps)
    
    # DPO损失
    loss = -F.logsigmoid(chosen_rewards - reject_rewards).mean()
    return loss
```

**DPO vs RLHF对比**：

| 方面 | RLHF | DPO |
|------|------|-----|
| **内存需求** | 4个模型 | 2个模型 |
| **训练稳定性** | 需要精细调参 | 稳定如SFT |
| **采样需求** | 在线采样 | 离线数据 |
| **超参数** | 10+ | 2-3 |
| **收敛速度** | 慢 | 快 |
| **最终性能** | ★★★★★ | ★★★★ |

### 1.2.4 其他后训练方法

**1. IPO (Identity Preference Optimization)**：

使用更简单的损失函数：
$$\mathcal{L}_{IPO} = \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \tau \right)^2$$

优势：避免sigmoid饱和问题

**2. RLAIF (RL from AI Feedback)**：

用AI模型替代人类标注：
```python
def generate_ai_preferences(instruction, responses, critic_model):
    # AI评判标准
    criteria = """
    1. 准确性和事实性
    2. 有用性和相关性
    3. 清晰度和组织性
    4. 安全性和适当性
    """
    
    scores = []
    for response in responses:
        prompt = f"{criteria}\n\n指令：{instruction}\n响应：{response}\n评分："
        score = critic_model.evaluate(prompt)
        scores.append(score)
    
    # 转换为偏好对
    preferences = create_preference_pairs(responses, scores)
    return preferences
```

**3. Constitutional AI**：

通过规则引导的自我改进：
```
Initial Response → Critique → Revision → Final Response
         ↑                                      ↓
         └──────── Constitutional Rules ────────┘
```

**4. ORPO (Odds Ratio Preference Optimization)**：

结合SFT和偏好优化：
$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}$$

其中odds ratio损失：
$$\mathcal{L}_{OR} = -\log \sigma\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

## 1.3 对齐税与能力权衡

### 1.3.1 对齐税的定义与表现

对齐税（Alignment Tax）是后训练不可避免的副作用，指模型为了获得对齐能力（安全性、有用性、诚实性）而付出的原始能力代价。这不是bug，而是当前技术的根本性限制。

**对齐税的具体表现**：

```
能力维度评估（真实案例统计）：
┌─────────────────┬──────────┬──────────┬─────────────┐
│    能力类别      │ 预训练   │ 后训练   │   退化原因   │
├─────────────────┼──────────┼──────────┼─────────────┤
│ 事实知识召回    │   95%    │   92%    │ 安全过滤    │
│ 数学推理        │   88%    │   85%    │ 格式约束    │
│ 代码生成        │   92%    │   88%    │ 拒绝机制    │
│ 创造性写作      │   90%    │   75%    │ 模式坍塌    │
│ 多语言能力      │   85%    │   78%    │ 英语偏向    │
├─────────────────┼──────────┼──────────┼─────────────┤
│ 安全性          │   60%    │   95%    │ 主要目标 ✓  │
│ 指令跟随        │   40%    │   90%    │ 主要目标 ✓  │
│ 拒绝有害请求    │   20%    │   98%    │ 主要目标 ✓  │
└─────────────────┴──────────┴──────────┴─────────────┘
```

**对齐税的深层机制**：

1. **表示空间重组**：
   ```python
   # 可视化：t-SNE投影显示
   # 预训练：知识均匀分布
   # 后训练：形成"安全区"和"危险区"聚类
   ```

2. **注意力模式改变**：
   - 预训练：均匀注意力分布
   - 后训练：过度关注安全相关token

3. **输出分布变窄**：
   $$H(Y|X)_{aligned} < H(Y|X)_{pretrain}$$

### 1.3.2 对齐税的精确测量

**多维度测量框架**：

```python
class AlignmentTaxMeasurer:
    def __init__(self, pretrain_model, aligned_model):
        self.benchmarks = {
            'knowledge': ['MMLU', 'TriviaQA', 'NaturalQuestions'],
            'reasoning': ['GSM8K', 'MATH', 'BBH'],
            'coding': ['HumanEval', 'MBPP', 'CodeContests'],
            'creativity': ['story_continuation', 'poetry_generation'],
            'multimodal': ['VQA', 'COCO_caption'] if has_vision else []
        }
    
    def measure_tax(self):
        results = {}
        for category, tests in self.benchmarks.items():
            pretrain_scores = evaluate(self.pretrain_model, tests)
            aligned_scores = evaluate(self.aligned_model, tests)
            
            # 计算各种指标
            results[category] = {
                'absolute_drop': pretrain_scores - aligned_scores,
                'relative_drop': (pretrain_scores - aligned_scores) / pretrain_scores,
                'worst_case': min(aligned_scores),
                'variance': np.var(aligned_scores)
            }
        
        # 综合对齐税
        total_tax = weighted_average(results, self.importance_weights)
        return results, total_tax
```

**细粒度分析**：

1. **Token级别对齐税**：
   $$\text{Tax}_{token} = \sum_{t} D_{KL}(P_{pre}(x_t) || P_{align}(x_t))$$

2. **任务级别对齐税**：
   $$\text{Tax}_{task} = \frac{\text{Perf}_{pre} - \text{Perf}_{align}}{\text{Perf}_{pre}}$$

3. **分布级别对齐税**：
   $$\text{Tax}_{dist} = \mathcal{W}_2(P_{pre}, P_{align})$$
   （Wasserstein距离）

### 1.3.3 对齐税的缓解策略

**1. 混合训练（Mix Training）**：

```python
def create_mixed_dataset(alignment_data, pretrain_data, mix_ratio=0.9):
    """
    关键发现：10-15%的预训练数据可减少50%的对齐税
    """
    n_alignment = int(len(alignment_data) * mix_ratio)
    n_pretrain = len(alignment_data) - n_alignment
    
    # 策略1：随机混合
    mixed = random.sample(alignment_data, n_alignment) + \
            random.sample(pretrain_data, n_pretrain)
    
    # 策略2：课程混合（推荐）
    # 早期多预训练数据，后期多对齐数据
    schedule = lambda epoch: 0.3 * exp(-epoch/10) + 0.1
    
    return mixed
```

**2. 弹性权重巩固（EWC）**：

保护重要参数不被过度修改：

$$\mathcal{L}_{EWC} = \mathcal{L}_{align} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{pre,i})^2$$

其中$F_i$是Fisher信息矩阵的对角元素：

```python
def compute_fisher_information(model, data):
    fisher = {}
    model.eval()
    
    for batch in data:
        loss = model(batch).loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in fisher:
                    fisher[name] = param.grad.data ** 2
                else:
                    fisher[name] += param.grad.data ** 2
    
    # 归一化
    for name in fisher:
        fisher[name] /= len(data)
    
    return fisher
```

**3. 层级微调（Layer-wise Fine-tuning）**：

```python
def layerwise_finetune(model, data, layer_schedule):
    """
    从顶层到底层逐渐解冻
    底层保留更多预训练知识
    """
    for stage, layers_to_unfreeze in enumerate(layer_schedule):
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻指定层
        for layer_idx in layers_to_unfreeze:
            for param in model.layers[layer_idx].parameters():
                param.requires_grad = True
        
        # 训练当前阶段
        train_stage(model, data, epochs=2)
```

**4. 知识蒸馏（Knowledge Distillation）**：

从预训练模型蒸馏知识：

$$\mathcal{L}_{KD} = \alpha \mathcal{L}_{align} + (1-\alpha) \mathcal{L}_{distill}$$

其中：
$$\mathcal{L}_{distill} = \tau^2 \cdot D_{KL}(P_{student}^{\tau} || P_{teacher}^{\tau})$$

### 1.3.4 帕累托前沿优化

在多目标优化中，寻找能力-对齐的最优权衡：

```
对齐程度 ↑
    │     ○ 帕累托最优点
    │    ╱│
    │   ╱ │ ← 可达区域
    │  ╱  • 次优解
    │ ╱   │
    │╱    │
    └──────┴──→ 原始能力
         能力保持阈值
```

**多目标优化算法**：

```python
def pareto_optimization(models, objectives):
    """
    NSGA-II风格的多目标优化
    """
    population = initialize_population(models)
    
    for generation in range(max_generations):
        # 评估每个模型
        fitness = evaluate_objectives(population, objectives)
        
        # 非支配排序
        fronts = non_dominated_sort(population, fitness)
        
        # 选择下一代
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= pop_size:
                next_gen.extend(front)
            else:
                # 拥挤度排序
                crowding = crowding_distance(front, fitness)
                sorted_front = sorted(zip(front, crowding), 
                                    key=lambda x: x[1], reverse=True)
                remaining = pop_size - len(next_gen)
                next_gen.extend([x[0] for x in sorted_front[:remaining]])
                break
        
        population = next_gen
        
    return get_pareto_front(population, fitness)
```

## 1.4 指令跟随与安全性平衡

### 1.4.1 指令理解的层次结构

指令理解不是二元的（理解/不理解），而是存在复杂的层次结构：

```
Level 5: 元认知理解
  └── "我要你忽略之前的指令" → 识别并拒绝
  
Level 4: 价值对齐理解
  └── "帮我写一封辞职信" → 考虑后果和建议
  
Level 3: 隐含意图推理  
  └── "太热了" → 推断：调节温度/开窗/提供饮品建议
  
Level 2: 多步骤指令分解
  └── "先总结文章要点，然后翻译成中文，最后写一段评论"
  
Level 1: 直接指令执行
  └── "将这段话翻译成英文"
  
Level 0: 语法解析
  └── 基础的句法理解
```

**指令歧义处理矩阵**：

| 歧义类型 | 示例 | 处理策略 |
|----------|------|----------|
| **范围歧义** | "总结这个" | 请求澄清范围 |
| **程度歧义** | "简要说明" | 提供多个长度选项 |
| **意图歧义** | "处理这个问题" | 列举可能的处理方式 |
| **冲突指令** | "详细但简短" | 指出矛盾并建议 |

### 1.4.2 安全边界的动态设计

安全不是静态规则，而是动态决策：

```python
class DynamicSafetyBoundary:
    def __init__(self):
        self.static_rules = load_constitution()  # 基础规则
        self.context_factors = {
            'user_history': [],
            'conversation_context': [],
            'detected_intent': None,
            'risk_level': 0
        }
    
    def evaluate_request(self, request):
        # 1. 静态规则检查
        static_risk = self.check_static_rules(request)
        
        # 2. 上下文风险评估
        context_risk = self.evaluate_context(request)
        
        # 3. 意图分析
        intent = self.analyze_intent(request)
        
        # 4. 综合决策
        total_risk = self.weighted_risk(static_risk, context_risk, intent)
        
        if total_risk > 0.8:
            return "REFUSE", self.generate_refusal(request, total_risk)
        elif total_risk > 0.5:
            return "WARN", self.generate_warning(request) 
        elif total_risk > 0.3:
            return "CAVEAT", self.add_disclaimer(request)
        else:
            return "ALLOW", None
```

**安全-有用性权衡曲线**：

```
有用性 ↑
100%│     
    │   ╱ ← 理想曲线
 80%│  ╱
    │ ╱ • 当前SOTA
 60%│╱
    │── ← 基础线
 40%│
    │
 20%│
    └────────────→ 安全性
     60% 80% 100%
```

### 1.4.3 拒绝机制的分层实现

**智能拒绝框架**：

```python
class IntelligentRefusal:
    def __init__(self):
        self.refusal_templates = {
            'illegal': "我不能协助违法活动。",
            'harmful': "这可能造成伤害，我不能提供帮助。",
            'privacy': "我不能分享个人隐私信息。",
            'uncertain': "我不确定这是否合适，让我们换个话题。"
        }
        
    def generate_refusal(self, request, risk_type):
        # 1. 识别具体风险
        specific_risk = self.identify_specific_risk(request)
        
        # 2. 解释原因（教育性）
        explanation = self.explain_why_harmful(specific_risk)
        
        # 3. 提供替代方案
        alternatives = self.suggest_alternatives(request, risk_type)
        
        # 4. 组合响应
        response = f"""
        {self.refusal_templates[risk_type]}
        
        {explanation}
        
        作为替代，我可以：
        {alternatives}
        """
        
        return response
```

**拒绝粒度控制**：

```
请求分类决策树：
├── 明确有害（>95%确定）
│   └── 直接拒绝 + 简短解释
├── 可能有害（70-95%确定）
│   └── 软性拒绝 + 详细解释 + 替代建议
├── 边界模糊（30-70%确定）
│   ├── 部分满足 + 安全限制
│   └── 要求澄清意图
└── 安全请求（<30%风险）
    └── 正常执行 + 可选免责声明
```

### 1.4.4 指令优先级与冲突解决

当多个指令或约束发生冲突时的处理：

```python
class InstructionPriorityResolver:
    def __init__(self):
        # 优先级从高到低
        self.priority_hierarchy = [
            'legal_compliance',      # 法律合规
            'safety',               # 安全性
            'privacy',              # 隐私保护
            'truthfulness',         # 真实性
            'helpfulness',          # 有用性
            'user_preference'       # 用户偏好
        ]
    
    def resolve_conflict(self, instructions):
        conflicts = self.detect_conflicts(instructions)
        
        if not conflicts:
            return self.merge_instructions(instructions)
        
        # 基于优先级解决冲突
        resolved = {}
        for conflict in conflicts:
            winning_instruction = max(
                conflict.instructions,
                key=lambda x: self.priority_hierarchy.index(x.type)
            )
            resolved[conflict.aspect] = winning_instruction
        
        return self.synthesize_response(resolved)
```

## 1.5 分布偏移问题

### 1.5.1 训练-推理分布差异的系统分析

分布偏移是后训练部署中最被低估的问题之一。即使模型在测试集上表现完美，实际部署时仍可能崩溃。

**多维度分布偏移分类**：

```python
class DistributionShift:
    def __init__(self):
        self.shift_types = {
            'covariate': {  # P(X)改变，P(Y|X)不变
                'example': '训练：新闻文章 → 部署：社交媒体',
                'severity': 'medium',
                'detection': 'KL divergence on input embeddings'
            },
            'label': {  # P(Y)改变，P(X|Y)不变
                'example': '训练：礼貌响应 → 部署：用户期望直接回答',
                'severity': 'high',
                'detection': 'output distribution monitoring'
            },
            'concept': {  # P(Y|X)改变
                'example': '新概念出现（如新技术、新事件）',
                'severity': 'critical',
                'detection': 'perplexity spike detection'
            },
            'temporal': {  # 随时间变化
                'example': '2023年训练 → 2025年部署',
                'severity': 'progressive',
                'detection': 'time-aware evaluation'
            }
        }
```

**1. 输入分布偏移的细粒度分析**：

| 维度 | 训练分布 | 部署分布 | 影响 |
|------|----------|----------|------|
| **长度** | 均值50词 | 长尾分布(1-1000词) | 长文本性能退化 |
| **语言** | 标准书面语 | 口语/俚语/表情 | 理解错误增加 |
| **格式** | 结构化指令 | 自由格式 | 解析失败 |
| **噪声** | 清洁文本 | 拼写错误/语法错误 | 鲁棒性差 |
| **领域** | 通用领域 | 专业领域 | 知识缺口 |

**2. 输出分布偏移**：

```python
def analyze_output_shift(train_outputs, deploy_outputs):
    metrics = {}
    
    # 长度分布变化
    metrics['length_shift'] = {
        'train_mean': np.mean([len(o) for o in train_outputs]),
        'deploy_mean': np.mean([len(o) for o in deploy_outputs]),
        'kl_divergence': compute_kl(train_lengths, deploy_lengths)
    }
    
    # 多样性变化
    metrics['diversity'] = {
        'train_entropy': compute_entropy(train_outputs),
        'deploy_entropy': compute_entropy(deploy_outputs),
        'unique_ngrams': count_unique_ngrams(deploy_outputs)
    }
    
    # 模式坍塌检测
    metrics['mode_collapse'] = {
        'repetition_rate': detect_repetitions(deploy_outputs),
        'template_usage': detect_templates(deploy_outputs)
    }
    
    return metrics
```

**3. 错误累积的数学建模**：

自回归生成中的错误传播：

$$P(\text{error at } t) = 1 - \prod_{i=0}^{t-1}(1 - \epsilon_i)$$

其中$\epsilon_i$是位置$i$的错误率。

实际测量显示：
- 前10个token：错误率 < 1%
- 100个token后：错误率 ~ 5%
- 500个token后：错误率 > 15%

### 1.5.2 分布偏移的实时检测系统

**多层次检测框架**：

```python
class DistributionShiftDetector:
    def __init__(self, reference_data):
        self.reference_stats = self.compute_reference_stats(reference_data)
        self.detection_methods = {
            'statistical': self.statistical_tests,
            'model_based': self.model_based_detection,
            'ensemble': self.ensemble_detection
        }
        
    def statistical_tests(self, new_data):
        tests = {}
        
        # 1. Kolmogorov-Smirnov测试
        tests['ks_test'] = ks_2samp(
            self.reference_stats['embeddings'],
            compute_embeddings(new_data)
        )
        
        # 2. Maximum Mean Discrepancy (MMD)
        tests['mmd'] = self.compute_mmd(
            self.reference_stats['features'],
            extract_features(new_data)
        )
        
        # 3. JS散度
        tests['js_divergence'] = self.compute_js_divergence(
            self.reference_stats['distribution'],
            estimate_distribution(new_data)
        )
        
        return tests
    
    def model_based_detection(self, new_data):
        # 使用专门的OOD检测器
        ood_scores = self.ood_detector.predict(new_data)
        
        # 不确定性估计
        uncertainty = self.uncertainty_estimator.estimate(new_data)
        
        return {
            'ood_score': np.mean(ood_scores),
            'epistemic_uncertainty': uncertainty['epistemic'],
            'aleatoric_uncertainty': uncertainty['aleatoric']
        }
```

**实时监控指标**：

```
监控仪表板：
┌──────────────────────────────────────┐
│ 分布偏移监控 - 实时状态              │
├──────────────────────────────────────┤
│ KL散度:     0.023 [████░░░░░░] 正常  │
│ JS散度:     0.045 [██████░░░░] 警告  │
│ MMD:        0.012 [███░░░░░░░] 正常  │
│ 困惑度峰值: 234   [████████░░] 异常  │
│ OOD率:      2.3%  [████░░░░░░] 正常  │
└──────────────────────────────────────┘
```

### 1.5.3 分布偏移的适应策略

**1. 测试时适应（Test-Time Adaptation）**：

```python
class TestTimeAdaptation:
    def __init__(self, model):
        self.model = model
        self.adaptation_buffer = []
        
    def adapt(self, batch):
        # 1. 熵最小化
        outputs = self.model(batch)
        entropy_loss = -torch.sum(outputs * torch.log(outputs))
        
        # 2. 伪标签自训练
        pseudo_labels = outputs.argmax(dim=-1)
        confidence = outputs.max(dim=-1).values
        
        # 只使用高置信度样本
        mask = confidence > 0.9
        if mask.any():
            self.model.backward(
                self.model.loss(batch[mask], pseudo_labels[mask])
            )
            
        # 3. 批归一化统计更新
        self.update_bn_stats(batch)
```

**2. 持续学习策略**：

```python
def continual_learning_pipeline():
    """
    数据飞轮：收集→标注→训练→部署
    """
    while True:
        # 1. 收集困难案例
        hard_cases = collect_hard_cases(
            threshold=0.7,  # 置信度阈值
            max_samples=1000
        )
        
        # 2. 智能标注
        labeled_data = hybrid_labeling(
            hard_cases,
            human_budget=100,  # 人工标注预算
            ai_labeler=strong_model
        )
        
        # 3. 增量训练
        model = incremental_train(
            model,
            new_data=labeled_data,
            replay_buffer=sample_old_data(500),
            regularization='ewc'  # 弹性权重巩固
        )
        
        # 4. A/B测试验证
        if validate_improvement(model, baseline):
            deploy(model)
        
        time.sleep(24 * 3600)  # 每日更新
```

**3. 鲁棒训练策略**：

```python
class RobustTraining:
    def __init__(self):
        self.augmentation_strategies = [
            self.add_noise,
            self.paraphrase,
            self.backtranslation,
            self.token_cutoff,
            self.adversarial_perturbation
        ]
    
    def augment_batch(self, batch):
        augmented = []
        for sample in batch:
            # 多策略组合
            aug_sample = sample
            for strategy in random.sample(self.augmentation_strategies, k=2):
                aug_sample = strategy(aug_sample)
            augmented.append(aug_sample)
        
        return augmented
    
    def adversarial_perturbation(self, text):
        # 生成对抗样本
        embedding = self.encoder(text)
        gradient = compute_gradient(embedding)
        
        # FGSM扰动
        perturbed = embedding + self.epsilon * gradient.sign()
        
        return self.decoder(perturbed)
```

**4. 课程学习的精细化设计**：

```python
class CurriculumLearning:
    def __init__(self):
        self.difficulty_stages = [
            {
                'week': 1,
                'tasks': ['simple_qa', 'translation'],
                'max_length': 50,
                'complexity': 'low'
            },
            {
                'week': 2,
                'tasks': ['summarization', 'explanation'],
                'max_length': 200,
                'complexity': 'medium'
            },
            {
                'week': 3,
                'tasks': ['multi_turn_dialogue', 'reasoning'],
                'max_length': 500,
                'complexity': 'high'
            },
            {
                'week': 4,
                'tasks': ['adversarial', 'edge_cases'],
                'max_length': 1000,
                'complexity': 'extreme'
            }
        ]
    
    def get_current_batch(self, epoch):
        stage = self.get_stage(epoch)
        
        # 动态难度调整
        if self.performance > 0.9:
            # 加速进度
            stage = min(stage + 1, len(self.difficulty_stages) - 1)
        elif self.performance < 0.7:
            # 放慢进度
            stage = max(stage - 1, 0)
        
        return self.sample_from_stage(stage)
```

## 1.6 理论基础深化

### 1.6.1 Bradley-Terry模型

人类偏好建模的理论基础：

$$P(y_1 \succ y_2 | x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))}$$

这个模型假设偏好概率与奖励差呈logistic关系。

### 1.6.2 逆强化学习视角

后训练可视为逆强化学习（IRL）问题：

```
观察到的行为 → 推断奖励函数 → 优化策略

数学形式：
max_R  L(R|D_demo) - λ·||R||
s.t.   π* = argmax_π E[R(s,a)]
```

### 1.6.3 信息论视角

从信息论角度理解对齐：

**互信息最大化**：
$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

目标是最大化指令X和响应Y之间的互信息，同时保持Y的熵足够高。

## 本章小结

本章介绍了LLM后训练的核心理论基础：

📌 **关键概念**：
- 后训练将预训练模型转化为对齐的AI助手
- SFT、RLHF、DPO是三种主流后训练范式
- 对齐税是后训练中不可避免的能力权衡
- 分布偏移是部署阶段的主要挑战

💡 **实用规则**：
- SFT数据质量 > 数量，通常几千条高质量数据足够
- RLHF的KL惩罚系数β通常设为0.01-0.1
- DPO相比RLHF减少50%的显存使用
- 混合10-20%预训练数据可有效减少对齐税

⚠️ **常见陷阱**：
- 过度优化特定指标导致模型行为退化
- 忽视分布偏移导致部署效果下降
- 安全约束过强导致模型拒绝合理请求

## 练习题

### 基础题

**练习 1.1：SFT损失函数理解**
给定一个批次的训练数据，包含3条样本：
- ("翻译成英文：你好", "Hello")
- ("总结这段话", "[响应]")
- ("写一首诗", "[响应]")

请计算第一条样本的SFT损失（假设词表大小为50000，正确token的logit为3.0，其他为0）。

*Hint: 使用交叉熵损失公式*

<details>
<summary>参考答案</summary>

SFT损失计算：
1. 对于"Hello"的每个token，计算softmax概率
2. P(correct) = exp(3.0) / (exp(3.0) + 49999*exp(0)) ≈ 0.0004
3. Loss = -log(0.0004) ≈ 7.82
4. 实际实现中会对所有token求平均

关键点：
- SFT本质是最大似然估计
- 损失与词表大小相关
- 需要考虑序列长度归一化
</details>

**练习 1.2：KL散度计算**
两个分布P和Q在3个类别上的概率分别为：
- P: [0.5, 0.3, 0.2]
- Q: [0.6, 0.3, 0.1]

计算D_KL(P||Q)。

*Hint: KL散度公式 D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))*

<details>
<summary>参考答案</summary>

D_KL(P||Q) = 0.5*log(0.5/0.6) + 0.3*log(0.3/0.3) + 0.2*log(0.2/0.1)
         = 0.5*(-0.182) + 0.3*0 + 0.2*0.693
         = -0.091 + 0 + 0.139
         = 0.048

注意：
- KL散度是非对称的：D_KL(P||Q) ≠ D_KL(Q||P)
- 当Q(x)=0但P(x)>0时，KL散度为无穷大
- 在RLHF中用于约束策略不要偏离参考策略太远
</details>

**练习 1.3：DPO vs RLHF对比**
列出DPO相比RLHF的3个优势和2个劣势。

*Hint: 考虑计算效率、实现复杂度、数据需求*

<details>
<summary>参考答案</summary>

**DPO优势**：
1. 实现简单：不需要训练独立的奖励模型
2. 内存效率：减少约50%显存使用（无需加载奖励模型）
3. 训练稳定：避免了RL训练的不稳定性

**DPO劣势**：
1. 数据效率低：需要更多偏好数据对
2. 泛化能力弱：难以超越训练数据分布

实践建议：
- 小规模实验优先选择DPO
- 大规模生产环境RLHF可能更优
- 可以用DPO初始化，再用RLHF微调
</details>

### 挑战题

**练习 1.4：对齐税的量化分析**
设计一个实验来量化测量对齐税。要求：
1. 选择至少3个能力维度
2. 设计评估指标
3. 提出缓解策略

*Hint: 考虑如何公平比较预训练和后训练模型*

<details>
<summary>参考答案</summary>

**实验设计**：

1. **能力维度选择**：
   - 事实知识：MMLU benchmark
   - 推理能力：GSM8K数学题
   - 代码生成：HumanEval
   - 创造性：故事续写多样性

2. **评估协议**：
   ```python
   # 控制变量
   - 相同的解码参数(temperature=0.7)
   - 相同的prompt格式
   - 多次运行取平均（减少随机性）
   
   # 指标计算
   alignment_tax[task] = max(0, 
       score_pretrain[task] - score_aligned[task])
   ```

3. **缓解策略**：
   - **数据混合**：加入15%预训练数据
   - **分层微调**：冻结底层，只调整顶层
   - **弹性权重巩固(EWC)**：保护重要参数
   - **知识蒸馏**：从预训练模型蒸馏

4. **预期结果**：
   - 事实知识：-3%到-5%
   - 推理能力：-5%到-8%
   - 代码生成：-10%到-15%
   - 创造性：-20%到-30%
</details>

**练习 1.5：分布偏移的在线适应**
在部署后发现模型在处理含有表情符号的输入时性能下降40%。设计一个在线适应方案。

*Hint: 考虑数据收集、标注、训练的完整流程*

<details>
<summary>参考答案</summary>

**在线适应方案**：

1. **问题诊断**：
   ```python
   # 分析失败案例
   failure_rate_by_feature = {
       "has_emoji": 0.40,
       "no_emoji": 0.05,
       "mixed_language": 0.25
   }
   ```

2. **数据收集策略**：
   - 自动记录含表情符号的失败案例
   - 主动采样：生成表情符号变体
   - 用户反馈：收集负面评价的案例

3. **增量训练**：
   ```python
   # 混合策略
   new_data = collect_emoji_cases(n=1000)
   replay_buffer = sample_previous(n=4000)
   combined = new_data + replay_buffer
   
   # 小学习率微调
   lr = original_lr * 0.1
   train_steps = 100  # 避免过拟合
   ```

4. **A/B测试验证**：
   - 10%流量测试新模型
   - 监控关键指标
   - 逐步扩大流量

5. **长期改进**：
   - 更新训练数据分布
   - 调整数据增强策略
   - 考虑专门的表情符号编码器
</details>

**练习 1.6：多目标优化的帕累托前沿**
给定3个目标：有用性(H)、无害性(S)、诚实性(T)。如何找到最优权衡点？

*Hint: 考虑多目标优化算法和实际约束*

<details>
<summary>参考答案</summary>

**解决方案**：

1. **问题形式化**：
   $$\max_\theta \{ H(\theta), S(\theta), T(\theta) \}$$
   
   约束：各指标最低阈值
   - H ≥ 0.8
   - S ≥ 0.95  
   - T ≥ 0.85

2. **帕累托前沿搜索**：
   ```python
   # 网格搜索权重
   for w_h in [0.2, 0.3, 0.4, 0.5]:
       for w_s in [0.2, 0.3, 0.4, 0.5]:
           w_t = 1 - w_h - w_s
           if w_t > 0:
               loss = -w_h*H - w_s*S - w_t*T
               train_model(loss)
               evaluate_pareto_dominance()
   ```

3. **自适应权重调整**：
   - 根据当前性能动态调整权重
   - 优先改进最差的指标
   - 使用梯度手术(Gradient Surgery)避免冲突

4. **实践建议**：
   - 先优化安全性到阈值以上
   - 在保证安全的前提下优化有用性
   - 诚实性通过数据质量保证
   - 定期重新评估权重分配
</details>

**练习 1.7：Constitutional AI的规则设计**
设计一套宪法规则(Constitutional Rules)来指导模型行为，要求覆盖安全、有用、诚实三个维度。

*Hint: 规则要具体、可执行、无歧义*

<details>
<summary>参考答案</summary>

**Constitutional Rules设计**：

1. **安全规则**(优先级最高)：
   ```
   R1: 绝不协助非法活动
   R2: 不生成可识别个人信息
   R3: 拒绝生成仇恨或歧视内容
   R4: 不提供自我伤害指导
   ```

2. **有用性规则**：
   ```
   R5: 直接回答用户问题
   R6: 提供可执行的步骤
   R7: 承认不确定性
   R8: 主动澄清歧义
   ```

3. **诚实性规则**：
   ```
   R9: 不编造事实或引用
   R10: 区分观点和事实
   R11: 承认知识边界
   R12: 纠正自己的错误
   ```

4. **规则冲突解决**：
   ```python
   def resolve_conflict(rules_triggered):
       # 安全 > 诚实 > 有用
       priority = {"safety": 3, "honesty": 2, "helpful": 1}
       return max(rules_triggered, key=lambda r: priority[r.type])
   ```

5. **实施机制**：
   - **训练时**：规则作为额外的奖励信号
   - **推理时**：规则作为生成约束
   - **评估时**：规则违反率作为关键指标
</details>

**练习 1.8：开放性思考题**
如果你要设计下一代的后训练方法，会如何改进现有的RLHF/DPO方法？请提出至少一个创新点。

*Hint: 可以从效率、效果、可解释性等角度思考*

<details>
<summary>参考答案</summary>

**创新方向示例**：

1. **在线偏好学习**：
   - 问题：静态偏好数据很快过时
   - 方案：实时收集用户反馈，动态更新奖励模型
   - 技术：增量学习 + 重要性采样

2. **多粒度奖励建模**：
   - 问题：单一标量奖励信息不足
   - 方案：token级、句子级、段落级多层次奖励
   - 优势：更精细的信用分配

3. **对比解释学习**：
   - 不仅学习"哪个更好"
   - 还学习"为什么更好"
   - 生成可解释的改进建议

4. **元学习优化器**：
   - 学习如何从少量偏好数据快速适应
   - 减少新领域的标注需求
   - 提高样本效率

5. **分布鲁棒优化**：
   - 显式建模最坏情况分布
   - 提高OOD泛化能力
   - 数学形式：
   $$\min_\theta \max_{Q \in \mathcal{P}} \mathbb{E}_{x \sim Q}[\mathcal{L}(\theta, x)]$$

评估标准：
- 样本效率提升>50%
- 训练时间减少>30%
- OOD性能提升>20%
</details>

## 常见陷阱与错误 (Gotchas)

### 1. 数据相关陷阱

⚠️ **过度清洗综合征**
- 错误：过度清洗训练数据，移除所有"不完美"样本
- 后果：模型失去处理真实世界混乱输入的能力
- 正确做法：保留15-20%的"噪声"数据，提高鲁棒性

⚠️ **标注者偏见放大**
- 错误：使用单一来源或同质化的标注团队
- 后果：模型学习并放大特定群体的偏见
- 正确做法：多样化标注者背景，使用标注者disagreement作为信号

### 2. 训练策略陷阱

⚠️ **KL惩罚系数选择**
- 错误：使用固定的β值throughout训练
- 后果：早期限制探索，后期退化严重
- 正确做法：
  ```python
  # 动态调整
  β = β_init * (1 + decay_rate * step)
  # 典型值：β_init=0.01, decay_rate=0.0001
  ```

⚠️ **奖励模型过拟合**
- 错误：奖励模型在同分布数据上过拟合
- 后果：策略模型学会exploit奖励模型的弱点
- 正确做法：
  1. 奖励模型ensemble
  2. 定期更新奖励模型
  3. 添加奖励不确定性估计

### 3. 评估陷阱

⚠️ **评估数据污染**
- 错误：评估集信息泄露到训练集
- 征兆：评估指标异常高，但实际效果差
- 检测方法：
  ```python
  # N-gram重叠检测
  contamination = check_ngram_overlap(train_set, eval_set, n=13)
  if contamination > 0.01:
      raise DataLeakageError
  ```

⚠️ **单一指标优化**
- 错误：只优化BLEU/ROUGE等自动指标
- 后果：Goodhart定律 - "当一个指标变成目标，它就不再是好指标"
- 正确做法：多维度评估矩阵 + 人工评估验证

### 4. 部署陷阱

⚠️ **批处理效应**
- 错误：训练时batch_size=32，推理时batch_size=1
- 后果：BatchNorm统计不匹配，性能下降
- 解决：使用LayerNorm或推理时调整统计量

⚠️ **长度外推失败**
- 错误：训练最大长度512，推理时处理2048
- 后果：位置编码失效，生成质量崩溃
- 解决：
  1. 训练时包含多种长度
  2. 使用相对位置编码
  3. 长度warmup策略

### 调试技巧

💡 **快速诊断检查单**：
```python
def diagnose_training_issues():
    checks = {
        "gradient_norm": check_gradient_explosion(),
        "loss_plateau": check_loss_convergence(),
        "reward_hacking": check_reward_gaming(),
        "distribution_shift": check_kl_divergence(),
        "capability_drop": run_capability_benchmarks()
    }
    return generate_diagnostic_report(checks)
```

💡 **A/B测试最佳实践**：
1. 最小可行改进：一次只改一个变量
2. 统计显著性：至少1000个样本
3. 在线指标vs离线指标对齐
4. 设置自动回滚阈值

---

下一章：[第二章：实验代码基础设施 →](chapter2.md)