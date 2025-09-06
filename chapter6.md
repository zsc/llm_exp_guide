# 第六章：强化学习与人类反馈

本章深入探讨基于人类反馈的强化学习（RLHF）及其变体在大语言模型后训练中的应用。我们将从奖励模型的构建开始，详细分析PPO、DPO等主流算法的实现细节，探讨Constitutional AI等自我改进方法，并讨论在线与离线强化学习的权衡。通过本章学习，您将掌握设计和实施RLHF系统的完整方法论，理解不同算法的适用场景，以及避免常见的实验陷阱。

## 6.1 RLHF的动机与核心挑战

### 6.1.1 为什么需要RLHF

监督微调（SFT）虽然能让模型学会遵循指令的基本格式，但存在几个根本性限制：

1. **行为模仿的局限性**：SFT本质上是让模型模仿训练数据中的行为模式。即使有高质量的示范数据，模型也只能学到"如何说"，而非真正理解"为什么这样说更好"。

2. **偏好的隐式性**：人类偏好往往是隐式的、多维的，很难通过示例完全表达。比如"有帮助"这个概念，包含准确性、完整性、清晰度等多个维度，且在不同上下文中权重不同。

3. **分布偏移问题**：SFT模型在生成时会累积误差，逐渐偏离训练分布。而RLHF通过在模型自己的生成分布上训练，能更好地处理这种偏移。

### 6.1.2 RLHF的核心组件

```
    Human Preferences
           ↓
    ┌──────────────┐
    │ Reward Model │ ← 偏好数据训练
    └──────────────┘
           ↓
       奖励信号
           ↓
    ┌──────────────┐
    │  RL Training │ ← PPO/DPO等算法
    └──────────────┘
           ↓
     Aligned Model
```

RLHF系统包含三个核心组件：

1. **偏好数据收集**：获取人类对不同回复的相对偏好判断
2. **奖励模型训练**：学习将文本映射到标量奖励值
3. **策略优化**：使用RL算法优化语言模型以最大化期望奖励

### 6.1.3 主要挑战

**挑战1：奖励过拟合（Reward Hacking）**

模型可能找到获得高奖励但实际质量差的捷径。例如：
- 过度使用奖励模型偏好的特定短语
- 生成看似完整但实际空洞的长回复
- 利用奖励模型的盲点生成有问题的内容

**挑战2：训练不稳定性**

RLHF训练过程容易出现：
- 策略崩溃：模型退化到重复简单模式
- 奖励爆炸：优化过程失控导致奖励值异常
- KL散度失控：生成分布过度偏离初始模型

**挑战3：评估困难**

- 奖励值不能完全代表真实质量
- 需要大量人工评估验证改进效果
- 不同评估指标可能相互冲突

## 6.2 奖励模型的训练与校准

### 6.2.1 Bradley-Terry偏好模型

奖励模型的理论基础是Bradley-Terry模型，它假设人类选择回复A优于回复B的概率为：

$$P(A \succ B) = \frac{\exp(r(A))}{\exp(r(A)) + \exp(r(B))} = \sigma(r(A) - r(B))$$

其中$r(\cdot)$是奖励函数，$\sigma$是sigmoid函数。

训练目标是最大化对数似然：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log\sigma(r_\theta(x,y_w) - r_\theta(x,y_l))\right]$$

其中$y_w$是被偏好的回复，$y_l$是较差的回复。

### 6.2.2 奖励模型架构设计

典型的奖励模型架构：

```
输入: [prompt] + [response]
  ↓
Transformer Encoder (预训练LM)
  ↓
最后一个token的隐状态
  ↓
Linear Head → 标量奖励值
```

**关键设计选择：**

1. **基座模型选择**：
   - 使用与策略模型相同规模的基座（如7B对7B）
   - 或使用更大的模型（如13B奖励模型指导7B策略）
   
2. **池化策略**：
   - 最后token池化（最常用）
   - 平均池化（对长文本更稳定）
   - 加权池化（考虑token重要性）

3. **归一化方案**：
   ```python
   # 每批次标准化
   rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
   
   # 或使用运行均值/方差
   self.running_mean = 0.99 * self.running_mean + 0.01 * batch_mean
   ```

### 6.2.3 训练技巧与过拟合预防

**技巧1：数据增强**

```python
def augment_preference_data(prompt, chosen, rejected):
    # 1. 顺序随机化
    if random.random() < 0.5:
        return prompt, rejected, chosen, -1  # 标签翻转
    
    # 2. 边际案例生成
    if similarity(chosen, rejected) > 0.9:
        # 为高度相似的对添加噪声
        rejected = add_noise(rejected)
    
    return prompt, chosen, rejected, 1
```

**技巧2：集成与不确定性估计**

训练多个奖励模型并使用集成：

```python
class EnsembleRewardModel:
    def __init__(self, models):
        self.models = models
    
    def predict(self, prompt, response):
        rewards = [m(prompt, response) for m in self.models]
        mean_reward = np.mean(rewards)
        uncertainty = np.std(rewards)
        
        # 高不确定性时降低奖励置信度
        if uncertainty > threshold:
            mean_reward *= 0.8
        
        return mean_reward, uncertainty
```

**技巧3：对抗验证**

定期用对抗样本测试奖励模型：

```python
def generate_adversarial_samples(reward_model, base_model):
    # 生成高奖励但质量差的样本
    prompt = "解释量子力学"
    
    # 策略1：重复关键词
    bad_response_1 = "量子力学量子力学..." * 100
    
    # 策略2：空洞的长回复
    bad_response_2 = generate_verbose_but_empty(prompt)
    
    # 检查奖励模型是否被欺骗
    if reward_model(prompt, bad_response_1) > threshold:
        log.warning("奖励模型对重复内容给出高分")
```

### 6.2.4 校准技术

**温度缩放（Temperature Scaling）**

```python
class CalibratedRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, prompt, response):
        logits = self.base_model(prompt, response)
        return logits / self.temperature
    
    def calibrate(self, val_data):
        # 在验证集上优化温度参数
        optimizer = torch.optim.LBFGS([self.temperature])
        
        def closure():
            loss = 0
            for prompt, chosen, rejected in val_data:
                prob = torch.sigmoid(
                    (self(prompt, chosen) - self(prompt, rejected)) 
                )
                loss -= torch.log(prob)
            return loss
        
        optimizer.step(closure)
```

**期望校准误差（ECE）监控**

```python
def compute_ece(predictions, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & \
               (predictions < bin_boundaries[i+1])
        
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = predictions[mask].mean()
            bin_weight = mask.sum() / len(predictions)
            
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece
```

## 6.3 PPO在LLM中的实现细节

### 6.3.1 PPO算法核心

PPO（Proximal Policy Optimization）通过限制每次更新的幅度来保证训练稳定性：

$$\mathcal{L}_{PPO} = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率
- $\hat{A}_t$ 是优势函数估计
- $\epsilon$ 是裁剪参数（通常0.1-0.2）

### 6.3.2 LLM特定的实现细节

**挑战1：序列生成的信用分配**

在LLM中，一个"动作"是生成一个token，"轨迹"是完整的回复。奖励通常只在序列末尾给出，需要合理的信用分配：

```python
def compute_advantages(rewards, values, gamma=1.0, lam=0.95):
    """
    计算广义优势估计(GAE)
    rewards: [batch_size, seq_len] 通常只有最后一个非零
    values: [batch_size, seq_len] 价值函数预测
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(len(rewards[0]))):
        if t == len(rewards[0]) - 1:
            next_values = 0  # 终止状态
        else:
            next_values = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        advantages[:, t] = lastgaelam = delta + gamma * lam * lastgaelam
    
    return advantages
```

**挑战2：KL散度约束**

防止策略偏离太远：

```python
def compute_kl_penalty(logprobs_new, logprobs_ref, kl_coef=0.1):
    """
    计算KL散度惩罚
    logprobs_new: 当前策略的对数概率
    logprobs_ref: 参考策略（通常是SFT模型）的对数概率
    """
    kl = (logprobs_ref - logprobs_new).sum(dim=-1)
    
    # 自适应KL系数
    if kl.mean() > target_kl * 1.5:
        kl_coef *= 1.5  # 增加惩罚
    elif kl.mean() < target_kl * 0.5:
        kl_coef *= 0.5  # 减少惩罚
    
    return kl * kl_coef
```

### 6.3.3 训练循环实现

```python
class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, 
                 lr=1e-6, eps=0.2, kl_coef=0.1):
        self.policy = policy_model
        self.ref = ref_model
        self.reward = reward_model
        self.optimizer = AdamW(policy_model.parameters(), lr=lr)
        self.eps = eps
        self.kl_coef = kl_coef
        
    def train_step(self, prompts, max_length=512):
        # 1. 生成回复
        with torch.no_grad():
            responses, old_logprobs = self.generate_responses(
                prompts, max_length
            )
            
            # 2. 计算奖励
            rewards = self.reward(prompts, responses)
            
            # 3. 计算参考模型的对数概率
            ref_logprobs = self.ref.compute_logprobs(prompts, responses)
        
        # 4. 多轮PPO更新
        for _ in range(4):  # PPO epochs
            # 计算当前策略的对数概率
            new_logprobs, values = self.policy.forward_with_value(
                prompts, responses
            )
            
            # 计算优势
            advantages = compute_advantages(rewards, values)
            
            # PPO损失
            ratio = torch.exp(new_logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # KL惩罚
            kl_loss = compute_kl_penalty(
                new_logprobs, ref_logprobs, self.kl_coef
            )
            
            # 价值函数损失
            value_loss = F.mse_loss(values, rewards + values.detach())
            
            # 总损失
            loss = policy_loss + kl_loss.mean() + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
```

### 6.3.4 PPO调试技巧

**技巧1：监控关键指标**

```python
def log_ppo_metrics(info):
    # 必须监控的指标
    metrics = {
        'kl_divergence': info['kl'].mean(),
        'clip_fraction': (info['ratio'] > 1.2).float().mean(),
        'approx_kl': (info['ratio'] - 1).pow(2).mean() / 2,
        'reward_mean': info['rewards'].mean(),
        'reward_std': info['rewards'].std(),
        'value_loss': info['value_loss'],
        'policy_loss': info['policy_loss'],
        'entropy': info['entropy'],  # 监控探索程度
    }
    
    # 异常检测
    if metrics['kl_divergence'] > 0.1:
        logger.warning("KL散度过大，可能导致训练不稳定")
    
    if metrics['clip_fraction'] > 0.3:
        logger.warning("裁剪比例过高，考虑减小学习率")
    
    return metrics
```

**技巧2：渐进式训练**

```python
def progressive_ppo_training(trainer, stages):
    """
    分阶段逐步增加训练难度
    """
    for stage in stages:
        # 阶段1：简单任务，大KL容忍度
        if stage == 1:
            trainer.kl_coef = 0.05
            prompts = get_simple_prompts()
            
        # 阶段2：中等难度，标准KL
        elif stage == 2:
            trainer.kl_coef = 0.1
            prompts = get_medium_prompts()
            
        # 阶段3：困难任务，严格KL
        else:
            trainer.kl_coef = 0.2
            prompts = get_hard_prompts()
        
        for step in range(stage_steps):
            trainer.train_step(prompts)
```

## 6.4 DPO与IPO的比较分析

### 6.4.1 DPO的理论基础

DPO（Direct Preference Optimization）通过重新参数化，将RLHF问题转换为监督学习问题，避免了显式训练奖励模型：

**关键洞察**：最优策略可以用封闭形式表达：

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)$$

反推奖励函数：

$$r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\log Z(x)$$

代入Bradley-Terry模型，得到DPO损失：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### 6.4.2 DPO实现细节

```python
class DPOTrainer:
    def __init__(self, model, ref_model, beta=0.1):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.optimizer = AdamW(model.parameters(), lr=5e-7)
    
    def compute_loss(self, prompts, chosen, rejected):
        # 计算策略模型的对数概率
        chosen_logps = self.model.compute_logprobs(prompts, chosen)
        rejected_logps = self.model.compute_logprobs(prompts, rejected)
        
        # 计算参考模型的对数概率
        with torch.no_grad():
            ref_chosen_logps = self.ref_model.compute_logprobs(
                prompts, chosen
            )
            ref_rejected_logps = self.ref_model.compute_logprobs(
                prompts, rejected
            )
        
        # 计算对数概率比
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        
        # DPO损失
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # 添加隐式奖励的监控
        with torch.no_grad():
            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        return loss, {
            'reward_accuracy': reward_accuracy,
            'reward_margin': reward_margin,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean()
        }
```

### 6.4.3 IPO的改进

IPO（Identity Preference Optimization）解决了DPO的一些问题：

1. **过拟合问题**：DPO倾向于让rejected样本的似然度趋近于0
2. **确定性偏好**：DPO假设偏好是确定性的，忽略了标注噪声

IPO的损失函数：

$$\mathcal{L}_{IPO} = \mathbb{E}\left[\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

```python
class IPOTrainer(DPOTrainer):
    def compute_loss(self, prompts, chosen, rejected):
        # 与DPO相同的对数概率计算
        chosen_logps = self.model.compute_logprobs(prompts, chosen)
        rejected_logps = self.model.compute_logprobs(prompts, rejected)
        
        with torch.no_grad():
            ref_chosen_logps = self.ref_model.compute_logprobs(
                prompts, chosen
            )
            ref_rejected_logps = self.ref_model.compute_logprobs(
                prompts, rejected
            )
        
        # IPO使用平方损失而非logistic损失
        log_ratio_chosen = chosen_logps - ref_chosen_logps
        log_ratio_rejected = rejected_logps - ref_rejected_logps
        
        # IPO损失：鼓励差值为1/(2β)
        losses = (log_ratio_chosen - log_ratio_rejected - 1/(2*self.beta))**2
        
        return losses.mean(), {
            'log_ratio_diff': (log_ratio_chosen - log_ratio_rejected).mean()
        }
```

### 6.4.4 DPO vs IPO实验对比

**实验设置对比表**：

| 维度 | DPO | IPO |
|------|-----|-----|
| 损失函数 | Logistic | MSE |
| β参数敏感度 | 高 | 中 |
| 训练稳定性 | 中 | 高 |
| 收敛速度 | 快 | 慢 |
| 过拟合风险 | 高 | 低 |
| 噪声鲁棒性 | 低 | 高 |

**选择指南**：

```python
def choose_optimization_method(dataset_properties):
    """
    根据数据集特性选择DPO或IPO
    """
    if dataset_properties['annotation_agreement'] < 0.7:
        # 标注一致性低，使用IPO
        return 'IPO', '标注噪声大，IPO更鲁棒'
    
    elif dataset_properties['size'] < 10000:
        # 数据量小，使用IPO避免过拟合
        return 'IPO', '数据量小，IPO泛化更好'
    
    elif dataset_properties['preference_strength'] > 0.9:
        # 偏好非常明确，使用DPO
        return 'DPO', '偏好明确，DPO收敛快'
    
    else:
        # 默认使用DPO
        return 'DPO', '标准场景，DPO效率高'
```

### 6.4.5 混合策略

```python
class HybridDPO_IPO:
    """
    结合DPO和IPO优点的混合方法
    """
    def __init__(self, model, ref_model, beta=0.1, alpha=0.5):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.alpha = alpha  # DPO和IPO的混合权重
    
    def compute_loss(self, prompts, chosen, rejected):
        # 计算对数概率
        chosen_logps = self.model.compute_logprobs(prompts, chosen)
        rejected_logps = self.model.compute_logprobs(prompts, rejected)
        
        with torch.no_grad():
            ref_chosen_logps = self.ref_model.compute_logprobs(prompts, chosen)
            ref_rejected_logps = self.ref_model.compute_logprobs(prompts, rejected)
        
        # 对数比
        log_ratio_diff = (chosen_logps - ref_chosen_logps) - \
                        (rejected_logps - ref_rejected_logps)
        
        # DPO损失
        dpo_loss = -F.logsigmoid(self.beta * log_ratio_diff).mean()
        
        # IPO损失
        ipo_loss = (log_ratio_diff - 1/(2*self.beta))**2.mean()
        
        # 混合损失
        loss = self.alpha * dpo_loss + (1 - self.alpha) * ipo_loss
        
        return loss
```

## 6.5 Constitutional AI与自我改进

### 6.5.1 Constitutional AI原理

Constitutional AI（CAI）使用一组原则来指导模型的自我改进，减少对人类标注的依赖：

```
原始回复 → AI自我批判 → 修订回复 → AI偏好判断 → 训练
```

核心组件：
1. **宪法原则**：定义模型应遵循的规则
2. **自我批判**：模型评估自己的输出
3. **自我修订**：基于批判改进回复
4. **自我偏好**：生成偏好数据用于训练

### 6.5.2 实现Constitutional AI

```python
class ConstitutionalAI:
    def __init__(self, model, principles):
        self.model = model
        self.principles = principles
    
    def critique_response(self, prompt, response):
        """
        使用宪法原则批判回复
        """
        critiques = []
        
        for principle in self.principles:
            critique_prompt = f"""
            原则：{principle}
            用户问题：{prompt}
            助手回复：{response}
            
            这个回复是否违反了上述原则？如果是，请说明如何改进。
            """
            
            critique = self.model.generate(critique_prompt)
            critiques.append(critique)
        
        return critiques
    
    def revise_response(self, prompt, response, critiques):
        """
        基于批判修订回复
        """
        revision_prompt = f"""
        原始问题：{prompt}
        原始回复：{response}
        
        批判意见：
        {' '.join(critiques)}
        
        请根据批判意见修订回复，使其更好地遵循原则。
        """
        
        revised = self.model.generate(revision_prompt)
        return revised
    
    def generate_preference_data(self, prompts):
        """
        生成自我标注的偏好数据
        """
        preference_data = []
        
        for prompt in prompts:
            # 生成初始回复
            response = self.model.generate(prompt)
            
            # 自我批判
            critiques = self.critique_response(prompt, response)
            
            # 如果有批判，生成修订版本
            if any(critiques):
                revised = self.revise_response(prompt, response, critiques)
                
                # 创建偏好对（修订版本 > 原始版本）
                preference_data.append({
                    'prompt': prompt,
                    'chosen': revised,
                    'rejected': response
                })
            
        return preference_data
```

### 6.5.3 RLAIF实践

RLAIF（RL from AI Feedback）完整流程：

```python
class RLAIFTrainer:
    def __init__(self, model, critic_model, principles):
        self.model = model
        self.critic = critic_model  # 可以是同一个模型
        self.principles = principles
        self.dpo_trainer = DPOTrainer(model, model.copy())
    
    def train_iteration(self, prompts, n_iterations=5):
        for iteration in range(n_iterations):
            print(f"RLAIF迭代 {iteration + 1}")
            
            # 1. 生成回复
            responses = []
            for prompt in prompts:
                response = self.model.generate(prompt)
                responses.append(response)
            
            # 2. AI评分和排序
            scored_responses = self.score_responses(prompts, responses)
            
            # 3. 构建偏好数据
            preference_data = self.create_preferences(scored_responses)
            
            # 4. DPO训练
            for batch in preference_data:
                loss = self.dpo_trainer.train_step(batch)
            
            # 5. 评估改进
            improvement = self.evaluate_improvement(prompts)
            print(f"改进幅度: {improvement:.2%}")
            
            if improvement < 0.01:  # 收敛
                break
    
    def score_responses(self, prompts, responses):
        """
        使用AI评分器给回复打分
        """
        scores = []
        
        for prompt, response in zip(prompts, responses):
            score_prompt = f"""
            根据以下原则评分（1-10分）：
            {self.principles}
            
            问题：{prompt}
            回复：{response}
            
            评分（只返回数字）：
            """
            
            score = float(self.critic.generate(score_prompt))
            scores.append(score)
        
        return list(zip(prompts, responses, scores))
```

### 6.5.4 原则设计最佳实践

**原则层次结构**：

```python
CONSTITUTIONAL_PRINCIPLES = {
    # 第一层：安全性原则（最高优先级）
    'safety': [
        "不提供可能造成伤害的信息",
        "避免生成歧视性内容",
        "保护用户隐私"
    ],
    
    # 第二层：有用性原则
    'helpfulness': [
        "提供准确的信息",
        "回答要切中要点",
        "承认不确定性"
    ],
    
    # 第三层：风格原则
    'style': [
        "保持专业语气",
        "避免过度自信",
        "适当使用例子"
    ]
}

def apply_principles_hierarchically(response, principles):
    """
    分层应用原则，高优先级原则可覆盖低优先级
    """
    for level in ['safety', 'helpfulness', 'style']:
        for principle in principles[level]:
            if violates_principle(response, principle):
                if level == 'safety':
                    # 安全问题必须修正
                    return revise_for_safety(response, principle)
                else:
                    # 其他问题尝试修正但不强制
                    response = soft_revise(response, principle)
    
    return response
```

## 6.6 在线与离线强化学习的权衡

### 6.6.1 在线vs离线RL的本质区别

**在线RL**：模型在训练过程中不断与环境交互，生成新数据并立即从中学习。

**离线RL**：仅使用预先收集的固定数据集进行训练，不与环境实时交互。

```
在线RL流程：
策略 → 生成 → 奖励 → 更新 → 策略（循环）

离线RL流程：
固定数据集 → 训练 → 策略（一次性）
```

### 6.6.2 在线RL的优势与挑战

**优势**：
1. **探索能力强**：能主动探索高奖励区域
2. **适应性好**：快速适应奖励函数变化
3. **无分布偏移**：在自己的生成分布上训练

**挑战**：
1. **计算成本高**：需要实时生成和评估
2. **不稳定性**：容易出现策略崩溃
3. **安全风险**：可能生成有害内容

**在线PPO实现**：

```python
class OnlinePPO:
    def __init__(self, policy, reward_model, buffer_size=1000):
        self.policy = policy
        self.reward_model = reward_model
        self.buffer = []
        self.buffer_size = buffer_size
    
    def collect_trajectories(self, prompts, n_samples=4):
        """
        实时收集轨迹数据
        """
        trajectories = []
        
        for prompt in prompts:
            for _ in range(n_samples):
                # 在线生成
                response = self.policy.generate(prompt)
                
                # 实时计算奖励
                reward = self.reward_model(prompt, response)
                
                # 计算优势（需要价值函数）
                value = self.policy.value_head(prompt, response)
                
                trajectories.append({
                    'prompt': prompt,
                    'response': response,
                    'reward': reward,
                    'value': value,
                    'logprobs': self.policy.get_logprobs(prompt, response)
                })
        
        return trajectories
    
    def train_step(self, prompts):
        # 收集新数据
        new_data = self.collect_trajectories(prompts)
        
        # 更新缓冲区（FIFO）
        self.buffer.extend(new_data)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        # 在缓冲区数据上训练
        for epoch in range(4):
            for batch in self.get_batches(self.buffer):
                loss = self.ppo_loss(batch)
                self.optimize(loss)
```

### 6.6.3 离线RL的优势与挑战

**优势**：
1. **安全可控**：不会生成未见过的有害内容
2. **计算高效**：数据可预处理和缓存
3. **可重复性**：相同数据得到相同结果

**挑战**：
1. **分布偏移**：训练和部署分布不匹配
2. **数据质量依赖**：完全依赖历史数据质量
3. **保守性**：难以超越数据集中的最佳表现

**离线DPO实现**：

```python
class OfflineDPO:
    def __init__(self, model, ref_model, dataset):
        self.model = model
        self.ref_model = ref_model
        self.dataset = dataset  # 预收集的偏好数据
        
    def train(self, n_epochs=3):
        """
        纯离线训练，不生成新数据
        """
        for epoch in range(n_epochs):
            for batch in self.dataset:
                # 使用固定数据集
                loss = self.compute_dpo_loss(
                    batch['prompts'],
                    batch['chosen'],
                    batch['rejected']
                )
                
                self.optimize(loss)
            
            # 离线评估
            val_loss = self.evaluate_offline()
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
    
    def compute_importance_weights(self, batch):
        """
        计算重要性权重以缓解分布偏移
        """
        with torch.no_grad():
            # 当前策略的概率
            current_probs = self.model.get_probs(batch['prompts'], batch['responses'])
            
            # 数据收集时的概率（如果有）
            old_probs = batch.get('old_probs', torch.ones_like(current_probs))
            
            # 重要性权重
            weights = current_probs / (old_probs + 1e-8)
            
            # 裁剪防止权重爆炸
            weights = torch.clamp(weights, 0.1, 10.0)
        
        return weights
```

### 6.6.4 混合策略：半在线学习

结合两者优势的实用方案：

```python
class SemiOnlineRL:
    def __init__(self, model, offline_data, online_ratio=0.2):
        self.model = model
        self.offline_data = offline_data
        self.online_ratio = online_ratio
        self.online_buffer = []
    
    def train_step(self, prompts):
        batch_size = len(prompts)
        
        # 1. 离线数据采样
        offline_size = int(batch_size * (1 - self.online_ratio))
        offline_batch = self.sample_offline(offline_size)
        
        # 2. 在线数据生成（少量）
        online_size = batch_size - offline_size
        online_batch = self.generate_online(
            prompts[:online_size]
        )
        
        # 3. 混合训练
        combined_batch = self.combine_batches(
            offline_batch, 
            online_batch
        )
        
        # 4. 加权更新
        loss = self.weighted_loss(combined_batch)
        self.optimize(loss)
    
    def weighted_loss(self, batch):
        """
        对在线和离线数据使用不同权重
        """
        losses = []
        
        for item in batch:
            if item['source'] == 'online':
                # 在线数据权重更高（更可信）
                weight = 1.5
            else:
                # 离线数据权重较低
                weight = 1.0
            
            loss = self.compute_loss(item) * weight
            losses.append(loss)
        
        return torch.stack(losses).mean()
```

### 6.6.5 实用决策框架

```python
def choose_rl_strategy(constraints):
    """
    根据实际约束选择RL策略
    """
    if constraints['safety_critical']:
        # 安全要求高，使用纯离线
        return 'offline', "安全第一，避免未知风险"
    
    elif constraints['compute_budget'] < 100:  # GPU小时
        # 计算预算有限，使用离线
        return 'offline', "计算资源受限"
    
    elif constraints['data_quality'] < 0.7:
        # 数据质量差，需要在线探索
        return 'online', "数据质量不足，需要主动改进"
    
    elif constraints['deployment_type'] == 'production':
        # 生产环境，使用半在线
        return 'semi_online', "平衡安全性和性能"
    
    else:
        # 研究环境，使用在线
        return 'online', "追求最佳性能"
```

### 6.6.6 分布偏移的缓解技术

**技术1：保守正则化**

```python
def conservative_regularization(policy, ref_policy, responses, alpha=0.1):
    """
    CQL-style保守正则化，防止离线RL过度乐观
    """
    # 计算OOD（out-of-distribution）动作的Q值
    ood_responses = policy.sample(temperature=1.5)  # 高温采样OOD
    ood_q_values = policy.q_function(ood_responses)
    
    # 惩罚OOD动作的高Q值
    conservative_loss = alpha * ood_q_values.mean()
    
    return conservative_loss
```

**技术2：分布感知采样**

```python
class DistributionAwareSampler:
    def __init__(self, offline_data, model):
        self.offline_data = offline_data
        self.model = model
        
        # 预计算数据分布特征
        self.data_embeddings = self.compute_embeddings(offline_data)
        self.distribution_stats = self.compute_stats(self.data_embeddings)
    
    def sample_in_distribution(self, n_samples):
        """
        优先采样分布内的数据
        """
        candidates = []
        
        for _ in range(n_samples * 5):  # 过采样
            response = self.model.generate()
            embedding = self.get_embedding(response)
            
            # 计算与训练分布的距离
            distance = self.compute_distance(
                embedding, 
                self.distribution_stats
            )
            
            candidates.append((response, distance))
        
        # 选择最接近训练分布的样本
        candidates.sort(key=lambda x: x[1])
        return [c[0] for c in candidates[:n_samples]]
```

## 本章小结

本章深入探讨了基于人类反馈的强化学习（RLHF）及其变体在大语言模型后训练中的应用。我们学习了：

### 核心概念回顾

1. **RLHF的三大支柱**：
   - 奖励模型：将人类偏好转化为可优化的标量信号
   - 策略优化：通过PPO等算法最大化期望奖励
   - KL约束：防止模型偏离初始分布过远

2. **关键算法对比**：
   - **PPO**：稳定但计算密集，需要显式奖励模型
   - **DPO**：直接优化，训练高效但可能过拟合
   - **IPO**：更鲁棒但收敛较慢
   - **Constitutional AI**：自我改进，减少人工标注

3. **实用权衡**：
   - 在线RL：探索能力强但成本高
   - 离线RL：安全高效但受限于数据质量
   - 混合策略：平衡探索与安全性

### 关键公式汇总

**Bradley-Terry偏好模型**：
$$P(A \succ B) = \sigma(r(A) - r(B))$$

**PPO目标函数**：
$$\mathcal{L}_{PPO} = \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)$$

**DPO损失函数**：
$$\mathcal{L}_{DPO} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

### 实践要点

1. **奖励模型训练**：使用集成和校准技术提高鲁棒性
2. **PPO调优**：监控KL散度和裁剪比例，采用渐进式训练
3. **DPO/IPO选择**：根据数据质量和标注一致性选择
4. **在线/离线决策**：基于安全需求和计算预算权衡

## 练习题

### 基础题（理解概念）

**练习6.1：奖励模型设计**
设计一个奖励模型架构，用于评估代码生成任务的质量。考虑如何处理语法正确性、功能完整性和代码风格等多个维度。

<details>
<summary>提示（Hint）</summary>
考虑多头架构，每个头负责一个质量维度，最终加权组合。
</details>

<details>
<summary>参考答案</summary>

奖励模型应包含：
1. 语法检查头：使用AST解析验证语法正确性（二值输出）
2. 功能评估头：通过测试用例执行评估功能完整性（0-1连续值）
3. 风格评分头：基于代码规范检查工具的输出（0-1连续值）
4. 效率评估头：分析时间/空间复杂度（可选）

最终奖励 = w1×语法 + w2×功能 + w3×风格 + w4×效率

其中权重可以根据任务需求调整，如生产代码更重视功能和语法，教学代码更重视风格。

</details>

**练习6.2：KL散度计算**
给定参考分布 $p_{ref} = [0.1, 0.2, 0.3, 0.4]$ 和当前分布 $p_{\theta} = [0.15, 0.25, 0.35, 0.25]$，计算KL散度 $D_{KL}(p_{\theta} || p_{ref})$。

<details>
<summary>提示（Hint）</summary>
使用公式：$D_{KL}(p||q) = \sum_i p_i \log(p_i/q_i)$
</details>

<details>
<summary>参考答案</summary>

$D_{KL}(p_{\theta} || p_{ref}) = \sum_i p_{\theta,i} \log(p_{\theta,i}/p_{ref,i})$

= 0.15×log(0.15/0.1) + 0.25×log(0.25/0.2) + 0.35×log(0.35/0.3) + 0.25×log(0.25/0.4)
= 0.15×0.405 + 0.25×0.223 + 0.35×0.155 + 0.25×(-0.470)
= 0.061 + 0.056 + 0.054 - 0.118
= 0.053

KL散度约为0.053，表示两个分布相对接近。

</details>

**练习6.3：DPO vs PPO场景选择**
列举三个适合使用DPO而非PPO的具体场景，并说明原因。

<details>
<summary>提示（Hint）</summary>
考虑计算资源、数据可用性和训练稳定性等因素。
</details>

<details>
<summary>参考答案</summary>

1. **学术研究环境**：计算资源有限，DPO不需要训练独立的奖励模型，节省GPU内存和训练时间。

2. **高质量偏好数据充足**：已有大量人工标注的偏好对，且标注一致性高（>85%），DPO可以直接利用这些数据。

3. **快速原型验证**：需要快速验证对齐方法的效果，DPO实现简单，收敛快，适合快速迭代。

不适合DPO的场景：标注噪声大、需要在线探索、安全性要求极高的场景。

</details>

### 挑战题（深入思考）

**练习6.4：奖励过拟合检测**
设计一个方法来自动检测RLHF训练过程中的奖励过拟合（reward hacking）现象。

<details>
<summary>提示（Hint）</summary>
考虑监控多个指标的相关性变化，如奖励值与人类评估的相关性。
</details>

<details>
<summary>参考答案</summary>

奖励过拟合检测系统：

1. **指标监控**：
   - 奖励值趋势：如果奖励持续上升但验证集性能下降
   - 响应长度分布：突然变长可能表示在利用长度偏好
   - 词汇多样性：下降表示模型在重复特定模式

2. **对抗测试**：
   - 定期生成对抗样本，检查奖励模型是否给予不合理高分
   - 使用简单的重复模式测试，如"非常好非常好..."

3. **人类评估对比**：
   - 定期抽样进行人类评估
   - 计算奖励值与人类评分的Spearman相关系数
   - 相关性下降是过拟合的强信号

4. **自动化检测规则**：
   ```python
   if (reward_increase > 50% and 
       human_eval_correlation < 0.5 and
       response_length_std > 2 * initial_std):
       trigger_alert("可能的奖励过拟合")
   ```

</details>

**练习6.5：Constitutional AI原则设计**
为一个医疗咨询AI助手设计一套宪法原则层次结构，确保安全性、准确性和有用性的平衡。

<details>
<summary>提示（Hint）</summary>
考虑医疗领域的特殊性：错误信息的严重后果、法律责任、患者隐私等。
</details>

<details>
<summary>参考答案</summary>

医疗AI宪法原则层次：

**第一层：安全性（不可违反）**
1. 绝不替代专业医疗诊断
2. 危急情况必须建议立即就医
3. 不推荐未经验证的治疗方法
4. 严格保护患者隐私信息

**第二层：准确性（强约束）**
1. 引用信息必须来自可靠医学来源
2. 明确区分常见情况和需要专业评估的情况
3. 承认医学不确定性，避免绝对化表述
4. 纠正明显的医学误解

**第三层：有用性（软约束）**
1. 提供易懂的医学知识解释
2. 给出合理的健康生活建议
3. 帮助准备就医问题清单
4. 提供情绪支持和安慰

**实施策略**：
- 第一层违反 → 立即拒绝输出
- 第二层违反 → 强制修改直到满足
- 第三层违反 → 尝试改进但可接受

</details>

**练习6.6：在线离线RL混合策略**
设计一个自适应的在线/离线RL混合训练策略，能够根据训练过程中的表现动态调整在线数据的比例。

<details>
<summary>提示（Hint）</summary>
考虑使用验证集性能、KL散度、计算成本等信号来调整混合比例。
</details>

<details>
<summary>参考答案</summary>

自适应混合策略：

```python
class AdaptiveMixedRL:
    def __init__(self):
        self.online_ratio = 0.1  # 初始10%在线
        self.performance_history = []
        
    def adjust_ratio(self, metrics):
        # 1. 性能改进率
        if len(self.performance_history) > 5:
            recent_improvement = np.mean(np.diff(self.performance_history[-5:]))
            
            if recent_improvement < 0.001:  # 性能停滞
                self.online_ratio = min(0.5, self.online_ratio * 1.5)
                
        # 2. KL散度监控
        if metrics['kl_divergence'] > 0.1:  # KL过大
            self.online_ratio = max(0.05, self.online_ratio * 0.8)
            
        # 3. 计算预算约束
        if metrics['compute_usage'] > 0.8:  # 接近预算上限
            self.online_ratio = max(0, self.online_ratio - 0.1)
            
        # 4. 数据分布匹配度
        if metrics['distribution_shift'] > 0.3:  # 分布偏移严重
            self.online_ratio = min(0.7, self.online_ratio + 0.1)
            
        return self.online_ratio
```

**动态调整规则**：
1. 初始阶段（前1000步）：纯离线，建立基线
2. 探索阶段（1000-5000步）：逐步增加在线比例
3. 稳定阶段（5000+步）：根据性能自适应调整
4. 异常处理：检测到训练不稳定时快速降低在线比例

</details>

**练习6.7：多目标RLHF优化**
设计一个方法来同时优化多个可能冲突的目标（如有用性、安全性、创造性），并处理它们之间的权衡。

<details>
<summary>提示（Hint）</summary>
考虑Pareto优化、多奖励模型、条件训练等方法。
</details>

<details>
<summary>参考答案</summary>

多目标RLHF框架：

1. **多奖励模型架构**：
```python
class MultiObjectiveRLHF:
    def __init__(self, objectives):
        self.reward_models = {
            'helpfulness': RewardModel(),
            'safety': RewardModel(),
            'creativity': RewardModel()
        }
        self.weights = {'helpfulness': 0.4, 'safety': 0.4, 'creativity': 0.2}
        
    def compute_pareto_rewards(self, responses):
        rewards = {}
        for obj, model in self.reward_models.items():
            rewards[obj] = model(responses)
        
        # 计算Pareto前沿
        pareto_mask = self.get_pareto_optimal(rewards)
        
        return rewards, pareto_mask
```

2. **动态权重调整**：
```python
def adjust_weights_by_context(prompt_type):
    if prompt_type == 'medical':
        return {'safety': 0.7, 'helpfulness': 0.3, 'creativity': 0.0}
    elif prompt_type == 'creative_writing':
        return {'creativity': 0.6, 'helpfulness': 0.3, 'safety': 0.1}
    else:
        return {'helpfulness': 0.4, 'safety': 0.4, 'creativity': 0.2}
```

3. **条件奖励函数**：
```python
def conditional_reward(response, objectives, context):
    base_rewards = compute_base_rewards(response, objectives)
    
    # 硬约束：安全性低于阈值时严重惩罚
    if base_rewards['safety'] < 0.3:
        return -10.0
    
    # 软约束：根据上下文加权
    weighted_reward = sum(
        base_rewards[obj] * weight 
        for obj, weight in context_weights.items()
    )
    
    return weighted_reward
```

4. **多策略集成**：
训练多个专门化的策略，推理时根据需求选择或插值。

</details>

## 常见陷阱与错误

### 1. 奖励模型过拟合

**错误表现**：
- 奖励值持续上升但实际质量下降
- 模型输出变得单一化（如总是生成极长回复）

**调试方法**：
```python
# 检测奖励过拟合
def detect_reward_overfitting(model, reward_model, test_prompts):
    responses = model.generate(test_prompts)
    rewards = reward_model(responses)
    
    # 检查1：奖励分布是否异常集中
    if rewards.std() < 0.1:
        print("警告：奖励分布过于集中")
    
    # 检查2：响应长度是否异常
    lengths = [len(r) for r in responses]
    if np.mean(lengths) > 2 * expected_length:
        print("警告：响应长度异常")
    
    # 检查3：词汇多样性
    vocab_diversity = compute_vocab_diversity(responses)
    if vocab_diversity < 0.3:
        print("警告：词汇多样性过低")
```

### 2. KL散度爆炸

**错误表现**：
- 训练后模型行为完全改变
- 失去基础能力（如语法正确性）

**预防措施**：
- 使用自适应KL系数
- 设置KL散度硬上限
- 定期重置到checkpoint

### 3. DPO训练不稳定

**错误表现**：
- 损失震荡不收敛
- chosen和rejected的概率都趋向于0

**解决方案**：
- 降低学习率（通常5e-7以下）
- 增加β参数（提高正则化）
- 过滤低质量偏好对

### 4. Constitutional AI的原则冲突

**错误表现**：
- 模型在满足一个原则时违反另一个
- 输出变得过于保守或模糊

**处理方法**：
- 建立清晰的原则优先级
- 使用分层原则结构
- 定期审查和调整原则

### 5. 在线RL的计算爆炸

**错误表现**：
- 训练时间指数增长
- GPU内存溢出

**优化技巧**：
- 使用经验回放缓冲区
- 批量生成和评估
- 实施早停机制

💡 **最佳实践建议**：
1. 始终保留SFT检查点作为回退方案
2. 使用多个独立的评估指标
3. 定期进行人工评估验证
4. 记录详细的实验日志便于调试
5. 从小规模实验开始逐步扩大