# 第三章：数据工程

## 本章概览

数据是后训练的基石。与预训练阶段追求规模和多样性不同，后训练数据工程聚焦于质量、对齐和任务覆盖。本章深入探讨后训练数据的全生命周期管理：从高质量指令数据的构造，到标注体系的设计，再到数据飞轮的搭建。我们将结合实际案例，阐述如何构建一个可持续、可扩展的数据工程体系，支撑模型能力的持续迭代。

**学习目标**：
- 掌握高质量指令数据的设计原则和构造方法
- 理解标注规范的制定流程和质量控制机制
- 学会搭建数据飞轮，实现数据的自动化迭代
- 掌握合成数据生成的各类技术和评估方法
- 理解数据配比和课程学习在后训练中的应用

## 3.1 高质量指令数据的构造方法

### 3.1.1 指令数据的核心要素

高质量的指令数据应包含三个核心要素：

```
┌─────────────────────────────────────┐
│         指令数据三要素                │
├─────────────────────────────────────┤
│  1. 指令(Instruction)               │
│     - 清晰的任务描述                 │
│     - 明确的约束条件                 │
│     - 期望的输出格式                 │
│                                     │
│  2. 输入(Input)                     │
│     - 任务相关的上下文               │
│     - 必要的背景信息                 │
│     - 多模态内容(可选)               │
│                                     │
│  3. 输出(Output)                    │
│     - 高质量的标准答案               │
│     - 思维过程(对于推理任务)         │
│     - 多样化的表达方式               │
└─────────────────────────────────────┘
```

### 3.1.2 数据源的选择策略

**1. 人工构造数据**

优势：质量可控、任务针对性强
劣势：成本高、规模受限

构造原则：
- **任务覆盖完整性**：系统性地覆盖目标能力矩阵
- **难度梯度设计**：从简单到复杂的渐进式分布
- **边界案例包含**：刻意包含异常和边界情况

**2. 现有数据改造**

从高质量的文本语料（如教科书、技术文档）中提取和改造：

```
原始文本 → 问答对提取 → 指令格式化 → 质量筛选
         ↓
    信息抽取规则
    (NER, 关系抽取等)
```

**3. 模型生成数据**

利用强大的基础模型生成训练数据：

```
种子任务 → LLM生成 → 人工验证 → 自动扩展
         ↓            ↓
    Self-Instruct  质量评分模型
```

### 3.1.3 数据多样性设计

📌 **多样性维度**：

1. **任务类型多样性**
   - 生成类：创作、翻译、总结
   - 理解类：分类、抽取、问答
   - 推理类：数学、逻辑、代码

2. **领域多样性**
   - 通用知识 vs 专业领域
   - 正式语境 vs 日常对话
   - 不同文化背景

3. **复杂度多样性**
   - 单轮 vs 多轮
   - 简单指令 vs 复合任务
   - 短文本 vs 长文本

### 3.1.4 数据质量评估框架

建立多维度的质量评估体系：

$$Q_{data} = \alpha \cdot Q_{correctness} + \beta \cdot Q_{helpfulness} + \gamma \cdot Q_{harmlessness} + \delta \cdot Q_{diversity}$$

其中：
- $Q_{correctness}$：事实准确性得分
- $Q_{helpfulness}$：有用性得分  
- $Q_{harmlessness}$：安全性得分
- $Q_{diversity}$：多样性得分
- $\alpha, \beta, \gamma, \delta$：权重系数，满足 $\sum = 1$

## 3.2 标注规范设计与质量控制

### 3.2.1 标注规范的层次化设计

```
┌─────────────────────────────────────┐
│          标注规范层次结构             │
├─────────────────────────────────────┤
│  Level 1: 基础规范                   │
│  - 格式要求                          │
│  - 长度限制                          │
│  - 语言风格                          │
├─────────────────────────────────────┤
│  Level 2: 任务规范                   │
│  - 任务特定要求                      │
│  - 评分标准                          │
│  - 示例案例                          │
├─────────────────────────────────────┤
│  Level 3: 质量规范                   │
│  - 准确性标准                        │
│  - 完整性要求                        │
│  - 一致性检查                        │
├─────────────────────────────────────┤
│  Level 4: 安全规范                   │
│  - 有害内容过滤                      │
│  - 偏见检测                          │
│  - 隐私保护                          │
└─────────────────────────────────────┘
```

### 3.2.2 标注者管理体系

**1. 标注者选择与培训**

- **背景筛选**：根据任务需求选择合适背景的标注者
- **培训流程**：
  ```
  理论学习 → 案例练习 → 测试考核 → 正式标注 → 持续反馈
  ```
- **能力分级**：初级、中级、高级标注者的任务分配

**2. 标注一致性保证**

Inter-Annotator Agreement (IAA) 计算：

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

其中：
- $P_o$：观察到的一致性比例
- $P_e$：随机一致性的期望比例

目标：$\kappa > 0.8$ 表示高度一致

### 3.2.3 质量控制机制

**1. 多重标注策略**

```
任务 → 标注者A → 结果A ↘
     → 标注者B → 结果B → 仲裁/投票 → 最终结果
     → 标注者C → 结果C ↗
```

**2. 质量抽检流程**

- **随机抽检**：按比例随机抽取样本复核
- **定向抽检**：针对特定标注者或任务类型
- **交叉验证**：标注者互相检查

**3. 动态质量评分**

标注者质量得分更新：

$$Q_t = \lambda \cdot Q_{t-1} + (1-\lambda) \cdot q_t$$

其中：
- $Q_t$：时刻 $t$ 的质量得分
- $q_t$：当前批次的质量评分
- $\lambda$：历史权重因子（典型值 0.7-0.9）

### 3.2.4 标注工具与平台

**关键功能需求**：

1. **任务分发**：智能分配、负载均衡
2. **进度追踪**：实时监控、瓶颈识别
3. **版本管理**：标注历史、变更追踪
4. **协作功能**：讨论、争议解决
5. **自动化辅助**：预标注、智能提示

⚠️ **常见陷阱**：
- 过度依赖单一标注源
- 忽视标注者疲劳导致的质量下降
- 标注规范过于复杂导致理解偏差
- 缺乏及时的反馈循环

## 3.3 数据飞轮（Data Flywheel）搭建

### 3.3.1 数据飞轮的核心理念

数据飞轮是一个自我强化的循环系统，通过模型部署收集新数据，不断改进模型性能：

```
     ┌──────────────────┐
     │   1. 模型部署     │
     │   收集用户交互    │
     └────────┬─────────┘
              ↓
     ┌──────────────────┐
     │   2. 数据筛选     │
     │   识别高价值样本  │
     └────────┬─────────┘
              ↓
     ┌──────────────────┐
     │   3. 数据标注     │
     │   人工/自动标注   │
     └────────┬─────────┘
              ↓
     ┌──────────────────┐
     │   4. 模型训练     │
     │   增量/全量训练   │
     └────────┬─────────┘
              ↓
     ┌──────────────────┐
     │   5. 评估验证     │
     │   A/B测试         │
     └────────┬─────────┘
              ↓
         循环继续 ←────────┘
```

### 3.3.2 数据收集策略

**1. 主动收集**

- **用户反馈按钮**：👍/👎 快速反馈
- **详细评价表单**：结构化的质量评估
- **对比评测**：A/B 模型输出对比

**2. 被动收集**

- **交互日志**：用户行为序列分析
- **修改痕迹**：用户对输出的编辑
- **使用模式**：高频查询、重试行为

**3. 隐式信号提取**

```python
# 伪代码示例
value_score = α * dwell_time + 
              β * copy_rate + 
              γ * share_rate - 
              δ * regeneration_rate
```

### 3.3.3 高价值数据识别

**价值评分模型**：

$$V(x) = w_1 \cdot \text{Uncertainty}(x) + w_2 \cdot \text{Diversity}(x) + w_3 \cdot \text{Difficulty}(x) + w_4 \cdot \text{Frequency}(x)$$

其中：
- **Uncertainty**：模型预测的不确定性（熵）
- **Diversity**：与现有数据的差异度
- **Difficulty**：任务复杂度评分
- **Frequency**：查询频率或重要性

**筛选策略**：

1. **不确定性采样**：选择模型最不确定的样本
2. **多样性采样**：最大化数据集的覆盖度
3. **对抗样本挖掘**：找出模型失败的案例
4. **边界探索**：接近决策边界的样本

### 3.3.4 自动化标注流水线

**混合标注策略**：

```
          高置信度
新数据 → 模型预标注 → 自动采纳
                   ↘
                    中置信度 → 人工复核 → 标注结果
                   ↗
         低置信度 → 人工标注
```

**置信度校准**：

使用温度缩放（Temperature Scaling）校准模型置信度：

$$p_i^{calibrated} = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

其中 $T$ 是通过验证集优化的温度参数。

### 3.3.5 增量训练与版本管理

**增量训练策略**：

1. **持续微调**：在新数据上继续训练
   ```
   Loss = λ * L_new + (1-λ) * L_replay
   ```
   
2. **经验回放**：混合历史数据防止遗忘
3. **弹性权重巩固（EWC）**：保护重要参数

**数据版本管理**：

```
data_v1.0/
├── raw/           # 原始数据
├── processed/     # 处理后数据
├── splits/        # 训练/验证/测试划分
├── metadata.json  # 数据集元信息
└── changelog.md   # 变更记录

data_v1.1/
├── incremental/   # 增量数据
├── merged/        # 合并后完整数据
└── diff_report/   # 变更分析
```

💡 **实用技巧**：
- 保持 10-20% 的验证集稳定，用于长期性能追踪
- 使用数据哈希值追踪数据血缘
- 定期进行全量重训，重置模型状态

## 3.4 合成数据生成策略

### 3.4.1 合成数据的价值与挑战

**价值**：
- 规模化：低成本大规模生成
- 可控性：精确控制数据特征
- 隐私性：避免真实数据隐私问题
- 平衡性：补充稀有类别数据

**挑战**：
- 质量保证：避免错误传播
- 多样性：防止模式坍缩
- 真实性：保持与真实分布一致

### 3.4.2 生成方法分类

**1. 模板基础生成**

```python
template = "将下列{语言A}翻译成{语言B}：{文本}"
instances = [
    {"语言A": "英文", "语言B": "中文", "文本": text}
    for text in source_texts
]
```

**2. 扰动基础生成**

原始样本通过系统性扰动生成变体：

- **词级扰动**：同义词替换、插入、删除
- **句级扰动**：改写、倒装、拆分合并
- **语义扰动**：否定、条件变换、视角转换

**3. 模型基础生成**

利用大模型生成训练数据：

```
┌────────────────────────────────┐
│     Self-Instruct Pipeline     │
├────────────────────────────────┤
│ 1. 种子任务 (175个)             │
│    ↓                           │
│ 2. 指令生成 (GPT生成新指令)     │
│    ↓                           │
│ 3. 指令过滤 (去重、质量筛选)    │
│    ↓                           │
│ 4. 实例生成 (输入输出对)        │
│    ↓                           │
│ 5. 质量验证                    │
│    ↓                           │
│ 6. 加入种子池 (迭代)           │
└────────────────────────────────┘
```

### 3.4.3 知识蒸馏与数据增强

**知识蒸馏流程**：

$$\mathcal{L}_{distill} = \alpha \cdot \mathcal{L}_{CE}(y, \hat{y}) + (1-\alpha) \cdot \mathcal{L}_{KL}(p_{teacher}, p_{student})$$

其中：
- $\mathcal{L}_{CE}$：交叉熵损失（hard label）
- $\mathcal{L}_{KL}$：KL散度（soft label）
- $\alpha$：硬标签权重

**数据增强技术**：

1. **回译增强**（Back-translation）
   ```
   原文 → 翻译到语言B → 翻译回语言A → 增强样本
   ```

2. **链式思维增强**（CoT Augmentation）
   ```
   问题 → 生成推理步骤 → 验证答案 → 筛选高质量CoT
   ```

3. **对比学习增强**
   生成正例和负例对：
   ```
   原始样本 → 正例（相似但不同）
           → 负例（表面相似但语义不同）
   ```

### 3.4.4 合成数据质量评估

**自动评估指标**：

1. **困惑度过滤**：
   $$PPL_{threshold} = \mu_{PPL} + k \cdot \sigma_{PPL}$$

2. **多样性度量**：
   - N-gram多样性
   - 语义嵌入多样性
   - 句法结构多样性

3. **一致性检验**：
   - 自洽性：多次生成的一致性
   - 事实一致性：与知识库对比
   - 逻辑一致性：推理链验证

**人工评估采样**：

采用分层采样确保覆盖：
```
总体 → 按难度分层 → 按类型分层 → 随机采样 → 人工评估
```

⚠️ **常见陷阱**：
- 过度依赖单一生成模型
- 忽视生成数据的分布偏移
- 缺乏系统性的质量控制
- 合成数据比例过高导致性能退化

## 3.5 数据配比与课程学习

### 3.5.1 数据配比的理论基础

**最优配比问题**：

给定 $K$ 类任务数据 $\{D_1, D_2, ..., D_K\}$，寻找最优混合比例 $\{\alpha_1, \alpha_2, ..., \alpha_K\}$：

$$\min_{\alpha} \sum_{i=1}^{K} w_i \cdot \mathcal{L}_i(\theta; \alpha_i \cdot D_i)$$

约束条件：$\sum_{i=1}^{K} \alpha_i = 1, \alpha_i \geq 0$

**配比策略**：

1. **均匀配比**：$\alpha_i = 1/K$
2. **按量配比**：$\alpha_i \propto |D_i|$
3. **按性能配比**：$\alpha_i \propto 1/\mathcal{L}_i$
4. **动态配比**：训练过程中调整

### 3.5.2 课程学习设计

**难度评估**：

```
难度指标 = f(长度, 复杂度, 稀有度, 歧义度)
```

**课程策略**：

1. **单调递增课程**：
   ```
   简单样本 → 中等样本 → 困难样本
   ```

2. **循环课程**：
   ```
   第1轮: 简单(100%)
   第2轮: 简单(50%) + 中等(50%)
   第3轮: 简单(25%) + 中等(50%) + 困难(25%)
   ```

3. **自适应课程**：
   根据模型当前性能动态调整：
   $$p(x) \propto \exp(-\lambda \cdot |difficulty(x) - competence(model)|)$$

### 3.5.3 多任务数据混合

**混合粒度选择**：

1. **样本级混合**：每个batch包含多种任务
   - 优点：梯度更新平滑
   - 缺点：可能相互干扰

2. **批次级混合**：不同batch来自不同任务
   - 优点：任务独立性好
   - 缺点：可能导致震荡

3. **阶段级混合**：分阶段训练不同任务
   - 优点：专注度高
   - 缺点：易遗忘早期任务

**采样算法**：

```python
# 温度采样
def temperature_sampling(task_sizes, temperature=1.0):
    probs = np.array(task_sizes) ** (1/temperature)
    probs = probs / probs.sum()
    return probs

# temperature > 1: 更均匀
# temperature < 1: 更偏向大任务
# temperature = 1: 按比例采样
```

### 3.5.4 数据配比的实验验证

**网格搜索法**：

```
配比实验矩阵：
┌─────────────────────────────┐
│ Task A │ Task B │ Task C │ Score │
├────────┼────────┼────────┼───────┤
│  0.33  │  0.33  │  0.34  │  0.85 │
│  0.50  │  0.25  │  0.25  │  0.87 │
│  0.25  │  0.50  │  0.25  │  0.86 │
│  0.60  │  0.20  │  0.20  │  0.88 │ ← 最优
└─────────────────────────────┘
```

**贝叶斯优化**：

使用高斯过程建模配比与性能的关系：

$$f(\alpha) \sim \mathcal{GP}(\mu(\alpha), k(\alpha, \alpha'))$$

通过最大化采集函数选择下一个实验点。

💡 **实用技巧**：
- 先用小规模实验快速探索配比空间
- 关注任务间的协同效应和负迁移
- 保留一定比例的"保护数据"防止能力退化
- 使用早停避免过拟合某类任务

## 本章小结

本章系统介绍了后训练数据工程的完整流程。核心要点包括：

1. **数据质量优于数量**：精心设计的小规模高质量数据往往优于大规模低质量数据

2. **标注体系是基石**：完善的标注规范、质量控制和标注者管理决定数据质量上限

3. **数据飞轮驱动迭代**：通过部署-收集-标注-训练的循环实现持续改进

4. **合成数据作为补充**：合理使用合成数据可以提升效率，但需要严格的质量控制

5. **配比与课程影响收敛**：数据配比和课程设计直接影响训练效率和最终性能

**关键公式回顾**：

- 数据质量评分：$Q_{data} = \sum \omega_i \cdot Q_i$
- 标注一致性：$\kappa = \frac{P_o - P_e}{1 - P_e}$
- 知识蒸馏损失：$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE} + (1-\alpha) \cdot \mathcal{L}_{KL}$
- 最优配比：$\min_{\alpha} \sum w_i \cdot \mathcal{L}_i(\theta; \alpha_i \cdot D_i)$

## 练习题

### 基础题

**练习 3.1**：设计一个多轮对话任务的标注规范，包括格式要求、质量标准和评分细则。

<details>
<summary>💡 提示</summary>

考虑以下要素：
- 对话的连贯性和上下文依赖
- 角色一致性
- 信息的累积性
- 错误传播的处理

</details>

<details>
<summary>📝 参考答案</summary>

标注规范应包括：

1. **格式规范**：
   - 明确的轮次标记
   - 角色标识（用户/助手）
   - 上下文窗口定义

2. **质量维度**：
   - 相关性：回复是否针对当前问题
   - 连贯性：是否与历史对话一致
   - 信息性：是否提供有价值信息
   - 自然度：语言是否流畅自然

3. **评分标准**：
   - 5分制，每个维度独立评分
   - 总分加权平均
   - 低于3分需要重新标注

4. **特殊情况处理**：
   - 话题转换的合理性判断
   - 指代消解的准确性
   - 多轮依赖的完整性检查

</details>

**练习 3.2**：计算两个标注者的 Cohen's Kappa 系数。标注者A和B对100个样本进行二分类标注，其中：
- 两者都标注为正例：40个
- 两者都标注为负例：35个  
- A正B负：15个
- A负B正：10个

<details>
<summary>💡 提示</summary>

Kappa 公式：$\kappa = \frac{P_o - P_e}{1 - P_e}$

其中：
- $P_o$ = 观察一致性
- $P_e$ = 期望一致性

</details>

<details>
<summary>📝 参考答案</summary>

计算步骤：

1. **观察一致性**：
   $P_o = \frac{40 + 35}{100} = 0.75$

2. **边际概率**：
   - A标正例：$\frac{40 + 15}{100} = 0.55$
   - B标正例：$\frac{40 + 10}{100} = 0.50$
   - A标负例：$\frac{35 + 10}{100} = 0.45$
   - B标负例：$\frac{35 + 15}{100} = 0.50$

3. **期望一致性**：
   $P_e = 0.55 \times 0.50 + 0.45 \times 0.50 = 0.275 + 0.225 = 0.50$

4. **Kappa系数**：
   $\kappa = \frac{0.75 - 0.50}{1 - 0.50} = \frac{0.25}{0.50} = 0.50$

结果：κ = 0.50，表示中等程度的一致性。

</details>

**练习 3.3**：设计一个数据飞轮的最小可行版本（MVP），包括数据收集、筛选、标注和训练的完整流程。

<details>
<summary>💡 提示</summary>

考虑：
- 最简单的反馈机制
- 自动化vs人工的平衡
- 迭代周期的设定
- 评估指标的选择

</details>

<details>
<summary>📝 参考答案</summary>

MVP 数据飞轮设计：

1. **数据收集（每日）**：
   - 用户查询日志
   - 简单的👍/👎反馈
   - 响应时间和重试次数

2. **数据筛选（每周）**：
   - 筛选标准：
     * 所有👎的样本
     * 响应时间>5秒的样本
     * 重试>2次的样本
   - 每周选择Top 100样本

3. **标注流程（每周）**：
   - 2名标注者独立标注
   - 不一致的由第3人仲裁
   - 生成改进的回复

4. **模型更新（每两周）**：
   - 累积200个新样本
   - 混合10%历史数据
   - 增量训练2个epoch

5. **评估验证**：
   - 保持固定测试集
   - A/B测试（5%流量）
   - 关键指标：满意度提升>2%则全量

</details>

### 挑战题

**练习 3.4**：设计一个自适应的数据配比算法，能够根据模型在不同任务上的表现动态调整训练数据的采样比例。

<details>
<summary>💡 提示</summary>

考虑：
- 如何度量各任务的学习进度
- 如何平衡探索与利用
- 如何避免某些任务被"饿死"
- 如何处理任务间的依赖关系

</details>

<details>
<summary>📝 参考答案</summary>

自适应配比算法设计：

1. **性能追踪**：
   ```python
   performance[task] = exponential_moving_average(
       current_loss, 
       historical_performance,
       alpha=0.1
   )
   ```

2. **学习进度评估**：
   ```python
   progress[task] = 1 - (current_loss / initial_loss)
   learning_rate[task] = d(progress) / d(steps)
   ```

3. **配比更新规则**：
   ```python
   # 基础配比
   base_ratio = 1 / num_tasks
   
   # 性能调整项
   perf_adjust = softmax(-performance / temperature)
   
   # 进度调整项
   progress_adjust = softmax(-learning_rate / temperature)
   
   # 最终配比
   ratio[task] = base_ratio + 
                 α * perf_adjust + 
                 β * progress_adjust
   
   # 归一化
   ratio = ratio / ratio.sum()
   
   # 保底机制
   ratio = max(ratio, min_ratio)
   ```

4. **探索机制**：
   - 每N个epoch随机提升某个任务比例
   - 使用ε-greedy策略
   - 记录探索结果用于未来决策

5. **约束条件**：
   - 最小比例：min_ratio = 0.05
   - 最大比例：max_ratio = 0.5
   - 平滑更新：限制相邻epoch的变化率

</details>

**练习 3.5**：设计一个合成数据质量评估系统，能够自动识别和过滤低质量的生成数据。

<details>
<summary>💡 提示</summary>

从多个维度评估：
- 语言质量
- 事实准确性
- 任务相关性
- 与真实数据的分布差异

</details>

<details>
<summary>📝 参考答案</summary>

质量评估系统设计：

1. **多维度评分器**：

   a) **流畅度评分**：
   ```python
   fluency = 1 / (1 + perplexity / baseline_ppl)
   ```

   b) **多样性评分**：
   ```python
   diversity = (unique_ngrams / total_ngrams) * 
              (1 - self_bleu_score)
   ```

   c) **一致性评分**：
   ```python
   # 多次生成的一致性
   consistency = mean_pairwise_similarity(
       multiple_generations
   )
   ```

   d) **相关性评分**：
   ```python
   relevance = cosine_similarity(
       task_embedding, 
       response_embedding
   )
   ```

2. **异常检测**：
   ```python
   # 使用孤立森林检测异常
   from sklearn.ensemble import IsolationForest
   
   features = extract_features(synthetic_data)
   detector = IsolationForest(contamination=0.1)
   outliers = detector.fit_predict(features)
   ```

3. **分布匹配**：
   ```python
   # KL散度检测分布偏移
   kl_div = kl_divergence(
       real_data_distribution,
       synthetic_data_distribution
   )
   
   if kl_div > threshold:
       flag_as_distribution_shift()
   ```

4. **集成决策**：
   ```python
   quality_score = weighted_sum([
       fluency * 0.2,
       diversity * 0.2,
       consistency * 0.3,
       relevance * 0.3
   ])
   
   if quality_score < 0.6 or is_outlier:
       filter_out()
   ```

5. **人工验证采样**：
   - 高置信度通过：直接使用
   - 中置信度：按比例采样验证
   - 低置信度：全部人工审核

</details>

**练习 3.6**：给定一个包含10个不同难度等级任务的数据集，设计一个课程学习策略，使模型训练效率最大化。

<details>
<summary>💡 提示</summary>

考虑：
- 难度评估的客观指标
- 课程的平滑过渡
- 防止灾难性遗忘
- 收敛速度vs最终性能的权衡

</details>

<details>
<summary>📝 参考答案</summary>

课程学习策略设计：

1. **难度量化**：
   ```python
   difficulty = 0.3 * length_score + 
                0.2 * vocabulary_score +
                0.3 * reasoning_steps +
                0.2 * ambiguity_score
   ```

2. **分桶策略**：
   - Level 1-3: 基础任务（30%）
   - Level 4-6: 中级任务（40%）
   - Level 7-10: 高级任务（30%）

3. **渐进式课程**：

   **阶段1（Epoch 1-5）**：
   ```
   Level 1-3: 70%
   Level 4-6: 25%
   Level 7-10: 5%
   ```

   **阶段2（Epoch 6-10）**：
   ```
   Level 1-3: 40%
   Level 4-6: 40%
   Level 7-10: 20%
   ```

   **阶段3（Epoch 11-15）**：
   ```
   Level 1-3: 20%
   Level 4-6: 40%
   Level 7-10: 40%
   ```

   **阶段4（Epoch 16+）**：
   ```
   均匀分布或基于性能的自适应采样
   ```

4. **反遗忘机制**：
   - 每个阶段保留20%的前期数据
   - 使用经验回放缓冲区
   - 定期在早期任务上评估

5. **早停条件**：
   ```python
   if (val_loss_increase_count > patience and
       all_levels_coverage > 0.8):
       stop_training()
   ```

6. **动态调整**：
   ```python
   # 基于学习曲线调整进度
   if learning_plateaued(level_k):
       increase_difficulty_ratio(level_k+1)
   ```

</details>

**练习 3.7**：设计一个数据血缘追踪系统，能够追踪每个训练样本从原始数据到最终使用的完整历史。

<details>
<summary>💡 提示</summary>

考虑：
- 数据的版本控制
- 变换操作的记录
- 性能归因分析
- 存储和查询效率

</details>

<details>
<summary>📝 参考答案</summary>

数据血缘系统设计：

1. **数据模型**：
   ```python
   class DataLineage:
       sample_id: str  # 唯一标识
       source_id: str  # 原始数据ID
       version: str    # 数据版本
       
       transformations: List[{
           'operation': str,
           'timestamp': datetime,
           'parameters': dict,
           'operator': str  # 人/模型
       }]
       
       quality_scores: List[{
           'metric': str,
           'score': float,
           'timestamp': datetime
       }]
       
       usage_history: List[{
           'model_version': str,
           'training_run': str,
           'epoch': int,
           'loss_contribution': float
       }]
   ```

2. **操作追踪**：
   ```python
   @track_lineage
   def transform_data(data, operation):
       result = operation(data)
       lineage.add_transformation({
           'input_hash': hash(data),
           'output_hash': hash(result),
           'operation': operation.__name__,
           'timestamp': now()
       })
       return result
   ```

3. **版本管理**：
   ```python
   # Git-like 版本控制
   data_version = hashlib.sha256(
       data_content + parent_version
   ).hexdigest()[:8]
   ```

4. **查询接口**：
   ```python
   # 正向追踪
   def trace_forward(sample_id):
       return lineage_db.query(
           source_id=sample_id
       )
   
   # 反向追踪
   def trace_backward(model_issue):
       problematic_samples = identify_issues()
       return [
           get_lineage(s) for s in problematic_samples
       ]
   ```

5. **性能归因**：
   ```python
   def attribute_performance(model_degradation):
       # 找出性能下降相关的数据变化
       recent_changes = get_recent_data_changes()
       correlation = calculate_correlation(
           recent_changes,
           model_degradation
       )
       return rank_by_impact(correlation)
   ```

6. **存储优化**：
   - 使用图数据库存储血缘关系
   - 定期压缩历史记录
   - 只保留关键节点的完整数据

</details>

**练习 3.8**：设计一个主动学习（Active Learning）策略，在标注预算有限的情况下选择最有价值的样本进行标注。

<details>
<summary>💡 提示</summary>

结合多种选择策略：
- 不确定性
- 多样性  
- 代表性
- 预期模型改进

</details>

<details>
<summary>📝 参考答案</summary>

主动学习策略设计：

1. **不确定性采样**：
   ```python
   def uncertainty_sampling(model, unlabeled_data):
       predictions = model.predict_proba(unlabeled_data)
       
       # 熵值计算
       entropy = -sum(p * log(p) for p in predictions)
       
       # 最小置信度
       least_confidence = 1 - max(predictions)
       
       # 边界采样
       margin = abs(top1_prob - top2_prob)
       
       uncertainty = α*entropy + β*least_confidence + γ/margin
       return uncertainty
   ```

2. **多样性采样**：
   ```python
   def diversity_sampling(selected_samples, candidate):
       # 基于嵌入的多样性
       min_distance = min([
           cosine_distance(candidate_emb, selected_emb)
           for selected_emb in selected_samples
       ])
       
       # 基于聚类的多样性
       cluster_coverage = len(unique_clusters) / total_clusters
       
       return min_distance * cluster_coverage
   ```

3. **代表性采样**：
   ```python
   def representativeness(sample, unlabeled_pool):
       # 密度估计
       density = mean([
           similarity(sample, other)
           for other in unlabeled_pool
       ])
       
       # 中心性
       centrality = 1 / mean_distance_to_others
       
       return density * centrality
   ```

4. **预期模型改进**：
   ```python
   def expected_model_change(model, sample):
       # 预期梯度长度
       gradient = compute_gradient(model, sample)
       egl = norm(gradient)
       
       # 预期误差减少
       eer = estimate_error_reduction(model, sample)
       
       return egl + eer
   ```

5. **混合策略**：
   ```python
   def hybrid_active_learning(
       model, 
       unlabeled_data, 
       budget,
       selected=[]
   ):
       scores = []
       for sample in unlabeled_data:
           u_score = uncertainty_sampling(model, sample)
           d_score = diversity_sampling(selected, sample)
           r_score = representativeness(sample, unlabeled_data)
           e_score = expected_model_change(model, sample)
           
           # 动态权重
           w1 = 0.4 if len(selected) < budget*0.3 else 0.2
           w2 = 0.2 if len(selected) < budget*0.3 else 0.4
           w3 = 0.2
           w4 = 0.2
           
           total = w1*u_score + w2*d_score + w3*r_score + w4*e_score
           scores.append(total)
       
       # 选择top-k
       top_indices = argsort(scores)[-budget:]
       return unlabeled_data[top_indices]
   ```

6. **批量选择优化**：
   ```python
   # 避免批次内冗余
   def batch_selection(candidates, batch_size):
       selected = []
       for _ in range(batch_size):
           best = argmax([
               score(c) - λ*max_similarity(c, selected)
               for c in candidates
           ])
           selected.append(candidates[best])
           candidates.remove(best)
       return selected
   ```

</details>

## 常见陷阱与错误 (Gotchas)

### 数据相关陷阱

1. **标注规范漂移**
   - 问题：随时间推移，标注标准逐渐偏离
   - 解决：定期校准会议，维护标注示例库

2. **数据泄露**
   - 问题：测试集信息泄露到训练集
   - 解决：严格的数据隔离，使用时间戳分割

3. **标注者偏见累积**
   - 问题：特定标注者的偏好被放大
   - 解决：标注者轮换，交叉验证

### 合成数据陷阱

4. **模型坍缩**
   - 问题：用模型生成的数据训练导致多样性降低
   - 解决：保持真实数据比例，定期注入新数据

5. **错误放大**
   - 问题：生成数据中的错误被学习和放大
   - 解决：严格的质量过滤，人工验证关键样本

6. **分布偏移未察觉**
   - 问题：合成数据分布逐渐偏离真实分布
   - 解决：持续监控分布指标，定期校准

### 工程实践陷阱

7. **数据版本混乱**
   - 问题：不同实验使用了不同版本的数据
   - 解决：严格的版本管理，实验配置记录

8. **增量训练的遗忘**
   - 问题：新数据训练后旧能力退化
   - 解决：经验回放，弹性权重巩固

9. **标注瓶颈**
   - 问题：标注速度跟不上数据产生速度
   - 解决：分优先级标注，自动标注辅助

10. **评估集污染**
    - 问题：评估集被用于训练决策
    - 解决：设置只有少数人知道的保留测试集