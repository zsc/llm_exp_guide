# 第五章：多模态任务实验设计

本章深入探讨多模态大语言模型的后训练实验设计，涵盖视觉、音频、视频等模态与语言的融合方法。我们将系统学习如何设计跨模态对齐实验、处理模态间的异质性挑战，以及构建统一的多模态表示空间。通过本章学习，您将掌握构建生产级多模态模型的完整实验流程。

## 5.1 视觉-语言对齐基础

视觉-语言对齐是多模态模型的核心挑战。不同于文本的离散表示，视觉信号是连续的高维数据，如何在保持语义一致性的同时实现高效对齐，是实验设计的关键考量。

### 5.1.1 对齐范式演进

多模态对齐经历了从简单特征拼接到深度语义融合的演进过程：

**早期方法（2015-2018）**：
- 特征级联：简单将 CNN 特征与词嵌入拼接
- 双塔架构：独立编码后计算相似度
- 问题：模态间隙（modality gap）严重，语义对齐效果差

**中期发展（2018-2021）**：
- 注意力机制引入：ViLBERT、LXMERT 等使用 co-attention
- 预训练任务设计：Masked Language Modeling + Masked Region Modeling
- 突破：CLIP 的对比学习范式，实现零样本迁移

**当前前沿（2021-2025）**：
- 统一架构：Flamingo、BLIP-2 的 Q-Former 设计
- 大规模预训练：LAION-5B 等十亿级数据集
- 效率优化：FLIP 的稀疏采样，节省 2/3 计算

```
演进路径：
Feature Concat → Dual Encoder → Cross Attention → Unified Architecture
     ↓              ↓                ↓                    ↓
  低效对齐      模态隔离         计算密集           统一表示空间
```

### 5.1.2 CLIP 及其变体

CLIP（Contrastive Language-Image Pre-training）奠定了现代视觉-语言对齐的基础：

**核心设计**：
```
Image Encoder: ResNet/ViT → I_emb ∈ R^d
Text Encoder: Transformer → T_emb ∈ R^d
对齐目标: maximize cos(I_emb, T_emb) for matched pairs
```

**关键创新**：
1. **对称损失**：图像→文本 和 文本→图像 双向对比
2. **温度参数**：τ = 0.07，控制分布锐度
3. **大批量训练**：32K batch size，充分利用负样本

**CLIP 变体对比**：

| 模型 | 创新点 | 性能提升 | 计算成本 |
|------|--------|---------|----------|
| CLIP | 基础对比学习 | Baseline | 1.0x |
| ALIGN | 噪声数据训练 | +2% Zero-shot | 1.2x |
| FLIP | 随机 Mask 50% patches | -1% 精度，-2.5x 训练时间 | 0.4x |
| OpenCLIP | 扩展到 ViT-G/14 | +5% ImageNet | 3.0x |
| EVA-CLIP | 改进初始化与优化器 | +3% 均值精度 | 1.1x |

**实验设计要点**：
- 数据质量 vs 数量权衡：ALIGN 证明 1B 噪声数据可行
- 编码器容量分配：视觉编码器通常需要更大容量
- 训练稳定性：梯度累积 + mixed precision 关键

### 5.1.3 对比学习损失设计

对比学习损失是多模态对齐的核心，其设计直接影响模型性能：

**InfoNCE 损失**：
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N} \exp(s_{ij}/\tau)}$$

其中 $s_{ij} = \text{cos}(f_I(x_i), f_T(t_j))$ 为相似度得分。

**改进方向**：

1. **难负样本挖掘**：
```python
# 伪代码
hard_negatives = top_k(similarity_scores, k=10, exclude_positive=True)
loss = -log(exp(pos_score/τ) / (exp(pos_score/τ) + sum(exp(neg/τ) for neg in hard_negatives)))
```

2. **多粒度对比**：
- Global：整图-整句对比
- Regional：区域-短语对比  
- Patch：图块-词汇对比

3. **软标签对比**：
考虑标注不确定性，使用软标签：
$$\mathcal{L}_{\text{soft}} = -\sum_{j} y_j \log p_j$$
其中 $y_j$ 为软标签分布。

**实验技巧**：
- 温度参数调优：τ ∈ [0.01, 0.1]，过小导致梯度爆炸
- 批量大小效应：批量翻倍，性能提升约 1-2%
- 负样本队列：维护动态负样本 bank，提升多样性

### 5.1.4 负样本采样策略

负样本质量直接决定对比学习效果：

**采样策略对比**：

```
随机采样：
├── 优点：实现简单，无偏
└── 缺点：包含大量简单负样本，学习效率低

难例挖掘：
├── 优点：加速收敛，提升区分能力
└── 缺点：可能过拟合，需要 curriculum

混合策略：
├── 80% 随机 + 20% 难例
└── 平衡探索与利用
```

**高级采样技术**：

1. **跨批次负样本**（MoCo 风格）：
```python
class NegativeBank:
    def __init__(self, size=65536):
        self.queue = deque(maxlen=size)
    
    def update(self, features):
        self.queue.extend(features.detach())
    
    def sample(self, n):
        return random.sample(self.queue, min(n, len(self.queue)))
```

2. **语义层次采样**：
- Level 1：完全无关（猫图片 vs 汽车描述）
- Level 2：领域相关（狗图片 vs 猫描述）  
- Level 3：细粒度区分（哈士奇 vs 阿拉斯加描述）

3. **动态难度调整**：
```
训练初期：70% easy + 30% hard
训练中期：50% easy + 50% hard  
训练后期：30% easy + 70% hard
```

**实验建议**：
- 监控负样本相似度分布，避免 collapse
- 使用 gradient accumulation 模拟大批量
- 定期评估 retrieval metrics，不只看 loss

## 5.2 图像理解与生成的统一建模

统一建模是多模态领域的圣杯——用单一模型同时处理理解和生成任务。这不仅简化了系统架构，还能实现任务间的知识迁移。

### 5.2.1 编码器-解码器架构设计

现代统一架构需要平衡理解的精确性和生成的多样性：

**架构演进**：

```
传统分离式：
Vision Encoder → Understanding Tasks
Text Decoder → Generation Tasks
问题：任务孤立，无法共享表示

早期统一（DALL-E）：
Text + Image Tokens → Autoregressive Transformer
问题：图像离散化损失严重

当前主流（Flamingo/BLIP-2）：
Frozen Vision Encoder → Perceiver Resampler → LLM Decoder
优势：保留预训练权重，高效适配
```

**关键设计选择**：

1. **Vision Encoder 选择**：
   - CLIP ViT-L/14：平衡性能与效率
   - EVA-02 ViT-E：极致性能，5B 参数
   - ConvNeXt V2：更好的局部特征

2. **连接层设计**：
   ```python
   class PerceiverResampler(nn.Module):
       def __init__(self, dim=1024, depth=6, num_latents=64):
           self.latents = nn.Parameter(torch.randn(num_latents, dim))
           self.cross_attend = CrossAttention(dim)
           self.self_attend = SelfAttention(dim)
       
       def forward(self, visual_features):
           x = self.latents
           for _ in range(depth):
               x = self.cross_attend(x, visual_features) + x
               x = self.self_attend(x) + x
           return x  # [64, 1024] 固定长度输出
   ```

3. **Decoder 适配**：
   - Prefix Tuning：视觉特征作为 prefix
   - Adapter Layers：在 FFN 后插入适配层
   - LoRA：低秩适配，参数效率最高

**实验技巧**：
- 冻结策略：先冻结 encoder，后期联合微调
- 学习率调度：encoder 1e-5, connector 1e-4, decoder 2e-5
- 数据配比：理解:生成 = 3:1 初期，逐步平衡

### 5.2.2 自回归 vs 扩散模型集成

两大生成范式的优劣与集成策略：

**范式对比**：

| 特性 | 自回归（AR） | 扩散模型（DM） | 混合方案 |
|------|-------------|---------------|----------|
| 生成质量 | 中等 | 高 | 高 |
| 推理速度 | 快（单次） | 慢（多步） | 中等 |
| 可控性 | 强（token级） | 弱（全局） | 强 |
| 训练稳定性 | 高 | 中等 | 中等 |
| 内存需求 | 低 | 高 | 高 |

**集成架构**：

```
输入文本 → LLM → 决策：{理解任务 → AR 输出}
                     {生成任务 → 触发 DM}
                     
Diffusion 分支：
LLM embeddings → Cross-Attention → U-Net → 图像
                     ↑
                 Text Condition
```

**CM3Leon 式统一**：
```python
class UnifiedModel:
    def forward(self, text, image=None, task="understand"):
        if task == "understand":
            # 图像 → 文本
            img_tokens = self.tokenize_image(image)
            return self.ar_generate(concat([img_tokens, text]))
        elif task == "generate":
            # 文本 → 图像
            if self.use_diffusion:
                text_emb = self.encode_text(text)
                return self.diffusion_decode(text_emb)
            else:
                return self.ar_generate_image(text)
```

**训练策略**：
1. 阶段一：分别预训练 AR 和 DM 分支
2. 阶段二：冻结 DM，训练 AR→DM 接口
3. 阶段三：联合微调，平衡两种损失

### 5.2.3 图像 Token 化策略

Token 化质量决定了模型理解和生成的上限：

**主流方法**：

1. **VQ-VAE 系列**：
   ```
   原始 VQ-VAE: 32×32 → 8×8 tokens (压缩率 16)
   VQ-VAE-2: 层级结构，top: 32×32, bottom: 64×64
   问题：码本坍塌，重建质量受限
   ```

2. **VQGAN**：
   ```python
   # 关键改进：感知损失 + 对抗训练
   L_total = L_recon + λ_p*L_perceptual + λ_g*L_gan + λ_c*L_codebook
   
   # 码本大小影响：
   |Codebook| = 1024: 重建 PSNR ~23dB
   |Codebook| = 8192: 重建 PSNR ~26dB
   |Codebook| = 16384: 重建 PSNR ~27dB (收益递减)
   ```

3. **连续 Token（SEED, LaVIT）**：
   ```python
   class ContinuousTokenizer:
       def encode(self, image):
           # 不量化，直接输出连续特征
           features = self.encoder(image)  # [B, 256, 16, 16]
           tokens = self.proj(features.flatten(2).transpose(1,2))  # [B, 256, D]
           return tokens  # 连续值，无码本
   ```

**分辨率处理**：

```
固定分辨率：
├── 简单但损失信息
└── 448×448 是常见选择

动态分辨率（Qwen-VL）：
├── 保持宽高比
├── Padding 到最近的 14 的倍数
└── 位置编码需要 2D 插值

NaViT 风格打包：
├── 不同分辨率图像打包成序列
├── 添加分辨率 token 标记边界
└── 计算效率最高
```

**实验要点**：
- Token 数量权衡：256 tokens 够用，1024 tokens 细节更好
- 重建质量监控：FID < 5 可接受，< 2 优秀
- 语义保持：用 CLIP score 验证语义一致性

### 5.2.4 分辨率自适应训练

处理真实世界多样化分辨率的关键技术：

**挑战与解决方案**：

```
挑战：
1. 训练数据分辨率不一（224×224 到 4K）
2. 推理需求多样（缩略图 vs 细节图）
3. 计算资源限制

解决方案矩阵：
              低分辨率      中分辨率      高分辨率
训练阶段1：    100%          -            -
训练阶段2：     60%         30%          10%
训练阶段3：     30%         40%          30%
```

**Pix2Struct 方法**：
```python
def variable_resolution_encode(image, max_patches=2048):
    h, w = image.shape[-2:]
    
    # 自适应决定 patch 数量
    aspect_ratio = w / h
    if aspect_ratio > 1:
        cols = int(sqrt(max_patches * aspect_ratio))
        rows = int(max_patches / cols)
    else:
        rows = int(sqrt(max_patches / aspect_ratio))
        cols = int(max_patches / rows)
    
    # 动态 patch embedding
    patches = extract_patches(image, (rows, cols))
    return patches, (rows, cols)  # 返回布局信息
```

**位置编码适配**：

1. **2D 正弦编码插值**：
   ```python
   def interpolate_pos_encoding(pos_embed, new_size):
       # pos_embed: [1, N, D], N = 14×14 = 196
       old_size = int(sqrt(pos_embed.shape[1]))
       pos_embed = pos_embed.reshape(1, old_size, old_size, -1)
       pos_embed = F.interpolate(pos_embed.permute(0,3,1,2), 
                                 size=new_size, mode='bicubic')
       return pos_embed.permute(0,2,3,1).flatten(1,2)
   ```

2. **RoPE 2D 扩展**：
   ```python
   def rope_2d(x, h, w):
       # 为 h 和 w 维度分别计算 RoPE
       pos_h = torch.arange(h).unsqueeze(1).repeat(1, w)
       pos_w = torch.arange(w).unsqueeze(0).repeat(h, 1)
       # 应用旋转位置编码
       x = apply_rope(x, pos_h, dim=-2)
       x = apply_rope(x, pos_w, dim=-1)
       return x
   ```

**多尺度训练策略**：

```
Curriculum Learning：
Week 1-2: 224×224 only
Week 3-4: 224×224 (70%) + 448×448 (30%)
Week 5-6: 224×224 (40%) + 448×448 (40%) + 896×896 (20%)
Week 7+:  动态采样，根据 loss 调整

数据增强：
- Random Crop: 保持目标可见前提下
- Multi-scale Training: 0.8x - 1.2x 缩放
- Mixup at Feature Level: 不同分辨率特征混合
```

**效率优化**：
- Flash Attention：长序列必备
- Gradient Checkpointing：用时间换显存
- Mixed Resolution Batch：相近分辨率分组

## 5.3 音频模态集成

音频模态带来独特挑战：时序依赖性强、采样率高、信号类型多样（语音、音乐、环境音）。有效的音频集成需要平衡时频域特征提取与计算效率。

### 5.3.1 语音识别与理解

语音是最重要的音频模态，其与文本的天然对应关系使其成为多模态融合的理想起点：

**技术路线演进**：

```
传统级联：ASR → 文本 → LLM
├── 优点：模块化，各部分可独立优化
└── 缺点：错误传播，丢失韵律信息

端到端集成：Audio → Encoder → LLM
├── 优点：保留完整音频信息
└── 缺点：需要大量配对数据
```

**Whisper 集成方案**：
```python
class WhisperLLMAdapter:
    def __init__(self, whisper_model="large-v3", llm_model="llama-7b"):
        self.whisper = load_whisper(whisper_model)
        self.llm = load_llm(llm_model)
        self.projection = nn.Linear(1280, 4096)  # Whisper → LLM dim
    
    def encode_audio(self, audio_waveform):
        # 提取 Whisper encoder 输出
        mel = log_mel_spectrogram(audio_waveform)
        audio_features = self.whisper.encoder(mel)  # [T, 1280]
        
        # 投影到 LLM 空间
        audio_tokens = self.projection(audio_features)  # [T, 4096]
        
        # 降采样（Whisper 50Hz → LLM ~10Hz）
        audio_tokens = audio_tokens[::5]  # 简单降采样
        return audio_tokens
```

**多语言处理策略**：

1. **语言感知编码**：
   ```python
   lang_embeddings = {
       "en": torch.randn(1, 768),
       "zh": torch.randn(1, 768),
       "es": torch.randn(1, 768),
   }
   
   def encode_with_lang(audio, detected_lang):
       features = self.encoder(audio)
       lang_emb = lang_embeddings[detected_lang]
       return features + lang_emb  # 语言偏置
   ```

2. **代码切换处理**（Code-switching）：
   ```
   输入："今天天气 really nice，我们去 shopping 吧"
   
   分段策略：
   [今天天气] → zh_encoder
   [really nice] → en_encoder  
   [我们去] → zh_encoder
   [shopping] → en_encoder
   [吧] → zh_encoder
   
   融合：Attention 机制自动学习边界
   ```

**韵律信息保留**：
- 音高（Pitch）：提取 F0 轨迹，作为额外 channel
- 能量（Energy）：RMS energy 曲线
- 语速（Duration）：音素时长信息
- 情感（Emotion）：预训练情感分类器特征

### 5.3.2 音乐与环境音理解

非语音音频理解需要不同的特征提取和建模策略：

**音乐理解层次**：

```
低级特征：
├── 节奏（Tempo, Beat）
├── 音高（Pitch, Key）
└── 音色（Timbre）

中级结构：
├── 和弦进行
├── 旋律线条
└── 节奏模式

高级语义：
├── 风格流派
├── 情感表达
└── 结构分析（Verse, Chorus, Bridge）
```

**MusicLM 式建模**：
```python
class MusicEncoder:
    def __init__(self):
        self.w2v_music = load_pretrained("wav2vec2-music")
        self.mulan = load_pretrained("mulan")  # 音乐-文本对齐
        
    def encode_hierarchical(self, audio):
        # 声学 tokens (w2v-BERT)
        acoustic_tokens = self.w2v_music(audio)  # 50Hz
        
        # 语义 tokens (MuLaN)
        semantic_tokens = self.mulan.encode(audio)  # 1Hz
        
        # 层次化表示
        return {
            "fine": acoustic_tokens,    # 细粒度
            "coarse": semantic_tokens,   # 粗粒度
        }
```

**环境音事件检测**：

```python
class AudioEventDetector:
    def __init__(self, num_classes=527):  # AudioSet classes
        self.encoder = nn.Sequential(
            ConvBlock(1, 64, kernel=3, stride=2),
            ConvBlock(64, 128, kernel=3, stride=2),
            ConvBlock(128, 256, kernel=3, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, spectrogram):
        # 多尺度特征提取
        features = self.encoder(spectrogram)
        
        # 帧级预测
        frame_logits = self.classifier(features)  # [T, 527]
        
        # 聚合为片段级
        segment_logits = torch.max(frame_logits, dim=0)[0]
        return segment_logits
```

**音频字幕生成**：
```
输入：[狗叫声] + [汽车经过] + [雨声]
输出："雨天街道上，一只狗对着经过的汽车吠叫"

关键：时序关系建模 + 事件共现学习
```

### 5.3.3 音频编码器选择

不同音频编码器的特点与适用场景：

**主流编码器对比**：

| 编码器 | 预训练数据 | 特点 | 最佳场景 |
|--------|-----------|------|----------|
| Wav2Vec2 | 960h LibriSpeech | 自监督，CPC loss | 英文语音 |
| HuBERT | 60k h Libri-Light | Masked prediction | 多语言语音 |
| WavLM | 94k h 混合 | 去噪 + 掩码 | 噪声鲁棒 |
| Whisper | 680k h 标注 | 有监督，多任务 | 通用语音 |
| BEATs | AudioSet + 私有 | 音频 MAE | 通用音频 |
| CLAP | LAION-Audio-630K | 对比学习 | 音频-文本对齐 |

**选择决策树**：
```
需要语音识别？
├─是→ 需要多语言？
│     ├─是→ Whisper
│     └─否→ Wav2Vec2
└─否→ 需要文本描述？
      ├─是→ CLAP
      └─否→ BEATs
```

**特征提取层级**：
```python
def extract_hierarchical_features(encoder, audio):
    features = {}
    
    # Whisper 示例
    if isinstance(encoder, WhisperModel):
        x = encoder.encoder.conv1(audio)
        features['conv'] = x  # 早期声学特征
        
        x = encoder.encoder.conv2(x)
        x = x.permute(0, 2, 1)
        
        for i, layer in enumerate(encoder.encoder.blocks):
            x = layer(x)
            if i in [6, 12, 18, 24]:  # 选择性保存
                features[f'layer_{i}'] = x
    
    return features  # 多层级特征
```

### 5.3.4 时频域特征融合

有效融合时域和频域信息是音频理解的关键：

**特征提取管线**：

```python
class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mel_bins = 128
        self.hop_length = 160  # 10ms
        
    def extract_all_features(self, waveform):
        features = {}
        
        # 时域特征
        features['zcr'] = zero_crossing_rate(waveform)
        features['energy'] = short_time_energy(waveform)
        
        # 频域特征  
        stft = torch.stft(waveform, n_fft=400, hop_length=160)
        features['spectral_centroid'] = spectral_centroid(stft)
        features['spectral_rolloff'] = spectral_rolloff(stft)
        
        # 时频特征
        features['mel_spec'] = mel_spectrogram(waveform)
        features['mfcc'] = mfcc(waveform, n_mfcc=13)
        
        return features
```

**多尺度时间建模**：

```
短时窗口（10-30ms）：
└── 捕获音素、音高变化

中时窗口（100-500ms）：  
└── 捕获音节、节拍

长时窗口（1-5s）：
└── 捕获句子、乐句

超长窗口（>10s）：
└── 捕获段落、音乐结构
```

**注意力融合机制**：
```python
class TimeFreqAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, 8)
        self.freq_attn = nn.MultiheadAttention(dim, 8)
        self.fusion = nn.Linear(dim * 2, dim)
        
    def forward(self, x):
        # x: [Batch, Time, Freq, Dim]
        B, T, F, D = x.shape
        
        # 时间维度注意力
        x_t = x.mean(dim=2)  # [B, T, D]
        x_t = self.time_attn(x_t, x_t, x_t)[0]
        
        # 频率维度注意力  
        x_f = x.mean(dim=1)  # [B, F, D]
        x_f = self.freq_attn(x_f, x_f, x_f)[0]
        
        # 广播并融合
        x_t = x_t.unsqueeze(2).expand(-1, -1, F, -1)
        x_f = x_f.unsqueeze(1).expand(-1, T, -1, -1)
        
        x_fused = self.fusion(torch.cat([x_t, x_f], dim=-1))
        return x_fused
```

**实验优化技巧**：
- SpecAugment：时频域数据增强
- 混合精度：节省显存，加速训练
- 渐进式分辨率：从低分辨率开始训练

## 5.4 视频理解的时序建模

视频作为最复杂的多模态数据，不仅包含视觉和音频信息，还具有强烈的时序依赖性。有效的时序建模是视频理解的核心，决定了模型能否捕获动作、事件和叙事结构。

### 5.4.1 帧采样策略

视频的高维特性（典型 30fps）使得处理所有帧在计算上不可行。智能的帧采样策略需要在信息保留和计算效率间取得平衡。

**采样方法对比**：

```
均匀采样（Uniform Sampling）：
├── 实现：每隔 k 帧采样一次
├── 优点：简单，保持时序均匀性
├── 缺点：可能错过关键帧
└── 适用：动作均匀分布的视频

密集采样（Dense Sampling）：
├── 实现：在短时间窗口内密集采样
├── 优点：捕获细粒度动作
├── 缺点：长程依赖建模困难
└── 适用：动作识别任务

稀疏采样（Sparse Sampling）：
├── 实现：大间隔采样，覆盖整个视频
├── 优点：捕获全局结构
├── 缺点：丢失局部细节
└── 适用：视频分类、摘要

自适应采样（Adaptive Sampling）：
├── 实现：基于内容变化动态调整
├── 优点：信息保留最优
├── 缺点：计算开销大
└── 适用：事件检测、高光提取
```

**TSN（Temporal Segment Networks）采样**：
```python
def tsn_sampling(video_frames, num_segments=8, mode='uniform'):
    """TSN 的分段采样策略"""
    total_frames = len(video_frames)
    segment_len = total_frames // num_segments
    
    sampled_frames = []
    for i in range(num_segments):
        start = i * segment_len
        end = min((i + 1) * segment_len, total_frames)
        
        if mode == 'uniform':
            # 每段中间帧
            frame_idx = (start + end) // 2
        elif mode == 'random':
            # 每段随机采样
            frame_idx = random.randint(start, end - 1)
        elif mode == 'dense':
            # 每段采样多帧
            indices = np.linspace(start, end - 1, 3, dtype=int)
            sampled_frames.extend([video_frames[idx] for idx in indices])
            continue
            
        sampled_frames.append(video_frames[frame_idx])
    
    return sampled_frames
```

**时序jittering增强**：
```python
class TemporalJitter:
    def __init__(self, max_jitter=3):
        self.max_jitter = max_jitter
    
    def __call__(self, frame_indices):
        """训练时添加时序扰动，提升泛化"""
        jittered = []
        for idx in frame_indices:
            offset = random.randint(-self.max_jitter, self.max_jitter)
            new_idx = np.clip(idx + offset, 0, max(frame_indices))
            jittered.append(new_idx)
        return jittered
```

**关键帧检测采样**：
```python
def keyframe_sampling(video, threshold=0.3):
    """基于内容变化的关键帧采样"""
    frames = []
    prev_frame = video[0]
    frames.append(prev_frame)
    
    for frame in video[1:]:
        # 计算帧间差异
        diff = compute_frame_difference(prev_frame, frame)
        
        if diff > threshold:
            frames.append(frame)
            prev_frame = frame
    
    return frames

def compute_frame_difference(frame1, frame2):
    """使用直方图差异 + 光流幅度"""
    # 颜色直方图差异
    hist1 = cv2.calcHist([frame1], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist2 = cv2.calcHist([frame2], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    # 光流幅度（可选）
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
    
    return 0.5 * hist_diff + 0.5 * magnitude
```

**实验建议**：
- 短视频（<10s）：8-16 帧足够
- 长视频（>60s）：32-64 帧，分层采样
- 动作识别：密集采样 + TSM（Temporal Shift Module）
- 视频问答：稀疏采样 + 关键帧检测

### 5.4.2 时序注意力机制

时序注意力是视频理解的核心，需要有效建模帧间关系while控制计算复杂度。

**时空注意力分解**：
```python
class SpaceTimeAttention(nn.Module):
    """时空注意力分解，降低复杂度 O(T²S²) → O(T²+S²)"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: [B, T, S, D] - Batch, Time, Space, Dim
        B, T, S, D = x.shape
        
        # 空间注意力（独立处理每帧）
        x_spatial = x.reshape(B*T, S, D)
        x_spatial = self.spatial_attn(x_spatial, x_spatial, x_spatial)[0]
        x_spatial = self.norm1(x_spatial + x.reshape(B*T, S, D))
        x_spatial = x_spatial.reshape(B, T, S, D)
        
        # 时间注意力（跨帧建模）
        x_temporal = x_spatial.permute(0, 2, 1, 3).reshape(B*S, T, D)
        x_temporal = self.temporal_attn(x_temporal, x_temporal, x_temporal)[0]
        x_temporal = self.norm2(x_temporal + x_temporal)
        x_temporal = x_temporal.reshape(B, S, T, D).permute(0, 2, 1, 3)
        
        return x_temporal
```

**TimeSformer架构变体**：
```
Divided Attention（分离注意力）：
Space Attn → Time Attn
复杂度：O(TS²) + O(T²S)

Joint Attention（联合注意力）：
SpaceTime Attn together
复杂度：O((TS)²) - 计算密集

Axial Attention（轴向注意力）：
Height → Width → Time
复杂度：O(TH²W) + O(THW²) + O(T²HW)
```

**局部时序注意力**：
```python
class LocalTemporalAttention(nn.Module):
    """局部窗口时序注意力，降低长视频复杂度"""
    def __init__(self, dim, window_size=5):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, 8)
        
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        output = torch.zeros_like(x)
        
        # 滑动窗口注意力
        for t in range(T):
            start = max(0, t - self.window_size // 2)
            end = min(T, t + self.window_size // 2 + 1)
            
            window = x[:, start:end]  # [B, W, D]
            center_idx = t - start
            
            # 计算窗口内注意力
            attn_out = self.attn(
                window[:, center_idx:center_idx+1],  # Query
                window,  # Key
                window   # Value
            )[0]
            
            output[:, t] = attn_out.squeeze(1)
        
        return output
```

**时序位置编码策略**：
```python
def temporal_position_encoding(num_frames, dim, max_period=10000):
    """为视频帧生成时序位置编码"""
    position = torch.arange(num_frames).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(max_period) / dim))
    
    pe = torch.zeros(num_frames, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # 添加帧率自适应
    if frame_rate != 30:  # 假设30fps为基准
        scale = 30.0 / frame_rate
        pe = pe * scale
    
    return pe
```

### 5.4.3 长视频处理优化

长视频（>5分钟）带来严重的内存和计算挑战，需要特殊的优化策略。

**层次化处理架构**：
```python
class HierarchicalVideoModel(nn.Module):
    """层次化长视频处理"""
    def __init__(self, clip_len=16, stride=8):
        super().__init__()
        self.clip_len = clip_len
        self.stride = stride
        
        # 局部编码器（处理短片段）
        self.local_encoder = VideoEncoder(num_frames=clip_len)
        
        # 全局聚合器（融合片段特征）
        self.global_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=4
        )
        
    def forward(self, video):
        # video: [B, T_total, H, W, C]
        clips = self.extract_clips(video)  # [B, N_clips, clip_len, H, W, C]
        
        # 编码每个片段
        clip_features = []
        for i in range(clips.shape[1]):
            feat = self.local_encoder(clips[:, i])  # [B, D]
            clip_features.append(feat)
        
        clip_features = torch.stack(clip_features, dim=1)  # [B, N_clips, D]
        
        # 全局聚合
        global_features = self.global_aggregator(clip_features)
        return global_features
```

**内存优化技术**：
```python
class MemoryEfficientVideoProcessor:
    def __init__(self, chunk_size=32):
        self.chunk_size = chunk_size
        
    def process_video_streaming(self, video_path, model):
        """流式处理，避免一次性加载全部帧"""
        cap = cv2.VideoCapture(video_path)
        features = []
        
        chunk = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            chunk.append(frame)
            
            if len(chunk) == self.chunk_size:
                # 处理当前chunk
                with torch.no_grad():
                    chunk_tensor = preprocess_frames(chunk)
                    chunk_features = model.encode(chunk_tensor)
                    features.append(chunk_features.cpu())
                
                # 清空chunk，保留部分重叠
                chunk = chunk[-8:]  # 保留8帧重叠
        
        # 处理剩余帧
        if chunk:
            chunk_tensor = preprocess_frames(chunk)
            chunk_features = model.encode(chunk_tensor)
            features.append(chunk_features.cpu())
        
        return torch.cat(features, dim=0)
```

**Token压缩策略**：
```python
class TokenMerging(nn.Module):
    """Token合并，减少序列长度"""
    def __init__(self, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        
    def forward(self, tokens, scores=None):
        # tokens: [B, N, D]
        B, N, D = tokens.shape
        num_keep = int(N * (1 - self.ratio))
        
        if scores is None:
            # 基于相似度的合并
            similarity = torch.cdist(tokens, tokens)
            scores = similarity.mean(dim=-1)  # 平均相似度
        
        # 保留重要tokens
        _, indices = scores.topk(num_keep, dim=1)
        kept_tokens = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        return kept_tokens
```

### 5.4.4 动作识别与事件定位

动作识别和时序定位是视频理解的核心任务，需要精确捕获动作边界和语义。

**双流网络（Two-Stream）架构**：
```python
class TwoStreamNetwork(nn.Module):
    """RGB流 + 光流流的经典架构"""
    def __init__(self, num_classes=400):
        super().__init__()
        # RGB流：外观信息
        self.rgb_stream = ResNet3D(input_channels=3)
        
        # 光流流：运动信息
        self.flow_stream = ResNet3D(input_channels=2)  # (u, v)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, rgb_frames, flow_frames):
        rgb_feat = self.rgb_stream(rgb_frames)
        flow_feat = self.flow_stream(flow_frames)
        
        # 晚期融合
        fused = torch.cat([rgb_feat, flow_feat], dim=1)
        return self.fusion(fused)
```

**时序动作定位（TAL）**：
```python
class TemporalActionLocalization(nn.Module):
    """时序动作定位，输出动作类别和时间边界"""
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = I3D()  # 3D CNN backbone
        self.classifier = nn.Conv1d(1024, num_classes, 1)
        self.regressor = nn.Conv1d(1024, 2, 1)  # 起止时间回归
        
    def forward(self, video):
        # 提取时序特征
        features = self.backbone(video)  # [B, C, T]
        
        # 逐帧分类
        class_scores = self.classifier(features)  # [B, num_classes, T]
        
        # 边界回归
        boundaries = self.regressor(features)  # [B, 2, T]
        start_offsets = boundaries[:, 0]
        end_offsets = boundaries[:, 1]
        
        return {
            'class_scores': class_scores,
            'start_offsets': start_offsets,
            'end_offsets': end_offsets
        }
```

**动作质量评估（AQA）**：
```python
class ActionQualityAssessment(nn.Module):
    """评估动作执行质量，如体操、跳水评分"""
    def __init__(self):
        super().__init__()
        self.encoder = VideoEncoder()
        
        # 多任务头
        self.score_head = nn.Linear(768, 1)  # 分数回归
        self.rank_head = nn.Linear(768, 128)  # 排序学习
        
    def forward(self, video, pairs=None):
        features = self.encoder(video)
        
        # 直接分数预测
        scores = self.score_head(features)
        
        # 相对排序学习
        if pairs is not None:
            feat_a, feat_b = pairs
            rank_a = self.rank_head(feat_a)
            rank_b = self.rank_head(feat_b)
            margin = torch.sigmoid(torch.sum(rank_a - rank_b, dim=-1))
            return scores, margin
        
        return scores
```

**实验优化要点**：
- 预训练初始化：Kinetics-400/600 预训练权重
- 多尺度测试：1x, 1.25x, 1.5x 尺度融合
- 时序增强：速度扰动 [0.5x, 2x]
- 类别平衡：Focal Loss 处理长尾分布

## 5.5 跨模态注意力机制设计

跨模态注意力是多模态模型的核心组件，决定了不同模态信息如何有效交互和融合。设计高效的跨模态注意力机制需要平衡表达能力、计算效率和训练稳定性。

### 5.5.1 早期融合 vs 晚期融合

融合时机的选择深刻影响模型的表达能力和计算效率：

**融合策略对比**：

```
早期融合（Early Fusion）：
输入层 → [Concat/Add] → 统一处理
├── 优点：充分交互，参数共享
├── 缺点：模态差异大，优化困难
└── 适用：模态相似度高的任务

晚期融合（Late Fusion）：
独立编码 → 高层特征 → [Fusion] → 输出
├── 优点：模块化，易于优化
├── 缺点：交互不充分，参数冗余
└── 适用：模态独立性强的任务

渐进融合（Progressive Fusion）：
多层级融合，逐步加深交互
├── 优点：平衡早期和晚期优势
├── 缺点：架构复杂，调参困难
└── 适用：复杂多模态理解任务
```

**早期融合实现**：
```python
class EarlyFusion(nn.Module):
    """早期融合：直接拼接输入"""
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=1024):
        super().__init__()
        # 投影到相同维度
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 融合后的处理
        self.fusion_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim * 2, nhead=8),
            num_layers=6
        )
        
    def forward(self, vision_features, text_features):
        # 投影
        v = self.vision_proj(vision_features)  # [B, Nv, D]
        t = self.text_proj(text_features)      # [B, Nt, D]
        
        # 拼接
        fused = torch.cat([v, t], dim=1)  # [B, Nv+Nt, D]
        
        # 统一处理
        output = self.fusion_layers(fused)
        return output
```

**晚期融合实现**：
```python
class LateFusion(nn.Module):
    """晚期融合：独立处理后融合"""
    def __init__(self):
        super().__init__()
        # 独立编码器
        self.vision_encoder = VisionTransformer()
        self.text_encoder = TextTransformer()
        
        # 融合策略
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768)
        )
        
    def forward(self, images, texts):
        # 独立编码
        v_out = self.vision_encoder(images).mean(dim=1)  # [B, D]
        t_out = self.text_encoder(texts).mean(dim=1)     # [B, D]
        
        # 高层融合
        fused = self.fusion(torch.cat([v_out, t_out], dim=-1))
        return fused
```

**渐进融合架构**：
```python
class ProgressiveFusion(nn.Module):
    """渐进式多层融合"""
    def __init__(self, num_layers=12):
        super().__init__()
        self.num_layers = num_layers
        
        # 每3层进行一次融合
        self.fusion_points = [3, 6, 9]
        
        self.vision_layers = nn.ModuleList([
            TransformerLayer(768) for _ in range(num_layers)
        ])
        self.text_layers = nn.ModuleList([
            TransformerLayer(768) for _ in range(num_layers)
        ])
        
        # 跨模态融合层
        self.cross_attention = nn.ModuleList([
            CrossModalAttention(768) for _ in range(len(self.fusion_points))
        ])
        
    def forward(self, v_input, t_input):
        fusion_idx = 0
        
        for i in range(self.num_layers):
            # 独立处理
            v_input = self.vision_layers[i](v_input)
            t_input = self.text_layers[i](t_input)
            
            # 在融合点进行跨模态交互
            if i + 1 in self.fusion_points:
                v_input, t_input = self.cross_attention[fusion_idx](v_input, t_input)
                fusion_idx += 1
        
        return v_input, t_input
```

### 5.5.2 交叉注意力优化

交叉注意力是实现模态间信息交换的核心机制：

**标准交叉注意力**：
```python
class CrossModalAttention(nn.Module):
    """双向交叉注意力"""
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(dim, num_heads)
        self.t2v_attn = nn.MultiheadAttention(dim, num_heads)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        
    def forward(self, vision, text):
        # Vision attending to Text
        v2t = self.v2t_attn(
            query=vision,
            key=text,
            value=text
        )[0]
        vision = self.norm_v(vision + v2t)
        
        # Text attending to Vision
        t2v = self.t2v_attn(
            query=text,
            key=vision,
            value=vision
        )[0]
        text = self.norm_t(text + t2v)
        
        return vision, text
```

**稀疏交叉注意力**（降低复杂度）：
```python
class SparseBlockAttention(nn.Module):
    """块稀疏模式的交叉注意力"""
    def __init__(self, dim=768, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.attention = nn.MultiheadAttention(dim, 8)
        
    def forward(self, query, key_value):
        B, N, D = query.shape
        _, M, _ = key_value.shape
        
        # 将序列分块
        num_blocks = N // self.block_size
        query_blocks = query.reshape(B, num_blocks, self.block_size, D)
        
        outputs = []
        for i in range(num_blocks):
            # 每个query块只attend到对应的key块
            start = i * self.block_size
            end = min((i + 1) * self.block_size, M)
            
            q_block = query_blocks[:, i]  # [B, block_size, D]
            kv_block = key_value[:, start:end]  # [B, block_size, D]
            
            out = self.attention(q_block, kv_block, kv_block)[0]
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)
```

**门控交叉注意力**：
```python
class GatedCrossAttention(nn.Module):
    """使用门控机制的交叉注意力"""
    def __init__(self, dim=768):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 8)
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, query, key_value):
        # 计算注意力
        attn_out = self.attention(query, key_value, key_value)[0]
        
        # 计算门控权重
        gate_input = torch.cat([query, attn_out], dim=-1)
        gate = self.gate_net(gate_input)
        
        # 门控输出
        output = gate * attn_out + (1 - gate) * query
        return output
```

**层次化注意力池化**：
```python
class HierarchicalAttentionPooling(nn.Module):
    """多粒度注意力池化"""
    def __init__(self, dim=768, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        
        # 不同粒度的池化头
        self.pool_heads = nn.ModuleList([
            nn.Linear(dim, dim // num_levels) 
            for _ in range(num_levels)
        ])
        
        # 粒度特定的注意力
        self.level_attention = nn.ModuleList([
            nn.MultiheadAttention(dim // num_levels, 4)
            for _ in range(num_levels)
        ])
        
    def forward(self, features):
        # features: [B, N, D]
        B, N, D = features.shape
        
        pooled_features = []
        for i in range(self.num_levels):
            # 不同步长的池化
            stride = 2 ** i
            pooled = features[:, ::stride]  # 降采样
            
            # 投影到子空间
            pooled = self.pool_heads[i](pooled)
            
            # 层级内注意力
            attended = self.level_attention[i](pooled, pooled, pooled)[0]
            pooled_features.append(attended.mean(dim=1))  # [B, D/num_levels]
        
        # 拼接多粒度特征
        return torch.cat(pooled_features, dim=-1)  # [B, D]
```

### 5.5.3 模态专家混合（MoME）

模态专家混合通过为不同模态分配专门的处理路径，提升模型的专业化能力：

**MoME 架构**：
```python
class ModalityMixtureOfExperts(nn.Module):
    """模态感知的专家混合"""
    def __init__(self, dim=768, num_experts=8, num_modalities=3):
        super().__init__()
        self.num_experts = num_experts
        self.num_modalities = num_modalities
        
        # 专家网络
        self.experts = nn.ModuleList([
            FeedForward(dim, dim * 4, dim)
            for _ in range(num_experts)
        ])
        
        # 模态特定的路由网络
        self.routers = nn.ModuleDict({
            'vision': nn.Linear(dim, num_experts),
            'text': nn.Linear(dim, num_experts),
            'audio': nn.Linear(dim, num_experts)
        })
        
        # Top-k 选择
        self.top_k = 2
        
    def forward(self, x, modality='vision'):
        # 计算路由权重
        router = self.routers[modality]
        routing_weights = F.softmax(router(x.mean(dim=1)), dim=-1)  # [B, num_experts]
        
        # 选择 top-k 专家
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 应用专家
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            
            # 批量处理每个专家
            for b in range(x.size(0)):
                expert = self.experts[expert_idx[b]]
                output[b] += weight[b] * expert(x[b])
        
        return output
```

**动态专家分配**：
```python
class DynamicExpertAllocation(nn.Module):
    """基于输入内容动态分配专家"""
    def __init__(self, dim=768, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        
        # 专家池
        self.experts = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_experts)
        ])
        
        # 负载均衡损失权重
        self.load_balance_weight = 0.01
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_experts)
        )
        
    def forward(self, x):
        B, N, D = x.shape
        
        # 计算每个token的路由
        router_logits = self.router(x)  # [B, N, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Gumbel-Softmax 采样（训练时）
        if self.training:
            router_probs = F.gumbel_softmax(router_logits, tau=1.0, hard=True)
        
        # 负载均衡损失
        expert_usage = router_probs.sum(dim=[0, 1]) / (B * N)
        load_balance_loss = self.load_balance_weight * (
            expert_usage.var() + (1.0 / self.num_experts - expert_usage).abs().mean()
        )
        
        # 应用专家
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = router_probs[:, :, i:i+1]  # [B, N, 1]
            output += mask * expert(x)
        
        return output, load_balance_loss
```

**模态特定专家设计**：
```python
class ModalitySpecificExperts(nn.Module):
    """为每个模态设计专门的专家"""
    def __init__(self, dim=768):
        super().__init__()
        
        # 视觉专家：擅长空间关系
        self.vision_expert = nn.Sequential(
            Conv2dAdapter(dim),  # 2D 卷积适配
            SpatialAttention(dim),
            FeedForward(dim, dim * 4, dim)
        )
        
        # 文本专家：擅长序列建模
        self.text_expert = nn.Sequential(
            PositionalEncoding(dim),
            CausalAttention(dim),  # 因果注意力
            FeedForward(dim, dim * 4, dim)
        )
        
        # 音频专家：擅长时频分析
        self.audio_expert = nn.Sequential(
            SpectralGating(dim),  # 频域门控
            TemporalConvNet(dim),
            FeedForward(dim, dim * 4, dim)
        )
        
        # 通用专家：处理跨模态信息
        self.general_expert = TransformerBlock(dim)
        
    def forward(self, x, modality_mask):
        """
        modality_mask: [B, N] 指示每个token的模态
        0: vision, 1: text, 2: audio, 3: mixed
        """
        output = torch.zeros_like(x)
        
        # 应用模态特定专家
        for modality, expert in enumerate([
            self.vision_expert,
            self.text_expert,
            self.audio_expert,
            self.general_expert
        ]):
            mask = (modality_mask == modality).unsqueeze(-1)
            output += mask * expert(x)
        
        return output
```

### 5.5.4 计算效率优化

跨模态注意力的计算开销巨大，优化策略至关重要：

**Flash Attention 集成**：
```python
class FlashCrossAttention(nn.Module):
    """使用 Flash Attention 的跨模态注意力"""
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key_value):
        B, N, _ = query.shape
        _, M, _ = key_value.shape
        
        # 投影
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim)
        kv = self.kv_proj(key_value).reshape(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        
        # Flash Attention (需要特定硬件支持)
        if torch.cuda.is_available():
            from flash_attn import flash_attn_func
            out = flash_attn_func(q, k, v, dropout_p=0.1 if self.training else 0.0)
        else:
            # Fallback to standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        
        out = out.reshape(B, N, -1)
        return self.out_proj(out)
```

**低秩分解优化**：
```python
class LowRankCrossAttention(nn.Module):
    """低秩分解的交叉注意力"""
    def __init__(self, dim=768, rank=64):
        super().__init__()
        self.rank = rank
        
        # 低秩投影
        self.q_down = nn.Linear(dim, rank, bias=False)
        self.q_up = nn.Linear(rank, dim, bias=False)
        
        self.kv_down = nn.Linear(dim, rank * 2, bias=False)
        self.kv_up = nn.Linear(rank * 2, dim * 2, bias=False)
        
        # 标准注意力（低维）
        self.attention = nn.MultiheadAttention(rank, 4)
        
    def forward(self, query, key_value):
        # 降维
        q_low = self.q_down(query)  # [B, N, rank]
        kv_low = self.kv_down(key_value)  # [B, M, rank*2]
        k_low, v_low = kv_low.chunk(2, dim=-1)
        
        # 低维注意力
        attn_out = self.attention(q_low, k_low, v_low)[0]
        
        # 升维
        output = self.q_up(attn_out)
        
        # 残差连接
        return output + query
```

**计算复用策略**：
```python
class ComputeReuseAttention(nn.Module):
    """复用计算结果的注意力"""
    def __init__(self, dim=768):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 8)
        self.cache = {}
        
    def forward(self, query, key_value, cache_key=None):
        if cache_key is not None and cache_key in self.cache:
            # 复用缓存的key/value投影
            k_cached, v_cached = self.cache[cache_key]
            output = self.attention.forward(
                query=query,
                key=k_cached,
                value=v_cached,
                need_weights=False
            )[0]
        else:
            # 正常计算
            output = self.attention(query, key_value, key_value)[0]
            
            # 缓存中间结果
            if cache_key is not None:
                with torch.no_grad():
                    k = self.attention.in_proj_weight[:dim] @ key_value.transpose(-2, -1)
                    v = self.attention.in_proj_weight[dim:2*dim] @ key_value.transpose(-2, -1)
                    self.cache[cache_key] = (k.transpose(-2, -1), v.transpose(-2, -1))
        
        return output
    
    def clear_cache(self):
        self.cache.clear()
```

**量化感知训练**：
```python
class QuantizedCrossAttention(nn.Module):
    """支持量化的交叉注意力"""
    def __init__(self, dim=768):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 量化感知层
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.attention = nn.MultiheadAttention(dim, 8)
        
    def forward(self, query, key_value):
        # 量化输入
        query = self.quant(query)
        key_value = self.quant(key_value)
        
        # 执行注意力（量化）
        output = self.attention(query, key_value, key_value)[0]
        
        # 反量化输出
        output = self.dequant(output)
        
        return output
    
    def prepare_quantization(self):
        """准备量化"""
        torch.quantization.prepare_qat(self, inplace=True)
    
    def convert_quantization(self):
        """转换为量化模型"""
        torch.quantization.convert(self, inplace=True)
```

## 本章小结

本章系统介绍了多模态任务的实验设计方法，涵盖了从基础的视觉-语言对齐到复杂的跨模态注意力机制设计。核心要点包括：

**📌 关键概念回顾**：
1. **视觉-语言对齐**：CLIP 的对比学习范式奠定基础，关键在于大批量训练和温度参数调优
2. **统一建模架构**：Perceiver Resampler 等连接层设计实现了高效的模态桥接
3. **音频集成策略**：时频域特征融合与层次化编码处理多样化音频信号
4. **视频时序建模**：帧采样策略与时空注意力分解平衡效率与效果
5. **跨模态注意力**：早期vs晚期融合、MoME专家混合、计算效率优化

**💡 实用公式总结**：
- InfoNCE 损失：$\mathcal{L} = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$
- 时空复杂度优化：$O(T^2S^2) \rightarrow O(T^2 + S^2)$
- 负载均衡约束：$\mathcal{L}_{balance} = \text{Var}(usage) + ||\frac{1}{K} - usage||_1$

**🔬 进阶探索方向**：
- 3D 视觉理解的多视角融合
- 实时多模态流处理优化
- 神经架构搜索（NAS）用于跨模态设计
- 模态缺失情况下的鲁棒推理

## 练习题

### 基础题

**练习 5.1**：实现一个简化版的 CLIP 模型，包括图像编码器、文本编码器和对比损失。
- *Hint*：使用预训练的 ResNet50 和 BERT-base，关注投影层设计
<details>
<summary>参考答案</summary>

关键实现要点：
1. 图像编码器输出需要全局池化得到 [B, D] 特征
2. 文本编码器使用 [CLS] token 或平均池化
3. 投影到相同维度空间（如 512）
4. 对比损失需要计算 batch 内所有配对的相似度
5. 温度参数初始化为 0.07，可学习
</details>

**练习 5.2**：设计一个帧采样策略，从 10 分钟的视频中采样 32 帧用于动作识别。
- *Hint*：考虑 TSN 的分段采样思想
<details>
<summary>参考答案</summary>

建议策略：
1. 将视频分为 32 个等长片段
2. 每个片段随机采样 1 帧（训练时）或中间帧（测试时）
3. 对于动作密集区域，可使用光流幅度加权采样
4. 保持最小 0.5 秒间隔避免冗余
</details>

**练习 5.3**：计算 ViT-L/14 处理 224×224 图像时产生的 token 数量。
- *Hint*：patch size = 14
<details>
<summary>参考答案</summary>

计算过程：
- 图像尺寸：224 × 224
- Patch 尺寸：14 × 14
- 每个维度的 patch 数：224 / 14 = 16
- 总 patch 数：16 × 16 = 256
- 加上 [CLS] token：256 + 1 = 257 tokens
</details>

**练习 5.4**：比较早期融合和晚期融合在参数量和计算量上的差异。
- *Hint*：假设视觉和文本序列长度分别为 Nv 和 Nt
<details>
<summary>参考答案</summary>

分析：
- 早期融合：
  - 序列长度：Nv + Nt
  - 自注意力复杂度：O((Nv + Nt)²)
  - 参数共享，总参数量较少
  
- 晚期融合：
  - 独立处理复杂度：O(Nv²) + O(Nt²)
  - 参数量约为早期融合的 2 倍
  - 当 Nv ≈ Nt 时，计算量约为早期融合的 50%
</details>

### 挑战题

**练习 5.5**：设计一个自适应的模态专家分配策略，根据输入动态决定使用哪些专家。
- *Hint*：考虑稀疏激活和负载均衡
<details>
<summary>参考答案</summary>

设计要点：
1. 使用可学习的路由网络，输出每个专家的分数
2. Top-k 选择（k=2），保持稀疏性
3. 添加负载均衡损失：鼓励均匀使用所有专家
4. 引入容量限制：每个专家处理的 token 数上限
5. 使用 Gumbel-Softmax 实现可微分的离散选择
6. 辅助损失：最小化未使用专家的比例
</details>

**练习 5.6**：如何处理视频中的音画不同步问题？设计一个对齐机制。
- *Hint*：考虑学习时间偏移
<details>
<summary>参考答案</summary>

解决方案：
1. 可学习的时间偏移预测器，输出音频相对视频的偏移量
2. 使用互相关计算音视频特征的最佳对齐点
3. 循环一致性约束：视频→音频→视频应该回到原点
4. 对比学习：同步的音视频对作为正样本
5. 滑动窗口注意力，允许 ±2 秒的偏移搜索
6. 数据增强：训练时人为引入时间偏移
</details>

**练习 5.7**：设计一个 token 压缩策略，将 1024 个视觉 tokens 压缩到 64 个。
- *Hint*：考虑语义聚类和重要性评分
<details>
<summary>参考答案</summary>

方案设计：
1. 学习 64 个可学习的聚类中心
2. 计算每个 token 到聚类中心的相似度
3. Soft assignment：每个 token 软分配到多个中心
4. 加权聚合：根据分配权重聚合 tokens
5. 保留部分原始重要 tokens（如 [CLS]）
6. 渐进压缩：1024→256→64，避免信息损失过快
7. 辅助重建损失：压缩后应能重建原始特征
</details>

**练习 5.8**：如何在保持性能的前提下，将跨模态注意力的内存占用降低 75%？
- *Hint*：结合多种优化技术
<details>
<summary>参考答案</summary>

综合优化策略：
1. **Flash Attention**：融合计算，减少中间结果存储（-50%）
2. **低秩分解**：将 768 维降到 128 维计算（-70%）
3. **块稀疏模式**：只计算局部和全局注意力（-60%）
4. **梯度检查点**：用计算换内存（-40%）
5. **混合精度**：FP16 计算，FP32 累加（-50%）
6. **KV 缓存复用**：跨层共享 key-value（-30%）

组合使用可达到 75% 以上的内存节省，性能损失 <2%。
</details>

## 常见陷阱与错误

⚠️ **常见错误与调试技巧**：

1. **模态不平衡问题**
   - 错误：一个模态主导，其他模态被忽略
   - 解决：分别计算每个模态的梯度范数，动态调整学习率

2. **视觉 Token 数量爆炸**
   - 错误：高分辨率图像产生过多 tokens
   - 解决：使用 Perceiver 降维或分层处理

3. **音视频不同步**
   - 错误：假设音视频完全对齐
   - 解决：允许可学习的时间偏移，使用 DTW 对齐

4. **负样本采样偏差**
   - 错误：负样本过于简单或过于困难
   - 解决：维护难度分布，curriculum 采样

5. **跨模态梯度不稳定**
   - 错误：不同模态梯度尺度差异大
   - 解决：分模态的 gradient clipping 和归一化

6. **推理速度瓶颈**
   - 错误：直接使用训练架构部署
   - 解决：知识蒸馏到轻量级学生模型

7. **数据泄露风险**
   - 错误：测试集的音视频对出现在训练集
   - 解决：基于视频 ID 而非帧级别划分数据集

8. **计算图内存泄露**
   - 错误：保存了过多的中间激活值
   - 解决：及时 detach()，使用 gradient checkpointing