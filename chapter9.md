# 第九章：生产部署与监控

将后训练模型从实验环境部署到生产系统是一个复杂的工程挑战。本章系统介绍模型压缩、服务化架构、实时监控、漂移检测以及发布策略等关键技术，帮助您构建稳定、高效、可维护的 LLM 生产系统。

## 9.1 模型压缩与加速技术

### 9.1.1 量化技术

量化是将模型权重和激活值从高精度（如 FP32）转换为低精度（如 INT8、INT4）的过程，可以显著减少模型大小和推理延迟。

#### 训练后量化（Post-Training Quantization, PTQ）

PTQ 在训练完成后对模型进行量化，无需重新训练：

```
原始权重 W ∈ [-α, α]
量化过程：W_q = round(W × s / α) 
反量化：W' = W_q × α / s

其中 s = 2^(b-1) - 1，b 为量化位数
```

**关键技术要点**：

1. **对称 vs 非对称量化**
   - 对称：零点固定在 0，实现简单但精度略低
   - 非对称：引入零点偏移，精度更高但计算复杂

2. **逐层 vs 逐通道量化**
   - 逐层：整层共享量化参数，压缩率高
   - 逐通道：每个通道独立量化，精度保持更好

3. **动态 vs 静态量化**
   - 静态：量化参数预先确定，推理速度快
   - 动态：运行时计算量化参数，精度更高

#### 量化感知训练（Quantization-Aware Training, QAT）

QAT 在训练过程中模拟量化效果，使模型适应低精度表示：

```
前向传播：使用量化权重
反向传播：使用全精度梯度更新

伪量化操作：
W_fake_quant = dequant(quant(W))
```

**实践技巧**：
- 💡 从 INT8 开始尝试，多数任务精度损失 < 1%
- 💡 关键层（如第一层、最后一层）保持高精度
- 💡 使用校准数据集优化量化参数

### 9.1.2 知识蒸馏

通过教师-学生框架将大模型知识迁移到小模型：

```
损失函数：
L = α × L_CE(y_student, y_true) + 
    β × L_KL(σ(z_student/T), σ(z_teacher/T))

其中：
- L_CE：交叉熵损失（硬标签）
- L_KL：KL 散度（软标签）
- T：温度参数，控制分布平滑度
- α, β：损失权重
```

**蒸馏策略优化**：

1. **逐层蒸馏**：不仅蒸馏最终输出，还对齐中间层表示
2. **注意力蒸馏**：传递注意力模式
3. **特征蒸馏**：对齐隐层特征分布

### 9.1.3 模型剪枝

移除冗余参数以减小模型大小：

#### 结构化剪枝
```
重要性评分：
- 权重幅度：|W|
- 梯度幅度：|∂L/∂W|
- Taylor 展开：ΔL ≈ |W × ∂L/∂W|

剪枝流程：
1. 训练基线模型
2. 计算重要性分数
3. 移除低分神经元/通道/层
4. 微调恢复性能
```

#### 非结构化剪枝
```
稀疏掩码：M ∈ {0,1}^d
稀疏权重：W_sparse = W ⊙ M

动态稀疏训练：
- 周期性更新掩码
- 保持固定稀疏度
```

⚠️ **常见陷阱**：
- 非结构化剪枝虽然压缩率高，但硬件加速困难
- 过度剪枝导致不可恢复的性能损失
- 不同任务对剪枝的敏感度差异巨大

### 9.1.4 推理优化技术

#### Flash Attention
减少注意力计算的内存访问：

```
标准注意力：O(N²) 内存
Flash Attention：O(N) 内存

通过分块计算和融合 kernel 实现：
- 减少 HBM 访问
- 提高 SRAM 利用率
```

#### KV Cache 优化
```
策略：
1. 多查询注意力（MQA）：共享 K、V 投影
2. 分组查询注意力（GQA）：K、V 头数 < Q 头数
3. 滑动窗口：限制注意力范围

内存计算：
Cache_size = batch × seq_len × n_layers × 
             (n_kv_heads × d_head) × 2 × dtype_size
```

#### 批处理优化
```
动态批处理：
- 连续批处理（Continuous Batching）
- 填充优化（Padding Optimization）
- 序列并行（Sequence Parallelism）

吞吐量优化：
Throughput = batch_size / latency
找到最优 batch_size 平衡延迟和吞吐
```

## 9.2 服务化架构设计

### 9.2.1 微服务架构

```
     ┌─────────────┐
     │   Gateway   │
     └──────┬──────┘
            │
    ┌───────┼───────┐
    │       │       │
┌───▼───┐ ┌▼────┐ ┌▼─────┐
│Router │ │Auth │ │Rate  │
│       │ │     │ │Limit │
└───┬───┘ └─────┘ └──────┘
    │
    ├──────────┬─────────┬──────────┐
    │          │         │          │
┌───▼───┐ ┌───▼──┐ ┌───▼───┐ ┌────▼────┐
│Model  │ │Model │ │Cache  │ │Monitor  │
│Server │ │Pool  │ │Service│ │Service  │
└───────┘ └──────┘ └───────┘ └─────────┘
```

**关键组件设计**：

1. **API Gateway**
   - 请求路由与负载均衡
   - 协议转换（HTTP/gRPC/WebSocket）
   - 认证授权

2. **Model Server**
   - 模型加载与版本管理
   - 批处理队列
   - 资源隔离

3. **Cache Layer**
   - 结果缓存（Redis/Memcached）
   - 嵌入向量缓存
   - KV Cache 共享

### 9.2.2 模型服务框架

#### TorchServe 架构
```python
# 模型处理器示例
class LLMHandler(BaseHandler):
    def initialize(self, context):
        self.model = load_model(context.model_dir)
        self.tokenizer = load_tokenizer()
        
    def preprocess(self, requests):
        # 批处理预处理
        texts = [req.get("text") for req in requests]
        return self.tokenizer(texts, return_tensors="pt")
    
    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return outputs
    
    def postprocess(self, outputs):
        # 解码和格式化
        return self.tokenizer.batch_decode(outputs)
```

#### Triton Inference Server
```
模型配置：
name: "llm_model"
platform: "pytorch_libtorch"
max_batch_size: 32
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [-1, -1]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

### 9.2.3 流式生成架构

处理 LLM 逐 token 生成的特殊需求：

```python
# SSE (Server-Sent Events) 实现
async def stream_generate(request):
    async def generate():
        for token in model.generate_stream(request.text):
            yield f"data: {json.dumps({'token': token})}\n\n"
            
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# WebSocket 实现
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            async for token in model.generate_stream(data):
                await websocket.send_json({"token": token})
    except WebSocketDisconnect:
        pass
```

**流式处理优化**：
- 使用环形缓冲区管理 token 流
- 实现背压（Backpressure）机制
- 支持中断和恢复

### 9.2.4 高可用设计

#### 多副本部署
```yaml
# Kubernetes 部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: model-server
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
          periodSeconds: 10
```

#### 故障转移策略
```
主备模式：
Primary ──┐
          ├──> Load Balancer ──> Client
Secondary ┘    (Health Check)

负载均衡算法：
- 轮询（Round Robin）
- 最少连接（Least Connections）
- 一致性哈希（Consistent Hashing）
- 基于延迟的路由
```

## 9.3 实时监控与告警系统

### 9.3.1 指标体系设计

#### 系统指标
```python
# Prometheus 指标定义
from prometheus_client import Counter, Histogram, Gauge

# 请求指标
request_count = Counter(
    'llm_requests_total',
    'Total requests',
    ['model', 'status']
)

request_latency = Histogram(
    'llm_request_duration_seconds',
    'Request latency',
    ['model', 'operation']
)

# 资源指标
gpu_utilization = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization',
    ['device_id']
)

memory_usage = Gauge(
    'llm_memory_usage_bytes',
    'Memory usage',
    ['type']  # gpu_memory, cpu_memory
)
```

#### 业务指标
```python
# 质量指标
output_quality_score = Histogram(
    'llm_output_quality_score',
    'Output quality score from feedback',
    ['task_type']
)

# Token 统计
token_usage = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['type']  # input, output
)

# 成本指标
api_cost = Counter(
    'llm_api_cost_dollars',
    'API cost in dollars',
    ['model', 'customer']
)
```

### 9.3.2 日志聚合与分析

#### 结构化日志设计
```python
import structlog

logger = structlog.get_logger()

# 请求日志
logger.info(
    "request_received",
    request_id=request_id,
    user_id=user_id,
    model_version=model_version,
    input_tokens=len(tokens),
    timestamp=time.time()
)

# 推理日志
logger.info(
    "inference_completed",
    request_id=request_id,
    latency_ms=latency * 1000,
    output_tokens=len(output_tokens),
    gpu_memory_mb=gpu_memory,
    cache_hit=cache_hit
)

# 错误日志
logger.error(
    "inference_failed",
    request_id=request_id,
    error_type=type(e).__name__,
    error_message=str(e),
    traceback=traceback.format_exc()
)
```

#### ELK Stack 集成
```
日志流水线：
Application ──> Filebeat ──> Logstash ──> Elasticsearch ──> Kibana
                              │
                              ├─> 解析和增强
                              ├─> 过滤和路由
                              └─> 告警触发
```

### 9.3.3 分布式追踪

使用 OpenTelemetry 实现全链路追踪：

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)

@app.post("/generate")
async def generate(request: GenerateRequest):
    with tracer.start_as_current_span("generate_request") as span:
        span.set_attribute("user_id", request.user_id)
        span.set_attribute("model", request.model)
        
        with tracer.start_as_current_span("tokenize"):
            tokens = tokenizer.encode(request.text)
            
        with tracer.start_as_current_span("inference"):
            output = model.generate(tokens)
            
        with tracer.start_as_current_span("decode"):
            result = tokenizer.decode(output)
            
        return result
```

### 9.3.4 告警策略

#### 分级告警
```yaml
# AlertManager 配置
route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: pagerduty
    continue: true
  - match:
      severity: warning
    receiver: slack
    
# Prometheus 告警规则
groups:
- name: llm_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, llm_request_duration_seconds) > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 延迟超过 5 秒"
      
  - alert: GPUMemoryLeak
    expr: rate(llm_gpu_memory_bytes[5m]) > 100000000
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "GPU 内存持续增长"
```

#### 智能告警降噪
```python
class AlertThrottler:
    def __init__(self, window_seconds=300, max_alerts=10):
        self.window = window_seconds
        self.max_alerts = max_alerts
        self.alert_times = defaultdict(deque)
    
    def should_alert(self, alert_key):
        now = time.time()
        times = self.alert_times[alert_key]
        
        # 清理过期时间
        while times and times[0] < now - self.window:
            times.popleft()
        
        # 检查是否超过阈值
        if len(times) >= self.max_alerts:
            return False
            
        times.append(now)
        return True

## 9.4 模型漂移检测

模型漂移是指模型在生产环境中性能随时间下降的现象，可能由数据分布变化、用户行为演变或外部环境改变引起。

### 9.4.1 漂移类型与检测方法

#### 数据漂移（Data Drift）
输入数据分布的变化：

```python
# KL 散度检测
def kl_divergence(p, q, epsilon=1e-10):
    """计算两个分布的 KL 散度"""
    p = p + epsilon
    q = q + epsilon
    return np.sum(p * np.log(p / q))

# Kolmogorov-Smirnov 检验
from scipy.stats import ks_2samp

def ks_test_drift(reference_data, current_data, threshold=0.05):
    """使用 KS 检验检测分布变化"""
    statistic, p_value = ks_2samp(reference_data, current_data)
    return p_value < threshold, statistic

# 特征分布监控
class FeatureDriftMonitor:
    def __init__(self, reference_window=1000):
        self.reference_window = reference_window
        self.feature_buffers = defaultdict(deque)
        self.reference_stats = {}
    
    def update(self, features):
        for name, value in features.items():
            buffer = self.feature_buffers[name]
            buffer.append(value)
            if len(buffer) > self.reference_window:
                buffer.popleft()
    
    def detect_drift(self, sensitivity=2.0):
        drifts = {}
        for name, buffer in self.feature_buffers.items():
            if len(buffer) < 100:
                continue
                
            current_mean = np.mean(buffer)
            current_std = np.std(buffer)
            
            if name in self.reference_stats:
                ref_mean, ref_std = self.reference_stats[name]
                z_score = abs(current_mean - ref_mean) / (ref_std + 1e-10)
                drifts[name] = z_score > sensitivity
            else:
                self.reference_stats[name] = (current_mean, current_std)
                drifts[name] = False
                
        return drifts
```

#### 概念漂移（Concept Drift）
输入-输出关系的变化：

```python
# 滑动窗口性能监控
class ConceptDriftDetector:
    def __init__(self, window_size=100, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.performance_window = deque(maxlen=window_size)
        self.baseline_performance = None
    
    def update(self, prediction, ground_truth):
        """更新性能指标"""
        correct = (prediction == ground_truth)
        self.performance_window.append(correct)
        
        if len(self.performance_window) == self.window_size:
            current_accuracy = np.mean(self.performance_window)
            
            if self.baseline_performance is None:
                self.baseline_performance = current_accuracy
                return False
            
            drift = abs(current_accuracy - self.baseline_performance)
            return drift > self.threshold
        
        return False

# ADWIN (Adaptive Windowing) 算法
class ADWIN:
    def __init__(self, delta=0.002):
        self.delta = delta
        self.window = []
        self.total = 0
        self.variance = 0
        self.width = 0
    
    def update(self, value):
        """检测概念漂移"""
        self.window.append(value)
        self.total += value
        self.width += 1
        
        if self.width > 1:
            mean = self.total / self.width
            self.variance += (value - mean) ** 2
            
            # 检测两个子窗口的显著差异
            if self._detect_change():
                # 丢弃旧数据
                self._shrink_window()
                return True
        
        return False
    
    def _detect_change(self):
        """使用 Hoeffding 界检测变化"""
        if self.width < 4:
            return False
            
        for split_point in range(1, self.width):
            n0 = split_point
            n1 = self.width - split_point
            
            sum0 = sum(self.window[:split_point])
            sum1 = sum(self.window[split_point:])
            
            mean0 = sum0 / n0
            mean1 = sum1 / n1
            
            epsilon = np.sqrt(
                (1/(2*n0) + 1/(2*n1)) * 
                np.log(2/self.delta)
            )
            
            if abs(mean0 - mean1) > epsilon:
                return True
                
        return False
```

### 9.4.2 预测质量监控

#### 置信度分析
```python
class ConfidenceMonitor:
    def __init__(self, calibration_bins=10):
        self.calibration_bins = calibration_bins
        self.confidence_scores = []
        self.accuracies = []
    
    def update_batch(self, probabilities, predictions, labels):
        """更新置信度统计"""
        max_probs = np.max(probabilities, axis=1)
        correct = (predictions == labels)
        
        self.confidence_scores.extend(max_probs)
        self.accuracies.extend(correct)
    
    def compute_ece(self):
        """计算期望校准误差 (ECE)"""
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [
                i for i, conf in enumerate(self.confidence_scores)
                if bin_lower <= conf < bin_upper
            ]
            
            if len(in_bin) > 0:
                bin_acc = np.mean([self.accuracies[i] for i in in_bin])
                bin_conf = np.mean([self.confidence_scores[i] for i in in_bin])
                ece += len(in_bin) * abs(bin_acc - bin_conf)
        
        return ece / len(self.confidence_scores)
```

#### 输出分布监控
```python
# 文本生成质量指标
class TextGenerationMonitor:
    def __init__(self):
        self.metrics_history = defaultdict(list)
    
    def compute_metrics(self, generated_text):
        """计算文本质量指标"""
        metrics = {}
        
        # 多样性指标
        tokens = generated_text.split()
        metrics['unique_tokens'] = len(set(tokens)) / len(tokens)
        
        # 重复度
        bigrams = zip(tokens[:-1], tokens[1:])
        metrics['unique_bigrams'] = len(set(bigrams)) / (len(tokens) - 1)
        
        # 长度分布
        metrics['avg_sentence_length'] = np.mean([
            len(sent.split()) 
            for sent in generated_text.split('.')
        ])
        
        # 困惑度代理指标
        metrics['entropy'] = self._compute_entropy(tokens)
        
        return metrics
    
    def detect_quality_degradation(self, metrics, window=100):
        """检测生成质量下降"""
        alerts = []
        
        for metric_name, value in metrics.items():
            history = self.metrics_history[metric_name]
            history.append(value)
            
            if len(history) > window:
                history.pop(0)
                
                # 计算移动平均和标准差
                mean = np.mean(history[:-10])
                std = np.std(history[:-10])
                recent_mean = np.mean(history[-10:])
                
                # Z-score 检验
                if abs(recent_mean - mean) > 2 * std:
                    alerts.append({
                        'metric': metric_name,
                        'baseline': mean,
                        'current': recent_mean,
                        'severity': 'high' if abs(recent_mean - mean) > 3 * std else 'medium'
                    })
        
        return alerts
```

### 9.4.3 自动化响应策略

```python
class DriftResponseOrchestrator:
    def __init__(self):
        self.drift_detectors = {}
        self.response_strategies = {}
        self.alert_manager = AlertManager()
    
    def register_detector(self, name, detector, strategy):
        """注册漂移检测器和响应策略"""
        self.drift_detectors[name] = detector
        self.response_strategies[name] = strategy
    
    def monitor_and_respond(self, data):
        """监控并自动响应"""
        for name, detector in self.drift_detectors.items():
            if detector.detect(data):
                self._handle_drift(name, data)
    
    def _handle_drift(self, detector_name, data):
        """处理检测到的漂移"""
        strategy = self.response_strategies[detector_name]
        
        if strategy == 'alert':
            self.alert_manager.send_alert(
                f"Drift detected by {detector_name}",
                severity='warning'
            )
        
        elif strategy == 'retrain':
            self._trigger_retraining(data)
        
        elif strategy == 'fallback':
            self._switch_to_fallback_model()
        
        elif strategy == 'recalibrate':
            self._recalibrate_model(data)
    
    def _trigger_retraining(self, data):
        """触发模型重训练"""
        # 收集新数据
        # 启动训练任务
        # 验证新模型
        pass
    
    def _recalibrate_model(self, data):
        """重新校准模型输出"""
        # 使用温度缩放
        # 更新后处理参数
        pass
```

## 9.5 回滚与灰度发布策略

### 9.5.1 蓝绿部署

```yaml
# Kubernetes 蓝绿部署配置
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm
    version: green  # 切换到 blue 或 green
  ports:
  - port: 80
    targetPort: 8080

---
# Blue 部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm
      version: blue
  template:
    metadata:
      labels:
        app: llm
        version: blue
    spec:
      containers:
      - name: model-server
        image: llm:v1.0

---
# Green 部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm
      version: green
  template:
    metadata:
      labels:
        app: llm
        version: green
    spec:
      containers:
      - name: model-server
        image: llm:v2.0
```

### 9.5.2 金丝雀发布

```python
class CanaryDeployment:
    def __init__(self, initial_traffic_percent=5):
        self.canary_percent = initial_traffic_percent
        self.metrics_collector = MetricsCollector()
        self.decision_threshold = 0.95  # 成功率阈值
    
    def route_request(self, request):
        """路由请求到金丝雀或稳定版本"""
        if random.random() < self.canary_percent / 100:
            response = self.canary_model.predict(request)
            self.metrics_collector.record('canary', response)
            return response
        else:
            response = self.stable_model.predict(request)
            self.metrics_collector.record('stable', response)
            return response
    
    def progressive_rollout(self):
        """渐进式发布"""
        stages = [5, 10, 25, 50, 100]  # 流量百分比阶段
        
        for target_percent in stages:
            self.canary_percent = target_percent
            time.sleep(300)  # 每阶段观察 5 分钟
            
            if not self._check_health():
                self._rollback()
                return False
        
        return True
    
    def _check_health(self):
        """检查金丝雀版本健康状态"""
        canary_metrics = self.metrics_collector.get_metrics('canary')
        stable_metrics = self.metrics_collector.get_metrics('stable')
        
        # 比较关键指标
        canary_success_rate = canary_metrics['success_rate']
        stable_success_rate = stable_metrics['success_rate']
        
        if canary_success_rate < self.decision_threshold:
            return False
        
        # 统计显著性检验
        p_value = self._statistical_test(
            canary_metrics['latencies'],
            stable_metrics['latencies']
        )
        
        return p_value > 0.05  # 无显著差异
    
    def _rollback(self):
        """回滚到稳定版本"""
        self.canary_percent = 0
        logger.warning("Canary rollback triggered")
```

### 9.5.3 特征开关（Feature Flags）

```python
class FeatureFlags:
    def __init__(self, config_source='redis'):
        self.flags = {}
        self.config_source = config_source
        self._load_flags()
    
    def _load_flags(self):
        """从配置源加载特征开关"""
        if self.config_source == 'redis':
            self.flags = {
                'new_tokenizer': {'enabled': True, 'rollout': 100},
                'attention_optimization': {'enabled': True, 'rollout': 50},
                'experimental_sampler': {'enabled': False, 'rollout': 0}
            }
    
    def is_enabled(self, flag_name, user_id=None):
        """检查特征是否启用"""
        if flag_name not in self.flags:
            return False
        
        flag = self.flags[flag_name]
        if not flag['enabled']:
            return False
        
        if user_id:
            # 基于用户 ID 的一致性哈希
            hash_value = hash(f"{flag_name}:{user_id}") % 100
            return hash_value < flag['rollout']
        
        return flag['rollout'] == 100
    
    def with_feature(self, flag_name):
        """装饰器模式"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if self.is_enabled(flag_name):
                    return func(*args, **kwargs)
                else:
                    # 返回默认行为
                    return self._default_behavior(flag_name)
            return wrapper
        return decorator
```

### 9.5.4 回滚策略实现

```python
class RollbackManager:
    def __init__(self):
        self.deployment_history = []
        self.health_checker = HealthChecker()
        self.max_rollback_window = 3600  # 1 小时
    
    def deploy(self, version, config):
        """部署新版本"""
        deployment = {
            'version': version,
            'config': config,
            'timestamp': time.time(),
            'status': 'deploying'
        }
        
        self.deployment_history.append(deployment)
        
        try:
            # 执行部署
            self._execute_deployment(version, config)
            
            # 健康检查
            if self._post_deployment_check(version):
                deployment['status'] = 'healthy'
                return True
            else:
                deployment['status'] = 'unhealthy'
                self.automatic_rollback()
                return False
                
        except Exception as e:
            deployment['status'] = 'failed'
            deployment['error'] = str(e)
            self.automatic_rollback()
            raise
    
    def automatic_rollback(self):
        """自动回滚到上一个健康版本"""
        for deployment in reversed(self.deployment_history[:-1]):
            if deployment['status'] == 'healthy':
                logger.info(f"Rolling back to version {deployment['version']}")
                self._execute_deployment(
                    deployment['version'],
                    deployment['config']
                )
                return True
        
        logger.error("No healthy version found for rollback")
        return False
    
    def _post_deployment_check(self, version):
        """部署后健康检查"""
        checks = [
            self.health_checker.check_endpoint_health(),
            self.health_checker.check_model_loading(),
            self.health_checker.check_inference_latency(),
            self.health_checker.check_error_rate()
        ]
        
        return all(checks)
    
    def _execute_deployment(self, version, config):
        """执行实际部署操作"""
        # 更新模型文件
        # 重启服务
        # 更新路由配置
        pass
```

## 本章小结

本章系统介绍了 LLM 生产部署与监控的核心技术：

### 📌 核心概念

1. **模型优化三大技术**：
   - 量化：PTQ vs QAT，权衡精度与效率
   - 蒸馏：知识传递，大模型能力迁移到小模型
   - 剪枝：结构化 vs 非结构化，移除冗余参数

2. **服务架构设计**：
   - 微服务架构：模块化、可扩展、高可用
   - 流式生成：SSE/WebSocket 实现逐 token 输出
   - 批处理优化：动态批处理提高吞吐量

3. **监控体系**：
   - 系统指标：延迟、吞吐、资源利用率
   - 业务指标：质量分数、成本统计
   - 分布式追踪：全链路性能分析

4. **漂移检测**：
   - 数据漂移：输入分布变化
   - 概念漂移：输入-输出关系变化
   - 自动响应：告警、重训练、回滚

5. **发布策略**：
   - 蓝绿部署：零停机切换
   - 金丝雀发布：渐进式流量迁移
   - 特征开关：细粒度功能控制

### 💡 关键公式

1. **量化误差**：
   $$\epsilon = \frac{\alpha}{2^{b-1}-1}$$
   其中 $\alpha$ 为数值范围，$b$ 为量化位数

2. **蒸馏损失**：
   $$L = \alpha L_{CE} + \beta \cdot T^2 \cdot L_{KL}$$

3. **期望校准误差（ECE）**：
   $$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$

4. **KL 散度漂移检测**：
   $$D_{KL}(P||Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

### 🔬 实用技巧

1. **压缩策略选择**：
   - 延迟敏感：优先量化（INT8）
   - 内存受限：结构化剪枝 + 量化
   - 精度优先：知识蒸馏

2. **监控告警设置**：
   - P50/P95/P99 分位数监控
   - 滑动窗口异常检测
   - 分级告警避免告警疲劳

3. **发布风险控制**：
   - 金丝雀起始流量 < 5%
   - 每阶段观察时间 ≥ 5 分钟
   - 自动回滚机制必须配置

## 常见陷阱与错误 (Gotchas)

### ⚠️ 模型压缩陷阱

1. **过度量化导致精度崩溃**
   - 错误：直接应用 INT4 量化到所有层
   - 正确：关键层保持高精度，使用混合精度策略

2. **忽视硬件加速支持**
   - 错误：使用非结构化剪枝期望加速
   - 正确：确认目标硬件支持的优化模式

3. **蒸馏温度设置不当**
   - 错误：固定温度 T=1
   - 正确：根据任务调整，通常 T∈[3,10]

### ⚠️ 部署架构陷阱

1. **批处理配置错误**
   - 错误：batch_size 越大越好
   - 正确：平衡延迟和吞吐，测试最优值

2. **忽视冷启动问题**
   - 错误：不预热模型直接服务
   - 正确：部署后进行预热请求

3. **KV Cache 内存溢出**
   - 错误：无限制缓存所有序列
   - 正确：设置最大序列长度和缓存淘汰策略

### ⚠️ 监控盲区

1. **只监控平均值**
   - 错误：只看平均延迟
   - 正确：监控 P95/P99 长尾延迟

2. **忽视业务指标**
   - 错误：只关注系统指标
   - 正确：结合业务质量指标综合评估

3. **告警风暴**
   - 错误：所有异常都触发告警
   - 正确：告警聚合、降噪、分级

### ⚠️ 发布风险

1. **跳过金丝雀阶段**
   - 错误：直接 100% 切换流量
   - 正确：渐进式增加流量比例

2. **回滚策略缺失**
   - 错误：手动回滚，耗时且易错
   - 正确：自动化回滚机制

3. **配置漂移**
   - 错误：手动修改生产配置
   - 正确：版本化配置，通过 CI/CD 部署

## 练习题

### 基础题

**练习 9.1：量化精度分析**
给定一个权重矩阵 W ∈ [-2.5, 3.7]，计算使用 INT8 对称量化后的最大量化误差。

<details>
<summary>Hint</summary>
考虑对称量化的范围确定方式和量化步长计算。
</details>

<details>
<summary>答案</summary>

对称量化需要范围对称于 0：
- 范围：[-3.7, 3.7]（取绝对值最大值）
- 量化步长：s = 3.7 / 127 ≈ 0.0291
- 最大量化误差：s/2 ≈ 0.0146

实际量化过程：
1. 缩放：W_scaled = W × 127/3.7
2. 取整：W_q = round(W_scaled)
3. 反量化：W' = W_q × 3.7/127

最大误差出现在量化边界，约为 0.0146。
</details>

**练习 9.2：KV Cache 内存估算**
计算以下配置的 KV Cache 内存需求：
- batch_size = 8
- max_seq_length = 2048
- num_layers = 24
- num_kv_heads = 8
- head_dim = 128
- dtype = float16

<details>
<summary>Hint</summary>
KV Cache 需要存储每层的 K 和 V 矩阵，注意 float16 占 2 字节。
</details>

<details>
<summary>答案</summary>

内存计算：
```
Cache_size = batch × seq_len × n_layers × n_kv_heads × d_head × 2 × dtype_size
         = 8 × 2048 × 24 × 8 × 128 × 2 × 2 bytes
         = 8 × 2048 × 24 × 8 × 128 × 2 × 2
         = 1,610,612,736 bytes
         ≈ 1.5 GB
```

解释：
- 2 表示 K 和 V 两个矩阵
- float16 占 2 字节
- 总计约 1.5 GB 显存
</details>

**练习 9.3：金丝雀发布流量计算**
金丝雀发布采用指数增长策略，初始流量 2%，每阶段翻倍。需要多少阶段才能达到 100% 流量？每阶段的具体流量比例是多少？

<details>
<summary>Hint</summary>
使用 2^n 计算阶段数，注意最后一个阶段直接到 100%。
</details>

<details>
<summary>答案</summary>

流量增长序列：
- 阶段 1：2%
- 阶段 2：4%
- 阶段 3：8%
- 阶段 4：16%
- 阶段 5：32%
- 阶段 6：64%
- 阶段 7：100%

共需要 7 个阶段。实际部署中，通常会在 50% 后直接跳到 100%，即：
2% → 5% → 10% → 25% → 50% → 100%（6 个阶段）
</details>

### 挑战题

**练习 9.4：漂移检测算法设计**
设计一个自适应的漂移检测算法，要求：
1. 能够区分渐进漂移和突发漂移
2. 自动调整检测灵敏度
3. 提供漂移严重程度评分

<details>
<summary>Hint</summary>
考虑结合多个时间窗口、使用不同的统计检验方法，以及如何量化漂移程度。
</details>

<details>
<summary>答案</summary>

多尺度自适应漂移检测器设计：

```python
class AdaptiveDriftDetector:
    def __init__(self):
        self.short_window = deque(maxlen=100)
        self.medium_window = deque(maxlen=500)
        self.long_window = deque(maxlen=2000)
        self.baseline_stats = None
        
    def detect(self, value):
        # 更新窗口
        self.short_window.append(value)
        self.medium_window.append(value)
        self.long_window.append(value)
        
        # 突发漂移检测（短期 vs 长期）
        sudden_drift = self._detect_sudden(
            self.short_window, 
            self.long_window
        )
        
        # 渐进漂移检测（中期趋势）
        gradual_drift = self._detect_gradual(
            self.medium_window
        )
        
        # 计算严重程度
        severity = self._compute_severity(
            sudden_drift, 
            gradual_drift
        )
        
        return {
            'sudden': sudden_drift,
            'gradual': gradual_drift,
            'severity': severity
        }
    
    def _detect_sudden(self, short, long):
        if len(short) < 50 or len(long) < 500:
            return False
        # KS 检验
        _, p_value = ks_2samp(list(short), list(long))
        return p_value < 0.01
    
    def _detect_gradual(self, window):
        if len(window) < 100:
            return False
        # Mann-Kendall 趋势检验
        return self._mann_kendall_test(window)
    
    def _compute_severity(self, sudden, gradual):
        if sudden and gradual:
            return 'critical'
        elif sudden:
            return 'high'
        elif gradual:
            return 'medium'
        return 'low'
```

关键设计点：
1. 多时间尺度窗口捕获不同类型漂移
2. KS 检验检测分布突变
3. Mann-Kendall 检验检测趋势
4. 组合多个信号评估严重程度
</details>

**练习 9.5：自动化容量规划**
设计一个系统，根据历史负载模式和业务增长预测，自动进行容量规划和资源调度。考虑：
1. 周期性模式（日/周/月）
2. 趋势增长
3. 突发流量
4. 成本优化

<details>
<summary>Hint</summary>
使用时间序列分解、预测模型、资源调度算法的组合。
</details>

<details>
<summary>答案</summary>

自动化容量规划系统设计：

```python
class CapacityPlanner:
    def __init__(self):
        self.load_history = []
        self.cost_model = CostModel()
        
    def plan(self, forecast_days=30):
        # 1. 时间序列分解
        trend, seasonal, residual = self._decompose_timeseries()
        
        # 2. 预测未来负载
        future_load = self._forecast_load(
            trend, seasonal, forecast_days
        )
        
        # 3. 计算所需资源
        required_resources = self._compute_resources(
            future_load,
            include_buffer=1.3  # 30% 缓冲
        )
        
        # 4. 优化资源配置
        optimal_config = self._optimize_allocation(
            required_resources,
            self.cost_model
        )
        
        # 5. 生成扩缩容计划
        scaling_plan = self._generate_scaling_plan(
            optimal_config,
            current_resources=self._get_current_resources()
        )
        
        return scaling_plan
    
    def _decompose_timeseries(self):
        # STL 分解
        from statsmodels.tsa.seasonal import STL
        stl = STL(self.load_history, seasonal=169)  # 周周期
        result = stl.fit()
        return result.trend, result.seasonal, result.resid
    
    def _forecast_load(self, trend, seasonal, days):
        # SARIMA 模型预测
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(
            self.load_history,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 24)
        )
        fitted = model.fit()
        forecast = fitted.forecast(steps=days * 24)
        
        # 添加季节性
        forecast_with_seasonal = forecast + seasonal[-days*24:]
        
        # 添加安全边际（P95）
        safety_margin = np.percentile(self.load_history, 95) / np.mean(self.load_history)
        return forecast_with_seasonal * safety_margin
    
    def _optimize_allocation(self, resources, cost_model):
        # 混合整数规划
        from scipy.optimize import milp
        
        # 定义决策变量：不同实例类型的数量
        # 目标：最小化成本
        # 约束：满足资源需求
        
        c = [cost_model.get_cost(t) for t in instance_types]
        A_ub = -np.array([t.capacity for t in instance_types])
        b_ub = -resources['compute']
        
        result = milp(c, integrality=1, bounds=(0, 100), 
                     constraints=[A_ub, b_ub])
        
        return result.x
```

核心组件：
1. **时间序列分解**：识别趋势、季节性、随机成分
2. **负载预测**：SARIMA 模型 + 安全边际
3. **资源映射**：负载 → GPU/CPU/内存需求
4. **成本优化**：考虑 Spot/Reserved/On-Demand 实例组合
5. **渐进式扩缩容**：避免资源抖动

实施要点：
- 预留 20-30% 缓冲应对突发
- 使用预留实例降低基线成本
- Spot 实例处理弹性负载
- 自动告警阈值：实际 > 预测 × 1.5
</details>

**练习 9.6：零停机迁移方案**
设计一个将 LLM 服务从数据中心 A 迁移到数据中心 B 的零停机方案，考虑：
1. 模型文件同步（数 GB）
2. 有状态的会话保持
3. 逐步流量迁移
4. 失败回滚

<details>
<summary>Hint</summary>
考虑双写、会话迁移、DNS 切换、数据一致性等问题。
</details>

<details>
<summary>答案</summary>

零停机跨数据中心迁移方案：

**阶段 1：准备（T-7 天）**
```yaml
tasks:
  - name: 部署 B 数据中心基础设施
    steps:
      - 配置网络和负载均衡器
      - 部署 Kubernetes 集群
      - 设置监控和日志系统
  
  - name: 模型文件同步
    method: 增量同步
    tools: rsync --daemon
    bandwidth_limit: 50%  # 避免影响 A 数据中心
```

**阶段 2：双写模式（T-3 天）**
```python
class DualWriteProxy:
    def __init__(self):
        self.primary = DataCenterA()
        self.secondary = DataCenterB()
        
    async def handle_request(self, request):
        # 主数据中心处理
        response = await self.primary.process(request)
        
        # 异步复制到副数据中心
        asyncio.create_task(
            self._replicate_to_secondary(request)
        )
        
        return response
    
    async def _replicate_to_secondary(self, request):
        try:
            await self.secondary.process(request)
        except Exception as e:
            logger.warning(f"Secondary replication failed: {e}")
```

**阶段 3：流量迁移（T-0）**
```python
class TrafficMigrator:
    def __init__(self):
        self.migration_percent = 0
        self.session_affinity = {}
        
    def route(self, request):
        # 会话保持
        session_id = request.session_id
        if session_id in self.session_affinity:
            return self.session_affinity[session_id]
        
        # 新会话按比例分配
        if random.random() < self.migration_percent / 100:
            destination = 'datacenter_b'
        else:
            destination = 'datacenter_a'
            
        self.session_affinity[session_id] = destination
        return destination
    
    def progressive_migration(self):
        stages = [1, 5, 10, 25, 50, 75, 90, 100]
        
        for percent in stages:
            self.migration_percent = percent
            
            # 健康检查
            if not self._health_check():
                self._rollback()
                return False
                
            # 会话迁移（长连接）
            self._migrate_sticky_sessions(percent)
            
            time.sleep(300)  # 5 分钟观察期
        
        return True
```

**阶段 4：会话迁移**
```python
class SessionMigrator:
    def migrate_session(self, session_id):
        # 1. 获取会话状态
        state = datacenter_a.get_session_state(session_id)
        
        # 2. 序列化 KV Cache
        kv_cache = self._serialize_kv_cache(state.kv_cache)
        
        # 3. 传输到 B 数据中心
        datacenter_b.restore_session(session_id, {
            'kv_cache': kv_cache,
            'context': state.context,
            'timestamp': state.timestamp
        })
        
        # 4. 验证一致性
        checksum_a = datacenter_a.compute_checksum(session_id)
        checksum_b = datacenter_b.compute_checksum(session_id)
        
        if checksum_a != checksum_b:
            raise ConsistencyError()
        
        # 5. 更新路由
        router.update_affinity(session_id, 'datacenter_b')
```

**阶段 5：验证和清理**
```bash
# DNS 切换
dig @8.8.8.8 api.example.com  # 验证 TTL
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123 \
  --change-batch file://dns-cutover.json

# 监控验证
- 错误率 < 0.01%
- P99 延迟 < 110% 基线
- 会话连续性 100%

# 清理旧资源（T+7 天）
kubectl --context=datacenter-a delete deployment llm-service
```

关键技术点：
1. **增量同步**：避免网络拥塞
2. **会话亲和性**：保持用户体验
3. **双写验证**：确保 B 数据中心就绪
4. **渐进式切换**：快速回滚能力
5. **状态迁移**：KV Cache 序列化传输

失败回滚机制：
- DNS 快速切回（TTL=60s）
- 会话路由表回滚
- 保留 A 数据中心 7 天
</details>

**练习 9.7：成本优化策略**
某 LLM 服务月度 GPU 成本 10 万美元，设计一个综合成本优化方案，目标降低 30% 成本而不影响 SLA。

<details>
<summary>Hint</summary>
从多个维度思考：实例类型、调度策略、缓存、模型优化等。
</details>

<details>
<summary>答案</summary>

综合成本优化方案：

**1. 实例类型优化（预期节省 15%）**
```python
# 当前：100% On-Demand
# 优化后：混合实例策略
instance_mix = {
    'reserved': 0.5,    # 50% 预留实例（3年期，节省 60%）
    'savings_plan': 0.2, # 20% Savings Plan（节省 40%）  
    'spot': 0.2,        # 20% Spot 实例（节省 70%）
    'on_demand': 0.1    # 10% 按需（保持弹性）
}

# 成本计算
original_cost = 100000  # 美元/月
optimized_cost = (
    100000 * 0.5 * 0.4 +   # Reserved
    100000 * 0.2 * 0.6 +   # Savings Plan
    100000 * 0.2 * 0.3 +   # Spot
    100000 * 0.1 * 1.0     # On-Demand
) = 48000  # 节省 52%，但考虑 Spot 中断，实际约 15%
```

**2. 智能调度和自动伸缩（预期节省 8%）**
```python
class CostAwareScheduler:
    def schedule(self, request):
        priority = self._get_priority(request)
        
        if priority == 'low':
            # 低优先级任务调度到 Spot 实例
            return self.spot_instances.schedule(request)
        elif priority == 'medium':
            # 中优先级使用预留实例
            return self.reserved_instances.schedule(request)
        else:
            # 高优先级保证 SLA
            return self.on_demand_instances.schedule(request)
    
    def auto_scale(self):
        # 基于负载预测的提前扩缩容
        predicted_load = self.predictor.forecast(horizon='1h')
        
        if predicted_load < 0.3:
            # 深夜低谷，关闭部分实例
            self.scale_down(target=0.5)
        elif predicted_load > 0.8:
            # 高峰期，提前扩容
            self.scale_up(target=1.2)
```

**3. 缓存优化（预期节省 5%）**
```python
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # 内存缓存（热点数据）
        self.l2_cache = Redis()  # 分布式缓存
        self.l3_cache = S3()  # 冷数据
        
    def get(self, key):
        # L1: 命中率 30%，节省 100% GPU 时间
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2: 命中率 20%，节省 95% GPU 时间
        if self.l2_cache.exists(key):
            value = self.l2_cache.get(key)
            self.l1_cache[key] = value
            return value
        
        # L3: 嵌入向量缓存
        if self._is_embedding_request(key):
            embedding = self.l3_cache.get(key)
            if embedding:
                return embedding
        
        # Cache miss，计算并存储
        result = self.compute(key)
        self._update_caches(key, result)
        return result
```

**4. 模型优化（预期节省 5%）**
```python
# 动态模型选择
class ModelSelector:
    def select(self, request):
        complexity = self._estimate_complexity(request)
        
        if complexity == 'simple':
            # 简单任务用小模型（7B）
            return self.small_model  # 成本 1x
        elif complexity == 'medium':
            # 中等任务用中模型（13B）
            return self.medium_model  # 成本 2x
        else:
            # 复杂任务用大模型（70B）
            return self.large_model  # 成本 10x

# 请求级量化
class DynamicQuantization:
    def process(self, request):
        if request.latency_requirement > 1000:  # ms
            # 宽松延迟要求，使用 INT4 量化
            return self.int4_model.generate(request)
        else:
            # 严格延迟要求，使用 FP16
            return self.fp16_model.generate(request)
```

**5. 批处理优化（预期节省 2%）**
```python
class BatchOptimizer:
    def optimize_batch_size(self):
        # 动态调整批大小
        current_latency = self.monitor.get_p95_latency()
        current_batch = self.config.batch_size
        
        if current_latency < SLA_LATENCY * 0.8:
            # 有余量，增加批大小提高吞吐
            self.config.batch_size = min(current_batch * 1.2, 64)
        elif current_latency > SLA_LATENCY * 0.95:
            # 接近 SLA，减小批大小
            self.config.batch_size = max(current_batch * 0.8, 1)
```

**实施计划**：
1. 第 1 个月：采购预留实例，部署缓存系统
2. 第 2 个月：实施智能调度，A/B 测试
3. 第 3 个月：模型优化，全面推广

**预期效果**：
- 总成本降低：32%
- SLA 保持：99.9%
- 用户体验：无感知

**风险缓解**：
- Spot 中断：自动故障转移到 On-Demand
- 缓存失效：降级到直接计算
- 模型切换：平滑过渡，监控质量
</details>

**练习 9.8：端到端延迟优化**
一个 LLM 服务的 P99 延迟为 5 秒，分析并优化到 2 秒以内。给出详细的分析方法和优化方案。

<details>
<summary>Hint</summary>
使用火焰图分析、分阶段优化、关注长尾延迟的特殊原因。
</details>

<details>
<summary>答案</summary>

端到端延迟优化方案：

**第一步：延迟分解分析**
```python
class LatencyProfiler:
    def profile_request(self, request):
        timeline = {}
        
        # 1. 网络接收
        t0 = time.time()
        data = receive_request(request)
        timeline['network_in'] = time.time() - t0
        
        # 2. 认证授权
        t1 = time.time()
        auth = authenticate(data)
        timeline['auth'] = time.time() - t1
        
        # 3. 预处理
        t2 = time.time()
        tokens = tokenize(data)
        timeline['tokenization'] = time.time() - t2
        
        # 4. 队列等待
        t3 = time.time()
        batch = queue.wait_for_batch(tokens)
        timeline['queue_wait'] = time.time() - t3
        
        # 5. 模型推理
        t4 = time.time()
        output = model.generate(batch)
        timeline['inference'] = time.time() - t4
        
        # 6. 后处理
        t5 = time.time()
        result = postprocess(output)
        timeline['postprocess'] = time.time() - t5
        
        # 7. 网络发送
        t6 = time.time()
        send_response(result)
        timeline['network_out'] = time.time() - t6
        
        return timeline

# 分析 P99 延迟组成
p99_breakdown = {
    'network_in': 50,      # ms
    'auth': 20,           # ms
    'tokenization': 100,  # ms
    'queue_wait': 500,    # ms (!)
    'inference': 4000,    # ms (!)
    'postprocess': 80,    # ms
    'network_out': 250    # ms (!)
}
```

**第二步：针对性优化**

**优化 1：推理加速（4000ms → 1500ms）**
```python
# Flash Attention 实现
class FlashAttentionOptimized:
    def forward(self, q, k, v):
        # 分块计算，减少内存访问
        BLOCK_SIZE = 64
        
        # 使用 Triton 核函数
        output = triton_flash_attn(
            q, k, v,
            causal=True,
            block_size=BLOCK_SIZE
        )
        return output

# KV Cache 优化
class OptimizedKVCache:
    def __init__(self):
        self.cache = {}
        self.gpu_cache = {}  # GPU 常驻
        
    def get(self, key):
        if key in self.gpu_cache:
            return self.gpu_cache[key]  # 0 拷贝
        elif key in self.cache:
            # 异步传输到 GPU
            self.gpu_cache[key] = self.cache[key].cuda(non_blocking=True)
            return self.gpu_cache[key]
        return None

# 算子融合
@torch.jit.script
def fused_gelu_linear(x, weight, bias):
    # 融合 GELU 激活和线性层
    return F.linear(F.gelu(x), weight, bias)
```

**优化 2：队列优化（500ms → 100ms）**
```python
class PriorityBatchQueue:
    def __init__(self):
        self.queues = {
            'high': deque(),     # SLA 严格
            'medium': deque(),   # 普通请求
            'low': deque()       # 批量任务
        }
        self.continuous_batching = True
        
    def add_request(self, request):
        priority = self._compute_priority(request)
        self.queues[priority].append(request)
        
        # 高优先级立即处理
        if priority == 'high':
            return self._immediate_batch(request)
        
        # 连续批处理
        if self.continuous_batching:
            return self._try_merge_batch(request)
    
    def _immediate_batch(self, request):
        # 高优先级请求不等待
        return [request]
    
    def _try_merge_batch(self, request):
        # 动态批处理，最大等待 100ms
        deadline = time.time() + 0.1
        batch = [request]
        
        while time.time() < deadline and len(batch) < 32:
            if self.queues['medium']:
                batch.append(self.queues['medium'].popleft())
            elif self.queues['low']:
                batch.append(self.queues['low'].popleft())
            else:
                break
                
        return batch if len(batch) > 1 else None
```

**优化 3：网络优化（250ms → 50ms）**
```python
# HTTP/2 服务器推送
class HTTP2Streaming:
    def stream_response(self, tokens):
        # 服务器推送，逐 token 发送
        for token in tokens:
            self.push_frame(token)
            
# 零拷贝发送
class ZeroCopySender:
    def send(self, data):
        # 使用 sendfile 系统调用
        os.sendfile(
            self.socket.fileno(),
            data.fileno(),
            offset=0,
            count=len(data)
        )

# 压缩传输
class CompressionMiddleware:
    def compress_response(self, response):
        if len(response) > 1024:  # 1KB 以上才压缩
            return gzip.compress(response, compresslevel=1)  # 快速压缩
        return response
```

**优化 4：长尾延迟专项优化**
```python
class TailLatencyOptimizer:
    def __init__(self):
        self.gc_controller = GCController()
        self.memory_pool = MemoryPool()
        
    def optimize(self):
        # 1. GC 调优
        self.gc_controller.set_gc_threshold(
            threshold0=10000,  # 延迟 GC
            threshold1=20,
            threshold2=20
        )
        
        # 2. 内存池化
        self.memory_pool.preallocate(
            tensor_sizes=[1024, 2048, 4096],
            count=100
        )
        
        # 3. CPU 亲和性
        os.sched_setaffinity(0, {0, 1, 2, 3})  # 绑定 CPU 核心
        
        # 4. 预热
        self._warmup_model()
        
    def _warmup_model(self):
        # JIT 编译预热
        dummy_input = torch.randn(1, 512)
        for _ in range(10):
            self.model(dummy_input)
```

**优化 5：自适应降级**
```python
class AdaptiveDegradation:
    def process(self, request):
        current_latency = self.monitor.get_current_p99()
        
        if current_latency > 3000:  # 严重延迟
            # 降级到小模型
            return self.small_model.generate(
                request,
                max_length=min(request.max_length, 256)
            )
        elif current_latency > 2000:  # 中度延迟
            # 减少生成长度
            return self.model.generate(
                request,
                max_length=min(request.max_length, 512)
            )
        else:
            # 正常处理
            return self.model.generate(request)
```

**最终效果**：
```
原始 P99: 5000ms
优化后 P99: 1850ms

分解：
- 网络接收: 50ms
- 认证: 20ms
- Tokenization: 100ms
- 队列: 100ms (优化 400ms)
- 推理: 1500ms (优化 2500ms)
- 后处理: 80ms
- 网络发送: 50ms (优化 200ms)

总计: 1900ms（达标）
```

**监控验证**：
```python
# A/B 测试验证
ab_test = ABTest(
    control='original',
    treatment='optimized',
    metrics=['p50', 'p95', 'p99', 'error_rate'],
    duration_hours=24
)

results = ab_test.run()
assert results['treatment']['p99'] < 2000
assert results['treatment']['error_rate'] < 0.001
```
</details>

---

**下一章**：[第十章：案例研究与最佳实践 →](chapter10.md)