# 第二章：实验代码基础设施

构建可维护、可扩展的实验代码架构是成功进行 LLM 后训练的基石。本章将深入探讨如何设计和实现一个健壮的实验基础设施，涵盖配置管理、版本控制、实验追踪等关键组件。我们将重点解决实际工程中的挑战：如何在快速迭代的同时保持代码质量，如何管理数百个实验的配置和结果，以及如何防止技术债务的累积。

## 2.1 实验配置管理

### 配置文件格式选择

在 LLM 后训练项目中，配置管理的复杂度远超传统深度学习项目。一个典型的实验可能包含上百个超参数，涉及模型架构、训练策略、数据处理、评估指标等多个维度。选择合适的配置格式至关重要。

**YAML 配置的优势与劣势**

YAML 因其可读性强而广受欢迎，特别适合嵌套结构的表达：

```yaml
model:
  architecture: llama2
  hidden_size: 4096
  num_layers: 32
  attention:
    num_heads: 32
    head_dim: 128
    rotary_embedding:
      base: 10000
      scaling_factor: 1.0
```

优势：
- 人类可读性最佳，适合配置审查
- 支持注释，便于文档化
- 层次结构清晰，适合复杂配置
- 生态系统成熟，工具链完善

劣势：
- 缩进敏感，容易出错
- 类型推断可能产生意外（如 "no" 被解析为布尔值）
- 不支持变量引用和计算表达式
- 大文件解析速度较慢

**TOML 配置的权衡**

TOML 在 Rust 和 Python 社区逐渐流行，提供了更严格的语法：

```toml
[model]
architecture = "llama2"
hidden_size = 4096
num_layers = 32

[model.attention]
num_heads = 32
head_dim = 128

[model.attention.rotary_embedding]
base = 10000
scaling_factor = 1.0
```

优势：
- 语法明确，歧义少
- 原生支持日期时间类型
- 表格数组语法适合批量实验配置

劣势：
- 深层嵌套可读性下降
- 数组和内联表的语法较复杂
- 生态系统相对较新

**Python 配置的灵活性**

直接使用 Python 文件作为配置提供了最大的灵活性：

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    architecture: str = "llama2"
    hidden_size: int = 4096
    num_layers: int = 32
    
    @property
    def total_params(self) -> int:
        # 动态计算参数量
        return self.calculate_params()
    
    def scale_model(self, factor: float):
        """动态调整模型规模"""
        self.hidden_size = int(self.hidden_size * factor)
        self.num_layers = int(self.num_layers * factor)
```

优势：
- 支持动态计算和条件逻辑
- 类型检查和 IDE 支持完善
- 可以复用代码和导入模块
- 支持配置验证和默认值

劣势：
- 安全性风险（执行任意代码）
- 非技术人员难以修改
- 版本控制中 diff 可读性较差

### 配置继承与覆盖机制

实践中，我们通常需要一个基础配置和多个实验变体。设计良好的继承机制可以大幅减少配置冗余：

```python
class ConfigManager:
    def __init__(self, base_config_path: str):
        self.base_config = self.load_config(base_config_path)
        self.inheritance_chain = [base_config_path]
    
    def inherit_from(self, parent_config_path: str):
        """支持多级继承"""
        parent_config = self.load_config(parent_config_path)
        self.base_config = self.deep_merge(parent_config, self.base_config)
        self.inheritance_chain.append(parent_config_path)
    
    def override(self, overrides: Dict[str, Any]):
        """支持命令行覆盖"""
        for key_path, value in overrides.items():
            self.set_nested_value(key_path, value)
    
    def deep_merge(self, base: Dict, override: Dict) -> Dict:
        """递归合并配置字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        return result
```

**配置覆盖的优先级设计**

一个清晰的优先级体系避免了配置冲突：

1. 命令行参数（最高优先级）
2. 环境变量
3. 实验特定配置文件
4. 用户配置文件
5. 项目默认配置（最低优先级）

```
优先级链：
CLI Args > ENV > experiment.yaml > user.yaml > default.yaml
```

### 配置验证与类型检查

使用 Pydantic 或 attrs 进行运行时验证可以及早发现配置错误：

```python
from pydantic import BaseModel, validator, Field
from typing import Literal

class TrainingConfig(BaseModel):
    learning_rate: float = Field(gt=0, le=1.0)
    batch_size: int = Field(gt=0, multiple_of=8)
    optimizer: Literal["adam", "sgd", "adamw"]
    gradient_accumulation_steps: int = Field(gt=0)
    
    @validator("batch_size")
    def validate_batch_size(cls, v, values):
        if "gradient_accumulation_steps" in values:
            effective_batch = v * values["gradient_accumulation_steps"]
            if effective_batch > 65536:
                raise ValueError(f"Effective batch size {effective_batch} too large")
        return v
    
    @validator("learning_rate")
    def validate_lr_schedule(cls, v, values):
        if values.get("optimizer") == "sgd" and v > 0.1:
            raise ValueError("SGD learning rate typically should be < 0.1")
        return v
```

## 2.2 Flag、环境变量与 Git 分支策略

### Command-line Flags 的设计原则

命令行参数是实验配置的第一接触点，良好的设计能显著提升实验效率。以下是经过大规模实验验证的设计原则：

**层次化的参数组织**

避免平铺所有参数，而是按功能域组织：

```python
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    
    # 使用参数组提高可读性
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model.name", default="llama2-7b")
    model_group.add_argument("--model.checkpoint", type=str)
    model_group.add_argument("--model.dtype", choices=["fp32", "fp16", "bf16"])
    
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--training.batch_size", type=int, default=32)
    training_group.add_argument("--training.learning_rate", type=float, default=1e-4)
    training_group.add_argument("--training.warmup_steps", type=int, default=1000)
    
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--data.train_path", required=True)
    data_group.add_argument("--data.val_path", required=True)
    data_group.add_argument("--data.num_workers", type=int, default=4)
    
    return parser
```

**智能默认值与必需参数**

区分必需参数和可选参数，为常见场景提供合理默认值：

```python
class FlagValidator:
    @staticmethod
    def validate_flags(args):
        # 自动推断相关参数
        if args.distributed and args.local_rank is None:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # 根据硬件自动设置
        if args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 批量大小自动调整
        if args.gradient_checkpointing and args.batch_size > 16:
            logger.warning(f"Reducing batch size from {args.batch_size} to 16 due to gradient checkpointing")
            args.batch_size = 16
        
        return args
```

**参数别名与简写**

为常用参数提供简写，提高命令行效率：

```python
parser.add_argument("-b", "--batch-size", "--training.batch_size", 
                   dest="batch_size", type=int, default=32,
                   help="Training batch size per device")
                   
parser.add_argument("-lr", "--learning-rate", "--training.lr",
                   dest="learning_rate", type=float, default=1e-4,
                   help="Peak learning rate")
                   
parser.add_argument("-e", "--epochs", "--num-epochs",
                   dest="num_epochs", type=int, default=3,
                   help="Number of training epochs")
```

### 环境变量的使用场景

环境变量适合管理跨实验的全局设置和敏感信息：

**分层的环境变量体系**

```bash
# 系统级别（集群配置）
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

# 项目级别（路径和凭证）
export LLM_DATA_ROOT=/mnt/data/llm_datasets
export LLM_CHECKPOINT_DIR=/mnt/checkpoints
export WANDB_API_KEY=your_api_key_here
export HF_TOKEN=your_huggingface_token

# 实验级别（运行时配置）
export LLM_EXPERIMENT_NAME=dpo_ablation_v3
export LLM_RUN_ID=$(date +%Y%m%d_%H%M%S)
export LLM_DEBUG_MODE=1
```

**环境变量的最佳实践**

```python
import os
from pathlib import Path
from typing import Optional

class EnvConfig:
    """统一管理环境变量"""
    
    @staticmethod
    def get_data_root() -> Path:
        """获取数据根目录，支持多级fallback"""
        candidates = [
            os.environ.get("LLM_DATA_ROOT"),
            os.environ.get("DATA_ROOT"),
            "/data/llm",
            "./data"
        ]
        for path in candidates:
            if path and Path(path).exists():
                return Path(path)
        raise ValueError("No valid data root found")
    
    @staticmethod
    def get_wandb_config() -> dict:
        """安全地获取 W&B 配置"""
        config = {}
        if api_key := os.environ.get("WANDB_API_KEY"):
            config["api_key"] = api_key
        if project := os.environ.get("WANDB_PROJECT"):
            config["project"] = project
        if entity := os.environ.get("WANDB_ENTITY"):
            config["entity"] = entity
        return config
    
    @staticmethod
    def is_debug_mode() -> bool:
        """检查调试模式"""
        return os.environ.get("LLM_DEBUG_MODE", "0").lower() in ("1", "true", "yes")
```

### Git 分支管理实践

在快速迭代的实验环境中，Git 分支策略需要平衡实验自由度和代码质量：

**实验分支命名规范**

```bash
# 功能开发分支
feature/distributed-dpo
feature/multimodal-alignment

# 实验分支（短期）
exp/20250105-lr-sweep
exp/20250106-batch-size-ablation

# 个人实验分支（更自由）
dev/alice/rope-scaling
dev/bob/attention-variants

# 长期研究分支
research/constitutional-ai
research/online-rlhf
```

**分支保护与合并策略**

```python
# .github/branch_protection.yml
protection_rules:
  main:
    required_reviews: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
    required_status_checks:
      - lint
      - type-check
      - unit-tests
    enforce_admins: false
    
  release/*:
    required_reviews: 3
    require_code_owner_reviews: true
    required_status_checks:
      - all-tests
      - integration-tests
      - benchmark-regression
```

**实验分支的生命周期管理**

```bash
#!/bin/bash
# scripts/manage_exp_branches.sh

# 自动清理过期的实验分支
cleanup_old_exp_branches() {
    local days_old=${1:-30}
    
    git for-each-ref --format='%(refname:short) %(committerdate:unix)' refs/heads/exp/ | \
    while read branch timestamp; do
        age_days=$(( ($(date +%s) - timestamp) / 86400 ))
        if [ $age_days -gt $days_old ]; then
            echo "Deleting old experimental branch: $branch (${age_days} days old)"
            git branch -D "$branch"
        fi
    done
}

# 归档重要实验分支
archive_exp_branch() {
    local branch=$1
    local archive_tag="archive/$(date +%Y%m%d)/${branch##*/}"
    
    git tag -a "$archive_tag" "$branch" -m "Archived experimental branch $branch"
    git push origin "$archive_tag"
    git branch -d "$branch"
    echo "Archived $branch as $archive_tag"
}
```

## 2.3 实验追踪与版本控制

### 实验追踪工具选择

选择合适的实验追踪工具是建立可重现研究流程的关键。主流工具各有特色，需要根据团队规模和需求选择。

**MLflow：开源标准的选择**

MLflow 提供了完整的实验生命周期管理：

```python
import mlflow
from mlflow.tracking import MlflowClient
import hashlib
import json

class MLflowExperimentTracker:
    def __init__(self, experiment_name: str, tracking_uri: str = "file:./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
    def start_run(self, config: dict, tags: dict = None):
        """开始一个新的实验运行"""
        # 生成配置哈希作为运行标识
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        run_name = f"{config.get('model_name', 'unknown')}_{config_hash}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # 记录配置参数
            mlflow.log_params(self.flatten_dict(config))
            
            # 记录标签
            if tags:
                mlflow.set_tags(tags)
            
            # 记录代码版本
            mlflow.set_tag("git_commit", self.get_git_commit())
            mlflow.set_tag("git_branch", self.get_git_branch())
            
            return run.info.run_id
    
    def log_metrics_batch(self, metrics: dict, step: int):
        """批量记录指标"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact_with_metadata(self, file_path: str, metadata: dict):
        """记录文件及其元数据"""
        mlflow.log_artifact(file_path)
        # 同时记录元数据
        metadata_path = f"{file_path}.metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(metadata_path)
```

**Weights & Biases：云原生的强大功能**

W&B 提供了更丰富的可视化和协作功能：

```python
import wandb
from typing import Any, Dict, Optional
import numpy as np

class WandBTracker:
    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity
        
    def init_run(self, config: dict, name: Optional[str] = None, 
                 resume: Optional[str] = None):
        """初始化 W&B 运行"""
        run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            name=name,
            resume=resume,  # 支持断点续训
            save_code=True,  # 自动保存代码
            tags=self.generate_tags(config)
        )
        
        # 定义自定义指标
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        
        return run
    
    def log_distribution(self, name: str, data: np.ndarray, step: int):
        """记录数据分布"""
        wandb.log({
            f"{name}/mean": np.mean(data),
            f"{name}/std": np.std(data),
            f"{name}/min": np.min(data),
            f"{name}/max": np.max(data),
            f"{name}/histogram": wandb.Histogram(data)
        }, step=step)
    
    def log_gradient_flow(self, model, step: int):
        """记录梯度流信息"""
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                gradients.append({
                    "name": name,
                    "grad_norm": grad_norm
                })
        
        # 创建梯度表格
        grad_table = wandb.Table(
            columns=["layer", "gradient_norm"],
            data=[[g["name"], g["grad_norm"]] for g in gradients]
        )
        wandb.log({"gradients": grad_table}, step=step)
```

**TensorBoard：轻量级本地方案**

对于不需要云服务的场景，TensorBoard 仍是可靠选择：

```python
from torch.utils.tensorboard import SummaryWriter
import torch
from pathlib import Path

class TensorBoardTracker:
    def __init__(self, log_dir: str, comment: str = ""):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=30  # 定期刷新到磁盘
        )
        
    def log_model_architecture(self, model: torch.nn.Module, input_shape: tuple):
        """记录模型架构"""
        dummy_input = torch.randn(input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def log_attention_weights(self, attention_weights: torch.Tensor, 
                            step: int, head_idx: int = 0):
        """可视化注意力权重"""
        # 选择特定的注意力头
        attn = attention_weights[0, head_idx].cpu().numpy()
        
        # 创建热力图
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(attn, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Attention Weights - Head {head_idx}')
        
        self.writer.add_figure(f'attention/head_{head_idx}', fig, step)
        plt.close()
    
    def log_learning_rate_schedule(self, optimizer, step: int):
        """记录学习率变化"""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)
```

### 实验元数据管理

完整的元数据记录是实验可重现性的基础：

```python
import platform
import subprocess
import datetime
import psutil
import GPUtil

class ExperimentMetadata:
    @staticmethod
    def collect_system_info() -> dict:
        """收集系统信息"""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "hostname": platform.node(),
            "user": os.environ.get("USER", "unknown")
        }
    
    @staticmethod
    def collect_gpu_info() -> list:
        """收集 GPU 信息"""
        gpus = GPUtil.getGPUs()
        return [{
            "id": gpu.id,
            "name": gpu.name,
            "memory_total": gpu.memoryTotal,
            "driver": gpu.driver,
            "compute_capability": f"{gpu.major}.{gpu.minor}"
        } for gpu in gpus]
    
    @staticmethod
    def collect_dependencies() -> dict:
        """收集依赖版本"""
        deps = {}
        try:
            import torch
            deps["torch"] = torch.__version__
            deps["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        except ImportError:
            pass
            
        try:
            import transformers
            deps["transformers"] = transformers.__version__
        except ImportError:
            pass
            
        # 从 requirements.txt 或 pyproject.toml 读取
        if Path("requirements.txt").exists():
            with open("requirements.txt") as f:
                for line in f:
                    if "==" in line:
                        pkg, version = line.strip().split("==")
                        deps[pkg] = version
                        
        return deps
    
    @staticmethod
    def create_experiment_card(config: dict) -> dict:
        """创建实验卡片"""
        return {
            "experiment_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
            "system": ExperimentMetadata.collect_system_info(),
            "gpus": ExperimentMetadata.collect_gpu_info(),
            "dependencies": ExperimentMetadata.collect_dependencies(),
            "git": {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"]
                ).decode().strip(),
                "branch": subprocess.check_output(
                    ["git", "branch", "--show-current"]
                ).decode().strip(),
                "diff": subprocess.check_output(
                    ["git", "diff", "HEAD"]
                ).decode()
            }
        }
```

### 模型检查点策略

高效的检查点管理对于长时间训练至关重要：

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self, model, optimizer, epoch: int, 
                       metrics: dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # 常规检查点
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """保留最新的N个检查点"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) > self.max_checkpoints:
            for ckpt in checkpoints[:-self.max_checkpoints]:
                ckpt.unlink()
                
    def resume_from_checkpoint(self, checkpoint_path: Path, 
                              model, optimizer) -> dict:
        """从检查点恢复"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"]
        }
```

## 2.4 防止代码腐化的最佳实践

### 技术债务管理

LLM 后训练项目的快速迭代容易累积技术债务。主动管理技术债务是保持项目长期健康的关键。

**技术债务的量化与追踪**

```python
from typing import List, Dict
import ast
import re

class TechnicalDebtAnalyzer:
    def __init__(self, codebase_path: Path):
        self.codebase_path = codebase_path
        self.debt_markers = ["TODO", "FIXME", "HACK", "XXX", "DEPRECATED"]
        
    def scan_codebase(self) -> Dict[str, List[Dict]]:
        """扫描代码库中的技术债务标记"""
        debt_items = {marker: [] for marker in self.debt_markers}
        
        for py_file in self.codebase_path.rglob("*.py"):
            with open(py_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    for marker in self.debt_markers:
                        if marker in line:
                            debt_items[marker].append({
                                "file": str(py_file.relative_to(self.codebase_path)),
                                "line": line_num,
                                "content": line.strip(),
                                "priority": self.estimate_priority(line)
                            })
        
        return debt_items
    
    def calculate_complexity_metrics(self, file_path: Path) -> dict:
        """计算代码复杂度指标"""
        with open(file_path, 'r') as f:
            source = f.read()
            
        tree = ast.parse(source)
        
        metrics = {
            "cyclomatic_complexity": self.calculate_cyclomatic_complexity(tree),
            "lines_of_code": len(source.splitlines()),
            "num_functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            "num_classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            "max_nesting_depth": self.calculate_max_nesting(tree)
        }
        
        return metrics
    
    def generate_debt_report(self) -> str:
        """生成技术债务报告"""
        debt_items = self.scan_codebase()
        total_debt = sum(len(items) for items in debt_items.values())
        
        report = f"""
# 技术债务报告
生成时间: {datetime.datetime.now().isoformat()}
总债务项: {total_debt}

## 按类型分布
"""
        for marker, items in debt_items.items():
            report += f"- {marker}: {len(items)} 项\n"
            
        # 高优先级项目
        high_priority = []
        for marker, items in debt_items.items():
            high_priority.extend([
                item for item in items 
                if item["priority"] == "high"
            ])
        
        if high_priority:
            report += "\n## 高优先级债务\n"
            for item in high_priority[:10]:  # 显示前10个
                report += f"- {item['file']}:{item['line']} - {item['content']}\n"
                
        return report
```

**代码质量门禁**

```python
class CodeQualityGate:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        
    def check_diff_quality(self, diff_file: str) -> bool:
        """检查代码变更的质量"""
        checks = {
            "no_print_statements": self.check_no_prints(diff_file),
            "has_tests": self.check_has_tests(diff_file),
            "docstring_coverage": self.check_docstrings(diff_file),
            "type_hints": self.check_type_hints(diff_file)
        }
        
        failures = [name for name, passed in checks.items() if not passed]
        
        if failures:
            print(f"Quality gate failed: {', '.join(failures)}")
            return False
            
        return True
    
    def check_no_prints(self, diff: str) -> bool:
        """检查是否包含调试用的print语句"""
        pattern = r'\+.*print\('
        return not re.search(pattern, diff)
    
    def check_has_tests(self, diff: str) -> bool:
        """检查是否包含对应的测试"""
        # 如果修改了src/下的文件，应该有对应的test/下的修改
        src_modified = "src/" in diff
        test_modified = "test/" in diff or "tests/" in diff
        
        if src_modified and not test_modified:
            return False
        return True
```

### 代码复用与模块化

良好的模块化设计是防止代码腐化的基础：

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')

class BaseExperiment(ABC, Generic[T]):
    """实验基类，强制规范化实验流程"""
    
    def __init__(self, config: dict):
        self.config = config
        self.setup()
        
    @abstractmethod
    def setup(self):
        """初始化实验环境"""
        pass
    
    @abstractmethod
    def prepare_data(self) -> T:
        """数据准备"""
        pass
    
    @abstractmethod
    def build_model(self):
        """构建模型"""
        pass
    
    @abstractmethod
    def train_step(self, batch: T) -> dict:
        """单步训练"""
        pass
    
    @abstractmethod
    def evaluate(self) -> dict:
        """评估"""
        pass
    
    def run(self):
        """标准化的实验流程"""
        data = self.prepare_data()
        model = self.build_model()
        
        for epoch in range(self.config["num_epochs"]):
            for batch in data:
                metrics = self.train_step(batch)
                self.log_metrics(metrics)
                
            eval_metrics = self.evaluate()
            self.log_eval_metrics(eval_metrics)
```

**组件注册机制**

```python
class ComponentRegistry:
    """统一的组件注册机制，避免代码分散"""
    
    _registry = {
        "models": {},
        "datasets": {},
        "trainers": {},
        "evaluators": {}
    }
    
    @classmethod
    def register(cls, category: str, name: str):
        """装饰器：注册组件"""
        def decorator(component_cls):
            if category not in cls._registry:
                raise ValueError(f"Unknown category: {category}")
                
            cls._registry[category][name] = component_cls
            return component_cls
        return decorator
    
    @classmethod
    def get(cls, category: str, name: str):
        """获取注册的组件"""
        if category not in cls._registry:
            raise ValueError(f"Unknown category: {category}")
            
        if name not in cls._registry[category]:
            available = list(cls._registry[category].keys())
            raise ValueError(f"Unknown {category}: {name}. Available: {available}")
            
        return cls._registry[category][name]

# 使用示例
@ComponentRegistry.register("models", "llama2")
class LLaMA2Model:
    pass

@ComponentRegistry.register("datasets", "alpaca")
class AlpacaDataset:
    pass
```

### 持续集成与测试

**分层测试策略**

```python
import pytest
from unittest.mock import Mock, patch

class TestStrategy:
    """分层测试策略"""
    
    @staticmethod
    def unit_test_example():
        """单元测试：测试独立函数"""
        def test_config_merge():
            base = {"a": 1, "b": {"c": 2}}
            override = {"b": {"c": 3, "d": 4}}
            result = deep_merge(base, override)
            assert result == {"a": 1, "b": {"c": 3, "d": 4}}
    
    @staticmethod
    def integration_test_example():
        """集成测试：测试组件交互"""
        def test_model_with_dataloader():
            model = create_model(config)
            dataloader = create_dataloader(config)
            
            batch = next(iter(dataloader))
            output = model(batch)
            
            assert output.shape == expected_shape
    
    @staticmethod
    def smoke_test_example():
        """冒烟测试：快速验证基本功能"""
        def test_training_loop_runs():
            config = get_minimal_config()
            trainer = Trainer(config)
            
            # 只运行几步
            trainer.train(max_steps=10)
            
            assert trainer.global_step == 10
```

**CI/CD 配置**

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Run linting
        run: |
          flake8 src/ --max-line-length=100
          black --check src/
          isort --check-only src/
      
      - name: Type checking
        run: |
          mypy src/ --ignore-missing-imports
      
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Check code complexity
        run: |
          radon cc src/ -s -nb
```

### 文档与知识传承

**自动化文档生成**

```python
class DocumentationGenerator:
    """自动生成实验文档"""
    
    def generate_experiment_doc(self, experiment_class):
        """从实验类生成文档"""
        doc = f"# {experiment_class.__name__}\n\n"
        
        # 提取类文档字符串
        if experiment_class.__doc__:
            doc += f"{experiment_class.__doc__}\n\n"
        
        # 提取配置参数
        doc += "## Configuration Parameters\n\n"
        config_schema = experiment_class.get_config_schema()
        for param, schema in config_schema.items():
            doc += f"- **{param}**: {schema['type']} "
            if 'default' in schema:
                doc += f"(default: {schema['default']})"
            doc += f"\n  {schema.get('description', '')}\n"
        
        # 提取方法文档
        doc += "\n## Methods\n\n"
        for method_name in dir(experiment_class):
            if not method_name.startswith('_'):
                method = getattr(experiment_class, method_name)
                if callable(method) and method.__doc__:
                    doc += f"### {method_name}\n"
                    doc += f"{method.__doc__}\n\n"
        
        return doc
```

## 本章小结

本章深入探讨了 LLM 后训练实验代码基础设施的构建。我们学习了：

📌 **配置管理的层次化设计**：通过 YAML/TOML/Python 配置文件的合理选择，配置继承机制，以及运行时验证，建立了灵活且健壮的配置体系。记住：配置的复杂度应该与实验的复杂度相匹配，过度设计和设计不足都会降低效率。

📌 **实验环境的多维管理**：通过命令行参数、环境变量和 Git 分支的协同使用，实现了实验的隔离性和可重现性。关键原则是：Flag 用于实验特定配置，环境变量用于系统级设置，Git 分支用于代码版本管理。

📌 **实验追踪的全生命周期覆盖**：从 MLflow、W&B 到 TensorBoard，不同工具适合不同场景。核心是要记录足够的元数据以支持实验重现，包括代码版本、依赖环境、硬件配置等。

📌 **技术债务的主动管理**：通过代码质量门禁、模块化设计、自动化测试和文档生成，建立了防止代码腐化的多重防线。记住：技术债务是不可避免的，关键是要可见、可控、可偿还。

### 关键公式与度量

1. **技术债务利息** = $\sum_{i=1}^{n} \text{complexity}_i \times \text{change_frequency}_i$

2. **实验可重现性得分** = $\frac{\text{成功重现的实验数}}{\text{总实验数}} \times \text{元数据完整度}$

3. **配置复杂度** = $\log_2(\text{配置参数数}) \times \text{嵌套深度}$

## 常见陷阱与错误 (Gotchas)

⚠️ **配置地狱（Configuration Hell）**
- 错误：为每个小实验创建完全独立的配置文件
- 后果：配置文件爆炸式增长，难以维护
- 解决：使用配置继承，只记录与基线的差异

⚠️ **实验追踪过度或不足**
- 错误：记录所有可能的指标 vs 只记录最终结果
- 后果：存储爆炸或信息不足无法调试
- 解决：分层记录策略，关键指标详细记录，辅助指标采样记录

⚠️ **Git 分支管理混乱**
- 错误：所有实验都在 main 分支进行
- 后果：代码历史混乱，难以回溯
- 解决：严格的分支命名规范和生命周期管理

⚠️ **硬编码路径和配置**
- 错误：在代码中硬编码数据路径、模型路径
- 后果：代码无法跨环境运行
- 解决：所有路径通过配置或环境变量管理

⚠️ **忽视代码复杂度增长**
- 错误：为了快速实验不断添加 if-else 分支
- 后果：代码变成意大利面条，无法维护
- 解决：定期重构，使用策略模式或注册机制

⚠️ **检查点管理不当**
- 错误：保存所有检查点或只保存最后一个
- 后果：磁盘空间耗尽或无法恢复最佳模型
- 解决：滚动窗口策略 + 最佳模型保存

💡 **实用技巧**

1. **配置验证前置**：在实验开始前验证所有配置，fail fast
2. **实验命名规范**：`{date}_{model}_{dataset}_{key_hyperparam}`
3. **自动化清理**：定期清理过期的实验分支和检查点
4. **增量式日志**：使用结构化日志，便于后续分析
5. **配置快照**：每次实验开始时保存完整配置快照

## 练习题

### 基础题

**练习 2.1：配置文件格式选择**
你的团队正在启动一个新的 LLM 后训练项目。项目需要支持：(1) 非技术人员调整超参数；(2) 复杂的嵌套配置；(3) 动态计算某些参数。请为这个项目选择配置文件格式，并说明理由。

*Hint: 考虑混合方案，不同层次使用不同格式。*

<details>
<summary>参考答案</summary>

建议采用混合配置方案：
- **基础配置层（YAML）**：用于非技术人员可调整的参数，如学习率、批次大小等
- **高级配置层（Python）**：用于需要动态计算的参数，如根据 GPU 内存自动调整批次大小
- **用户覆盖层（TOML）**：用于用户特定的环境配置

实现方式：
1. 先加载 YAML 基础配置
2. 通过 Python 配置类进行动态计算和验证
3. 最后应用 TOML 用户覆盖

这样既保证了易用性，又提供了足够的灵活性。
</details>

**练习 2.2：实验追踪工具集成**
设计一个统一的接口，能够同时向 MLflow 和 W&B 记录实验指标。要求支持批量记录和异步写入。

*Hint: 使用适配器模式和队列机制。*

<details>
<summary>参考答案</summary>

```python
from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread

class ExperimentTracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int): pass

class UnifiedTracker:
    def __init__(self):
        self.trackers = []
        self.queue = Queue()
        self.worker = Thread(target=self._process_queue)
        self.worker.start()
    
    def add_tracker(self, tracker: ExperimentTracker):
        self.trackers.append(tracker)
    
    def log_metrics(self, metrics: dict, step: int):
        # 异步记录
        self.queue.put(("metrics", metrics, step))
    
    def _process_queue(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            event_type, data, step = item
            for tracker in self.trackers:
                try:
                    tracker.log_metrics(data, step)
                except Exception as e:
                    print(f"Tracker failed: {e}")
```

关键点：
1. 统一接口抽象
2. 异步队列避免阻塞训练
3. 错误隔离，单个 tracker 失败不影响其他
</details>

**练习 2.3：Git 分支清理策略**
编写一个脚本，自动清理实验分支。要求：(1) 保留最近 30 天的分支；(2) 保留有未合并提交的分支；(3) 归档重要实验结果。

*Hint: 使用 git for-each-ref 和 git cherry 命令。*

<details>
<summary>参考答案</summary>

```bash
#!/bin/bash

cleanup_experimental_branches() {
    local cutoff_date=$(date -d "30 days ago" +%s)
    
    git for-each-ref --format='%(refname:short) %(committerdate:unix)' refs/heads/exp/ | \
    while read branch timestamp; do
        # 检查年龄
        if [ $timestamp -lt $cutoff_date ]; then
            # 检查是否有未合并的提交
            unmerged=$(git cherry main $branch | grep "^+" | wc -l)
            
            if [ $unmerged -eq 0 ]; then
                # 检查是否标记为重要
                if git tag --list "important/$branch" | grep -q .; then
                    # 归档而非删除
                    git tag -a "archive/$(date +%Y%m)/$branch" $branch -m "Auto-archived"
                fi
                git branch -D $branch
                echo "Deleted: $branch"
            else
                echo "Kept (unmerged): $branch"
            fi
        fi
    done
}
```

关键检查：
1. 时间戳比较
2. 未合并提交检测
3. 重要性标记识别
</details>

### 挑战题

**练习 2.4：配置差异分析**
实现一个工具，能够：(1) 比较两个实验的配置差异；(2) 识别哪些配置变化导致了性能提升；(3) 生成配置优化建议。

*Hint: 考虑使用决策树或 SHAP 值分析配置重要性。*

<details>
<summary>参考答案</summary>

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap

class ConfigAnalyzer:
    def __init__(self, experiments: List[Dict]):
        self.experiments = experiments
        self.feature_names = self._extract_features()
        
    def _extract_features(self):
        # 提取所有配置键
        all_keys = set()
        for exp in self.experiments:
            all_keys.update(self._flatten_dict(exp['config']).keys())
        return sorted(all_keys)
    
    def _flatten_dict(self, d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def analyze_importance(self, metric='accuracy'):
        # 准备数据
        X = []
        y = []
        for exp in self.experiments:
            flat_config = self._flatten_dict(exp['config'])
            features = [flat_config.get(k, 0) for k in self.feature_names]
            X.append(features)
            y.append(exp['metrics'][metric])
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        # SHAP 分析
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # 生成重要性排名
        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = np.abs(shap_values[:, i]).mean()
        
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    def suggest_optimization(self, current_config: dict):
        importance = self.analyze_importance()
        suggestions = []
        
        # 找出表现最好的配置
        best_exp = max(self.experiments, key=lambda x: x['metrics']['accuracy'])
        best_config = self._flatten_dict(best_exp['config'])
        current_flat = self._flatten_dict(current_config)
        
        # 基于重要性生成建议
        for param, imp_score in importance[:5]:  # Top 5 重要参数
            if param in best_config and param in current_flat:
                if best_config[param] != current_flat[param]:
                    suggestions.append({
                        'parameter': param,
                        'current': current_flat[param],
                        'suggested': best_config[param],
                        'importance': imp_score
                    })
        
        return suggestions
```

核心思路：
1. 使用随机森林学习配置到性能的映射
2. SHAP 值量化每个配置的贡献
3. 基于历史最佳实践生成优化建议
</details>

**练习 2.5：实验代码版本隔离**
设计一个系统，能够为每个实验创建隔离的代码环境，支持：(1) 代码快照；(2) 依赖版本锁定；(3) 快速切换和恢复。

*Hint: 结合 Git worktree、Docker 或 Python 虚拟环境。*

<details>
<summary>参考答案</summary>

```python
import subprocess
import json
from pathlib import Path
import venv

class ExperimentEnvironment:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.envs_dir = base_dir / "environments"
        self.envs_dir.mkdir(exist_ok=True)
        
    def create_environment(self, exp_id: str, config: dict):
        env_path = self.envs_dir / exp_id
        
        # 1. 创建 Git worktree
        worktree_path = env_path / "code"
        subprocess.run([
            "git", "worktree", "add", 
            str(worktree_path), 
            config.get("git_commit", "HEAD")
        ])
        
        # 2. 创建 Python 虚拟环境
        venv_path = env_path / "venv"
        venv.create(venv_path, with_pip=True)
        
        # 3. 锁定依赖版本
        pip_path = venv_path / "bin" / "pip"
        requirements = config.get("requirements", [])
        
        # 生成 requirements.txt
        req_file = env_path / "requirements.txt"
        with open(req_file, 'w') as f:
            for pkg in requirements:
                f.write(f"{pkg}\n")
        
        # 安装依赖
        subprocess.run([
            str(pip_path), "install", "-r", str(req_file)
        ])
        
        # 4. 保存环境元数据
        metadata = {
            "exp_id": exp_id,
            "created_at": datetime.now().isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=worktree_path
            ).decode().strip(),
            "config": config
        }
        
        with open(env_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return env_path
    
    def activate_environment(self, exp_id: str):
        env_path = self.envs_dir / exp_id
        
        # 生成激活脚本
        activate_script = f"""
#!/bin/bash
export EXPERIMENT_ID={exp_id}
export PYTHONPATH={env_path}/code:$PYTHONPATH
source {env_path}/venv/bin/activate
cd {env_path}/code
echo "Environment {exp_id} activated"
"""
        
        script_path = env_path / "activate.sh"
        with open(script_path, 'w') as f:
            f.write(activate_script)
        
        script_path.chmod(0o755)
        return script_path
    
    def cleanup_environment(self, exp_id: str, archive: bool = True):
        env_path = self.envs_dir / exp_id
        
        if archive:
            # 归档重要文件
            archive_path = self.base_dir / "archives" / f"{exp_id}.tar.gz"
            archive_path.parent.mkdir(exist_ok=True)
            
            subprocess.run([
                "tar", "czf", str(archive_path),
                "-C", str(self.envs_dir),
                exp_id,
                "--exclude", "venv",
                "--exclude", ".git"
            ])
        
        # 清理 worktree
        worktree_path = env_path / "code"
        subprocess.run(["git", "worktree", "remove", str(worktree_path)])
        
        # 删除环境目录
        import shutil
        shutil.rmtree(env_path)
```

关键特性：
1. Git worktree 提供代码隔离
2. Python venv 提供依赖隔离  
3. 元数据记录保证可追溯性
4. 归档机制保留重要实验
</details>

**练习 2.6：分布式实验协调**
设计一个分布式实验管理系统，支持：(1) 多机器上的实验调度；(2) 资源（GPU）分配；(3) 实验失败自动重试。

*Hint: 考虑使用消息队列和状态机。*

<details>
<summary>参考答案</summary>

```python
from enum import Enum
from dataclasses import dataclass
import redis
import json
from typing import Optional

class ExperimentState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class ExperimentJob:
    exp_id: str
    config: dict
    state: ExperimentState
    assigned_worker: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class DistributedExperimentScheduler:
    def __init__(self, redis_host: str):
        self.redis = redis.Redis(host=redis_host, decode_responses=True)
        self.job_queue = "experiment_queue"
        self.worker_status = "worker_status"
        
    def submit_experiment(self, exp_id: str, config: dict):
        job = ExperimentJob(
            exp_id=exp_id,
            config=config,
            state=ExperimentState.PENDING
        )
        
        # 加入队列
        self.redis.lpush(self.job_queue, json.dumps({
            'exp_id': job.exp_id,
            'config': job.config,
            'state': job.state.value,
            'retry_count': job.retry_count
        }))
        
        # 记录作业状态
        self.redis.hset(f"job:{exp_id}", mapping={
            'state': job.state.value,
            'submitted_at': datetime.now().isoformat()
        })
    
    def worker_loop(self, worker_id: str, resources: dict):
        """工作节点主循环"""
        
        # 注册工作节点
        self.redis.hset(self.worker_status, worker_id, json.dumps({
            'status': 'idle',
            'resources': resources,
            'last_heartbeat': datetime.now().isoformat()
        }))
        
        while True:
            # 获取任务
            job_data = self.redis.brpop(self.job_queue, timeout=5)
            
            if job_data:
                _, job_str = job_data
                job = json.loads(job_str)
                
                # 检查资源需求
                if self._can_run(job['config'], resources):
                    self._run_experiment(worker_id, job)
                else:
                    # 放回队列末尾
                    self.redis.lpush(self.job_queue, job_str)
            
            # 发送心跳
            self._heartbeat(worker_id)
    
    def _can_run(self, config: dict, resources: dict) -> bool:
        """检查资源是否满足需求"""
        required_gpus = config.get('num_gpus', 1)
        available_gpus = resources.get('gpus', 0)
        
        required_memory = config.get('memory_gb', 16)
        available_memory = resources.get('memory_gb', 0)
        
        return (available_gpus >= required_gpus and 
                available_memory >= required_memory)
    
    def _run_experiment(self, worker_id: str, job: dict):
        exp_id = job['exp_id']
        
        try:
            # 更新状态
            self.redis.hset(f"job:{exp_id}", mapping={
                'state': ExperimentState.RUNNING.value,
                'worker': worker_id,
                'started_at': datetime.now().isoformat()
            })
            
            # 更新工作节点状态
            self.redis.hset(self.worker_status, worker_id, json.dumps({
                'status': 'busy',
                'current_job': exp_id
            }))
            
            # 执行实验
            result = self._execute_experiment(job['config'])
            
            # 标记完成
            self.redis.hset(f"job:{exp_id}", mapping={
                'state': ExperimentState.COMPLETED.value,
                'completed_at': datetime.now().isoformat(),
                'result': json.dumps(result)
            })
            
        except Exception as e:
            # 处理失败
            self._handle_failure(exp_id, job, str(e))
    
    def _handle_failure(self, exp_id: str, job: dict, error: str):
        retry_count = job.get('retry_count', 0)
        max_retries = job.get('max_retries', 3)
        
        if retry_count < max_retries:
            # 重试
            job['retry_count'] = retry_count + 1
            job['state'] = ExperimentState.RETRYING.value
            
            # 延迟重试（指数退避）
            delay = 2 ** retry_count
            self.redis.lpush(f"{self.job_queue}:delayed:{delay}", 
                           json.dumps(job))
            
            self.redis.hset(f"job:{exp_id}", mapping={
                'state': ExperimentState.RETRYING.value,
                'retry_count': retry_count + 1,
                'last_error': error
            })
        else:
            # 最终失败
            self.redis.hset(f"job:{exp_id}", mapping={
                'state': ExperimentState.FAILED.value,
                'failed_at': datetime.now().isoformat(),
                'error': error
            })
```

系统设计要点：
1. Redis 作为中央协调器
2. 基于资源的任务分配
3. 状态机管理实验生命周期
4. 指数退避的重试机制
5. 心跳机制检测工作节点健康
</details>

**练习 2.7：代码腐化度量与预警**
开发一个系统，能够：(1) 量化代码腐化程度；(2) 预测技术债务增长趋势；(3) 自动生成重构建议。

*Hint: 结合静态分析、Git 历史和复杂度度量。*

<details>
<summary>参考答案</summary>

```python
import ast
import git
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

class CodeHealthMonitor:
    def __init__(self, repo_path: str):
        self.repo = git.Repo(repo_path)
        self.metrics_history = []
        
    def calculate_health_score(self) -> float:
        """计算代码健康度得分 (0-100)"""
        metrics = {
            'complexity': self._measure_complexity(),
            'duplication': self._measure_duplication(),
            'test_coverage': self._measure_test_coverage(),
            'debt_density': self._measure_technical_debt(),
            'change_frequency': self._measure_change_frequency()
        }
        
        # 加权计算总分
        weights = {
            'complexity': -0.3,      # 复杂度越高分数越低
            'duplication': -0.2,     # 重复代码越多分数越低
            'test_coverage': 0.25,   # 测试覆盖率越高分数越高
            'debt_density': -0.15,   # 技术债务越多分数越低
            'change_frequency': -0.1 # 频繁修改的代码分数越低
        }
        
        score = 50  # 基准分
        for metric, value in metrics.items():
            score += weights[metric] * value
            
        return max(0, min(100, score))
    
    def _measure_complexity(self) -> float:
        """测量圈复杂度"""
        total_complexity = 0
        file_count = 0
        
        for py_file in Path(self.repo.working_dir).rglob("*.py"):
            with open(py_file) as f:
                tree = ast.parse(f.read())
                complexity = self._calculate_cyclomatic_complexity(tree)
                total_complexity += complexity
                file_count += 1
        
        return total_complexity / max(file_count, 1)
    
    def _measure_duplication(self) -> float:
        """测量代码重复率"""
        # 使用简化的方法：相似代码块检测
        code_blocks = []
        
        for py_file in Path(self.repo.working_dir).rglob("*.py"):
            with open(py_file) as f:
                content = f.read()
                # 提取函数体作为代码块
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        code_blocks.append(ast.unparse(node))
        
        # 计算相似度
        duplicates = 0
        for i, block1 in enumerate(code_blocks):
            for block2 in code_blocks[i+1:]:
                if self._similarity(block1, block2) > 0.8:
                    duplicates += 1
        
        return duplicates / max(len(code_blocks), 1) * 100
    
    def predict_debt_growth(self, days_ahead: int = 30) -> dict:
        """预测技术债务增长趋势"""
        
        # 收集历史数据
        history = []
        for days_ago in range(90, 0, -7):  # 过去90天，每周采样
            date = datetime.now() - timedelta(days=days_ago)
            commit = self._get_commit_at_date(date)
            
            if commit:
                self.repo.git.checkout(commit.hexsha)
                metrics = {
                    'date': date,
                    'debt_count': self._count_debt_markers(),
                    'complexity': self._measure_complexity()
                }
                history.append(metrics)
        
        # 回到当前分支
        self.repo.git.checkout('main')
        
        # 线性回归预测
        X = np.array([i for i in range(len(history))]).reshape(-1, 1)
        y = np.array([h['debt_count'] for h in history])
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测未来
        future_x = len(history) + days_ahead // 7
        predicted_debt = model.predict([[future_x]])[0]
        
        return {
            'current_debt': history[-1]['debt_count'] if history else 0,
            'predicted_debt': predicted_debt,
            'growth_rate': model.coef_[0],
            'confidence': model.score(X, y)
        }
    
    def generate_refactoring_suggestions(self) -> list:
        """生成重构建议"""
        suggestions = []
        
        # 分析热点文件（频繁修改且复杂度高）
        hotspots = self._identify_hotspots()
        
        for file_path, metrics in hotspots[:5]:  # Top 5 热点
            suggestion = {
                'file': file_path,
                'reason': [],
                'actions': []
            }
            
            if metrics['complexity'] > 10:
                suggestion['reason'].append(f"高复杂度: {metrics['complexity']}")
                suggestion['actions'].append("拆分大函数为更小的功能单元")
            
            if metrics['change_frequency'] > 20:
                suggestion['reason'].append(f"频繁修改: {metrics['change_frequency']}次/月")
                suggestion['actions'].append("考虑抽象出稳定接口")
            
            if metrics['duplication'] > 20:
                suggestion['reason'].append(f"代码重复: {metrics['duplication']}%")
                suggestion['actions'].append("提取公共代码到工具模块")
            
            if metrics['test_coverage'] < 50:
                suggestion['reason'].append(f"测试覆盖率低: {metrics['test_coverage']}%")
                suggestion['actions'].append("增加单元测试覆盖")
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _identify_hotspots(self) -> list:
        """识别代码热点"""
        file_metrics = {}
        
        # 分析 Git 历史
        for commit in self.repo.iter_commits('main', max_count=100):
            for file in commit.stats.files:
                if file.endswith('.py'):
                    if file not in file_metrics:
                        file_metrics[file] = {
                            'change_frequency': 0,
                            'complexity': 0,
                            'duplication': 0,
                            'test_coverage': 0
                        }
                    file_metrics[file]['change_frequency'] += 1
        
        # 计算当前指标
        for file_path in file_metrics:
            full_path = Path(self.repo.working_dir) / file_path
            if full_path.exists():
                with open(full_path) as f:
                    tree = ast.parse(f.read())
                    file_metrics[file_path]['complexity'] = \
                        self._calculate_cyclomatic_complexity(tree)
        
        # 按综合得分排序
        scored = []
        for file, metrics in file_metrics.items():
            score = (metrics['change_frequency'] * 0.4 + 
                    metrics['complexity'] * 0.4 +
                    metrics['duplication'] * 0.2)
            scored.append((file, metrics, score))
        
        return sorted(scored, key=lambda x: x[2], reverse=True)
```

监控系统特点：
1. 多维度健康评分
2. 基于历史数据的趋势预测
3. 热点分析识别问题区域
4. 可操作的重构建议
5. 持续监控和预警机制
</details>

**练习 2.8：实验结果自动分析**
实现一个系统，能够自动分析实验结果，识别：(1) 异常实验；(2) 性能瓶颈；(3) 最优配置组合。

*Hint: 使用异常检测、性能剖析和贝叶斯优化。*

<details>
<summary>参考答案</summary>

```python
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

class ExperimentAnalyzer:
    def __init__(self, experiment_history: list):
        self.history = experiment_history
        
    def detect_anomalies(self) -> list:
        """检测异常实验"""
        anomalies = []
        
        # 提取关键指标
        metrics = ['loss', 'accuracy', 'training_time']
        
        for metric in metrics:
            values = [exp['metrics'].get(metric, 0) for exp in self.history]
            
            if len(values) < 3:
                continue
                
            # 使用 IQR 方法检测异常
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, exp in enumerate(self.history):
                value = exp['metrics'].get(metric, 0)
                if value < lower_bound or value > upper_bound:
                    anomalies.append({
                        'exp_id': exp['id'],
                        'metric': metric,
                        'value': value,
                        'expected_range': (lower_bound, upper_bound),
                        'severity': 'high' if abs(value - np.mean(values)) > 3 * np.std(values) else 'medium'
                    })
        
        # 检测训练曲线异常
        for exp in self.history:
            if 'training_curve' in exp:
                curve_anomalies = self._detect_curve_anomalies(exp['training_curve'])
                if curve_anomalies:
                    anomalies.extend(curve_anomalies)
        
        return anomalies
    
    def _detect_curve_anomalies(self, curve: list) -> list:
        """检测训练曲线异常"""
        anomalies = []
        
        # 检测 loss 爆炸
        losses = [point['loss'] for point in curve]
        if any(np.isnan(losses)) or any(np.isinf(losses)):
            anomalies.append({
                'type': 'loss_explosion',
                'severity': 'critical'
            })
        
        # 检测过拟合
        if len(curve) > 10:
            train_acc = [p.get('train_acc', 0) for p in curve[-10:]]
            val_acc = [p.get('val_acc', 0) for p in curve[-10:]]
            
            if train_acc and val_acc:
                gap = np.mean(train_acc) - np.mean(val_acc)
                if gap > 0.1:  # 10% 差距
                    anomalies.append({
                        'type': 'overfitting',
                        'severity': 'medium',
                        'train_val_gap': gap
                    })
        
        return anomalies
    
    def identify_bottlenecks(self) -> dict:
        """识别性能瓶颈"""
        bottlenecks = {
            'data_loading': [],
            'forward_pass': [],
            'backward_pass': [],
            'optimizer_step': []
        }
        
        for exp in self.history:
            if 'profiling' not in exp:
                continue
                
            prof = exp['profiling']
            
            # 分析各阶段耗时
            total_time = sum(prof.values())
            
            for stage, time in prof.items():
                percentage = (time / total_time) * 100
                
                # 如果某阶段占比异常高
                expected_percentages = {
                    'data_loading': 10,
                    'forward_pass': 30,
                    'backward_pass': 40,
                    'optimizer_step': 20
                }
                
                if stage in expected_percentages:
                    expected = expected_percentages[stage]
                    if percentage > expected * 1.5:  # 超过预期 50%
                        bottlenecks[stage].append({
                            'exp_id': exp['id'],
                            'percentage': percentage,
                            'expected': expected,
                            'suggestions': self._get_optimization_suggestions(stage)
                        })
        
        return bottlenecks
    
    def _get_optimization_suggestions(self, stage: str) -> list:
        """获取优化建议"""
        suggestions = {
            'data_loading': [
                "增加数据加载的 num_workers",
                "使用更高效的数据格式 (如 HDF5)",
                "实现数据预取和缓存",
                "考虑使用 DALI 或其他加速库"
            ],
            'forward_pass': [
                "使用混合精度训练 (AMP)",
                "启用 cudnn.benchmark",
                "考虑模型剪枝或量化",
                "使用更高效的算子实现"
            ],
            'backward_pass': [
                "使用梯度累积减少显存占用",
                "启用梯度检查点",
                "考虑使用 ZeRO 优化器",
                "检查是否有不必要的梯度计算"
            ],
            'optimizer_step': [
                "使用融合优化器 (如 FusedAdam)",
                "减少参数更新频率",
                "考虑使用 LARS 或 LAMB",
                "检查权重衰减设置"
            ]
        }
        
        return suggestions.get(stage, [])
    
    def find_optimal_config(self) -> dict:
        """使用贝叶斯优化找到最优配置"""
        
        # 准备数据
        configs = []
        scores = []
        
        param_names = set()
        for exp in self.history:
            flat_config = self._flatten_config(exp['config'])
            param_names.update(flat_config.keys())
            configs.append(flat_config)
            scores.append(exp['metrics'].get('accuracy', 0))
        
        param_names = sorted(param_names)
        
        # 转换为数值矩阵
        X = []
        for config in configs:
            x = []
            for param in param_names:
                value = config.get(param, 0)
                # 简单的数值化
                if isinstance(value, bool):
                    x.append(float(value))
                elif isinstance(value, str):
                    x.append(hash(value) % 100)  # 简化处理
                else:
                    x.append(float(value))
            X.append(x)
        
        X = np.array(X)
        y = np.array(scores)
        
        # 高斯过程回归
        kernel = Matern(length_scale=1.0, nu=2.5)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gpr.fit(X, y)
        
        # 贝叶斯优化：找下一个最佳点
        def acquisition_function(x):
            """Expected Improvement"""
            mu, sigma = gpr.predict(x.reshape(1, -1), return_std=True)
            
            best_y = np.max(y)
            z = (mu - best_y - 0.01) / sigma
            ei = (mu - best_y - 0.01) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            
            return ei[0]
        
        # 随机搜索找最大 EI
        best_ei = -np.inf
        best_config = None
        
        for _ in range(1000):
            # 在已有配置附近采样
            idx = np.random.randint(len(X))
            candidate = X[idx] + np.random.randn(len(param_names)) * 0.1
            
            ei = acquisition_function(candidate)
            if ei > best_ei:
                best_ei = ei
                best_config = candidate
        
        # 转换回配置字典
        optimal_config = {}
        for i, param in enumerate(param_names):
            optimal_config[param] = best_config[i]
        
        # 预测性能
        predicted_score, uncertainty = gpr.predict(
            best_config.reshape(1, -1), 
            return_std=True
        )
        
        return {
            'config': optimal_config,
            'predicted_score': predicted_score[0],
            'uncertainty': uncertainty[0],
            'expected_improvement': best_ei
        }
    
    def _flatten_config(self, config: dict, prefix: str = '') -> dict:
        """展平嵌套配置"""
        flat = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value
        return flat
```

分析系统功能：
1. 异常检测：IQR 方法 + 曲线分析
2. 瓶颈识别：性能剖析 + 阶段分析
3. 配置优化：贝叶斯优化 + 高斯过程
4. 可操作建议：针对性优化方案
5. 不确定性量化：预测置信度
</details>

---

通过完成这些练习，你将掌握构建健壮的 LLM 后训练实验基础设施的核心技能。记住，好的基础设施是高效实验的基石，值得在项目初期投入时间建设。
