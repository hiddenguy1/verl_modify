# GROUP PPO: Token-Level Advantage Computation Implementation

## 概述

本项目实现了基于token级别的GROUP PPO算法，该算法是对传统PPO算法的改进，通过动态确定组边界并只在组的末端计算advantage值来提高训练效率。实现基于verl框架，支持大规模语言模型的强化学习训练。

## 算法原理

GROUP PPO算法的核心思想是：
1. **动态分组**: 根据Algorithm 2中的variance ratio (r_grpo/r_ppo) 来动态确定组边界
2. **末端计算**: 只在每个组的末端token位置计算advantage值，其他位置的advantage为0
3. **方差优化**: 通过控制GRPO方差与PPO方差的比值来减少计算复杂度

## 环境要求

- GPU: 至少24GB显存（推荐RTX 4090, A100, H100）
- Docker支持
- CUDA 12.1+
- Python 3.10

## 快速开始

### 1. 拉取Docker镜像

我们提供了预编译的Docker镜像，包含所有必要的依赖：

```bash
# 拉取stable版本的verl镜像（推荐）
docker pull whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3

# 或者拉取最新版本(暂未尝试)
docker pull hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0
```

### 2. 启动Docker容器

```bash
# 创建并启动容器
docker run --rm -it \       ## --rm 代表关闭容器会直接remove掉，下次不可使用
    --gpus all \            ## 可以用多少gpu(options: “--gpus all”、“--gpus='"device=0,1,2,3"'”)
    --shm-size 128G \       
    -p 8265:8265 \          ## 映射端口
    -v $HOME:$HOME \
    -v $(pwd):/workspace \  ## 这是挂载目录。 ":"冒号前是自己目录（我给你的代码）所在地；冒号后是挂载到docker的具体位置
    whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3 \ ## 镜像名称
    /bin/bash
```
具体示例：

```bash
docker run -tid \
    --name verl_new \
    --gpus all \
    --shm-size=10G \
    --entrypoint=/bin/bash \
    -v F:/tasks/verl_raw:/workspace/verl_raw \
    whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3 \
```

### 3. 安装项目代码

在容器内执行：

```bash
# 克隆原始verl库
git clone https://github.com/volcengine/verl.git
cd verl

# 备份原始文件
cp verl/trainer/ppo/ray_trainer.py verl/trainer/ppo/ray_trainer_original.py
cp verl/trainer/ppo/core_algos.py verl/trainer/ppo/core_algos_original.py
cp examples/ppo_trainer/config/ppo_trainer.yaml examples/ppo_trainer/config/ppo_trainer_original.yaml

# 替换为修改版本
# 把/workspace/verl_raw中的 ray_trainer.py、core_algos.py、ppo_trainer.yaml放到clone的verl中

pip install git+https://github.com/NVIDIA/nvidia-dlfw-inspect.git@v0.1

pip3 install -e .[vllm]


```

### 4. 准备数据集

```bash
# 下载并预处理GSM8K数据集
python3 examples/data_preprocess/gsm8k.py --local_dir /workspace/verl/data/gsm8k
```


### 5. 下载模型

```bash
## docker中已经内置modelscope直接下载就行
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir /workspace/verl/models
```

## 运行TOKEN级别的GROUP PPO训练

### 1. 单GPU训练（测试用）

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/workspace/verl/data/gsm8k/train.parquet \
 data.val_files=/workspace/verl/data/gsm8k/test.parquet \
 data.train_batch_size=8 \
 data.max_prompt_length=256 \                   ## 建议后续加大
 data.max_response_length=128 \                 ## 建议后续加大
 actor_rollout_ref.model.path=/workspace/verl/models/Qwen2.5-0.5B-Instruct \ ## 改为你自己模型的地址
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=4 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \## 这里是你希望上述模型用vllm部署的话占用总显存的百分比（最好是0.4-0.6,不然容易报OOM）
 actor_rollout_ref.rollout.n=1 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 critic.optim.lr=1e-5 \
 critic.model.path=/workspace/verl/models/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=1 \
 algorithm.adv_estimator=group_ppo \            ## options:gae、group_ppo、grpo...详细ppo_trainer.yaml
 algorithm.group_params.group_variance_threshold=0.8 \
 algorithm.group_params.group_max_size=8 \
 algorithm.group_params.group_epsilon=1e-6 \
 algorithm.group_params.adaptive_threshold=false \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.project_name=group_qwen \              ## 记得修改project_name，这是你存checkpoints的目录
 trainer.logger=['console'] \                   ## 这里如果要wandb，可以修改为wandb
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=1 \                    ## 这里是单卡
 trainer.nnodes=1 \                             ## 目前verl只能支持单个node
 trainer.save_freq=100 \                        ## checkpoints和log的保存频率
 trainer.test_freq=100 \                        ## test频率
 trainer.total_epochs=1 2>&1 | tee verl_group_demo.log  ## log保存地...
```

### 2. 多GPU训练（这里我没有测试过，需要你自己调整参数）

```bash
# 4卡训练脚本
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=5e-6 \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.adv_estimator=group_ppo \
    algorithm.group_params.group_variance_threshold=1.0 \
    algorithm.group_params.group_max_size=32 \
    algorithm.group_params.adaptive_threshold=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \                 
    trainer.test_freq=100 \
    trainer.total_epochs=1 2>&1 | tee verl_group.log 
```

## 关键配置参数说明

### GROUP PPO算法参数

```yaml
algorithm:
  adv_estimator: group_ppo  # 使用GROUP PPO算法
  group_params:
    group_variance_threshold: 1.0      # r_grpo/r_ppo的阈值
    group_max_size: 16                 # 最大组大小
    group_epsilon: 1e-6                # 数值稳定性参数
    adaptive_threshold: true           # 是否启用自适应阈值
    threshold_lr: 0.01                 # 阈值调整学习率
    min_threshold: 0.1                 # 最小阈值
    max_threshold: 10.0                # 最大阈值
```

### 重要技术细节

1. **Token级别实现**: 与batch级别不同，我们的实现在每个序列的token级别进行分组
2. **末端计算**: 只有满足组边界条件的token位置才计算advantage，大大减少了计算量
3. **动态分组**: 基于Algorithm 2的variance ratio来动态确定组边界
4. **Critic依赖**: 训练时需要critic model提供value estimates

## 监控和调试

### 1. 查看训练日志

```bash
# 查看wandb面板（如果启用）
# 访问 https://wandb.ai/your-project/runs
```

### 2. 关键指标监控

- `token_group/total_tokens`: 总token数
- `token_group/advantage_positions`: 计算advantage的位置数
- `token_group/compression_ratio`: 压缩比率（越高越好）
- `token_group/speedup_factor`: 加速因子

### 3. 常见问题排查

**内存不足 (OOM)**:
```bash
# 减少batch size
data.train_batch_size=64
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
```

**收敛慢**:
```bash
# 调整学习率
actor_rollout_ref.actor.optim.lr=5e-7
# 调整group参数
algorithm.group_params.group_variance_threshold=0.5
```

**性能问题**:
```bash
# 启用优化选项
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.free_cache_engine=False
```

## 实验验证

### 1. 性能对比测试

```bash
# 运行标准PPO作为baseline
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    # ... 其他参数相同

# 运行GROUP PPO
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=group_ppo \
    # ... 其他参数相同

# 比较训练时间和最终性能
```

## 参考内容
1. https://verl.readthedocs.io/en/latest/start/install.html
2. https://verl.readthedocs.io/en/latest/start/quickstart.html
3. https://github.com/volcengine/verl
