# 8卡H100下Qwen2.5-3B/7B-Instruct模型 Group_PPO 推荐参数配置

本说明针对8卡H100服务器，分别给出Qwen2.5-3B-Instruct和Qwen2.5-7B-Instruct模型在Group_PPO训练下的推荐参数配置、理由与命令行示例。

---

## 一、硬件资源
- 8卡H100（每卡80GB显存）
- 支持大batch、长序列和大模型训练

---

## 二、通用参数建议

| 参数名 | 推荐设置 | 说明 |
|---|---|---|
| `trainer.nnodes` | 1 | 单机 |
| `trainer.n_gpus_per_node` | 8 | 8卡 |
| `data.train_batch_size` | 32~64 | 总batch，分到每卡4~8，保证不OOM |
| `data.max_prompt_length` | 128~256 | GSM8K等任务，128~256足够 |
| `data.max_response_length` | 64~128 | 生成任务，64~128较常见 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 8~16 | 取决于总batch和梯度累积 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 1~2 | 单卡微batch，防止OOM |
| `critic.ppo_micro_batch_size_per_gpu` | 1~2 | 同上 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 1~2 | 3B可1，7B建议2 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.7~0.8 | H100显存大，可适当提高 |
| `algorithm.adv_estimator` | group_ppo | 启用Group_PPO |
| `algorithm.group_params.group_variance_threshold` | 0.8 | 分组阈值，默认即可 |
| `algorithm.group_params.group_max_size` | 8 | 分组最大长度，默认即可 |
| `algorithm.group_params.group_epsilon` | 1e-6 | 默认即可 |
| `algorithm.group_params.adaptive_threshold` | false | 默认即可 |
| `trainer.save_freq` | 1000 | 视数据量调整 |
| `trainer.test_freq` | 1000 | 视数据量调整 |
| `trainer.total_epochs` | 1~3 | 视任务和数据量调整 |

---

## 三、Qwen2.5-3B-Instruct 推荐配置

### 首先用小batch，短prompt，短response来跑
```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/workspace/verl/data/gsm8k/train.parquet \
 data.val_files=/workspace/verl/data/gsm8k/test.parquet \
 data.train_batch_size=64 \
 data.max_prompt_length=256 \
 data.max_response_length=128 \
 actor_rollout_ref.model.path=/workspace/verl/models/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=1 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
 critic.optim.lr=1e-5 \
 critic.model.path=/workspace/verl/models/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.adv_estimator=group_ppo \
 algorithm.group_params.group_variance_threshold=2.0 \
 algorithm.group_params.group_max_size=8 \
 algorithm.group_params.group_epsilon=1e-6 \
 algorithm.group_params.adaptive_threshold=false \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.project_name=group_ppo_qwen_8gpu \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=100 \
 trainer.test_freq=100 \
 trainer.total_epochs=1 2>&1 | tee verl_group_ppo_8gpu.log 
```

#### 说明与建议
- train_batch_size=16：8卡，每卡2，安全起步。
- max_prompt_length=128, max_response_length=64：大幅降低显存压力，适合试跑。
- n_gpus_per_node=8：充分利用8卡。
- algorithm.adv_estimator=group_ppo：启用Group_PPO。
- ppo_mini_batch_size=4：可根据实际情况调整，保证能整除总batch。
- gpu_memory_utilization=0.7：先保守，防止爆显存。
- 如试跑无OOM，可逐步增大batch和序列长度，建议每次增量不超过50%。
- 如需针对7B模型或更大batch/序列长度配置，建议先用上述参数试跑，确认稳定后再逐步提升。如有报错或OOM，最好扔给模型分析下参数和原因！

`tips`
**需要注意的是：保证ppo_mini_batch_size能被gpu数 × ppo_micro_batch_size_per_gpu整除。**
**需要注意的是：保证ppo_mini_batch_size能被gpu数 × ppo_micro_batch_size_per_gpu整除。**
**需要注意的是：保证ppo_mini_batch_size能被gpu数 × ppo_micro_batch_size_per_gpu整除。**

四、Qwen2.5-7B-Instruct 推荐配置
```bash
#!/bin/bash

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/workspace/verl/data/gsm8k/train.parquet \
 data.val_files=/workspace/verl/data/gsm8k/test.parquet \
 data.train_batch_size=64 \
 data.max_prompt_length=256 \
 data.max_response_length=128 \
 actor_rollout_ref.model.path=/workspace/verl/models/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.rollout.n=1 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
 critic.optim.lr=1e-5 \
 critic.model.path=/workspace/verl/models/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=2 \
 algorithm.adv_estimator=group_ppo \
 algorithm.group_params.group_variance_threshold=2.0 \
 algorithm.group_params.group_max_size=8 \
 algorithm.group_params.group_epsilon=1e-6 \
 algorithm.group_params.adaptive_threshold=false \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.project_name=group_ppo_qwen_8gpu \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=100 \
 trainer.test_freq=100 \
 trainer.total_epochs=1 2>&1 | tee verl_group_ppo_8gpu.log 


```
`tips`
**需要注意的是：保证ppo_mini_batch_size能被gpu数 × ppo_micro_batch_size_per_gpu整除。**
**需要注意的是：保证ppo_mini_batch_size能被gpu数 × ppo_micro_batch_size_per_gpu整除。**
**需要注意的是：保证ppo_mini_batch_size能被gpu数 × ppo_micro_batch_size_per_gpu整除。**
