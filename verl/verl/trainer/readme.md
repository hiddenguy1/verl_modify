# Group_PPO 说明与使用指南

## 一、Group_PPO 原理简介

Group_PPO（分组PPO）是一种高效的PPO变体，其核心思想是：
- **只在分组端点（endpoints）上计算PPO损失和反向传播**，其余token不参与loss计算。
- **advantage（优势）依然用GAE（Generalized Advantage Estimation）在所有token上计算**，但只有端点位置的loss会被用于参数更新。
- 这样可以大幅减少反向传播的token数，提高训练效率，尤其适合长序列和大模型。

---

## 二、详细代码修改位置与原因

### 1. verl/verl/trainer/ppo/core_algos.py

- **文件头部**
  - 新增注释，说明Group_PPO的整体原理和端点mask的作用。
  - 目的：方便理解Group_PPO的设计思想。

- **compute_policy_loss 函数**
  - 新增 `group_endpoints` 参数，并在函数内部用 `effective_mask = response_mask * group_endpoints` 替换原有的 response_mask。
  - 目的：只在端点位置（mask为1）计算PPO损失，其余位置loss为0，不参与反向传播。

- **compute_group_advantage 函数**
  - 已删除。
  - 目的：旧实现只在端点有非零advantage，不再符合GAE全token计算的Group_PPO新逻辑。

- **_find_group_endpoints 函数**
  - 保留，用于生成端点mask（endpoint_mask）。
  - 目的：作为分组策略的核心工具，决定哪些token为端点。

---

### 2. verl/verl/trainer/ppo/ray_trainer.py

- **GROUP分支（AdvantageEstimator.GROUP）**
  - 删除了对 compute_group_advantage 的调用。
  - 新增：先用 GAE 算 advantage（全token），再用 _find_group_endpoints 生成端点mask（endpoint_mask），并存入 data.batch["endpoint_mask"]。
  - 目的：保证advantage全token计算，端点mask只用于loss，完全符合Group_PPO新范式。

---

### 3. verl/verl/workers/fsdp_workers.py

- **_should_use_group_optimization 方法**
  - 优先判断 data.batch 是否有 endpoint_mask 字段，有则直接返回True。
  - 目的：确保只要有端点mask就走Group_PPO优化路径。

- **_update_policy_with_group_optimization 方法**
  - 优先使用 data.batch['endpoint_mask'] 作为端点mask，只有没有时才fallback到(advantages!=0)。
  - 目的：保证端点mask的来源权威且一致，避免因advantage稀疏性误判。

---

### 4. verl/verl/workers/actor/dp_actor.py

- **update_policy 方法**
  - 在函数开头通过 `endpoint_mask = data.meta_info.get("endpoint_mask", None)` 获取端点mask。
  - 新增 group_optimization = endpoint_mask is not None 逻辑，简化判断。
  - 目的：只要有端点mask就走Group_PPO分支，保证loss和反向传播只在端点上进行。

---

## 三、Group_PPO 运行流程

1. **采样数据**：与普通PPO一致，采集prompt+response样本。
2. **奖励计算**：与普通PPO一致。
3. **advantage计算**：用GAE算法对所有token计算advantage。
4. **端点mask生成**：用`_find_group_endpoints`生成端点mask（endpoint_mask），只在端点为1。
5. **损失计算**：只在端点mask为1的位置做PPO损失和反向传播，其余token loss为0，不参与参数更新。
6. **参数更新**：如常规PPO。

---

## 四、与标准PPO的区别

- **PPO**：所有token都参与loss和反向传播，计算量大。
- **Group_PPO**：只在端点token参与loss和反向传播，极大减少计算量，提升大模型训练效率。

---

## 五、使用建议

- 推荐在`config`中设置`algorithm.adv_estimator=group_ppo`，并合理调整`group_params`（如`group_variance_threshold`、`group_max_size`等）以获得更高的压缩率。
- 可通过日志中的`compression_ratio`、`active_positions`等指标监控端点mask的稀疏度和加速效果。
- 对于7B/8B等大模型，建议多卡并行，适当减小batch和序列长度，优先保证显存不溢出。

---

如需进一步定制分组策略或有其它优化需求，欢迎联系开发者或提交issue。 
