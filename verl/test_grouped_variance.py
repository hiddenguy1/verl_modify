# test_grouped_variance.py
import torch
import numpy as np
from verl.trainer.ppo.core_algos import compute_grouped_variance_advantage

def test_grouped_variance():
    batch_size, seq_len = 4, 10
    token_level_rewards = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    index = np.array([0, 0, 1, 1])  # 两个prompt，每个有两个response
    values = torch.randn(batch_size, seq_len)
    
    config = {
        "max_group_size": 4,
        "min_group_size": 2, 
        "variance_threshold": 0.1,
        "gamma": 0.99,
        "lam": 0.95,
    }
    
    advantages, returns = compute_grouped_variance_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=config,
        values=values
    )
    
    print(f"Input shape: {token_level_rewards.shape}")
    print(f"Output advantages shape: {advantages.shape}")
    print(f"Non-zero advantages: {(advantages != 0).sum().item()}")
    print("✓ 测试通过！")

if __name__ == "__main__":
    test_grouped_variance()