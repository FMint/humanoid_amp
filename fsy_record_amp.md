```bash
python -m humanoid_amp.train --task Isaac-G1-AMP-Walk-Direct-v0 --headless --num_envs 4096
```

```bash
python -m humanoid_amp.play --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 4 --checkpoint logs/skrl/g1_amp_walk/2026-03-30_10-36-33_ppo_torch/checkpoints/best_agent.pt
```