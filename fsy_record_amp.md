```bash
python -m humanoid_amp.train --task Isaac-G1-AMP-Walk-Direct-v0 --headless --num_envs 4096

python -m humanoid_amp.train --task Isaac-G1-AMP-Dance-Direct-v0 --headless --num_envs 4096
```

train1: no push
train2: force(50-600)
train3: no push dance
train4: force(50-300)
train5: force(50-900)


amp-train1:
```bash
python -m humanoid_amp.play --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 4 --checkpoint logs/skrl/g1_amp_walk/2026-03-30_10-36-33_ppo_torch/checkpoints/best_agent.pt
```
amp-train3:
```bash
python -m humanoid_amp.play --task Isaac-G1-AMP-Dance-Direct-v0 --num_envs 4 --checkpoint logs/skrl/g1_amp_dance/2026-04-16_05-57-54_ppo_torch/checkpoints/best_agent.pt
```


damp-train2:
```bash
python -m humanoid_amp.play --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 4 --checkpoint logs/skrl/g1_amp_walk/2026-04-09_08-18-50_ppo_torch/checkpoints/best_agent.pt
```
train4：
```bash
python -m humanoid_amp.play --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 4 --checkpoint logs/skrl/g1_amp_walk/2026-04-16_10-40-36_ppo_torch/checkpoints/best_agent.pt
```
train5：
```bash
python -m humanoid_amp.play --task Isaac-G1-AMP-Walk-Direct-v0 --num_envs 4 --checkpoint logs/skrl/g1_amp_walk/2026-04-16_10-44-26_ppo_torch/checkpoints/best_agent.pt
```