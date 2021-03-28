# bomberman_rl
Setup for a project/competition amongst students to train a winning 
Reinforcement Learning agent for the classic game Bomberman.

The agent that was uploaded for the final tournament is custom_sarsa_agent. 
It was trained using

```bash
python main.py play --agents custom_sarsa_agent --no-gui --train 1 --n-rounds 20000
```

which was enough to fit the model 4 times.

To play:
```bash
python main.py play --agents custom_sarsa_agent --n-rounds 1
```

This agent performs relatively well on a field with no crates and no opponents 
(CRATE_DENSITY = 0 in settings.py).

The custom_sarsa_pca_agent was an attempt at creating an agent that could play 
the game on a field with crates. For the kernel PCA transformation and training,
100000 states per action were first collected using
```bash
python main.py play --agents rule_based_agent_pca_sarsa --no-gui --train 1 --n-rounds 25000
```
Then, custom_sarsa_pca_agent/pca.py and custom_sarsa_pca_agent/train_with_saved_states.py
were executed.