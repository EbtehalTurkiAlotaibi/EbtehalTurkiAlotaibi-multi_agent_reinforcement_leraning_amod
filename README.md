# Multi Agent Autonomous Mobility on Demand


This version is an extension of the classic Taxi by gym with additional multi-agent setting and recharging action. 
This version is compatible with [PyMARL](https://github.com/oxwhirl/pymarl) state-of-art algorithms

PyMARL is WhiRL's framework for deep multi-agent reinforcement learning and includes implementations of the following algorithms:

QMIX: [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)

COMA: [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

VDN: [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296)

IQL: [Independent Q-Learning](https://arxiv.org/abs/1511.08779)

QTRAN: [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)


To test the environment:

1- Go to config/algs and adjust algorithm setting.

2- Go to config/env and adjust environment setting.

3- run with 
```
python3 main.py --env-config=driver --config=vdn
```
where --config could have : iql/dmix/vdn/coma/qtran

To try your own version of reward shaping:

1- Go to envs/driver.py

2- You can edit rewards per agent in

```
def calculate_reward(self,state,action):
```
or per step in
```
def step(self, actions):
```

