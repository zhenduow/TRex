# TRex
A personal implementation of the paper "Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations" (TRex) for conversational search policy agent. To use it for your task. You need to trim down some parameters and customize it for you task a little bit.

Inside this repo is a RL agent class that support TRex reward estimation and PPO policy optimization.

## How to use it
A typical call order of the functions in an epoch should be like:
your_rollout_function()         # to get on-policy training data as well as the trajectories for reward estimation net training
TRexAgent.trex_reward_update()  # to train the reward estimator
TRexAgent.train_policy()        # to train actor critic (AC) policy network
TRexAgent.train_value()         # to train AC value network

Please see the comments in each function for more information.
