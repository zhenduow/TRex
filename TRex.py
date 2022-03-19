from math import e
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import warnings 
import math
import random
from torch.distributions import Categorical
from collections import OrderedDict
import statistics

class TRexAgent():
    '''
    The multi-objective Inverse reinforcement learning Agent for conversational search.
    This agent has multiple policies each represented by one <agent> object.
    '''
    def __init__(self, n_action, observation_dim, top_n, lr, lrdc, gamma, weight_decay, dropout_ratio, n_rewards, traj_sample_length, epsilon=1.0, eps_dec=1e-3, eps_min=0.01, max_experience_length = 8000,
        ppo_clip_val = 0.2,
        kl_div_val = 0.01,
        max_policy_train_iter = 40,
        max_value_train_iter = 40,
        policy_lr = 1e-4,
        value_lr = 1e-4,
        ):
        self.lr = lr
        self.lrdc = lrdc
        self.weight_decay = weight_decay
        self.n_action = n_action
        self.top_n = top_n
        self.loss = nn.MSELoss()    
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.observation_dim = observation_dim
        self.policy = LinearDeepNetwork(n_actions = n_action, input_dims = (2+2*self.top_n) * observation_dim + (2) * self.top_n)
        self.n_rewards = n_rewards
        self.reward = []
        for i in range(n_rewards):
            self.reward.append(LinearDeepNetwork_no_activation(n_actions = 1, input_dims = (2+self.top_n) * observation_dim + self.top_n, hidden_size = 256, dropout_ratio = dropout_ratio))
        self.traj_sample_length = traj_sample_length
        self.policy_params = self.policy.parameters()
        self.reward_params = [self.reward[i].parameters() for i in range(n_rewards)]
        self.reward_optimizer = [optim.RMSprop(self.reward_params[i], lr=self.lr, weight_decay = self.weight_decay) for i in range(n_rewards)]
        self.policy_optimizer = optim.RMSprop(self.policy_params, lr=self.lr, weight_decay = self.weight_decay)
        self.scheduler = [optim.lr_scheduler.ExponentialLR(self.reward_optimizer[i], gamma=self.lrdc) for i in range(n_rewards)]
        self.expert_traj_history = []
        self.self_traj_history = []
        self.experiences = []
        self.experiences_replay_times = 3
        self.max_experience_length = max_experience_length
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = gamma
        self.reward_history = {}
        for k in range(self.n_rewards):
            self.reward_history[k] = []
  
        self.ac = ActorCriticNetwork(obs_space_size = (2+2*self.top_n) * observation_dim + (2) * self.top_n, action_space_size = self.n_action)
        self.ppo_clip_val = ppo_clip_val
        self.kl_div_val = kl_div_val
        self.max_policy_train_iter = max_policy_train_iter
        self.max_value_train_iter = max_value_train_iter

        policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layers.parameters())
        self.policy_optim = optim.Adam(policy_params, lr=policy_lr)
    
        value_params = list(self.ac.shared_layers.parameters()) + list(self.ac.value_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr=value_lr)
    
    def save(self, path):
        T.save(self.policy.state_dict(), path+'_policy')
        for i in range(self.n_rewards):
            T.save(self.reward[i].state_dict(), path+'_reward'+str(i))
        T.save(self.ac.state_dict(), path+'_ac')
    
    def load(self, path):
        self.policy.load_state_dict(T.load(path+'_policy'))
        for i in range(self.n_rewards):
            self.reward[i].load_state_dict(T.load(path+'_reward'+str(i)))
        self.ac.load_state_dict(T.load(path+'_ac'))

    def trex_reward_update(self, all_traj_pairs):
        '''
        Take a trex training step
        '''
        batch_loss = 0
        error = 0
        for k in range(self.n_rewards):
            L = T.zeros(1).to(self.device)
            for traj_pair in all_traj_pairs:
                pos_traj, neg_traj = [(traj[0].tolist(), traj[1]) for traj in traj_pair[0]], [(traj[0].tolist(), traj[1]) for traj in traj_pair[1]]
                #print("pos",pos_traj)
                #print("neg",neg_traj)
                traj_sample_length = min(min(len(pos_traj), len(neg_traj)), self.traj_sample_length)
                pos_traj, neg_traj = [(T.tensor(traj[0]).to(self.device), traj[1]) for traj in pos_traj if traj not in neg_traj], [(T.tensor(traj[0]).to(self.device), traj[1]) for traj in neg_traj if traj not in pos_traj]                #pos_traj = random.sample(pos_traj, traj_sample_length)
                pos_traj = random.sample(pos_traj, traj_sample_length)
                neg_traj = random.sample(neg_traj, traj_sample_length)
                #print("random pos",pos_traj)
                #print("random neg",neg_traj)
                pos_input, neg_input = [], []
                for i, row in enumerate(pos_traj):
                    context = row[0][:2*self.observation_dim]
                    candidate_start = 2 * self.observation_dim if int(row[1]) > 0  else (2 + self.top_n) * self.observation_dim
                    candidate_end = candidate_start + self.top_n * self.observation_dim 
                    score_start = - self.top_n * (1 + int(row[1])) - 1 
                    score_end = score_start + self.top_n
                    candidate = T.cat((row[0][candidate_start: candidate_end], row[0][score_start: score_end]))
                    reward_input = T.cat((context, candidate))
                    pos_input.append(reward_input)

                for i, row in enumerate(neg_traj):
                    context = row[0][:2*self.observation_dim]
                    candidate_start = 2 * self.observation_dim if int(row[1]) > 0  else (2 + self.top_n) * self.observation_dim
                    candidate_end = candidate_start + self.top_n * self.observation_dim 
                    score_start = - self.top_n * (1 + int(row[1])) - 1 
                    score_end = score_start + self.top_n
                    candidate = T.cat((row[0][candidate_start: candidate_end], row[0][score_start: score_end]))
                    reward_input = T.cat((context, candidate))
                    neg_input.append(reward_input)

                pos_input_tensor = T.stack(pos_input).to(self.device)
                neg_input_tensor = T.stack(neg_input).to(self.device)
            
                pos_output = self.reward[k].forward(pos_input_tensor)
                neg_output = self.reward[k].forward(neg_input_tensor)
                for error_iter in range(len(pos_output)):
                    if pos_output[error_iter] < neg_output[error_iter]:
                        error += 1
                L -= T.log(T.exp(pos_output.sum()) / (T.exp(pos_output.sum()) + T.exp(neg_output.sum())))
                #print(pos_output, neg_output)
                #print("loss", L)
            self.reward_optimizer[k].zero_grad()
            L.backward(retain_graph = True)
            self.reward_optimizer[k].step()
            self.scheduler[i].step()
            batch_loss += L.detach().item()

        return batch_loss, error


    def inference_step(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores, mode):
        '''
        The inference step of moirl agent first computes the posterior distribution of policies given existing conversation trajectory.
        Then the distribution of policies can be used to compute a weighted reward function.
        '''
        encoded_state = T.cat((query_embedding, context_embedding), dim=0)
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, questions_embeddings[i]), dim=0)
        for i in range(self.top_n):
            encoded_state = T.cat((encoded_state, answers_embeddings[i]), dim=0)
        encoded_state = T.cat((encoded_state, questions_scores[:self.top_n]), dim=0)
        encoded_state = T.cat((encoded_state, answers_scores[:self.top_n]), dim=0)
        state = T.tensor(encoded_state, dtype=T.float).to(self.device)
        pp = self.ac.policy(state)
        action = T.argmax(pp).item()
        return state, action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def estimate_reward(self, state, act):
        context = state[:2*self.observation_dim]
        candidate_start = 2 * self.observation_dim if act > 0  else (2 + self.top_n) * self.observation_dim
        candidate_end = candidate_start + self.top_n * self.observation_dim 
        score_start = - self.top_n * (1 + act) - 1 
        score_end = score_start + self.top_n
        candidate = T.cat((state[candidate_start: candidate_end], state[score_start: score_end]))
        state_act = T.cat((context, candidate))

        reward = T.zeros(1).to(self.device)
        for k in range(self.n_rewards):
            prediction = self.reward[k].forward(state_act)
            self.reward_history[k].append(prediction.detach().item())
            mean_est = statistics.mean(self.reward_history[k][-1000:])
            stdv_est = statistics.pstdev(self.reward_history[k][-1000:])
            if stdv_est <= 1e-4:
                stdv_est = 1e-4
            normalized_prediction = (prediction - mean_est)/stdv_est
            reward += normalized_prediction

        reward = reward/self.n_rewards
        return reward  
    
    def train_policy(self, states, acts, old_log_probs, gaes):
        for _ in range(self.max_policy_train_iter):
            self.policy_optim.zero_grad()
            new_logits = self.ac.policy(states)
            new_logits = Categorical(logits=new_logits)
            new_log_probs = new_logits.log_prob(acts)
            policy_ratio = T.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            policy_loss = -T.min(full_loss, clipped_loss).mean()
            policy_loss.backward()
            self.policy_optim.step()

            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.kl_div_val:
                break

    def train_value(self, state, returns):
        for _ in range(self.max_value_train_iter):
            self.value_optim.zero_grad()

            values = self.ac.value(state)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward()
            self.value_optim.step()
        return value_loss.detach().item()
        