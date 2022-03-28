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
warnings.filterwarnings("ignore")

T.set_printoptions(sci_mode=False, edgeitems=5, threshold=1000)

def entropy_e(ts):
    '''
    Assuming ts is 2d tensor
    '''
    ent = 0
    for t in ts:
        for p in t:
            ent -= p * T.log(p)
    return ent

def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = T.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = T.cat([t.view(-1) for t in g])
    
    g[g != g] = 0
    return g

def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = T.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    x[x != x] = 0
    return x

class LinearDeepNetwork(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size = 16):
        super(LinearDeepNetwork, self).__init__()
        self.net = nn.Sequential(OrderedDict({
            'fc_1': nn.Linear(input_dims, hidden_size),
            'act_1': nn.ReLU(),
            'fc_2': nn.Linear(hidden_size, n_actions),
            'act_2': nn.Softmax(),
        }))
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                T.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
    
        self.net.apply(init_weights)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

class LinearRDeepNetwork(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size, dropout_ratio):
        super(LinearRDeepNetwork, self).__init__()
        self.net = nn.Sequential(OrderedDict({
            'fc_1': nn.Linear(input_dims, hidden_size),
            #'norm_1': nn.LayerNorm(hidden_size),
            'act_1': nn.ReLU(),
            'dropout1': nn.Dropout(p=dropout_ratio),
            'fc_2': nn.Linear(hidden_size, n_actions),
            #'norm_2': nn.LayerNorm(n_actions),
            'act_2': nn.Sigmoid()
        }))
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                T.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        def init_zero(m):
            if isinstance(m, nn.Linear):
                T.nn.init.constant_(m.weight, 0)
                m.bias.data.zero_()

        def init_kaiming(m):
            if isinstance(m, nn.Linear):
                T.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        self.net.apply(init_weights)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)

class LinearDeepNetwork_no_activation(nn.Module):
    '''
    The linear deep network used by the agent.
    '''
    def __init__(self, n_actions, input_dims, hidden_size, dropout_ratio):
        super(LinearDeepNetwork_no_activation, self).__init__()
        self.net = nn.Sequential(OrderedDict({
            'fc_1': nn.Linear(input_dims, hidden_size),
            'act_1': nn.ReLU(),
            'dropout1': nn.Dropout(p=dropout_ratio),
            'fc_2': nn.Linear(hidden_size, hidden_size),
            'act_2': nn.ReLU(),
            'dropout2': nn.Dropout(p=dropout_ratio),
            'fc_3': nn.Linear(hidden_size, n_actions),
        }))
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                T.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
    
        self.net.apply(init_weights)

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.net(state)


class ActorCriticNetwork(nn.Module):
  def __init__(self, obs_space_size, action_space_size, dropout_ratio):
    super().__init__()

    self.shared_layers = nn.Sequential(
        nn.Linear(obs_space_size, 64),
        nn.ReLU(),
        nn.Dropout(p=dropout_ratio),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(p=dropout_ratio)
    )
    
    self.policy_layers = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(p=dropout_ratio),
        nn.Linear(64, action_space_size)
        )
    
    self.value_layers = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(p=dropout_ratio),
        nn.Linear(64, 1)
        )

    self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    self.to(self.device)
    
  def value(self, obs):
    z = self.shared_layers(obs)
    value = self.value_layers(z)
    return value
        
  def policy(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    value = self.value_layers(z)
    return policy_logits, value


class GAILAgent():
    '''
    The multi-objective Inverse reinforcement learning Agent for conversational search.
    This agent has multiple policies each represented by one <agent> object.
    '''
    def __init__(self, n_action, observation_dim, top_n, lr, lrdc, weight_decay, max_d_kl, entropy_weight, pmax, disc_weight_clip, policy_weight_clip, gan_name, disc_pretrain_epochs, dropout_ratio):
        self.lr = lr
        self.lrdc = lrdc
        self.weight_decay = weight_decay
        self.n_action = n_action
        self.top_n = top_n
        self.loss = nn.MSELoss()    
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.entropy_weight = entropy_weight*T.ones(1).to(self.device)
        self.observation_dim = observation_dim
        self.disc_weight_clip = disc_weight_clip
        self.policy_weight_clip = policy_weight_clip
        self.policy = LinearDeepNetwork(n_actions = n_action, input_dims = (2+2*self.top_n) * observation_dim + (2) * self.top_n)
        self.gan_name = gan_name
        self.disc_pretrain_epochs = disc_pretrain_epochs
        #self.disc = LinearDeepNetwork_no_activation(n_actions = 1, input_dims = (2+self.top_n) * observation_dim + self.top_n) if gan_name == 'WGAN' else LinearRDeepNetwork(n_actions = 1, input_dims = (2+self.top_n) * observation_dim + self.top_n)
        self.disc = LinearRDeepNetwork(n_actions = 1, input_dims = (2+self.top_n) * observation_dim + self.top_n, hidden_size = 16, dropout_ratio = dropout_ratio)
        self.policyparams = self.policy.parameters()
        self.discparams = self.disc.parameters()
        self.max_d_kl = max_d_kl
        self.disc_optimizer = optim.RMSprop(self.discparams, lr=self.lr, weight_decay = self.weight_decay)
        self.policy_optimizer = optim.RMSprop(self.policyparams, lr=self.lr, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.disc_optimizer, gamma=self.lrdc)
        self.pmax = pmax
        self.expert_traj_history = []
        self.self_traj_history = []
    
    def save(self, path):
        T.save(self.policy.state_dict(), path+'_policy')
        T.save(self.disc.state_dict(), path+'_disc')
    
    def load(self, path):
        self.policy.load_state_dict(T.load(path+'_policy'))
        self.disc.load_state_dict(T.load(path+'_disc'))

    def gail_update(self, all_expert_traj, all_self_traj, disc_train_ratio, epoch):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0
        eps = 1e-4

        # filter out self traj that is best traj
        all_self_traj = [self_traj for self_traj in all_self_traj if self_traj[-1][-1] < 0.3]
        
        self_a_list = T.LongTensor([a for self_traj in all_self_traj for _,a,_ in self_traj]).to(self.device)
        self_s_list = T.stack(([s for self_traj in all_self_traj for s,_,_ in self_traj])).to(self.device)
        self_p_list = T.tensor([1-p for self_traj in all_self_traj for _,_,p in self_traj]).to(self.device)
        predicted_a_list = self.policy.forward(self_s_list).to(self.device)
        predicted_probs = T.max(predicted_a_list, 1).values

        L = T.zeros(1).to(self.device)

        # compute self state_action input for discriminator
        self_disc_s_a_list = []
        for i, row in enumerate(self_s_list):
            disc_context = row[:2*self.observation_dim]
            column_start = 2 * self.observation_dim if int(self_a_list[i]) > 0 else (2 + self.top_n) * self.observation_dim
            column_end = (2 + self.top_n) * self.observation_dim if int(self_a_list[i]) > 0 else (2 + 2 * self.top_n) * self.observation_dim  
            score_start = - self.top_n * (1 + int(self_a_list[i])) - 1 
            disc_candidate = T.cat((row[column_start: column_end], row[score_start: score_start + self.top_n]))
            self_disc_s_a_list.append(T.cat((disc_context, disc_candidate)))
        self_disc_s_a = T.stack(self_disc_s_a_list).to(self.device)

        # compute expert state_action input for discriminator
        expert_s_list = T.stack(([s for expert_traj in all_expert_traj for s,_ in expert_traj])).to(self.device)
        expert_a_list = T.LongTensor([a for expert_traj in all_expert_traj for _,a in expert_traj]).to(self.device)
        expert_disc_s_a_list = []
        for i, row in enumerate(expert_s_list):
            disc_context = row[:2*self.observation_dim]
            column_start = 2 * self.observation_dim if int(expert_a_list[i]) > 0  else (2 + self.top_n) * self.observation_dim
            column_end = (2 + self.top_n) * self.observation_dim if int(expert_a_list[i]) > 0  else (2 + 2 * self.top_n) * self.observation_dim
            score_start = - self.top_n * (1 + int(expert_a_list[i])) - 1 
            disc_candidate = T.cat((row[column_start: column_end], row[score_start: score_start + self.top_n]))
            expert_disc_s_a_list.append(T.cat((disc_context, disc_candidate)))
        expert_disc_s_a = T.stack(expert_disc_s_a_list).to(self.device)


        if self.gan_name == 'WGAN':
            disc_train_ratio = 5

        for di in range(disc_train_ratio):
            disc_self_p = self.disc.forward(self_disc_s_a).to(self.device)
            disc_expert_p = self.disc.forward(expert_disc_s_a).to(self.device)
            # update discriminator
            if self.gan_name == 'GAN':
                L_disc = - T.log(disc_expert_p).mean() - T.log(1-disc_self_p).mean() # log discriminator loss
            elif self.gan_name == 'LSGAN':
                L_disc = self.loss(disc_expert_p, T.tensor([1.0]*len(disc_expert_p)).to(self.device)) + self.loss(disc_self_p, T.tensor([0.0]*len(disc_self_p)).to(self.device)) # mse discriminator loss
            elif self.gan_name == 'WGAN':
                L_disc = -( disc_expert_p.mean() - disc_self_p.mean() )
            self.disc_optimizer.zero_grad()
            L_disc.backward(retain_graph = True)
            self.disc_optimizer.step()
            if self.gan_name == 'WGAN':
                for p in self.disc.parameters():
                    p.data.clamp_(-self.disc_weight_clip, self.disc_weight_clip)
            
        if epoch < self.disc_pretrain_epochs:
            return L_disc.detach().item(), -1000 * epoch

        # update policy
        L_pol = T.zeros(1).to(self.device)
        for k in range(len(all_self_traj)):
            expert_a = [a for _,a in all_expert_traj[k]]
            conv_a_list = T.LongTensor([a for _,a,_ in all_self_traj[k]]).to(self.device)
            conv_s_list = T.stack(([s for s,_,_ in all_self_traj[k]])).to(self.device)
            conv_disc_s_a_list = []
            conv_disc_s_a_list_2 = []
            for ck, row in enumerate(conv_s_list):
                disc_context = row[:2*self.observation_dim]
                column_start = 2 * self.observation_dim if int(conv_a_list[ck]) > 0 else (2 + self.top_n) * self.observation_dim
                column_end = (2 + self.top_n) * self.observation_dim if int(conv_a_list[ck]) > 0 else (2 + 2 * self.top_n) * self.observation_dim
                score_start = - self.top_n * (1 + int(conv_a_list[ck])) - 1 
                disc_candidate = T.cat((row[column_start: column_end], row[score_start: score_start + self.top_n]))
                conv_disc_s_a_list.append(T.cat((disc_context, disc_candidate)))

                column_start_2 = (2 + self.top_n) * self.observation_dim if int(conv_a_list[ck]) > 0 else 2 * self.observation_dim
                column_end_2 = (2 + 2 * self.top_n) * self.observation_dim if int(conv_a_list[ck]) > 0 else (2 + self.top_n) * self.observation_dim
                score_start_2 = - self.top_n * (2 - int(conv_a_list[ck])) - 1 
                disc_candidate_2 = T.cat((row[column_start_2: column_end_2], row[score_start_2: score_start_2 + self.top_n]))
                conv_disc_s_a_list_2.append(T.cat((disc_context, disc_candidate_2)))
            conv_disc_s_a = T.stack(conv_disc_s_a_list).to(self.device)
            conv_disc_s_a_2 = T.stack(conv_disc_s_a_list_2).to(self.device)

            for j in range(len(all_self_traj[k])):
                distrib = self.policy.forward(all_self_traj[k][j][0]).to(self.device)

                if conv_a_list[j] > 0:
                    Qs = self.disc.forward(conv_disc_s_a[j:])
                    Q = Qs.mean()
                    Qs_2 = self.disc.forward(conv_disc_s_a_2[j])
                    Q_2 = Qs_2.mean()
                else:
                    Qs = self.disc.forward(conv_disc_s_a[j])
                    Q = Qs.mean()
                    Qs_2 = self.disc.forward(conv_disc_s_a_2[j:])
                    Q_2 = Qs_2.mean()
                    
                L_pol -= T.log(distrib[conv_a_list[j]])* Q + T.log(distrib[1-conv_a_list[j]])*Q_2 + self.entropy_weight * entropy_e([distrib]).to(self.device)
                #L_pol -= T.log(p_s_a)*Q + T.log(p_s_a_2)*Q_2 + self.entropy_weight * entropy_e([distrib]).to(self.device)
                #L_pol -= T.log(p_s_a)*Q  + self.entropy_weight * entropy_e([distrib]).to(self.device)

            
        # l1 penalty or try l2.
        l1 = 0
        for p in self.policy.parameters():
            l1 += p.abs().sum()
            
        L_pol = L_pol + self.weight_decay * l1

        self.policy_optimizer.zero_grad()
        L_pol.backward()
        self.policy_optimizer.step()
        
          
        for p in self.policy.parameters():
            p.data.clamp_(-self.policy_weight_clip, self.policy_weight_clip)
        
        batch_loss += L_pol.detach().item()

        return L_disc.detach().item(), batch_loss


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
        pp = self.policy.forward(state)
        action = T.argmax(pp).item()
        return state, action
    
    def sample_step(self, query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores):
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
        pp = self.policy.forward(state)
        if random.random() > pp[0]:
            action = 1
        else:
            action = 0
        return state, action


    def ecrr_update(self, all_expert_traj, all_self_traj):
        '''
        Take a TRPO step to minimize E_i[log\pi(a|s)*p]-\lambda H(\pi)
        '''
        # update policy
        batch_loss = 0
        self.policy_optimizer.zero_grad()

        L = T.zeros(1).to(self.device)
        for k in range(len(all_self_traj)):
            expert_a = [a for _,a in all_expert_traj[k]]
            for j in range(len(all_self_traj[k])):
                if j < len(expert_a):
                    distrib = self.policy.forward(all_self_traj[k][j][0]).to(self.device)
                    L -= T.log(distrib[expert_a[j]])*(1-all_self_traj[k][j][2]) + self.entropy_weight * entropy_e([distrib]).to(self.device)

        L.backward()
        self.policy_optimizer.step()
        self.scheduler.step()
        batch_loss += L.detach().item()

        return 0, batch_loss

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
        value_lr = 1e-5,
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
  
        self.ac = ActorCriticNetwork(obs_space_size = (2+2*self.top_n) * observation_dim + (2) * self.top_n, action_space_size = self.n_action, dropout_ratio = dropout_ratio)
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
        
