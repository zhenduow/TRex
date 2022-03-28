from user import User
from dataset import ConversationDataset
from agent import Agent, BaseAgent
from ilagent import TRexAgent, GAILAgent
import logging
import numpy as np
import random
import os
import time
import torch as T
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel
from parlai.scripts.interactive import Interactive, rerank
from copy import deepcopy
import argparse
import gc
import matplotlib.pyplot as plt

def generate_embedding_no_grad(text, tokenizer, embedding_model, device):
    '''
    Generate embedding using torch transformer.
    '''
    with T.no_grad():
        tokenized_context_ = T.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
        context_embedding_ = T.squeeze(embedding_model(tokenized_context_)[0])[0].detach().cpu()
        del tokenized_context_
        T.cuda.empty_cache()
        gc.collect()
        return context_embedding_

def read_from_memory(query, context, memory):
    '''
    Read query, context, question, answer ranks and ranking scores from memory
    '''
    return memory[query]['embedding'], memory[query][context]['embedding'],\
        memory[query][context]['questions'], memory[query][context]['answers'],\
        memory[query][context]['questions_embeddings'],memory[query][context]['answers_embeddings'],\
        memory[query][context]['questions_scores'], memory[query][context]['answers_scores']

def save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device):
    '''
    Save query, context, question, answer ranks and ranking scores to memory for running speed.
    '''
    if query not in memory.keys():
        memory[query] = {}
        with T.no_grad():
            tokenized_query = T.tensor([tokenizer.encode(query, add_special_tokens=True)]).to(device)
            memory[query]['embedding'] = T.squeeze(embedding_model(tokenized_query)[0])[0].detach().cpu()
            T.cuda.empty_cache()
    memory[query][context] = {}
    with T.no_grad():
        memory[query][context]['embedding'] = T.squeeze(embedding_model(T.tensor([tokenizer.encode(context, add_special_tokens=True)]).to(device))[0])[0].detach().cpu()
        memory[query][context]['questions_embeddings'] = [T.squeeze(embedding_model(T.tensor([tokenizer.encode(questions[i], add_special_tokens=True)]).to(device))[0])[0].detach().cpu() for i in range(10)] # hard coding max tolerance to save memory
        memory[query][context]['answers_embeddings'] = [T.squeeze(embedding_model(T.tensor([tokenizer.encode(answers[i], add_special_tokens=True)]).to(device))[0])[0].detach().cpu() for i in range(10)]
        memory[query][context]['questions'] = questions
        memory[query][context]['answers'] = answers
        memory[query][context]['questions_scores'] = T.tensor(questions_scores).detach().cpu()
        memory[query][context]['answers_scores'] = T.tensor(answers_scores).detach().cpu()
        T.cuda.empty_cache()
    gc.collect()
    return memory

def generate_batch_question_candidates(batch, conversation_id, ignore_questions, total_candidates):
    '''
    Generate positive and negative clarifying question answers.
    '''
    positives = [batch['conversations'][conversation_id][turn_id] for turn_id in range(len(batch['conversations'][conversation_id])) if turn_id % 2 == 1 and turn_id != len(batch['conversations'][conversation_id])-1]
    filtered_positives = [cand for cand in positives if cand not in ignore_questions]
    negatives = [response for response in batch['responses_pool'] if response not in positives][:total_candidates - len(filtered_positives)]
    return filtered_positives + negatives

def generate_batch_answer_candidates(batch, conversation_id, total_candidates):
    '''
    Generate positive and negative answer candidates.
    '''
    positives = [batch['conversations'][conversation_id][-1]]
    negatives = [answer for answer in batch['answers_pool'] if answer not in positives][:total_candidates - len(positives)] 
    return positives + negatives

def create_rerankers(dataset_name, reranker_name):
    '''
    Create rerankers.
    '''
    if dataset_name == 'MSDialog':
        reranker_prefix = ''
    elif dataset_name == 'UDC' or 'UDC-ActionParsnip' or 'UDC-2012':
        reranker_prefix = 'udc'
    elif dataset_name == 'Opendialkg':
        reranker_prefix = 'open'
    
    question_reranker = Interactive.main(model = 'transformer/' + reranker_name + 'encoder', 
                    model_file = 'zoo:pretrained_transformers/model_'+ reranker_name + '/' + reranker_prefix + 'question',  
                    encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                    return_cand_scores = True)

    answer_reranker = Interactive.main(model = 'transformer/' + reranker_name + 'encoder', 
                    model_file = 'zoo:pretrained_transformers/model_' + reranker_name + '/' + reranker_prefix + 'answer',  
                    encode_candidate_vecs = False,  eval_candidates = 'inline', interactive_candidates = 'inline',
                    return_cand_scores = True)

    return question_reranker, answer_reranker

def initialize_dirs(dataset_name, reranker_name, cv):
    '''
    Create folders for saving cache, checkpoints, and logs.
    '''
    if not os.path.exists(dataset_name + '_experiments/'):
        os.makedirs(dataset_name + '_experiments/')
    if not os.path.exists(dataset_name + '_experiments/checkpoints/'):
        os.makedirs(dataset_name + '_experiments/checkpoints/')
    if not os.path.exists(dataset_name + '_experiments/embedding_cache/'):
        os.makedirs(dataset_name + '_experiments/embedding_cache/')
    if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name ):
        os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name )
    if cv != '':
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv)
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv + '/train')
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv + '/val')
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/' + cv + '/test')
    else:
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/train' ):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/train')
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/val' ):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/val' )
        if not os.path.exists(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/test' ):
            os.makedirs(dataset_name + '_experiments/embedding_cache/' + reranker_name + '/test' )

def find_best_trajectory(answer_traj, question_traj, p):
    '''
    Find the best conversation trajectory given all the answer rank and question rank.
    answer_traj is a list of ((state, action), reward) tuples.
    question_traj is a list of ((state, action), correct question rank) tuples.
    '''
    best_ecrr, best_step = 0, 0
    cumulative_ecrr = 1
    assert len(answer_traj) == len(question_traj)
    for step in range(len(answer_traj)):
        if cumulative_ecrr * answer_traj[step][1] > best_ecrr:
            best_ecrr = cumulative_ecrr * answer_traj[step][1]
            best_step = step
        cumulative_ecrr *= p ** question_traj[step][1]
    return best_step, best_ecrr

def generate_trajectory_ranked_pairs(answer_traj, question_traj, p):
    '''
    generate pairs of ranked trajectories for T-Rex training
    answer_traj is a list of ((state, reward) tuples.
    question_traj is a list of ((state, correct question rank) tuples.
    Return a list of trajectory pairs.
    '''
    traj_list = []
    traj_pair_list = []
    cumulative_ecrr = 1
    assert len(answer_traj) == len(question_traj)
    for step in range(len(answer_traj)):
        step_ecrr = cumulative_ecrr * answer_traj[step][1]
        traj_list.append(([(a_traj[0], 1) if i < step else (a_traj[0], 0) for i, a_traj in enumerate(answer_traj[:step+1]) ], step_ecrr))
        cumulative_ecrr *= p ** question_traj[step][1]
    for step1 in range(len(traj_list)):
        for step2 in range(step1, len(traj_list)):
            if traj_list[step1][1] > traj_list[step2][1]:
                traj_pair_list.append((traj_list[step1][0], traj_list[step2][0]))
            elif traj_list[step2][1] > traj_list[step1][1]:
                traj_pair_list.append((traj_list[step2][0], traj_list[step1][0]))
    
    return traj_pair_list

def compute_trajectory_ecrr(traj, continue_p):
    p = 1
    for s, a, r in traj:
        if a == 0:
            p *= r
            return p
        else:
            p *= continue_p ** r
    p *= 0 # if no answer is returned.
    return p

def compute_self_trajectory_p(self_traj, a_traj, q_traj, p, best_ecrr):
    '''
    Compute the normalized trajectory ecrr.
    '''
    self_traj_w_p = []
    ecrr = 1
    for i, (s,a,r) in enumerate(self_traj):
        if a == 0:
            ecrr *= r
            self_traj_w_p.append((s, a, ecrr/(best_ecrr+1e-6)))
        elif a == 1:
            ecrr *= p ** r
            _, best_future_ecrr = find_best_trajectory(a_traj[i+1:], q_traj[i+1:], p)
            self_traj_w_p.append((s, a, ecrr*best_future_ecrr/(best_ecrr+1e-6)))

    return self_traj_w_p

def find_least_effort(answer_traj, question_traj):
    '''
    Find the conversation trajectory with the least user effort given all the answer rank and question rank.
    answer_traj is a list of (state, action, reward) tuples.
    question_traj is a list of (state, action, correct question rank) tuples.
    '''
    least_effort, best_step = 0, 0
    cumulative_effort = 0
    assert len(answer_traj) == len(question_traj)
    for step in range(len(answer_traj)):
        if cumulative_effort + int(1/answer_traj[step][1]) < least_effort:
            least_effort = cumulative_effort + int(1/answer_traj[step][1])
            best_step = step
        cumulative_effort += (question_traj[step][1] + 1)
    
    return best_step, least_effort

def compute_trajectory_effort(traj):
    eff = 0
    for s, a, r in traj:
        if a == 0:
            p += int(1/r)
            return p
        else:
            p += (r+1)
    p += 10000 # if no answer is returned.
    return p

def compute_self_trajectory_normalized_effort(self_traj, a_traj, q_traj, least_effort):
    '''
    Compute the normalized trajectory effort.
    '''
    self_traj_w_p = []
    effort = 0
    for i, (s,a,r) in enumerate(self_traj):
        if a == 0:
            effort += int(1/r)
            self_traj_w_p.append((s, a, least_effort/effort))
        elif a == 1:
            effort += (r+1)
            _, least_future_effort = find_least_effort(a_traj[i+1:], q_traj[i+1:])
            self_traj_w_p.append((s, a, least_effort/(effort+least_future_effort)))

    return self_traj_w_p

def discount_rewards(rewards, gamma=0.8):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.8, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])
    return np.array(gaes[::-1])

def main(args):
    logging.getLogger().setLevel(logging.INFO)
    random.seed(2020)
    device = "cuda:0" if T.cuda.is_available() else "cpu"

    # initialize log directories
    initialize_dirs(args.dataset_name, args.reranker_name, args.cv)
    print(args, flush = True)

    output = open(args.dataset_name+'_experiments/'+ 'cv' + args.cv + '_' + args.gan_name + '_r' + str(args.cq_reward) +'_cas'+str(args.cascade_p)+'_entropyw'+str(args.entropy_weight)+'_discratio'+str(args.disc_train_ratio), 'w')
    #train_output = open(args.dataset_name+'_experiments/'+args.reranker_name + '_r' + str(args.cq_reward) +'_cas'+str(args.cascade_p) +'_max_dkl'+str(args.max_d_kl)+'_entropyw'+str(args.entropy_weight)+'_discratio'+str(args.disc_train_ratio)+ '_train', 'w')
    #val_output = open(args.dataset_name+'_experiments/'+args.reranker_name + '_r' + str(args.cq_reward)+'_cas'+str(args.cascade_p) +'_max_dkl'+str(args.max_d_kl)+'_entropyw'+str(args.entropy_weight)+'_discratio'+str(args.disc_train_ratio)+ '_val', 'w')
    #test_output = open(args.dataset_name+'_experiments/'+args.reranker_name + '_r' + str(args.cq_reward)+'_cas'+str(args.cascade_p)+'_max_dkl'+str(args.max_d_kl)+'_entropyw'+str(args.entropy_weight)+'_discratio'+str(args.disc_train_ratio) + '_test', 'w')


    # load data
    train_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/train' + args.cv + '/', args.batch_size, args.max_data_size)
    val_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/val' + args.cv + '/', args.batch_size, args.max_data_size)
    test_dataset = ConversationDataset('data/' + args.dataset_name + '-Complete/test' + args.cv + '/', args.batch_size, args.max_data_size)
    data_size = sum([len(b['conversations'].keys()) for b in train_dataset.batches]) 
    test_size = sum([len(b['conversations'].keys()) for b in test_dataset.batches]) 

    # load rerankers
    question_reranker, answer_reranker = create_rerankers(args.dataset_name, args.reranker_name)

    # initialize agents
    agent = Agent(lr = 1e-4, input_dims = (3 + args.user_tolerance) * args.observation_dim + 1 + args.user_tolerance, top_k = args.user_tolerance, n_actions=args.n_action, gamma = 1 - args.cq_reward, weight_decay = 0.01, max_experience_length = 5 * data_size) # query, context, answer, and topn questions embedding + 1 answer score and topn question score
    base_agent = BaseAgent(lr = 1e-4, input_dims = 2 * args.observation_dim, n_actions = args.n_action, weight_decay = args.weight_decay)
    ilagent = TRexAgent(n_action = args.n_action, observation_dim = args.observation_dim, top_n = args.il_topn, lr= args.lr, lrdc=args.lrdc, gamma = 1 - args.cq_reward, weight_decay= args.weight_decay, dropout_ratio = args.dropout_ratio, n_rewards = args.trex_n_rewards, traj_sample_length = args.trex_traj_sample_length)
    
    # initialize embedding model
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
    embedding_model = AutoModel.from_pretrained('xlnet-base-cased').to(device)

    if args.load_checkpoint == True:
        #agent.load(args.dataset_name + '_experiments/checkpoints/' + args.checkpoint)
        #base_agent.load(args.dataset_name + '_experiments/checkpoints/' + args.checkpoint_base)
        ilagent.load(args.dataset_name + '_experiments/checkpoints/' + args.checkpoint_il)
    
    X = []
    train_il_ecrr_hist, train_il_loss_hist = [],[]
    val_il_mrr_hist, val_il_ecrr_hist = [],[]
    test_il_mrr_hist, test_il_ecrr_hist = [],[]
    
    print('training')
    for i in range(args.max_iter):
        train_scores, train_q0_scores, train_q1_scores, train_q2_scores, train_oracle_scores, train_base_scores, train_il_scores  = [],[],[],[],[],[],[]
        train_ecrr, train_q0_ecrr, train_q1_ecrr, train_q2_ecrr, train_oracle_ecrr, train_base_ecrr, train_il_ecrr = [],[],[],[],[],[],[]
        il_loss = 0
        il_error = 0
        disc_loss = 0
        value_loss = 0
        for batch_serial, batch in enumerate(train_dataset.batches):
            ''' GAIL variables
            all_expert_traj = [] # all conversation trajectories used for il
            all_self_traj = []
            '''
            all_traj_pairs = [] # T-Rex training data

            # load state information from cached memory
            if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/train/memory.batchsave' + str(batch_serial)):
                with T.no_grad():
                    memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/train/memory.batchsave' + str(batch_serial))
            else:
                memory = {}

            update_memory = False

            train_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = args.cq_reward, cq_penalty = args.cq_reward - 1)
            for conv_serial, train_id in enumerate(train_ids):
                query = user.initialize_state(train_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                q1_score, q2_score = 100, 100
                a_traj, q_traj, il_traj = [], [], []
                stop, base_stop, il_stop = False, False, False
                ecrr, base_ecrr, il_ecrr = 1, 1, 1
                correct_question_rank = 0
                rollout_data = [[],[],[],[],[]] # state, act, reward, value, act_log_probs
                output.write('-------- train batch %.0f conversation %.0f/%.0f --------\n' % (batch_serial, args.batch_size*(batch_serial) + conv_serial + 1, data_size))
                rollout_done = False
                #while not q_done:
                while n_round < len(batch['conversations'][train_id]) / 2 and correct_question_rank < args.reranker_return_length:
                    output.write('-------- round %.0f --------\n' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, train_id, args.batch_size)
                            # get reranker results   
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                            update_memory = True
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, args.batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, train_id, args.batch_size)
                        # get reranker results   
                        questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                        answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                        update_memory = True
                    
                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    if i < args.risk_run_epoch:
                        _, action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    else:
                        action = 0
                    if i < args.base_run_epoch:
                        base_action = base_agent.choose_action(query_embedding, context_embedding)
                    else:
                        base_action = 0
                    state, il_action = ilagent.inference_step(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores, mode = 'train')
                    # use predefined rewards
                    context_, answer_reward, question_reward, correct_question_rank, q_done_this_round, good_question, patience_this_round = user.update_state(train_id, context, questions, answers, use_top_k = max(args.user_tolerance - patience_used, 1))
                    patience_used = max(patience_used + patience_this_round, args.user_tolerance)
                    q_done = q_done or q_done_this_round
                    output.write('act '+str(action)+' base act '+str(base_action)+' il act '+str(il_action)+' a reward '+str(answer_reward)+' q reward '+str(question_reward) +' cq rank '+str(correct_question_rank) +'\n')
                    #train_output.write(str(i)+' '+str(train_id)+' '+str(n_round)+' '+str(action)+' '+str(base_action)+' '+str(il_action)+' '+str(answer_reward)+' '+str(correct_question_rank) +'\n')
                    a_traj.append((state, answer_reward))
                    q_traj.append((state, correct_question_rank))
                    il_traj.append((state, il_action,[answer_reward, correct_question_rank][il_action]))

                    if n_round >= args.user_patience:
                        q_done = True

                    # rollout the conversation 
                    if not rollout_done:
                        logits, value = ilagent.ac(state)
                        act_distribution = Categorical(logits=logits)
                        act = act_distribution.sample()
                        act_log_prob = act_distribution.log_prob(act).item()
                        reward = ilagent.estimate_reward(state, act)
                        for rollout_id, item in enumerate((state, act, reward, value, act_log_prob)):
                            rollout_data[rollout_id].append(item)

                        if q_done or act == 0:
                            rollout_done = True

                    if n_round < len(batch['conversations'][train_id]) / 2 and correct_question_rank < args.reranker_return_length:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, train_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, train_id, args.batch_size)
                            # get reranker results
                            questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                            answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model, device)
                            update_memory = True
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)

                    else:
                        context_embedding_ = generate_embedding_no_grad(context_, tokenizer, embedding_model, device)
                        questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = None, None, None, None, None, None

                    if i < args.risk_run_epoch:
                        agent.joint_learn((query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores),\
                            answer_reward, question_reward,\
                            (query_embedding, context_embedding_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_))
                    if i < args.base_run_epoch:
                        base_agent.learn(query_embedding, context_embedding, 0 if (n_round + 1) == len(user.dataset[train_id])/2 else 1)

                    # trex training
                    #ilagent.PPO_learn((query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores),\
                    #        (query_embedding, context_embedding_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_))

                    # non-deterministic methods evaluation
                    if not stop:
                        if (action == 0 or (action == 1 and question_reward == args.cq_reward - 1)):
                            stop = True 
                            train_scores.append(answer_reward if action == 0 else 0)
                    
                    if not base_stop:
                        if (base_action == 0 or (base_action == 1 and question_reward == args.cq_reward - 1)):
                            base_stop = True
                            train_base_scores.append(answer_reward if base_action == 0 else 0)

                    
                    if not il_stop:
                        if (il_action == 0 or (il_action == 1 and question_reward == args.cq_reward - 1)):
                            il_stop = True 
                            train_il_scores.append(answer_reward if il_action == 0 else 0)
                    
                    # deterministic methods evaluation and store optimal trajectory 
                    if n_round == 0:
                        train_q0_scores.append(answer_reward)
                        train_q0_ecrr.append(answer_reward)
                        train_q1_ecrr.append(args.cascade_p**correct_question_rank)
                        train_q2_ecrr.append(args.cascade_p**correct_question_rank)
                        if q_done:
                            q1_score, q2_score = 0, 0
                            if not correct_question_rank < args.reranker_return_length:
                                train_q1_scores.append(0)
                                train_q2_scores.append(0)
                    elif n_round == 1:
                        train_q1_scores.append(min(answer_reward, q1_score))
                        train_q1_ecrr[-1] *= answer_reward
                        train_q2_ecrr[-1] *= args.cascade_p**correct_question_rank
                        if q_done:
                            q2_score = 0
                            if not correct_question_rank < args.reranker_return_length:
                                train_q2_scores.append(0)
                    elif n_round == 2:
                        train_q2_scores.append(min(answer_reward, q2_score))
                        train_q2_ecrr[-1] *= answer_reward

                    # ecrr evaluation
                    if 'ecrr' in locals():
                        if action == 1:
                            ecrr *= args.cascade_p**correct_question_rank
                            if  correct_question_rank >= args.reranker_return_length:
                                ecrr *= answer_reward
                                train_ecrr.append(ecrr)
                                del ecrr
                        else:
                            ecrr *= answer_reward
                            train_ecrr.append(ecrr)
                            del ecrr
                    
                    if 'base_ecrr' in locals():
                        if base_action == 1:
                            base_ecrr *= args.cascade_p**correct_question_rank
                            if  correct_question_rank >=  args.reranker_return_length:
                                base_ecrr *= answer_reward
                                train_base_ecrr.append(base_ecrr)
                                del base_ecrr
                        else:
                            base_ecrr *= answer_reward
                            train_base_ecrr.append(base_ecrr)
                            del base_ecrr
                    
                    if 'il_ecrr' in locals():
                        if il_action == 1:
                            il_ecrr *= args.cascade_p**correct_question_rank
                            if  correct_question_rank >= args.reranker_return_length:
                                il_ecrr *= answer_reward
                                train_il_ecrr.append(il_ecrr)
                                del il_ecrr
                        else:
                            il_ecrr *= answer_reward
                            train_il_ecrr.append(il_ecrr)
                            del il_ecrr

                    context = context_
                    n_round += 1
                    
                # simulation ends
                # find the optimal trajectory using ecrr
                best_answer_step, best_ecrr = find_best_trajectory(a_traj, q_traj, args.cascade_p)
                best_answer_reward = a_traj[best_answer_step][1]
                train_oracle_scores.append(best_answer_reward)
                train_oracle_ecrr.append(best_ecrr)
                
                # T-Rex training
                conv_traj_pairs = generate_trajectory_ranked_pairs(a_traj, q_traj, args.cascade_p)
                if len(all_traj_pairs) == 0:
                    all_traj_pairs = conv_traj_pairs
                else:
                    all_traj_pairs.extend(conv_traj_pairs)

                print("------------new conversation-------------")
                # TRex PPO training   
                
                states = T.stack(rollout_data[0]).to(device)
                acts = T.tensor(rollout_data[1], dtype=T.int32, device=device)
                rewards = rollout_data[2]
                values = rollout_data[3]
                old_log_probs = T.tensor(rollout_data[4], dtype=T.float32, device=device)
                returns = discount_rewards(rewards)
                returns = T.tensor(returns, dtype=T.float32, device=device)
                gaes = calculate_gaes(rewards, values)
                gaes = T.tensor(gaes.astype('float32'), dtype = T.float32, device=device)
                   
                print("-----------------")
                print("state", states)
                print("act", acts)
                print("rewards", rewards)
                print("values", values)
                print("returns", returns)
                print("gaes", gaes)
                print("-----------------")
                ilagent.train_policy(
                        states,
                        acts,
                        old_log_probs,
                        gaes
                        )
                value_loss += ilagent.train_value(
                        states,
                        returns
                        )
                    
                ''' GAIL training content
                expert_traj = []
                for step in range(best_answer_step):
                    expert_traj.append((q_traj[step][0], 1))
                expert_traj.append((a_traj[best_answer_step][0], 0))
                all_expert_traj.append(expert_traj)
                        
                self_traj = []
                # create self trajectories
                for step in range(len(il_traj)):
                    self_traj.append((il_traj[step]))
                    if int(il_traj[step][1]) == 0:
                        break
                # add positive and self trajectory to batch
                self_traj_w_p = compute_self_trajectory_p(self_traj, a_traj, q_traj, args.cascade_p, best_ecrr)
                all_self_traj.append(self_traj_w_p)
                '''
                

            # save memory per batch and clear some cache
            if update_memory:
                T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/train/memory.batchsave' + str(batch_serial))
            del memory
            del user
            
            #d_loss, batch_loss = ilagent.gail_update(all_expert_traj, all_self_traj, disc_train_ratio = args.disc_train_ratio, epoch = i)
            #d_loss, batch_loss = ilagent.ecrr_update(all_expert_traj, all_self_traj)
            #il_loss += batch_loss
            #disc_loss += d_loss

            batch_loss, error = ilagent.trex_reward_update(all_traj_pairs)
            il_loss += batch_loss
            il_error += error
            
            T.cuda.empty_cache()

        output.write("Train epoch " + str(i)+'\n')
        output.write("risk\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in train_scores]), np.mean(train_scores), np.mean(train_ecrr)))
        output.write("q0\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in train_q0_scores]), np.mean(train_q0_scores), np.mean(train_q0_ecrr)))
        output.write("q1\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in train_q1_scores]), np.mean(train_q1_scores), np.mean(train_q1_ecrr)))
        output.write("q2\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in train_q2_scores]), np.mean(train_q2_scores), np.mean(train_q2_ecrr)))
        output.write("base\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in train_base_scores]), np.mean(train_base_scores), np.mean(train_base_ecrr)))
        output.write("il\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in train_il_scores]), np.mean(train_il_scores), np.mean(train_il_ecrr)))
        output.write("oracle\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in train_oracle_scores]), np.mean(train_oracle_scores), np.mean(train_oracle_ecrr)))
        

        output.write("avg loss " + str(np.mean(agent.loss_history)) + '\n')
        output.write("il loss " + str(il_loss) + " value loss " + str(value_loss) + " il error " + str(il_error) + '\n')
        #output.write("disc loss " + str(disc_loss) + '\n')

        # save checkpoint
        #agent.save(args.dataset_name + '_experiments/checkpoints/' + str(np.mean(agent.loss_history)))
        #base_agent.save(args.dataset_name + '_experiments/checkpoints/' + 'base')
        ilagent.save(args.dataset_name + '_experiments/checkpoints/' + 'il_' + str(il_loss))
        
        
        # plotting
        X.append(i+1)
        train_il_ecrr_hist.append(np.mean(train_il_ecrr))
        train_il_loss_hist.append(il_loss)
        if i > 30:
            plt.figure()
            plt.plot(X[31:], train_il_ecrr_hist[31:], label="train_ecrr")
            #plt.plot(X[31:], train_il_loss_hist[31:], label="train_loss")
            plt.legend()
            plt.savefig('fig_'+args.dataset_name +'_cv'+ str(args.cv)+'_top'+str(args.user_tolerance)+'_lr'+str(args.lr)+'_r'+str(args.cq_reward)+'_caps'+str(args.cascade_p)+'_train.png')
            

        ## val
        val_scores, val_q0_scores, val_q1_scores, val_q2_scores, val_oracle_scores, val_base_scores, val_il_scores = [],[],[],[],[],[],[]
        val_ecrr, val_q0_ecrr, val_q1_ecrr, val_q2_ecrr, val_oracle_ecrr, val_base_ecrr, val_il_ecrr = [],[],[],[],[],[],[]
        agent.epsilon = 0
        
        for batch_serial, batch in enumerate(val_dataset.batches):
            if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/val/memory.batchsave' + str(batch_serial)):
                with T.no_grad():
                    memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/val/memory.batchsave' + str(batch_serial))
            else:
                memory = {}
            update_memory = False

            val_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = args.cq_reward, cq_penalty = args.cq_reward - 1)
            for conv_serial, val_id in enumerate(val_ids):   
                query = user.initialize_state(val_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                q1_score, q2_score = 100, 100
                stop, base_stop, il_stop = False,False,False
                a_traj, q_traj, il_traj = [], [], []
                ecrr, base_ecrr, il_ecrr = 1, 1, 1
                correct_question_rank = 0
                output.write('-------- val batch %.0f conversation %.0f/%.0f --------\n' % (batch_serial, args.batch_size*(batch_serial) + conv_serial + 1, test_size))
                # while not q_done:
                while n_round < len(batch['conversations'][val_id]) / 2 and correct_question_rank < args.reranker_return_length:
                    output.write('-------- round %.0f --------\n' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, val_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, val_id, args.batch_size)
                            # get reranker results   
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                            update_memory = True
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, val_id, ignore_questions, args.batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, val_id, args.batch_size)

                        # get reranker results
                        questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                        answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                        update_memory = True
                    
                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    if i < args.risk_run_epoch:
                        state, action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    else:
                        action = 0
                    if i < args.base_run_epoch:
                        base_action = base_agent.choose_action(query_embedding, context_embedding)
                    else:
                        base_action = 0
                    # convil
                    state, il_action = ilagent.inference_step(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores, mode = 'val')

                    context_, answer_reward, question_reward, correct_question_rank, q_done_this_round, good_question, patience_this_round = user.update_state(val_id, context, questions, answers, use_top_k = max(args.user_tolerance - patience_used, 1))
                    q_done = q_done or q_done_this_round
                    patience_used = max(patience_used + patience_this_round, args.user_tolerance)
                    output.write('act '+str(action)+' base act '+str(base_action)+' il act '+str(il_action)+' a reward '+str(answer_reward)+' q reward '+str(question_reward) +' cq rank '+str(correct_question_rank) +'\n')
                    #val_output.write(str(i)+' '+str(val_id)+' '+str(n_round)+' '+str(action)+' '+str(base_action)+' '+str(il_action)+' '+str(answer_reward)+' '+str(correct_question_rank) +'\n')

                    a_traj.append((state, answer_reward))
                    q_traj.append((state, correct_question_rank))
                    il_traj.append((state, il_action,[answer_reward, correct_question_rank][il_action]))

                    if n_round >= args.user_patience:
                        q_done = True

                    if n_round < len(batch['conversations'][val_id]) / 2 and correct_question_rank < args.reranker_return_length:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling    
                            question_candidates = generate_batch_question_candidates(batch, val_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, val_id, args.batch_size)
                            # get reranker results
                            questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                            answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model, device)
                            update_memory = True
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)

                    # non-deterministic models evaluation
                    if not stop:
                        if (action == 0 or (action == 1 and question_reward == args.cq_reward - 1)):
                            stop = True
                            val_scores.append(answer_reward if action == 0 else 0)

                    if not base_stop:
                        if (base_action == 0 or (base_action == 1 and question_reward == args.cq_reward - 1)):
                            base_stop = True
                            val_base_scores.append(answer_reward if base_action == 0 else 0)

                    if not il_stop:
                        if (il_action == 0 or (il_action == 1 and question_reward == args.cq_reward - 1)):
                            il_stop = True
                            val_il_scores.append(answer_reward if il_action == 0 else 0)

                    # baseline models evaluations
                    if n_round == 0:
                        val_q0_scores.append(answer_reward)
                        val_q0_ecrr.append(answer_reward)
                        val_q1_ecrr.append(args.cascade_p**correct_question_rank)
                        val_q2_ecrr.append(args.cascade_p**correct_question_rank)
                        if q_done:
                            q1_score, q2_score = 0, 0
                            if not correct_question_rank < args.reranker_return_length:
                                val_q1_scores.append(0)
                                val_q2_scores.append(0)
                    elif n_round == 1:
                        val_q1_scores.append(min(answer_reward, q1_score))
                        val_q1_ecrr[-1] *= answer_reward
                        val_q2_ecrr[-1] *= args.cascade_p**correct_question_rank
                        if q_done:
                            q2_score = 0
                            if not correct_question_rank < args.reranker_return_length:
                                val_q2_scores.append(0)
                    elif n_round == 2:
                        val_q2_scores.append(min(answer_reward, q2_score))
                        val_q2_ecrr[-1] *= answer_reward

                    # ecrr evaluation
                    if 'ecrr' in locals():
                        if action == 1:
                            ecrr *= args.cascade_p**correct_question_rank
                            if correct_question_rank >= args.reranker_return_length:
                                ecrr *= answer_reward
                                val_ecrr.append(ecrr)
                                del ecrr
                        else:
                            ecrr *= answer_reward
                            val_ecrr.append(ecrr)
                            del ecrr
                    
                    if 'base_ecrr' in locals():
                        if base_action == 1 :
                            base_ecrr *= args.cascade_p**correct_question_rank
                            if correct_question_rank >= args.reranker_return_length:
                                base_ecrr *= answer_reward
                                val_base_ecrr.append(base_ecrr)
                                del base_ecrr
                        else:
                            base_ecrr *= answer_reward
                            val_base_ecrr.append(base_ecrr)
                            del base_ecrr
                    
                    if 'il_ecrr' in locals():
                        if il_action == 1 :
                            il_ecrr *= args.cascade_p**correct_question_rank
                            if correct_question_rank >= args.reranker_return_length:
                                il_ecrr *= answer_reward
                                val_il_ecrr.append(il_ecrr)
                                del il_ecrr
                        else:
                            il_ecrr *= answer_reward
                            val_il_ecrr.append(il_ecrr)
                            del il_ecrr

                    n_round += 1
                    context = context_

                # find the optimal trajectory
                best_answer_step, best_ecrr = find_best_trajectory(a_traj, q_traj, args.cascade_p)
                best_answer_reward = a_traj[best_answer_step][1]
                val_oracle_scores.append(best_answer_reward)
                val_oracle_ecrr.append(best_ecrr)

            # save batch cache
            if update_memory:
                T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/val/memory.batchsave' + str(batch_serial))
            del memory
            T.cuda.empty_cache()


        output.write("Val epoch " + str(i)+'\n')
        output.write("risk\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in val_scores]), np.mean(val_scores), np.mean(val_ecrr)))
        output.write("q0\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in val_q0_scores]), np.mean(val_q0_scores), np.mean(val_q0_ecrr)))
        output.write("q1\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in val_q1_scores]), np.mean(val_q1_scores), np.mean(val_q1_ecrr)))
        output.write("q2\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in val_q2_scores]), np.mean(val_q2_scores), np.mean(val_q2_ecrr)))
        output.write("base\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in val_base_scores]), np.mean(val_base_scores), np.mean(val_base_ecrr)))
        output.write("il\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in val_il_scores]), np.mean(val_il_scores), np.mean(val_il_ecrr)))
        output.write("oracle\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in val_oracle_scores]), np.mean(val_oracle_scores), np.mean(val_oracle_ecrr)))
        

        
        # plotting
        val_il_mrr_hist.append(np.mean(val_il_scores))
        val_il_ecrr_hist.append(np.mean(val_il_ecrr))
        if i > 30:
            plt.figure()
            plt.plot(X[31:], val_il_mrr_hist[31:], label="val_mrr")
            plt.plot(X[31:], val_il_ecrr_hist[31:], label="val_ecrr")
            plt.legend()
        #plt.savefig('fig_'+args.dataset_name +'_cv'+ str(args.cv)+'_top'+str(args.user_tolerance)+'_lr'+str(args.lr)+'_r'+str(args.cq_reward)+'_caps'+str(args.cascade_p)+'_val.png')
        

        ## test 
        test_scores, test_q0_scores, test_q1_scores, test_q2_scores, test_oracle_scores, test_base_scores, test_il_scores = [],[],[],[],[],[],[]
        test_ecrr, test_q0_ecrr, test_q1_ecrr, test_q2_ecrr, test_oracle_ecrr, test_base_ecrr, test_il_ecrr = [],[],[],[],[],[],[]
        agent.epsilon= 0
        
        print('testing')
        for batch_serial, batch in enumerate(test_dataset.batches):
    
            if os.path.exists(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/test/memory.batchsave' + str(batch_serial)):
                with T.no_grad():
                    memory = T.load(args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/test/memory.batchsave' + str(batch_serial))
            else:
                memory = {}
            update_memory = False
            
            test_ids = list(batch['conversations'].keys())
            user = User(batch['conversations'], cq_reward = args.cq_reward, cq_penalty = args.cq_reward - 1)
            for conv_serial, test_id in enumerate(test_ids):
                query = user.initialize_state(test_id)
                if query == '': # UDC dataset has some weird stuff
                    continue
                context = ''
                ignore_questions = []
                n_round = 0
                patience_used = 0
                q_done = False
                q1_score, q2_score = 100, 100
                stop, base_stop, il_stop = False,False,False
                ecrr, base_ecrr, il_ecrr = 1, 1, 1
                a_traj, q_traj, il_traj = [], [], []
                correct_question_rank = 0
                output.write('-------- test batch %.0f conversation %.0f/%.0f --------\n' % (batch_serial, args.batch_size*(batch_serial) + conv_serial + 1, test_size))
                # while not q_done:
                while n_round < len(batch['conversations'][test_id]) / 2 and correct_question_rank < args.reranker_return_length:
                    output.write('-------- round %.0f --------\n' % (n_round))
                    if query in memory.keys():
                        if context not in memory[query].keys():
                            # sampling
                            question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, test_id, args.batch_size)
                            # get reranker results   
                            questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                            answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                            memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                            update_memory = True
                            
                    else:
                        # sampling
                        question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, args.batch_size)
                        answer_candidates = generate_batch_answer_candidates(batch, test_id, args.batch_size)
                        # get reranker results
                        questions, questions_scores = rerank(question_reranker, query, context, question_candidates)
                        answers, answers_scores = rerank(answer_reranker, query, context, answer_candidates)
                        memory = save_to_memory(query, context, memory, questions, answers, questions_scores, answers_scores, tokenizer, embedding_model, device)
                        update_memory = True

                    query_embedding, context_embedding, questions, answers, questions_embeddings, answers_embeddings, questions_scores, answers_scores = read_from_memory(query, context, memory)
                    if i < args.risk_run_epoch:
                        state, action = agent.choose_action(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores)
                    else:
                        action = 0
                    if i < args.base_run_epoch:
                        base_action = base_agent.choose_action(query_embedding, context_embedding) 
                    else:
                        base_action = 0                   
                    # convil
                    state, il_action = ilagent.inference_step(query_embedding, context_embedding, questions_embeddings, answers_embeddings, questions_scores, answers_scores, mode = 'test')
                    context_, answer_reward, question_reward, correct_question_rank, q_done_this_round, good_question, patience_this_round = user.update_state(test_id, context, questions, answers, use_top_k = max(args.user_tolerance - patience_used, 1))
                    q_done = q_done or q_done_this_round
                    patience_used = max(patience_used + patience_this_round, args.user_tolerance)
                    
                    output.write('act '+str(action)+' base act '+str(base_action)+' il act '+str(il_action)+' a reward '+str(answer_reward)+' q reward '+str(question_reward) +' cq rank '+str(correct_question_rank) +'\n')
                    #test_output.write(str(i)+' '+str(test_id)+' '+str(n_round)+' '+str(action)+' '+str(base_action)+' '+str(il_action)+' '+str(answer_reward)+' '+str(correct_question_rank) +'\n')

                    a_traj.append((state, answer_reward))
                    q_traj.append((state, correct_question_rank))
                    il_traj.append((state, il_action, [answer_reward, correct_question_rank][il_action]))

                    if n_round >= args.user_patience:
                        q_done = True

                    if n_round < len(batch['conversations'][test_id]) / 2 and correct_question_rank < args.reranker_return_length:
                        ignore_questions.append(good_question)
                        if context_ not in memory[query].keys():
                            # sampling    
                            question_candidates = generate_batch_question_candidates(batch, test_id, ignore_questions, args.batch_size)
                            answer_candidates = generate_batch_answer_candidates(batch, test_id, args.batch_size)
                            # get reranker results
                            questions_, questions_scores_ = rerank(question_reranker, query, context_, question_candidates)
                            answers_, answers_scores_ = rerank(answer_reranker, query, context_, answer_candidates)
                            
                            memory = save_to_memory(query, context_, memory, questions_, answers_, questions_scores_, answers_scores_, tokenizer, embedding_model, device)
                            update_memory = True
                        query_embedding, context_embedding_, questions_, answers_, questions_embeddings_, answers_embeddings_, questions_scores_, answers_scores_ = read_from_memory(query, context_, memory)
                    # non-deterministic models evaluation
                    if not stop:              
                        if (action == 0 or (action == 1 and question_reward == args.cq_reward - 1)):
                            stop = True
                            test_scores.append(answer_reward if action == 0 else 0)

                    if not base_stop:
                        if (base_action == 0 or (base_action == 1 and question_reward == args.cq_reward - 1)):
                            base_stop = True
                            test_base_scores.append(answer_reward if base_action == 0 else 0)
                    
                    if not il_stop:
                        if (il_action == 0 or (il_action == 1 and question_reward == args.cq_reward - 1)):
                            il_stop = True
                            test_il_scores.append(answer_reward if il_action == 0 else 0)

                    # baseline models evaluations
                    if n_round == 0:
                        test_q0_scores.append(answer_reward)
                        test_q0_ecrr.append(answer_reward)
                        test_q1_ecrr.append(args.cascade_p**correct_question_rank)
                        test_q2_ecrr.append(args.cascade_p**correct_question_rank)
                        if q_done:
                            q1_score = 0
                            q2_score = 0
                            if not correct_question_rank < args.reranker_return_length:
                                test_q1_scores.append(0)
                                test_q2_scores.append(0)
                    elif n_round == 1:
                        test_q1_scores.append(min(answer_reward, q1_score))
                        test_q1_ecrr[-1] *= answer_reward
                        test_q2_ecrr[-1] *= args.cascade_p**correct_question_rank
                        if q_done:
                            q2_score = 0
                            if not correct_question_rank < args.reranker_return_length:
                                test_q2_scores.append(0)
                    elif n_round == 2:
                        test_q2_scores.append(min(answer_reward, q2_score))
                        test_q2_ecrr[-1] *= answer_reward

                    # ecrr evaluation
                    if 'ecrr' in locals():
                        if action == 1:
                            ecrr *= args.cascade_p**correct_question_rank
                            if correct_question_rank >= args.reranker_return_length:
                                ecrr *= answer_reward
                                test_ecrr.append(ecrr)
                                del ecrr
                        else:
                            ecrr *= answer_reward
                            test_ecrr.append(ecrr)
                            del ecrr
                    
                    if 'base_ecrr' in locals():
                        if base_action == 1 :
                            base_ecrr *= args.cascade_p**correct_question_rank
                            if correct_question_rank >= args.reranker_return_length:
                                base_ecrr *= answer_reward
                                test_base_ecrr.append(base_ecrr)
                                del base_ecrr
                        else:
                            base_ecrr *= answer_reward
                            test_base_ecrr.append(base_ecrr)
                            del base_ecrr
                    
                    if 'il_ecrr' in locals():
                        if il_action == 1 :
                            il_ecrr *= args.cascade_p**correct_question_rank
                            if correct_question_rank >= args.reranker_return_length:
                                il_ecrr *= answer_reward
                                test_il_ecrr.append(il_ecrr)
                                del il_ecrr
                        else:
                            il_ecrr *= answer_reward
                            test_il_ecrr.append(il_ecrr)
                            del il_ecrr

                    n_round += 1
                    context = context_
                # find the optimal trajectory
                best_answer_step, best_ecrr = find_best_trajectory(a_traj, q_traj, args.cascade_p)
                best_answer_reward = a_traj[best_answer_step][1]
                test_oracle_scores.append(best_answer_reward)
                test_oracle_ecrr.append(best_ecrr)

            

            # save batch cache
            if update_memory:
                T.save(memory, args.dataset_name + '_experiments/embedding_cache/' + args.reranker_name + '/' + args.cv + '/test/memory.batchsave' + str(batch_serial))
            del memory
            T.cuda.empty_cache()
        
        output.write("Test epoch " + str(i)+'\n')
        output.write("risk\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in test_scores]), np.mean(test_scores), np.mean(test_ecrr)))
        output.write("q0\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in test_q0_scores]), np.mean(test_q0_scores), np.mean(test_q0_ecrr)))
        output.write("q1\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in test_q1_scores]), np.mean(test_q1_scores), np.mean(test_q1_ecrr)))
        output.write("q2\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in test_q2_scores]), np.mean(test_q2_scores), np.mean(test_q2_ecrr)))
        output.write("base\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in test_base_scores]), np.mean(test_base_scores), np.mean(test_base_ecrr)))
        output.write("il\t\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in test_il_scores]), np.mean(test_il_scores), np.mean(test_il_ecrr)))   
        output.write("oracle\tacc %.6f, mrr %.6f, ecrr %.6f\n" % 
            (np.mean([1 if score == 1 else 0 for score in test_oracle_scores]), np.mean(test_oracle_scores),  np.mean(test_oracle_ecrr)))

        
        # plotting
        test_il_mrr_hist.append(np.mean(test_il_scores))
        test_il_ecrr_hist.append(np.mean(test_il_ecrr))
        if i > 30:
            plt.figure()
            plt.plot(X[31:], test_il_mrr_hist[31:], label="test_mrr")
            plt.plot(X[31:], test_il_ecrr_hist[31:], label="test_ecrr")
            plt.legend()
            plt.savefig('fig_'+args.dataset_name +'_cv'+ str(args.cv)+'_top'+str(args.user_tolerance)+'_lr'+str(args.lr)+'_r'+str(args.cq_reward)+'_caps'+str(args.cascade_p)+'_test.png')
        

    output.close()
    #train_output.close()
    #val_output.close()
    #test_output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type = str, default = 'MSDialog')
    parser.add_argument('--user_tolerance', type = int, default = 1)
    parser.add_argument('--user_patience', type = int, default = 10)
    parser.add_argument('--il_topn', type = int, default = 10)
    parser.add_argument('--cv', type = str, default = '')
    parser.add_argument('--reranker_name', type = str, default = 'poly')
    parser.add_argument('--cascade_p', type = float, default = 0.5)
    parser.add_argument('--reranker_return_length', type = int, default = 10)
    parser.add_argument('--observation_dim', type = int, default = 768)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--lrdc', type = float, default = 0.95)
    parser.add_argument('--weight_decay', type = float, default = 0.01)
    parser.add_argument('--n_action', type = int, default = 2)
    parser.add_argument('--cq_reward', type = float, default = 0.1)
    parser.add_argument('--max_iter', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--max_data_size', type = int, default = 10000)
    parser.add_argument('--checkpoint', type = str, default = '')
    parser.add_argument('--checkpoint_base', type = str, default = '')
    parser.add_argument('--checkpoint_il', type = str, default = '')
    parser.add_argument('--load_checkpoint', type = bool, default = False)
    parser.add_argument('--max_d_kl', type = float, default = 0.01)
    parser.add_argument('--entropy_weight', type = float, default = 1e-3)
    parser.add_argument('--pmax', type = float, default = 0.8)
    parser.add_argument('--disc_weight_clip', type = float, default = 0.01)
    parser.add_argument('--policy_weight_clip', type = float, default = 5)
    parser.add_argument('--disc_train_ratio', type = int, default = 1)
    parser.add_argument('--gan_name', type = str, default = 'GAN')
    parser.add_argument('--disc_pretrain_epochs', type = int, default = 30)
    parser.add_argument('--risk_run_epoch', type = int, default = 20)
    parser.add_argument('--base_run_epoch', type = int, default = 20)
    parser.add_argument('--dropout_ratio', type = float, default = 0.2)
    parser.add_argument('--trex_n_rewards', type = int, default = 5)
    parser.add_argument('--trex_traj_sample_length', type = int, default = 1)

    args = parser.parse_args()
    main(args)
