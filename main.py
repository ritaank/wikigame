""" main.py

Handles the training of the Reinforcement Learning Agent for the Wiki Game

TODO:
    - fix destination node for quicker training
    - settle on good hyperparams
    - make sure this is runnabel on colab
"""
import math
import random
import sys
import warnings
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymEnv.wikiGame.envs.wikiGame import wikiGame
from parse import parser
from qnetwork import QNetwork
from replay_utils import ReplayMemory, Transition
from tqdm import tqdm

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from allennlp.modules.elmo import Elmo, batch_to_ids
    from sacremoses import MosesTokenizer

warnings.filterwarnings('ignore')
sys.setrecursionlimit(10**3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#tokenizer + elmo model
mt = MosesTokenizer(lang='en')
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

def randomly_select_action(state):
    random_action = np.random.choice([int(neighbor) for neighbor in state.out_neighbors()], 1)
    return random_action

def get_elmo_embedding(text):
    # print(f"text {text}")
    tokenized_text = mt.tokenize(text, escape=False)
    character_ids = batch_to_ids([tokenized_text])

    embedding = elmo(character_ids)
    embedding = torch.stack(embedding['elmo_representations'], dim=0)
    embedding = embedding.view(embedding.size(-1), -1).mean(1)
    return embedding

def evaluate_expected_rewards(policy_net, current_state, goal_state_embedding, vertex_to_title):
    current_state_embedding = get_elmo_embedding(vertex_to_title[int(current_state)])
    rewards = torch.zeros((sum(1 for _ in current_state.out_neighbors()), 1))
    indexes = torch.zeros((sum(1 for _ in current_state.out_neighbors()), 1))
    for i, neighbor in enumerate(current_state.out_neighbors()):
        next_state_embedding = get_elmo_embedding(vertex_to_title[int(neighbor)])
        indexes[i] = int(neighbor)
        combined_state_action = torch.cat((goal_state_embedding, current_state_embedding, next_state_embedding), dim=0).unsqueeze(0)
        x = policy_net(combined_state_action)
        rewards[i] = x
    return rewards, indexes

def select_action(policy_net, state, goal_state_embedding, eps_threshold, vertex_to_title):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            reward_vector, ix_vector = evaluate_expected_rewards(policy_net, state, goal_state_embedding, vertex_to_title)
            max_reward_ix  =  np.argmax(reward_vector, axis=0) #reward_vector.max(dim=1)['indices'].view(1,1)
            out = ix_vector[max_reward_ix].long()
            return out
    else:
        out = torch.tensor([randomly_select_action(state)], device=device, dtype=torch.long)
        return out

def optimize_model(args, memory, policy_net, target_net, optimizer, vertex_to_title):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)

    losses = []
    loss_fn = nn.SmoothL1Loss()
    for state, _, next_state, reward, goal_state_embedding in transitions:
        cur_reward_vector, _ = evaluate_expected_rewards(policy_net, state, goal_state_embedding, vertex_to_title)
        expected_reward_vector, _ = evaluate_expected_rewards(target_net, next_state, goal_state_embedding, vertex_to_title)
        
        future_val = reward + args.gamma * expected_reward_vector.max()
        temporal_diff = loss_fn(cur_reward_vector.max(), future_val)
        losses.append(temporal_diff)
    loss = sum(losses)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train(args, env, memory, policy_net, target_net, optimizer):
    num_episodes = 50
    episode_durations = []
    steps_done = 0
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        state, goal_state, vertex_to_title = env.reset()
        goal_state_embedding = get_elmo_embedding(vertex_to_title[int(goal_state)])
        for t in tqdm(range(args.max_ep_length), position=0, leave=True):
            # Select and perform an action
            eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * steps_done / args.eps_decay)
            action = select_action(policy_net, state, goal_state_embedding, eps_threshold, vertex_to_title)

            steps_done += 1
            _, reward, done, info_dict = env.step(action.item())
            next_state = info_dict['next_vertex']

            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            #print("PUSH TO MEM", state, next_state, goal_state, reward,)
            memory.push(state, action, next_state, reward, goal_state_embedding)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # print("about to optimize model", flush=True)
            optimize_model(memory, policy_net, target_net, optimizer, vertex_to_title)
            if done or t == args.max_ep_length-1:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Training Complete')
    env.render()
    env.close()

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def main(args):
    memory = ReplayMemory(args.buffer_capacity)
    policy_net = QNetwork(args.state_size, args.fc1_units, args.fc2_units)
    target_net = QNetwork(args.state_size, args.fc1_units, args.fc2_units)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    print("creating wikigame", flush=True)
    env = wikiGame()
    train(env, memory, policy_net, target_net, optimizer)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)