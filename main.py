""" main.py

Handles the training of the Reinforcement Learning Agent for the Wiki Game

TODO:

- put all hyperparameters into argparse
- glove representations for policy evaluation
- figure out how to capture state for vectors

"""
import sys
sys.setrecursionlimit(10**3)
import warnings
warnings.filterwarnings('ignore')
import time

from parse import parser
import random
import math
from qnetwork import QNetwork
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gymEnv.wikiGame.envs.wikiGame import wikiGame
from tqdm import tqdm

from replay_utils import Transition, ReplayMemory

from allennlp.modules.elmo import Elmo, batch_to_ids
from sacremoses import MosesTokenizer

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
WIKIPEDIA_DIAMETER = 70
BUFFER_CAPACITY = 1000
STATE_SIZE = 1024*3
SEED = 6884
FC1_UNITS = 1024
FC2_UNITS = 256
LR = 3e-4

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
    print(f"text {text}")
    tokenized_text = mt.tokenize(text, escape=False)
    print(f"tokenized_text {tokenized_text}")
    character_ids = batch_to_ids(tokenized_text)

    embedding = elmo(character_ids)
    print(f"char ids: \n type {type(character_ids)} \n size{character_ids.size()}")
    print(f"embeddding type {type(embedding)} \n length{len(embedding)}")
    print(f"each tensor in embedding has dims{ embedding['elmo_representations'][0].size()}")
    # time.sleep(60)
    embedding = torch.stack(embedding['elmo_representations'], dim=0)
    embedding = embedding.mean(dim=(0, 1))
    return embedding

def evaluate_expected_rewards(policy_net, current_state, goal_state_embedding, vertex_to_title):
    current_state_embedding = get_elmo_embedding(vertex_to_title[int(current_state)])
    rewards = torch.zeros((len(current_state.out_neighbors(), 1)))
    indexes = torch.zeros((len(current_state.out_neighbors(), 1)))
    for i, neighbor in enumerate(current_state.out_neighbors()):
        next_state_embedding = get_elmo_embedding(vertex_to_title[int(neighbor)])
        indexes[i] = int(neighbor)
        rewards[i] = policy_net(torch.cat((goal_state_embedding, current_state_embedding, next_state_embedding), 0))
    return rewards, indexes

def select_action(policy_net, state, goal_state, goal_state_embedding, steps_done, vertex_to_title):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            reward_vector, ix_vector = evaluate_expected_rewards(policy_net, state, goal_state_embedding, vertex_to_title)
            max_reward_ix  = reward_vector.max(dim=1)['indices'].view(1,1) #np.argmax(reward_vector)
            return ix_vector[max_reward_ix]
    else:
        return torch.tensor([[randomly_select_action(state)]], device=device, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple( #transforms iterator to tuple, then to tensor
                                        map(lambda s: s is not None, batch.next_state) #iterator
                                        ), 
                                device=device, 
                                dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train(env, memory, policy_net, target_net, optimizer):
    num_episodes = 50
    episode_durations = []
    steps_done = 0
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and state
        state, goal_state, vertex_to_title = env.reset()
        goal_state_embedding = get_elmo_embedding(vertex_to_title[int(goal_state)])
        print(f"goal state embedding {goal_state_embedding.size()}")
        for t in range(WIKIPEDIA_DIAMETER):
            # Select and perform an action
            action = select_action(policy_net, state, goal_state, goal_state_embedding, steps_done, vertex_to_title)
            steps_done += 1
            _, reward, done, info_dict = env.step(action.item())
            next_state = info_dict['next_vertex']

            reward = torch.tensor([reward], device=device)

            # Observe new state
            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
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
    print("main start", flush=True)
    memory = ReplayMemory(BUFFER_CAPACITY)
    policy_net = QNetwork(STATE_SIZE, SEED, FC1_UNITS,  FC2_UNITS)
    target_net = QNetwork(STATE_SIZE, SEED, FC1_UNITS,  FC2_UNITS)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    print("creating wikigame", flush=True)
    env = wikiGame()
    print("wikigame created", flush=True)
    train(env, memory, policy_net, target_net, optimizer)
    return

if __name__ == "__main__":
    print("entry", flush=True)
    args = parser.parse_args()
    main(args)