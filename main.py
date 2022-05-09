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

from transformers import DistilBertTokenizer, DistilBertModel

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=DeprecationWarning)
#     print("FUCK ME")
#     from allennlp.modules.elmo import Elmo, batch_to_ids
#     print("FUCK ME 2")
#     from sacremoses import MosesTokenizer

warnings.filterwarnings('ignore')
sys.setrecursionlimit(10**3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display




tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
cached = {}

def get_neural_embedding(text):
    if text in cached:
        output = cached[text]
    else:
        encoded_input = tokenizer(text, return_tensors='pt').to(device)
        output = model(**encoded_input).last_hidden_state.mean(dim=1).view(-1).to(device)
        cached[text] = output
    return output

def evaluate_expected_rewards(policy_net, state, goal_state_embedding, possible_actions):
    current_state_embedding = get_neural_embedding(state)
    rewards = torch.empty(size=(len(possible_actions), 1))
    for i, neighbor in enumerate(possible_actions):
        next_state_embedding = get_neural_embedding(neighbor)
        combined_state_action = torch.cat((goal_state_embedding, current_state_embedding, next_state_embedding), dim=0).unsqueeze(0)
        x = policy_net(combined_state_action)
        rewards[i] = x
    return rewards

def select_action(policy_net, state, goal_state_embedding, eps_threshold, possible_actions):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            reward_vector = evaluate_expected_rewards(policy_net, state, goal_state_embedding, possible_actions)
            max_reward_ix = np.argmax(reward_vector, axis=0) 
            out = possible_actions[max_reward_ix]
            return out
    else:
        out = np.random.choice(list(possible_actions), 1)[0]
        return out

def optimize_model(env, args, memory, policy_net, target_net, optimizer):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)

    loss = torch.zeros(size=(1, 1))
    loss_fn = nn.SmoothL1Loss()
    for state, _, next_state, reward, goal_state_embedding in transitions:
        cur_possible_actions = list(env.graph.successors(state))
        cur_reward_vector = evaluate_expected_rewards(policy_net, state, goal_state_embedding, cur_possible_actions)
        next_possible_actions = list(env.graph.successors(next_state))
        expected_reward_vector = evaluate_expected_rewards(target_net, next_state, goal_state_embedding, next_possible_actions)
        future_val = reward + args.gamma * expected_reward_vector.max()
        temporal_diff = loss_fn(cur_reward_vector.max(), future_val)
        loss = loss + temporal_diff

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train(args, env, memory, policy_net, target_net, optimizer):
    episode_durations = []
    steps_done = 0
    for i_episode in tqdm(range(args.num_episodes)):
        print(i_episode, args.num_episodes)
        # Initialize the environment and state
        state, goal_state = env.reset()
        goal_state_embedding = get_neural_embedding(goal_state)
        for t in tqdm(range(args.max_ep_length), position=0, leave=True):
            # Select and perform an action
            eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * steps_done / args.eps_decay)
            possible_actions = list(env.graph.successors(state))
            
            if sum(1 for _ in possible_actions) == 0:
                episode_durations.append(args.max_ep_length)
                plot_durations(episode_durations)
                break
            action = select_action(policy_net, state, goal_state_embedding, eps_threshold, possible_actions)
            # print("done selectign action")
            steps_done += 1
            _, reward, done, _ = env.step(action)


            next_state = action #by virtue of deterministic observed transitions 

            reward = torch.tensor([reward], device=device)

            # Store the transition in memory
            
            memory.push(state, action, next_state, reward, goal_state_embedding)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # print("about to optimize model", flush=True)
            optimize_model(env, args, memory, policy_net, target_net, optimizer)
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
    policy_net = QNetwork(args.state_size, args.fc1_units, args.fc2_units).to(device)
    target_net = QNetwork(args.state_size, args.fc1_units, args.fc2_units).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    print("creating wikigame", flush=True)
    env = wikiGame(has_fixed_dest_node=args.has_fixed_dest_node, 
                    fixed_dest_node=args.fixed_dest_node, 
                    wiki_year=args.wiki_year)
    train(args, env, memory, policy_net, target_net, optimizer)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)