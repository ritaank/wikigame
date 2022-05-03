""" main.py

Handles the training of the Reinforcement Learning Agent for the Wiki Game

"""

from parse import parser
import random
import math
import torch

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

def select_action(policy_net, state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            expected_reward_vector = policy_net(state)
            valid_reward_vector = filter_valid_actions(state, expected_reward_vector)
            max_reward_ix  = valid_reward_vector.max(dim=1)['indices'].view(1,1)
            return max_reward_ix
    else:
        return torch.tensor([[randomly_select_action(state)]], device=device, dtype=torch.long)

def main(args):
    pass


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)