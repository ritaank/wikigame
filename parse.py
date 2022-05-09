import argparse

parser = argparse.ArgumentParser("WALDO - Wikipedia Agent for Learning Definition-informed Objectives")

parser.add_argument("--save", type=str, default="../results/tmp")
parser.add_argument("--max_ep_length", type=int, default=25)
parser.add_argument("--num_episodes", type=int, default=300)
parser.add_argument("--buffer_capacity", type=int, default=1000)
parser.add_argument("--wiki_year", type=int, default=2006)

#Hyperparams
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--gamma", type=float, default=.999)
parser.add_argument("--eps_start", type=float, default=.9)
parser.add_argument("--eps_end", type=float, default=.05)
parser.add_argument("--eps_decay", type=int, default=50)
parser.add_argument("--target_update", type=int, default=10)

#QNetwork Params
parser.add_argument("--state_size", type=int, default=1024*3)
parser.add_argument("--fc1_units", type=int, default=1024)
parser.add_argument("--fc2_units", type=int, default=256)

#optimizer params
parser.add_argument("--lr", type=float, default=3e-4)

#gym params
parser.add_argument("--has_fixed_dest_node", type=bool, default=False)
parser.add_argument("--fixed_dest_node", type=str, default="Applied Statistics")

