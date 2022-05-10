import argparse
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm

from gymEnv.wikiGame.envs.wikiGame import wikiGame
from qnetwork import QNetwork

from transformers import DistilBertTokenizer, DistilBertModel

eval_parser = argparse.ArgumentParser("WALDO evaluation")
eval_parser.add_argument('-p','--path', type=str, help='path to trained model (.pt) file', required=True)
eval_parser.add_argument('--num_tests', type=int, help='how many tests to run?', default=5)
eval_parser.add_argument('--dist_levels', type=list, help='what levels to run tests at', default=[2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PATH = '/content/wikigame/models/2006_fixednode-True_2022_05_10-12_07_38_PM.pt'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda()
for param in model.parameters():
    param.requires_grad = False

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

cached = {}

def get_neural_embedding(text):
    if text in cached:
        output = cached[text]
    else:
        with torch.no_grad():
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

def select_action(qnet, state, goal_state_embedding, possible_actions):
    with torch.no_grad():
        reward_vector = evaluate_expected_rewards(qnet, state, goal_state_embedding, possible_actions)
        max_reward_ix = np.argmax(reward_vector, axis=0)
        out = possible_actions[max_reward_ix]
        return out

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

def evaluate(qnet, env, args, potential_start_nodes):

        episode_durations = []
        distance_ratios = []
        cos_sims = []

        steps_done = 0
        env = wikiGame(args)
        wins = fails = reached_sink = 0

        for _ in tqdm(range(args.num_tests)):
            # print(f"i {i}")

            source = np.random.choice(potential_start_nodes, 1)[0]
            print("our source is ", source)
            state, goal_state = env.reset(evalMode=True, node=source)
            path_taken = [state]
            # print(f"starting at {state} and going to {goal_state}")

            best_path_list = nx.shortest_path(env.graph, source=state, target=goal_state, weight=None, method='dijkstra')
            print("best", best_path_list)
            best_len = len(best_path_list)

            goal_state_embedding = get_neural_embedding(goal_state)
            eval_goal_state_embedding = torch.clone(get_neural_embedding(goal_state)).unsqueeze(0).cpu()
            curr_state_embedding = get_neural_embedding(state).cpu()
            initial_cos_sim = cosine_similarity(curr_state_embedding.unsqueeze(0), eval_goal_state_embedding)

            if args.max_bfs_dist > 0:
                limit = 2*args.max_bfs_dist #what should this multiplier be?
            else:
                limit = args.max_ep_length

            # loop over limit number of steps to find the dest
            for t in range(limit+1):

                # Select and perform an action
                possible_actions = list(env.graph.successors(state))

                if (len(possible_actions) == 0):
                    # we lose this game
                    # print("losing")
                    episode_durations.append(limit)

                    path_list = nx.shortest_path(env.graph, source=state, target=goal_state, weight=None, method='dijkstra')
                    distance_ratios.append(len(path_list)/best_len)

                    cos_sim = cosine_similarity(get_neural_embedding(state).cpu().unsqueeze(0), eval_goal_state_embedding)
                    cos_sims.append(cos_sim - initial_cos_sim)

                    print(path_taken, "\n")

                    fails +=1
                    reached_sink += 1
                    break
                
                action = select_action(qnet, state, goal_state_embedding, possible_actions)
                steps_done += 1
                _, reward, done, _ = env.step(action)

                if done:
                    # we win
                    # print("winning")
                    episode_durations.append(t+1)

                    distance_ratios.append(0/best_len)
                    cos_sims.append(1 - initial_cos_sim)

                    print(path_taken + [goal_state], "\n")

                    wins +=1
                    break

                # print(f"going to state {action}")
                state = action
                path_taken.append(state)

                if t > limit-1:
                    # we out of steps, lose the game
                    episode_durations.append(t+1)

                    path_list = nx.shortest_path(env.graph, source=state, target=goal_state, weight=None, method='dijkstra')
                    distance_ratios.append(len(path_list)/best_len)

                    cos_sim = cosine_similarity(get_neural_embedding(state).cpu().unsqueeze(0), eval_goal_state_embedding)
                    cos_sims.append(cos_sim - initial_cos_sim)

                    print(path_taken, "\n")

                    fails +=1
                    break

        assert wins + fails == args.num_tests, f"w:{wins} and f:{fails} big bug"
        print("settings\t", args)
        print("success rate:\t", wins/args.num_tests)
        print("rate we reached a dead end:\t", reached_sink/args.num_tests)
        # print("distance ratios, lower is better:\n", distance_ratios)
        print("average distance ratio:\t",sum(distance_ratios)/len(distance_ratios))
        # print("cos sims, higher is better:\n", cos_sims)
        print("avg improvement in cos sim:\t",sum(cos_sims)/len(cos_sims))
        # plot_durations(episode_durations)


def main(eval_args):
    model_and_args = torch.load(eval_args.path)
    args = model_and_args['args']
    args.num_tests = eval_args.num_tests
    state_dict = model_and_args['state_dict']
    print("loaded model from ", eval_args.path)
    trained_net = QNetwork(args.state_size, args.fc1_units, args.fc2_units).to(device)
    trained_net.load_state_dict(state_dict)

    try:
        args.bfs_center_node = args.fixed_dest_node
    except:
        pass

    print(args)

    with torch.no_grad():
        trained_net.eval()
        env = wikiGame(args)
        nodes_by_dist = env.get_nodes_by_distances(eval_args.dist_levels) #args.tiers should be a list
        for level in eval_args.dist_levels:
            evaluate(trained_net, env, args, nodes_by_dist[level])
        
        env.close()

if __name__ == "__main__":
    eval_args = eval_parser.parse_args()
    main(eval_args)