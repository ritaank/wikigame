import math
import sys
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import networkx as nx
import pickle

def create_wiki_graph(graph_source):
    df = pd.read_csv(graph_source, sep='\t', header=0)
    df = df[df['page_id_from'] != df['page_id_to']]
    df = df.dropna()
    g = nx.from_pandas_edgelist(df, source='page_title_from', target='page_title_to', create_using=nx.DiGraph)
    remove = (node for node in list(g) if g.out_degree(node) == 0)
    g.remove_nodes_from(remove)
    return g

class wikiGame(gym.Env):
    """
    will have fixed action space, but not all actions are valid within each state
    step function should have a function that tests if the chosen action is valid
    the observation returned will be the graph_tool graph, but the state will just be
    the adjacency matrix (? maybe, currently have obs space as the matrix)
    maybe step function just alters the given graph
    """
    metadata = {'render.modes': ['human', 'graph', 'interactive']}

    def __init__(self, args):
        graph_file_txt = f"gymEnv/wikiGame/envs/wikiGraph_{args.wiki_year}.gpickle"
        graph_file = Path(graph_file_txt)
        if graph_file.is_file():
            print("loading graph file")
            self.graph = nx.read_gpickle(graph_file_txt)
        else:
            print("creating graph file")
            graph_source_text = f"gymEnv/wikiGame/envs/enwiki.wikilink_graph.{args.wiki_year}-03-01.csv.gz"
            self.graph = create_wiki_graph(graph_source_text)
            nx.write_gpickle(self.graph, graph_file_txt)

        self.current_vertex, self.goal_vertex = None, None

        #DEAL WITH FIXED DESTINATION AND MAX BFS DIST
        self.has_fixed_dest_node = args.has_fixed_dest_node
        self.bfs_center_node = args.bfs_center_node
        self.max_bfs_dist = args.max_bfs_dist
        if self.max_bfs_dist > 0: #if we put a bfs dist, we want to build graph from some center node
            print("OLD GRAPH SIZE", len(self.graph.nodes()))
            self.graph = self.trim_graph(graph=self.graph, target=self.bfs_center_node, cutoff=self.max_bfs_dist)
            print(f"NEW GRAPH SIZE (all within distance {self.max_bfs_dist} of {self.bfs_center_node})", len(self.graph.nodes()))
        
        # DEAL WITH EXPANDING BFS PROGRESSIVE SPECTRAL DEMASKING 
        self.expanding_bfs = args.expanding_bfs
        self.num_episodes = 0
        self.max_episodes = args.num_episodes
        self.full_graph = self.graph
        self.last_trim_call_params = None
        self.reset()


    def render(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.graph
        elif mode == 'interactive':
            interactive_window(self.graph)
        elif mode == 'human':
            # filename = "./renders/render" + str(self.time_step) + ".png"
            # graph_draw(self.graph, vertex_text=self.graph.vertex_index, vertex_font_size=18,
            #             output_size=(1000, 1000), output=filename)
            nx.draw(self.graph)

    def step(self, action):
        self.current_vertex = action
        done = 0
        reward = -1
        if self.goal_vertex == self.current_vertex:
            reward = 1
            done = 1
        return None, reward, done, {} #no observations, this is an MDP not POMDP

    def reset(self):
        self.goal_vertex = self.bfs_center_node if self.has_fixed_dest_node else np.random.choice(self.graph.nodes(), 1)[0]
        if self.expanding_bfs:
            curr_bfs_dist =  self.calc_bfs_dist_schedule()
            if curr_bfs_dist == self.max_bfs_dist:#WE ALREADY REACHED FULL SIZE, DONT BOTHER RECALCULATING
                pass 
            elif self.last_trim_call_params != (self.goal_vertex, curr_bfs_dist): #IF NEW GOAL OR NEW DiSTANCE, RETRIM
                self.graph = self.trim_graph(self.full_graph, self.goal_vertex, curr_bfs_dist)
                self.last_trim_call_params = (self.goal_vertex, curr_bfs_dist)

        self.current_vertex = np.random.choice(self.graph.nodes(), 1)[0]
        while self.current_vertex == self.goal_vertex:
            self.current_vertex = np.random.choice(self.graph.nodes(), 1)[0]
        
        self.num_episodes += 1
        return self.current_vertex, \
                self.goal_vertex

    def calc_bfs_dist_schedule(self):
        #first, calculate accoridng to exponential increase based on the fitting points
        k = 1 / (0 - .6 * self.max_episodes) * math.log(1/self.max_bfs_dist)
        curr_bfs_dist = min([math.floor(math.exp(k)), self.max_bfs_dist])

        return curr_bfs_dist

    def trim_graph(self, graph, target, cutoff):
        path = nx.single_target_shortest_path(graph, target=target, cutoff=cutoff)
        desired_nodes = [k for k in path]
        ret_graph = graph.subgraph(desired_nodes)
        return ret_graph
