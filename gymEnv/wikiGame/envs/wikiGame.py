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

    def __init__(self, has_fixed_dest_node=False, fixed_dest_node='Massachusetts Institute of Technology', wiki_year=2006):
        graph_file_txt = f"gymEnv/wikiGame/envs/wikiGraph_{wiki_year}.gpickle"
        graph_file = Path(graph_file_txt)
        if graph_file.is_file():
            print("loading graph file")
            self.graph = nx.read_gpickle(graph_file_txt)
        else:
            print("creating graph file")
            graph_source_text = f"gymEnv/wikiGame/envs/enwiki.wikilink_graph.{wiki_year}-03-01.csv.gz"
            self.graph = create_wiki_graph(graph_source_text)
            nx.write_gpickle(self.graph, graph_file_txt)

        self.current_vertex, self.goal_vertex = None, None
        self.has_fixed_dest_node = has_fixed_dest_node
        self.fixed_dest_node = fixed_dest_node
        self.reset()


    def render(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.graph
        elif mode == 'interactive':
            interactive_window(self.graph)
        elif mode == 'human':
            filename = "./renders/render" + str(self.time_step) + ".png"
            graph_draw(self.graph, vertex_text=self.graph.vertex_index, vertex_font_size=18,
                        output_size=(1000, 1000), output=filename)

    def step(self, action):
        self.current_vertex = action
        done = 0
        reward = -1
        if self.goal_vertex == self.current_vertex:
            reward = 1
            done = 1
        return None, reward, done, {} #no observations, this is an MDP not POMDP

    def reset(self):
        self.goal_vertex = self.fixed_dest_node if self.has_fixed_dest_node else np.random.choice(self.graph.nodes(), 1)[0]
        self.current_vertex = np.random.choice(self.graph.nodes(), 1)[0]
        while self.current_vertex == self.goal_vertex:
            self.current_vertex = np.random.choice(self.graph.nodes(), 1)[0]
        return self.current_vertex, \
                self.goal_vertex