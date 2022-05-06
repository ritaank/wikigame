import sys
sys.path.append('/opt/homebrew/Cellar/graph-tool/2.44_1/lib/python3.9/site-packages')

import gym
from gym import spaces
from pathlib import Path
from graph_tool.all import Graph, load_graph
import pandas as pd
import numpy as np
import os

graph_source = 'enwiki.wikilink_graph.2001-03-01.csv'
# 'enwiki.wikilink_graph.2018-03-01.csv.gz

def create_wiki_graph():
    df = pd.read_csv(f'gymEnv/wikiGame/envs/{graph_source}', sep='\t')
    print(df.head())
    all_pages = df['page_title_from'].unique()
    g = Graph()
    v_prop = g.new_vertex_property("string")
    vertices = g.add_vertex(len(all_pages))
    ix_to_name_d = {}
    name_to_ix_d = {}
    for vertex, page_name in zip(vertices, all_pages):
        v_prop[vertex] = page_name
        ix_to_name_d[int(vertex)] = page_name
        name_to_ix_d[page_name] = int(vertex)

    #assign properties as a dic value
    g.vertex_properties["name"] = v_prop
    #remap page titles to graph vertex indexes... will make adding edges significantly easier
    
    df["page_title_from_ix"] = df["page_title_from"].map(name_to_ix_d)
    df["page_title_to_ix"] = df["page_title_to"].map(name_to_ix_d)

    #add edges from remapped dataframe
    g.add_edge_list(df[["page_title_from_ix", "page_title_to_ix"]].values)

    return g, ix_to_name_d, name_to_ix_d, len(all_pages)

class wikiGame(gym.Env):
    """
    will have fixed action space, but not all actions are valid within each state
    step function should have a function that tests if the chosen action is valid
    the observation returned will be the graph_tool graph, but the state will just be
    the adjacency matrix (? maybe, currently have obs space as the matrix)
    maybe step function just alters the given graph
    """
    metadata = {'render.modes': ['human', 'graph', 'interactive']}

    def __init__(self):

        graph_file = Path("gymEnv/wikiGame/envs/wikiGraph.xml.gz")
        if graph_file.is_file():
            self.graph = load_graph("gymEnv/wikiGame/envs/wikiGraph.xml.gz")
            self.ix_to_name_d = {}
            self.name_to_ix_d = {}
            v_prop = self.graph.vertex_properties["name"]
            self.n_vertices = 0
            for vertex in self.graph.vertices():
                self.ix_to_name_d[int(vertex)] = v_prop[vertex]
                self.name_to_ix_d[v_prop[vertex]] = int(vertex)
                self.n_vertices += 1
        else:
            self.graph, self.ix_to_name_d, self.name_to_ix_d, self.n_vertices = create_wiki_graph()
            self.graph.save("gymEnv/wikiGame/envs/wikiGraph.xml.gz")

        self.current_vertex, self.goal_vertex = None, None
        print("in wikigame() init, before reset")

        # self.reset()

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
        self.current_vertex = self.graph.vertex(action)
        done = 0
        reward = -1
        if self.goal_vertex == self.current_vertex:
            reward = 1
            done = 1
        return None, reward, done, {"next_vertex": self.current_vertex} #no observations, this is an MDP not POMDP

    def reset(self):
        init_ix, goal_ix = np.random.choice(self.n_vertices, 2, replace=False)
        self.current_vertex = self.graph.vertex(init_ix)
        self.goal_vertex = self.graph.vertex(goal_ix)
        print(f"current vtx: {self.ix_to_name_d[int(self.current_vertex)]},\tgoal vtx: {self.ix_to_name_d[int(self.goal_vertex)]}")
        return self.current_vertex, \
                self.goal_vertex, \
                self.ix_to_name_d