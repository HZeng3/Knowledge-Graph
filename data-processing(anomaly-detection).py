import pprint
pp = pprint.PrettyPrinter(indent=4)
import time
from collections import defaultdict
import torch
import random
import pickle
import argparse
from Embedding import *


def split_pos(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


class Data:

    def __init__(self, filepath_triple, filepath_label, max_ngb, embed_dim, entity_idxs, relation_idxs, undirected, islabel, max_hops):

        self.filepath_triple = filepath_triple
        self.filepath_label = filepath_label

        self.undirected = undirected

        self.triple_list = []
        self.triple_map = {}
        self.relation_dict = {}
        self.relation_triple_dict = {}
        self.relation_list = ['NONE']
        self.embed_dim = embed_dim
        self.relation_idxs = relation_idxs
        self.entity_idxs = entity_idxs

        # self.unique_relation_num = -1 # k, number of relations (without coucnting the inverses))
        # self.unique_entity_num = -1 # total number of unique entities in the KG

        self.max_ngb = max_ngb
        self.graph = defaultdict(list)  # stores the whole KG in a graph

        self.ego_graphs = []  # a list of smaller EgoGraph instances, list len = number of unique entities in the KG
        self.max_hops = max_hops

        with open(self.filepath_triple, 'r') as fp:
            self.triple_list = fp.readlines()  # raw triple list from file
        entity_list = []
        for triple in self.triple_list:
            head, rel, tail = triple.strip().split('\t')  # split the head, tail, relations
            # head, rel, tail = triple.strip().split(' ') # split the head, tail, relations
            # the separator may change depending on the dataset

            entity_list.extend([head, tail])

            self.add_edge(head, tail, self.undirected)

            if rel not in self.relation_dict.keys():
                # add if already not in relation dictionaries
                self.relation_dict[rel] = 1
                self.relation_triple_dict[rel] = [triple]

                if self.undirected:
                    self.relation_list.extend([rel, rel + '-inv'])
                else:
                    self.relation_list.append(rel)
            else:
                # increase and append if in dictionaries
                self.relation_dict[rel] += 1
                self.relation_triple_dict[rel].append(triple)

            if (head, tail) in self.triple_map.keys():
                self.triple_map[(head, tail)].append(rel)
            else:
                self.triple_map[(head, tail)] = [rel]

            if self.undirected:
                if (tail, head) in self.triple_map.keys():
                    self.triple_map[(tail, head)].append(rel + '-inv')
                else:
                    self.triple_map[(tail, head)] = [rel + '-inv']

        # self.unique_entity_list = list(set(entity_list))
        # prepare unique entity list based on the entity idxs returned from tucker
        self.unique_entity_list = []
        for key in self.entity_idxs.keys():
            # print(f"key: {key} and value: {self.entity_idxs[key]}")
            # print(f"key: {type(key)} and value: {type(self.entity_idxs[key])}")

            idx = int(self.entity_idxs[key])
            self.unique_entity_list.insert(idx, key)
        if verbose:
            print(self.unique_entity_list[:10])
        self.unique_entity_num = len(self.unique_entity_list)

        # self.x = torch.rand(self.unique_entity_num, self.embed_dim) # use tucker or random
        # self.X = torch.randn((self.unique_entity_num, self.embed_dim))

        if islabel:
            self.labels = []
            self.labels_dict = {}  # key: entity, value: its labels in a list

            # process and store labels
            data = []  # stores all data from labels.txt file
            with open(self.filepath_label, 'r') as fp:
                data = fp.readlines()
            for entry in data:
                entity, labels_str = split_pos(entry, ' ', 1)[0], split_pos(entry, ' ', 1)[1]
                self.labels_dict[entity] = []
                if labels_str == labels_str.split(' '):  # means there's only one label, no split
                    self.labels_dict[entity] = labels_str
                else:
                    self.labels_dict[entity].extend(labels_str.strip().split(' '))
                self.labels.extend(entry.strip().split(' ')[1:])
            self.unique_labels = list(set(self.labels))

        if verbose:
            print('|V| -> Number of unique entities = {}'.format(self.unique_entity_num))
            print('|E| -> Number of triples/edges = {}'.format(len(self.triple_list)))
            if islabel:
                print('|L_v| -> Number of unique node labels = {}'.format(len(self.unique_labels)))
            print('|L_e| -> Number of edge labels or unique relations = {}'.format(len(self.relation_dict)))

    def add_edge(self, u, v, undirected):
        self.graph[u].append(v)
        if undirected:
            self.graph[v].append(u)

    def BFS(self, start, shuffle=True):
        # m is the maximum size of the neighborhood
        # visited = [False] * (2*m + 1)
        visited = {start}
        queue = [start]
        new_queue = []
        neighbors = [start]

        # visited[start] = True
        flag = True
        current_hops = 0

        while queue:
            s = queue.pop(0)
            for i in self.graph[s]:

                if i not in visited:

                    new_queue.append(i)
                    neighbors.append(i)
                    visited.add(i)
                    if len(neighbors) == self.max_ngb + 1:  # +1 because the source node is included in the
                        flag = False  # break loop if maximum neighborhood size is reached
                        break
            if len(queue) == 0:
                queue = new_queue
                new_queue = []
                current_hops += 1
                if current_hops > self.max_hops:
                    flag = False  # break loop if all neighbors in n-hop distance have been reached
            if not flag:
                break

        if shuffle:
            center_node = neighbors[0]
            random.shuffle(neighbors)
            center_node_pos = neighbors.index(center_node)
            neighbors[center_node_pos], neighbors[0] = neighbors[0], center_node

        node_indices = [self.unique_entity_list.index(i) for i in neighbors]

        return neighbors, node_indices

    def print_class(self):
        pp.pprint(vars(self))

    def create_ego_classes(self):
        count = 0
        min = 51
        max = 1
        n_size = 0
        for entity in self.unique_entity_list:
            self.ego_graphs.append(EgoGraphFeat(self, entity))

            sample_g = EgoGraphFeat(self, entity)
            if len(sample_g.neighbor_nodes) < min:
                min = len(sample_g.neighbor_nodes)
            if len(sample_g.neighbor_nodes) > max:
                max = len(sample_g.neighbor_nodes)
            count += sample_g.connection
            n_size += len(sample_g.neighbor_nodes)

        print(
            f"{round(count * 100 / self.unique_entity_num, 2)}%: {count} out of {(self.unique_entity_num)} graphs have a connection from center to last node")
        print(f"max neighborhood: {max} and min neighborhood: {min}")
        print(f"average neighborhood: {n_size / self.unique_entity_num}")

        with open(path + dataset_name + "_ego_graphs.pkl","wb") as f:
            pickle.dump(self.ego_graphs, f)
            f.close()
            print("ego_graphs file saved in pkl format")


class EgoGraphFeat:
  def __init__(self, kg, node):
    self.center_node = node
    self.neighbor_nodes, self.node_indices = kg.BFS(node)
    # the first item returned by the BFS is always the start node of BFS (center node)
    # store whether there is a connection or not
    self.connection = 0
    if (self.center_node, self.neighbor_nodes[-1]) in kg.triple_map.keys():
      self.connection = 1
    # if len(self.neighbor_nodes) == 1: # means the node has no neighbors
      # print(f"node with no neighbor detected")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora_all_train", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--max_hops", type=int, default=2, nargs="?",
                        help="Max number of hops in ego-graph.")
    parser.add_argument("--embedding_dim", type=int, default=50, nargs="?",
                        help="Embedding dimension (hyper-parameter).")
    parser.add_argument("--max_ngb", type=int, default=30, nargs="?",
                        help="Max number of neighbors in ego-graph")

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset

    verbose = True
    filepath_triple = './data/'+ dataset_name + '/cora_triples.txt'
    filepath_label = './data/' + dataset_name + '/nell_labels.txt'

    path = 'embeddings/'

    entity_idxs = pickle.load(open(path + dataset_name + '_entity_idxs.pkl', 'rb'))
    relation_idxs = pickle.load(open(path + dataset_name + '_relation_idxs.pkl', 'rb'))

    max_ngb = args.max_ngb
    embedding_dim = args.embedding_dim
    undirected = True
    islabel = False
    max_hops = args.max_hops

    d = Data(filepath_triple, filepath_label, max_ngb, embedding_dim, entity_idxs, relation_idxs, undirected, islabel, max_hops)
    d.create_ego_classes()

    # if verbose:
    #   for eg in d.ego_graphs:
    #     print(f'center node: {eg.center_node}, neighbors: {eg.neighbor_nodes}, neighbor indices: {eg.node_indices}')