"""Simple Prefix tree"""

class Node:
    def __init__(self):
        self.k:int
        self.v:int
        self.child: dict[int, "Node"] = dict()
    def insert(self, k: int, v: int):
        self.k = k
        self.v = v
    def add_child(self, edge_val: int, child_node:"Node"):
        self.child[edge_val] = child_node

class PrefixTree:
    def __init__(self):
        self.root = Node()
    def insert_token(self, embedding, k_val, v_val, start_node=None):
        assert len(k_val) == len(v_val) == len(embedding)
        curr_node = self.root if start_node is None else start_node
        for token, k, v in zip(embedding, k_val, v_val):
            if token in curr_node.child:
                curr_node = curr_node.child[token]
            else:
                new_node = Node()
                new_node.insert(k,v)
                curr_node.child[token] = new_node
                curr_node = new_node
    def longest_match(self, embedding):
        """return the k, v value; length can be infered from len(k), also
        return the node to make insert the found token, maybe faster"""
        curr_node = self.root
        k_val=[]
        v_val=[]
        for token in embedding:
            if token in curr_node.child:
                k_val.append(curr_node.child[token].k)
                v_val.append(curr_node.child[token].v)
                curr_node = curr_node.child[token]
            else:
                break
        return k_val, v_val, curr_node
    