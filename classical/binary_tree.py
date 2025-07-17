# This module builds includes functions for building a binary tree that encode the values necessary for transforming |0>^n into any vector |v>.
from classical.calc_weights import calc_weight_vec
import numpy as np

class Node:
    """A node in the binary tree."""
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.level = None
        self.position = None # position in level
        self.edge_from_parent = None # 0 for left, 1 for right
        self.path = [] # path from root to this node

class BinaryTree:
    """A full binary tree to store values."""
    def __init__(self):
        self.depth = 0
        self.levels = []  # Store nodes by level for O(1) access

    def build_tree(self, vector):
        """Build the binary tree from a vector.
        
        Args:
            vector (np.ndarray): The vector to build the tree from, expecting it to be of dimension 2^n."""

        depth = int(np.log2(len(vector)))
        self.depth = depth
        self.levels = [[] for _ in range(depth + 1)] # intialise levels

        # populate leaf nodes with values first
        for i, value in enumerate(vector):
            node = Node(np.abs(value)**2) # probability of measurement
            node.level = depth
            node.position = i
            self.levels[depth].append(node)
        
        for level in range(depth-1, -1, -1):
            for j in range(2**(level)):
                left_child = self.levels[level + 1][2 * j]
                right_child = self.levels[level + 1][2 * j + 1]
                left_value = left_child.value
                right_value = right_child.value
                node = Node(left_child.value + right_child.value)
                node.left = left_value
                node.right = right_value
                node.level = level
                node.position = j
                self.levels[level].append(node)

        for level in range(1, depth + 1):
            for j, node in enumerate(self.levels[level]):
                if j % 2 == 0:
                    node.edge_to_parent = 0
                else:
                    node.edge_to_parent = 1
                node.path = self.levels[level - 1][j // 2].path + [node.edge_to_parent]
        
        
        