#!/usr/bin/env python3
"""Depth of a decision tree"""
import numpy as np


class Node:
    """Node class"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None,
                 is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ find the maximum of the depths of the nodes
        (including the leaves) in a decision tree.
        """
        if self.is_leaf:
            return self.depth
        left_depth = self.left_child.max_depth_below(

        )if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below(
        ) if self.right_child else self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """counts the number of nodes below this node"""
        if self.is_leaf:
            return 1 if only_leaves else 1

        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def __str__(self):
        """String representation of the node"""
        node_repr = f"[feature={self.feature}, threshold={self.threshold}] depth={self.depth}\n"
        if self.left_child:
            node_repr += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            node_repr += self.right_child_add_prefix(self.right_child.__str__())
        return node_repr
                

    def left_child_add_prefix(self, text):
        """Prefixes the text with the left child prefix"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)
    
    def right_child_add_prefix(self, text):
        """Prefixes the text with the right child prefix"""
        lines = text.split("\n")
        new_text = "    \\--" + lines[0] + "\n"
        for x in lines[1:]:
                new_text += ("    |  " + x) + "\n"
        return new_text


class Leaf(Node):
    """Leaf class"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def __str__(self):
        """String representation of the leaf"""
        return (f"-> leaf [value={self.value}] ")


class Decision_Tree():
    """Decision Tree class"""

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None
    ):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Compute the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts all nodes or only leaves in tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """String representation of the tree"""
        return self.root.__str__()
