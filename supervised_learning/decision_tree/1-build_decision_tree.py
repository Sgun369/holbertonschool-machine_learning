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
