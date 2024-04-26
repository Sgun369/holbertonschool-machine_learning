#!/usr/bin/env python3
"""Depth of a decision tree"""
import numpy as np


class Node:
    """Node class for representing decision nodes within a tree."""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth

    def __str__(self):
        """Return structured string representation of the Node and its children."""
        result = f"root [feature={self.feature}, threshold={self.threshold}]" if self.is_root else f"-> node [feature={self.feature}, threshold={self.threshold}]"
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        """Format left child string representation with branches for visual structure."""
        lines=text.split("\n")
        new_text="    +--"+lines[0]+"\n"
        for x in lines[1:-1] :
            new_text+=("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Format right child string representation with branches for visual structure."""
        lines = text.split("\n")
        return "    +--" + lines[0] + "\n" + "\n".join("     " + " " * 3 + line for line in lines[1:])

class Leaf(Node):
    """Leaf class representing terminal nodes of a decision tree."""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return structured string representation of a Leaf."""
        return f"-> leaf [value={self.value}] "

class Decision_Tree:
    """Decision Tree class managing the overall tree structure."""
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def __str__(self):
        """String representation of the entire Decision Tree."""
        return str(self.root)
