#!/usr/bin/env python3
"""Module to implement a decision tree with node depth calculations and formatted output."""

import numpy as np

class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth among the node's children."""
        if self.is_leaf:
            return self.depth
        left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
        right_depth = self.right_child.max_depth_below() if self.right_child else self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count the total or leaf-only nodes below this node."""
        if self.is_leaf:
            return 1
        count = 1 if not only_leaves else 0
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def __str__(self):
        node_label = ("root" if self.is_root else "node") + f" [feature={self.feature}, threshold={self.threshold}]"
        parts = [node_label]
        if self.left_child:
            parts.append(self.left_child_add_prefix(str(self.left_child)))
        if self.right_child:
            parts.append(self.right_child_add_prefix(str(self.right_child)))
        return "\n".join(parts)

    def left_child_add_prefix(self, text):
        """Format left child string representation with indentation."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |     " + x + "\n"
        return new_text.rstrip()

    def right_child_add_prefix(self, text):
        """Format right child string representation with indentation."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "          " + x + "\n"
        return new_text.rstrip()

class Leaf(Node):
    """Represents a leaf in the decision tree."""
    def __init__(self, value, depth=0):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of the leaf."""
        return self.depth

    def __str__(self):
        return f" leaf [value={self.value}]"

class Decision_Tree:
    """Implements the decision tree."""
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """Compute the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count all nodes or only leaves in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the entire tree."""
        return str(self.root)
