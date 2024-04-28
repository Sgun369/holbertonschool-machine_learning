#!/usr/bin/env python3
"""Depth of a decision tree"""
import numpy as np


class Node:
    """Node class for representing decision nodes within a tree."""

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0,
        bounds=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.depth = depth
        self.bounds = bounds if bounds is not None else {
            'upper': {}, 'lower': {}}
        self.lower = self.bounds['lower']
        self.upper = self.bounds['upper']

    def max_depth_below(self):
        """find the maximum of the depths of the nodes
        (including the leaves) in a decision tree.
        """
        if self.is_leaf:
            return self.depth
        left_depth = (self.left_child.max_depth_below()
                      if self.left_child else self.depth)
        right_depth = (self.right_child.max_depth_below()
                       if self.right_child else self.depth)
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

    def get_leaves_below(self):
        """Returns a list of all leaf nodes below this Node"""
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
            return leaves

    def __str__(self):
        """Return structured string representation of
        the Node and its children.
        """
        result = (
            f"root [feature={self.feature}, threshold={self.threshold}]"
            if self.is_root
            else
            f"-> node [feature={self.feature}, threshold={self.threshold}]"
        )
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result

    def left_child_add_prefix(self, text):
        """Format left child string representation with branches
        for visual structure.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Format right child string representation
        with branches for visual structure.
        """
        lines = text.split("\n")
        return (
            "    +--"
            + lines[0]
            + "\n"
            + "\n".join("     " + " " * 3 + line for line in lines[1:])
        )

    def update_bounds_below(self):
        """Updates bounds"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = max(
                        child.lower.get(self.feature, -np.inf), self.threshold)
                else:  # right child
                    child.upper[self.feature] = min(
                        child.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Compute the indicator function for
        the node based on the node's bounds"""

        def is_large_enough(x):
            """Returns a boolean"""
            return np.all(np.array([x[:, key] > self.lower[key]
                          for key in self.lower]), axis=0)

        def is_small_enough(x):
            """Returns a boolean array where
            each element is True if the corresponding
            individual's features are less than
            or equal to the upper bounds."""
            return np.all(np.array([x[:, key] <= self.upper[key]
                          for key in self.upper]), axis=0)

        self.indicator = lambda x: np.logical_and(
            is_large_enough(x), is_small_enough(x))

    def pred(self, x):
        """Recursively predict based on feature and threshold"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Leaf class representing terminal nodes of a decision tree."""

    def __init__(self, value, depth=None, bounds=None):
        super().__init__(
            depth=depth,
            bounds=bounds if bounds is not None else {
                'upper': {},
                'lower': {}})
        self.value = value
        self.is_leaf = True

    def max_depth_below(self):
        """Returns the depth of the leaf"""
        return self.depth

    def get_leaves_below(self):
        """Return a list containing jst this leaf"""
        return [self]

    def update_bounds_below(self):
        """Updates bounds """
        pass

    def pred(self, x):
        """Return the leaf's value as the prediction"""
        return self.value

    def __str__(self):
        """Return structured string representation of a Leaf."""
        return f"-> leaf [value={self.value}] "


class Decision_Tree:
    """Decision Tree class managing the overall tree structure."""

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def depth(self):
        """Compute the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Counts all nodes or only leaves in tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Return all leaves in the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Start the bounds update process from the root."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Update tree for prediction"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0)

    def __str__(self):
        """String representation of the entire Decision Tree."""
        return str(self.root)
