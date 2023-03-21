#Adaptive Random Forest (ARF) is an ensemble machine learning algorithm that combines Random Forest
# (RF) with the concept of concept drift. In ARF, the decision trees are grown dynamically, and the older 
# trees are replaced with the new ones as the data distribution changes. In this implementation, we will 
# be using the River library, which is a lightweight and efficient Python library for online machine learning.

#Here's how you can implement ARF using River in Python without using any external libraries:

import random
from river import tree

class ARF:
    def __init__(self, n_trees=10, max_size=1000, grace_period=50):
        self.n_trees = n_trees
        self.max_size = max_size
        self.grace_period = grace_period
        self.forest = [tree.HoeffdingTreeClassifier() for _ in range(n_trees)]
        self.buffer = []
        self.weights = [1.0 / n_trees] * n_trees

    def fit(self, x, y):
        self.buffer.append((x, y))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

        for i, tree in enumerate(self.forest):
            tree.fit_one(x, y, sample_weight=self.weights[i])

        if len(self.buffer) % self.grace_period == 0:
            self.adapt()

    def adapt(self):
        new_forest = []
        for i in range(self.n_trees):
            tree_idx = self.choose_tree()
            new_tree = tree.HoeffdingTreeClassifier()
            new_tree.learn_from(self.forest[tree_idx].root)
            new_forest.append(new_tree)

        self.forest = new_forest
        self.weights = [1.0 / self.n_trees] * self.n_trees

    def choose_tree(self):
        return random.choices(range(self.n_trees), weights=self.weights)[0]

    def predict_proba(self, x):
        probas = [tree.predict_proba_one(x) for tree in self.forest]
        return sum(probas) / len(probas)





