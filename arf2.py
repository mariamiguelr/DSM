
#Define a class for the adaptive random forest. The forest should have methods for training on a batch of data,
#  making predictions on a batch of data, and updating its structure based on new data. The forest should maintain a collection of decision trees, 
# and each tree should be trained on a random subset of the data.
from river import tree

class AdaptiveRandomForest:
    def __init__(self, n_trees, tree_size):
        self.n_trees = n_trees
        self.tree_size = tree_size
        self.trees = []

        for i in range(n_trees):
           tree = HoeffdingTreeClassifier()
        self.trees.append(tree)
    

    def train(self, X, y):
        for i in range(self.n_trees):
            indices = np.random.choice(len(X), self.tree_size, replace=False)
            self.trees[i].train(X[indices], y[indices])

    def predict(self, X):
        predictions = np.zeros((len(X),))

        for i in range(self.n_trees):
            predictions += self.trees[i].predict(X)

        return np.round(predictions / self.n_trees)

    def update(self, X, y):
        for i in range(self.n_trees):
            indices = np.random.choice(len(X), self.tree_size, replace=False)
            self.trees[i].update(X[indices], y[indices])

