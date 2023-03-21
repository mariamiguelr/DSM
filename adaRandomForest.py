from river import datasets
from river import ensemble
from river import metrics
from river import tree
import random
from river.metrics import Accuracy
from river.model_selection import train_test_split
import pandas as pd
from skmultiflow.trees import HoeffdingTreeClassifier
from river.metrics import Accuracy, Precision, Recall, F1Score, AUC, LogLoss

# Set parameters
n_trees = 10
tree_size = 100
window_size = 1000

####################################### TREINO E TESTE ###########################################
dataset = pd.read_csv("river_dataset.csv")
X_y = [(x, y) for x, y in dataset]
X_train, X_test, y_train, y_test = train_test_split(X_y, test_size=0.3)


####################################### MODELO #################################################


AdaptiveRandomForestClassifier(
     
     
)


# Create the adaptive random forest
model = AdaptiveRandomForestClassifier(
    n_models=n_trees,
    ht = HoeffdingTreeClassifier(
        grace_period=tree_size,
        split_criterion='gini'
    ),
    window_size=window_size
)

model.train(X_train, y_train)


######################## METRICAS ################################

model = LogisticRegression()
metrics = {
    'accuracy': Accuracy(),
    'precision': Precision(),
    'recall': Recall(),
    'f1score': F1Score(),
    'auc': AUC(),
    'logloss': LogLoss()
}

for x, y in X_train:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    for metric in metrics.values():
        metric.update(y, y_pred)

for name, metric in metrics.items():
    score = metric.get()
    print(f"{name.capitalize()}: {score:.4f}")
