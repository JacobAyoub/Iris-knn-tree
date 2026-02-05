from collections import Counter
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from sklearn.tree import DecisionTreeClassifier, plot_tree

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


def load_iris_dataset(url=IRIS_URL):
    raw = urlopen(url).read().decode("utf-8").strip().splitlines()
    data = [line.split(",") for line in raw if line]
    features = np.array([[float(v) for v in row[:4]] for row in data])
    labels = np.array([row[4] for row in data])
    return features, labels


def train_test_split(X, y, train_size=120, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def knn_predict(kd_tree, y_train, X_query, k=5):
    #print(X_query)
    points, indices = kd_tree.query(X_query, k=k) #Find k nearest neighbors
    #print(points, y_train[indices])
    classification = Counter(y_train[indices]).most_common(1) #Finds majority label
    #print(classification)
    return classification[0][0]


def evaluate_knn(X_train, y_train, X_test, y_test, k=5):
    kd_tree = KDTree(X_train)
    success = 0
    for i in range(len(X_test)):
        prediction = knn_predict(kd_tree, y_train, X_test[i]) #Get prediction for each test sample
        if (prediction == y_test[i]): #Test if prediction is correct
            success += 1
    accuracy = success / len(X_test) #Calculate accuracy
    return accuracy
    """
    kd_tree = KDTree(X_train)
    prediction = knn_predict(kd_tree, y_train, X_test[:1])
    print(prediction, y_test[0])
    if (prediction == y_test[0]):
        print("success")
    """


def evaluate_decision_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train) #Train decision tree
    predictions = clf.predict(X_test) #Get predictions for all test samples
    #print(predictions)
    success = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]: #Test if each prediction is correct
            success += 1
    accuracy = success / len(y_test) #Calculate accuracy
    return accuracy


def visualize_decision_tree(X_train, y_train, output_path="decision_tree.png", show=False):
    """Train a decision tree on the training split and export a labeled diagram."""
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plot_tree(
        clf,
        feature_names=["sepal length", "sepal width", "petal length", "petal width"],
        class_names=clf.classes_.tolist(),
        filled=True,
        rounded=True,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def main():
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    results = [
        ("k-NN (k=5)", evaluate_knn(X_train, y_train, X_test, y_test, k=5)),
        ("Decision tree", evaluate_decision_tree(X_train, y_train, X_test, y_test)),
    ]

    for name, accuracy in results:
        print(f"{name} test accuracy: {accuracy:.3f}")

    visualize_decision_tree(X_train, y_train, output_path="decision_tree.png")


if __name__ == "__main__":
    main()

"""
Summary of Iris Decision Tree:
-The root splits on petal width <= 0.8, which was the best split available when looking at the dataset.
-For the wider flowers, the tree splits based on petal length <= 4.95, separating most of the Versicolor and Virginica labels.
-The left subtree usually led to Iris-Versicolor, while the right subtree mostly led to Iris-Virginica.
-The tree has a depth of 4, meaning there is a maximum of 4 decisions from the root to a leaf node.
-There is 9 decision nodes and 7 leaf nodes in the tree.
"""