# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Function to plot confusion matrix
def plot_confusion_matrix(y, y_predict, normalize=False):
    cm = confusion_matrix(y, y_predict)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.2f' if normalize else 'd')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['did not land', 'landed'])
    ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.show()

# Function to print classification report and plot confusion matrix
def evaluate_model(y_true, y_pred, model_name):
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=['did not land', 'landed']))
    plot_confusion_matrix(y_true, y_pred)

# Load the data
URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
data = pd.read_csv(URL1)
X = pd.read_csv(URL2)

# Check for missing values
print("Missing values in dataset:", data.isnull().sum().sum(), X.isnull().sum().sum())

# Task 1: Create NumPy array from 'Class' column
Y = data["Class"].to_numpy()

# Task 2: Standardize the data
transform = preprocessing.StandardScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
X_train = transform.fit_transform(X_train)
X_test = transform.transform(X_test)

# Logistic Regression
parameters = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10, scoring="accuracy")
logreg_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for Logistic Regression
print("Logistic Regression best parameters:", logreg_cv.best_params_)
print("Logistic Regression best accuracy:", logreg_cv.best_score_)

# Evaluate Logistic Regression
yhat_logreg = logreg_cv.predict(X_test)
evaluate_model(Y_test, yhat_logreg, "Logistic Regression")

# Support Vector Machine
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': np.logspace(-3, 3, 5), 'gamma': np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(estimator=svm, param_grid=parameters, cv=10, scoring="accuracy")
svm_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for SVM
print("SVM best parameters:", svm_cv.best_params_)
print("SVM best accuracy:", svm_cv.best_score_)

# Evaluate SVM
yhat_svm = svm_cv.predict(X_test)
evaluate_model(Y_test, yhat_svm, "Support Vector Machine")

# Decision Tree
parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2*n for n in range(1, 10)], 'max_features': ['auto', 'sqrt'], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10, scoring="accuracy")
tree_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for Decision Tree
print("Decision Tree best parameters:", tree_cv.best_params_)
print("Decision Tree best accuracy:", tree_cv.best_score_)

# Evaluate Decision Tree
yhat_tree = tree_cv.predict(X_test)
evaluate_model(Y_test, yhat_tree, "Decision Tree")

# K-Nearest Neighbors
parameters = {'n_neighbors': list(range(1, 11)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(estimator=knn, param_grid=parameters, cv=10, scoring="accuracy")
knn_cv.fit(X_train, Y_train)

# Output best parameters and accuracy for KNN
print("KNN best parameters:", knn_cv.best_params_)
print("KNN best accuracy:", knn_cv.best_score_)

# Evaluate KNN
yhat_knn = knn_cv.predict(X_test)
evaluate_model(Y_test, yhat_knn, "K-Nearest Neighbors")

# Compare models
model_grid_searches = [logreg_cv, svm_cv, tree_cv, knn_cv]
models = ["Logistic Regression", "Support Vector Machine", "Decision Tree", "K-Nearest Neighbors"]
scores = [logreg_cv.best_score_, svm_cv.best_score_, tree_cv.best_score_, knn_cv.best_score_]
best_model_index = np.argmax(scores)
best_model = models[best_model_index]
best_score = scores[best_model_index]

# Display results
print(f"Best Model: {best_model} with score {best_score}")

score_df = pd.DataFrame({"Model": models, "Score": scores})

# Plot accuracy scores
sns.barplot(data=score_df, x="Model", y="Score")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.ylabel("Scores")
plt.title("Accuracy scores of Machine Learning Models")
plt.show()
