import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import multiprocessing as mp

warnings.filterwarnings("ignore")

# Ensure the dataset file exists or create one
dataset_path = r'C:\Users\prath\Downloads\IDS_dataset.csv'

if not os.path.exists(dataset_path):
    np.random.seed(42)
    num_samples = 1000
    num_features = 13
    data = np.random.rand(num_samples, num_features)
    labels = np.random.choice(["Normal", "Attack"], num_samples)
    
    df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(num_features)])
    df["Label"] = labels
    df.to_csv(dataset_path, index=False)
    print(f"Dataset created at {dataset_path}")

# Load dataset
dataset = pd.read_csv(dataset_path)
X = dataset.iloc[:, :-1].values  # All feature columns
y = dataset.iloc[:, -1].values   # Target column

# Encode categorical features
def Multilabelencoder(X, k):
    X[:, k] = LabelEncoder().fit_transform(X[:, k])
    return X

for i in range(min(3, X.shape[1])):  # Encode up to 3 features if applicable
    X = Multilabelencoder(X, i)

# Encode target variable
y = LabelEncoder().fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Apply PCA (for SVM models)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define classifiers
classifiers = [
    ("Logistic Regression", LogisticRegression(random_state=0)),
    ("Random Forest (10 Trees)", RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)),
    ("SVM - Linear", SVC(kernel='linear', random_state=0)),
    ("SVM - Poly", SVC(kernel='poly', random_state=0)),
    ("SVM - RBF", SVC(kernel='rbf', random_state=0)),
    ("SVM - Sigmoid", SVC(kernel='sigmoid', random_state=0)),
    ("Decision Tree", DecisionTreeClassifier(criterion='entropy', random_state=0)),
    ("Naive Bayes", GaussianNB()),
    ("Random Forest (20 Trees)", RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0))
]

# Train classifiers
e_models = []
for name, clf in classifiers:
    if "SVM" in name:
        clf.fit(X_train_pca, y_train)
    else:
        clf.fit(X_train, y_train)
    e_models.append((name, clf))

def evaluate_model(index):
    name, model = e_models[index]
    X_input = X_test_pca if "SVM" in name else X_test  # Ensure correct input size
    y_pred = model.predict(X_input)
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum() * 100
    precision = cm[0][0] / (cm[0][0] + cm[1][0]) * 100 if cm[0][0] + cm[1][0] != 0 else 0
    recall = cm[0][0] / (cm[0][0] + cm[0][1]) * 100 if cm[0][0] + cm[0][1] != 0 else 0
    print(f"Accuracy of {name}: {accuracy:.2f}%")
    print(f"Precision of {name}: {precision:.2f}%")
    print(f"Recall of {name}: {recall:.2f}%")

def master():
    while True:
        choice = int(input("""
Welcome to Network Intrusion Detection System
Choose an option:
0. Logistic Regression
1. Random Forest (10 Trees)
2. SVM - Linear
3. SVM - Poly
4. SVM - RBF
5. SVM - Sigmoid
6. Decision Tree
7. Naive Bayes
8. Random Forest (20 Trees)
9. Run all algorithms
10. Quit
"""))
        if choice in range(len(e_models)):
            evaluate_model(choice)
        elif choice == 9:
            run_all()
        elif choice == 10:
            break
        else:
            print("Invalid choice. Try again.")

def run_all():
    print("Running all models in parallel...")
    with mp.Pool(len(e_models)) as pool:
        pool.map(evaluate_model, range(len(e_models)))

if __name__ == "__main__":
    master()
