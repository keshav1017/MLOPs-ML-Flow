import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
import dagshub


dagshub.init(repo_owner='keshav1017', repo_name='MLOPs-ML-Flow', mlflow=True)

# Load wine dataset
wine = load_wine()
x = wine.data  
y = wine.target

# train and test split 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.10, random_state = 42)

# define the params for RF model
max_depth = 10
n_estimators = 5

mlflow.set_tracking_uri("https://dagshub.com/keshav1017/MLOPs-ML-Flow.mlflow")

# mention your experiment below
mlflow.set_experiment("MLOPs Exp1")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, random_state = 42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    # creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (6, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = "Blues", xticklabels = wine.target_names, yticklabels = wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # save the plot
    plt.savefig("Confusion-Matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-Matrix.png")
    mlflow.log_artifact(__file__)

    # set tags
    mlflow.set_tags({"Author": "Keshav", "Project": "Wine Classfication"})

    # log model
    mlflow.sklearn.log_model(rf, "Random-Forest-Classifier")

    print(accuracy)