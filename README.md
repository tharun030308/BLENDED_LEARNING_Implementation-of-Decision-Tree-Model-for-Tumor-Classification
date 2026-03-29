# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset.<br/>
2. Separate the dataset into input features (X) and target variable (y).<br/>
3. Split the dataset into training and testing sets.<br/>
4. Create and train the Decision Tree Classifier model using the training data.<br/>
5. Predict the class labels using the test data.<br/>
6. Evaluate the model using accuracy score and classification report.<br/>
7. Generate the confusion matrix to analyze model performance.<br/>
8. Visualize the confusion matrix using a heatmap.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('tumor.csv')
X = data.drop(columns=['Class'])
y = data['Class'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="620" height="239" alt="image" src="https://github.com/user-attachments/assets/c44b0394-af96-4888-ab02-2a2d403bf594" />
<img width="782" height="600" alt="image" src="https://github.com/user-attachments/assets/26e2f6fe-b039-4bbc-9ef7-5644cb6ba793" />


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
