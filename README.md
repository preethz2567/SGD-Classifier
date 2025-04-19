# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**  
   Import `pandas`, `numpy`, `load_iris`, `SGDClassifier`, `train_test_split`, and evaluation metrics from `sklearn`.

2. **Load Dataset**  
   Load the Iris dataset using `load_iris()`.

3. **Create DataFrame**  
   Convert the data into a pandas DataFrame and add the target labels as a new column.

4. **Inspect Data**  
   Use `df.head()` to preview the first few rows.

5. **Define Features and Target**  
   Set `x` as all feature columns and `y` as the `target` column.

6. **Split the Data**  
   Use `train_test_split()` to split into training and testing sets (80/20 split).

7. **Initialize Classifier**  
   Create an `SGDClassifier` with `max_iter=1000` and `tol=1e-3`.

8. **Train the Model**  
   Fit the classifier on the training data.

9. **Make Predictions**  
   Predict target values for the test set.

10. **Evaluate the Model**  
   Use `accuracy_score`, `confusion_matrix`, and `classification_report` to assess performance.

## Program and Output:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: PREETHI D 
RegisterNumber:  212224040250
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
```
```
iris  = load_iris()
```
```
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/8cc28bf3-3e70-47b4-896b-6e482b7f0dbc)

```
x = df.drop('target', axis=1)
y = df['target']
```
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
```
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(x_train, y_train)
```
![image](https://github.com/user-attachments/assets/a4a67caf-1cf9-47a3-8d4c-f2e33828d263)
```
y_pred = sgd_clf.predict(x_test)
```
```
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
![image](https://github.com/user-attachments/assets/4512cf53-6608-411a-bb3e-fb4c627caf7d)
```
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
![image](https://github.com/user-attachments/assets/2a68d350-9bd2-4f25-b94d-a674ec2e7716)

```
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
```
![image](https://github.com/user-attachments/assets/a36d15e9-6a52-479c-abb9-cf41233455a5)




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
