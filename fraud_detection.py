import pandas as pd
from sklearn.model_selection import train_test_split #splits the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier #works by building many decision trees and combining their transactions
from sklearn.metrics import classification_report #help us to evaluate how good the model is

# Load dataset
data = pd.read_csv("creditcard.csv")

# Features and label
X = data.drop("fraud", axis=1) # removes the "fraud" coulmn from the dataset and assigns the remaining columns to X, which will be used as features for training the model.
y = data["fraud"]  #selects the fraud column only

# Split data
X_train, X_test, y_train, y_test = train_test_split( #20% testing and 80% training
    X, y, test_size=0.2, random_state=42
)  

# Train model
model = RandomForestClassifier() #think of it is an empty machine learning model it has not learned anything yet
model.fit(X_train, y_train)  #random forest builds many decision trees to learn these patterns

# Predict
predictions = model.predict(X_test) # model sees new transaction it has never seen before

# Evaluate
print("\nModel Evaluation:")
print(classification_report(y_test, predictions)) #this tells how good the model is 