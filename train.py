import pandas as pd
df=pd.read_csv('train.csv')

df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1 , inplace=True)

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())


# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])





df['Sex']=df['Sex'].map({'male': 0,'female':1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print(df.head())
print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model= LogisticRegression(max_iter=(1000))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
 

accuracy=accuracy_score(y_test,y_pred)
print("logistic regrtesion accuracy:",accuracy)


##Now check the accuracy with random forest


from sklearn.ensemble import RandomForestClassifier

# Create and train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_acc)
##now try it with mlp

from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Predict and evaluate
mlp_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
print("MLP Classifier Accuracy:", mlp_acc)