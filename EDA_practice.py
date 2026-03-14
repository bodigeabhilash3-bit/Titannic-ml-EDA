#Tatanic Survival prediction 
#EDA +Logistic Regression 
#Accuracy :80.4%
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\HP\Desktop\cse\cse_c\train.csv")
                 
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"]=df["Sex"].map({"male":0,"female":1})
df = df.drop(["Name","Cabin","Ticket","Embarked"],axis=1)
df = df.dropna()
print(df.shape)
print(df["Sex"])
print(df.head())
print(df.isnull().sum())
print(df["Survived"].value_counts())
print(df.groupby("Sex")["Survived"].mean())
print(df.groupby("Pclass")["Survived"].mean())
df["Survived"].value_counts().plot(kind="bar")
plt.title("dead vs survied")
plt.xlabel("dead=0,survived = 1")
plt.ylabel("count")
plt.show()
x = df.drop("Survived",axis = 1)
y = df["Survived"]
X_train,X_test,Y_train,Y_test = train_test_split(
    x,y,random_state=42,test_size=0.2
)
model = LogisticRegression(max_iter=2000)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print("Accuracy:",accuracy_score(Y_test,y_pred))
