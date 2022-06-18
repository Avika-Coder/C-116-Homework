from google.colab import files
data_to_load=files.upload()

import pandas as pd
import csv
import plotly.express as px
import statistics

df=pd.read_csv("Admission_Predict.csv")
TOEFL_Score=df["TOEFL Score"].tolist()
GRE_Score=df["GRE Score"].tolist()
graph1=px.scatter(x=GRE_Score,y=TOEFL_Score)
graph1.show()

import plotly.graph_objects as go
Chance_of_admit=df["Chance of admit"].tolist()
TOEFL_Score=df["TOEFL Score"].tolist()
GRE_Score=df["GRE Score"].tolist()
colors=[]
for data in Chance_of_admit :
  if data == 1 :
    colors.append("green")
  else :
    colors.append("red")
graph2=go.Figure(data=go.Scatter(x=TOEFL_Score,y=GRE_Score,mode='markers',marker=dict(color=colors)))
graph2.show()

score=df[["TOEFL Score","GRE Score"]]
Chance_of_admit=df["Chance of admit"]
from sklearn.model_selection import train_test_split
TOEFL_Score_train,TOEFL_Score_test,Chance_of_admit_train,Chance_of_admit_test=train_test_split(score,Chance_of_admit,test_size=0.25,random_state=0)
print(TOEFL_Score_train[0:10])
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
TOEFL_Score_train=sc_x.fit_transform(TOEFL_Score_train)
TOEFL_Score_test=sc_x.transform(TOEFL_Score_test)
print(TOEFL_Score_train[0:10])
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(TOEFL_Score_train,Chance_of_admit_train)
Chance_of_admit_prediction=classifier.predict(TOEFL_Score_test)
from sklearn.metrics import accuracy_score
print("accuracy=",accuracy_score(Chance_of_admit_test,Chance_of_admit_prediction))
