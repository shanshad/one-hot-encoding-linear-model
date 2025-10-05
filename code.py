import pandas as pd
from sklearn.linear_model import LinearRegression
model=LinearRegression()
df=pd.read_csv(r"C:\Users\LENOVO\Desktop\Datasets for machine lerning\carprices.csv")
dummies=pd.get_dummies(df['Car Model'])
merged=pd.concat([df,dummies],axis='columns')
final=merged.drop(['Mercedez Benz C class','Car Model'],axis='columns')
X=final.drop('Sell Price($)',axis='columns')
Y=final['Sell Price($)']
model.fit(X,Y)
print(X)
print(model.predict([[45000,4,0,0]]))
print(model.predict([[86000,7,0,1]]))
print(model.score(X,Y))