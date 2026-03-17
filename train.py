from linear_regression import LinearRegression, StandarScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df= pd.read_csv('train.csv',index_col='Id')
test= pd.read_csv('test.csv')
test_x= test.drop(columns=['Id'])
cat_col= df.select_dtypes('object').columns

df= pd.get_dummies(df,columns=cat_col,drop_first=True,dummy_na=True)
test_x= pd.get_dummies(test_x,columns=cat_col,drop_first=True,dummy_na=True)

y= df['SalePrice']
X= df.drop(columns=['SalePrice'])
test_x = test_x.reindex(columns=X.columns, fill_value=0)
num_col= X.select_dtypes('number').columns
X[num_col]= X[num_col].fillna(X[num_col].mean())
test_x[num_col]= test_x[num_col].fillna(test_x[num_col].mean())
X*=1
test_x*=1

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1,random_state=42)
X_train,X_test,y_train,y_test= X_train.to_numpy(),X_test.to_numpy(),y_train.to_numpy(),y_test.to_numpy()

test_x= test_x.to_numpy()

scaler= StandarScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
test_x= scaler.transform(test_x)

model= LinearRegression()
model.fit(X_train,y_train,0.001,10000)
prediction= model.predict(X_test)

print(f"R2 Score: {r2_score(y_test, prediction)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, prediction)}")

prediction2= model.predict(test_x)
submission= pd.DataFrame({
    'Id':test['Id'],'SalePrice':prediction2
})
submission.to_csv('submission.csv',index=False)