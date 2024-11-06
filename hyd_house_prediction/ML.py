import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRFRegressor
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor
import pickle

#Loadin the Dataset
df = pd.read_csv(r"C:\Users\jayas\OneDrive\Desktop\New folder\hyd_house_prediction\updated_land_data.csv")

#Splitting the dataset into depedent and indepedent variable
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

le = LabelEncoder()
for col in x.select_dtypes(include='object').columns:
    x[col] = le.fit_transform(x[col])
    
st = StandardScaler()
x = st.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0,test_size=0.2)

lasso_param_grid = {
    'alpha': [0.1, 0.5, 1, 5, 10]
}

ls = GridSearchCV(Lasso(),lasso_param_grid, cv=5, scoring='r2', n_jobs=-1)
ls.fit(x_train,y_train)
y_pred_lasso = ls.predict(x_test)
acc_lass = r2_score(y_test,y_pred_lasso)
mse_lasso = mean_squared_error(y_test,y_pred_lasso)

ridge_param_grid = {
    'alpha': [0.1, 0.5, 1, 5, 10]
}

rd = GridSearchCV(Ridge(),ridge_param_grid, cv=5, scoring='r2', n_jobs=-1)
rd.fit(x_train,y_train)
y_pred_rid = rd.predict(x_test)
acc_rid = r2_score(y_test,y_pred_rid)
mse_rid = mean_squared_error(y_test,y_pred_rid)

poly = PolynomialFeatures(degree = 2)
x_trian_p = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

reg = LinearRegression()
reg.fit(x_trian_p, y_train)
y_pred = reg.predict(x_test_poly)
acc = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

knn = KNeighborsRegressor(n_neighbors = 4,weights="distance",p = 1,algorithm="ball_tree")
knn.fit(x_train,y_train)
y_pred1 = knn.predict(x_test)
acc1 = r2_score(y_test,y_pred1)
mse1 = mean_squared_error(y_test,y_pred1)

dt = DecisionTreeRegressor(criterion = "friedman_mse",random_state = 0,splitter = "best")
dt.fit(x_train,y_train)
y_pred2 = dt.predict(x_test)
acc2 = r2_score(y_test,y_pred2)
mse2 = mean_squared_error(y_test,y_pred2)

xgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9]
}

xgb = GridSearchCV(XGBRFRegressor(),xgb_param_grid,cv=5, scoring='r2', n_jobs=-1)
xgb.fit(x_train,y_train)
y_pred3 = xgb.predict(x_test)
acc3 = r2_score(y_test,y_pred3)
mse3 = mean_squared_error(y_test,y_pred3)

lgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 10],
    'num_leaves': [31, 50, 100]
}

lgb = GridSearchCV(lgb.LGBMRegressor(),lgb_param_grid,cv=5, scoring='r2', n_jobs=-1)
lgb.fit(x_train,y_train)
y_pred4 = lgb.predict(x_test)
acc4 = r2_score(y_test,y_pred4)
mse4 = mean_squared_error(y_test,y_pred4)

gbr = GradientBoostingRegressor(learning_rate=0.1,min_samples_split=2,max_depth=3,n_estimators=50,random_state=0,max_leaf_nodes=2)
gbr.fit(x_train,y_train)
y_pred5 = gbr.predict(x_test)
acc5 = r2_score(y_test,y_pred5)
mse5 = mean_squared_error(y_test,y_pred5)

rnd = RandomForestRegressor(n_estimators=50,max_depth=4,min_samples_leaf=1)
rnd.fit(x_train,y_train)
y_pred6 = rnd.predict(x_test)
acc6 = r2_score(y_test,y_pred6)
mse6 = mean_squared_error(y_test,y_pred6)

ada = AdaBoostRegressor(learning_rate=0.1,random_state=0,loss = "linear")
ada.fit(x_train,y_train)
y_pred7 = ada.predict(x_test)
acc7 = r2_score(y_test,y_pred7)
mse7 = mean_squared_error(y_test,y_pred7)

results = {
    "Lasso Regression": {"R2": acc_lass, "MSE": mse_lasso},
    "Ridge Regression": {"R2": acc_rid, "MSE": mse_rid},
    "Polynomial Regression": {"R2": acc, "MSE": mse},
    "K-Nearest Neighbors": {"R2": acc1, "MSE": mse1},
    "Decision Tree Regressor": {"R2": acc2, "MSE": mse2},
    "XGBRF Regressor": {"R2": acc3, "MSE": mse3},
    "LightGBM Regressor": {"R2": acc4, "MSE": mse4},
    "Gradient Boosting Regressor": {"R2": acc5, "MSE": mse5},
    "Random Forest Regressor": {"R2": acc6, "MSE": mse6},
    "AdaBoost Regressor": {"R2": acc7, "MSE": mse7}
}

results_df = pd.DataFrame(results).T
print("Model Comparison Results:")
print(results_df)

#Finding the Best fit model
best_model_r2 = results_df['R2'].idxmax()  
best_model_mse = results_df['MSE'].idxmin()

print("\nBest model based on R2 score:", best_model_r2)
print("Best model based on MSE score:", best_model_mse)

#Pickle the File
filename = "House_Model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(knn,file)
print("Model Created")

cv_scores = cross_val_score(reg, x, y, cv=5, scoring='r2')

