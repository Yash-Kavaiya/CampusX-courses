
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb



df = pd.read_excel ("C:\\Users\\rohu1\\Desktop\\depv1\\EPS_Dataset.xlsx",'v8')
df = df. drop( ['Bank name', 'Year'] , axis=1)


# Data cleaning

# Outliers detection
col_list = df.columns 
for i in col_list:
    q1 = np.percentile(df[i] , 25)
    q3 = np.percentile(df[i] , 75)
    iqr = q3-q1
    lower_bound = q1 - (5 * iqr)
    upper_bound = q3 + (5 * iqr)
   
    df = df.loc [ (df[i] >= lower_bound) & (df[i] <= upper_bound) ]
   
   



# Extract x and y
y = df ['Basic EPS (Rs.)']
x = df. drop( ['Basic EPS (Rs.)'] , axis=1)





# VIF for numerical columns

from statsmodels.stats.outliers_influence import variance_inflation_factor


numeric_columns = x.columns
vif_data = x
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0


for i in range (0,total_columns):
   
    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,vif_value)
   
   
    if vif_value <= 280:
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1
   
    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)




x = x [columns_to_be_kept]



# Apply StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x = x_scaled




# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)




# Define the hyperparameter grid
param_grid = {
  'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
  'learning_rate'   : [0.001, 0.01, 0.1, 1],
  'max_depth'       : [3, 5, 8, 10],
  'alpha'           : [1, 10, 100],
  'n_estimators'    : [10,50,100]
}




index = 0




answers_grid = {
    'combination'       :[],
    'train_RMSE'        :[],
    'test_RMSE'         :[],
    'train_R2'          :[],
    'test_R2'           :[],
    'train_std_diff'    :[],
    'test_std_diff'     :[],
    'colsample_bytree'  :[],
    'learning_rate'     :[],
    'max_depth'         :[],
    'alpha'             :[],
    'n_estimators'      :[]

    }





# Loop through each combination of hyperparameters
for colsample_bytree in param_grid['colsample_bytree']:
  for learning_rate in param_grid['learning_rate']:
    for max_depth in param_grid['max_depth']:
      for alpha in param_grid['alpha']:
          for n_estimators in param_grid['n_estimators']:
             
              index = index + 1
             
              # Define and train the XGBoost model
              model = xgb.XGBRegressor(objective='reg:squarederror',
                                      colsample_bytree=colsample_bytree,
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,
                                      alpha=alpha,
                                      n_estimators=n_estimators)
               
       
     


              model.fit(x_train, y_train)
       
              # Predict on training and testing sets
              y_pred_train = model.predict(x_train)
              y_pred_test = model.predict(x_test)
       
       
              # Calculate train and test results
              train_rmse = np.sqrt (mean_squared_error(y_train, y_pred_train))
              test_rmse= np.sqrt (mean_squared_error(y_test, y_pred_test))
              train_r2 = r2_score(y_train, y_pred_train)
              test_r2  = r2_score(y_test, y_pred_test)
              train_std_diff = train_rmse  / np.std(y_train)
              test_std_diff = test_rmse / np.std(y_test)
       
       
              # Include into the lists
              answers_grid ['combination']   .append(index)
              answers_grid ['train_RMSE']    .append(train_rmse)
              answers_grid ['test_RMSE']     .append(test_rmse)
              answers_grid ['train_R2']      .append(train_r2)
              answers_grid ['test_R2']       .append(test_r2)
              answers_grid ['train_std_diff'].append(train_std_diff)
              answers_grid ['test_std_diff'] .append(test_std_diff)
              answers_grid ['colsample_bytree']   .append(colsample_bytree)
              answers_grid ['learning_rate']      .append(learning_rate)
              answers_grid ['max_depth']          .append(max_depth)
              answers_grid ['alpha']              .append(alpha)
              answers_grid ['n_estimators']       .append(n_estimators)
       
       
              # Print results for this combination
              print(f"Combination {index}")
              print(f"colsample_bytree: {colsample_bytree}, learning_rate: {learning_rate}, max_depth: {max_depth}, alpha: {alpha}, n_estimators: {n_estimators}")
              print(f"Train RMSE: {train_rmse:.2f}")
              print(f"Test RMSE : {test_rmse:.2f}")
              print(f"Train R2  : {train_r2:.2f}")
              print(f"Test R2   : {test_r2:.2f}")
              print(f"Train std_diff: {train_std_diff:.2f}")
              print(f"Test std_diff : {test_std_diff:.2f}")
              print("-" * 30)
       
       
       
       

answers_grid_df = pd.DataFrame(answers_grid)
answers_grid_df .to_excel ('C:\\Users\\rohu1\\Desktop\\9K.xlsx', index=False)


# Getting best results with these-
# colsample_bytree  0.7
# learning_rate     1.0
# max_depth         8
# alpha             10.0
# n_estimators      10



# Retrain on the new parameters
model = xgb.XGBRegressor(objective='reg:squarederror',
                         colsample_bytree = 0.7,
                         learning_rate    = 1.0,
                         max_depth        = 8,
                         alpha            = 10.0,
                         n_estimators     = 10)
   

model.fit(x_train, y_train)




# Predicting on the testing set
y_pred = model.predict(x_test)


# Calculate loss metrics
print()
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)
stddev = y_test.std()
print('Stddev difference', rmse/stddev)








# save the model
import pickle
filename = 'eps_v1.sav'
pickle.dump(model, open(filename,'wb'))


load_model = pickle.load(open(filename,'rb'))


arg = x_train[:2]
load_model.predict(arg)














# ROCE (%)
# 1.91


# CASA (%)
# 39.47


# Return on Equity / Networth (%)
# 14.36


# Non-Interest Income/Total Assets (%)
# 0.68


# Operating Profit/Total Assets (%)
# 0.27


# Operating Expenses/Total Assets (%)
# 1.68

# Interest Expenses/Total Assets (%)
# 3.3


# Face_value
# 2



# Basic EPS (Rs.)
# 27.28






