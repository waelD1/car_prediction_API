import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import pickle

#import the data
data = pd.read_csv('bmw_pricing_challenge.csv')

# Drop useless features
data.drop(['maker_key', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8'], axis=1, inplace=True)

#Create the features that indicates the time between registration and sale
data["registration_date"]= pd.to_datetime(data["registration_date"])
data["sold_at"]= pd.to_datetime(data["sold_at"])
data['time_on_sale'] = (data['sold_at'] - data['registration_date']).dt.days


#Encoding features
labelencoder = preprocessing.LabelEncoder()

# Assigning numerical values and storing in another column
data['model_key_cat'] = labelencoder.fit_transform(data['model_key'])

# Save the mapping to use it into the dropdown lists
le_model_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
#with open('mapping_model.pkl', 'wb') as f:
#  pickle.dump(le_model_name_mapping, f)'''

# Create lists of unique elements 
list_color = data['paint_color'].unique()
list_car_type = data['car_type'].unique()
list_fuel = data['fuel'].unique()

dict_lists = {
    'list_color' : list_color, 
    'list_car_type' : list_car_type, 
    'list_fuel' : list_fuel
    }

#Save the lists to use it into the dropdown lists
# for keys, values in dict_lists.items():
#   with open(f'{keys}.pkl', 'wb') as f:
#     pickle.dump(values, f)

#Encoding categorical variables
fuel = pd.get_dummies(data['fuel'], prefix='fuel')
color = pd.get_dummies(data['paint_color'], prefix='color')
car_type = pd.get_dummies(data['car_type'], prefix='car_type')

# Concatenate the new variables with the other variables
train_data = pd.concat([data, fuel, color, car_type], axis = 1)
train_data.drop(['model_key', 'registration_date', 'fuel', 'paint_color', 'car_type', 'sold_at'], axis = 1, inplace = True)

# Finds correlation between independent and dependent features
plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(), annot = True, cmap = "RdYlGn")
plt.show()

# Important feature using ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(train_data.loc[:, train_data.columns != 'price'], train_data['price'])

#plot graph of feature importances for better visualization
plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=train_data.loc[:, train_data.columns != 'price'].columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(train_data.loc[:, train_data.columns != 'price'], train_data['price'], test_size=0.1, random_state = 42, shuffle = True)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


#XGBoost hyper-parameter tuning
def hyperParameterTuning(X_train, Y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 10.0, 50.0],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           cv = 5,
                           n_jobs = -1,
                           verbose = 30)
    

    gsearch.fit(X_train,Y_train, verbose=True)

    return gsearch.best_params_

hyperParameterTuning(X_train, Y_train)

# Selecting the best parameters
xgb_model = XGBRegressor(
 learning_rate = 0.1,
 reg_alpha = 0.01,
 gamma = 0,
 reg_lambda = 40,
 min_child_weight= 1,
 n_estimators= 100,
 colsample_bytree= 0.5,
 max_depth = 3,
 objective= 'reg:squarederror',
 subsample= 0.7)

# Model training
xgb_model.fit(X_train, Y_train, early_stopping_rounds=5, eval_set=[(X_val, Y_val)], verbose=False)

# Model Evaluation 
y_pred_xgb = xgb_model.predict(X_val)
mae_xgb = mean_absolute_error(Y_val, y_pred_xgb)
print("MAE: ", mae_xgb)
print('SCORE' , xgb_model.score(X_test, Y_test))

# Save the model to use it directly in our API
xgb_model.save_model("xgb_model.bin")