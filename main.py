# Importing modules
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score

import joblib

# Import Data
students_data = pd.read_csv("data/Student_Performance.csv")

# Since, None column has max value greater than 128 we can reduce the datatype to int8 and float16 to save memory.
students_data['Hours Studied'] = students_data['Hours Studied'].astype('int8')
students_data['Previous Scores'] = students_data['Previous Scores'].astype('int8')
students_data['Sleep Hours'] = students_data['Sleep Hours'].astype('int8')
students_data['Sample Question Papers Practiced'] = students_data['Sample Question Papers Practiced'].astype('int8')
students_data['Performance Index'] = students_data['Performance Index'].astype('float16')

# Handling Duplicate Rows
students_data = students_data.drop_duplicates()

# Split Dataset into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(students_data.drop(columns='Performance Index'),
                                                    students_data['Performance Index'])

# Handling Categorical Data
transform1 = ColumnTransformer([("OneHotEncoding", OneHotEncoder(), [2])], remainder="passthrough")

# Scaling Dataset
transform2 = ColumnTransformer([
    ("MinMaxScaling", MinMaxScaler(), slice(0, 5))
])

# Creating a Pipeline
pipe = Pipeline([
    ("Transform1", transform1),
    ("Transform2", transform2)
])

# Transforming/ Performing Feature Engineering on x_train and x_test
x_train = pipe.fit_transform(x_train)
x_test = pipe.transform(x_test)

start = time.time()
estimators = [
    ('sgd', SGDRegressor(penalty='l1', max_iter=900, loss='squared_error',
                        learning_rate='invscaling', eta0=0.01)),
    ('ridge', Ridge(alpha=0.06)),
    ('svr', SVR(kernel='linear', gamma='scale')),
    ('gradboost', GradientBoostingRegressor(n_estimators=650, max_depth=1,
                                           loss='absolute_error', learning_rate=0.96))
]
streg = StackingRegressor(estimators=estimators,
                      final_estimator=LinearRegression(),
                       cv=10)
streg.fit(x_train, y_train)
y_pred = streg.predict(x_test)

print("\nTime Taken for Stacking is", np.round(time.time()-start, 2), "seconds")
print("R2 Score:", np.round(r2_score(y_test, y_pred)*100, 2))

# Saving the model
joblib.dump(streg, 'students_performance_model')
joblib.dump(pipe, 'pipe')