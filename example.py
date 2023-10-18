from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scitime import RuntimeEstimator
from sklearn import datasets

# example for rf regressor
estimator = RuntimeEstimator(meta_algo='NN', verbose=3)
rf = RandomForestRegressor()

X, y  = datasets.load_diabetes(return_X_y = True)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(rf, X, y)

rf.fit(X,y)
