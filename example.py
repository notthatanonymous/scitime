from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

from scitime import RuntimeEstimator

# example for rf regressor
estimator = RuntimeEstimator(meta_algo='RF', verbose=3)
rf = RandomForestRegressor()

X,y = np.random.rand(100000,10),np.random.rand(100000,1)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(rf, X, y)

# compare to the actual training time
start_time = time.time()
rf.fit(X,y)
elapsed_time = time.time() - start_time
print("elapsed time: {:.2}".format(elapsed_time))
