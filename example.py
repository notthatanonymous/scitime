from sklearn.cluster import KMeans
import numpy as np
import time

from scitime import RuntimeEstimator

# example for kmeans clustering
estimator = RuntimeEstimator(meta_algo='RF', verbose=3)
km = KMeans()

X = np.random.rand(100000,10)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(km, X)

# compare to the actual training time
start_time = time.time()
km.fit(X)
elapsed_time = time.time() - start_time
print("elapsed time: {:.2}".format(elapsed_time))
