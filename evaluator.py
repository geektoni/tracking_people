import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


def compute_distance(points):
    distances = []
    for index, row in points.iterrows():
        point_a = np.array([row["x"], row["y"]])
        point_b = np.array([row["X"], row["Y"]])
        distance = np.linalg.norm(point_a - point_b)
        distances.append(distance)
    return np.array(distances)

if (len(sys.argv) == 1):
    print("Wrong number of parameters!")
    print("Usage: evaluator.py <result_dir>")

# Data types
dtypes = {"frame": np.int32, "id": np.int32, "x":np.float64, "y":np.float64}

# Read the ground truth file
gt = pd.read_csv("./data/A1_assignment/A1_groundtruthC.txt", header=None, names=["frame", "id", "x", "y"], index_col=["frame"], dtype=dtypes)

# Users we want to track
users = [10, 36, 42]

# Compute the count variations
count = pd.read_csv(sys.argv[1]+"/people_count.csv", index_col=["frame"])
gt_count = gt.groupby(["frame"]).count().rename(columns={"id": "count"}).drop(["x", "y"], axis=1)
count_j = gt_count.join(count, lsuffix="_pred")

# Print RMSE of counting
rmse = sqrt(mean_squared_error(count_j["count"], count_j["count_pred"]))
print("Count RMSE: {}".format(rmse))


# Iterate over the users
for u in users:

    # Open the result file
    user_file = pd.read_csv(sys.argv[1]+"/track_"+str(u)+".csv", index_col=["frame"], dtype=dtypes)

    # Join the result with the ground truth
    result = gt[gt.id==u].join(user_file, rsuffix="_pred").drop(["id_pred"], axis=1)

    # Remove NaN values
    result = result.fillna(method='ffill')
    result = result.fillna(method='bfill')

    # Compute mean and std of the distances
    distances = compute_distance(result)
    mean = distances.mean()
    stddev = distances.std()

    # Return the mean +- stddev
    print("User({}): {:.2f} +- {:.2f}".format(u, mean, stddev))