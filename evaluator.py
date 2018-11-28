import pandas as pd
import numpy as np


def compute_distance(points):
    distances = []
    for index, row in points.iterrows():
        point_a = np.array([row["x"], row["y"]])
        point_b = np.array([row["X"], row["Y"]])
        distance = np.linalg.norm(point_a - point_b)
        distances.append(distance)
    return np.array(distances)


# Data types
dtypes = {"frame": np.int32, "id": np.int32, "x":np.float64, "y":np.float64}

# Read the ground truth file
gt = pd.read_csv("./data/A1_assignment/A1_groundtruthC.txt", header=None, names=["frame", "id", "x", "y"], index_col=["frame"], dtype=dtypes)

# Users we want to track
users = [10, 36, 42]

# Iterate over the users
print("# Tracking displacement #")
for u in users:

    # Open the result file
    user_file = pd.read_csv("./track_"+str(u)+".csv", index_col=["frame"], dtype=dtypes)

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