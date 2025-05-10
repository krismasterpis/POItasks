import numpy as np
import os
import pandas as pd

def generateHorizontalPlane(nPoints, width, length, z=0):
    xArr = np.random.uniform(-width / 2, width / 2, nPoints)
    yArr = np.random.uniform(-length / 2, length / 2, nPoints)
    zArr = np.full(nPoints, z)
    return np.column_stack((xArr, yArr, zArr))

def generateVerticalPlane(nPoints, width, height, invert=False):
    if invert:
        x = np.random.uniform(-width / 2, width / 2, nPoints)
        z = np.random.uniform(-height / 2, height / 2, nPoints)
        y = np.full(nPoints, 0)
    else:
        y = np.random.uniform(-width / 2, width / 2, nPoints)
        z = np.random.uniform(-height / 2, height / 2, nPoints)
        x = np.full(nPoints, 0)
    return np.column_stack((x, y, z))

def generateCylinderSurface(nPoints, radius, height):
    a = np.random.uniform(0, 2 * np.pi, nPoints)
    z = np.random.uniform(-height / 2, height / 2, nPoints)
    x = radius * np.cos(a)
    y = radius * np.sin(a)
    return np.column_stack((x, y, z))

def generateMixSurface(nPoints, radius, height, width, length):
    xArr = np.random.uniform(-width / 2, width / 2, nPoints)
    yArr = np.random.uniform(-length / 2, length / 2, nPoints)
    zArr = np.full(nPoints, 0)

    y = np.random.uniform(-width / 2, width / 2, nPoints)
    z = np.random.uniform(-height / 2, height / 2, nPoints)
    x = np.full(nPoints, 0)

    a = np.random.uniform(0, 2 * np.pi, nPoints)
    zc = np.random.uniform(-height / 2, height / 2, nPoints)
    xc = radius * np.cos(a)
    yc = radius * np.sin(a)

    X = np.concatenate([xArr, x, xc])
    Y = np.concatenate([yArr, y, yc])
    Z = np.concatenate([zArr, z, zc])

    return np.column_stack((X, Y, Z))

def save_as_csv(filename, points):
    if not os.path.exists("./cloudcompare_data"):
        os.makedirs("./cloudcompare_data/")
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".csv":
        filename = name + ".csv"

    path = os.path.join("./cloudcompare_data", filename)
    cnt = 1
    while os.path.exists(path):
        path = os.path.join("./cloudcompare_data", f"{name}_{cnt}.csv")
        cnt += 1

    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df.to_csv(path, index=False)
    print(f"Zapisano: {path}")

horizontalCloud = generateHorizontalPlane(1000, 10, 10)
verticalCloudX = generateVerticalPlane(1000, 10, 10)
verticalCloudY = generateVerticalPlane(1000, 10, 10, invert=True)
cylinderCloud = generateCylinderSurface(1000, 5, 10)
mixCloud = generateMixSurface(1000, 5, 10, 10, 10)

save_as_csv("horizontal_cloud.csv", horizontalCloud)
save_as_csv("vertical_cloud_X.csv", verticalCloudX)
save_as_csv("vertical_cloud_Y.csv", verticalCloudY)
save_as_csv("cylinder_cloud.csv", cylinderCloud)
save_as_csv("mix.csv", mixCloud)
