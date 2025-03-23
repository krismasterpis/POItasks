import numpy as np
import os

def generateHorizontalPlane(nPoints, width, length, z=0):
    xArr = np.random.uniform(-width / 2, width / 2, nPoints)
    yArr = np.random.uniform(-length / 2, length / 2, nPoints)
    zArr = np.full(nPoints, z)
    return np.column_stack((xArr, yArr, zArr))

def generateVerticalPlane(nPoints, width, height,invert = False):
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

def save(filename, points):
    if not os.path.exists("./cloudcompare_data"):
        os.makedirs("./cloudcompare_data/")
    name, ext = os.path.splitext(filename)
    cnt = 1
    while os.path.exists("./cloudcompare_data/"+filename):
        filename = f"{name}_{cnt}{ext}"
        cnt += 1
    np.savetxt("./cloudcompare_data/"+filename, points, delimiter=' ', header='X Y Z', comments='', fmt='%.6f')


horizontalCloud = generateHorizontalPlane(1000, 10, 10)
verticalCloudX = generateVerticalPlane(1000, 10, 10)
verticalCloudY = generateVerticalPlane(1000, 10, 10, invert = True)
cylinderCloud = generateCylinderSurface(1000, 5, 10)
save("horizontal_cloud.txt", horizontalCloud)
save("vertical_cloud_X.txt", verticalCloudX)
save("vertical_cloud_Y.txt", verticalCloudY)
save("cylinder_cloud.txt", cylinderCloud)