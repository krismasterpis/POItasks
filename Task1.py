import numpy as np
import os

def generate_horizontal_plane(nPoints, width, length, z=0):
    xArr = np.random.uniform(-width / 2, width / 2, nPoints)
    yArr = np.random.uniform(-length / 2, length / 2, nPoints)
    zArr = np.full(nPoints, z)
    return np.column_stack((xArr, yArr, zArr))

def generate_vertical_plane(nPoints, width, height,invert = False):
    if invert:
        x = np.random.uniform(-width / 2, width / 2, nPoints)
        z = np.random.uniform(-height / 2, height / 2, nPoints)
        y = np.full(nPoints, 0)
    else:
        y = np.random.uniform(-width / 2, width / 2, nPoints)
        z = np.random.uniform(-height / 2, height / 2, nPoints)
        x = np.full(nPoints, 0)
    return np.column_stack((x, y, z))

def generate_cylinder_surface(nPoints, radius, height):
    a = np.random.uniform(0, 2 * np.pi, nPoints)
    z = np.random.uniform(-height / 2, height / 2, nPoints)
    x = radius * np.cos(a)
    y = radius * np.sin(a)
    return np.column_stack((x, y, z))

def generate_mix_surface(nPoints, radius, height,width,length, dist=0):
    xArr = np.random.uniform(-width / 2, width / 2, nPoints)
    yArr = np.random.uniform(-length / 2, length / 2, nPoints)
    zArr = np.full(nPoints, 0)
    x = np.random.uniform(-width / 2, width / 2, nPoints)+dist
    z = np.random.uniform(-height / 2, height / 2, nPoints)
    y = np.full(nPoints, 0)
    a = np.random.uniform(0, 2 * np.pi, nPoints)
    zc = np.random.uniform(-height / 2, height / 2, nPoints)
    xc = (radius * np.cos(a))+(dist*2)
    yc = (radius * np.sin(a))
    X = []
    Y = []
    Z = []

    X.extend(xArr)
    X.extend(x)
    X.extend(xc)

    Y.extend(yArr)
    Y.extend(y)
    Y.extend(yc)

    Z.extend(zArr)
    Z.extend(z)
    Z.extend(zc)

    return np.column_stack((X, Y, Z))


def save(filename, points):
    if not os.path.exists("./cloudcompare_data"):
        os.makedirs("./cloudcompare_data/")
    name, ext = os.path.splitext(filename)
    cnt = 1
    while os.path.exists("./cloudcompare_data/"+filename):
        filename = f"{name}_{cnt}{ext}"
        cnt += 1
    np.savetxt("./cloudcompare_data/"+filename, points, delimiter=' ', comments='', fmt='%.6f')


horizontalCloud = generate_horizontal_plane(1000, 10, 10)
verticalCloudX = generate_vertical_plane(1000, 10, 10)
verticalCloudY = generate_vertical_plane(1000, 10, 10, invert = True)
cylinderCloud = generate_cylinder_surface(1000, 5, 10)
mixCloud = generate_mix_surface(1000,5,10,10,10)
combinedCloud = generate_mix_surface(1000,5,10,10,10,15)

save("horizontal_cloud_X.xyz", horizontalCloud)
save("vertical_cloud_X.xyz", verticalCloudX)
save("vertical_cloud_Y.xyz", verticalCloudY)
save("cylinder_cloud.xyz", cylinderCloud)
save("mix.xyz", mixCloud)
save("combined.xyz",combinedCloud)