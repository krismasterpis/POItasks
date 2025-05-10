import numpy as np
import matplotlib.pyplot as plt


def load_xyz_txt(_filename):
    points = []

    with open(_filename, 'r') as f:
        next(f)

        for line in f:
            values = line.strip().split()
            if len(values) == 3:
                try:
                    points.append([float(values[0]), float(values[1]), float(values[2])])
                except ValueError:
                    continue

    np_points = np.array(points)
    return np_points[:, 0], np_points[:, 1], np_points[:, 2]

def fit_plane(x, y, z):

    points = np.column_stack((x, y, z))  # Tworzy macierz punktów
    p1, p2, p3 = points  # Wybiera 3 punkty
    normal = np.cross(p2 - p1, p3 - p1)  # Wektor normalny
    normal = normal / np.linalg.norm(normal)  # Normalizacja
    D = -np.dot(normal, p1)  # Obliczenie D
    return abs(np.append(normal, D))  # (A, B, C, D)


def ransac(x, y, z, iterations=1000, threshold=0.01):
    points = np.column_stack((x, y, z))
    best_plane = None
    best_inliers = []

    for _ in range(iterations):
        # Wybór losowo 3 punktów
        indices = np.random.choice(len(x), 3, replace=False)
        sample_x, sample_y, sample_z = x[indices], y[indices], z[indices]

        # Dopasowanie płaszczyzny
        plane = fit_plane(sample_x, sample_y, sample_z)
        A, B, C, D = plane

        # Obliczanie odległości wszystkich punktów od płaszczyzny
        distances = np.abs(A * x + B * y + C * z + D) / np.linalg.norm([A, B, C])

        # Znajdywanie inlierów (punkty bliskie płaszczyzny)
        inlier_indices = distances < threshold
        inliers = points[inlier_indices]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = plane

    return best_plane, best_inliers


def interpret_plane(plane_coeffs, threshold=0.1):
    A, B, C, D = plane_coeffs  # Współczynniki płaszczyzny

    normal_vector = np.array([A, B, C])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    if abs(normal_vector[2]) > 1 - threshold:
        return "Płaszczyzna jest pozioma."
    elif abs(normal_vector[0]) > 1 - threshold or abs(normal_vector[1]) > 1 - threshold:
        return "Płaszczyzna jest pionowa."
    else:
        return "Płaszczyzna jest skośna lub nierozpoznana."

x, y, z = load_xyz_txt("./cloudcompare_data/combined.xyz")
points = np.column_stack((x, y, z))

# Dopasowanie pierwszej płaszczyzny
plane1, inliers1 = ransac(x, y, z, iterations=100, threshold=0.0001)
print("Płaszczyzna 1:", interpret_plane(plane1))
print(plane1)
# Usuwanie inlierów pierwszej płaszczyzny
mask1 = np.array([tuple(p) in map(tuple, inliers1) for p in points])
remaining = points[~mask1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*inliers1.T, c='blue', label='Płaszczyzna 1')

# Dopasowanie drugiej płaszczyzny
if remaining.size > 0:
    plane2, inliers2 = ransac(remaining[:,0], remaining[:,1], remaining[:,2], iterations=10000, threshold=0.0001)
    print("Płaszczyzna 2:", interpret_plane(plane2))
    print(plane2)
    # Usuwanie inlierów drugiej płaszczyzny
    mask2 = np.array([tuple(p) in map(tuple, inliers2) for p in remaining])
    remaining2 = remaining[~mask2]
    ax.scatter(*inliers2.T, c='green', label='Płaszczyzna 2')

    if remaining2.size > 0:
        print(f"Pozostało {len(remaining2)} punktów – to może być cylinder")
        ax.scatter(*remaining2.T, c='red', label='Cylinder (?)', alpha=0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()