import numpy as np
import pandas as pd
from pyransac3d import Plane
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_points(_path):
    df = pd.read_table(_path, delim_whitespace=True, header=None, usecols=[0, 1, 2])
    df.columns = ['x', 'y', 'z']
    return df.values

def check_if_plane(points, thresh=0.1, min_inliers_ratio=0.8):
    try:
        plane_model, inliers = Plane().fit(points, thresh=thresh, maxIteration=1000)
        inlier_pts = points[inliers]
        if len(inlier_pts) < len(points) * min_inliers_ratio:
            return False, None, None, None
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        distances = np.abs((points @ normal + d)) / np.linalg.norm(normal)
        avg_distance = np.mean(distances)
        return True, normal, avg_distance, plane_model
    except Exception as e:
        print(f"Błąd podczas dopasowania płaszczyzny: {e}")
        return False, None, None, None

def classify_normal_orientation(normal):
    if abs(normal[2]) > abs(normal[0]) and abs(normal[2]) > abs(normal[1]):
        return "pozioma"
    else:
        return "pionowa"


def segment_and_classify(points, n_clusters=3):
    print("Podział chmury punktów za pomocą KMeans...")
    #scaled = StandardScaler().fit_transform(points)
    kmeans = KMeans(n_clusters=n_clusters,init="k-means++",n_init="auto", random_state=42)
    labels = kmeans.fit_predict(points)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap("tab10")

    for label in np.unique(labels):
        cluster_points = points[labels == label]

        color = cmap(label % 10)[:3]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   color=color, label=f"Klaster {label}", s=1)
        print(f"\nKlaster {label} – {len(cluster_points)} punktów")

        is_plane, normal, avg_dist, plane_model = check_if_plane(cluster_points)
        if is_plane:
            orient = classify_normal_orientation(normal)
            print(f"   Płaszczyzna wykryta")
            print(f"   Wektor normalny: {np.round(normal, 4)}")
            print(f"   Średnia odległość: {avg_dist:.5f}")
            print(f"   Orientacja: {orient}")
        else:
            print(f"Płaszczyzna nie wykryta")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Chmura punktów i wyniki segmentacji")
    ax.legend()

    print("\nWyświetlam klastry 3D...")
    plt.show()

if __name__ == "__main__":
    path = "./cloudcompare_data/combined.xyz"
    pts = load_points(path)
    segment_and_classify(pts, n_clusters=3)
