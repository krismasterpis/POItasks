import numpy as np
import pandas as pd
import open3d as o3d
from pyransac3d import Plane, Cylinder
import os
import matplotlib.pyplot as plt

def load_points(_path):
    df = pd.read_table(_path, delim_whitespace=True, header=None, usecols=[0, 1, 2])
    df.columns = ['x', 'y', 'z']
    return df.values

def save_points_to_csv(points, name, output_dir="planes_out"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.csv")
    pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(path, index=False)
    print(f"  Zapisano: {path}")

def visualize_segments(segments):
    geometries = []
    cmap = plt.get_cmap("tab10")
    for i, pts in enumerate(segments):
        color = cmap(i % 10)[:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)

def segment_planes_and_cylinder(points, max_planes=6, thresh_plane=0.01, min_inliers=100):
    remaining_points = points.copy()
    all_segments = []
    pl = Plane()

    for i in range(1, max_planes + 1):
        if len(remaining_points) < 3:
            print("  Za mało punktów do dalszego dopasowania.")
            break

        try:
            plane_model, inliers = pl.fit(remaining_points, thresh=thresh_plane, maxIteration=1000)
        except Exception as e:
            print(f"  Błąd w iteracji {i}: {e}")
            break

        inlier_pts = remaining_points[inliers]
        if len(inlier_pts) < min_inliers:
            print(f"Iteracja {i}: za mało dopasowanych punktów ({len(inlier_pts)})")
            break

        a, b, c, d = plane_model
        orientation = "pozioma" if abs(c) > abs(a) and abs(c) > abs(b) else "pionowa"

        print(f"\nPłaszczyzna {i}")
        print(f"  Równanie: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print(f"  Orientacja: {orientation} | Liczba punktów: {len(inlier_pts)}")

        save_points_to_csv(inlier_pts, f"plane_{i}")
        all_segments.append(inlier_pts)

        mask = np.ones(len(remaining_points), dtype=bool)
        mask[inliers] = False
        remaining_points = remaining_points[mask]

    if len(remaining_points) > 0:
        remainder_name = "cylinder_noise" # Nazwa dla pozostałych punktów
        save_points_to_csv(remaining_points, "noise")
        all_segments.append(remaining_points)
    else:
        print("  Nie pozostały żadne punkty nieprzypisane do płaszczyzn.")
    return all_segments

if __name__ == "__main__":
    csv_path = "./cloudcompare_data/combined.xyz"
    point_cloud = load_points(csv_path)

    segments = segment_planes_and_cylinder(
        point_cloud,
        max_planes=3,
        thresh_plane=0.0001,
        min_inliers=10
    )

    visualize_segments(segments)
