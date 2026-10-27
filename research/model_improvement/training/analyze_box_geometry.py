import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "training" / "model_9" / "ninth_model_data" / "annotations.csv"

def analyze_geometry():
    df = pd.read_csv(CSV_PATH)
    print(f"Total annotations in Model 9 dataset: {len(df)}")
    print(df["class_id"].value_counts().to_dict())

    # Filter valid positive boxes
    pos = df[(df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])].copy()
    pos["w"] = pos["xmax"] - pos["xmin"]
    pos["h"] = pos["ymax"] - pos["ymin"]
    pos["area"] = pos["w"] * pos["h"]
    pos["scale"] = np.sqrt(pos["area"])
    pos["aspect_ratio"] = pos["w"] / pos["h"]

    class_map = {1: "handgun", 2: "knife"}

    for cid, label in class_map.items():
        sub = pos[pos["class_id"] == cid]
        print(f"\n{'='*60}")
        print(f"GEOMETRY ANALYSIS: {label.upper()} (class_id={cid}, N = {len(sub)})")
        print(f"{'='*60}")
        print(f"Width  (px): mean={sub['w'].mean():.1f}, min={sub['w'].min():.1f}, 25%={sub['w'].quantile(0.25):.1f}, median={sub['w'].median():.1f}, 75%={sub['w'].quantile(0.75):.1f}, max={sub['w'].max():.1f}")
        print(f"Height (px): mean={sub['h'].mean():.1f}, min={sub['h'].min():.1f}, 25%={sub['h'].quantile(0.25):.1f}, median={sub['h'].median():.1f}, 75%={sub['h'].quantile(0.75):.1f}, max={sub['h'].max():.1f}")
        print(f"Scale  (px): mean={sub['scale'].mean():.1f}, min={sub['scale'].min():.1f}, 25%={sub['scale'].quantile(0.25):.1f}, median={sub['scale'].median():.1f}, 75%={sub['scale'].quantile(0.75):.1f}, max={sub['scale'].max():.1f}")
        print(f"Aspect Ratio (w/h): min={sub['aspect_ratio'].min():.2f}, 10%={sub['aspect_ratio'].quantile(0.10):.2f}, median={sub['aspect_ratio'].median():.2f}, 90%={sub['aspect_ratio'].quantile(0.90):.2f}, max={sub['aspect_ratio'].max():.2f}")
        
        # Check how many fall outside default aspect ratios (0.5 to 2.0)
        extreme_vertical = (sub['aspect_ratio'] < 0.5).mean() * 100
        extreme_horizontal = (sub['aspect_ratio'] > 2.0).mean() * 100
        extreme_small = (sub['scale'] < 24).mean() * 100
        print(f"Proportion with Aspect Ratio < 0.5 (tall/slender): {extreme_vertical:.1f}%")
        print(f"Proportion with Aspect Ratio > 2.0 (wide/horizontal): {extreme_horizontal:.1f}%")
        print(f"Proportion with Scale < 24px (distant small): {extreme_small:.1f}%")

if __name__ == "__main__":
    analyze_geometry()
