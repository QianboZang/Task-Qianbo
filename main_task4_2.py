"""
Task 4: Pose Estimation Methods Comparison Under Occlusion
分析不同姿态估计方法在强遮挡下的行为差异

直接运行: python task4_pose_analysis.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================
# 固定路径配置
# ============================================================
LMO_ROOT = Path("./lmo")
PRED_DIR = Path("./pose")  # 修改为你的目录名

# 方法信息 - 根据BOP Pose Estimation Leaderboard排名
METHODS = {
    "freezev21":          {"file": "freezev21_lmo-test.csv",                 "rank": 6,  "ar_lmo": 0.777},
    "sam6d":              {"file": "sam6d_lmo-test.csv",                     "rank": 10, "ar_lmo": 0.778},
    "waprv2multi-2d":     {"file": "waprv2multi-2d-detections_lmo-test.csv", "rank": 8,  "ar_lmo": 0.768},
}

OCCLUSION_LEVELS = {
    "low": (0.7, 1.0),
    "medium": (0.3, 0.7),
    "high": (0.0, 0.3)
}

ADD_THRESHOLD_FACTOR = 0.1

MODEL_DIAMETERS = {
    1: 102.1, 5: 108.0, 6: 164.6, 8: 129.5,
    9: 104.0, 10: 115.3, 11: 90.9, 12: 130.0,
}

SYMMETRIC_OBJECTS = {10, 11}


def load_gt_poses(lmo_root):
    """加载GT姿态和遮挡信息"""
    gt_data = {}
    test_dir = lmo_root / "test"
    
    for scene_dir in sorted(test_dir.glob("*")):
        if not scene_dir.is_dir():
            continue
        
        scene_id = int(scene_dir.name)
        gt_file = scene_dir / "scene_gt.json"
        gt_info_file = scene_dir / "scene_gt_info.json"
        
        if not gt_file.exists() or not gt_info_file.exists():
            continue
        
        with open(gt_file) as f:
            scene_gt = json.load(f)
        with open(gt_info_file) as f:
            scene_gt_info = json.load(f)
        
        for img_id_str, objs_gt in scene_gt.items():
            img_id = int(img_id_str)
            objs_info = scene_gt_info[img_id_str]
            
            for obj_idx, (gt_obj, info) in enumerate(zip(objs_gt, objs_info)):
                obj_id = gt_obj["obj_id"]
                R_gt = np.array(gt_obj["cam_R_m2c"]).reshape(3, 3)
                t_gt = np.array(gt_obj["cam_t_m2c"]).reshape(3)
                
                key = (scene_id, img_id, obj_id, obj_idx)
                gt_data[key] = {
                    "R": R_gt,
                    "t": t_gt,
                    "visib_fract": info.get("visib_fract", 1.0),
                    "obj_id": obj_id
                }
    
    print(f"Loaded {len(gt_data)} GT poses")
    return gt_data


def load_predictions(pred_file):
    """加载预测结果"""
    with open(pred_file) as f:
        first_line = f.readline().strip()
    
    if first_line.startswith("scene_id"):
        df = pd.read_csv(pred_file)
    else:
        df = pd.read_csv(pred_file, header=None,
                        names=["scene_id", "im_id", "obj_id", "score", "R", "t", "time"])
    
    predictions = []
    for _, row in df.iterrows():
        try:
            R_str = str(row["R"]).strip()
            R_values = [float(x) for x in R_str.split()]
            R = np.array(R_values).reshape(3, 3)
            
            t_str = str(row["t"]).strip()
            t_values = [float(x) for x in t_str.split()]
            t = np.array(t_values)
            
            im_id = int(row.get("im_id", row.get("image_id", 0)))
            
            predictions.append({
                "scene_id": int(row["scene_id"]),
                "im_id": im_id,
                "obj_id": int(row["obj_id"]),
                "score": float(row["score"]),
                "R": R,
                "t": t
            })
        except:
            continue
    
    print(f"  Loaded {len(predictions)} predictions")
    return predictions


def compute_add(R_pred, t_pred, R_gt, t_gt, points):
    pred_points = (R_pred @ points.T).T + t_pred
    gt_points = (R_gt @ points.T).T + t_gt
    return np.mean(np.linalg.norm(pred_points - gt_points, axis=1))


def compute_adds(R_pred, t_pred, R_gt, t_gt, points):
    pred_points = (R_pred @ points.T).T + t_pred
    gt_points = (R_gt @ points.T).T + t_gt
    distances = [np.min(np.linalg.norm(gt_points - p, axis=1)) for p in pred_points]
    return np.mean(distances)


def generate_sphere_points(n=500):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.stack([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)], axis=1)


def analyze_method(predictions, gt_data):
    sphere_points = generate_sphere_points(500)
    
    best_preds = {}
    for pred in predictions:
        key = (pred["scene_id"], pred["im_id"], pred["obj_id"])
        if key not in best_preds or pred["score"] > best_preds[key]["score"]:
            best_preds[key] = pred
    
    results = []
    for gt_key, gt_info in gt_data.items():
        scene_id, img_id, obj_id, obj_idx = gt_key
        pred_key = (scene_id, img_id, obj_id)
        
        if pred_key not in best_preds:
            results.append({
                "scene_id": scene_id, "image_id": img_id, "obj_id": obj_id,
                "visib_fract": gt_info["visib_fract"], "add_error": np.inf, "matched": False
            })
            continue
        
        pred = best_preds[pred_key]
        R_pred, t_pred = pred["R"], pred["t"]
        R_gt, t_gt = gt_info["R"], gt_info["t"]
        
        diameter = MODEL_DIAMETERS.get(obj_id, 100.0)
        model_points = sphere_points * (diameter / 2)
        
        if obj_id in SYMMETRIC_OBJECTS:
            add_error = compute_adds(R_pred, t_pred, R_gt, t_gt, model_points)
        else:
            add_error = compute_add(R_pred, t_pred, R_gt, t_gt, model_points)
        
        threshold = ADD_THRESHOLD_FACTOR * diameter
        
        results.append({
            "scene_id": scene_id, "image_id": img_id, "obj_id": obj_id,
            "visib_fract": gt_info["visib_fract"], "add_error": add_error,
            "matched": add_error < threshold
        })
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("Task 4: Pose Estimation Methods Comparison Under Occlusion")
    print("=" * 70)
    print(f"\nLMO Root: {LMO_ROOT.absolute()}")
    print(f"Predictions: {PRED_DIR.absolute()}")
    
    if not PRED_DIR.exists():
        print(f"\n[ERROR] Directory not found: {PRED_DIR}")
        return
    
    print("\n[1] Loading data...")
    gt_data = load_gt_poses(LMO_ROOT)
    
    print("\n[2] Analyzing methods...")
    all_results = []
    
    for method_name, info in METHODS.items():
        pred_file = PRED_DIR / info["file"]
        if not pred_file.exists():
            print(f"  Skipping {method_name}: {pred_file} not found")
            continue
        
        print(f"  Processing {method_name} (Leaderboard Rank: {info['rank']})...")
        predictions = load_predictions(pred_file)
        
        if len(predictions) == 0:
            continue
            
        df = analyze_method(predictions, gt_data)
        
        row = {"method": method_name, "leaderboard_rank": info["rank"], "ar_lmo": info.get("ar_lmo", np.nan)}
        
        for level, (lo, hi) in OCCLUSION_LEVELS.items():
            subset = df[(df["visib_fract"] > lo) & (df["visib_fract"] <= hi)]
            row[f"{level}_count"] = len(subset)
            row[f"{level}_recall"] = subset["matched"].mean() if len(subset) > 0 else np.nan
            valid_add = subset[subset["add_error"] < np.inf]["add_error"]
            row[f"{level}_mean_add"] = valid_add.mean() if len(valid_add) > 0 else np.nan
        
        row["total_recall"] = df["matched"].mean()
        all_results.append(row)
    
    if len(all_results) == 0:
        print("\n[ERROR] No methods analyzed!")
        return
    
    df_results = pd.DataFrame(all_results).sort_values("leaderboard_rank")
    
    # 输出结果
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n[Table 1] ADD(-S) Recall by Occlusion Level")
    print("-" * 80)
    cols1 = ["method", "leaderboard_rank", "ar_lmo", "low_recall", "medium_recall", "high_recall", "total_recall"]
    print(df_results[cols1].round(4).to_string(index=False))
    
    print("\n[Table 2] Mean ADD(-S) Error (mm)")
    print("-" * 80)
    cols2 = ["method", "leaderboard_rank", "low_mean_add", "medium_mean_add", "high_mean_add"]
    print(df_results[cols2].round(2).to_string(index=False))
    
    print("\n[Table 3] Sample Counts")
    print("-" * 70)
    cols3 = ["method", "low_count", "medium_count", "high_count"]
    print(df_results[cols3].to_string(index=False))
    
    # 排名分析
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Rank Changes Under High Occlusion")
    print("=" * 70)
    
    df_rank = df_results[["method", "leaderboard_rank", "ar_lmo", "high_recall"]].copy()
    df_rank["high_occ_rank"] = df_rank["high_recall"].rank(ascending=False).astype(int)
    df_rank["rank_change"] = df_rank["leaderboard_rank"] - df_rank["high_occ_rank"]
    
    print("\n(positive = BETTER under high occlusion)")
    print(df_rank.round(4).to_string(index=False))
    
    # 性能下降
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Performance Degradation")
    print("=" * 70)
    
    df_drop = df_results[["method", "leaderboard_rank", "low_recall", "high_recall"]].copy()
    df_drop["recall_drop"] = df_drop["low_recall"] - df_drop["high_recall"]
    df_drop["drop_percent"] = (df_drop["recall_drop"] / df_drop["low_recall"] * 100).round(1)
    
    print("\n(lower drop = more robust)")
    print(df_drop.round(4).to_string(index=False))
    
    # 关键发现
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Key Findings")
    print("=" * 70)
    
    improved = df_rank[df_rank["rank_change"] > 0]
    if len(improved) > 0:
        print("\n[!] Better under high occlusion:")
        for _, row in improved.iterrows():
            print(f"    - {row['method']}: #{int(row['leaderboard_rank'])} -> #{int(row['high_occ_rank'])} (+{int(row['rank_change'])})")
    
    degraded = df_rank[df_rank["rank_change"] < 0]
    if len(degraded) > 0:
        print("\n[!] Worse under high occlusion:")
        for _, row in degraded.iterrows():
            print(f"    - {row['method']}: #{int(row['leaderboard_rank'])} -> #{int(row['high_occ_rank'])} ({int(row['rank_change'])})")
    
    valid_drop = df_drop.dropna(subset=["drop_percent"])
    if len(valid_drop) > 0:
        most_robust = valid_drop.loc[valid_drop["drop_percent"].idxmin()]
        least_robust = valid_drop.loc[valid_drop["drop_percent"].idxmax()]
        print(f"\n[!] Most robust: {most_robust['method']} (drop: {most_robust['drop_percent']}%)")
        print(f"[!] Least robust: {least_robust['method']} (drop: {least_robust['drop_percent']}%)")
    
    # 结论
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    best = df_rank.loc[df_rank["high_occ_rank"].idxmin()]
    print(f"""
Best under HIGH OCCLUSION:
  - {best['method']} (Leaderboard #{int(best['leaderboard_rank'])})
  - High-occ Recall: {best['high_recall']:.4f}
  - High-occ Rank: #{int(best['high_occ_rank'])}
  - Rank change: {int(best['rank_change'])}
""")
    
    df_results.to_csv("task4_pose_results.csv", index=False)
    print("Saved to task4_pose_results.csv")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()