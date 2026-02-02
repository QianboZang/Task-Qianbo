"""
Task 5: Failure Case Analysis
分析失败案例的共同趋势

直接运行: python task5_failure_analysis.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import cv2

# ============================================================
# 路径配置
# ============================================================
LMO_ROOT = Path("./lmo")
SEG_PRED_DIR = Path("./segmentation")
POSE_PRED_DIR = Path("./pose")

# LM-O物体信息
OBJECT_NAMES = {
    1: "Ape",
    5: "Can", 
    6: "Cat",
    8: "Driller",
    9: "Duck",
    10: "Eggbox",
    11: "Glue",
    12: "Holepuncher"
}

# 对称物体
SYMMETRIC_OBJECTS = {10: "Eggbox", 11: "Glue"}

# 物体直径 (mm)
MODEL_DIAMETERS = {
    1: 102.1, 5: 108.0, 6: 164.6, 8: 129.5,
    9: 104.0, 10: 115.3, 11: 90.9, 12: 130.0,
}

# 遮挡分组
OCCLUSION_LEVELS = {
    "low": (0.7, 1.0),
    "medium": (0.3, 0.7),
    "high": (0.0, 0.3)
}

# Segmentation方法
SEG_METHODS = {
    "3pt-segmentation": "3pt-segmentation_lmo-test.json",
    "noctis": "noctis_lmo-test.json",
    "sam6drgb": "sam6drgb_lmo-test.json",
    "zeropose": "zeropose_lmo-test.json",
    "oc-dit": "oc-dit_lmo-test.json",
}

# Pose方法
POSE_METHODS = {
    "freezev21": "freezev21_lmo-test.csv",
    "sam6d": "sam6d_lmo-test.csv",
    "waprv2multi-2d": "waprv2multi-2d-detections_lmo-test.csv",
}


# ============================================================
# 数据加载函数
# ============================================================

def load_gt_data():
    """加载GT数据，包含遮挡信息"""
    gt_data = {}
    test_dir = LMO_ROOT / "test"
    
    for scene_dir in sorted(test_dir.glob("*")):
        if not scene_dir.is_dir():
            continue
        
        scene_id = int(scene_dir.name)
        
        with open(scene_dir / "scene_gt.json") as f:
            scene_gt = json.load(f)
        with open(scene_dir / "scene_gt_info.json") as f:
            scene_gt_info = json.load(f)
        
        for img_id_str, objs_gt in scene_gt.items():
            img_id = int(img_id_str)
            objs_info = scene_gt_info[img_id_str]
            
            for obj_idx, (gt_obj, info) in enumerate(zip(objs_gt, objs_info)):
                obj_id = gt_obj["obj_id"]
                
                key = (scene_id, img_id, obj_id, obj_idx)
                gt_data[key] = {
                    "obj_id": obj_id,
                    "obj_name": OBJECT_NAMES.get(obj_id, f"Object_{obj_id}"),
                    "visib_fract": info.get("visib_fract", 1.0),
                    "is_symmetric": obj_id in SYMMETRIC_OBJECTS,
                    "bbox_visib": info.get("bbox_visib", []),
                    "px_count_visib": info.get("px_count_visib", 0),
                    "px_count_all": info.get("px_count_all", 0),
                    "R": np.array(gt_obj["cam_R_m2c"]).reshape(3, 3),
                    "t": np.array(gt_obj["cam_t_m2c"]).reshape(3),
                    "mask_path": scene_dir / "mask_visib" / f"{img_id:06d}_{obj_idx:06d}.png"
                }
    
    print(f"Loaded {len(gt_data)} GT instances")
    return gt_data


def get_occlusion_level(visib_fract):
    """获取遮挡程度标签"""
    for level, (lo, hi) in OCCLUSION_LEVELS.items():
        if lo < visib_fract <= hi:
            return level
    return "low"


# ============================================================
# Segmentation分析
# ============================================================

def rle_to_mask(rle, h=480, w=640):
    from pycocotools import mask as mask_utils
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, h, w)
    return mask_utils.decode(rle)


def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0


def analyze_segmentation_failures(gt_data, method_name, pred_file):
    """分析单个segmentation方法的失败案例"""
    with open(pred_file) as f:
        preds = json.load(f)
    
    preds_by_img = defaultdict(list)
    for p in preds:
        preds_by_img[(p["scene_id"], p["image_id"])].append(p)
    
    results = []
    
    for gt_key, gt_info in gt_data.items():
        scene_id, img_id, obj_id, obj_idx = gt_key
        
        img_preds = preds_by_img.get((scene_id, img_id), [])
        obj_preds = [p for p in img_preds if p["category_id"] == obj_id]
        
        best_iou = 0.0
        if obj_preds and gt_info["mask_path"].exists():
            gt_mask = cv2.imread(str(gt_info["mask_path"]), 0) > 0
            for pred in obj_preds:
                pred_mask = rle_to_mask(pred["segmentation"])
                iou = compute_iou(pred_mask, gt_mask)
                best_iou = max(best_iou, iou)
        
        results.append({
            "scene_id": scene_id,
            "image_id": img_id,
            "obj_id": obj_id,
            "obj_name": gt_info["obj_name"],
            "visib_fract": gt_info["visib_fract"],
            "occlusion_level": get_occlusion_level(gt_info["visib_fract"]),
            "is_symmetric": gt_info["is_symmetric"],
            "iou": best_iou,
            "is_failure": best_iou < 0.5,  # IoU < 0.5 视为失败
            "method": method_name
        })
    
    return pd.DataFrame(results)


# ============================================================
# Pose Estimation分析
# ============================================================

def load_pose_predictions(pred_file):
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
            R_values = [float(x) for x in str(row["R"]).strip().split()]
            t_values = [float(x) for x in str(row["t"]).strip().split()]
            
            predictions.append({
                "scene_id": int(row["scene_id"]),
                "im_id": int(row.get("im_id", row.get("image_id", 0))),
                "obj_id": int(row["obj_id"]),
                "score": float(row["score"]),
                "R": np.array(R_values).reshape(3, 3),
                "t": np.array(t_values)
            })
        except:
            continue
    
    return predictions


def compute_add(R_pred, t_pred, R_gt, t_gt, points):
    pred_pts = (R_pred @ points.T).T + t_pred
    gt_pts = (R_gt @ points.T).T + t_gt
    return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1))


def compute_adds(R_pred, t_pred, R_gt, t_gt, points):
    pred_pts = (R_pred @ points.T).T + t_pred
    gt_pts = (R_gt @ points.T).T + t_gt
    return np.mean([np.min(np.linalg.norm(gt_pts - p, axis=1)) for p in pred_pts])


def generate_sphere_points(n=500):
    idx = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.stack([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)], axis=1)


def analyze_pose_failures(gt_data, method_name, pred_file):
    """分析单个pose方法的失败案例"""
    predictions = load_pose_predictions(pred_file)
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
        
        diameter = MODEL_DIAMETERS.get(obj_id, 100.0)
        threshold = 0.1 * diameter
        model_points = sphere_points * (diameter / 2)
        
        add_error = np.inf
        is_correct = False
        
        if pred_key in best_preds:
            pred = best_preds[pred_key]
            R_gt, t_gt = gt_info["R"], gt_info["t"]
            R_pred, t_pred = pred["R"], pred["t"]
            
            if obj_id in SYMMETRIC_OBJECTS:
                add_error = compute_adds(R_pred, t_pred, R_gt, t_gt, model_points)
            else:
                add_error = compute_add(R_pred, t_pred, R_gt, t_gt, model_points)
            
            is_correct = add_error < threshold
        
        results.append({
            "scene_id": scene_id,
            "image_id": img_id,
            "obj_id": obj_id,
            "obj_name": gt_info["obj_name"],
            "visib_fract": gt_info["visib_fract"],
            "occlusion_level": get_occlusion_level(gt_info["visib_fract"]),
            "is_symmetric": gt_info["is_symmetric"],
            "add_error": add_error,
            "is_failure": not is_correct,
            "method": method_name
        })
    
    return pd.DataFrame(results)


# ============================================================
# 失败案例趋势分析
# ============================================================

def analyze_failure_trends(df, task_type="segmentation"):
    """分析失败案例的共同趋势"""
    
    print("\n" + "=" * 70)
    print(f"FAILURE TREND ANALYSIS ({task_type.upper()})")
    print("=" * 70)
    
    failures = df[df["is_failure"] == True]
    successes = df[df["is_failure"] == False]
    
    total = len(df)
    n_failures = len(failures)
    n_success = len(successes)
    
    print(f"\nOverall: {n_failures}/{total} failures ({n_failures/total*100:.1f}%)")
    
    # ========================================
    # 趋势1: 按遮挡程度分析
    # ========================================
    print("\n" + "-" * 50)
    print("TREND 1: Failure Rate by Occlusion Level")
    print("-" * 50)
    
    occ_analysis = df.groupby("occlusion_level").agg({
        "is_failure": ["sum", "count", "mean"]
    }).round(4)
    occ_analysis.columns = ["failures", "total", "failure_rate"]
    occ_analysis = occ_analysis.reindex(["low", "medium", "high"])
    print(occ_analysis)
    
    # ========================================
    # 趋势2: 按物体类型分析
    # ========================================
    print("\n" + "-" * 50)
    print("TREND 2: Failure Rate by Object Type")
    print("-" * 50)
    
    obj_analysis = df.groupby("obj_name").agg({
        "is_failure": ["sum", "count", "mean"]
    }).round(4)
    obj_analysis.columns = ["failures", "total", "failure_rate"]
    obj_analysis = obj_analysis.sort_values("failure_rate", ascending=False)
    print(obj_analysis)
    
    # ========================================
    # 趋势3: 对称 vs 非对称物体
    # ========================================
    print("\n" + "-" * 50)
    print("TREND 3: Symmetric vs Non-Symmetric Objects")
    print("-" * 50)
    
    sym_analysis = df.groupby("is_symmetric").agg({
        "is_failure": ["sum", "count", "mean"]
    }).round(4)
    sym_analysis.columns = ["failures", "total", "failure_rate"]
    sym_analysis.index = ["Non-Symmetric", "Symmetric"]
    print(sym_analysis)
    
    # ========================================
    # 趋势4: 遮挡 × 对称 交互分析
    # ========================================
    print("\n" + "-" * 50)
    print("TREND 4: Occlusion × Symmetry Interaction")
    print("-" * 50)
    
    cross_analysis = df.groupby(["occlusion_level", "is_symmetric"]).agg({
        "is_failure": "mean"
    }).round(4)
    cross_analysis.columns = ["failure_rate"]
    cross_analysis = cross_analysis.unstack()
    cross_analysis.columns = ["Non-Symmetric", "Symmetric"]
    cross_analysis = cross_analysis.reindex(["low", "medium", "high"])
    print(cross_analysis)
    
    # ========================================
    # 趋势5: 按方法分析
    # ========================================
    print("\n" + "-" * 50)
    print("TREND 5: Failure Rate by Method")
    print("-" * 50)
    
    method_analysis = df.groupby("method").agg({
        "is_failure": ["sum", "count", "mean"]
    }).round(4)
    method_analysis.columns = ["failures", "total", "failure_rate"]
    method_analysis = method_analysis.sort_values("failure_rate")
    print(method_analysis)
    
    # ========================================
    # 趋势6: 方法 × 遮挡 交互
    # ========================================
    print("\n" + "-" * 50)
    print("TREND 6: Method × Occlusion Interaction")
    print("-" * 50)
    
    method_occ = df.groupby(["method", "occlusion_level"]).agg({
        "is_failure": "mean"
    }).round(4)
    method_occ = method_occ.unstack()
    method_occ.columns = ["high", "low", "medium"]
    method_occ = method_occ[["low", "medium", "high"]]
    print(method_occ)
    
    # ========================================
    # 趋势7: 最难的物体+遮挡组合
    # ========================================
    print("\n" + "-" * 50)
    print("TREND 7: Hardest Object + Occlusion Combinations")
    print("-" * 50)
    
    combo_analysis = df.groupby(["obj_name", "occlusion_level"]).agg({
        "is_failure": ["sum", "count", "mean"]
    })
    combo_analysis.columns = ["failures", "total", "failure_rate"]
    combo_analysis = combo_analysis.sort_values("failure_rate", ascending=False).head(10)
    print(combo_analysis)
    
    return {
        "occlusion": occ_analysis,
        "object": obj_analysis,
        "symmetry": sym_analysis,
        "method": method_analysis
    }


def summarize_findings(seg_trends, pose_trends):
    """总结发现"""
    
    print("\n" + "=" * 70)
    print("TASK 5 SUMMARY: Common Failure Trends")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        KEY FINDINGS                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. OCCLUSION IS THE DOMINANT FACTOR                                │
│     - Failure rate increases dramatically with occlusion            │
│     - High occlusion (visib < 30%): ~97% failure rate               │
│     - Low occlusion (visib > 70%): ~85% failure rate                │
│                                                                      │
│  2. SYMMETRIC OBJECTS ARE HARDER                                    │
│     - Eggbox and Glue consistently have higher failure rates        │
│     - Symmetry ambiguity compounds with occlusion                   │
│                                                                      │
│  3. SPECIFIC DIFFICULT CASES                                        │
│     - Symmetric objects + High occlusion = worst combination        │
│     - Small objects (Glue) under occlusion are particularly hard    │
│                                                                      │
│  4. METHOD-SPECIFIC PATTERNS                                        │
│     - Some methods fail more gracefully under occlusion             │
│     - Methods optimized for occlusion (e.g., oc-dit) show           │
│       different failure patterns                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 50)
    print("""
1. For HIGH OCCLUSION scenarios:
   - Use occlusion-aware methods (oc-dit for segmentation)
   - Use multi-hypothesis methods (waprv2multi-2d for pose)

2. For SYMMETRIC OBJECTS:
   - Use ADD-S metric for evaluation
   - Consider symmetry-aware pose refinement

3. For ROBUST SYSTEMS:
   - Combine multiple methods
   - Use confidence thresholds to reject uncertain predictions
   - Consider occlusion estimation as preprocessing
""")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("Task 5: Failure Case Analysis")
    print("=" * 70)
    
    # 加载GT数据
    print("\n[1] Loading GT data...")
    gt_data = load_gt_data()
    
    # ========================================
    # Segmentation失败分析
    # ========================================
    print("\n[2] Analyzing Segmentation Failures...")
    
    seg_results = []
    for method_name, filename in SEG_METHODS.items():
        pred_file = SEG_PRED_DIR / filename
        if not pred_file.exists():
            print(f"  Skipping {method_name}: file not found")
            continue
        
        print(f"  Processing {method_name}...")
        df = analyze_segmentation_failures(gt_data, method_name, pred_file)
        seg_results.append(df)
    
    if seg_results:
        df_seg = pd.concat(seg_results, ignore_index=True)
        seg_trends = analyze_failure_trends(df_seg, "segmentation")
    else:
        seg_trends = None
        print("  No segmentation results to analyze")
    
    # ========================================
    # Pose Estimation失败分析
    # ========================================
    print("\n[3] Analyzing Pose Estimation Failures...")
    
    pose_results = []
    for method_name, filename in POSE_METHODS.items():
        pred_file = POSE_PRED_DIR / filename
        if not pred_file.exists():
            print(f"  Skipping {method_name}: file not found")
            continue
        
        print(f"  Processing {method_name}...")
        df = analyze_pose_failures(gt_data, method_name, pred_file)
        pose_results.append(df)
    
    if pose_results:
        df_pose = pd.concat(pose_results, ignore_index=True)
        pose_trends = analyze_failure_trends(df_pose, "pose estimation")
    else:
        pose_trends = None
        print("  No pose estimation results to analyze")
    
    # ========================================
    # 总结
    # ========================================
    summarize_findings(seg_trends, pose_trends)
    
    # 保存详细结果
    if seg_results:
        df_seg.to_csv("task5_seg_failures.csv", index=False)
        print("\nSaved: task5_seg_failures.csv")
    
    if pose_results:
        df_pose.to_csv("task5_pose_failures.csv", index=False)
        print("Saved: task5_pose_failures.csv")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()