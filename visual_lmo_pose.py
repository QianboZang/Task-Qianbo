"""
三种6D Pose算法对比可视化 (WAPR.v2 vs FreeZe.v2 vs SAM6D)

直接运行: python compare_methods.py

依赖:
    pip install numpy opencv-python trimesh pandas
"""

import os
import json
import random
import numpy as np
import pandas as pd
import cv2
import trimesh

# ==================== 配置 ====================
LMO_PATH = "./lmo"
OUTPUT_DIR = "./lmo_compare_pose"

# 三个算法的预测结果
CSV_PATHS = {
    "WAPR.v2": "./pose/waprv2multi-2d-detections_lmo-test.csv",
    "FreeZe.v2": "./pose/freezev21_lmo-test.csv",
    "SAM6D": "./pose/sam6d_lmo-test.csv",
}

# 采样参数
NUM_SAMPLES = 10           # 采样图像数量
RANDOM_SEED = 42           # 随机种子，保证可复现
SCORE_THRESHOLD = 0.0      # 置信度阈值（设为0显示所有预测）
TOP_K = 1                  # 每个物体显示top-k个预测
# ==============================================


# LM-O物体名称映射
LMO_OBJECTS = {
    1: "Ape", 5: "Can", 6: "Cat", 8: "Driller",
    9: "Duck", 10: "Eggbox", 11: "Glue", 12: "Holepuncher"
}

# 物体颜色 (BGR)
OBJECT_COLORS = {
    1: (255, 0, 0), 5: (0, 255, 0), 6: (0, 0, 255), 8: (255, 255, 0),
    9: (255, 0, 255), 10: (0, 255, 255), 11: (128, 0, 255), 12: (255, 128, 0)
}


def load_csv(path):
    """
    加载CSV文件，自动处理不同格式
    """
    # 先读取第一行判断是否有header
    with open(path, 'r') as f:
        first_line = f.readline().strip()
    
    # 标准列名
    columns = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
    
    # 判断是否有header（检查第一个字段是否是数字）
    first_field = first_line.split(',')[0]
    has_header = not first_field.isdigit()
    
    if has_header:
        df = pd.read_csv(path)
    else:
        # SAM6D格式：没有header
        df = pd.read_csv(path, header=None, names=columns)
    
    # 确保列名标准化
    df.columns = [c.strip() for c in df.columns]
    
    # 确保数值类型正确
    df['scene_id'] = df['scene_id'].astype(int)
    df['im_id'] = df['im_id'].astype(int)
    df['obj_id'] = df['obj_id'].astype(int)
    df['score'] = df['score'].astype(float)
    
    return df


def load_camera_intrinsics(scene_id, im_id):
    scene_camera_path = os.path.join(LMO_PATH, f"test/{scene_id:06d}/scene_camera.json")
    with open(scene_camera_path, 'r') as f:
        scene_camera = json.load(f)
    cam_info = scene_camera[str(im_id)]
    return np.array(cam_info['cam_K']).reshape(3, 3)


def load_object_model(obj_id):
    model_path = os.path.join(LMO_PATH, f"models/obj_{obj_id:06d}.ply")
    return trimesh.load(model_path)


def load_rgb_image(scene_id, im_id):
    img_path = os.path.join(LMO_PATH, f"test/{scene_id:06d}/rgb/{im_id:06d}.png")
    return cv2.imread(img_path)


def parse_pose(row):
    """解析R和t，兼容不同格式"""
    R_str = str(row['R']).strip()
    t_str = str(row['t']).strip()
    
    # 解析R (9个值)
    R_values = [float(x) for x in R_str.split()]
    R = np.array(R_values).reshape(3, 3)
    
    # 解析t (3个值)
    t_values = [float(x) for x in t_str.split()]
    t = np.array(t_values).reshape(3, 1)
    
    return R, t


def project_points(points_3d, K, R, t):
    points_cam = (R @ points_3d.T + t).T
    points_2d = (K @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    return points_2d


def render_bbox_3d(img, mesh, K, R, t, color, thickness=2):
    """渲染3D包围盒"""
    bbox = mesh.bounding_box.vertices
    points_2d = project_points(bbox, K, R, t).astype(np.int32)
    
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        cv2.line(img, tuple(points_2d[edge[0]]), tuple(points_2d[edge[1]]), color, thickness)
    return img


def render_predictions(img, predictions_df, models, K, show_label=True):
    """在图像上渲染所有预测"""
    img = img.copy()
    
    for obj_id in predictions_df['obj_id'].unique():
        obj_preds = predictions_df[predictions_df['obj_id'] == obj_id].nlargest(TOP_K, 'score')
        
        if obj_id not in models:
            continue
        
        mesh = models[obj_id]
        color = OBJECT_COLORS.get(obj_id, (0, 255, 0))
        
        for _, pred in obj_preds.iterrows():
            R, t = parse_pose(pred)
            score = pred['score']
            
            img = render_bbox_3d(img, mesh, K, R, t, color)
            
            if show_label:
                obj_name = LMO_OBJECTS.get(obj_id, f"obj_{obj_id}")
                label = f"{obj_name}: {score:.2f}"
                center_2d = project_points(np.array([[0, 0, 0]]), K, R, t)[0].astype(int)
                cv2.putText(img, label, (center_2d[0], center_2d[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return img


def add_title(img, title, font_scale=0.8, thickness=2):
    """在图像顶部添加标题"""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return img


def create_comparison_image(scene_id, im_id, all_predictions, models):
    """创建单张对比图"""
    img_original = load_rgb_image(scene_id, im_id)
    if img_original is None:
        return None
    
    K = load_camera_intrinsics(scene_id, im_id)
    h, w = img_original.shape[:2]
    
    images = []
    
    # 原图
    img_orig = img_original.copy()
    img_orig = add_title(img_orig, "Original")
    images.append(img_orig)
    
    # 三种方法的预测
    for method_name, df in all_predictions.items():
        img_pred = img_original.copy()
        
        preds = df[(df['scene_id'] == scene_id) & (df['im_id'] == im_id)]
        preds = preds[preds['score'] >= SCORE_THRESHOLD]
        
        if len(preds) > 0:
            img_pred = render_predictions(img_pred, preds, models, K)
        
        img_pred = add_title(img_pred, f"{method_name} ({len(preds)} preds)")
        images.append(img_pred)
    
    # 2x2 拼接
    row1 = np.hstack([images[0], images[1]])
    row2 = np.hstack([images[2], images[3]])
    combined = np.vstack([row1, row2])
    
    return combined


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    
    # 加载所有预测结果
    print("Loading predictions...")
    all_predictions = {}
    for name, path in CSV_PATHS.items():
        print(f"  Loading {name} from {path}")
        df = load_csv(path)
        all_predictions[name] = df
        print(f"    -> {len(df)} predictions, columns: {list(df.columns)}")
    
    # 找到所有方法共有的图像
    print("\nFinding common images...")
    image_sets = []
    for name, df in all_predictions.items():
        images = set(zip(df['scene_id'], df['im_id']))
        image_sets.append(images)
        print(f"  {name}: {len(images)} unique images")
    
    common_images = list(set.intersection(*image_sets))
    print(f"  Common images: {len(common_images)}")
    
    if len(common_images) == 0:
        print("\nError: No common images found! Checking image ranges...")
        for name, df in all_predictions.items():
            print(f"  {name}: scene_id={df['scene_id'].unique()}, im_id range=[{df['im_id'].min()}, {df['im_id'].max()}]")
        return
    
    # 采样
    if len(common_images) > NUM_SAMPLES:
        sampled_images = random.sample(common_images, NUM_SAMPLES)
    else:
        sampled_images = common_images
    
    print(f"\nSampled {len(sampled_images)} images for comparison")
    
    # 预加载所有需要的物体模型
    all_obj_ids = set()
    for df in all_predictions.values():
        all_obj_ids.update(df['obj_id'].unique())
    
    print("\nLoading 3D models...")
    models = {}
    for obj_id in all_obj_ids:
        try:
            models[obj_id] = load_object_model(obj_id)
            print(f"  Loaded obj_{obj_id:06d} ({LMO_OBJECTS.get(obj_id, 'Unknown')})")
        except Exception as e:
            print(f"  Warning: Could not load obj_{obj_id}: {e}")
    
    # 生成对比图
    print(f"\nGenerating comparison images...")
    for i, (scene_id, im_id) in enumerate(sampled_images):
        print(f"  [{i+1}/{len(sampled_images)}] scene_{scene_id:06d}_im_{im_id:06d}")
        
        try:
            combined = create_comparison_image(scene_id, im_id, all_predictions, models)
            
            if combined is not None:
                output_path = os.path.join(OUTPUT_DIR, f"compare_{scene_id:06d}_{im_id:06d}.png")
                cv2.imwrite(output_path, combined)
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\nDone! Results saved to {OUTPUT_DIR}/")
    print(f"Open the images to compare: WAPR.v2 vs FreeZe.v2 vs SAM6D")


if __name__ == '__main__':
    main()