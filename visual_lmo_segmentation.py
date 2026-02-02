"""
Task 2: LM-O Segmentation Visualization
- 对比5个方法的分割结果
- 每张图显示: RGB, GT, 5个方法的预测, GT mask, 5个方法的IoU对比
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

# ============== 配置 ==============
LMO_BASE = "./lmo"
SEGMENTATION_DIR = "./segmentation"
RGB_DIR = os.path.join(LMO_BASE, "test/000002/rgb")
MASK_DIR = os.path.join(LMO_BASE, "test/000002/mask_visib")
SCENE_GT_PATH = os.path.join(LMO_BASE, "test/000002/scene_gt.json")
SCENE_GT_INFO_PATH = os.path.join(LMO_BASE, "test/000002/scene_gt_info.json")
OUTPUT_DIR = "./lmo_compare"

# 要对比的方法
METHODS = ['3pt-segmentation', 'oc-dit', 'noctis', 'sam6drgb', 'zeropose']

# 要可视化的样本数量
NUM_SAMPLES = 12

# ============== 工具函数 ==============
def decode_rle(rle_data):
    """解码RLE - 使用pycocotools"""
    try:
        from pycocotools import mask as mask_utils
        if isinstance(rle_data, dict):
            # 确保格式正确
            rle = {
                'counts': rle_data['counts'],
                'size': rle_data.get('size', [480, 640])
            }
            return mask_utils.decode(rle)
    except:
        pass
    
    # Fallback: 手动解码
    if isinstance(rle_data, dict):
        counts = rle_data['counts']
        h, w = rle_data.get('size', [480, 640])
    else:
        counts = rle_data
        h, w = 480, 640
    
    if isinstance(counts, str):
        cnts = []
        p = 0
        while p < len(counts):
            x, k, more = 0, 0, True
            while more:
                c = ord(counts[p]) - 48
                x |= (c & 0x1f) << (5 * k)
                more = c > 31
                p += 1
                k += 1
                if p > len(counts):
                    break
            if k > 2 and (x & (1 << (5*k - 1))):
                x |= -1 << (5*k)
            if len(cnts) > 2:
                x += cnts[-2]
            cnts.append(x)
        counts = cnts
    
    mask = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        if c > 0:
            mask[pos:pos+c] = val
        pos += c
        val = 1 - val
    return mask.reshape((h, w), order='F')

def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return inter / union if union > 0 else 0.0

def overlay_mask(img, mask, color, alpha=0.5):
    """在图像上叠加mask"""
    result = img.copy()
    mask_bool = mask > 0
    for c in range(3):
        result[:,:,c] = np.where(mask_bool, 
                                  result[:,:,c] * (1-alpha) + color[c] * alpha,
                                  result[:,:,c])
    return result.astype(np.uint8)

def create_comparison_mask(pred_mask, gt_mask):
    """创建对比mask: Green=Pred only, Red=GT only, Yellow=Overlap"""
    h, w = gt_mask.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    overlap = pred_bool & gt_bool
    pred_only = pred_bool & ~gt_bool
    gt_only = gt_bool & ~pred_bool
    
    result[overlap] = [255, 255, 0]    # Yellow
    result[pred_only] = [0, 255, 0]    # Green
    result[gt_only] = [255, 0, 0]      # Red
    
    return result

# ============== 主函数 ==============
def load_all_predictions():
    """加载所有方法的预测结果"""
    all_preds = {}
    for method in METHODS:
        json_path = os.path.join(SEGMENTATION_DIR, f"{method}_lmo-test.json")
        if os.path.exists(json_path):
            preds = json.load(open(json_path))
            # 按(image_id, obj_id)索引
            preds_dict = defaultdict(list)
            for p in preds:
                preds_dict[(p['image_id'], p['category_id'])].append(p)
            all_preds[method] = preds_dict
            print(f"Loaded {method}: {len(preds)} predictions")
        else:
            print(f"Warning: {json_path} not found")
    return all_preds

def visualize_sample(image_id, obj_id, all_preds, scene_gt, scene_gt_info):
    """为单个样本生成可视化"""
    # 加载RGB图像
    rgb_path = os.path.join(RGB_DIR, f"{image_id:06d}.png")
    rgb = np.array(Image.open(rgb_path))
    
    # 获取GT信息
    gt_list = scene_gt.get(str(image_id), [])
    gt_info_list = scene_gt_info.get(str(image_id), [])
    
    # 找到对应obj_id的GT
    gt_idx = None
    visib_fract = None
    for i, gt in enumerate(gt_list):
        if gt['obj_id'] == obj_id:
            gt_idx = i
            visib_fract = gt_info_list[i].get('visib_fract', 1.0)
            break
    
    if gt_idx is None:
        print(f"  GT not found for image {image_id}, obj {obj_id}")
        return
    
    # 加载GT mask
    gt_mask_path = os.path.join(MASK_DIR, f"{image_id:06d}_{gt_idx:06d}.png")
    gt_mask = (np.array(Image.open(gt_mask_path)) > 0).astype(np.uint8)
    
    # 获取每个方法的预测
    method_results = {}
    for method in METHODS:
        if method not in all_preds:
            continue
        preds = all_preds[method].get((image_id, obj_id), [])
        if preds:
            # 取最高分的预测
            best_pred = max(preds, key=lambda x: x.get('score', 0))
            pred_mask = decode_rle(best_pred['segmentation'])
            iou = compute_iou(pred_mask, gt_mask)
            method_results[method] = {
                'mask': pred_mask,
                'score': best_pred.get('score', 0),
                'iou': iou
            }
    
    # 创建图
    n_methods = len(METHODS)
    fig = plt.figure(figsize=(4 * (n_methods + 2), 8))
    
    # 第一行: RGB, GT overlay, 各方法overlay
    # 第二行: 空, GT mask, 各方法comparison mask
    
    # Row 1, Col 1: RGB
    ax = fig.add_subplot(2, n_methods + 2, 1)
    ax.imshow(rgb)
    ax.set_title(f'RGB Image (id={image_id})', fontsize=10)
    ax.axis('off')
    
    # Row 1, Col 2: Ground Truth overlay
    ax = fig.add_subplot(2, n_methods + 2, 2)
    gt_overlay = overlay_mask(rgb, gt_mask, [139, 0, 0], alpha=0.5)  # Dark red
    ax.imshow(gt_overlay)
    ax.set_title(f'Ground Truth\nobj_id={obj_id}, visib={visib_fract:.3f}', fontsize=10)
    ax.axis('off')
    
    # Row 1, Col 3+: 各方法overlay
    for i, method in enumerate(METHODS):
        ax = fig.add_subplot(2, n_methods + 2, 3 + i)
        if method in method_results:
            pred_mask = method_results[method]['mask']
            score = method_results[method]['score']
            pred_overlay = overlay_mask(rgb, pred_mask, [0, 255, 0], alpha=0.5)
            ax.imshow(pred_overlay)
            ax.set_title(f'{method}\nscore={score:.4f}', fontsize=10, color='green')
        else:
            ax.imshow(rgb)
            ax.set_title(f'{method}\nNo prediction', fontsize=10, color='red')
        ax.axis('off')
    
    # Row 2, Col 1: IoU Comparison box
    ax = fig.add_subplot(2, n_methods + 2, n_methods + 3)
    ax.axis('off')
    
    # 创建IoU对比文本
    text_lines = ['IoU Comparison\n']
    ious = {}
    for method in METHODS:
        if method in method_results:
            iou = method_results[method]['iou']
            ious[method] = iou
            text_lines.append(f'{method}: {iou:.4f}')
    
    # 找最佳方法
    if ious:
        best_method = max(ious, key=ious.get)
        second_best = sorted(ious.values(), reverse=True)[1] if len(ious) > 1 else 0
        diff = ious[best_method] - second_best
        text_lines.append(f'\n{best_method} wins by {diff:.4f}')
    
    ax.text(0.5, 0.5, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 2, Col 2: GT Mask
    ax = fig.add_subplot(2, n_methods + 2, n_methods + 4)
    gt_mask_display = np.ones((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8) * 255
    gt_mask_display[gt_mask > 0] = [139, 0, 0]
    ax.imshow(gt_mask_display)
    ax.set_title('GT Mask', fontsize=10)
    ax.axis('off')
    
    # Row 2, Col 3+: 各方法comparison mask
    for i, method in enumerate(METHODS):
        ax = fig.add_subplot(2, n_methods + 2, n_methods + 5 + i)
        if method in method_results:
            pred_mask = method_results[method]['mask']
            iou = method_results[method]['iou']
            comp_mask = create_comparison_mask(pred_mask, gt_mask)
            ax.imshow(comp_mask)
            ax.set_title(f'{method} vs GT\nIoU={iou:.4f}', fontsize=10, color='green')
        else:
            ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            ax.set_title(f'{method} vs GT\nNo prediction', fontsize=10, color='red')
        ax.axis('off')
    
    plt.suptitle(f'LM-O Segmentation Comparison - Image {image_id}, Object {obj_id}', 
                 fontsize=14, fontweight='bold')
    
    # 添加图例
    fig.text(0.5, 0.02, 'Comparison: Green=Pred only, Red=GT only, Yellow=Overlap', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, f'lmo_compare_{image_id}_obj{obj_id}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def find_valid_samples(all_preds, scene_gt, num_samples=12):
    """找出所有方法都有预测的样本"""
    # 获取每个方法的(image_id, obj_id)集合
    method_keys = {}
    for method, preds_dict in all_preds.items():
        method_keys[method] = set(preds_dict.keys())
    
    # 找交集：所有方法都有预测的样本
    if not method_keys:
        return []
    
    common_keys = set.intersection(*method_keys.values())
    
    # 过滤：只保留GT中存在的样本
    valid_samples = []
    for (img_id, obj_id) in common_keys:
        gt_list = scene_gt.get(str(img_id), [])
        for gt in gt_list:
            if gt['obj_id'] == obj_id:
                valid_samples.append((img_id, obj_id))
                break
    
    # 随机选取
    import random
    random.seed(42)
    if len(valid_samples) > num_samples:
        valid_samples = random.sample(valid_samples, num_samples)
    
    # 排序
    valid_samples.sort()
    return valid_samples

def main():
    print("="*60)
    print("Task 2: LM-O Segmentation Comparison (5 Methods)")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载GT
    scene_gt = json.load(open(SCENE_GT_PATH))
    scene_gt_info = json.load(open(SCENE_GT_INFO_PATH))
    
    # 加载所有预测
    all_preds = load_all_predictions()
    
    # 自动选取有效样本
    samples = find_valid_samples(all_preds, scene_gt, NUM_SAMPLES)
    print(f"\nFound {len(samples)} valid samples (all methods have predictions)")
    
    # 为每个样本生成可视化
    print(f"Generating visualizations...")
    for image_id, obj_id in samples:
        print(f"Processing image {image_id}, obj {obj_id}...")
        visualize_sample(image_id, obj_id, all_preds, scene_gt, scene_gt_info)
    
    print(f"\nAll results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()