import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from scipy import stats
from sklearn.cluster import KMeans


# ============== 配置 ==============
LMO_BASE = "./lmo"
PRED_PATH = "segmentation/3pt-segmentation_lmo-test.json"
SCENE_GT_PATH = os.path.join(LMO_BASE, "test/000002/scene_gt.json")
SCENE_GT_INFO_PATH = os.path.join(LMO_BASE, "test/000002/scene_gt_info.json")
MASK_DIR = os.path.join(LMO_BASE, "test/000002/mask_visib")
OUTPUT_DIR = "./lmo_occlusion"


# ============== 工具函数 ==============
def decode_rle(rle_data):
    """解码RLE格式mask（支持字符串和list两种格式）"""
    if isinstance(rle_data, dict):
        counts = rle_data['counts']
        h, w = rle_data.get('size', [480, 640])
    else:
        counts = rle_data
        h, w = 480, 640
    
    # 字符串格式RLE (COCO compressed)
    if isinstance(counts, str):
        # 解码LEB128压缩格式
        cnts = []
        p = 0
        while p < len(counts):
            x = 0
            k = 0
            more = True
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
    
    # list格式RLE解码
    mask = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        if c > 0:
            mask[pos:pos+c] = val
        pos += c
        val = 1 - val
    return mask.reshape((h, w), order='F')


def compute_iou(mask1, mask2):
    """计算IoU"""
    inter = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    return inter / union if union > 0 else 0.0


def load_mask(image_id, gt_idx):
    """加载GT mask"""
    path = os.path.join(MASK_DIR, f"{int(image_id):06d}_{int(gt_idx):06d}.png")
    if os.path.exists(path):
        return (np.array(Image.open(path)) > 0).astype(np.uint8)
    return None


# ============== 主分析流程 ==============
def match_predictions():
    """匹配预测与GT，返回结果列表"""
    predictions = json.load(open(PRED_PATH))
    scene_gt = json.load(open(SCENE_GT_PATH))
    scene_gt_info = json.load(open(SCENE_GT_INFO_PATH))
    
    # 按图像分组预测
    preds_by_img = defaultdict(list)
    for p in predictions:
        preds_by_img[p['image_id']].append(p)
    
    results = []
    for img_id, preds in preds_by_img.items():
        gt_list = scene_gt.get(str(img_id), [])
        gt_info = scene_gt_info.get(str(img_id), [])
        gt_matched = [False] * len(gt_list)
        
        # 按置信度排序
        for pred in sorted(preds, key=lambda x: -x.get('score', 0)):
            pred_mask = decode_rle(pred['segmentation'])
            if pred_mask is None:
                continue
            
            # 找最佳匹配GT
            best_iou, best_idx, best_visib = 0, -1, 0
            for i, gt in enumerate(gt_list):
                if gt['obj_id'] != pred['category_id'] or gt_matched[i]:
                    continue
                gt_mask = load_mask(img_id, i)
                if gt_mask is None:
                    continue
                iou = compute_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
                    best_visib = gt_info[i].get('visib_fract', 1.0)
            
            if best_idx >= 0:
                gt_matched[best_idx] = True
                results.append({
                    'image_id': img_id,
                    'iou': best_iou,
                    'visib_fract': best_visib
                })
    
    print(f"Matched {len(results)} prediction-GT pairs")
    return results


def cluster_by_occlusion(results):
    """使用K-Means对visib_fract聚类"""
    visib = np.array([r['visib_fract'] for r in results]).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visib)
    centers = kmeans.cluster_centers_.flatten()
    
    # 根据聚类中心排序：中心越大 -> low occlusion
    order = np.argsort(centers)[::-1]  # 从大到小
    label_names = ['low', 'medium', 'heavy']
    label_map = {order[i]: label_names[i] for i in range(3)}
    
    for i, r in enumerate(results):
        r['cluster'] = label_map[labels[i]]
    
    print(f"\nK-Means Cluster Centers (visib_fract):")
    for i, name in enumerate(label_names):
        orig_idx = order[i]
        print(f"  {name}: center={centers[orig_idx]:.3f}")
    
    return results, centers, order


def compute_statistics(results):
    """按聚类分组计算统计"""
    groups = defaultdict(list)
    for r in results:
        groups[r['cluster']].append(r)
    
    stats = {}
    for level in ['low', 'medium', 'heavy']:
        data = groups[level]
        if data:
            ious = [r['iou'] for r in data]
            stats[level] = {
                'count': len(data),
                'mean_iou': np.mean(ious),
                'std_iou': np.std(ious),
                'detection_rate': np.mean(np.array(ious) >= 0.5)
            }
        else:
            stats[level] = {'count': 0, 'mean_iou': 0, 'std_iou': 0, 'detection_rate': 0}
    return stats


def compute_correlation(results):
    """计算visib_fract与IoU的相关性"""
    visib = [r['visib_fract'] for r in results]
    ious = [r['iou'] for r in results]
    pearson_r, pearson_p = stats.pearsonr(visib, ious)
    spearman_r, spearman_p = stats.spearmanr(visib, ious)
    return {
        'pearson': {'r': pearson_r, 'p': pearson_p},
        'spearman': {'r': spearman_r, 'p': spearman_p}
    }


# ============== 可视化 ==============
def plot_results(results, statistics, correlation, centers, order):
    """生成可视化图"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 图1: 分组对比
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    levels = ['low', 'medium', 'heavy']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # 获取每个cluster的visib_fract范围
    cluster_ranges = {}
    for level in levels:
        visibs = [r['visib_fract'] for r in results if r['cluster'] == level]
        if visibs:
            cluster_ranges[level] = (min(visibs), max(visibs))
        else:
            cluster_ranges[level] = (0, 0)
    
    labels = [f'{l.capitalize()}\n({cluster_ranges[l][0]:.2f}-{cluster_ranges[l][1]:.2f})' 
              for l in levels]
    
    # Mean IoU
    mean_ious = [statistics[l]['mean_iou'] for l in levels]
    std_ious = [statistics[l]['std_iou'] for l in levels]
    bars = axes[0].bar(labels, mean_ious, yerr=std_ious, capsize=5, color=colors)
    axes[0].set_ylabel('Mean IoU')
    axes[0].set_title('Segmentation Accuracy by Cluster')
    axes[0].set_ylim(0, 1.1)
    for bar, v in zip(bars, mean_ious):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{v:.3f}', ha='center', fontweight='bold')
    
    # Detection Rate
    det_rates = [statistics[l]['detection_rate'] * 100 for l in levels]
    bars = axes[1].bar(labels, det_rates, color=colors)
    axes[1].set_ylabel('Detection Rate (%)')
    axes[1].set_title('Detection Rate (IoU≥0.5)')
    axes[1].set_ylim(0, 110)
    for bar, v in zip(bars, det_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Sample Count
    counts = [statistics[l]['count'] for l in levels]
    bars = axes[2].bar(labels, counts, color=colors)
    axes[2].set_ylabel('Sample Count')
    axes[2].set_title('Sample Distribution')
    for bar, v in zip(bars, counts):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02, 
                    str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison.png'), dpi=150)
    plt.close()
    
    # 图2: 散点图
    fig, ax = plt.subplots(figsize=(10, 7))
    color_map = {'low': '#2ecc71', 'medium': '#f39c12', 'heavy': '#e74c3c'}
    
    for level in levels:
        data = [r for r in results if r['cluster'] == level]
        if data:
            ax.scatter([r['visib_fract'] for r in data], [r['iou'] for r in data],
                      c=color_map[level], alpha=0.6, s=30, label=f'{level.capitalize()} Occlusion')
    
    # 趋势线
    all_visib = [r['visib_fract'] for r in results]
    all_ious = [r['iou'] for r in results]
    z = np.polyfit(all_visib, all_ious, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), 'k--', lw=2, label=f'Trend (r={correlation["pearson"]["r"]:.3f})')
    
    # 标记聚类边界
    sorted_centers = np.sort(centers)
    for c in sorted_centers[:-1]:
        # 找相邻两个center的中点作为边界
        pass
    
    ax.set_xlabel('Visibility Fraction')
    ax.set_ylabel('IoU')
    ax.set_title(f'IoU vs Visibility (Pearson r={correlation["pearson"]["r"]:.4f}, p={correlation["pearson"]["p"]:.2e})')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scatter.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {OUTPUT_DIR}/")


# ============== Main ==============
def main():
    print("="*60)
    print("Task 2: LM-O Occlusion Analysis with K-Means Clustering")
    print("="*60)
    
    # 1. 匹配预测与GT
    results = match_predictions()
    
    # 2. K-Means聚类
    results, centers, order = cluster_by_occlusion(results)
    
    # 3. 计算统计
    statistics = compute_statistics(results)
    correlation = compute_correlation(results)
    
    # 4. 打印结果
    print("\n" + "="*60)
    print("Statistics by Occlusion Cluster:")
    print("="*60)
    for level in ['low', 'medium', 'heavy']:
        s = statistics[level]
        print(f"{level.upper():8s}: n={s['count']:4d}, Mean IoU={s['mean_iou']:.4f} ± {s['std_iou']:.4f}, Det Rate={s['detection_rate']*100:.1f}%")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson:  r={correlation['pearson']['r']:.4f}, p={correlation['pearson']['p']:.2e}")
    print(f"  Spearman: r={correlation['spearman']['r']:.4f}, p={correlation['spearman']['p']:.2e}")
    
    # 5. 可视化
    plot_results(results, statistics, correlation, centers, order)
    
    # 6. 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({'statistics': statistics, 'correlation': correlation}, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()