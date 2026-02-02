"""
Task 3: LM-O Pose Estimation Analysis with K-Means Clustering
- 使用K-Means对visib_fract进行聚类（3类）
- 分析遮挡与位姿估计置信度的相关性
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans

# ============== 配置 ==============
LMO_BASE = "./lmo"
RESULTS_CSV = "pose/waprv2multi-2d-detections_lmo-test.csv"
GT_INFO_PATH = os.path.join(LMO_BASE, "test/000002/scene_gt_info.json")
GT_PATH = os.path.join(LMO_BASE, "test/000002/scene_gt.json")
OUTPUT_DIR = "./lmo_pose_analysis"

# ============== 数据加载与匹配 ==============
def load_and_match():
    """加载CSV结果并与GT可见性信息匹配"""
    # 加载数据
    df = pd.read_csv(RESULTS_CSV)
    gt_info = json.load(open(GT_INFO_PATH))
    gt_poses = json.load(open(GT_PATH))
    
    print(f"Loaded {len(df)} predictions")
    
    # 每个(scene, image, object)只取最高分的预测
    df = df.sort_values('score', ascending=False)
    df = df.groupby(['scene_id', 'im_id', 'obj_id']).head(1).reset_index(drop=True)
    
    matched = []
    for _, row in df.iterrows():
        im_id = str(int(row['im_id']))
        obj_id = int(row['obj_id'])
        
        if im_id not in gt_info:
            continue
        
        gt_instances = gt_info[im_id]
        gt_pose_list = gt_poses.get(im_id, [])
        
        # 找到对应obj_id的GT实例
        for idx, pose in enumerate(gt_pose_list):
            if pose.get('obj_id') == obj_id and idx < len(gt_instances):
                visib = gt_instances[idx].get('visib_fract')
                if visib is not None:
                    matched.append({
                        'im_id': im_id,
                        'obj_id': obj_id,
                        'score': row['score'],
                        'visib_fract': visib
                    })
                break
    
    print(f"Matched {len(matched)} prediction-GT pairs")
    return pd.DataFrame(matched)

# ============== K-Means聚类 ==============
def cluster_by_visibility(df):
    """使用K-Means对visib_fract聚类"""
    visib = df['visib_fract'].values.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(visib)
    centers = kmeans.cluster_centers_.flatten()
    
    # 根据聚类中心排序：中心越大 -> low occlusion
    order = np.argsort(centers)[::-1]
    label_names = ['low', 'medium', 'heavy']
    label_map = {order[i]: label_names[i] for i in range(3)}
    
    df['cluster'] = [label_map[l] for l in labels]
    
    print(f"\nK-Means Cluster Centers (visib_fract):")
    for i, name in enumerate(label_names):
        print(f"  {name}: center={centers[order[i]]:.3f}")
    
    return df, centers, order

# ============== 统计分析 ==============
def compute_statistics(df):
    """按聚类分组计算统计"""
    stats_dict = {}
    for level in ['low', 'medium', 'heavy']:
        data = df[df['cluster'] == level]
        if len(data) > 0:
            scores = data['score'].values
            stats_dict[level] = {
                'count': len(data),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'high_conf_rate': np.mean(scores > 0.5) * 100,
                'very_high_conf_rate': np.mean(scores > 0.7) * 100
            }
        else:
            stats_dict[level] = {'count': 0, 'mean_score': 0, 'std_score': 0, 
                                'high_conf_rate': 0, 'very_high_conf_rate': 0}
    return stats_dict

def compute_correlation(df):
    """计算visib_fract与score的相关性"""
    visib = df['visib_fract'].values
    scores = df['score'].values
    pearson_r, pearson_p = stats.pearsonr(visib, scores)
    spearman_r, spearman_p = stats.spearmanr(visib, scores)
    return {
        'pearson': {'r': pearson_r, 'p': pearson_p},
        'spearman': {'r': spearman_r, 'p': spearman_p}
    }

# ============== 可视化 ==============
def plot_results(df, statistics, correlation):
    """生成两张图：boxplot和scatter"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    levels = ['low', 'medium', 'heavy']
    colors = {'low': '#27ae60', 'medium': '#f39c12', 'heavy': '#e74c3c'}
    
    # 图1: 散点图 + Pearson相关性
    fig, ax = plt.subplots(figsize=(10, 7))
    for level in levels:
        data = df[df['cluster'] == level]
        if len(data) > 0:
            ax.scatter(data['visib_fract'], data['score'], 
                      c=colors[level], alpha=0.5, s=20, label=f'{level.capitalize()} Occlusion')
    
    # 趋势线
    z = np.polyfit(df['visib_fract'], df['score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), 'k--', lw=2, label=f'Trend line')
    
    ax.set_xlabel('Visibility Fraction', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title(f'Score vs Visibility\nPearson r={correlation["pearson"]["r"]:.4f}, p={correlation["pearson"]["p"]:.2e}', 
                fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_pearson.png'), dpi=150)
    plt.close()
    
    # 图2: 箱线图
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [df[df['cluster'] == l]['score'].values for l in levels]
    counts = [statistics[l]['count'] for l in levels]
    
    bp = ax.boxplot(box_data, labels=[f'{l.capitalize()}\n(n={c})' for l, c in zip(levels, counts)], 
                   patch_artist=True)
    for patch, level in zip(bp['boxes'], levels):
        patch.set_facecolor(colors[level])
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Occlusion Level (by K-Means)', fontsize=12)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Score Distribution by Occlusion Cluster', fontsize=14)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {OUTPUT_DIR}/")

# ============== Main ==============
def main():
    print("="*60)
    print("Task 3: LM-O Pose Estimation Analysis with K-Means")
    print("="*60)
    
    # 1. 加载并匹配数据
    df = load_and_match()
    
    # 2. K-Means聚类
    df, centers, order = cluster_by_visibility(df)
    
    # 3. 计算统计
    statistics = compute_statistics(df)
    correlation = compute_correlation(df)
    
    # 4. 打印结果
    print("\n" + "="*60)
    print("Statistics by Occlusion Cluster:")
    print("="*60)
    for level in ['low', 'medium', 'heavy']:
        s = statistics[level]
        print(f"{level.upper():8s}: n={s['count']:4d}, Mean Score={s['mean_score']:.4f} ± {s['std_score']:.4f}, "
              f"High Conf(>0.5)={s['high_conf_rate']:.1f}%")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Pearson:  r={correlation['pearson']['r']:.4f}, p={correlation['pearson']['p']:.2e}")
    print(f"  Spearman: r={correlation['spearman']['r']:.4f}, p={correlation['spearman']['p']:.2e}")
    
    # 5. 可视化
    plot_results(df, statistics, correlation)
    
    # 6. 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({'statistics': statistics, 'correlation': correlation}, f, indent=2)
    
    df.to_csv(os.path.join(OUTPUT_DIR, 'matched_data.csv'), index=False)
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()