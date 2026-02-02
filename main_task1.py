import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from PIL import Image


# =============================================================================
# 第一步：读取 LM-O 数据集
# =============================================================================

def load_lmo_camera(camera_json_path):
    """读取 LM-O 的相机内参"""
    with open(camera_json_path, 'r') as f:
        camera_params = json.load(f)
    
    print("=" * 50)
    print("相机参数 (camera.json):")
    print("=" * 50)
    print(f"  f_x = {camera_params['fx']}")
    print(f"  f_y = {camera_params['fy']}")
    print(f"  c_x = {camera_params['cx']}")
    print(f"  c_y = {camera_params['cy']}")
    print(f"  depth_scale = {camera_params['depth_scale']}")
    print(f"  width = {camera_params['width']}")
    print(f"  height = {camera_params['height']}")
    
    return camera_params


def build_intrinsic_matrix(camera_params):
    """
    构建内参矩阵 K
    
    K = [fx   0  cx]
        [ 0  fy  cy]
        [ 0   0   1]
    """
    f_x = camera_params['fx']
    f_y = camera_params['fy']
    c_x = camera_params['cx']
    c_y = camera_params['cy']
    
    K = np.array([
        [f_x,   0, c_x],
        [  0, f_y, c_y],
        [  0,   0,   1]
    ], dtype=np.float64)
    
    print("\n内参矩阵 K:")
    print(K)
    
    return K


def compute_K_inverse(K):
    """
    计算 K 的逆矩阵
    
    K^(-1) = [1/fx    0    -cx/fx]
             [  0   1/fy   -cy/fy]
             [  0     0       1  ]
    """
    K_inv = np.linalg.inv(K)
    
    print("\nK 的逆矩阵 K^(-1):")
    print(K_inv)
    
    return K_inv


def load_rgb_image(rgb_path):
    """读取 RGB 图像"""
    rgb_image = np.array(Image.open(rgb_path))
    print(f"\nRGB 图像: {rgb_path}")
    print(f"  形状: {rgb_image.shape}, 范围: [{rgb_image.min()}, {rgb_image.max()}]")
    return rgb_image


def load_depth_image(depth_path, depth_scale):
    """读取深度图并转换为米"""
    depth_raw = np.array(Image.open(depth_path))
    depth_in_meters = depth_raw.astype(np.float32) * depth_scale
    
    print(f"\n深度图: {depth_path}")
    print(f"  原始值范围: [{depth_raw.min()}, {depth_raw.max()}]")
    print(f"  × depth_scale ({depth_scale})")
    print(f"  转换后范围: [{depth_in_meters.min():.4f}, {depth_in_meters.max():.4f}] 米")
    
    return depth_in_meters


def load_mask(mask_path):
    """读取物体掩码"""
    mask = np.array(Image.open(mask_path))
    print(f"\n物体掩码: {mask_path}")
    print(f"  形状: {mask.shape}, 物体像素数: {np.sum(mask > 0)}")
    return mask


# =============================================================================
# 第二步：使用 K^(-1) 矩阵进行反投影
# =============================================================================

def rgbd_to_pointcloud_matrix(rgb_image, depth_in_meters, K_inv, camera_params, mask=None):
    """
    使用矩阵运算将 RGBD 图像转换为 3D 彩色点云
    
    数学公式:
        [X]            [u]
        [Y] = Z × K^(-1) × [v]
        [Z]            [1]
    
    参数:
        rgb_image: (H, W, 3) RGB 图像
        depth_in_meters: (H, W) 深度图，单位：米
        K_inv: (3, 3) 内参矩阵的逆
        camera_params: 相机参数字典
        mask: (H, W) 物体掩码，可选
    
    返回:
        points: (N, 3) 3D 点坐标
        colors: (N, 3) RGB 颜色
    """
    H = camera_params['height']
    W = camera_params['width']
    
    print("\n" + "=" * 50)
    print("使用 K^(-1) 矩阵进行反投影")
    print("=" * 50)
    
    # -------------------------------------------------------------------------
    # 第1步：创建像素坐标的齐次形式
    # -------------------------------------------------------------------------
    # u: 列号 (0 到 W-1)
    # v: 行号 (0 到 H-1)
    u_coords = np.arange(W)
    v_coords = np.arange(H)
    u, v = np.meshgrid(u_coords, v_coords)  # 都是 (H, W)
    
    # 创建齐次坐标 [u, v, 1]
    # ones 形状是 (H, W)
    ones = np.ones_like(u)
    
    # 堆叠成 (3, H, W)
    # pixel_coords[0] = u
    # pixel_coords[1] = v
    # pixel_coords[2] = 1
    pixel_coords = np.stack([u, v, ones], axis=0)  # (3, H, W)
    
    print(f"像素齐次坐标形状: {pixel_coords.shape}")  # (3, H, W)
    
    # -------------------------------------------------------------------------
    # 第2步：应用 K^(-1) 矩阵
    # -------------------------------------------------------------------------
    # K_inv: (3, 3)
    # pixel_coords: (3, H, W)
    #
    # 我们需要对每个像素计算: K^(-1) × [u, v, 1]^T
    #
    # 方法：reshape pixel_coords 为 (3, H*W)，做矩阵乘法，再 reshape 回来
    
    pixel_coords_flat = pixel_coords.reshape(3, -1)  # (3, H*W)
    
    # 矩阵乘法: (3, 3) × (3, H*W) = (3, H*W)
    normalized_coords = K_inv @ pixel_coords_flat  # (3, H*W)
    
    # reshape 回 (3, H, W)
    normalized_coords = normalized_coords.reshape(3, H, W)
    
    print(f"归一化坐标形状: {normalized_coords.shape}")  # (3, H, W)
    
    # -------------------------------------------------------------------------
    # 第3步：乘以深度 Z 得到 3D 坐标
    # -------------------------------------------------------------------------
    # normalized_coords[0] = (u - cx) / fx
    # normalized_coords[1] = (v - cy) / fy
    # normalized_coords[2] = 1
    #
    # 3D坐标 = normalized_coords × Z
    
    Z = depth_in_meters  # (H, W)
    
    X = normalized_coords[0] * Z  # (H, W)
    Y = normalized_coords[1] * Z  # (H, W)
    # Z 本身就是深度
    
    print(f"\n3D 坐标计算完成:")
    print(f"  X 形状: {X.shape}")
    print(f"  Y 形状: {Y.shape}")
    print(f"  Z 形状: {Z.shape}")
    
    # -------------------------------------------------------------------------
    # 第4步：组合成点云
    # -------------------------------------------------------------------------
    points = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    
    # -------------------------------------------------------------------------
    # 第5步：处理颜色
    # -------------------------------------------------------------------------
    colors = rgb_image.astype(np.float32)
    if colors.max() > 1:
        colors = colors / 255.0
    
    # -------------------------------------------------------------------------
    # 第6步：应用掩码
    # -------------------------------------------------------------------------
    if mask is not None:
        valid = (mask > 0) & (Z > 0)
    else:
        valid = Z > 0
    
    points = points[valid]  # (N, 3)
    colors = colors[valid]  # (N, 3)
    
    print(f"\n最终点云:")
    print(f"  点数: {len(points)}")
    print(f"  X 范围: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] 米")
    print(f"  Y 范围: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}] 米")
    print(f"  Z 范围: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}] 米")
    
    return points, colors


# =============================================================================
# 第三步：可视化函数
# =============================================================================

def visualize_input_data(rgb_image, depth_image, mask=None, save_path=None):
    """可视化输入数据"""
    n_plots = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    im = axes[1].imshow(depth_image, cmap='plasma')
    axes[1].set_title('Depth Image (meters)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, label='Depth (m)')
    
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Object Mask', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    plt.suptitle('LM-O Dataset Input Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"保存: {save_path}")
    
    plt.show()


def visualize_pointcloud(points, colors, title="3D Point Cloud", 
                         n_samples=10000, save_path=None):
    """可视化彩色点云"""
    
    if len(points) > n_samples:
        indices = np.random.choice(len(points), n_samples, replace=False)
        pts = points[indices]
        cols = colors[indices]
    else:
        pts = points
        cols = colors
    
    fig = plt.figure(figsize=(14, 12))
    
    viewpoints = [
        (0, 0, 'Front View'),
        (0, 90, 'Side View'),
        (25, 135, 'Perspective View'),
        (90, 0, 'Top View')
    ]
    
    for i, (elev, azim, view_title) in enumerate(viewpoints):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], 
                   c=cols, s=2, alpha=1.0, edgecolors='none')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=8)
        ax.set_ylabel('Z (m)', fontsize=10, labelpad=8)
        ax.set_zlabel('Y (m)', fontsize=10, labelpad=8)
        ax.set_title(view_title, fontsize=13, fontweight='bold', pad=15)
        ax.view_init(elev=elev, azim=azim)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, 
                        wspace=0.15, hspace=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"保存: {save_path}")
    
    plt.show()


def save_pointcloud_ply(points, colors, filename):
    """保存点云为 PLY 文件"""
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")
    
    print(f"保存: {filename}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # 配置路径 - 根据你的目录结构
    # =========================================================================
    
    # LM-O 数据集根目录
    LMO_ROOT = "lmo"  # 修改为你的实际路径
    
    # 场景和帧编号
    SCENE_ID = "000002"
    FRAME_ID = "000000"
    OBJECT_ID = "000000"
    
    # 构建文件路径（根据你的目录结构）
    camera_json_path = os.path.join(LMO_ROOT, "lmo", "camera.json")  # lmo/lmo/camera.json
    rgb_path = os.path.join(LMO_ROOT, "test", SCENE_ID, "rgb", f"{FRAME_ID}.png")
    depth_path = os.path.join(LMO_ROOT, "test", SCENE_ID, "depth", f"{FRAME_ID}.png")
    mask_path = os.path.join(LMO_ROOT, "test", SCENE_ID, "mask_visib", f"{FRAME_ID}_{OBJECT_ID}.png")
    
    print("=" * 70)
    print("Task 1: LM-O Dataset - RGBD to 3D Point Cloud (Using K^(-1) Matrix)")
    print("=" * 70)
    print(f"\n文件路径:")
    print(f"  相机参数: {camera_json_path}")
    print(f"  RGB 图像: {rgb_path}")
    print(f"  深度图:   {depth_path}")
    print(f"  物体掩码: {mask_path}")
    
    # =========================================================================
    # 配置输出目录
    # =========================================================================
    OUTPUT_DIR = "lmo_cloud_point"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"结果将保存到: {OUTPUT_DIR}")

    # =========================================================================
    # 第1步：加载相机参数
    # =========================================================================
    camera_params = load_lmo_camera(camera_json_path)
    
    # =========================================================================
    # 第2步：构建内参矩阵 K 和 K^(-1)
    # =========================================================================
    K = build_intrinsic_matrix(camera_params)
    K_inv = compute_K_inverse(K)
    
    # =========================================================================
    # 第3步：加载图像数据
    # =========================================================================
    rgb_image = load_rgb_image(rgb_path)
    depth_in_meters = load_depth_image(depth_path, camera_params['depth_scale'])
    mask = load_mask(mask_path)
    
    # =========================================================================
    # 第4步：可视化输入数据
    # =========================================================================
    visualize_input_data(rgb_image, depth_in_meters, mask, 
                         save_path=os.path.join(OUTPUT_DIR, "lmo_input_data.png"))
    
    # =========================================================================
    # 第5步：使用 K^(-1) 矩阵进行 RGBD 转 3D 点云
    # =========================================================================
    points, colors = rgbd_to_pointcloud_matrix(
        rgb_image, depth_in_meters, K_inv, camera_params, mask
    )
    
    # =========================================================================
    # 第6步：可视化点云
    # =========================================================================
    visualize_pointcloud(
        points, colors, 
        title=f"LM-O Scene {SCENE_ID} Frame {FRAME_ID}\nTarget Object Point Cloud (Using K⁻¹ Matrix)",
        save_path=os.path.join(OUTPUT_DIR, "lmo_pointcloud.png")
    )
    
    # =========================================================================
    # 第7步：保存点云
    # =========================================================================
    save_pointcloud_ply(points, colors, os.path.join(OUTPUT_DIR, "lmo_target_object.ply"))
    
    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 70)
    print("Task 1 完成!")
    print("=" * 70)
    print(f"""
使用的数学公式:

    内参矩阵 K:
    {K}

    K 的逆矩阵 K^(-1):
    {K_inv}

    反投影公式:
    [X]            [u]
    [Y] = Z × K^(-1) × [v]
    [Z]            [1]

输出文件:
  1. lmo_input_data.png    - 输入数据可视化
  2. lmo_pointcloud.png    - 3D 点云可视化
  3. lmo_target_object.ply - 点云文件
""")