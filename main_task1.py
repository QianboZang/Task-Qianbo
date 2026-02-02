"""
Task 1: RGBD to 3D Colored Point Cloud Conversion
==================================================

This script converts RGBD pixels to a 3D colored point cloud using camera intrinsic parameters.

Mathematical Background:
------------------------
Given an RGBD image and camera intrinsics, each pixel (u, v) with depth d can be 
"lifted" (back-projected) to 3D coordinates (X, Y, Z) using:

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

Where:
    - (fx, fy): focal lengths in pixels
    - (cx, cy): principal point (optical center)
    - (u, v): pixel coordinates
    - d: depth value at that pixel

Usage:
------
1. With your own data:
    points, colors = rgbd_to_pointcloud(rgb_image, depth_image, intrinsics, mask)
    
2. Run this script directly to see a demo with synthetic data:
    python task1_rgbd_to_pointcloud.py

Author: Claude (Anthropic)
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Core Function: RGBD to Point Cloud Conversion
# =============================================================================

def rgbd_to_pointcloud(rgb_image, depth_image, intrinsics, mask=None):
    """
    Convert RGBD image to colored 3D point cloud.
    
    Parameters
    ----------
    rgb_image : np.ndarray
        RGB image of shape (H, W, 3), values in [0, 255] or [0, 1]
    depth_image : np.ndarray
        Depth image of shape (H, W), values in meters
    intrinsics : dict
        Camera intrinsic parameters with keys: 'fx', 'fy', 'cx', 'cy'
        - fx, fy: focal lengths in pixels
        - cx, cy: principal point coordinates
    mask : np.ndarray, optional
        Binary mask of shape (H, W) for the target object 
        (1 = object, 0 = background)
    
    Returns
    -------
    points : np.ndarray
        3D points of shape (N, 3) containing XYZ coordinates
    colors : np.ndarray
        RGB colors of shape (N, 3), normalized to [0, 1]
    
    Example
    -------
    >>> intrinsics = {'fx': 615.0, 'fy': 615.0, 'cx': 320.0, 'cy': 240.0}
    >>> points, colors = rgbd_to_pointcloud(rgb, depth, intrinsics, mask)
    """
    # Extract intrinsic parameters
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    H, W = depth_image.shape
    
    # Create pixel coordinate grids
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)  # u: (H, W), v: (H, W)
    
    # Back-project to 3D using pinhole camera model
    Z = depth_image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack into point cloud (H, W, 3)
    points = np.stack([X, Y, Z], axis=-1)
    
    # Process colors
    colors = rgb_image.copy().astype(np.float32)
    if colors.max() > 1:
        colors = colors / 255.0  # Normalize to [0, 1]
    
    # Create valid mask
    if mask is not None:
        valid = (mask > 0) & (depth_image > 0)
    else:
        valid = depth_image > 0  # Only keep points with valid depth
    
    # Flatten and filter invalid points
    points = points[valid]
    colors = colors[valid]
    
    return points, colors


def get_intrinsic_matrix(intrinsics):
    """
    Convert intrinsics dict to 3x3 camera intrinsic matrix K.
    
    Parameters
    ----------
    intrinsics : dict
        Camera intrinsic parameters with keys: 'fx', 'fy', 'cx', 'cy'
    
    Returns
    -------
    K : np.ndarray
        3x3 camera intrinsic matrix
    """
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_input_data(rgb_image, depth_image, mask=None, save_path=None):
    """
    Visualize the input RGBD data and mask.
    
    Parameters
    ----------
    rgb_image : np.ndarray
        RGB image of shape (H, W, 3)
    depth_image : np.ndarray
        Depth image of shape (H, W)
    mask : np.ndarray, optional
        Binary mask of shape (H, W)
    save_path : str, optional
        Path to save the figure
    """
    n_plots = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    # RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Depth image
    im = axes[1].imshow(depth_image, cmap='plasma')
    axes[1].set_title('Depth Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046)
    cbar.set_label('Depth (meters)', fontsize=10)
    
    # Mask (if provided)
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Object Mask', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    plt.suptitle('Input Data for Point Cloud Generation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()


def visualize_colored_pointcloud(points, colors, title="Colored 3D Point Cloud", 
                                  n_samples=10000, save_path=None):
    """
    Visualize colored 3D point cloud from multiple viewpoints.
    
    Parameters
    ----------
    points : np.ndarray
        3D points of shape (N, 3)
    colors : np.ndarray
        RGB colors of shape (N, 3), values in [0, 1]
    title : str
        Title for the plot
    n_samples : int
        Number of points to display (for performance)
    save_path : str, optional
        Path to save the figure
    """
    # Subsample for faster rendering
    if len(points) > n_samples:
        indices = np.random.choice(len(points), n_samples, replace=False)
        pts = points[indices]
        cols = colors[indices]
    else:
        pts = points
        cols = colors
    
    fig = plt.figure(figsize=(16, 12))
    
    # Define viewpoints: (elevation, azimuth, title)
    viewpoints = [
        (0, 0, 'Front View'),
        (0, 90, 'Side View'),
        (25, 135, 'Perspective View'),
        (90, 0, 'Top View')
    ]
    
    for i, (elev, azim, view_title) in enumerate(viewpoints):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], 
                   c=cols, s=3, alpha=1.0, edgecolors='none')
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Z - Depth (m)', fontsize=11)
        ax.set_zlabel('Y (m)', fontsize=11)
        ax.set_title(view_title, fontsize=13, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
    
    plt.suptitle(f'{title}\n(Each point colored with original RGB value)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()


def visualize_pointcloud_single(points, colors, title="Colored 3D Point Cloud",
                                 n_samples=15000, elev=20, azim=135, save_path=None):
    """
    Visualize colored 3D point cloud from a single viewpoint (high quality).
    
    Parameters
    ----------
    points : np.ndarray
        3D points of shape (N, 3)
    colors : np.ndarray
        RGB colors of shape (N, 3), values in [0, 1]
    title : str
        Title for the plot
    n_samples : int
        Number of points to display
    elev : float
        Elevation angle for viewing
    azim : float
        Azimuth angle for viewing
    save_path : str, optional
        Path to save the figure
    """
    # Subsample
    if len(points) > n_samples:
        indices = np.random.choice(len(points), n_samples, replace=False)
        pts = points[indices]
        cols = colors[indices]
    else:
        pts = points
        cols = colors
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], 
               c=cols, s=5, alpha=1.0, edgecolors='none')
    
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Z - Depth (meters)', fontsize=14)
    ax.set_zlabel('Y (meters)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    
    # Set equal aspect ratio
    max_range = np.array([
        pts[:, 0].max() - pts[:, 0].min(),
        pts[:, 2].max() - pts[:, 2].min(),
        pts[:, 1].max() - pts[:, 1].min()
    ]).max() / 2.0
    
    mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
    mid_y = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
    mid_z = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()


# =============================================================================
# Utility Functions
# =============================================================================

def save_pointcloud_ply(points, colors, filename):
    """
    Save point cloud to PLY file format.
    
    Parameters
    ----------
    points : np.ndarray
        3D points of shape (N, 3)
    colors : np.ndarray
        RGB colors of shape (N, 3), values in [0, 1]
    filename : str
        Output filename (should end with .ply)
    """
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    with open(filename, 'w') as f:
        # Write PLY header
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
        
        # Write point data
        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
            f.write(f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")
    
    print(f"Saved point cloud to: {filename}")


def create_synthetic_data(H=480, W=640, intrinsics=None):
    """
    Create synthetic RGBD data for testing.
    
    Parameters
    ----------
    H : int
        Image height
    W : int
        Image width
    intrinsics : dict, optional
        Camera intrinsics. If None, default values are used.
    
    Returns
    -------
    rgb_image : np.ndarray
        Synthetic RGB image
    depth_image : np.ndarray
        Synthetic depth image
    object_mask : np.ndarray
        Object segmentation mask
    intrinsics : dict
        Camera intrinsic parameters
    """
    if intrinsics is None:
        intrinsics = {
            'fx': 615.0,
            'fy': 615.0,
            'cx': W / 2.0,
            'cy': H / 2.0
        }
    
    np.random.seed(42)
    
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    
    # Background depth
    depth_image = np.ones((H, W)) * 3.0
    
    # Create circular object in center
    center_u, center_v = W // 2, H // 2
    radius = min(H, W) // 6
    dist_from_center = np.sqrt((u_grid - center_u)**2 + (v_grid - center_v)**2)
    
    # Object mask
    object_mask = (dist_from_center < radius).astype(np.uint8)
    
    # Sphere-like depth (closer in center)
    sphere_depth = 1.5 - 0.3 * np.sqrt(np.maximum(0, 1 - (dist_from_center / radius)**2))
    depth_image[object_mask > 0] = sphere_depth[object_mask > 0]
    
    # Create RGB image with color gradient
    rgb_image = np.ones((H, W, 3), dtype=np.uint8) * 50  # Dark background
    
    for v in range(H):
        for u in range(W):
            if object_mask[v, u] > 0:
                # Color gradient: red -> green -> blue
                r = int(255 * (1 - (u - center_u + radius) / (2 * radius)))
                g = int(255 * (1 - abs(v - center_v) / radius))
                b = int(255 * (u - center_u + radius) / (2 * radius))
                rgb_image[v, u] = [
                    np.clip(r, 0, 255),
                    np.clip(g, 0, 255),
                    np.clip(b, 0, 255)
                ]
    
    return rgb_image, depth_image, object_mask, intrinsics


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Task 1: RGBD to Colored 3D Point Cloud Conversion")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Create or load data
    # -------------------------------------------------------------------------
    print("\n[Step 1] Creating synthetic RGBD data...")
    
    rgb_image, depth_image, object_mask, intrinsics = create_synthetic_data()
    
    print(f"  - RGB image shape: {rgb_image.shape}")
    print(f"  - Depth image shape: {depth_image.shape}")
    print(f"  - Object mask shape: {object_mask.shape}")
    print(f"  - Camera intrinsics: fx={intrinsics['fx']}, fy={intrinsics['fy']}, "
          f"cx={intrinsics['cx']}, cy={intrinsics['cy']}")
    
    # Print intrinsic matrix
    K = get_intrinsic_matrix(intrinsics)
    print(f"\n  Camera Intrinsic Matrix K:")
    print(f"    [{K[0, 0]:8.2f} {K[0, 1]:8.2f} {K[0, 2]:8.2f}]")
    print(f"    [{K[1, 0]:8.2f} {K[1, 1]:8.2f} {K[1, 2]:8.2f}]")
    print(f"    [{K[2, 0]:8.2f} {K[2, 1]:8.2f} {K[2, 2]:8.2f}]")
    
    # -------------------------------------------------------------------------
    # Step 2: Visualize input data
    # -------------------------------------------------------------------------
    print("\n[Step 2] Visualizing input data...")
    visualize_input_data(rgb_image, depth_image, object_mask, 
                         save_path="input_data.png")
    
    # -------------------------------------------------------------------------
    # Step 3: Convert RGBD to 3D point cloud
    # -------------------------------------------------------------------------
    print("\n[Step 3] Converting RGBD to 3D point cloud...")
    
    points, colors = rgbd_to_pointcloud(rgb_image, depth_image, intrinsics, 
                                        mask=object_mask)
    
    print(f"  - Generated {len(points)} 3D points")
    print(f"  - Points shape: {points.shape} (N x 3: X, Y, Z)")
    print(f"  - Colors shape: {colors.shape} (N x 3: R, G, B)")
    print(f"  - X range: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] meters")
    print(f"  - Y range: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}] meters")
    print(f"  - Z range: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}] meters")
    
    # -------------------------------------------------------------------------
    # Step 4: Visualize colored point cloud
    # -------------------------------------------------------------------------
    print("\n[Step 4] Visualizing colored 3D point cloud...")
    
    # Multi-view visualization
    visualize_colored_pointcloud(points, colors, 
                                  title="Target Object: Colored Point Cloud",
                                  save_path="colored_pointcloud_multiview.png")
    
    # Single high-quality view
    visualize_pointcloud_single(points, colors,
                                 title="Target Object Point Cloud\n(Lifted from RGBD using camera intrinsics)",
                                 save_path="colored_pointcloud_main.png")
    
    # -------------------------------------------------------------------------
    # Step 5: Save point cloud to file
    # -------------------------------------------------------------------------
    print("\n[Step 5] Saving point cloud to PLY file...")
    save_pointcloud_ply(points, colors, "target_object_pointcloud.ply")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 1 COMPLETE!")
    print("=" * 70)
    print(f"""
Summary:
--------
- Input: RGBD image ({rgb_image.shape[1]}x{rgb_image.shape[0]}) + Object mask
- Output: Colored 3D point cloud ({len(points)} points)
- Each point has: XYZ coordinates (meters) + RGB color

Output files:
-------------
1. input_data.png           - Visualization of input RGB, depth, and mask
2. colored_pointcloud_multiview.png - Point cloud from multiple viewpoints
3. colored_pointcloud_main.png      - High-quality single view
4. target_object_pointcloud.ply     - Point cloud file (can open in MeshLab, CloudCompare)

The point cloud can now be used for:
- Feature extraction
- Correspondence matching with query object
- 6D pose estimation via RANSAC
""")