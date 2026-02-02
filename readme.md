# Task 2: Segmentation Accuracy vs. Occlusion Analysis on LM-O Dataset

## Dataset
**Dataset**: LM-O (Linemod-Occluded) from BOP Challenge

## Methodology
### Occlusion Clustering (K-Means)

We used **K-Means clustering (k=3)** on the `visib_fract` (visibility fraction) values to automatically partition objects into three occlusion groups:

| Cluster | Visibility Range | Cluster Center | Interpretation |
|---------|------------------|----------------|----------------|
| **Low Occlusion** | 0.85 - 1.00 | 0.956 | Object mostly visible |
| **Medium Occlusion** | 0.59 - 0.84 | 0.734 | Partial occlusion |
| **Heavy Occlusion** | 0.08 - 0.58 | 0.424 | Significant occlusion |

### Evaluation Metrics

- **IoU (Intersection over Union)**: Measures segmentation mask overlap
- **Detection Rate**: Percentage of predictions with IoU ≥ 0.5
- **Pearson/Spearman Correlation**: Quantifies relationship between visibility and IoU

## Results

### Segmentation Accuracy by Occlusion Level

| Occlusion Level | Sample Count | Mean IoU | Std IoU | Detection Rate |
|-----------------|--------------|----------|---------|----------------|
| **Low** | 809 | 0.8258 | ±0.0875 | 99.5% |
| **Medium** | 341 | 0.7474 | ±0.1223 | 95.9% |
| **Heavy** | 174 | 0.5975 | ±0.1984 | 73.6% |

### Key Findings

1. **Clear Performance Degradation**: Segmentation accuracy drops significantly as occlusion increases:
   - Low → Medium: **-9.5%** IoU decrease
   - Medium → Heavy: **-20.0%** IoU decrease
   - Total degradation (Low → Heavy): **-27.6%** IoU

2. **Detection Rate Impact**: The percentage of successful detections (IoU ≥ 0.5) drops dramatically:
   - Low occlusion: 99.5%
   - Heavy occlusion: 73.6% (**-25.9 percentage points**)

3. **Increased Variance**: Standard deviation increases with occlusion (0.0875 → 0.1984), indicating more unpredictable performance under heavy occlusion.

### Correlation Analysis

| Correlation Type | Coefficient (r) | p-value | Interpretation |
|------------------|-----------------|---------|----------------|
| **Pearson** | 0.6015 | 4.32e-131 | Strong positive correlation |
| **Spearman** | 0.5551 | 7.23e-108 | Strong monotonic relationship |

**Answer to Task Question**: **Yes, there is a statistically significant positive correlation** between object visibility (inverse of occlusion) and segmentation accuracy. The Pearson correlation coefficient of **r = 0.60** with an extremely small p-value (p < 10⁻¹³⁰) confirms that:

> Objects with higher visibility (lower occlusion) consistently achieve better segmentation accuracy.

## Visualizations

### Figure 1: Segmentation Metrics by Occlusion Cluster
![Comparison](lmo_occlusion/comparison.png)

This figure shows three metrics across occlusion levels:
- **Left**: Mean IoU with standard deviation error bars
- **Center**: Detection rate (IoU ≥ 0.5 threshold)
- **Right**: Sample distribution across clusters

### Figure 2: IoU vs. Visibility Scatter Plot
![Scatter](lmo_occlusion/scatter.png)

This scatter plot visualizes:
- Individual data points colored by occlusion cluster
- Linear trend line showing the positive correlation
- Correlation statistics in the title