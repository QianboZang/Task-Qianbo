# Task 2: Segmentation Accuracy vs. Occlusion Analysis on LM-O Dataset
**Dataset**: LM-O (Linemod-Occluded) from BOP Challenge

## Occlusion Clustering (K-Means)
We used **K-Means clustering (k=3)** to automatically partition objects into three occlusion groups:
| Cluster | Visibility Range | Cluster Center | Interpretation |
|---------|------------------|----------------|----------------|
| **Low Occlusion** | 0.85 - 1.00 | 0.956 | Object mostly visible |
| **Medium Occlusion** | 0.59 - 0.84 | 0.734 | Partial occlusion |
| **Heavy Occlusion** | 0.08 - 0.58 | 0.424 | Significant occlusion |

## Segmentation Accuracy by Occlusion Level
| Occlusion Level | Sample Count | Mean IoU | Std IoU | Detection Rate |
|-----------------|--------------|----------|---------|----------------|
| **Low** | 809 | 0.8258 | ±0.0875 | 99.5% |
| **Medium** | 341 | 0.7474 | ±0.1223 | 95.9% |
| **Heavy** | 174 | 0.5975 | ±0.1984 | 73.6% |

**Clear Performance Degradation**: Segmentation accuracy drops significantly as occlusion increases:
   - Low → Medium: **-9.5%** IoU decrease
   - Medium → Heavy: **-20.0%** IoU decrease
   - Total degradation (Low → Heavy): **-27.6%** IoU

## Correlation Analysis

| Correlation Type | Coefficient (r) | p-value | Interpretation |
|------------------|-----------------|---------|----------------|
| **Pearson** | 0.6015 | 4.32e-131 | Strong positive correlation |
| **Spearman** | 0.5551 | 7.23e-108 | Strong monotonic relationship |

There is a statistically significant **positive correlation** between occlusion and segmentation accuracy. The Pearson correlation coefficient of **r = 0.60** with an extremely small p-value (p < 10⁻¹³⁰) confirms that:

## Visualizations
![Comparison](lmo_occlusion/comparison.png)
![Scatter](lmo_occlusion/scatter.png)
