Edge-Aware SAR Ship Detection: Multi-Scale Edge-Semantic Fusion and Contrast-Driven Feature Aggregation
✨ Highlights
General and Plug-and-Play: Our framework is a universal component that can be easily integrated into mainstream object detection architectures, including the YOLO and RT-DETR series.

Significant Performance Improvement: On public datasets like SSDD, our method significantly improves detection accuracy while maintaining high computational efficiency.

Edge-Aware: A unique modular design that effectively leverages crucial edge information in SAR images, solving issues of blurry edges and inaccurate localization common in traditional methods.

🎯 Methodology Overview
To address the challenges in Synthetic Aperture Radar (SAR) image ship detection, we propose a general framework called Edge-Aware SAR Ship Detection. The framework consists of two key modules:

Multi-Scale Edge-Semantic Fusion (MEF) Module:

Function: Solves the problem of edge information loss during deep network propagation.

How it works: Extracts high-resolution edge features from shallow convolutional layers and progressively fuses them into the Feature Pyramid Network (FPN), working synergistically with deep semantic features.

Contrast-Driven Feature Aggregation (CDFA) Module:

Function: Enhances the model's feature discriminability in complex backgrounds, reducing false positives and missed detections.

How it works: Uses Haar wavelet transforms for multi-scale decomposition, combined with an attention mechanism for dynamic weighting, to aggregate features that enhance the contrast between targets and the background.

🖼️ Framework Illustration
🚀 Results
We conducted extensive experiments on the SSDD and other public datasets to evaluate the proposed framework. Our method achieved significant performance gains on multiple baseline architectures, including a 1.97% AP improvement on RT-DETR, demonstrating its generalizability and robustness.

Model



📖 Citation
If this research is helpful to you, please cite our paper:

[Pending]

🌐 Open Source
We plan to open-source the detailed implementation of this method for the community to use and for further research. Please follow our GitHub repository for the latest updates.
