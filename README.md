Edge-Aware SAR Ship Detection: Multi-Scale Edge-Semantic Fusion and Contrast-Driven Feature Aggregation
📖 Abstract
Accurate target localization in Synthetic Aperture Radar (SAR) ship detection heavily relies on robust edge information, yet conventional detection networks lack dedicated mechanisms to enhance edge feature awareness. To address this limitation, we propose a novel framework that integrates two synergistic modules:

Multi-Scale Edge-Semantic Fusion (MEF): This module extracts edge features from shallow convolutional layers and propagates them across the backbone network, enabling multi-scale edge-semantic fusion.

Contrast-Driven Feature Aggregation (CDFA): This module integrates Haar wavelet transforms and attention mechanisms to dynamically enhance edge detection and feature discriminability through multi-scale spectral analysis and contrast-aware weighting.

Extensive experiments on SSDD and other public datasets demonstrate the effectiveness of our approach. The proposed framework achieves state-of-the-art accuracy on multiple benchmarks with computational efficiency.

[Paper PDF] (coming soon)
[Project Page]

🚀 Features
General and Plug-and-Play: Our framework is a universal component that can be easily integrated into mainstream object detection architectures, including the YOLO and RT-DETR series.

Significant Performance Improvement: On public datasets like SSDD, our method significantly improves detection accuracy while maintaining high computational efficiency.

Edge-Aware: A unique modular design that effectively leverages crucial edge information in SAR images, solving issues of blurry edges and inaccurate localization common in traditional methods.

📂 Datasets
We evaluate on several public SAR ship detection datasets, including:

SSDD: SAR Ship Detection Dataset

HRSID: High-Resolution SAR Ship Detection Dataset

⚙️ Installation
# Clone the repository
git clone [https://github.com/ZJ-Song-Lab/MEF-CDFA.git](https://github.com/ZJ-Song-Lab/MEF-CDFA.git)
cd MEF-CDFA

# Install dependencies
pip install -r requirements.txt

🚀 Results
Our method achieved significant performance gains on multiple baseline architectures, including a 1.97% AP improvement on RT-DETR, demonstrating its generalizability and robustness.

Main Results (SSDD Dataset)
Model

AP@0.5 (%)

AP@0.5:0.95 (%)

RT-DETR

XX.XX

XX.XX

Our Method

XX.XX

XX.XX

YOLOv8

XX.XX

XX.XX

Our Method

XX.XX

XX.XX

📸 Visualization
📜 License
This project is released under the MIT License.
