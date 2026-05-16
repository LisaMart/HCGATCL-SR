**HCGATCL-SR - Hypergraph Convolution + Graph Attention + Contrastive Learning for Session-based Recommendation**

📌 **Task**
* Predict the next item from short, anonymous sessions. 

HCGATCL-SR uses a dual-view graph: Item-level hypergraph for high-order co-occurrences and Session-level graph for inter-session similarity

Contrastive Learning with Hard Negative Mining aligns the two views:
* Pulls semantically similar sessions together
* Pushes hard negatives apart

**Metrics:** Precision@N, MRR@N
**Datasets:** Diginetica, Nowplaying, Tmall

## Acknowledgements

This repository contains the implementation of HCGATCL-SR, a session-based recommendation model developed as part of my master's dissertation.

The model design is inspired by several lines of prior research on hypergraph neural networks, graph attention networks, and contrastive learning for session-based recommendation. The hypergraph convolution component is related to prior work on hypergraph neural networks and hypergraph-based session recommendation, including HGCN and DHCN. The session-level attention component refers to graph attention mechanisms for session-based recommendation. The contrastive learning objective follows the InfoNCE-style formulation introduced in Contrastive Predictive Coding, while the hard negative mining strategy is motivated by prior work on hard negative mining in metric learning and graph contrastive learning.

The original papers are listed below for proper acknowledgment.

## Related Papers

- **Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation**  
  Xin Xia, Hongzhi Yin, Junliang Yu, Qinyong Wang, Lizhen Cui, Xiangliang Zhang. AAAI 2021.  
  DOI: https://doi.org/10.1609/aaai.v35i5.16578

- **Hypergraph Neural Networks**  
  Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao.  AAAI 2019: 3558-3565. 
  DOI: https://doi.org/10.1609/aaai.v33i01.33013558

- **Personalized Session-Based Recommendation Using Graph Attention Networks**  
  Yongquan Xie, Zhengru Li, Tian Qin, Finn Tseng, Johannes Kristinsson, Shiqi Qiu, Yi Lu Murphey. IJCNN 2021.  
  DOI: https://doi.org/10.1109/IJCNN52387.2021.9533533

- **Representation Learning with Contrastive Predictive Coding**  
  Aaron van den Oord, Yazhe Li, Oriol Vinyals. arXiv:1807.03748, 2018.

- **Hard Negative Mining for Metric Learning Based Zero-Shot Classification**  
  Maxime Bucher, Stéphane Herbin, Frédéric Jurie. arXiv:1608.07441, 2016.  
  DOI: https://doi.org/10.48550/arXiv.1608.07441

- **Hardness-Aware Deep Metric Learning**  
  Wenzhao Zheng, Zhaodong Chen, Jiwen Lu, Jie Zhou. arXiv:1903.05503, 2019.  
  DOI: https://doi.org/10.48550/arXiv.1903.05503

- **ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning**  
  Jun Xia, Lirong Wu, Ge Wang, Jintao Chen, Stan Z. Li. ICML 2022.  
  DOI: https://doi.org/10.48550/arXiv.2110.02027
