**HCGATCL-SR - Hypergraph Convolution + Graph Attention + Contrastive Learning for Session-based Recommendation**

📌 **Task**
* Predict the next item from short, anonymous sessions. 

HCGATCL-SR uses a dual-view graph: Item-level hypergraph for high-order co-occurrences and Session-level graph for inter-session similarity

Contrastive Learning with Hard Negative Mining aligns the two views:
* Pulls semantically similar sessions together
* Pushes hard negatives apart

**Metrics:** Precision@N, MRR@N
**Datasets:** Diginetica, Nowplaying, Tmall
