🔍 Project: HCGATCL-SR  
**Hypergraph Convolution + Graph Attention + Contrastive Learning for Session-based Recommendation**

📌 Problem
Session-based recommender systems must predict the **next item** using only **a few anonymous clicks**.  
Pure GNN models:  
- Over-smooth item embeddings  
- Ignore **high-order item correlations**  
- Fail to **separate similar sessions** → fragile representations under noise/sparsity

✅ Solution
HCGATCL-SR builds a **dual-view graph**:  
1. **Item-level hypergraph** – captures **high-order** item co-occurrence via hyperedges  
2. **Session-level graph** – models **inter-session** similarity with GAT  

**Contrastive Learning (InfoNCE)** explicitly aligns the two views:  
- Pulls **semantically similar sessions** together  
- Pushes **dissimilar sessions** apart  
→ **No extra latency** at inference

📊 Results
**+1.8 % Precision@10** and **+1.9 % MRR@10** vs. strongest baseline **without** extra parameters or inference cost.

Datatsets - Diginetica, Nowplaying, Tmall
