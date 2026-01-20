# 🔬 Experimental Hybrid Recommender — PySpark

This directory contains **experimental variants** of the Yelp hybrid recommendation system, used to explore **feature richness, model capacity, and ensemble strategies** beyond the stable baseline.

These implementations are **not the default submission**, but serve as a controlled environment for performance experimentation and ablation.

---

## 🧠 Experimental Focus
The experimental pipeline extends the stable hybrid model with:

- 📊 Expanded user and business side features  
- 🧩 PCA-based dimensionality reduction for high-cardinality metadata  
- 🧮 Higher-capacity matrix factorization models  
- 🔗 Regularized linear blending strategies  

The goal is to evaluate **accuracy vs. complexity trade-offs** under sparse user–item interactions.

---

## 🏗️ Architecture Highlights

- 📐 **Baseline:** Global mean with regularized user/business biases  
- 🔁 **Collaborative Filtering:** Residual item–item CF (Pearson correlation + shrinkage)  
- 🧮 **Matrix Factorization:** SGD-trained latent factor model with bias terms  
- 📊 **Feature Engineering:**  
  - User activity and engagement statistics  
  - Business metadata (categories, location, popularity signals)  
  - Optional PCA projections for dimensionality control  
- 🌲 **XGBoost:** Feature-based regression for cold-start robustness  
- 🔗 **Ensemble:** Linear blending with ridge regularization and clamping  

---


## 🛠️ Tech Stack
- 🐍 Python  
- ⚡ PySpark (RDD API)  
- 🌲 XGBoost  
- 📊 NumPy  

---

## 📝 Notes
- Experimental variants may increase runtime or memory usage  
- Results can vary across splits and hyperparameters  
- Intended for **analysis and learning**, not default deployment
