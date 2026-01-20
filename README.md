🍽️ Yelp Hybrid Recommendation System (CF · MF · XGBoost)


🚀 Scalable hybrid recommender system for Yelp ⭐ rating prediction using PySpark (RDD-only), combining collaborative filtering, matrix factorization, and XGBoost for sparse user–item data.

---

## 🔍 Overview
Predicts Yelp star ratings for `(user_id, business_id)` pairs by combining multiple recommendation paradigms into a single robust pipeline.

**Core techniques:**
- 🤝 Item–item collaborative filtering (Pearson correlation + shrinkage)
- 📉 SGD-based matrix factorization
- 🌲 XGBoost regression on user/business features
- 🧊 Bias-based cold-start handling
- 🔗 Linear ensemble blending

📊 Achieves **~0.98 RMSE** on validation.

---

## 🏗️ Architecture
- 📐 **Baseline:** Global mean + regularized user/business biases  
- 🔁 **CF:** Residual item–item CF with top-K similarity pruning  
- 🧮 **MF:** Latent factor model trained via SGD  
- 🤖 **ML:** XGBoost for feature-based generalization  
- 🔗 **Ensemble:** Linear blending with prediction clamping  

---

## 🛠️ Tech Stack
- 🐍 Python
- ⚡ PySpark (RDD API)
- 🌲 XGBoost
- 📊 NumPy

---

## ▶️ How to Run
```bash
spark-submit Stable_Hybrid_Baseline.py <data_folder> <test_file> <output_file>
```
📤 Output
CSV format ->
user_id,business_id,prediction

⭐ Predictions are clamped to [1.0, 5.0].
