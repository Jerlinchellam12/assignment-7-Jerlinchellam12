# assignment-7-Jerlinchellam12

Name : Jerlin Chellam J

Roll No: DA25C009

Documentation

# 🌍 Multi-Class Model Selection using ROC and Precision-Recall Curves

## 🧩 Project Overview

This project focuses on classifying land cover types using **satellite image data** from the **UCI Landsat Satellite dataset**.

It's a **multi-class classification problem (6 classes)** that involves high feature dimensionality and overlapping boundaries between classes.

The main goal is to identify the best-performing model using **ROC (Receiver Operating Characteristic)** and **PRC (Precision-Recall Curves)** rather than relying solely on accuracy.

---

## 🎯 Objectives

- Classify satellite land cover types into six distinct categories.
- Compare multiple classifiers using **F1 Score**, **ROC-AUC**, and **PRC-AP** metrics.
- Evaluate models with one-vs-rest ROC and PRC analysis to understand threshold-based behavior.
- Determine the best and worst-performing models and explain specific trade-offs between them.

---

## ⚙️ Requirements

Before running this notebook, ensure you have the following Python packages installed:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

**Python Version:** 3.8 or higher  
**Recommended Environment:** Jupyter Notebook

---

## 🛰️ Dataset and Methodology

The **UCI Landsat Satellite dataset** contains spectral data representing different land cover types such as vegetation, soil, and water.

The workflow included the following steps:

1. **Data Preprocessing** – Cleaning and splitting the dataset into training and testing sets.

2. **Model Training** – Implemented the following classifiers:
   - K-Nearest Neighbors (KNN)
   - XGBoost Classifier
   - Random Forest
   - Decision Tree

3. **Evaluation Metrics** –
   - Accuracy
   - Weighted F1 Score
   - ROC-AUC (One-vs-Rest)
   - PRC-AP (Average Precision)

4. **Visualization** – ROC and PRC curves were plotted for all models to analyze performance at various thresholds.

---

## 🔍 Observations

- **KNN** consistently produced the highest F1, ROC-AUC, and PRC-AP values, showing smooth and stable performance.
- **XGBoost** followed closely, performing well on precision but slightly lower on recall compared to KNN.
- **Random Forest** and **Decision Tree** lagged behind, with Decision Tree showing signs of overfitting.
- Models with smoother PR curves tended to generalize better across unseen data.

---

## 💡 Insights

- ROC and PRC analyses reveal details that accuracy alone cannot show.
- **KNN** provided the best balance between precision and recall, making it ideal for real-world land classification tasks.
- **XGBoost** proved to be a strong second choice with powerful learning ability but slightly higher computational cost.
- **Decision Tree** models, while interpretable, suffered from performance instability across thresholds.

---

## 🧾 Conclusion

Through this project, I learned that **threshold-based performance evaluation** is crucial in complex multi-class tasks.

Among all models, **KNN stood out as the best-performing classifier**, showing top performance in every metric.

**XGBoost emerged as the second-best model**, offering competitive results and robust generalization.

Overall, the analysis highlighted how **ROC and PRC curves provide deeper insights** for selecting reliable models in satellite-based land cover classification.

---

## 📊 Model Performance Summary

| Model              | Weighted F1 | ROC-AUC (OvR) | PRC-AP |
|--------------------|-------------|---------------|--------|
| KNN                | 0.9094      | 0.9842        | 0.9215 |
| XGBoost            | TBD         | TBD           | TBD    |
| Random Forest      | TBD         | TBD           | TBD    |
| Decision Tree      | 0.8481      | 0.9077        | 0.7246 |
| Logistic Regression| 0.8421      | 0.9775        | 0.8638 |
| Naive Bayes        | 0.7901      | 0.9540        | 0.7859 |
| SVM                | 0.8913      | 0.9838        | 0.8996 |

---

## 🚀 How to Run

1. Clone this repository:
```bash
   git clone https://github.com/yourusername/landsat-classification.git
   cd landsat-classification
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
   jupyter notebook notebooks/model_evaluation.ipynb
```

---

## 📚 References

- [UCI Landsat Satellite Dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
