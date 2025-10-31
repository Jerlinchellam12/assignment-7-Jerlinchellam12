# assignment-7-Jerlinchellam12

Name : Jerlin Chellam J

Roll No: DA25C009

Documentation

# Multi-Class Model Selection using ROC and Precision-Recall Curves

## Project Overview

This project focuses on classifying land cover types using **satellite image data** from the **UCI Landsat Satellite dataset**.

It's a **multi-class classification problem (6 classes)** that involves high feature dimensionality and overlapping boundaries between classes.

The main goal is to identify the best-performing model using **ROC (Receiver Operating Characteristic)** and **PRC (Precision-Recall Curves)** rather than relying solely on accuracy.

---

## Objectives

- Classify satellite land cover types into six distinct categories.
- Compare multiple classifiers using **F1 Score**, **ROC-AUC**, and **PRC-AP** metrics.
- Evaluate models with one-vs-rest ROC and PRC analysis to understand threshold-based behavior.
- Determine the best and worst-performing models and explain specific trade-offs between them.

---

## Requirements

Before running this notebook, ensure you have the following Python packages installed:
```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

**Python Version:** 3.8 or higher  
**Recommended Environment:** Jupyter Notebook

---

## Dataset and Methodology

The **UCI Landsat Satellite dataset** contains spectral data representing different land cover types such as vegetation, soil, and water.

The workflow included the following steps:

1. **Data Preprocessing** – Splitting the dataset into training and testing sets.

2. **Model Training** – Implemented the following classifiers:
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier
   - Dummy Classifier
   - Logistic Regression
   - Naive Bayes
   - Support Vector Machine(SVM)
   - Random Forest Classifier
   - XGBoost classifier


3. **Evaluation Metrics** –
   - Accuracy
   - Weighted F1 Score
   - ROC-AUC (One-vs-Rest)
   - PRC-AP (Average Precision)

4. **Visualization** – ROC and PRC curves were plotted for all models to analyze performance at various thresholds.

---

## Observations

- **KNN** consistently produced the highest F1, ROC-AUC, and PRC-AP values, showing smooth and stable performance.
- **XGBoost** followed closely, performing well on precision but slightly lower on recall compared to KNN.
- **Random Forest** and **Decision Tree** lagged behind, with Decision Tree showing signs of overfitting.
- Models with smoother PR curves tended to generalize better across unseen data.

---

## Insights

- ROC and PRC analyses reveal details that accuracy alone cannot show.
- **KNN** provided the best balance between precision and recall, making it ideal for real-world land classification tasks.
- **XGBoost** proved to be a strong second choice with powerful learning ability but slightly higher computational cost.
- **Decision Tree** models, while interpretable, suffered from performance instability across thresholds.
- **Dummy Classifier** was worst model.

---

## 🧾 Conclusion

Through this project, I learned the importance of analyzing model behavior across different thresholds, especially in multi-class problems where accuracy alone can be misleading.

Among all models, **KNN stood out as the best-performing classifier**, showing top performance in every metric.

**XGBoost emerged as the second-best model**, offering competitive results and robust generalization.

Overall, the analysis highlighted how **ROC and PRC curves provide deeper insights** for selecting reliable models in satellite-based land cover classification.

---

## References

- [UCI Landsat Satellite Dataset](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
  

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
