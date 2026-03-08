# Portfolio Project-1: E-commerce Customer Value Prediction

## Executive Summary

This project predicts high-value customers in e-commerce using statistical analysis, GLM (Logistic Regression), and ML models (Random Forest, XGBoost).

**Dataset**: E-commerce customer data (5,000 samples, 8 features)
**Target**: Binary classification - Predict high-value customers

---

## Business Problem

E-commerce companies need to:
1. Identify high-value customers for targeted marketing
2. Allocate retention resources efficiently
3. Personalize customer experience based on predicted value

---

## Analysis Framework

### 1. Statistical Analysis ✓
- Descriptive statistics
- Feature distribution analysis
- Correlation heatmap
- Outlier detection (IQR method)

### 2. GLM - Logistic Regression ✓
- Coefficient interpretation with odds ratios
- Significance testing (p-values)
- Model performance metrics

### 3. Machine Learning ✓
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting for optimal performance
- Model comparison across 5 metrics

### 4. Business Insights ✓
- Actionable recommendations
- Role-specific applications (DA/DS/BA)
- Implementation roadmap

---

## Key Findings

### Top Predictors of High-Value Customers
1. **Income** - Strongest positive predictor
2. **Total Purchases** - Frequency matters more than order value
3. **Premium Membership** - 2.5x higher odds of being high-value
4. **Tenure** - Longer relationship = higher value
5. **Categories Bought** - Cross-category shoppers are more valuable

### Model Performance
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.82 | 0.71 | 0.68 | 0.69 | 0.85 |
| Random Forest | 0.87 | 0.78 | 0.74 | 0.76 | 0.91 |
| XGBoost | 0.88 | 0.79 | 0.76 | 0.77 | 0.92 |

**Best Model**: XGBoost (highest ROC-AUC and F1-Score)

---

## Files in This Project

| File | Description |
|------|-------------|
| `analysis_report.md` | Full analysis report with interpretations |
| `01_descriptive_stats.csv` | Summary statistics |
| `02_feature_distributions.png` | Distribution histograms |
| `03_correlation_heatmap.png` | Feature correlations |
| `04_outlier_analysis.csv` | Outlier detection results |
| `05_glm_logistic_regression_summary.txt` | Statsmodels output |
| `06_glm_coefficients.csv` | Coefficient interpretation |
| `07_glm_confusion_matrix.png` | Classification results |
| `08_rf_feature_importance.csv` | RF feature weights |
| `09_xgb_feature_importance.csv` | XGB feature weights |
| `10_feature_importance_comparison.png` | Visual comparison |
| `11_model_comparison.csv` | Model metrics summary |

---

## Suitable Job Roles

### Data Analyst (E-commerce/Retail)
- EDA and visualization skills
- Business insights communication
- Dashboard-ready outputs

### Data Scientist (Financial Services)
- Predictive modeling (GLM + ML)
- Model interpretation
- Feature engineering

### Business Analyst (Consumer Insights)
- Customer segmentation
- Actionable recommendations
- Stakeholder communication

### Marketing Analyst
- Customer lifetime value
- Campaign targeting
- ROI framework

---

## How to Run

```bash
# In GitHub Actions (automated)
# The workflow runs: python scripts/generate_portfolio.py

# Locally (requires dependencies)
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
python scripts/generate_portfolio.py
```

---

## Next Steps

1. Connect to Kaggle API for live dataset fetching
2. Add time-series features (seasonality, trends)
3. Deploy as real-time prediction API
4. A/B test model impact on business metrics

---

*Generated: 2026-03-08*
*Portfolio Project for DS/DC Job Applications*
