# customer-satisfaction-ml-nlp-project
End-to-end Machine Learning + NLP project to predict customer satisfaction using XGBoost, BERT embeddings, and SHAP explainability.

---

# Customer Support Ticket Satisfaction Analysis (NLP + ML)

**Author:** Deepika Priya K  

---

## Executive Summary

This project applies **Python, NLP, and Machine Learning** to analyze customer support tickets and understand the key reasons behind **low satisfaction ratings**. A predictive model is built to flag dissatisfaction early, helping support teams take proactive actions.

### Key Components
- Cleaning and preprocessing raw customer support data
- Feature engineering for **response time**, **resolution time**, and text metadata
- Generating text embeddings using **MPNet (SentenceTransformers)**
- Training ML models (Logistic Regression, Random Forest, XGBoost)
- **Model explainability** using SHAP
- Saving and exporting the final ML model for deployment

---

## Business Problem

Support teams handle **thousands of customer tickets** each day. Manually identifying dissatisfaction patterns and predicting low satisfaction scores is difficult.

This project aims to solve:

- **Why** customers are dissatisfied  
- **Which** tickets are likely to receive low satisfaction ratings  
- **Which factors** most influence customer experience  

The result is a combination of **data-driven insights** and a **predictive ML model** that helps support teams act early and improve customer satisfaction.

---

## Methodology

### 1. Data Cleaning & Preparation
- Resolved missing and inconsistent values  
- Standardized and converted timestamp fields  
- Created duration features such as *first response time* and *resolution time*  
- Combined subject and description fields to prepare clean ticket text  
- Saved the processed datasets for modeling and reuse  

### 2. Exploratory Data Analysis (EDA)
- Analyzed satisfaction score distribution  
- Identified patterns across ticket priority, type, and support channels  
- Explored time-based behaviors to understand workload and response delays  
- Generated a correlation heatmap to study relationships among features  

### 3. NLP Text Embedding (MPNet)
- Used the `"all-mpnet-base-v2"` SentenceTransformer model  
- Generated **768-dimension embeddings** for each support ticket  
- Merged text embeddings with numeric and categorical features to build a unified dataset  

### 4. Machine Learning Models
- Trained models including Logistic Regression, Random Forest, and XGBoost  
- Selected **XGBoost** as the final model due to best performance  
- Evaluated all models using accuracy, precision, recall, and F1-score  

### 5. Model Explainability
- Used SHAP bar plots to show feature importance  
- Used SHAP beeswarm plots to analyze how each feature influences predictions  

### 6. Model Export
- Saved the final trained model as:  
  `data/models/xgb_model.pkl`

---

## Skills & Tools Used
- **Programming & Analytics:** Python, Pandas, NumPy  
- **Machine Learning:** Scikit-learn, XGBoost  
- **NLP:** SentenceTransformers (MPNet)  
- **Visualization:** Seaborn, Matplotlib  
- **Model Explainability:** SHAP  
- **Development Environment:** Google Colab, GitHub  

---

## Key Insights

- **High first response time** is strongly linked to customer dissatisfaction  
- **Long resolution time** is one of the biggest drivers of low satisfaction  
- **Complex or lengthy customer messages** often carry more negative sentiment  
- **Product type** influences dissatisfaction patterns  
- **Text sentiment** provides strong predictive power for the ML model  

## 📊 Satisfaction Distribution (Satisfied vs Dissatisfied)
<p align="centre">
<img src="https://github.com/deepikapriyak30/customer-satisfaction-ml-nlp-project/blob/main/image/Satisfaction%20Distribution%20(Satisfied%20vs%20Dissatisfied).jpg?raw=true" width="350">
</p>

---

## Machine Learning Results

- **Logistic Regression** → Best recall (captures more dissatisfied cases)  
- **Random Forest** → Best precision (lowest false positives)  
- **XGBoost** → Best overall balance across metrics  


## Overall Model Comparison

| Model               | Accuracy | Precision (1) | Recall (1) | F1-score |
|---------------------|----------|---------------|------------|----------|
| Logistic Regression | 0.777    | 0.386         | 0.661      | 0.487    |
| Random Forest       | 0.842    | 0.556         | 0.084      | 0.147    |
| XGBoost             | 0.837    | 0.489         | 0.406      | 0.444    |

---

## Business Recommendations

1. **Reduce response & resolution delays**  
   Implement better triage, workload balancing, and SLA monitoring.

2. **Prioritize high-risk tickets**  
   Long descriptions + high waiting time → route to senior agents.

3. **Improve agent communication quality**  
   Provide templates, tone guidelines, and proactive messaging.

4. **Strengthen product-specific support**  
   Some product categories have higher dissatisfaction patterns.

5. **Use ML predictions in real-time**  
   Alert supervisors for tickets likely to receive low ratings.

---

## Next Steps

1. Build a Power BI dashboard for insights  
2. Train an advanced transformer model (BERT fine-tuning)  
3. Deploy the model as a REST API  
4. Add text sentiment SHAP explanations  

---

## Dataset Source

Customer support dataset used for learning purposes.  
*I do not own the data — full credit to original creators.*


