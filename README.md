# customer-satisfaction-ml-nlp-project
End-to-end Machine Learning + NLP project to predict customer satisfaction using XGBoost, BERT embeddings, and SHAP explainability.

---

# customer-satisfaction-ml-nlp-project  
End-to-end Machine Learning + NLP project to predict customer satisfaction from support tickets and identify key drivers of dissatisfaction.

# Customer Support Ticket Satisfaction Analysis (NLP + ML)

**Author:** Deepika Priya K  

---

## Executive Summary  

Using Python, NLP, and Machine Learning, this project analyzes customer support tickets to understand **why customers give low satisfaction ratings** and builds a **predictive model** to detect dissatisfaction early.

The project includes:  
- Cleaning & preparing raw customer support data  
- Feature engineering for response & resolution times  
- Text embeddings using MPNet (SentenceTransformers)  
- Machine Learning models (Logistic Regression, Random Forest, XGBoost)  
- Model explainability using SHAP  
- Saving final ML model for deployment  

---

## Business Problem  

Companies receive thousands of support tickets daily.  
It’s difficult to manually understand:  
- Why customers are dissatisfied  
- Which tickets will receive low ratings  
- Which factors drive poor customer experience  

This project provides **data-driven insights** + a **predictive ML model** to help support teams act proactively.

---

## Methodology  

### **1. Data Cleaning & Preparation**
- Handled missing & inconsistent values  
- Converted timestamps  
- Created duration features: *first response time*, *resolution time*  
- Cleaned ticket text: combined subject + description  
- Saved processed datasets  

### **2. Exploratory Data Analysis (EDA)**
- Distribution of satisfaction  
- Ticket priority, channel, type analysis  
- Time-based behavior patterns  
- Correlation heatmap  

### **3. NLP Text Embedding (MPNet)**
- Used `"all-mpnet-base-v2"` SentenceTransformer  
- Generated **768-dimensional embeddings** for each ticket  
- Combined numeric + categorical + text embeddings  

### **4. Machine Learning Models**
- Logistic Regression  
- Random Forest  
- **XGBoost (final model)**  
- Evaluated using accuracy, precision, recall, F1-score  

### **5. Model Explainability**
- SHAP bar plot → feature importance  
- SHAP beeswarm → direction of impact  

### **6. Model Export**
Final trained model saved as:  
`data/models/xgb_model.pkl`

---

## Skills & Tools Used  

- **Python:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, XGBoost  
- **NLP:** SentenceTransformers (MPNet)  
- **Visualization:** Seaborn, Matplotlib  
- **Explainability:** SHAP  
- **Development:** Google Colab, GitHub  

---

## Key Insights  

- **High first response time** strongly increases dissatisfaction  
- **Long resolution time** is one of the biggest drivers  
- **Complex / lengthy messages** → more negative sentiment  
- **Product type** influences dissatisfaction  
- Text sentiment carries strong predictive signal  

---

## Machine Learning Results  

- **Logistic Regression** → Best recall  
- **Random Forest** → Best precision  
- **XGBoost** → Best overall balance  

### **Overall Model Comparison**

| Model               | Accuracy | Precision (1) | Recall (1) | F1-score |  
|---------------------|----------|----------------|-------------|----------|  
| Logistic Regression | 0.777    | 0.386          | 0.661       | 0.487    |  
| Random Forest       | 0.842    | 0.556          | 0.084       | 0.147    |  
| XGBoost             | 0.837    | 0.489          | 0.406       | 0.444    |  

---

## Business Recommendations  

**1. Reduce response & resolution delays:**  
Implement better triage, workload balancing, and SLA monitoring.

**2. Prioritize high-risk tickets:**  
Long descriptions + high waiting time → route to senior agents.

**3. Improve agent communication quality:**  
Provide templates, tone guidelines, proactive messaging.

**4. Strengthen product-specific support:**  
Some product categories have more dissatisfaction.

**5. Use ML predictions in real-time:**  
Alert supervisors for tickets likely to receive low ratings.

---

## Next Steps  

1. Build Power BI dashboard for insights  
2. Train an advanced transformer model (BERT fine-tuning)  
3. Deploy model as REST API  
4. Add text sentiment SHAP explanations  

---

## Dataset Source  

Customer support dataset used for learning purposes.  
*I do not own the data — full credit to original creators.*

