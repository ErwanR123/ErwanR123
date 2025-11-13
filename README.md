
# **Welcome to My GitHub Profile**

Hi! I am Erwan, a **Second year Master's student in AI and statistical learning at Sorbonne University**, passionate about **data science** and **artificial intelligence**. I am always looking for opportunities to learn, collaborate, and work on innovative projects. 

Here, you'll find a selection of my main projects that reflect my interest in AI and my technical skills.

## 🎨 About Me  

- **🔧 Programming**: Python, R, SQL, SAS, MATLAB, VBA.  
- **🔬 Librairies**: Matplotlib, NumPy, Pandas, Beautiful Soup, TensorFlow Keras, PyTorch, Scikit-Learn, Seaborn.  

---

## 📚 **My Projects**  

## 📘 **1. Master’s Thesis – Kernel PCA**  
- **Title**: *Kernel Principal Component Analysis: Theory and Applications*  
- **Summary**: A full theoretical and experimental exploration of Kernel PCA, focusing on its advantages over classical PCA for nonlinear dimensionality reduction.  
- **Applications**:  
  - **Sentiment classification** on IMDb movie reviews  
  - **Anomaly detection** in handwritten digits (MNIST)  
  - **Signal denoising** on ECG data (MIT-BIH)  
- **Deliverables**: Complete report (in French) and all source code (Python, Jupyter Notebooks)  
- 🔗 [**View the GitHub Repository**](https://github.com/ErwanR123/Master-Thesis-Kernel-PCA/tree/main)

---

### 🧪 **2. Toxic Gas Detection – Data Challenge Bertin Technologies**  
**Objective**: Predict 23 toxic gas concentrations from multichannel sensor signals in a context of **strong domain shift caused by humidity variations**.

**Key Contributions**:  
- Developed an **external de-humidification module** based on *polynomial Ridge regression* to correct systematic sensor drift.  
- Designed a **geometry-aware feature engineering pipeline** leveraging the spatial organization of the sensor array:  
  local gradients, inter-block contrasts, shape moments, L2-normalized profiles, etc.  
- Built a modular, reusable **train/test-compatible processing pipeline**.  
- Optimized a **multi-output Random Forest** robust to domain shift (adaptive clipping, max-samples tuning, feature selection).  
- Performed stratified cross-validation across humidity regimes to assess inter-domain robustness.  

**Results**:  
- Significant reduction of humidity-induced domain shift.  
- Stable model performance even under extreme humidity conditions.  
- **Public leaderboard score**: **0.14318**  
- **Current ranking**: **10ᵗʰ out of 148 participants**  
  → [View ranking](https://challengedata.ens.fr/participants/challenges/156/ranking/public)  
- [Project Link](https://github.com/ErwanR123/Toxic_Gas_Identification_Challenge_Bertin_Technologies-)
---

### 🚀 **3. Hackathon Gen AI - Sia Partners & SFIL**  

- **Objective**: Develop an AI-powered pipeline to automatically generate structured reports for French local authorities (SPLs) based on financial, demographic, and investment data.  
- **Details**:  
  - **Data Sources**: Open government datasets, financial reports, Wikipedia, LinkedIn, and news articles.  
  - **Technologies**: AWS Lambda, S3, SerpAPI, BeautifulSoup, Pandas, Mistral AI (LLM for text summarization and chatbot).  
  - **Key Features**:  
    - Automated data extraction and enrichment using **SerpAPI & Wikipedia API**.  
    - Integration of **Mistral AI** to summarize and extract key insights from reports.  
    - Classification of local authorities using **ELECTRE multi-criteria decision analysis**.  
    - **Interactive Streamlit dashboard** displaying structured reports, financial indicators, and ELECTRE results.  
  - **Future Enhancements**:  
    - **Chatbot powered by Mistral AI** for interactive queries on local authority data.  
    - Improved API efficiency and optimization of financial data processing.  
- **Skills Used**: Cloud Computing (AWS Lambda, S3), NLP (Mistral AI), Web Scraping (BeautifulSoup, SerpAPI), Data Processing (Pandas), Decision Analysis (ELECTRE).  
- [Project Link](https://github.com/ErwanR123/Hackathon_Gen_AI_SIA_Partners_Silf/tree/main)

---
### 🧠 4. Sentiment Analysis with Kernel PCA

- **Objective**: Evaluate whether Kernel PCA (cosine kernel) can enhance the performance of standard classifiers in a binary sentiment classification task on IMDb reviews.
- **Approach**:  
  - Applied a complete NLP preprocessing pipeline: lowercasing, lemmatization with POS-tagging, stopword removal.  
  - Converted text into numerical features using Bag-of-Words (`CountVectorizer`) with a fixed vocabulary size.  
  - Reduced dimensionality using Kernel PCA (cosine kernel), and compared with both classical PCA and no reduction.  
  - Trained and evaluated three classifiers: Logistic Regression, SVM (RBF kernel), and K-Nearest Neighbors.  
  - Analyzed how performance varies depending on the number of components used in Kernel PCA.
- **Findings**:  
  - Kernel PCA significantly improved KNN classification performance.  
  - Logistic Regression and SVM showed minor but consistent improvements with Kernel PCA over PCA.  
  - An optimal number of components was identified to retain over 90% of the explained variance with the cosine kernel.
- **Skills Used**: NLP (NLTK), Text Vectorization, Dimensionality Reduction (Kernel PCA), Scikit-learn, Model Evaluation, Experimental Design

- [Project Link](https://github.com/ErwanR123/Sentiment-Analysis-on-IMDb-Movie-Reviews-using-Kernel-PCA)
---
### 🧪 **5. Breast Cancer Classification (Statistical Learning Project)**

- **Objective**: Predict breast cancer presence based on clinical biomarkers from the Breast Cancer Coimbra dataset.
- **Details**:
  - Performed detailed **exploratory data analysis**: histograms, boxplots, scatter matrices, and correlation matrices.
  - Applied **log transformation** to right-skewed variables (`Insulin`, `HOMA`, `MCP.1`, `Resistin`) to improve normality and reduce outliers.
  - Standardized all features before model training to ensure fair comparison across algorithms.
  - Built and evaluated several models:
    - **Logistic Regression** (baseline and L2-regularized with `GridSearchCV`),
    - **k-Nearest Neighbors (KNN)**,
    - **Naïve Bayes**,
    - **MLP Classifier**.
  - Used **stratified train/test split** and **5-fold cross-validation** for robustness.
  - Evaluation metrics included **F1-score**, **recall**, **AUC**, **ROC curves**, and **coefficient analysis**.
- **Results**:
  - Logistic regression with L2 regularization (`C=100`) achieved an **F1-score of 0.75**, with balanced recall across classes.
  - MLP and KNN provided comparable performance, suggesting that **nonlinear methods** could enhance predictive power.
- **Bonus**:
  - Initial SVM experiments included (`svm.ipynb`), but not detailed in the final report.
- **Skills Used**: Scikit-learn, Data Preprocessing, Model Evaluation, Hyperparameter Tuning, ROC Analysis.
- [Project Link](https://github.com/ErwanR123/breast-cancer-detection)

---

## 🌐 **Where to Find Me**  

- **LinkedIn**: [My Profile](https://www.linkedin.com/in/erwan-ouabdesselam/)
----
⬇️⬇️ MORE PROJECTS HERE ⬇️⬇️
