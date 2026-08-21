# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING

This Machine Learning project exploits the *K-Means Clustering algorithm* to predict customer segments in order to generate actionable corporate strategies

# K-MEANS THEORY

K-Means is a popular Unsupervised algorithm to partition a dataset into K distinct non-developing subsets known as clusters. The purpose is to categorize data into groups such that points within a specific cluster are more similar to each other than to those belonging to other subsets. Unlike Supervised Learning which learns from input and labeled data to make predictions on new unseen data for a correct output prediction (*Linear Regression*, *Logistic Regression*, *Decision Trees*, *Random Forests*, *Neural Networks*), the Unsupervised Learning strives to identify hidden trends or patterns in the input data without any specific output labels. Examples are *Clustering* and *Dimensionality Reduction*.

A customer segmentation is a common application of Clustering to identify distinct groups for easier interpretation, data retrieval and generation of useful insights.

# DATASET PRESENTATION

It's a publicly file available on Kaggle.com (*CUSTOMER ANALYTICS DATASET*) designed for a real-world business application by including variables commonly used in Retail, Marketing and Financial Analytics.

# BUSINESS PROBLEM

As Data Scientist for a retail company, the I.T department relies on your competences to ask several questions about customers' behavior. The concerns are to the possible existing of people who aren't buying anymore from us and switching to competitors. What kind of clients group requires special promotions or the adoption of retention strategies? Who deserves loyalty rewards?

# EXPLORATORY DATA ANALYSIS (EDA)

After checking for null and duplicated values, I started to see possible correlations between three numerical variables: Age, Annual Income and Spending Score. The capacity of expense is key driver for future interpretations, so I decided to keep it static and giving results by doing comparisons respectively with age and Annual Income through two separate scatterplots:

### AGE VS SPENDING SCORE


<img width="1376" height="382" alt="image" src="https://github.com/user-attachments/assets/524dd810-8a00-4490-b53a-f657b8801b09" />


### ANNUAL INCOME VS SPENDING SCORE


<img width="1378" height="383" alt="image" src="https://github.com/user-attachments/assets/13947e29-6324-4eaa-b0c2-bb3a9fa66bbc" />


We don't need to preprocess categorical features, as only three numerical variables are used to deploy the algorithmic model

# ELBOW METHOD

<img width="1405" height="381" alt="image" src="https://github.com/user-attachments/assets/1d4e20bf-adcf-42ec-ac9d-69ba70a35372" />


