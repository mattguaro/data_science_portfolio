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


# ELBOW METHOD - AGE VS SPENDING SCORE


<img width="1405" height="381" alt="image" src="https://github.com/user-attachments/assets/1d4e20bf-adcf-42ec-ac9d-69ba70a35372" />


A technique called ***Elbow Method*** is used to determine the optimal number of clusters, such that adding one more cluster doesn't provide anymore a better modelling of the data. The value is found by computing the Within-Cluster Sum of Squares (WCSS) for different numbers of subsets. It's identified by one ideal point meeting the sum of squares in y-axis and K number of clusters in x-axis. The advantages are that this is simple and intuitive and avoids overfitting by not choosing too many groups. The limitation is that Elbow Method might not be always well-defined in a dataset where a decrease in the Sum of Squares is not gradual. The curve begins to be flat at ***k = 4***, indicating that 4 clusters provide a good balance between model complexity and explanatory power.


# VISUAL RESULTS - PART ONE


<img width="1121" height="388" alt="image" src="https://github.com/user-attachments/assets/ae7f9ebb-457a-4e4a-a489-f4dca98c521b" />


By minimizing the distance within customers of the same group, the first trained K-Means model has been able to identify four distinct customer segments:

1) **20-40 years old customers who are moderate spenders**
2) **20-40 years old customers who are high spenders**
3) **Wide age customer group who are low spenders**
4) **Middle-aged and old customers who are moderate spenders**

The evidences are that younger customers tend to overspend based on their budget, they are the most valuable and are more likely to respond to premium promotions. Instead, mature customers should require different retention strategies in order not to lose them


# ELBOW METHOD - ANNUAL INCOME VS SPENDING SCORE


<img width="1409" height="391" alt="image" src="https://github.com/user-attachments/assets/04e1157d-dc20-4f37-815f-7b28a97ce1b0" />


The curve begins to be flat at ***k = 5***, indicating that 5 clusters provide a good balance between model complexity and explanatory power.


# VISUAL RESULTS - PART TWO


<img width="1069" height="394" alt="image" src="https://github.com/user-attachments/assets/9eb94a52-db09-4b05-8611-ad04bfcfa346" />


By minimizing the distance within customers of the same group, the second trained K-Means model has been able to identify five distinct customer segments:

- **Low Income / Low Spenders** 
- **Low Income / High Spenders** 
- **Moderate Income / Moderate Spenders** 
- **High Income / Low Spenders** 
- **High Income / High Spenders** 

The 1° group are budget-conscious or infrequent shoppers. The second group are individuals who prioritize spending only for important occasions despite having a very low income. No criticism for the 3° group, not problematic because they just have a balanced spending habits. The 4° group probably save money or shop somewhere else (considered a red flag!). Finally, the last one is made up of premium customers who are more frequent shoppers and more likely to buy luxury products

Possible recommendations are the following:

**1) The VIP customers, belonging to the 5° cluster, must keep going to have priority or early access on the last products on sale**
**2) Special discount promotions for budget-conscious households can be targeted to cluster 1**
**3) For cluster 4, the purpose is to investigate why they spend few. The solution is to conduct surveys to know their money habits, how do they save, invest, spend and finally knowing their shopping preferences in order to conduct specific retention strategies**


# TECHNOLOGIES USED

***Python***
***Pandas***
***NumPy***
***Matplotlib***
***Seaborn***
***Scikit-learn***
***Jupyter Notebook***

# FUTURE ADJUSTMENTS


The project might end here, as you can disclose simple but reasonable insights for business strategies over the short-term. However, in order to enhance it, it's possible to include additional features in the model to make it more explainable. Alternatively, deploying other clustering algorithms (e.g. **Hierarchical Clustering**, **DBSCAN**) is also important to compare the results with K-Means and make more precise conclusions.


# KEY TAKEAWAYS

Using ***Age VS Spending Score*** and ***Annual Income VS Spending Score***, K-Means clustering successfully identified four and five meaningful customer groups. These segmentations can support more *targeted marketing campaigns*, a *better allocation of advertising budgets* and *tailored promotional strategies*
