# CUSTOMER CHURN USING LOGISTIC REGRESSION MODEL

An end-to-end machine learning project using **logistic regression model** has been created to identify the probability that a customer will churn the bank in order to generate actionable corporate recommendations  

# BUSINESS PROBLEM

Imagine you're working as data scientist for a very important financial institution and the management wants to investigate its customers so they can answer to questions like:

• Which customers are likely to leave?

• What retention strategies does the management need to adopt?

In order to preserve the reputation, it's possible to identify noticeable customer groups with similar characteristics who are unstable and more likely to switching towards different competitors. In this way, the bank can prevent and avoid dissatisfaction by adopting different kind of technical strategies

# DATASET

The project uses a publicly Bank Churn Modelling dataset available from **Kaggle**. The dataset contains information about 10000 clients.


<img width="1230" height="200" alt="image" src="https://github.com/user-attachments/assets/b3aa6838-164e-4afe-93d6-3f8c60c4fd79" />


The data primarily focuses on **Exited**, the variable that has been predicted.

# EXPLORATORY DATA ANALYSIS (EDA)

Before exploring and building the regression model, I checked if there were null or duplicated values and, luckily, the problem didn't appear. Then, I start to better understand customer features.

### CREDIT SCORE

<img width="1439" height="407" alt="image" src="https://github.com/user-attachments/assets/5a6a23e3-b3a8-4164-9f3e-7aa5b544cdda" />


<img width="1433" height="429" alt="image" src="https://github.com/user-attachments/assets/52e86211-a9de-4d68-9333-23903d6837dd" />


The median credit score is approximately 650. The middle 50% of customers have a credit score between 580 and 720:

- Lower side outliers are visible, representing a small percentage 
- Since credit scores are concentrated around upper-side range, the bank serves mostly financially stable customers
- Are customers with low credit score more likely to leave the bank?

### AGE

<img width="1433" height="410" alt="image" src="https://github.com/user-attachments/assets/887f7e85-b6b8-4897-bd8d-d952d43fd565" />


<img width="1406" height="420" alt="image" src="https://github.com/user-attachments/assets/8d9e66a9-5e79-4135-83c5-42a5cbddaee0" />


Important observations and insights are the following:

- The majority of customers are 32-44 years old, so the banking clients are approximately middle-aged, neither too young nor too old 
- Higher-side outliers are visible, meaning the clients from 64 years old to afterwards aren't concentrated in the central band of data points
- Why does the bank decide to not serve elder customers? Are they more likely to leave the bank?

### BALANCE 

<img width="1420" height="422" alt="image" src="https://github.com/user-attachments/assets/4d19ef9c-c37b-4a78-ac76-cb749ed51c93" />


<img width="1403" height="248" alt="image" src="https://github.com/user-attachments/assets/15f1d150-c396-41e6-a2f5-3bf0847ca23f" />


Other important considerations:

- Despite the large spike of many customers at zero account balance, the distribution is approximately bell-shaped with a massive spread between data points
- Customers with zero balances may require targeted marketing campaigns or account activation strategies

<img width="158" height="132" alt="image" src="https://github.com/user-attachments/assets/3f02b8af-df52-4846-8679-33bc4d00678f" />


<img width="175" height="220" alt="image" src="https://github.com/user-attachments/assets/1146bbc0-b70c-4d45-80f4-fd50ce8ae135" />


Customers with no zero account balance have a churn rate of 24.1% and are more likely to leave than customers with zero balances. So, they're more likely to switch to competitors offering better interest rates, investment opportunities or premium banking services.
The bank should focus on retaining high-value customers by improving loyalty programs, personalized offers and customer service. The 51-60 age group has the highest churn rate (56%) and more likely to leave the bank than younger customers. It should be prioritized in customer retention strategies by offering tailored products or loyalty rewards

<img width="183" height="160" alt="image" src="https://github.com/user-attachments/assets/a78615f7-62a6-4b09-ab57-4099d058fb3e" />


<img width="1445" height="414" alt="image" src="https://github.com/user-attachments/assets/47c9c1df-8cb0-4d7d-97cc-846063aa1763" />


OBSERVATION AND BUSINESS INSIGHTS:

- *German clients* of 51-60 age segment are likely to churn 
- *Customer service issues* or *customer service dissatisfaction* might be two of main possible reasons

<img width="218" height="134" alt="image" src="https://github.com/user-attachments/assets/44b92a6a-93c0-4192-aa6f-357e76a0dfe5" />


No active members are more likely to leave. Will this result be confirmed when predicting the target variable on new test data?

<img width="892" height="547" alt="image" src="https://github.com/user-attachments/assets/f9f898f3-08a8-48fa-bebe-50e3fcac33fa" />


- Female customers have higher probability to churn than male ones due to the high spread difference
- Product gender differences and service expectations may contribute to this gap
- In order to reduce it, the bank should conduct feedback surveys targeting specifically female customers and design tailored solutions for retention

<img width="226" height="201" alt="image" src="https://github.com/user-attachments/assets/812d9a2a-afc0-4beb-aa36-32042180a30d" />


<img width="1392" height="512" alt="image" src="https://github.com/user-attachments/assets/48791fe3-38ec-45cf-b6b8-df07b9fe0f60" />


- Customers with 4 products have the highest churn rate but the sample size is too small and should be interpreted with caution
- Based on 266 customers, these have a very high churn rate at 82.71%. Whereas customers with two banking products appear to be the most loyal segment
- Possible recomendations might be to encourage customers with one product to adopt a second banking product, as customers with two products have the lowest churn rate
- Assist customers with 3 to 4 products by talking to high-experienced financial consultants to better manage their money

<img width="179" height="143" alt="image" src="https://github.com/user-attachments/assets/2a237ded-0e28-4255-979b-4b0caaa0e2eb" />

- A client owning a credit card or not isn't a strong predictor to be considered as one of the variables for considering a possible bank churning
- The difference is only 0.6 percentage points
- Instead, considering age, geography, balance, estimated salary, number of products, tenure and isactivemember as key decision variable in logistic regression model

# CORRELATION HEATMAP


<img width="1324" height="757" alt="image" src="https://github.com/user-attachments/assets/094ce13e-4e0a-43b6-81e2-03464c9df539" />

# BUILDING THE MODEL

I split the data into train and test set, preprocess some variables and deployed a **Logistic Regression**, which is mainly used for **classification problems** rather than predicting continuous values.

Instead of predicting a numerical value such as a price, Logistic Regression estimates the probability that an observation belongs to a particular class. For example, it can be used to predict whether a customer is likely to **leave a bank or remain a customer**, based on characteristics such as age, account activity, balance or number of products used.

The model uses the **logistic (sigmoid) function** to transform the output into a probability between 0 and 1. This makes Logistic Regression particularly useful when the business problem has a binary outcome, such as *Yes/No*, *Churn/No Churn* or *Default/No Default*.

# RESULTS

- **High German Churn (32%)**: Audit regional competitor pricing and localized branch services in Germany
- **Product Overload (3+ Products)**: Simplify multi-product packaging. Train advisors to avoid aggressive cross-selling without onboarding support
- **Middle-Aged Segment Attrition**: Create targeted wealth management, retirement planning, and premium loyalty perks for customers aged 40–60
- **Inactive Member Exposure**: Trigger automated re-engagement email workflows offering complimentary financial consultations to inactive users

# TOOLS AND FRAMEWORKS USED

- *Python*
- *Pandas*
- *NumPy*
- *Matplotlib*
- *Seaborn*
- *Scikit-learn*
- *Jupyter Notebook*

# FUTURE IMPROVEMENT

* **Temporal Data:** Incorporate longitudinal time-series features (e.g., transaction volume trends over 3/6/12 months)
* **Additional variables:** Including new customer demographical or geographic features
* **Model comparison:** Build a RandomForestClassifier to evaluate the impact on performance and exploit the hyperparameter tuning to adjust the overall accuracy

# KEY TAKEAWAYS

1. **Optimize for Business Impact, Not Just Accuracy:** In imbalanced classification, tuning for Recall and adjusting probability thresholds yields measurable revenue preservation
2. **Quality > Quantity in Cross-Selling:** Expanding client product usage works up to 2 products; beyond that, poor user experience accelerates churn
