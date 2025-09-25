
# Sephora Product Price Predictions 

## 📜 Project Overview
The goal of this project is to predict skincare product prices using a regression model trained on a dataset of Sephora products and reviews. By analyzing features such as brand, ratings, number of reviews, and product type, I aim to uncover insights into the factors driving product pricing.

## 📌 Problem Statement 
### With this project, I am looking to answer these questions through regression:

*1. How accurately can a regression model predict the price of skincare products based on features such as brand, rating, number of reviews, and product type?*

*2. What features most significantly impact the pricing of skincare products, and how do these features vary across different product categories?*

The goal of this project is to predict skincare product prices using a regression model trained on a dataset of Sephora Products and Skincare Reviews Dataset. By analyzing features such as brand, ratings, number of reviews, and product type, I aim to uncover insights into the factors driving product pricing.

---

## 📊 Introducing the Data   
The dataset used is sourced from [Kaggle: Instagram Data](https://www.kaggle.com/datasets/propriyam/instagram-data).  

**Key Features**:  
- **Product Name:** The name of the Sephora product.
- **Brand:** The manufacturer or brand of the product.
- **Category:** The type of product, such as moisturizers, cleansers, or sunscreens.
- **Price:** The cost of the product in USD.
- **Rating:** Average customer rating on a scale from 1 to 5.
- **Number of Reviews:** Total number of reviews for the product.
- **Review Text:** Individual customer feedback provided as text.
- **Skin Type:** Information about the skin type of the reviewer (e.g., dry, oily, combination).
- **Skin Tone:** Descriptions of skin tone, if provided by the reviewer.
- **Hair Color:** Reviewers' hair color, if available.
- **Eye Color:** Reviewers' eye color, if provided.

## 💡 What is Regression and How Does it Work? 
Regression analysis predicts a dependent variable (e.g., price) based on independent variables (e.g., ratings, reviews). It identifies trends and quantifies relationships between variables.

Coefficients(B) are estimated using Least Squares, which minimizes the sum of squared errors.

### Assumptions for Linear Regression:
- Linearity between predictors and target.
- No multicollinearity among predictors.
- Homoscedasticity (constant variance of residuals).
- Residuals follow a normal distribution.


<img width="866" height="396" alt="Screenshot 2025-09-25 105425" src="https://github.com/user-attachments/assets/f0145279-e984-469a-b4aa-8eaf9ed4254c" />


---

## 📋 Experiment #1 
### Data Understanding 
Before beginning the preprocessing phase, I conducted an exploratory data analysis to understand the dataset's structure, distributions, and underlying patterns. This builds upon insights gained from Project 1 while focusing specifically on features relevant to price prediction.

### Dataset Overview
- **Total Products:** 8,000 skincare products from Sephora
- **Price Range:** $[min] to $[max] with average price of $[average]
- **Rating Distribution:** Average rating of [average] out of 5 across all products
- **Key Features Analyzed:** Price, rating, number of reviews, brand, category, and customer demographics

### Key Insights from Preliminary Analysis
#### Correlation Analysis
Building on Project 1 findings, I confirmed a **weak correlation between price and rating** (correlation coefficient: 0.1110412755157950284). This suggests that while price may influence purchasing decisions, it has minimal impact on customer satisfaction ratings.

This is the bar graph I previously created in my first project: 
<img width="503" height="478" alt="Screenshot 2025-09-25 105749" src="https://github.com/user-attachments/assets/324d3b74-b2ae-47c1-a95a-91b89bfa74b3" />


Scatterplot that shows the correlation between price and rating of a product 
<img width="577" height="358" alt="Screenshot 2025-09-25 110008" src="https://github.com/user-attachments/assets/80e3d407-568f-4781-9ed7-747a7c8112ab" />

#### Category Performance Trends
The analysis revealed distinct patterns across product categories:
- **Highest Rated Categories:** Treatments (avg rating:  4.326238), followed by masks (avg rating: 4.225166)  moisturizers (avg rating: 4.149276)  
- **Price Variations:** Significant price differences observed across categories, with [specific category] being the most expensive on average

#### Business Implications
These findings suggest that:
- Customer satisfaction is driven more by product efficacy than price point
- Different categories may warrant distinct pricing strategies
- Brand reputation and specific features likely play larger roles in pricing than customer ratings alone

### Connection to Current Project
These insights inform our price prediction approach by:
1. Highlighting which features show meaningful relationships with price
2. Identifying potential confounding variables to consider
3. Providing baseline expectations for model performance
4. Suggesting that non-linear relationships may exist beyond simple correlations


## ⚙️ Preprocessing 
Preprocessing is a crucial step in the OSEMN pipeline, as it involves “cleaning” the data to ensure accuracy and reliability. By addressing errors, handling missing or inaccurate values, and standardizing data, this step creates a solid foundation for meaningful analysis. Here's how I approached preprocessing for my project:

1. Organizing the Data
The first step is to gather and organize the data in a way that reveals the key information needed to create visualizations and explore the relationship between different features such as price and ratings. By doing this, I can begin to uncover patterns between different products and their ratings, which is key to the analysis.

2. Cleaning the Data
To ensure data quality and reliability, I implemented a comprehensive cleaning pipeline with the following steps:
## 2. Data Cleaning Process

To ensure data quality and reliability, I implemented a comprehensive cleaning pipeline with the following steps:

###  Dataset Inspection
- Conducted thorough examination of the dataset structure and completeness
- Identified missing values, inconsistencies, and data type issues
- Analyzed value distributions across all features

### Handling Missing Values
- For fields like skin tone, eye color, skin type, and hair color, I replaced missing values with placeholders labeled **'unknown'**.
- For text fields, such as review titles or content, missing values were filled with placeholders like **'no review'** or **'no title'**.


### Standardized Categories
- Categories such as **"moisturizer"** and **"Moisturizer"** were inconsistent. To address this, I standardized these entries by converting all values to lowercase, preventing duplicate entries due to case sensitivity.
  
```python
# Example: Converting inconsistent category labels
df['category'] = df['category'].str.lower()  # 'Moisturizer' → 'moisturizer'
```

- Numerical features (ratings, number of reviews)
  
### Ensuring Numerical Data Types
- I confirmed that all numerical fields were correctly formatted to facilitate further analysis.
  
### Trained a Linear Regression model using features such as rating, number of reviews, and category 
### One-hot encoded categorical values like brand and category. 

By following these steps, I transformed the raw dataset into a clean, standardized version, ready for visualization and analysis. This pre-processing part of the project was essential for ensuring that results were meaningful and reliable. 


## 📈 Modeling 
In the first experiment, I applied a simple Linear Regression model to predict the price of skincare products using features such as rating, number of reviews, and price. After preprocessing the data (handling missing values and encoding categorical variables), I split the dataset into training and testing sets to train the regression model. The linear regression model used the following equation:

<img width="619" height="359" alt="Screenshot 2025-09-25 111218" src="https://github.com/user-attachments/assets/3628d06e-d616-4b45-918c-efbbd3c96581" />

## 📝Evaluation
The model was evaluated based on the Root Mean Squared Error (RMSE), which quantifies the difference between the predicted and actual prices.


RMSE for this model was **47.78**, which represents the average error in predicting the price of skincare products. This indicates that, on average, the predicted prices deviate from the actual prices by approximately $47.78.


This value serves as a baseline for model accuracy but indicates potential for improvement, specifically if most product prices in the dataset fall within a similar range. To boost performance, I plan to refine preprocessing steps, explore additional features, and test more advanced models in future experiments. Through this process, I aim to reduce the RMSE and enhance predictive accuracy by better capturing the complexities of pricing behavior.


## 📋 Experiment #2
After evaluating the first model, I chose to use a *Random Forest Regressor* instead of Linear Regression to evaluate whether a nonlinear model could better capture the underlying patterns in the dataset. This model was trained on the same features as the first experiment.

This is what I found: 
- Using a Random Forest model gave a  **Root Mean Squared Error (RMSE) of 59.33**

- Comparing this to the Linear Regression model, which was a RMSE of 47.78-- the Random Forest model gave a higher RMSE. 

- This reveals that the Random Forest model did not perform as well compared to the Linear Regression model in predicating the target variable. 

One reason for this result could be that the dataset doesn’t have strong nonlinear patterns for the Random Forest model to take advantage of, or the model’s default settings might not be the best fit for this problem. Another possible factor is the size of the dataset, as Random Forest usually works better with larger datasets and a wider variety of features.

## 📋 Experiment #3

In this experiment, I focused on using the **brand_name** feature as a predictor. Since brand_name is a categorical variable. I applied one-hot encoding to convert the unique brands into numerical representations that could be used for training.I chose linear regression as the model to evaluate how well the brand alone can predict the target variable, which in this case was the product rating.

Result(s): 
- The **Root Mean Squared Error (RMSE)** for this experiment was 0.4452. 

Insights: 
- This brand feature alone significantly predicts ratings, suggesting that certain features like brand can have stronger individual predictive power for specific outcomes. 

Comparison from previous experiment: 
- Experiment 1 (Linear Regression with multiple features) resulted in a much higher RMSE (47.78), indicating that the combination of features may have included some noise or irrelevant data that hindered the model's predictive ability.

- Experiment 2 (Random Forest with multiple features) yielded an RMSE of 59.33, which was even higher than Experiment 1, showing that Random Forest might not be the most suitable model for this specific task with the selected features.

Interpretation:
- This experiment shows that focusing on a single well-defined feature, such as brand_name, can significantly reduce error when predicting ratings. This suggests that certain features might have a stronger predictive power on their own, and using a simpler approach can sometimes outperform models that include more features.

##  Impact Analysis 
This section discusses the potential broader implications of the project, considering both positive applications and unintended negative consequences.

### ✅ Positive Impacts

#### **For Consumers**
- **Informed Purchasing Decisions:** Helps customers compare products by identifying fair pricing strategies based on objective features rather than marketing alone.
- **Price Transparency:** Provides insights into what factors actually drive product costs, empowering consumers to understand what they're paying for.

#### **For Businesses**
- **Data-Driven Pricing Strategies:** Offers retailers insights into how different factors (brand reputation, product category, customer ratings) influence perceived value and price sensitivity.
- **Market Analysis:** Helps identify pricing gaps and opportunities in the skincare market by analyzing competitor positioning.

### ⚠️ Negative Impacts & Ethical Considerations

#### **Price Discrimination Risk**
- **Barriers to Entry:** Insights from the model could be used to justify premium pricing for established brands, potentially creating market barriers for smaller or emerging brands.
- **Dynamic Pricing Abuse:** The methodology could be adapted for real-time price optimization that maximizes profit at the expense of consumer fairness.

#### **Algorithmic Bias Concerns**
- **Popularity Bias:** The dataset's inherent bias toward popular, well-reviewed products could skew predictions, disadvantaging newer or niche products with fewer reviews.
- **Representation Gaps:** If the dataset lacks diversity in product types or price ranges, the model may perform poorly for underrepresented segments.

#### **Market Concentration**
- **Competitive Disadvantages:** Larger companies with more resources could leverage these insights to further dominate the market, potentially reducing competition and consumer choice.



## Conclusion

This project explored various approaches to predicting skincare product prices using machine learning regression models. Through multiple experiments, several key insights emerged about model performance, feature importance, and data preprocessing.

### 🔑 Key Learnings

#### **Model Performance**
- **Linear Regression** provided a strong baseline (RMSE: 47.78), demonstrating that linear relationships can capture fundamental pricing patterns in the dataset.
- **Random Forest** underperformed compared to linear regression (RMSE: 59.33), suggesting that either the dataset lacks strong nonlinear patterns or requires more sophisticated hyperparameter tuning.
- **Feature-specific modeling** using only brand information achieved significantly better results for rating prediction (RMSE: 0.4452), highlighting the power of targeted feature selection.

#### **Feature Importance**
- **Brand dominance** emerged as a critical factor, with brand-alone models outperforming multi-feature approaches for certain prediction tasks.
- **Feature selection** proved crucial - simpler models with well-chosen features sometimes outperformed more complex models with numerous variables.
- **Data quality** matters significantly: proper handling of missing values and categorical encoding directly impacted model reliability.

#### **Methodological Insights**
- The **iterative experimentation process** was valuable for understanding the problem space and model limitations.
- **Baseline models** are essential for establishing performance benchmarks before moving to more complex approaches.

### 🚀 Next Steps & Future Work

Based on the findings from this project, several directions for future improvement include:

#### **Feature Engineering**
- Explore additional features such as product ingredients, marketing data, or temporal trends
- Investigate text mining of review content for sentiment analysis features
- Create composite features that capture brand-category interactions

#### **Model Enhancement**
- Experiment with advanced ensemble methods (Gradient Boosting, XGBoost, LightGBM)
- Implement hyperparameter optimization for Random Forest and other models
- Test neural network approaches for capturing complex feature interactions

#### **Data Expansion**
- Expand the dataset to include more product types, brands, and price ranges
- Incorporate external data sources such as competitor pricing or market trends
- Address dataset biases to improve model generalizability

#### **Production Considerations**
- Develop model interpretation tools to explain pricing factors to stakeholders
- Create a pipeline for continuous model retraining with new data
- Implement monitoring systems to detect model drift and performance degradation

This project demonstrates that while simple models can provide valuable insights, there remains significant opportunity for improvement through more sophisticated feature engineering, model selection, and data collection strategies.


## References

### Technical Documentation & Libraries
- [scikit-learn: Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [scikit-learn: Random Forest Classifier](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### Educational Resources
- [IBM: What is Linear Regression?](https://www.ibm.com/topics/linear-regression)
- [Statistics by Jim: Root Mean Square Error (RMSE)](https://statisticsbyjim.com/regression/root-mean-square-error-rmse/)
- [DataCamp: One-Hot Encoding in Python](https://www.datacamp.com/tutorial/one-hot-encoding-python-tutorial)

### Dataset Source
- [Kaggle: Sephora Products and Skincare Reviews Dataset](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data)


