
# Sephora Product Price Predictions 

## 📜 Project Overview
The goal of this project is to predict skincare product prices using a regression model trained on a dataset of Sephora products and reviews. By analyzing features such as brand, ratings, number of reviews, and product type, I aim to uncover insights into the factors driving product pricing.

## Problem Statement 
With this project, I am looking to answer these questions through regression:

1. How accurately can a regression model predict the price of skincare products based on features such as brand, rating, number of reviews, and product type?

2. What features most significantly impact the pricing of skincare products, and how do these features vary across different product categories?

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

## What is Regression and How Does it Work? 
Regression analysis predicts a dependent variable (e.g., price) based on independent variables (e.g., ratings, reviews). It identifies trends and quantifies relationships between variables.

Coefficients(B) are estimated using Least Squares, which minimizes the sum of squared errors.

<img width="866" height="396" alt="Screenshot 2025-09-25 105425" src="https://github.com/user-attachments/assets/f0145279-e984-469a-b4aa-8eaf9ed4254c" />

Assumptions for Linear Regression:
- Linearity between predictors and target.
- No multicollinearity among predictors.
- Homoscedasticity (constant variance of residuals).
- Residuals follow a normal distribution.

---

## Experiment #1 
### Data Understanding 
Before diving into preprocessing for this project, I revisited the initial steps I completed in Project 1 to gain a thorough understanding of my dataset. Here's a review of the key steps and findings from that phase:

With Project 1, I discovered that completing the data cleaning and categorization process helped me analyze and visualize the relationship between product categories and ratings. I was able to answer my initial questions through data visualization.

These were my problem statement questions:
- Is there a correlation between the price of a skincare product and its customer rating on Sephora? 
- Which product categories (ex. cleansers, moisturizers, serums) tend to have the highest customer ratings? 

I discovered that there was a weak correlation between price and rating. This implied that price may influence purchasing decisions, but it does not strongly affect the reviews customers leave on a scale of 1–5.  Within the bar graph, I discovered that the skincare products with the highest rating were moisturizers, a top-rated category, followed by cleansers and sunscreens. This discovery has helped me gain an understanding of customer satisfaction and can help guide businesses in optimizing their product offerings.


This is the bar graph I previously created in my first project: 
<img width="503" height="478" alt="Screenshot 2025-09-25 105749" src="https://github.com/user-attachments/assets/324d3b74-b2ae-47c1-a95a-91b89bfa74b3" />


<img width="617" height="370" alt="Screenshot 2025-09-25 105847" src="https://github.com/user-attachments/assets/10424788-96ed-43ab-b33c-b482f338d3a1" />

Scatterplot that shows the correlation between price and rating of a product 
<img width="577" height="358" alt="Screenshot 2025-09-25 110008" src="https://github.com/user-attachments/assets/80e3d407-568f-4781-9ed7-747a7c8112ab" />

## Preprocessing 
Preprocessing is a crucial step in the OSEMN pipeline, as it involves “cleaning” the data to ensure accuracy and reliability. By addressing errors, handling missing or inaccurate values, and standardizing data, this step creates a solid foundation for meaningful analysis. Here's how I approached preprocessing for my project:

1. Organizing the Data
The first step is to gather and organize the data in a way that reveals the key information needed to create visualizations and explore the relationship between different features such as price and ratings. By doing this, I can begin to uncover patterns between different products and their ratings, which is key to the analysis.

2. Cleaning the Data










