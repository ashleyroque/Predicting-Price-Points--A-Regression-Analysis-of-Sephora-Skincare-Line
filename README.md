# Sephora Product Price Predictions 💄💰

---

## Project Overview

The goal of this project is to predict skincare product prices using a regression model trained on a dataset of Sephora products and reviews. By analyzing features such as brand, ratings, number of reviews, and product type, I aim to uncover insights into the factors driving product pricing.


---

## Problem Statement & Data Introduction

With this project, I am looking to answer these questions through regression:

### Question #1
How accurately can a regression model predict the price of skincare products based on features such as brand, rating, number of reviews, and product type?

### Question #2
What features most significantly impact the pricing of skincare products, and how do these features vary across different product categories?

This project is an extension of Project 1 but it has the goal of predicting skincare product prices using a regression model trained on a dataset of Sephora Products and Skincare Reviews Dataset. By analyzing features such as brand, ratings, number of reviews, and product type, I aim to uncover insights into the factors driving product pricing.

### Dataset Key Features

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

---

## What is Regression and How Does it Work

Explain what regression is and how it works (specifically linear regression as it was covered in class).

### Definition and Purpose
Regression analysis predicts a dependent variable (e.g., price) based on independent variables (e.g., ratings, reviews). It identifies trends and quantifies relationships between variables.

Coefficients(B) are estimated using Least Squares, which minimizes the sum of squared errors.

### Focus on Linear Regression
This is the formula for linear regression. Note: random error is also written as "u".

**Formula:**
