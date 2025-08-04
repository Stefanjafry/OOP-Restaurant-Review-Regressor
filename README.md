Restaurant Review Rating Predictor – Regression-Based ML System in Python (No Machine Learning Libraries)

This project implements a complete machine learning system in pure Python to predict restaurant review ratings using custom-built regression models. It is designed around strong Object-Oriented Programming (OOP) principles and a fully modular architecture. The entire system is built from scratch using only NumPy.

The goal is to model both individual user preferences and general review behavior by learning from user-restaurant-review triples. The system supports per-user regression models as well as a global model trained across users.

Key Features:

Object-Oriented Design:

Fully encapsulated abstractions for Review, Restaurant, and User

Clean modular class structure, designed for scalability and maintainability

Robust handling of restaurant features, user metadata, and review histories

Custom Regression Models (No ML Libraries Used):

Manual implementation of Ordinary Least Squares (OLS) regression

Ridge Regression with L2 regularization

Huber Regression for robust fitting and outlier resistance

Closed-form solution for each model using matrix algebra and regularization

Feature Engineering and Interaction Terms:

Raw features: restaurant mean rating, price, review count, number of categories

Interaction terms: price multiplied by mean rating, price multiplied by category count, and more

Jaccard similarity between user and restaurant categories

Log-transformed review count for scale normalization

Distance normalization between user’s central location and restaurant location (with z-score scaling)

Evaluation and Visualization:

Manual calculation of evaluation metrics: R-squared (R²), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)

Automatic plotting of predicted vs. actual review ratings for each user using matplotlib

Global plot across all users who meet a review count threshold

Output plots are generated in the "plots" folder and cleaned with each new execution

Dataset Integration:

JSON-based ingestion of Yelp-style review datasets

Restaurant-level metadata: name, price, categories, coordinates, reviews

User-level metadata: name, review history, average rating

Review-level data: restaurant link, rating given

Scalable Evaluation:

Batch processing of all eligible users

Ability to control which users are modeled using a dynamic review count threshold

Efficient design with data preloading and mapping

Code Structure:

main.py: end-to-end controller for training and evaluating models

abstractions.py: clean data classes for Users, Restaurants, and Reviews

recommended.py: contains regression logic and manual training code

data.py: handles JSON I/O and linkage between users, reviews, and restaurants

utilities.py: helper functions for sorting, filtering, and aggregation

No External ML Libraries:

All modeling logic is implemented using basic linear algebra via NumPy

No scikit-learn, no external fitting libraries — just pure mathematics and matrix operations


How to Use:

Install dependencies from requirements.txt

Place your dataset files (users.json, reviews.json, restaurants.json) into the datafolder directory

Run main.py and provide a minimum review count when prompted

Results will be displayed interactively and saved as plots

License: MIT License
