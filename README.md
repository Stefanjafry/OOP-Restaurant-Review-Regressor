Restaurant Review Rating Predictor Regression-Based ML System in Python

This project implements a complete restaurant rating prediction system using Object-Oriented Programming (OOP) and mathematical regression techniques. The system models user preferences and restaurant metadata to predict review ratings using hand-crafted feature engineering and linear regression, all built from scratch using Python and NumPy only (no scikit-learn or other ML libraries).

Overview:

The application showcases:

A fully modular and object-oriented design for users, restaurants, and reviews

Manual implementation of linear regression, ridge regression, and Huber regression

End-to-end data ingestion from real-world JSON datasets

Feature engineering including interaction terms, log transforms, and geospatial normalization

Model evaluation and visualization with R², MAE, RMSE, and predicted vs. actual rating plots

Key Components:

Review: Represents a single user’s rating of a restaurant, storing the restaurant name and numerical score.

Restaurant: Encapsulates all metadata about a restaurant, including location (latitude/longitude), category list, price, and all received reviews. Computes internal stats like average rating and category count.

User: Represents a review-writing user. Stores reviews and exposes functionality to compute average rating, number of reviews, and review history.

LinearRegression (in recommended.py):

Implements closed-form OLS regression with optional Ridge or Huber loss

Trains user-specific and global models using NumPy matrix algebra

Supports robust fitting, outlier handling, and feature normalization

Feature Engineering:

Raw features: price, category count, mean rating, review count

Transforms: log(review_count), z-scored geographic distance from user center

Interactions: price × rating, price × category count, user average × review count

Jaccard similarity between user and restaurant categories

One-hot category encoding with inverse frequency weighting

Evaluation and Visualization:

R², MAE, RMSE computed manually for each user and the global model

Plots of predicted vs. actual ratings using matplotlib

Plots are displayed interactively and saved to disk (./plots/)

Data Pipeline (in data.py):

Parses and links three JSON datasets: users.json, reviews.json, and restaurants.json

Converts raw JSON objects into structured Python classes

Associates reviews with both users and restaurants

Computes global category set and user category preferences

Utilities (in utilities.py):

Provides functional-style tools like map_and_filter, sorted_restaurants, and average calculations

File Structure:

datafolder/
users.json : User profiles and metadata
reviews.json : Individual review records
restaurants.json : Restaurant attributes, prices, coordinates, and categories

main.py : Main program launcher and evaluation script
recommended.py : Linear, Ridge, and Huber regression implementation with feature generation
abstractions.py : Data models for User, Restaurant, and Review
utilities.py : Functional helpers and aggregate operations
data.py : Dataset loading, linkage, and cleaning logic
README.md : Project documentation (this file)
requirements.txt : Python dependencies (NumPy + Matplotlib)
.gitignore : Common build artifacts and data exclusions

Setup Instructions:

Clone the repository:

git clone https://github.com/your-username/restaurant-review-regressor.git
cd restaurant-review-regressor

(Optional) Create and activate a virtual environment:

python -m venv .venv
On Windows: .venv\Scripts\activate
On Mac/Linux: source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Running the Project:

To launch the model evaluation and visualization script, run:

python main.py

You will be prompted to enter a minimum number of reviews per user (e.g., 5). The system will then:

Train models for all eligible users

Evaluate prediction accuracy (R², MAE, RMSE)

Train and evaluate a global model

Plot predicted vs. actual ratings

Save outputs to ./plots/ and predicted_ratings.csv

License:

This project is licensed under the MIT License. You are free to use, modify, and distribute the code for personal, academic, or commercial use.
