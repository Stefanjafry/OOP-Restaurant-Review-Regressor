"""
recommended.py
==============

This module defines a custom `LinearRegression` class used to predict user-specific
restaurant review ratings based on rich engineered features.

The regression is implemented from scratch using NumPy and supports:
- Feature normalization (z-score)
- Ridge-like numerical stability with λ-regularization
- R² score computation
- User-personalized feature engineering
- Robust inference and evaluation

Feature Engineering Highlights:
-------------------------------
- Restaurant features:
    - Mean rating
    - Price level
    - Review count (raw and log-transformed)
    - Number of categories
    - Jaccard similarity with user preferences
    - Euclidean distance to city center (normalized)
- Interaction terms:
    - price × rating
    - price × category count
    - user_avg × rest_review_count
- Category embeddings:
    - Inverse-frequency weighted one-hot vectors across known categories

Main Class:
-----------
LinearRegression:
    - train(): fits a linear model to a user's reviews.
    - predict(): predicts the user's rating for a restaurant.
    - rate_all(): returns predicted and actual ratings for all restaurants.

Dependencies:
-------------
- NumPy
- Custom data abstractions: User, Restaurant, Review (imported from `abstractions.py`)

Usage:
------
The LinearRegression model is used inside `main.py` to generate per-user models and evaluate
prediction performance, both individually and globally.
"""

import numpy as np
from abstractions import Restaurant, User, Review

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.r_squared = 0
        self.X_mean = None
        self.X_std = None

    def _feature_vector(self, restaurant: Restaurant, location_center: np.ndarray, categories: list[str],
                        user_avg: float = 0.0, user_review_count: int = 0, user_categories: set[str] = None,
                        all_category_counts: dict[str, int] = None, distance_stats: tuple = None) -> list[float]:

        mean_rating = restaurant.restaurant_mean_rating()
        price = restaurant.get_price() or 0
        rest_review_count = restaurant.get_review_count()
        num_categories = len(restaurant.get_categories())
        loc = restaurant.get_location() or location_center

        # Distance 
        distance = float(np.linalg.norm(np.array(loc) - location_center)) if loc is not None else 0.0
        if distance_stats is not None:
            dist_mean, dist_std = distance_stats
            distance = (distance - dist_mean) / dist_std if dist_std > 0 else 0.0

        # Log transform 
        log_review_count = np.log1p(rest_review_count)

        # --- Interaction features ---
        interaction_price_rating = price * mean_rating
        interaction_price_cats = price * num_categories
        interaction_user_restcount = user_avg * rest_review_count

        # Category features 
        rest_categories = set(restaurant.get_categories())

        # Jaccard similarity (user categories vs restaurant categories)
        jaccard = 0.0
        if user_categories:
            inter = rest_categories.intersection(user_categories)
            union = rest_categories.union(user_categories)
            jaccard = len(inter) / len(union) if union else 0.0

        # One-hot category vector with inverse frequency weighting
        category_vector = []
        for cat in categories:
            if cat in rest_categories:
                weight = 1.0 / (1 + all_category_counts.get(cat, 0))
                category_vector.append(weight)
            else:
                category_vector.append(0.0)

        return [
            mean_rating,
            price,
            rest_review_count,
            num_categories,
            log_review_count,
            distance,
            interaction_price_rating,
            interaction_price_cats,
            interaction_user_restcount,
            jaccard,
            user_avg,
            user_review_count
        ] + category_vector

    def _design_matrix_and_target(self, reviews: list[Review], restaurants: list[Restaurant],
                                  user: User, categories: list[str]) -> tuple[np.ndarray, np.ndarray]:

        restaurant_map = {r.get_name(): r for r in restaurants}
        locations = [r.get_location() for r in restaurants if r.get_location() is not None]
        location_center = np.mean(locations, axis=0) if locations else np.array([0.0, 0.0])

        user_avg = user.get_average_stars()
        user_review_count = user.get_review_count()
        user_categories = set(cat for r in user.get_reviewed_restaurants(restaurants) for cat in r.get_categories())

        all_category_counts = {}
        for r in restaurants:
            for c in r.get_categories():
                all_category_counts[c] = all_category_counts.get(c, 0) + 1

        dists = [float(np.linalg.norm(np.array(r.get_location()) - location_center)) for r in restaurants if
                 r.get_location()]
        dist_mean = np.mean(dists)
        dist_std = np.std(dists)

        X = []
        y = []

        for review in reviews:
            rname = review.get_restaurant_name()
            rating = review.get_rating()
            restaurant = restaurant_map.get(rname)
            if restaurant is None:
                continue
            features = self._feature_vector(
                restaurant,
                location_center,
                categories,
                user_avg=user_avg,
                user_review_count=user_review_count,
                user_categories=user_categories,
                all_category_counts=all_category_counts,
                distance_stats=(dist_mean, dist_std)
            )
            X.append(features)
            y.append(rating)

        return np.array(X), np.array(y)

    def _least_squares_fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        lambda_identity = 1e-8 * np.eye(X.shape[1])
        XTX = X.T @ X + lambda_identity
        XTy = X.T @ y
        return np.linalg.solve(XTX, XTy)

    def train(self, user: User, restaurants: list[Restaurant], categories: list[str]) -> None:
        reviews = user.get_reviews()
        if len(reviews) < 2:
            self.coefficients = None
            self.r_squared = 0
            return

        X, y = self._design_matrix_and_target(reviews, restaurants, user, categories)
        if len(y) < 2:
            self.coefficients = None
            self.r_squared = 0
            return

        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_normalized = (X - self.X_mean) / np.where(self.X_std == 0, 1, self.X_std)
        X_b = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])
        beta = self._least_squares_fit(X_b, y)
        self.coefficients = beta

        y_pred = X_b @ beta
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        self.r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    def predict(self, restaurant: Restaurant, restaurants: list[Restaurant],
                categories: list[str], user: User) -> float:
        if self.coefficients is None:
            return 0.0

        locations = [r.get_location() for r in restaurants if r.get_location() is not None]
        location_center = np.mean(locations, axis=0) if locations else np.array([0.0, 0.0])

        user_avg = user.get_average_stars()
        user_review_count = user.get_review_count()
        user_categories = set(cat for r in user.get_reviewed_restaurants(restaurants) for cat in r.get_categories())

        all_category_counts = {}
        for r in restaurants:
            for c in r.get_categories():
                all_category_counts[c] = all_category_counts.get(c, 0) + 1

        dists = [float(np.linalg.norm(np.array(r.get_location()) - location_center)) for r in restaurants if
                 r.get_location()]
        dist_mean = np.mean(dists)
        dist_std = np.std(dists)

        features = np.array(self._feature_vector(
            restaurant,
            location_center,
            categories,
            user_avg=user_avg,
            user_review_count=user_review_count,
            user_categories=user_categories,
            all_category_counts=all_category_counts,
            distance_stats=(dist_mean, dist_std)
        ))

        features = (features - self.X_mean) / np.where(self.X_std == 0, 1, self.X_std)
        features_b = np.insert(features, 0, 1.0)
        return float(np.dot(features_b, self.coefficients))

    def rate_all(self, user: User, restaurants: list[Restaurant], categories: list[str]) -> dict[str, tuple[float, float | None]]:
        self.train(user, restaurants, categories)
        predictions = {}
        reviewed_names = {r.get_restaurant_name() for r in user.get_reviews()}
        for r in restaurants:
            predicted = self.predict(r, restaurants, categories, user)
            actual = None
            if r.get_name() in reviewed_names:
                for review in user.get_reviews():
                    if review.get_restaurant_name() == r.get_name():
                        actual = review.get_rating()
                        break
            predictions[r.get_name()] = (predicted, actual)
        return predictions
