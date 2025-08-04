"""
main.py
=======

This is the entry point for the Restaurant Review Regression system.

It evaluates how well restaurant review ratings can be predicted using
object-oriented models of users, restaurants, and reviews.

Key Features:
-------------
- Trains per-user regression models using custom OLS/Ridge/Huber regression.
- Trains a global regression model across all users.
- Extracts restaurant features (location, categories, review counts, etc.).
- Generates scatter plots for each user and the global model:
    - Shows predicted vs. actual ratings
    - Annotates R², MAE, and RMSE on each plot
    - Color-coded by residual magnitude
- Automatically deletes and regenerates the `./plots/` folder each run.
- Exports all predictions to `predicted_ratings.csv`.

Workflow:
---------
1. Users with ≥ N reviews (user-defined) are selected for plotting.
2. For each eligible user:
   - A personalized model is trained on their reviews.
   - A plot is generated showing prediction accuracy.
3. A global model is trained using all review data (with robust Huber loss).
4. Evaluation metrics (R², MAE, RMSE) are printed and visualized.

Modules Used:
-------------
- `abstractions.py`: Defines User, Restaurant, and Review objects.
- `recommended.py`: Contains regression models (OLS, Ridge, Huber).
- `data.py`: Loads and links user, review, and restaurant data.
- `utilities.py`: Optional helper functions.
- Standard Python libraries: NumPy, Matplotlib, CSV, Concurrency.
Author:
-------
Stefan Jafry, 2025
"""

import numpy as np
import csv
import os
import shutil
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from abstractions import Restaurant, User
from recommended import LinearRegression
from data import USERS, ALL_RESTAURANTS, CATEGORIES
from scipy.optimize import minimize

# === Clean plots folder ===
if os.path.exists("plots"):
    shutil.rmtree("plots")
os.makedirs("plots", exist_ok=True)

def normalize(name: str) -> str:
    return name.strip().lower()

restaurant_lookup = {normalize(r.get_name()): r for r in ALL_RESTAURANTS}

def plot_user(user_name, actuals, predicted, r2):
    plt.figure(figsize=(7, 6))
    jitter = np.random.normal(0, 0.05, size=len(actuals))
    residuals = np.array(predicted) - np.array(actuals)
    colors = np.abs(residuals)

    actuals_np = np.array(actuals)
    predicted_np = np.array(predicted)
    mae = np.mean(np.abs(predicted_np - actuals_np))
    rmse = np.sqrt(np.mean((predicted_np - actuals_np) ** 2))

    plt.scatter(actuals_np + jitter, predicted_np, c=colors, cmap='viridis', edgecolor='k', s=60)
    plt.plot([1, 5], [1, 5], linestyle='--', color='red', label='Ideal')
    for a, p in zip(actuals, predicted):
        plt.plot([a, a], [a, p], color='gray', alpha=0.3)

    plt.xlabel("Actual Rating", fontsize=12)
    plt.ylabel("Predicted Rating", fontsize=12)
    plt.title(f"{user_name} | R² = {r2:.2f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}", fontsize=13)
    cbar = plt.colorbar(label="|Residual|")
    cbar.ax.tick_params(labelsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{user_name.replace(' ', '_')}_plot.png")
    plt.close()

def process_user(user: User, min_reviews_to_plot: int):
    model = LinearRegression()
    reviewed = user.get_reviewed_restaurants(ALL_RESTAURANTS)
    if len(reviewed) < 2:
        return None

    is_eligible = len(reviewed) >= min_reviews_to_plot

    results = model.rate_all(user, ALL_RESTAURANTS, list(CATEGORIES))
    if model.coefficients is None:
        return None

    user_actuals = []
    user_preds = []
    preds = []

    for rest in reviewed:
        name = rest.get_name()
        pred, actual = results[name]
        if actual is not None:
            user_actuals.append(actual)
            user_preds.append(pred)
            preds.append({
                "User": user.get_name(),
                "Restaurant": name,
                "PredictedRating": round(pred, 3),
                "ActualRating": round(actual, 3),
                "R2": round(model.r_squared, 4)
            })

    return {
        "user_name": user.get_name(),
        "is_eligible": is_eligible,
        "r2": model.r_squared,
        "actuals": user_actuals,
        "preds": user_preds,
        "records": preds
    }

def evaluate(min_reviews_to_plot=5):
    all_predictions = []
    user_r2s = []
    plotted_users = 0
    eligible_users = 0

    print("Evaluating each user (parallelized)...")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_user, user, min_reviews_to_plot) for user in USERS]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue

            user_r2s.append(result["r2"])
            all_predictions.extend(result["records"])

            if result["is_eligible"]:
                eligible_users += 1
                if len(result["actuals"]) >= min_reviews_to_plot:
                    plot_user(result["user_name"], result["actuals"], result["preds"], result["r2"])
                    plotted_users += 1

    with open("predicted_ratings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["User", "Restaurant", "PredictedRating", "ActualRating", "R2"])
        writer.writeheader()
        writer.writerows(all_predictions)

    if plotted_users == 0:
        print(f"Sorry, no users with ≥{min_reviews_to_plot} reviews available to plot.")

    avg_r2 = sum(user_r2s) / len(user_r2s) if user_r2s else 0
    print(f"\nUsers with ≥{min_reviews_to_plot} reviews: {eligible_users}")
    print(f"Users successfully plotted: {plotted_users}")
    print(f"Average R² across users: {avg_r2:.3f}")
    print(f"Saved predictions to predicted_ratings.csv")
    print(f"Saved plots to ./plots/")

    return all_predictions

def train_global_model(predicted_records):
    print("\nTraining global model across all users...")

    model = LinearRegression()
    features = []
    targets = []

    locations = [r.get_location() for r in ALL_RESTAURANTS if r.get_location() is not None]
    location_center = np.mean(locations, axis=0) if locations else np.array([0.0, 0.0])

    for rec in predicted_records:
        name = normalize(rec["Restaurant"])
        restaurant = restaurant_lookup.get(name)
        if not restaurant:
            continue
        try:
            vec = model._feature_vector(
                restaurant,
                location_center,
                list(CATEGORIES),
                user_avg=3.5,
                user_review_count=20,
                user_categories=None,
                all_category_counts={},
                distance_stats=(0.0, 1.0)
            )
            features.append(vec)
            targets.append(rec["ActualRating"])
        except Exception:
            continue

    if not features:
        print("[GLOBAL MODEL] No valid training data.")
        return

    X = np.array(features)
    y = np.array(targets)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    def huber_loss(beta):
        y_pred = X @ beta
        residual = y - y_pred
        delta = 1.0
        loss = np.where(np.abs(residual) <= delta,
                        0.5 * residual**2,
                        delta * (np.abs(residual) - 0.5 * delta))
        return np.sum(loss)

    result = minimize(huber_loss, np.zeros(X.shape[1]), method='BFGS')
    beta = result.x
    y_pred = X @ beta

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - ss_res / ss_total if ss_total > 0 else 0.0
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    plt.figure(figsize=(7, 6))
    jitter = np.random.normal(0, 0.05, size=len(y))
    residuals = y_pred - y
    colors = np.abs(residuals)

    plt.scatter(y + jitter, y_pred, c=colors, cmap='plasma', edgecolor='k', s=60)
    plt.plot([1, 5], [1, 5], linestyle='--', color='red')
    for a, p in zip(y, y_pred):
        plt.plot([a, a], [a, p], color='gray', alpha=0.2)

    plt.xlabel("Actual Rating", fontsize=12)
    plt.ylabel("Predicted Rating", fontsize=12)
    plt.title(f"Global Model | R² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}", fontsize=13)
    cbar = plt.colorbar(label="|Residual|")
    cbar.ax.tick_params(labelsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("plots/global_model_plot.png")
    plt.close()

    print(f"[GLOBAL MODEL] R² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f} | Plot saved to plots/global_model_plot.png")

if __name__ == "__main__":
    try:
        n = int(input("Minimum reviews required to generate user plot (default=5): ") or "5")
    except ValueError:
        n = 5
    all_preds = evaluate(min_reviews_to_plot=n)
    train_global_model(all_preds)
