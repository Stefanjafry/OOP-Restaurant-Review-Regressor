"""
Data Loading Module

Parses JSON files containing user, restaurant, and review data and constructs
Python objects (`User`, `Restaurant`, `Review`) used throughout the system.

This module initializes:
- USERS: list of all users
- REVIEWS: all review objects
- ALL_RESTAURANTS: dict of restaurant objects keyed by name
- CATEGORIES: set of all restaurant categories
"""

import collections
import os
from json import loads, dumps

from abstractions import User, Restaurant, Review

DATA_DIRECTORY = 'datafolder'

def load(fp, **kw):
    return [loads(obj, **kw) for obj in fp]

def dump(objs, fp, **kw):
    for obj in objs:
        fp.write(dumps(obj, **kw))
        fp.write('\n')

def load_data(user_dataset, review_dataset, restaurant_dataset):
    with open(os.path.join(DATA_DIRECTORY, user_dataset)) as f:
        user_data = load(f)
    with open(os.path.join(DATA_DIRECTORY, review_dataset)) as f:
        review_data = load(f)
    with open(os.path.join(DATA_DIRECTORY, restaurant_dataset)) as f:
        restaurant_data = load(f)

    # Load users
    userid_to_user = {}
    for user in user_data:
        name = user['name']
        _user_id = user['user_id']
        user = User(name, [])
        userid_to_user[_user_id] = user

    # Load restaurants
    busid_to_restaurant = {}
    for restaurant in restaurant_data:
        name = restaurant['name']
        location = [float(restaurant['latitude']), float(restaurant['longitude'])]
        categories = restaurant['categories']
        price = int(restaurant['price']) if restaurant['price'] is not None else None
        _business_id = restaurant['business_id']
        restaurant = Restaurant(name, location, categories, price, [])
        busid_to_restaurant[_business_id] = restaurant

    # Load reviews
    reviews = []
    busid_to_reviews = collections.defaultdict(list)
    userid_to_reviews = collections.defaultdict(list)

    for review in review_data:
        _user_id = review['user_id']
        _business_id = review['business_id']
        restaurant = busid_to_restaurant.get(_business_id)
        if restaurant is None:
            continue
        rating = float(review['stars'])
        review_obj = Review(restaurant.get_name(), rating)
        reviews.append(review_obj)
        busid_to_reviews[_business_id].append(review_obj)
        userid_to_reviews[_user_id].append(review_obj)

    # Finalize restaurants
    restaurants = {}
    for busid, restaurant in busid_to_restaurant.items():
        name = restaurant.get_name()
        location = restaurant.get_location()
        categories = restaurant.get_categories()
        price = restaurant.get_price()
        restaurant_reviews = busid_to_reviews[busid]
        restaurant = Restaurant(name, location, categories, price, restaurant_reviews)
        restaurants[name] = restaurant

    # Finalize users
    users = []
    for userid, user in userid_to_user.items():
        name = user.get_name()
        user_reviews = userid_to_reviews[userid]
        user = User(name, user_reviews)
        users.append(user)

    return users, reviews, list(restaurants.values())


# Load + Expose Global Variables

USERS, REVIEWS, ALL_RESTAURANTS = load_data('users.json', 'reviews.json', 'restaurants.json')

if len(ALL_RESTAURANTS) > 0:
    CATEGORIES = {c for r in ALL_RESTAURANTS for c in r.get_categories()}
else:
    CATEGORIES = set()
