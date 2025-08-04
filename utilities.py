"""
utilities.py
============

Utility functions for manipulating and analyzing restaurant data.

This module includes:
- `map_and_filter`: generic mapping and filtering on lists
- `sorted_restaurants`: sorting restaurants by mean rating
- `average_restaurant`: computing average of average ratings

Functions are demonstrated using the `Restaurant` class from `abstractions.py`,
but the implementations (except sorting and averaging) are generic.

Used extensively in `main.py` to support evaluation and display logic.

Dependencies:
-------------
- abstractions.Restaurant
- data.ALL_RESTAURANTS
"""

from __future__ import annotations
from typing import Any, Callable
from abstractions import Restaurant
from data import ALL_RESTAURANTS


def map_and_filter(s: list[Any], map_fn: Callable, filter_fn: Callable) -> list[Any]:
    """
    Apply `map_fn` to elements in `s` that satisfy `filter_fn`.

    >>> getname = lambda x: x.get_name()
    >>> is_good = lambda x: x.restaurant_mean_rating() > 4
    >>> map_and_filter(ALL_RESTAURANTS[:10], getname, is_good)
    ['Happy Valley']
    >>> is_bad = lambda x: x.restaurant_mean_rating() < 3
    >>> map_and_filter(ALL_RESTAURANTS[:10], getname, is_bad)
    ['Cafe 3', 'Jasmine Thai', 'Fondue Fred', 'Peppermint Grill', 'Viengvilay Thai Cuisine']
    """
    return list(map(map_fn, filter(filter_fn, s)))


def sorted_restaurants(restaurants: list[Restaurant]) -> list[tuple[str, float]]:
    """
    Return a list of (restaurant_name, mean_rating) tuples, sorted by rating (ascending).

    >>> r = sorted_restaurants(ALL_RESTAURANTS)
    >>> print("lowest rating: " + str(r[0]))
    lowest rating: ('Subway', 2.0)
    >>> print("highest rating: " + str(r[-1]))
    highest rating: ("Foley's Deli", 5.0)
    """
    return sorted([(r.get_name(), r.restaurant_mean_rating()) for r in restaurants],
                  key=lambda x: x[1])


def average_restaurant(restaurants: list[Restaurant]) -> float:
    """
    Return the average of all restaurant mean ratings.

    >>> r = round(average_restaurant(ALL_RESTAURANTS), 1)
    >>> print("average rating: " + str(r))
    average rating: 3.4
    """
    if not restaurants:
        return 0.0
    return sum(r.restaurant_mean_rating() for r in restaurants) / len(restaurants)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
