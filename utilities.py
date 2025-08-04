"""Utilities"""
from __future__ import annotations

from typing import Any, Callable
from abstractions import Restaurant
from data import ALL_RESTAURANTS


def map_and_filter(s: list[Any], map_fn: Callable, filter_fn: Callable) -> list[Restaurant]:
    """Accept a list. Return a new list containing
    the result of calling `map_fn` on each item in the sequence `s`
    for which `filter_fn` returns a true value.
    We will demonstrate this in the doctest on
    lists of Restaurants, but this method
    should work with lists of other types.
    >>> getname = lambda x: x.get_name()
    >>> is_good = lambda x: x.restaurant_mean_rating() > 4
    >>> map_and_filter(ALL_RESTAURANTS[:10], getname, is_good)
    ['Happy Valley']
    >>> is_bad = lambda x: x.restaurant_mean_rating() < 3
    >>> map_and_filter(ALL_RESTAURANTS[:10], getname, is_bad)
    ['Cafe 3', 'Jasmine Thai', 'Fondue Fred', 'Peppermint Grill', 'Viengvilay Thai Cuisine']
    """
    return list(map(map_fn, filter(filter_fn,s)))


def sorted_restaurants(restaurants: list[Restaurant]) -> list[tuple[str, float]]:
    """Accept a list of restaurants.  Return a list of tuples of restaurant names
    and the corrsesponding average rating of the restaurants.  The
    list that is returned should be sorted by average rating, from lowest
    to highest.
    >>> r = sorted_restaurants(ALL_RESTAURANTS)
    >>> print("lowest rating: " + str(r[0]))
    lowest rating: ('Subway', 2.0)
    >>> print("highest rating: " + str(r[-1]))
    highest rating: ("Foley's Deli", 5.0)
    """
    p


def average_restaurant(restaurants: list[Restaurant]) -> float:
    """Return the average of all restaurant averages in the
    input list of restaurants.
    >>> r = round(average_restaurant(ALL_RESTAURANTS),1)
    >>> print("average rating: " + str(r))
    average rating: 3.4
    """
    pass  # replace this with your implementation


if __name__ == "__main__":
    import doctest
    doctest.testmod()

