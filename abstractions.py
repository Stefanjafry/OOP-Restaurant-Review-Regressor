"""Data Abstractions"""

class Review:
    def __init__(self, restaurant_name: str, rating: float):
        self._restaurant_name = restaurant_name
        self._rating = rating

    def get_restaurant_name(self) -> str:
        return self._restaurant_name

    def get_rating(self) -> float:
        return self._rating


class Restaurant:
    def __init__(self, name: str, location: list[float], categories: list[str], price: int,
                 reviews: list[Review]) -> None:
        self._name = name
        self._location = location
        self._categories = categories
        self._price = price
        self._reviews = reviews

    def get_name(self) -> str:
        return self._name

    def get_location(self) -> list[float]:
        return self._location

    def get_categories(self) -> list[str]:
        return self._categories

    def get_price(self) -> int:
        return self._price

    def get_reviews(self) -> list[Review]:
        return self._reviews

    def get_review_count(self) -> int:
        return len(self._reviews)

    def get_ratings(self) -> list[float]:
        return [r.get_rating() for r in self._reviews]

    def restaurant_num_ratings(self) -> int:
        return len(self._reviews)

    def restaurant_mean_rating(self) -> float:
        ratings = [r.get_rating() for r in self._reviews]
        return sum(ratings) / len(ratings) if ratings else 0.0


class User:
    def __init__(self, name: str, reviews: list[Review]) -> None:
        self._name = name
        self._reviews = reviews

    def get_name(self) -> str:
        return self._name

    def get_reviews(self) -> list[Review]:
        return self._reviews

    def get_review_count(self) -> int:
        return len(self._reviews)

    def get_reviewed_restaurants(self, restaurants: list[Restaurant]) -> list[Restaurant]:
        return [r for r in restaurants if self.user_rating(r.get_name()) is not None]

    def user_rating(self, restaurant_name: str) -> float | None:
        for r in self._reviews:
            if r.get_restaurant_name().strip().lower() == restaurant_name.strip().lower():
                return r.get_rating()
        return None

    def get_average_stars(self) -> float:
        ratings = [r.get_rating() for r in self._reviews]
        return sum(ratings) / len(ratings) if ratings else 0.0

