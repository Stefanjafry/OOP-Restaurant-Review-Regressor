from matplotlib import pyplot as plt

from data import ALL_RESTAURANTS
from recommended import LinearRegression
from utilities import map_and_filter
from abstractions import User, Review, Restaurant

test_mapfilter = True
test_restaurant = True
test_rate_all = True
test_predict = True


# map_and_filter tests
def test_mapfilterA():
    square = lambda x: x * x
    is_odd = lambda x: x % 2 == 1
    assert ([1, 9, 25] == map_and_filter([1, 2, 3, 4, 5], square, is_odd))
    assert (['o', 'd'] == map_and_filter(['hi', 'hello', 'hey', 'world'], lambda x: x[4], lambda x: len(x) > 4))


# Restaurant test
def test_restaurantA():
    soda_reviews = [Review('Soda', 4.5), Review('Soda', 4)]
    soda = Restaurant('Soda', [127.0, 0.1], ['Restaurants', 'Breakfast & Brunch'], 1, soda_reviews)
    assert ([4.5, 4] == soda.get_ratings())


# Restaurant test
def test_restaurantB():
    woz_reviews = [Review('Wozniak Lounge', 4), Review('Wozniak Lounge', 3), Review('Wozniak Lounge', 5)]
    woz = Restaurant('Wozniak Lounge', [127.0, 0.1], ['Restaurants', 'Pizza'], 1, woz_reviews)
    assert (woz.restaurant_num_ratings() == 3)  # should be an integer
    assert (type(woz.restaurant_num_ratings()) == int)  # should be an integer
    assert (woz.restaurant_mean_rating() == 4.0)  # should be a decimal
    assert (type(woz.restaurant_mean_rating()) == float)  # should be a decimal


# Regression tests
def test_predictA():
    user = User('John D.', [Review(ALL_RESTAURANTS[0].get_name(), 1), Review(ALL_RESTAURANTS[1].get_name(), 5),
                            Review(ALL_RESTAURANTS[2].get_name(), 2), Review(ALL_RESTAURANTS[3].get_name(), 2.5)])
    restaurant = Restaurant('New', [-10, 2], [], 2, [Review('New', 4)])
    lr = LinearRegression()
    lr.train(user, user.get_reviewed_restaurants(ALL_RESTAURANTS))

    assert (round(lr.predict(restaurant), 2) == 6.97)
    assert (round(lr.r_squared, 2) == 0.55)


def test_predictB():
    user = User('John D.', [Review(ALL_RESTAURANTS[-1].get_name(), 1), Review(ALL_RESTAURANTS[-2].get_name(), 5),
                            Review(ALL_RESTAURANTS[-3].get_name(), 2), Review(ALL_RESTAURANTS[-4].get_name(), 2.5)])
    restaurant = Restaurant('New', [-10, 2], [], 2, [Review('New', 4)])

    lr = LinearRegression()
    lr.train(user, user.get_reviewed_restaurants(ALL_RESTAURANTS))

    assert (2.41 == round(lr.predict(restaurant), 2))
    assert (0.02 == round(lr.r_squared, 2))


def test_predictC():
    user = User('John D.', [Review(ALL_RESTAURANTS[4].get_name(), 1), Review(ALL_RESTAURANTS[5].get_name(), 5),
                            Review(ALL_RESTAURANTS[6].get_name(), 2), Review(ALL_RESTAURANTS[7].get_name(), 2.5)])
    restaurant = Restaurant('New', [-10, 2], [], 2, [Review('New', 4)])

    lr = LinearRegression()
    lr.train(user, user.get_reviewed_restaurants(ALL_RESTAURANTS))

    assert (4.72 == round(lr.predict(restaurant), 2))
    assert (0.99 == round(lr.r_squared, 2))


# Rate all tests
def test_rateallA():
    user = User('Mr. Mean Rating Minus One',
                [Review('Cafe 3', 1.0), Review('Jasmine Thai', 2.0), Review('Fondue Fred', 2.0)])
    to_rate = ALL_RESTAURANTS[:6]
    c = LinearRegression()

    # ADDED LINES BELOW
    reviewed = user.get_reviewed_restaurants(ALL_RESTAURANTS)
    c.train(user, reviewed)
    # END OF ADDED LINES BELOW

    predictions = [round(n, 1) for n in list(c.rate_all(user, to_rate).values())]
    correct = [1.0, 2.0, 2.0, 1.4, 1.6, 3.3]

    mean_ratings = [r.restaurant_mean_rating() for r in to_rate]
    plt.plot(mean_ratings, predictions, 'bo')
    plt.plot(c.xs, c.ys, 'ro')
    plt.xlabel('Mean Rating for Restaurant')
    plt.ylabel('User Rating')
    plt.legend(['Predicted Rating', 'Actual Rating'])
    plt.show()

    for c in range(len(correct)):
        assert (correct[c] == predictions[c])


def test_rateallB():
    user = User('Mr. Nice Rating Plus One',
                [Review('Cafe 3', 3.0), Review('Jasmine Thai', 4.0), Review('Fondue Fred', 4.0)])
    to_rate = ALL_RESTAURANTS[:6]
    c = LinearRegression()

    # ADDED LINES BELOW
    reviewed = user.get_reviewed_restaurants(ALL_RESTAURANTS)
    c.train(user, reviewed)
    # END OF ADDED LINES BELOW

    predictions = [round(n, 1) for n in list(c.rate_all(user, to_rate).values())]
    correct = [3.0, 4.0, 4.0, 3.4, 3.6, 5.3]
    mean_ratings = [r.restaurant_mean_rating() for r in to_rate]
    plt.plot(mean_ratings, predictions, 'bo')
    plt.plot(c.xs, c.ys, 'ro')
    plt.xlabel('Mean Rating for Restaurant')
    plt.ylabel('User Rating')
    plt.legend(['Predicted Rating', 'Actual Rating'])
    plt.show()

    for c in range(len(correct)):
        assert (correct[c] == predictions[c])


if __name__ == '__main__':

    if test_mapfilter:
        test_mapfilterA()
    if test_restaurant:
        test_restaurantA()
        test_restaurantB()
    if test_predict:
        test_predictA()
        test_predictB()
        test_predictC()
    if test_rate_all:
        test_rateallA()
        test_rateallB()

