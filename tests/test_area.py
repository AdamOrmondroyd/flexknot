import numpy as np
from flexknot import FlexKnot, AdaptiveKnot
from flexknot.helper_functions import get_theta_n, intersection


def test_area():
    x_min = 1
    x_max = 2

    fk = FlexKnot(x_min, x_max)

    # rectangle
    theta0 = np.array([0])
    theta1 = np.array([1])
    assert fk.area(theta0, theta1) == 1

    # crossover
    theta0 = np.array([0, 1])
    theta1 = np.array([1, 0])
    assert fk.area(theta0, theta1) == 0.5
    assert fk.area(theta0, theta1) == fk.area(theta0, theta1)

    # Check same result both ways around
    fk = FlexKnot(0, 5)
    theta0 = np.array([0, 1, 1, 2, -2, 3, 3, 4, -4, 5])
    theta1 = np.array([0, 1, -1, 2, 2, 3, -3, 4, 4, -5])
    assert fk.area(theta0, theta1) == fk.area(theta1, theta0)

    assert np.all(np.isclose(fk.intersections(theta0, theta1),
                             fk.intersections(theta1, theta0)))
    assert np.isclose(fk.area(theta0, theta1),
                      fk.area(theta1, theta0))

    # compare old and new with this example
    assert np.all(np.isclose(fk.intersections(theta0, theta1),
                             [1+1/3, 2+2/5, 3+3/7, 4+4/9]))
    assert np.isclose(fk.area(theta0, theta1),
                      2*(1/2+5/6+13/10+25/14+41/18))
    assert np.isclose(fk.area(theta0, np.array([0, 0])),
                      1/2+5/6+13/10+25/14+41/18)
    # assert np.isclose(fk.area(theta0, theta1), fk.area(theta0, theta1))

    # Test adaptive
    ak = AdaptiveKnot(0, 5)
    theta0 = np.array([3, 0, 1, 1, 2, -2, 3, 3, 4, -4, 5])
    theta1 = np.array([3, 0, 1, -1, 2, 2, 3, -3, 4, 4, -5])
    assert ak.area(theta0, theta1) == ak.area(theta1, theta0)
    assert ak.area(theta0, theta1) == fk.area(get_theta_n(theta0),
                                              get_theta_n(theta1))


def test_intersection():
    p0 = np.array([0, 0])
    p1 = np.array([1, 1])
    q0 = np.array([0, 1])
    q1 = np.array([1, 0])

    assert np.all(intersection(p0, p1, q0, q1) == np.array([0.5, 0.5]))


def test_intersections():
    fk = FlexKnot(0, 1)
    ints = fk.intersections(
            np.array([0, 1]),
            np.array([1, 0]),
            )
    assert np.all(ints == np.array([[0.5, 0.5]]))
    ints = fk.intersections(
            np.array([0, 0.5, 1, 0]),
            np.array([1, 0.5, 0, 1]),
            )
    assert np.all(np.isclose(ints, np.array([0.25, 0.75])))
    # print(fk.area(np.array([0, 0.5, 1, 0]), np.array([1, 0.5, 0, 1])))

    # Check same result both ways around
    theta0 = np.sort(np.random.rand(10))
    theta1 = np.sort(np.random.rand(10))
    assert np.all(np.isclose(fk.intersections(theta0, theta1),
                             fk.intersections(theta1, theta0)))

    # Should be equal at intersections
    assert np.all(np.isclose(fk(fk.intersections(theta0, theta1), theta0),
                             fk(fk.intersections(theta0, theta1), theta1)))
