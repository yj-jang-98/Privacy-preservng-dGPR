from secrets import randbelow
import numpy as np


def get_rand(min: int, max: int) -> int:
    """
    Generates a random integer in `[min, max)`.

    Parameters
    ----------
    min : int
        The minimum value of the range (inclusive).
    max : int
        The maximum value of the range (exclusive).

    Returns
    -------
    int
        Generated random integer in `[min, max)`.
    """

    return randbelow(max - min) + min

def mod(x, q):
    """
    Takes the modulo operation. Z_q = [-q/2,q/2)

    Parameters
    ----------
    x : int
        Input value
    q : int
        Modulus

    Returns
    -------
    int
        x mod q
    """
    r = x % q
    if r >= q // 2:
        r -= q
    return r

def mod_vec(x, q):
    """
    Takes the modulo operation for each element of a vector

    Parameters
    ----------
    x : nparray of int
    q : int
        modulus

    Returns
    -------
    nparray of int
        x mod q
    """
    r = np.mod(x, q)
    r = np.where(r >= q // 2, r - q, r)
    return r
