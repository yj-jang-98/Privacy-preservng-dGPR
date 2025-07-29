from secrets import randbelow
import numpy as np
import torch

def get_rand(min: int, max: int, device) -> int:
    """
    Generates a random integer in `[min, max)`.

    Parameters
    ----------
        min : The minimum value of the range (inclusive).
        max : The maximum value of the range (exclusive).

    Returns
    -------
        Generated random integer in `[min, max)`.
    """

    return torch.randint(min, max, (1,), dtype=torch.int64, device=device).item()

def mod(x, q):
    """
    Takes the modulo operation. Z_q = [-q/2,q/2)

    Parameters
    ----------
        x : Input value
        q : Modulus

    Returns
    -------
        x mod q
    """
    r = x % q
    r = torch.where(r >= q // 2, r - q, r)  # Element-wise conditional mod operation
    return r

def mod_vec(x, q):
    """
    Takes the modulo operation for each element of a vector

    Parameters
    ----------
        x : nparray of int
        q : modulus

    Returns
    -------
        x mod q
    """
    r = x % q
    r = torch.where(r >= q // 2, r - q, r)  # Element-wise conditional mod operation
    return r