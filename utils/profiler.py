# This module will be used when necessary to profile algorithms I implement to optimise their performance.


import numpy as np
import cProfile
import pstats
from io import StringIO

from utils.params import p, B
from classical.min_distance import min_distance 



if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    min_distance(B.T)

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    print(s.getvalue())