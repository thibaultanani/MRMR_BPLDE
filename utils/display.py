# Pretty print functions

from datetime import timedelta
from typing import Union


def pretty_print(print_out: str, ID: int, method: str, mean: float, bestScore: float, numCols: int,
                 time_exe: timedelta, time_total: timedelta, entropy: list[float], iteration: int, p: Union[int, str]):

    display = "PID: {:3d} [{:3}]    G: {:5d}    mean: {:2.4%}    best: {:2.4%}    features: {:6d}    G time: {}" \
              "    total time: {}    entropy : {:2.3%}    p: {}".format(ID, method, iteration, mean, bestScore, numCols,
                                                                        time_exe, time_total, sum(entropy)/len(entropy),
                                                                        p)
    print_out = print_out + display
    print(display)
    return print_out
