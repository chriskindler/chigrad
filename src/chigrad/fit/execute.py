# File: chigrad/src/chigrad/execute.py
import iminuit
import numpy as np
from typing import Callable, Literal, Optional

from chigrad.fit.minimise import ConvergenceError
from chigrad.statistics.jackknife import jackknife_variance, jackknife_covariance 


