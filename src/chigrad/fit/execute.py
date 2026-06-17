# File: chigrad/src/chigrad/execute.py
import iminuit
import numpy as np
from typing import Callable, Literal, Optional

from chigrad.log import message
from chigrad.fit.minimise import ConvergenceError, minimise
from chigrad.statistics.jackknife import jackknife_variance, jackknife_covariance 
