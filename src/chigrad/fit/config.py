from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class FitConfig:
    # parameters 
    param_start:      dict[str, float]
    param_limit:      Optional[dict[str, tuple]] = None
    priors:           Optional[dict[str, tuple[float, float]]] = None

    # minimiser settings
    tolerance:        float = 0.1
    strategy:         int   = 1
    iterations:       int   = 10000
    enable_simplex:   bool  = True

    # run control
    execute_resample: bool = False
    raise_failure:    bool = True
    resample_type:    Literal["bootstrap", "jackknife"] = "jackknife"
    silent_output:    bool = False

    def __post_init__(self):
        if self.strategy not in (0, 1, 2):
            raise ValueError(f"Minuit requires strategy 0, 1, or 2. Received {self.strategy}.")
        if not self.param_start:
            raise ValueError("No start parameters received.")
