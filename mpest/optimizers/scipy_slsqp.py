"""Module which contains Sequential Least Squares Programming (SLSQP) optimizer"""

from typing import Callable

from scipy.optimize import minimize

from mpest.optimizers.abstract_optimizer import AOptimizer
from mpest.types import Params


class ScipySLSQP(AOptimizer):
    """Class which represents SciPy Sequential Least Squares Programming (SLSQP) optimizer"""

    @property
    def name(self):
        return "ScipySLSQP"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="SLSQP").x
