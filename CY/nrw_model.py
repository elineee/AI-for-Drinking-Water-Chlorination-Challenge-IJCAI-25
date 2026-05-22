"""
Modeling of non-revenue water (NRW) -- taken from the Battle of Water Futures competition at WDSA CCWI 2026
"""
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray


# NRW bounds per category in cubic meter per km per day (mc/km/day)
_NRW_CLASSES_BOUNDS_LIST = [0.2, 2., 3., 4., 8., float('inf')]
_NRW_CLASSES_AGES_LIST = [0., 25., 43., 54., 60., float('inf')]
_NRW_CLASSES_BOUNDS = {
    i: (_NRW_CLASSES_BOUNDS_LIST[i], _NRW_CLASSES_BOUNDS_LIST[i+1])
    for i in range(0, len(_NRW_CLASSES_BOUNDS_LIST)-1)
}


class NRWClass(IntEnum):
    """
    Class models the properties of leaks and other losses in a municipality (i.e., its base demand to be added
    to the normal consumer demands).

    Parameters
    ----------
    nrw_class_id : `int`
        NRW category/class.
        Must be one of the following constants:

            - A = 0
            - B = 1
            - C = 2
            - D = 3
            - E = 4
    """
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4

    @classmethod
    def determine_class(cls, age: float) -> 'NRWClass':
        """Determines the NRW class based on the network age"""
        assert age >= 0., "Age parameter to determine the NRW class can't be negative"

        for size_class in cls:
            upper_bound = _NRW_CLASSES_AGES_LIST[size_class.value+1]

            if age < upper_bound:
                return size_class

        # Fallback for error handling (e.g., NaNs)
        raise ValueError(f"Age to determine the NRW class {age} is outside the defined bounds.")

    def demand_factor_bounds(self) -> tuple[float, float]:
        return _NRW_CLASSES_AGES_LIST[self.value], _NRW_CLASSES_AGES_LIST[self.value+1]

    def sample_demand(self, n_points: int, RNG: np.random.Generator) -> NDArray[np.float64]:
        """
        Returns a sampled nrw demand.

        Returns
        -------
        `float`
            NRW demand.
        """
        base_demand_range = _NRW_CLASSES_BOUNDS[self.value]

        if self != NRWClass.E:
            if self != NRWClass.A:
                return RNG.uniform(
                    low=base_demand_range[0],
                    high=base_demand_range[1],
                    size=n_points
                )
            else:
                # First nrw class: Shift probability mass towards upper bound -- we use a Beta distribution.
                return base_demand_range[1] * RNG.beta(5, 1, size=n_points)
        else:
            # Use the exponential distribution for sampling demands because there is no upper bound!
            # Ensures that the demand will always greater or equal then the specified
            # lower bound from 'nrw_base_demands'
            return base_demand_range[0] + base_demand_range[0] * .3 * RNG.exponential(size=n_points)
