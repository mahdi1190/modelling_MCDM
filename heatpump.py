import pandas as pd
import math
import pyomo


def lorenz(Th, Tc):
    """Find the COP_lorenz of a heat pump based on hot and cold stream temperatures in Kelvin"""

    COP_lorenz = ((Th)/(Th-Tc))

    return COP_lorenz
