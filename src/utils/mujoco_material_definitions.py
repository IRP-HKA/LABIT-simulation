from dataclasses import dataclass, field
from typing import List


@dataclass
class Material:
    density: int = 1000     # kg/m3

    # [timeconst,dampratio] or [−stiffness,−damping]
    solref: List[float] = field(default_factory=lambda: [0.02, 1.0])

    #  [d0, dwidth​, width, midpoint, power]
    solimp: List[float] = field(default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2])

    # [sliding, torsional, rolling]
    friction: List[float] = field(default_factory=lambda: [1, 0.005, 0.0001])

    # used for elasticity simulation only
    young: int = 5e4       # Pa
    poisson: float = 0.2   # dimensionless


MATERIALS = {
    "default": Material(),
    "steel": Material(
        density=7850,
        solref=[0.001, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        friction=[1.5, 0.005, 0.0001], # 0.8
        young=2e11,
        poisson=0.28,
    ),
    "plastic": Material(
        density=1190,
        solref=[0.001, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        friction=[1.5, 0.005, 0.0001],
        young=2.5e9,
        poisson=0.4,
    ),
    "feather": Material(
        density=.01,
        solref=[0.001, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2],
        friction=[1.0, 0.005, 0.0001],
        young=2.5e9,
        poisson=0.4,
    ),
}
