from enum import IntEnum

from numpy import typing as npt
from pypower import idx_bus, idx_gen


class BusType(IntEnum):
    PQ = 1
    PV = 2
    REF = 3
    ISOLATED = 4


def get_bus_types(ppc: dict) -> npt.NDArray[int]:
    return ppc["bus"][:, idx_bus.BUS_TYPE].astype(int)


def get_gen_bus_indices(ppc: dict) -> npt.NDArray[int]:
    return ppc["gen"][:, idx_gen.GEN_BUS].astype(int)
