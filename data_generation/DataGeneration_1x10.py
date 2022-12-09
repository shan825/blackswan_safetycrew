"""
Generate initial data for xTemp, direction, and target variables used in NN model

@author: Doug Cady

Created on: Aug 30, 2022

"""

import numpy as np
import pandas as pd
from typing import NamedTuple


NUM_TEMP_POSITIONS = 10
DIRECTIONS = ['RIGHT', 'LEFT']
NUM_MOVEMENTS = 11

RIGHT = 0
LEFT = 1

OUTPUT_COLUMNS = [
    'xTemp',
    'direction',
    'target'
]

DIRECTION_DELTA_MAP = {
    'RIGHT': 1,
    'LEFT': -1,
}


def left_edge(xTemp: int) -> bool:
    return xTemp == 0


def right_edge(xTemp: int) -> bool:
    return xTemp == NUM_TEMP_POSITIONS - 1


def calc_target(xTemp: int, direction: int) -> int:
    """Calculate target position and return as an int."""
    for i in range(NUM_MOVEMENTS):
        if left_edge(xTemp=xTemp) & (direction == 'LEFT'):
            xTemp = xTemp + 1
            direction = 'RIGHT'

        elif right_edge(xTemp=xTemp) & (direction == 'RIGHT'):
            xTemp = xTemp - 1
            direction = 'LEFT'

        # In middle of grid, not on corner or edge
        else:
            xTemp = xTemp + DIRECTION_DELTA_MAP[direction]

    return xTemp


def main() -> int:
    out_list = []

    for direction in DIRECTIONS:
        for xTemp in range(NUM_TEMP_POSITIONS):
            target = calc_target(xTemp=xTemp, direction=direction)
            row_list = [xTemp, direction, target]
            out_list.append(row_list)

    data = pd.DataFrame(out_list, columns = OUTPUT_COLUMNS)
    data.to_csv("../data/testOneByTenModelData.csv", index=False)


if __name__ == '__main__':
    main()
