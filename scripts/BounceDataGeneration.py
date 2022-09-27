"""
Generate initial data for xTemp, yVolume, direction, and target variables used in NN model

This script also keeps track of:
1) a particle bouncing on its move
2) how many times a particle bounces

@author: Doug Cady

Created on: Sep 27, 2022

"""

import numpy as np
import pandas as pd
from typing import NamedTuple, Tuple


NUM_TEMP_POSITIONS = 10
NUM_VOLUME_POSITIONS = 10
NUM_DIRECTIONS = 8
NUM_MOVEMENTS = 19

INPUT_NORMALIZATION_CONSTANT = 10

NORTH = 0
NORTHEAST = 1
EAST = 2
SOUTHEAST = 3
SOUTH = 4
SOUTHWEST = 5
WEST = 6
NORTHWEST = 7

NORTH_SOUTH = set([NORTH, SOUTH])
EAST_WEST = set([EAST, WEST])
DIAGONAL_NORTH = set([NORTHEAST, NORTHWEST])
DIAGONAL_EAST = set([NORTHEAST, SOUTHEAST])

ALL_NORTH = set([NORTH, NORTHEAST, NORTHWEST])
ALL_SOUTH = set([SOUTH,  SOUTHEAST, SOUTHWEST])
ALL_EAST = set([EAST, NORTHEAST, SOUTHEAST])
ALL_WEST = set([WEST, NORTHWEST, SOUTHWEST])
    
OUTPUT_COLUMNS = [
    'xTemp',
    'yVol',
    'direction',
    'target',
    'bounceFirstMove',
    'numBounces',
    'troublePair'
]


class DeltaPosition(NamedTuple):
    x: int
    y: int
    name: str

DIRECTION_DELTA_MAP = {
    0: DeltaPosition(x=0,  y=-1, name='north'),
    4: DeltaPosition(x=0,  y=1,  name='south'),
    
    2: DeltaPosition(x=1,  y=0, name='east'),
    6: DeltaPosition(x=-1, y=0, name='west'),

    1: DeltaPosition(x=1,  y=-1,  name='north-east'),
    3: DeltaPosition(x=1,  y=1,   name='south-east'),
    5: DeltaPosition(x=-1, y=1,   name='south-west'),
    7: DeltaPosition(x=-1, y=-1,  name='north-west'),
}


def left_edge(xTemp: int) -> bool:
    return xTemp == 0


def right_edge(xTemp: int) -> bool:
    return xTemp == NUM_TEMP_POSITIONS - 1


def top_edge(yVol: int) -> bool:
    return yVol == 0


def bot_edge(yVol: int) -> bool:
    return yVol == NUM_VOLUME_POSITIONS - 1


def on_edge(xTemp: int, yVol: int) -> bool:
    return left_edge(xTemp) | right_edge(xTemp) | top_edge(yVol) | bot_edge(yVol)


def top_left_corner(xTemp: int, yVol: int) -> bool:
    return top_edge(yVol) & left_edge(xTemp)


def top_right_corner(xTemp: int, yVol: int) -> bool:
    return top_edge(yVol) & right_edge(xTemp)


def bot_right_corner(xTemp: int, yVol: int) -> bool:
    return bot_edge(yVol) & right_edge(xTemp)


def bot_left_corner(xTemp: int, yVol: int) -> bool:
    return bot_edge(yVol) & left_edge(xTemp)


def is_first_move_bounce(xTemp: int, yVol: int, direction: int) -> bool:
    if top_edge(yVol=yVol) & (direction in ALL_NORTH):
        return True
    
    elif bot_edge(yVol=yVol) & (direction in ALL_SOUTH):
        return True
    
    elif left_edge(xTemp=xTemp) & (direction in ALL_WEST):
        return True
    
    elif right_edge(xTemp=xTemp) & (direction in ALL_EAST):
        return True

    else:
        return False


def calc_target(xTemp: int, yVol: int, direction: int) -> Tuple[int, bool, int]:
    """Calculate target position, first move bounce, and number of bounces."""
    first_move_bounce = is_first_move_bounce(xTemp=xTemp, yVol=yVol, direction=direction)
    num_bounces = 1 if first_move_bounce else 0

    for i in range(NUM_MOVEMENTS):
        if direction in NORTH_SOUTH:
            if top_edge(yVol=yVol) & (direction == NORTH):
                yVol = yVol + 1
                direction = SOUTH
                num_bounces += 1

            elif bot_edge(yVol=yVol) & (direction == SOUTH):
                yVol = yVol - 1
                direction = NORTH
                num_bounces += 1

            else:
                yVol += DIRECTION_DELTA_MAP[direction].y

        elif direction in EAST_WEST:
            if left_edge(xTemp=xTemp) & (direction == WEST):
                xTemp = xTemp + 1
                direction = EAST
                num_bounces += 1

            elif right_edge(xTemp=xTemp) & (direction == EAST):
                xTemp = xTemp - 1
                direction = WEST
                num_bounces += 1

            else:
                xTemp += DIRECTION_DELTA_MAP[direction].x
            
        # Diagonal movement
        else:
            if top_left_corner(xTemp=xTemp, yVol=yVol):
                xTemp = xTemp + 1
                yVol = yVol + 1
                direction = SOUTHEAST
                num_bounces += 1

            elif top_right_corner(xTemp=xTemp, yVol=yVol):
                xTemp = xTemp - 1
                yVol = yVol + 1
                direction = SOUTHWEST
                num_bounces += 1

            elif bot_right_corner(xTemp=xTemp, yVol=yVol):
                xTemp = xTemp - 1
                yVol = yVol - 1
                direction = NORTHWEST
                num_bounces += 1

            elif bot_left_corner(xTemp=xTemp, yVol=yVol):
                xTemp = xTemp + 1
                yVol = yVol - 1
                direction = NORTHEAST
                num_bounces += 1

            # Not in a corner, on an edge
            elif top_edge(yVol=yVol):
                if direction in DIAGONAL_EAST:
                    xTemp = xTemp + 1
                    yVol = yVol + 1
                    direction = SOUTHEAST
                    num_bounces += 1
                else:
                    xTemp = xTemp - 1
                    yVol = yVol + 1
                    direction = SOUTHWEST
                    num_bounces += 1

            elif bot_edge(yVol=yVol):
                if direction in DIAGONAL_EAST:
                    xTemp = xTemp + 1
                    yVol = yVol - 1
                    direction = NORTHEAST
                    num_bounces += 1
                else:
                    xTemp = xTemp - 1
                    yVol = yVol - 1
                    direction = NORTHWEST
                    num_bounces += 1

            elif left_edge(xTemp=xTemp):
                if direction in DIAGONAL_NORTH:
                    xTemp = xTemp + 1
                    yVol = yVol - 1
                    direction = NORTHEAST
                    num_bounces += 1
                else:
                    xTemp = xTemp + 1
                    yVol = yVol + 1
                    direction = SOUTHEAST
                    num_bounces += 1

            elif right_edge(xTemp=xTemp):
                if direction in DIAGONAL_NORTH:
                    xTemp = xTemp - 1
                    yVol = yVol - 1
                    direction = NORTHWEST
                    num_bounces += 1
                else:
                    xTemp = xTemp - 1
                    yVol = yVol + 1
                    direction = SOUTHWEST
                    num_bounces += 1

            # In middle of grid, not on corner or edge
            else:
                xTemp = xTemp + DIRECTION_DELTA_MAP[direction].x
                yVol = yVol + DIRECTION_DELTA_MAP[direction].y

    return (10 * yVol) + xTemp, first_move_bounce, num_bounces


def main() -> int:
    out_list = []

    for xTemp in range(NUM_TEMP_POSITIONS):
        for yVol in range(NUM_VOLUME_POSITIONS):
            for direction in range(NUM_DIRECTIONS):
                target, first_move_bounce, num_bounces = calc_target(xTemp=xTemp, yVol=yVol, 
                                                                     direction=direction)
                trouble_pair = True if target == 99 else False

                row_list = [xTemp / INPUT_NORMALIZATION_CONSTANT, 
                            yVol / INPUT_NORMALIZATION_CONSTANT, 
                            direction / INPUT_NORMALIZATION_CONSTANT,
                            target,
                            first_move_bounce, 
                            num_bounces,
                            trouble_pair]

                out_list.append(row_list)

    data = pd.DataFrame(out_list, columns = OUTPUT_COLUMNS)
    data.to_csv("../data/bounce_tenByTenModelData.csv", index=False)


if __name__ == '__main__':
    main()