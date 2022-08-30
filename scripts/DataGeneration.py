"""
Generate initial data for xTemp, yVolume, direction, and target variables used in NN model

@author: Doug Cady

Created on: Aug 28, 2022

"""

import numpy as np
import pandas as pd
from typing import NamedTuple


NUM_TEMP_POSITIONS = 10
NUM_VOLUME_POSITIONS = 10
NUM_DIRECTIONS = 8
NUM_MOVEMENTS = 19

UP = 0
DOWN = 4
RIGHT = 2
LEFT = 6

UP_DOWN = set([0, 4])
RIGHT_LEFT = set([2, 6])
DIAGONAL_UP = set([1, 7])
DIAGONAL_RIGHT = set([1, 3])

OUTPUT_COLUMNS = [
	'xTemp',
	'yVolume',
	'direction',
	'target'
]


class DeltaPosition(NamedTuple):
	x: int
	y: int
	name: str

DIRECTION_DELTA_MAP = {
	0: DeltaPosition(x=0,  y=-1, name='up'),
	4: DeltaPosition(x=0,  y=1,  name='down'),
	
	2: DeltaPosition(x=1,  y=0, name='right'),
	6: DeltaPosition(x=-1, y=0, name='left'),

	1: DeltaPosition(x=1,  y=-1,  name='up-right'),
	3: DeltaPosition(x=1,  y=1,   name='down-right'),
	5: DeltaPosition(x=-1, y=1,   name='down-left'),
	7: DeltaPosition(x=-1,  y=-1, name='up-left'),
}


def left_edge(xTemp: int) -> bool:
	return xTemp == 0


def right_edge(xTemp: int) -> bool:
	return xTemp == NUM_TEMP_POSITIONS - 1


def top_edge(yVolume: int) -> bool:
	return yVolume == 0


def bot_edge(yVolume: int) -> bool:
	return yVolume == NUM_VOLUME_POSITIONS - 1


def on_edge(xTemp: int, yVolume: int) -> bool:
	return left_edge(xTemp) | right_edge(xTemp) | top_edge(yVolume) | bot_edge(yVolume)


def top_left_corner(xTemp: int, yVolume: int) -> bool:
	return top_edge(yVolume) & left_edge(xTemp)


def top_right_corner(xTemp: int, yVolume: int) -> bool:
	return top_edge(yVolume) & right_edge(xTemp)


def bot_right_corner(xTemp: int, yVolume: int) -> bool:
	return bot_edge(yVolume) & right_edge(xTemp)


def bot_left_corner(xTemp: int, yVolume: int) -> bool:
	return bot_edge(yVolume) & left_edge(xTemp)


def calc_target(xTemp: int, yVolume: int, direction: int) -> int:
	"""Calculate target position and return as an int."""
	for i in range(NUM_MOVEMENTS):
		if direction in UP_DOWN:
			if top_edge(yVolume=yVolume) & (direction == UP):
				yVolume = yVolume + 1
				direction = DOWN

			elif bot_edge(yVolume=yVolume) & (direction == DOWN):
				yVolume = yVolume - 1
				direction = UP

			else:
				yVolume += DIRECTION_DELTA_MAP[direction].y

		elif direction in RIGHT_LEFT:
			if left_edge(xTemp=xTemp) & (direction == LEFT):
				xTemp = xTemp + 1
				direction = RIGHT

			elif right_edge(xTemp=xTemp) & (direction == RIGHT):
				xTemp = xTemp - 1
				direction = LEFT

			else:
				xTemp += DIRECTION_DELTA_MAP[direction].x
			
		# Diagonal movement
		else:
			if top_left_corner(xTemp=xTemp, yVolume=yVolume):
				xTemp = xTemp + 1
				yVolume = yVolume + 1
				direction = 3

			elif top_right_corner(xTemp=xTemp, yVolume=yVolume):
				xTemp = xTemp - 1
				yVolume = yVolume + 1
				direction = 5

			elif bot_right_corner(xTemp=xTemp, yVolume=yVolume):
				xTemp = xTemp - 1
				yVolume = yVolume - 1
				direction = 7

			elif bot_left_corner(xTemp=xTemp, yVolume=yVolume):
				xTemp = xTemp + 1
				yVolume = yVolume - 1
				direction = 1

			# Not in a corner, on an edge
			elif top_edge(yVolume=yVolume):
				if direction in DIAGONAL_RIGHT:
					xTemp = xTemp + 1
					yVolume = yVolume + 1
					direction = 3
				else:
					xTemp = xTemp - 1
					yVolume = yVolume + 1
					direction = 5

			elif bot_edge(yVolume=yVolume):
				if direction in DIAGONAL_RIGHT:
					xTemp = xTemp + 1
					yVolume = yVolume - 1
					direction = 1
				else:
					xTemp = xTemp - 1
					yVolume = yVolume - 1
					direction = 7

			elif left_edge(xTemp=xTemp):
				if direction in DIAGONAL_UP:
					xTemp = xTemp + 1
					yVolume = yVolume - 1
					direction = 1
				else:
					xTemp = xTemp + 1
					yVolume = yVolume + 1
					direction = 3

			elif right_edge(xTemp=xTemp):
				if direction in DIAGONAL_UP:
					xTemp = xTemp - 1
					yVolume = yVolume - 1
					direction = 7
				else:
					xTemp = xTemp - 1
					yVolume = yVolume + 1
					direction = 5

			# In middle of grid, not on corner or edge
			else:
				xTemp = xTemp + DIRECTION_DELTA_MAP[direction].x
				yVolume = yVolume + DIRECTION_DELTA_MAP[direction].y

	return (10 * yVolume) + xTemp


def main() -> int:
	out_list = []

	for xTemp in range(NUM_TEMP_POSITIONS):
		for yVolume in range(NUM_VOLUME_POSITIONS):
			for direction in range(NUM_DIRECTIONS):
				target = calc_target(xTemp=xTemp, yVolume=yVolume, direction=direction)
				row_list = [xTemp, yVolume, direction, target]
				out_list.append(row_list)

	data = pd.DataFrame(out_list, columns = OUTPUT_COLUMNS)
	data.to_csv("../data/tenByTenModelData.csv", index=False)


if __name__ == '__main__':
	main()
