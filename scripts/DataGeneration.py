"""
Generate initial data for xPos, yPos, dir, and 99pos variables used in NN model

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
	'yVol',
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


def calc_target(xTemp: int, yVol: int, direction: int) -> int:
	"""Calculate target position and return as an int."""
	for i in range(NUM_MOVEMENTS):
		if direction in UP_DOWN:
			if top_edge(yVol=yVol) & (direction == UP):
				yVol = yVol + 1
				direction = DOWN

			elif bot_edge(yVol=yVol) & (direction == DOWN):
				yVol = yVol - 1
				direction = UP

			else:
				yVol += DIRECTION_DELTA_MAP[direction].y

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
			if top_left_corner(xTemp=xTemp, yVol=yVol):
				xTemp = xTemp + 1
				yVol = yVol + 1
				direction = 3

			elif top_right_corner(xTemp=xTemp, yVol=yVol):
				xTemp = xTemp - 1
				yVol = yVol + 1
				direction = 5

			elif bot_right_corner(xTemp=xTemp, yVol=yVol):
				xTemp = xTemp - 1
				yVol = yVol - 1
				direction = 7

			elif bot_left_corner(xTemp=xTemp, yVol=yVol):
				xTemp = xTemp + 1
				yVol = yVol - 1
				direction = 1

			# Not in a corner, on an edge
			elif top_edge(yVol=yVol):
				if direction in DIAGONAL_RIGHT:
					xTemp = xTemp + 1
					yVol = yVol + 1
					direction = 3
				else:
					xTemp = xTemp - 1
					yVol = yVol + 1
					direction = 5

			elif bot_edge(yVol=yVol):
				if direction in DIAGONAL_RIGHT:
					xTemp = xTemp + 1
					yVol = yVol - 1
					direction = 1
				else:
					xTemp = xTemp - 1
					yVol = yVol - 1
					direction = 7

			elif left_edge(xTemp=xTemp):
				if direction in DIAGONAL_UP:
					xTemp = xTemp + 1
					yVol = yVol - 1
					direction = 1
				else:
					xTemp = xTemp + 1
					yVol = yVol + 1
					direction = 3

			elif right_edge(xTemp=xTemp):
				if direction in DIAGONAL_UP:
					xTemp = xTemp - 1
					yVol = yVol - 1
					direction = 7
				else:
					xTemp = xTemp - 1
					yVol = yVol + 1
					direction = 5

			# In middle of grid, not on corner or edge
			else:
				xTemp = xTemp + DIRECTION_DELTA_MAP[direction].x
				yVol = yVol + DIRECTION_DELTA_MAP[direction].y

		# print(f"i: {i+1} | xTemp: {xTemp} | yVol: {yVol} | direction: {direction}")

	return (10 * yVol) + xTemp


def main() -> int:
	out_list = []

	for xTemp in range(NUM_TEMP_POSITIONS):
		for yVol in range(NUM_VOLUME_POSITIONS):
			for direction in range(NUM_DIRECTIONS):
				target = calc_target(xTemp=xTemp, yVol=yVol, direction=direction)
				row_list = [xTemp, yVol, direction, target]
				out_list.append(row_list)

	data = pd.DataFrame(out_list, columns = OUTPUT_COLUMNS)
	data.to_csv("../data/tenByTenModelData.csv", index=False)

	# xTemp = 1
	# yVol = 0
	# direction = 5
	# target = calc_target(xTemp=xTemp, yVol=yVol, direction=direction)
	# row_list = [xTemp, yVol, direction, target]
	# out_list.append(row_list)

	# print(out_list)

if __name__ == '__main__':
	main()