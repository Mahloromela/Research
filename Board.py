# Constants
import copy

WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 5
BOARD_ROWS, BOARD_COLS = 7, 7
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 6
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4
BG_COLOR = (173, 216, 230)
LINE_COLOR = (23, 145, 135)
CIRCLE_BG_COLOR = (200, 200, 200)
CIRCLE_COLOR = (255, 0, 0)
CROSS_COLOR = (84, 84, 84)

positions = [
    (0, 0), (0, 3), (0, 6),
    (1, 1), (1, 3), (1, 5),
    (2, 2), (2, 3), (2, 4),
    (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4),
    (5, 1), (5, 3), (5, 5),
    (6, 0), (6, 3), (6, 6)
]

adjacent_points = {
    (0, 0): [(0, 3), (3, 0), (1, 1)],
    (0, 3): [(0, 0), (0, 6), (1, 3)],
    (0, 6): [(0, 3), (3, 6), (1, 5)],
    (1, 1): [(1, 3), (3, 1), (0, 0), (2, 2)],
    (1, 3): [(1, 1), (1, 5), (2, 3), (0, 3)],
    (1, 5): [(1, 3), (3, 5), (0, 6), (2, 4)],
    (2, 2): [(2, 3), (3, 2), (1, 1)],
    (2, 3): [(2, 2), (2, 4), (1, 3)],
    (2, 4): [(2, 3), (3, 4), (1, 5)],
    (3, 0): [(0, 0), (3, 1), (6, 0)],
    (3, 1): [(3, 0), (3, 2), (1, 1), (5, 1)],
    (3, 2): [(3, 1), (2, 2), (4, 2)],
    (3, 4): [(3, 5), (2, 4), (4, 4)],
    (3, 5): [(3, 4), (3, 6), (1, 5), (5, 5)],
    (3, 6): [(3, 5), (0, 6), (6, 6)],
    (4, 2): [(3, 2), (4, 3), (5, 1)],
    (4, 3): [(4, 2), (4, 4), (5, 3)],
    (4, 4): [(4, 3), (3, 4), (5, 5)],
    (5, 1): [(4, 2), (5, 3), (3, 1), (6, 0)],
    (5, 3): [(5, 1), (5, 5), (4, 3), (6, 3)],
    (5, 5): [(5, 3), (4, 4), (3, 5), (6, 6)],
    (6, 0): [(3, 0), (6, 3), (5, 1)],
    (6, 3): [(6, 0), (6, 6), (5, 3)],
    (6, 6): [(6, 3), (3, 6), (5, 5)]
}

mills = [
    # Horizontal mills
    [(0, 0), (0, 3), (0, 6)],
    [(1, 1), (1, 3), (1, 5)],
    [(2, 2), (2, 3), (2, 4)],
    [(3, 0), (3, 1), (3, 2)],
    [(3, 4), (3, 5), (3, 6)],
    [(4, 2), (4, 3), (4, 4)],
    [(5, 1), (5, 3), (5, 5)],
    [(6, 0), (6, 3), (6, 6)],

    # Vertical mills
    [(0, 0), (3, 0), (6, 0)],
    [(1, 1), (3, 1), (5, 1)],
    [(2, 2), (3, 2), (4, 2)],
    [(0, 3), (1, 3), (2, 3)],
    [(4, 3), (5, 3), (6, 3)],
    [(2, 4), (3, 4), (4, 4)],
    [(1, 5), (3, 5), (5, 5)],
    [(0, 6), (3, 6), (6, 6)],

    # Diagonal Mills
    [(0, 0), (1, 1), (2, 2)],
    [(6, 0), (5, 1), (4, 2)],
    [(6, 6), (5, 5), (4, 4)],
    [(0, 6), (1, 5), (2, 4)],

]


class Board:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.empty_moves = [True for _ in range(BOARD_COLS * BOARD_ROWS)]
        self.board_representation = [1 if self.board[row][col] == 'X' else (-1 if self.board[row][col] == 'O' else 0)
                                     for row, col in positions]

    def mark_square(self, row, col, player):
        self.board[row][col] = player

    def is_square_empty(self, row, col):
        return self.board[row][col] == ' '

    def remove_piece(self, row, col):
        if not self.is_square_empty(row, col):
            self.mark_square(row, col, ' ')

    def get_index(self, row, col):
        return row * BOARD_ROWS + col

    def copy(self):
        return copy.deepcopy(self.board)
