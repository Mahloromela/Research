import random

from Research.Board import *

from Research.Algorithms.Player import Player


class RandomAgent(Player):
    def __init__(self, piece):
        super().__init__(piece)
        self.name = "Random Agent"

    def make_move(self, game):
        if game.removal_mode:
            return self.remove_random_piece(game)
        elif self.stage == 1:
            return self.place_random_piece(game)
        else:
            return self.move_random_piece(game)

    def place_random_piece(self, game):
        return random.choice(list(game.free_positions)) if game.free_positions else None

    def move_random_piece(self, game):
        my_pieces = list(self.my_pieces)
        moves = random.sample(my_pieces, len(my_pieces))
        for from_pos in moves:
            if self.stage == 2:
                valid_moves = [pos for pos in adjacent_points[from_pos] if game.Board.is_square_empty(*pos)]
            else:  # stage 3, can fly to any empty position
                valid_moves = list(game.free_positions)
            if valid_moves:
                to_pos = random.choice(valid_moves)

                return [from_pos, to_pos]
        return None

    def remove_random_piece(self, game):
        opponent_pieces = list(self.opp.my_pieces)
        return random.choice(opponent_pieces) if opponent_pieces else None
