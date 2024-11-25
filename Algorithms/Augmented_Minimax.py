import random
import time
from typing import Dict, Tuple
from Research.Board import *
import numpy as np
from sklearn.preprocessing import StandardScaler

from Research.Preprocessing import count_mills

from Research.Board import positions

central_positions = [(1, 1), (1, 3), (1, 5), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3),
                     (4, 4), (5, 1), (5, 3), (5, 5)]
from Research.Algorithms.Player import Player
from Research.Algorithms.Neural_network import NNAgent, load_most_recent_model

scaler = StandardScaler()


class AugmentMinimaxAgent5(Player):
    def __init__(self, piece, max_depth=1, time_limit=15):
        super().__init__(piece)
        self.piece = piece
        self.max_depth = max_depth
        self.time_limit = time_limit  # Time limit in seconds
        self.transposition_table1: Dict[str, Tuple[float, int]] = {}
        self.transposition_table2: Dict[str, Tuple[float, int]] = {}
        self.transposition_table3: Dict[str, Tuple[float, int]] = {}
        self.opp = None
        self.my_pieces = set()
        self.name = "ANN"
        self.agent = NNAgent()
        self.agent.placement_value_model = load_most_recent_model('placement_value_model')
        self.agent.movement_value_model = load_most_recent_model('movement_value_model')
        self.agent.jumping_value_model = load_most_recent_model('jumping_value_model')

    def minimax(self, game, depth, alpha, beta, maximizing_player, start_time):
        state_hash = self.hash_state(game)

        # check from the h_table
        if state_hash in self.transposition_table1 and game.current_player.stage == 1:
            stored_value, stored_depth = self.transposition_table1[state_hash]
            if stored_depth >= depth:
                return stored_value

        elif state_hash in self.transposition_table2 and game.current_player.stage == 2:
            stored_value, stored_depth = self.transposition_table2[state_hash]
            if stored_depth >= depth:
                return stored_value

        elif state_hash in self.transposition_table3 and game.current_player.stage == 3:
            stored_value, stored_depth = self.transposition_table3[state_hash]
            if stored_depth >= depth:
                return stored_value

        if depth == 0 or game.check_winner() != 'In Progress' or self.time_limit < time.time() - start_time:
            return self.evaluate(game)

        if maximizing_player:
            max_eval = float('-inf')
            possible_moves = self.get_possible_moves(game.current_player, game)
            # possible_moves = self.get_ordered_moves(game, game.current_player, True)
            for move in possible_moves:
                new_game = self.apply_move(game, move)
                eval = self.minimax(new_game, depth - 1, alpha, beta, False, start_time)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            # #add to the h_table
            if game.current_player.stage == 1:
                self.transposition_table1[state_hash] = (max_eval, depth)
            elif game.current_player.stage == 2:
                self.transposition_table2[state_hash] = (max_eval, depth)
            elif game.current_player.stage == 3:
                self.transposition_table3[state_hash] = (max_eval, depth)
            return max_eval


        else:
            min_eval = float('inf')
            possible_moves = self.get_possible_moves(game.current_player, game)
            # possible_moves = self.get_ordered_moves(game, game.current_player, False)
            for move in possible_moves:
                new_game = self.apply_move(game, move)
                eval = self.minimax(new_game, depth - 1, alpha, beta, True, start_time)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            if game.current_player.stage == 1:
                self.transposition_table1[state_hash] = (min_eval, depth)
            elif game.current_player.stage == 2:
                self.transposition_table2[state_hash] = (min_eval, depth)
            elif game.current_player.stage == 3:
                self.transposition_table3[state_hash] = (min_eval, depth)
            return min_eval

    def best_move(self, game):
        start_time = time.time()
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        board_state = np.array(
            [scaler.fit_transform(self.board_to_numpy(game.Board.board, game).reshape(-1, 1)).flatten()])
        print('The intial board estimation is',
              self.agent.movement_predict_value(board_state * self.revert(game)))
        # possible_moves = self.get_ordered_moves(game, game.current_player, True)
        possible_moves = self.get_possible_moves(game.current_player, game)
        start_time = time.time()
        for move in possible_moves:
            new_game = self.apply_move(game, move)
            score = self.minimax(new_game, self.max_depth, alpha, beta, False, start_time)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        print(best_score, best_move)
        return best_move

    def get_ordered_moves(self, game, player, maximizing_player):
        moves = self.get_possible_moves(player, game)

        def move_score(move):
            new_game = self.apply_move(game, move)
            score = self.quick_evaluate(new_game)
            return score

        return sorted(moves, key=move_score, reverse=maximizing_player)

    def quick_evaluate(self, game):
        player1_pieces = sum(1 for row, col in positions if game.Board.board[row][col] == 'X')
        player2_pieces = sum(1 for row, col in positions if game.Board.board[row][col] == 'O')
        # return (game.player1.pieces_usable if piece == 'X' else game.Player2.pieces_usable) - \
        # (game.Player2.pieces_usable if piece == 'X' else game.player1.pieces_usable)
        return (player1_pieces - player2_pieces) * 5 * self.revert(game)

    def hash_state(self, game):
        return hash(tuple(map(tuple, game.Board.board)))

    def evaluate(self, game):
        revert = self.revert(game)
        current_state = np.array(
            [scaler.fit_transform(self.board_to_numpy(game.Board.board, game).reshape(-1, 1)).flatten()]) * revert

        if self.stage == 1:
            return self.agent.placement_predict_value(current_state)
        elif self.stage == 2:
            return self.agent.movement_predict_value(current_state)

        return self.agent.jumping_predict_value(current_state)

    def revert(self, game):
        return 1 if self.piece == game.player1.piece else -1

    def board_to_numpy(self, board, game):
        # Piece counts
        player1_pieces = sum(1 for row, col in positions if board[row][col] == 'X')
        player2_pieces = sum(1 for row, col in positions if board[row][col] == 'O')

        # Potential mills
        player1_potential_mills = self.count_potential_mills(board, 'X')
        player2_potential_mills = self.count_potential_mills(board, 'O')

        # Closed mills
        player1_closed_mills = self.count_mills(board, 'X')
        player2_closed_mills = self.count_mills(board, 'O')

        # Blocked mills
        player1_blocked_mills = self.blocked_potential_mills(board, 'X', 'O')
        player2_blocked_mills = self.blocked_potential_mills(board, 'O', 'X')

        # Configuration counts
        player1_configuration = self.configuration_count(board, 'X')
        player2_configuration = self.configuration_count(board, 'O')

        # Additional Features
        player1_mobility = self.mobility(board, 'X')
        player2_mobility = self.mobility(board, 'O')

        player1_central_control = self.central_control(board, 'X')
        player2_central_control = self.central_control(board, 'O')

        player1_blockades = self.blockades(board, 'O')
        player2_blockades = self.blockades(board, 'X')

        player1_stability = self.piece_stability(board, 'X')
        player2_stability = self.piece_stability(board, 'O')

        pieces_usable = 0
        if game.current_player.stage == 1:
            pieces_usable = game.player1.pieces_usable - game.player2.pieces_usable
        # Difference-based features
        piece_diff = player1_pieces - player2_pieces + pieces_usable
        closed_mills_diff = player1_closed_mills - player2_closed_mills
        potential_mills_diff = player1_potential_mills - player2_potential_mills
        blocked_mills_diff = player1_blocked_mills - player2_blocked_mills
        configuration_diff = player1_configuration - player2_configuration
        mobility_diff = player1_mobility - player2_mobility
        central_control_diff = player1_central_control - player2_central_control
        blockades_diff = player1_blockades - player2_blockades
        stability_diff = player1_stability - player2_stability

        # Combine all features into a single array
        parameters = [
            piece_diff,
            closed_mills_diff,
            potential_mills_diff,
            blocked_mills_diff,
            configuration_diff,
            mobility_diff,
            central_control_diff,
            blockades_diff,
            stability_diff

        ]
        board_representation = [1 if board[row][col] == 'X' else (-1 if board[row][col] == 'O' else 0)
                                for row, col in positions]

        # Combine all features
        return np.concatenate((parameters, board_representation))

    def get_possible_moves(self, player, game):
        if game.removal_mode:
            return list(player.opp.my_pieces)
        elif player.stage == 1:
            return list(game.free_positions)
        else:
            moves = []
            my_pieces = player.my_pieces
            for from_pos in my_pieces:
                if player.stage == 2:
                    valid_moves = [pos for pos in adjacent_points[from_pos] if game.Board.is_square_empty(*pos)]
                else:  # stage 3, can fly to any empty position
                    valid_moves = game.free_positions
                moves.extend([from_pos, to_pos] for to_pos in valid_moves)
            return moves

    def apply_move(self, game, move):
        new_game = game.copy()
        if new_game.removal_mode:
            new_game.handle_removal(*move)
        elif new_game.current_player.stage == 1:
            new_game.handle_placement(*move)
        else:
            new_game.handle_movement(*move[0], *move[1])
        return new_game

    def make_move(self, game):
        if game.removal_mode:
            return self.remove_piece(game)
        elif self.stage == 1:
            return self.place_piece(game)
        else:
            return self.move_piece(game)

    def place_piece(self, game):
        return self.best_move(game)

    def move_piece(self, game):
        return self.best_move(game)

    def remove_piece(self, game):
        return self.best_move(game)

    def update_stage(self):
        if self.pieces == 0 and self.stage == 1:
            self.stage = 2
        elif self.pieces_usable == 3 and self.stage == 2:
            self.stage = 3

    def mobility(self, board, piece):
        moves = 0
        for r, c in positions:
            if board[r][c] == piece:
                for adj_r, adj_c in adjacent_points[(r, c)]:
                    if board[adj_r][adj_c] == ' ':
                        moves += 1
        return moves

    def blockades(self, board, opponent_piece):
        blocked = 0
        for r, c in positions:
            if board[r][c] == opponent_piece:
                if all(board[adj_r][adj_c] != ' ' for adj_r, adj_c in adjacent_points[(r, c)]):
                    blocked += 1
        return blocked

    def central_control(self, board, piece):
        return sum(1 for r, c in central_positions if board[r][c] == piece)

    def piece_stability(self, board, piece):
        stable = 0
        for r, c in positions:
            if board[r][c] == piece:
                # Define stability criteria, e.g., part of multiple mills or not adjacent to opponent pieces
                if count_mills(board, piece) > 3:
                    stable += 1
        return stable

    def count_mills(self, board, piece):
        mill_count = 0
        for mill in mills:
            if all(board[r][c] == piece for r, c in mill):
                mill_count += 1
        return mill_count

    def configuration_count(self, board, piece):
        configuration_count = 0
        for r, c in positions:
            if board[r][c] == piece:
                # Check all adjacent points for the same piece
                if all(board[adj_r][adj_c] == piece for adj_r, adj_c in adjacent_points[(r, c)]):
                    configuration_count += 1
        return configuration_count

    def count_potential_mills(self, board, piece):
        potential_mills = 0
        for mill in mills:  # Assuming mills is a class attribute
            pieces = [board[r][c] for r, c in mill]
            if pieces.count(piece) == 2 and pieces.count(' ') == 1:
                empty_spot_coords = mill[pieces.index(' ')]
                if any([board[pos[0]][pos[1]] == piece for pos in adjacent_points[empty_spot_coords]]):
                    potential_mills += 1
        return potential_mills

    def blocked_potential_mills(self, board, piece1, piece2):
        potential_mills = 0
        for mill in mills:
            pieces = [board[r][c] for r, c in mill]
            if pieces.count(piece2) == 2 and pieces.count(piece1) == 1:
                current_player_spot_coords = mill[pieces.index(piece1)]
                if any([board[pos[0]][pos[1]] == piece2 for pos in adjacent_points[current_player_spot_coords]]):
                    potential_mills += 1
        return potential_mills
