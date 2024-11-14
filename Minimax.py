import copy
import random
from Board import positions, adjacent_points, mills
from Player import Player


class MinimaxAgent(Player):
    def __init__(self, piece, max_depth=1):
        super().__init__(piece)
        self.max_depth = max_depth
        self.name = "Alpha-Beta Minimax"

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.check_winner() != 'In Progress':
            if depth == 3:
                print('The depth is at ', depth)

            return self.evaluate(game)

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_possible_moves(game.current_player, game):
                new_game = self.apply_move(game, move)
                evaluate = self.minimax(new_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, evaluate)
                alpha = max(alpha, evaluate)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_possible_moves(game.current_player, game):
                new_game = self.apply_move(game, move)
                evaluate = self.minimax(new_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, evaluate)
                beta = min(beta, evaluate)
                if beta <= alpha:
                    break
            return min_eval

    def best_move(self, game):
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        possible_moves = self.get_possible_moves(self, game)
        random.shuffle(possible_moves)

        is_maximizing = False
        j = 1
        for move in possible_moves:
            new_game = self.apply_move(game, move)
            score = self.minimax(new_game, self.max_depth, alpha, beta, is_maximizing)
            if score > best_score:
                best_score = score
                best_move = move
            # alpha = max(alpha, score)
            # beta = min(beta, score)
        return best_move

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

    def evaluate(self, game):
        winner = game.check_winner()
        # val = -1
        # if self.is_player_1:
        # val = 1
        score = 0

        if winner == "Draw":
            return 0
        elif winner == self.piece:
            score += 1000  # if self.is_player_1 else -1000
        elif winner == self.opp.piece:
            return -1000

        # Count pieces on the Board
        player1_pieces = sum(1 for row, col in positions if game.Board.board[row][col] == self.piece)
        player2_pieces = sum(1 for row, col in positions if game.Board.board[row][col] == self.opp.piece)

        # Count potential mills
        player1_mills = self.count_potential_mills(game, self.piece)
        player2_mills = self.count_potential_mills(game, self.opp.piece)

        # Combine factors
        player1_closed_mills = self.closed_mill(game, self)
        player2_closed_mills = self.closed_mill(game, self.opp)
        #counting the pieces for stage 2 is more proper

        # score += (self.pieces_usable - self.opp.pieces_usable) * 15
        score += (player1_pieces - player2_pieces) * 5
        if self.stage == 1:
            score += player1_closed_mills*20

        if self.opp.stage == 3:
            score += (player1_mills - player2_mills) * 15
            score -= player2_closed_mills*20
        if self.stage == 3:
            score += (player1_mills - player2_mills) * 15
            score += player1_closed_mills*20
        return score

    def count_potential_mills(self, game, player):
        potential_mills = 0
        for mill in mills:  # Assuming mills is a class attribute
            pieces = [game.Board.board[r][c] for r, c in mill]
            if pieces.count(player) == 2 and pieces.count(' ') == 1:
                empty_spot_coords = mill[pieces.index(' ')]
                if game.current_player.stage == 1:
                    potential_mills += 1
                elif any([game.Board.board[pos[0]][pos[1]] == player for pos in adjacent_points[empty_spot_coords]]):
                    potential_mills += 1

        return potential_mills

    def closed_mill(self, game, player):
        # Iterate through all mill combinations
        for mill in mills:
            # Check if the player's last move is part of the mill
            if player.last_move in mill:
                # Check if all positions in the mill contain the player's piece
                if all(game.Board.board[r][c] == player.piece for r, c in mill):
                    return True
        # If no mills are found, return False
        return False

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
                    pos = ()
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

    def count_mills(self, game, player):
        mill_count = 0
        for mill in mills:
            if all(game.Board.Board[r][c] == player.piece for r, c in mill):
                mill_count += 1
        return mill_count
