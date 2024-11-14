import copy

import pygame
from Monte_Carlo_Agent import MCTSAgent
from Minimax import MinimaxAgent
from Player import Player

# from Morabaraba.Training3.Augmented_Minimax import AugmentMinimaxAgent3
# from Morabaraba.Training4.Augmented_Minimax import AugmentMinimaxAgent4
from Augmented_Minimax import AugmentMinimaxAgent5
from NN_Minimmax import AugmentedMinimaxAgent

from RandomAgent import RandomAgent
from Board import *

# Initialize Pygame
pygame.init()
# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Morabaraba')


class Game:
    def __init__(self, player1=MinimaxAgent('X', max_depth=3), player2=RandomAgent('O')):
        self.game_over = False
        self.Board = Board()
        self.game_states = {1: [], 2: [], 3: [], 'Winner': ''}
        self.move_history = []
        self.player1 = player1
        self.player2 = player2
        self.player1.piece = 'X'
        self.player2.piece = 'O'
        self.player1.opp = self.player2
        self.player2.opp = self.player1
        self.current_player = self.player1
        self.removal_mode = False
        self.choose_mode = False
        self.free_positions = set(positions)

    def switch_player(self):
        if self.current_player == self.player1:
            self.current_player = self.player2
        else:
            self.current_player = self.player1

    def return_score(self):
        return self.player1.pieces_usable - self.player2.pieces_usable

    def check_for_mill(self, position):
        relevant_mills = [mill for mill in mills if position in mill]
        return any(
            self.Board.board[mill[0][0]][mill[0][1]] ==
            self.Board.board[mill[1][0]][mill[1][1]] ==
            self.Board.board[mill[2][0]][mill[2][1]] != ' '
            for mill in relevant_mills
        )

    def handle_removal(self, row, col):
        if (row, col) in positions and not self.Board.is_square_empty(row, col) and \
                self.Board.board[row][col] == self.current_player.opp.piece:
            self.Board.remove_piece(row, col)
            self.free_positions.add((row, col))
            self.current_player.opp.my_pieces.remove((row, col))
            self.current_player.opp.update_stage()
            self.removal_mode = False

            self.switch_player()

    def handle_placement(self, row, col):
        if self.Board.is_square_empty(row, col):
            self.Board.mark_square(row, col, self.current_player.piece)
            self.free_positions.remove((row, col))
            self.current_player.my_pieces.add((row, col))
            self.current_player.pieces -= 1
            self.current_player.update_stage()

            if self.check_for_mill((row, col)):
                self.removal_mode = True
                self.current_player.opp.pieces_usable -= 1
            else:
                self.switch_player()

    def handle_movement(self, from_row, from_col, to_row, to_col):

        if (to_row, to_col) in adjacent_points[(from_row, from_col)] and self.Board.is_square_empty(to_row,
                                                                                                    to_col) and self.current_player.stage == 2:
            self.Board.mark_square(to_row, to_col, self.current_player.piece)

            self.free_positions.remove((to_row, to_col))
            self.current_player.my_pieces.add((to_row, to_col))
            self.current_player.my_pieces.remove((from_row, from_col))
            self.current_player.update_stage()

            self.Board.remove_piece(from_row, from_col)
            self.free_positions.add((from_row, from_col))
            # Record the move in history
            self.current_player.move_history.append((from_row, from_col, to_row, to_col))

            if self.check_for_mill((to_row, to_col)):
                self.removal_mode = True
                self.current_player.opp.pieces_usable -= 1


            else:
                self.switch_player()


        elif self.Board.is_square_empty(to_row, to_col) and self.current_player.stage == 3:
            self.Board.mark_square(to_row, to_col, self.current_player.piece)
            self.free_positions.remove((to_row, to_col))
            self.current_player.my_pieces.add((to_row, to_col))
            self.current_player.my_pieces.remove((from_row, from_col))

            self.Board.remove_piece(from_row, from_col)
            self.free_positions.add((from_row, from_col))
            self.current_player.move_history.append((from_row, from_col, to_row, to_col))

            if self.check_for_mill((to_row, to_col)):
                self.removal_mode = True
                self.current_player.opp.pieces_usable -= 1
            else:
                self.switch_player()

    def handle_move(self, row, col):

        if self.removal_mode:
            self.handle_removal(row, col)

        elif self.get_current_player().stage == 1:
            self.handle_placement(row, col)

        elif self.current_player.stage in [2, 3]:
            if not hasattr(self, 'selected_piece'):
                # First click: Select a piece to move
                if self.Board.board[row][col] == self.current_player.piece:
                    self.selected_piece = (row, col)
                    print(f"Selected piece at {row}, {col}")
                else:
                    print("Select your own piece to move")

            else:
                # Second click: Move the selected piece
                from_row, from_col = self.selected_piece
                if self.is_valid_move(from_row, from_col, row, col):
                    self.handle_movement(from_row, from_col, row, col)
                    delattr(self, 'selected_piece')
                    # Record the   subprocess.run(["python3", r"/home/vmuser/Pictures/Code/Morabaraba/Training5/Fine_Tune.py"])move in history
                    # self.current_player.move_history.append((row, col))

                else:
                    print("Invalid move. Try again.")
                    delattr(self, 'selected_piece')

    def check_repeated_moves(self):
        last_moves1 = self.player1.move_history[-5:]
        last_moves2 = self.player2.move_history[-5:]
        if (len(last_moves1) == 5 and last_moves1[0] == last_moves1[2] == last_moves1[4]) and \
                (len(last_moves2) == 5 and last_moves2[0] == last_moves2[2] == last_moves2[4]):
            return True
        return False

    def is_valid_move(self, from_row, from_col, to_row, to_col):
        if self.Board.board[to_row][to_col] != ' ':
            return False
        if self.current_player.stage == 2:
            return (to_row, to_col) in adjacent_points[(from_row, from_col)]
        elif self.current_player.stage == 3:
            return True  # Can move to any empty spot in stage 3
        return False

    def check_winner(self):
        # Check current and opposition pieces usable
        if self.current_player.pieces_usable < 3 and self.current_player.opp.pieces_usable > 2:
            self.game_over = True
            return self.current_player.opp.piece
        elif self.current_player.pieces_usable > 3 and self.current_player.opp.pieces_usable < 2:
            self.game_over = True
            return self.current_player.piece

        # Checking Draw where there are equal points on the Board
        elif self.current_player.pieces_usable > 2 and self.current_player.opp.pieces_usable > 2:
            count = sum(1 for position in positions if self.Board.board[position[0]][position[1]] != " ")
            if count == len(positions) and self.removal_mode == False:
                self.game_over = True
                return "Draw"

            # Check if all pieces are blocked
            for player in [self.player1, self.player2]:
                can_move = False
                for position in positions:
                    if self.Board.board[position[0]][position[1]] == player.piece:
                        adjacent_positions = adjacent_points[position]
                        if any(self.Board.board[adj[0]][adj[1]] == ' ' for adj in adjacent_positions):
                            can_move = True
                            break
                if (((not can_move and player.stage == 2)
                     or (self.check_repeated_moves() and player.stage in [2, 3])) and
                        self.removal_mode == False):  # Only check for blocked pieces in stage 2
                    self.game_over = True
                    return "Draw"

        return "In Progress"

    def get_current_player(self):
        if self.current_player == self.player1:
            return self.player1
        return self.player2

    def draw_board(self):
        screen.fill(BG_COLOR)
        self.draw_mills()
        self.draw_circles()
        self.draw_figures()
        self.draw_status()
        pygame.display.update()
        self.game_over = False

    def draw_mills(self):
        for line in mills:
            start = (line[0][1] * SQUARE_SIZE + SQUARE_SIZE // 2, line[0][0] * SQUARE_SIZE + SQUARE_SIZE // 2)
            end = (line[2][1] * SQUARE_SIZE + SQUARE_SIZE // 2, line[2][0] * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.line(screen, LINE_COLOR, start, end, LINE_WIDTH)

    def draw_circles(self):
        for position in positions:
            pygame.draw.circle(screen, CIRCLE_BG_COLOR,
                               (position[1] * SQUARE_SIZE + SQUARE_SIZE // 2,
                                position[0] * SQUARE_SIZE + SQUARE_SIZE // 2),
                               CIRCLE_RADIUS + 5)
            pygame.draw.circle(screen, BG_COLOR,
                               (position[1] * SQUARE_SIZE + SQUARE_SIZE // 2,
                                position[0] * SQUARE_SIZE + SQUARE_SIZE // 2),
                               CIRCLE_RADIUS, CIRCLE_WIDTH)

    def draw_figures(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.Board.board[row][col] == 'X':
                    pygame.draw.line(screen, CROSS_COLOR,
                                     (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                     (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
                    pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE),
                                     (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE),
                                     CROSS_WIDTH)
                elif self.Board.board[row][col] == 'O':
                    pygame.draw.circle(screen, CIRCLE_COLOR,
                                       (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                       CIRCLE_RADIUS, CIRCLE_WIDTH)

    def draw_status(self):
        font = pygame.font.Font(None, 36)
        text = f"Player {self.current_player.piece}'s turn"
        if self.removal_mode:
            text += " - Remove opponent's piece"
            if self.current_player == self.player2:
                move = self.player2.make_move(self)
                self.handle_click(move[0], move[1])
        text_surface = font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (30, HEIGHT - 10))

    def handle_click(self, row, col):
        try:
            if (row, col) not in positions:
                raise ValueError("Invalid position")
            if self.current_player == self.player1:
                self.handle_move(row, col)
                pygame.display.update()
            Human = False
            if self.current_player == self.player2 and not self.game_over and Human:
                self.ai_turn()
            else:
                self.handle_move(row, col)
                pygame.display.update()
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def ai_turn(self):
        move = self.player2.make_move(self)
        if move is not None:
            if self.removal_mode:
                self.handle_removal(*move)
            elif self.player2.stage == 1:
                self.handle_placement(*move)
            else:
                self.handle_movement(*move[0], *move[1])

    def copy(self):
        return copy_state(game=self)

    def reset(self):
        # Reset the game state for a new game
        self.Board = Board()
        self.current_player = self.player1
        self.removal_mode = False
        self.game_over = False
        self.free_positions = set(positions)
        self.player1.pieces = 12
        self.player2.pieces = 12
        self.player1.my_pieces.clear()
        self.player2.my_pieces.clear()
        self.player1.pieces_usable = 12
        self.player2.pieces_usable = 12
        self.player1.stage = 1
        self.player2.stage = 1
        self.player1.move_history = []
        self.player2.move_history = []


def copy_state(game):
    new_state = Game(game.player1.copy(), game.player2.copy())
    new_state.Board.board = game.Board.copy()
    new_state.current_player = new_state.player1
    if game.current_player == game.player2:
        new_state.current_player = new_state.player2
    new_state.removal_mode = game.removal_mode
    new_state.free_positions = game.free_positions.copy()
    return new_state
