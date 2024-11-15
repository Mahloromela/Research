import sys
import time
from collections import OrderedDict
import subprocess
from Game import *
from Save_Game_States import save_game_states
import pygame
import logging
import Fine_Tune
from NN_Minimmax import AugmentedMinimaxAgent

class GameSimulator:
    """
    Simulates multiple games of Morabaraba between two agents and tracks outcomes.

    Attributes:
        num_games (int): Number of games to simulate.
        visualize (bool): Whether to visualize the games using Pygame.
        player1_wins (int): Count of Player1's wins.
        player2_wins (int): Count of Player2's wins.
        draws (int): Count of drawn games.
        game_states (dict): Dictionary to store game states categorized by stages.
    """

    def __init__(self, num_games=100, visualize=True):
        """
        Initializes the GameSimulator.

        Args:
            num_games (int): Number of games to simulate.
            visualize (bool): Whether to visualize the games using Pygame.
        """
        self.num_games = num_games
        self.visualize = visualize
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0
        self.game_states = {1: [], 2: [], 3: []}
        self.stage_pieces_usable = []

        if self.visualize:
            pygame.init()
            self.WIDTH, self.HEIGHT = 800, 600  # Define appropriate dimensions
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Morabaraba Game Simulation")

    def simulate(self):
        """
        Runs the simulation of multiple games and tracks the outcomes.
        """
        try:
            for i in range(self.num_games):
                game_instance = Game(player1=AugmentedMinimaxAgent('O', max_depth=1), player2=MinimaxAgent('X', max_depth=2))  # Initialize game
                winner = self.play_game(game_instance)
                game_instance.reset()
                # Update win/loss counters based on the winner
                if winner == 'X':  # Assuming 'X' is Player1
                    self.player1_wins += 1
                elif winner == 'O':  # Assuming 'O' is Player2
                    self.player2_wins += 1
                elif winner == "Draw":
                    self.draws += 1
                else:
                    logging.error(f"Game {i + 1}: Unexpected winner value '{winner}'")

                logging.info(f"Completed game {i + 1}/{self.num_games}")

        except Exception as e:
            logging.error(f"An error occurred during simulation: {e}")

        finally:
            # Print final results
            print(f"\n{game_instance.player1.name} Agent won {self.player1_wins} games")
            print(f"{game_instance.player2.name} Agent won {self.player2_wins} games")
            print(f"Number of draws: {self.draws}")

            if self.visualize:
                pygame.quit()

    def board_to_tuple(self, board):
        """
        Converts the board from a list of lists to a tuple of tuples for hashing.

        Args:
            board (list): The game board.

        Returns:
            tuple: A hashable representation of the board.
        """
        return tuple(tuple(row) for row in board)

    def play_game(self, game_instance):
        """
        Plays a single game instance.

        Args:
            game_instance (Game): An instance of the Morabaraba game.

        Returns:
            str: The winner ('X', 'O', or 'Draw').
        """
        if self.visualize:
            clock = pygame.time.Clock()

        unique_states = {1: OrderedDict(), 2: OrderedDict(), 3: OrderedDict()}
        all_states = {1: [], 2: [], 3: []}
        pieces_usable = []
        moves_no = 0
        start_time = time.time()

        if self.visualize:
            game_instance.draw_board()
            pygame.display.update()

        while game_instance.check_winner() == "In Progress":
            if self.visualize:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            current_player = game_instance.current_player
            move = current_player.make_move(game_instance)

            if move is not None:
                if game_instance.removal_mode:
                    game_instance.handle_removal(*move)
                    logging.info(f'Player {current_player.piece} removed piece at {move}')
                elif current_player.stage == 1:
                    game_instance.handle_placement(*move)
                    pieces_usable.append(
                        game_instance.player1.pieces_usable - game_instance.player2.pieces_usable)
                    logging.info(f'Player {current_player.piece} placed piece at {move}')
                    current_player.last_move = move
                else:
                    game_instance.handle_movement(*move[0], *move[1])
                    logging.info(f'Player {current_player.piece} moved piece from {move[0]} to {move[1]}')
                    current_player.last_move = move[1]

                # Convert the board to a tuple for hashing
                board_state = self.board_to_tuple(game_instance.Board.board)

                # Add the state to all_states (allowing duplicates)
                all_states[current_player.stage].append(board_state)

                # Only add the state if it's unique for the current stage
                if board_state not in unique_states[current_player.stage]:
                    unique_states[current_player.stage][board_state] = len(unique_states[current_player.stage])
                    logging.info(f'New unique state added for stage {current_player.stage}')
                else:
                    logging.warning(f'Duplicate state not added for stage {current_player.stage}')

                logging.debug(f"Current Board State:\n{game_instance.Board.board}")

            moves_no += 1

            if self.visualize:
                game_instance.draw_board()
                # pygame.display.update()
                # clock.tick(60)  # Limit FPS to 60

            if not game_instance.game_over:
                winner = game_instance.check_winner()
                if winner != "In Progress":
                    end_time = time.time()
                    logging.info(f"{winner} wins!" if winner != "Draw" else "It's a draw!")
                    game_instance.game_over = True
                    logging.info('Total Time taken is {:.2f} seconds'.format(end_time - start_time))

            if game_instance.game_over:
                States = {'states': all_states, 'Winner': winner, 'pieces_usable': pieces_usable}
                save_game_states(States, f"game_states")

                # Print summary of unique states for each stage
                for stage in [1, 2, 3]:
                    logging.info(f"Unique states for Stage {stage}: {len(unique_states[stage])}")
                logging.info(f'The number of moves taken are {moves_no}')

                if self.visualize:
                    self.display_game_over()

                break
            # clock.tick(30)  # limits FPS to 30
            # time.sleep(1)
        return winner  # Return the winner ("X", "O", or "Draw")

    def display_game_over(self):
        """
        Displays the 'Game Over' message on the Pygame screen.
        """
        font = pygame.font.Font(None, 74)
        text = font.render("Game Over", True, (255, 0, 0))
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text, text_rect)

        font = pygame.font.Font(None, 36)
        text = font.render("Proceeding to next game...", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 50))
        self.screen.blit(text, text_rect)
        pygame.display.update()

        # Pause briefly to allow visualization
        time.sleep(1)  # Adjust the duration as needed


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse command-line arguments if desired (optional)
    # For simplicity, we're directly instantiating with parameters
    simulator = GameSimulator(num_games=1200, visualize=False)  # Set visualize=True for graphical display
    simulator.simulate()
    # Fine_Tune.main()
    
