import os
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import pygame
from typing_extensions import TypedDict
import statistics
from datetime import datetime
from Game2 import Game
from Minimax import MinimaxAgent
from Augmented_Minimax import AugmentMinimaxAgent5
from NN_Minimmax import AugmentedMinimaxAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class GameState(TypedDict):
    states: Dict[int, List[Tuple[Tuple[str, ...], float]]]
    Winner: str
    metrics: Dict[str, Any]


@dataclass
class GameMetrics:
    total_duration: float
    moves_count: int
    avg_decision_time: float
    max_decision_time: float
    unique_states_count: Dict[int, int]
    stage_durations: Dict[int, float]
    agent_total_duration: float
    agent_avg_decision_time: float
    agent_max_decision_time: float


@dataclass
class SimulationStats:
    player1_wins: int = 0
    player2_wins: int = 0
    draws: int = 0
    total_games: int = 0
    avg_game_duration: float = 0.0
    avg_moves_per_game: float = 0.0
    metrics_history: List[GameMetrics] = None

    def __post_init__(self):
        self.metrics_history = []

    def add_game_metrics(self, metrics: GameMetrics) -> None:
        self.metrics_history.append(metrics)
        self.update_averages()

    def update_averages(self) -> None:
        if self.metrics_history:
            self.avg_game_duration = statistics.mean(m.total_duration for m in self.metrics_history)
            self.avg_moves_per_game = statistics.mean(m.moves_count for m in self.metrics_history)
            self.total_games = len(self.metrics_history)


class GameStateManager:
    """Enhanced manager for saving and analyzing game states."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_session_path(self) -> Path:
        """Create and return a session-specific directory."""
        session_path = self.base_path / self.session_id
        session_path.mkdir(exist_ok=True)
        return session_path

    def save_game_states(self, game_states: GameState, game_number: int) -> None:
        """Save game states with enhanced metrics to a session-specific JSON file."""
        try:
            session_path = self.get_session_path()
            filename = session_path / f"game_{game_number}.json"

            with open(filename, 'w') as f:
                json.dump(game_states, f, indent=2)
            logging.info(f"Game states saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save game states: {e}")

    def save_session_summary(self, stats: SimulationStats) -> None:
        """Save overall session statistics and analysis."""
        try:
            session_path = self.get_session_path()
            summary = {
                'total_games': stats.total_games,
                'player1_wins': stats.player1_wins,
                'player2_wins': stats.player2_wins,
                'draws': stats.draws,
                'avg_game_duration': stats.avg_game_duration,
                'avg_moves_per_game': stats.avg_moves_per_game,
                'metrics_history': [asdict(m) for m in stats.metrics_history]
            }

            with open(session_path / 'session_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            logging.info(f"Session summary saved to {session_path / 'session_summary.json'}")
        except Exception as e:
            logging.error(f"Failed to save session summary: {e}")


class GameVisualizer:
    """Enhanced game visualization with performance metrics display."""

    def __init__(self, width: int = 1024, height: int = 700):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Morabaraba Game Simulation")
        self.fonts = {
            'large': pygame.font.Font(None, 74),
            'medium': pygame.font.Font(None, 48),
            'small': pygame.font.Font(None, 20)
        }

    def draw_metrics(self, metrics: GameMetrics, current_stage: int, player_name: str) -> None:
        """Display real-time game metrics."""
        metrics_surface = pygame.Surface((600, self.height-200))
        metrics_surface.fill((240, 240, 240))
        y_pos = 20

        for text, value in [
            ("Current Stage", f"{current_stage}"),
            ("Moves", f"{metrics.moves_count}"),
            ("Avg Decision Time", f"{metrics.avg_decision_time:}s"),
            ("Max Decision Time", f"{metrics.max_decision_time:}s"),
            (f"{player_name} Avg Decision Time", f"{metrics.agent_avg_decision_time:}s"),
            (f"{player_name} Max Decision Time", f"{metrics.agent_max_decision_time:}s"),
        ]:
            text_surface = self.fonts['small'].render(f"{text}: {value}", True, (0, 0, 0))
            metrics_surface.blit(text_surface, (10, y_pos))
            y_pos += 30

        self.screen.blit(metrics_surface, (self.width - 400, 0))

    def draw_game_over(self, message: str, stats: SimulationStats) -> None:
        """Display enhanced game over screen with statistics."""
        self.screen.fill((255, 255, 255))

        # Game over message
        text_large = self.fonts['large'].render("Game Over", True, (255, 0, 0))
        text_rect_large = text_large.get_rect(center=(self.width / 2, self.height / 3))
        self.screen.blit(text_large, text_rect_large)

        # Statistics
        y_pos = self.height / 2
        for text, value in [
            ("Winner", message),
            ("Games Played", f"{stats.total_games}"),
            ("Player 1 Wins", f"{stats.player1_wins}"),
            ("Player 2 Wins", f"{stats.player2_wins}"),
            ("Draws", f"{stats.draws}")
        ]:
            text_surface = self.fonts['medium'].render(f"{text}: {value}", True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(self.width / 2, y_pos))
            self.screen.blit(text_surface, text_rect)
            y_pos += 40

        pygame.display.update()

    def cleanup(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()


class GameSimulator:
    """Enhanced Morabaraba game simulator with comprehensive metrics tracking."""

    def __init__(self,
                 num_games: int = 100,
                 visualize: bool = False,
                 save_path: Path = Path("game_data"),
                 log_level: int = logging.INFO):
        self.num_games = num_games
        self.stats = SimulationStats()
        self.visualizer = GameVisualizer() if visualize else None
        self.state_manager = GameStateManager(save_path)

        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(save_path / "simulation.log"),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def board_to_tuple(board: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
        """Convert board state to hashable tuple format."""
        return tuple(tuple(row) for row in board)

    def handle_game_move(self, game_instance: 'Game', move: Optional[Tuple],
                         stage_metrics: defaultdict, player_metrics: defaultdict) -> None:
        """Process and record a game move with enhanced metrics tracking."""
        if move is None:
            return

        current_player = game_instance.current_player
        stage = current_player.stage

        # Time the move execution
        start_time = time.time()

        try:
            if game_instance.removal_mode:
                game_instance.handle_removal(*move)
                logging.info(f'Player {current_player.piece} removed piece at {move}')
            elif stage == 1:
                game_instance.handle_placement(*move)
                logging.info(f'Player {current_player.piece} placed piece at {move}')
                current_player.last_move = move
            else:
                game_instance.handle_movement(*move[0], *move[1])
                logging.info(f'Player {current_player.piece} moved from {move[0]} to {move[1]}')
                current_player.last_move = move[1]

            # Record metrics
            move_duration = time.time() - start_time
            stage_metrics[stage]['durations'].append(move_duration)
            stage_metrics[stage]['states'].append(self.board_to_tuple(game_instance.Board.board))
            player_metrics[current_player.name]['durations'].append(move_duration)

        except Exception as e:
            logging.error(f"Error handling move: {e}")
            raise

    def calculate_game_metrics(self, stage_metrics: defaultdict, player_metrics: defaultdict,
                               total_duration: float, name: str) -> GameMetrics:
        """Calculate comprehensive game metrics."""
        all_durations = [d for metrics in stage_metrics.values()
                         for d in metrics['durations']]

        unique_states = {
            stage: len(set(metrics['states']))
            for stage, metrics in stage_metrics.items()
        }

        stage_durations = {
            stage: sum(metrics['durations'])
            for stage, metrics in stage_metrics.items()
        }

        player_durations = {
            player_name: metrics['durations']
            for player_name, metrics in player_metrics.items()
        }
        logging.info(msg=player_durations)
        if name in player_durations:
            agent_avg_decision_time = statistics.mean(player_durations[name]) if player_durations[name] else 0
            agent_total_duration = sum(player_durations[name])
            agent_max_decision_time = max(player_durations[name]) if player_durations[name] else 0
        else:
            logging.warning(f"Player name {name} not found in player_durations.")


        return GameMetrics(
                    total_duration=total_duration,
                    moves_count=len(all_durations),
                    avg_decision_time=statistics.mean(all_durations) if all_durations else 0,
                    max_decision_time=max(all_durations) if all_durations else 0,
                    unique_states_count=unique_states,
                    stage_durations=stage_durations,
                    agent_avg_decision_time=agent_avg_decision_time,
                    agent_total_duration=agent_total_duration,
                    agent_max_decision_time=agent_max_decision_time
                )

    def play_game(self, game_instance: 'Game', game_number: int) -> str:
        """Play a single game with enhanced metrics tracking."""
        stage_metrics = defaultdict(lambda: {'durations': [], 'states': []})
        player_metrics = defaultdict(lambda: {'durations': []})
        start_time = time.time()

        try:
            while game_instance.check_winner() == "In Progress":
                if self.visualizer:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise SystemExit

                    game_instance.draw_board()
                    if game_instance.current_player.name not in player_metrics:
                        player_metrics[game_instance.current_player.name] = {'durations': []}

                    if stage_metrics:
                        current_metrics = self.calculate_game_metrics(
                            stage_metrics, player_metrics, time.time() - start_time, game_instance.current_player.name)
                        self.visualizer.draw_metrics(
                            current_metrics, game_instance.current_player.stage, game_instance.current_player.name)
                    pygame.display.update()

                move = game_instance.current_player.make_move(game_instance)
                self.handle_game_move(game_instance, move, stage_metrics, player_metrics)

                winner = game_instance.check_winner()
                if winner != "In Progress":
                    game_instance.game_over = True
                    total_duration = time.time() - start_time

                    # Calculate final metrics
                    game_metrics = self.calculate_game_metrics(stage_metrics, player_metrics, total_duration,game_instance.current_player.name)
                    self.stats.add_game_metrics(game_metrics)

                    # Save game states with metrics
                    self.state_manager.save_game_states(
                        GameState(
                            states={stage: [(s, d) for s, d in zip(metrics['states'],
                                                                   metrics['durations'])]
                                    for stage, metrics in stage_metrics.items()},
                            Winner=winner,
                            metrics=asdict(game_metrics)
                        ),
                        game_number
                    )

                    if self.visualizer:
                        self.visualizer.draw_game_over(f"Winner: {winner}", self.stats)
                        pygame.time.wait(1000)

                    return winner

        except Exception as e:
            logging.error(f"Error during game play: {e}")
            return "Error"

    def simulate(self) -> SimulationStats:
        """Run the complete simulation with enhanced metrics tracking."""
        try:
            for i in range(self.num_games):
                logging.info(f"Starting game {i + 1}/{self.num_games}")

                game_instance = Game(player1=AugmentMinimaxAgent5('X', max_depth=1), player2=MinimaxAgent('O', max_depth=1))
                winner = self.play_game(game_instance, i + 1)

                # Update statistics
                if winner == 'X':
                    self.stats.player1_wins += 1
                elif winner == 'O':
                    self.stats.player2_wins += 1
                elif winner == "Draw":
                    self.stats.draws += 1

                game_instance.reset()
                logging.info(f"Completed game {i + 1}/{self.num_games}")

        except Exception as e:
            logging.error(f"Simulation error: {e}")

        finally:
            if self.visualizer:
                self.visualizer.cleanup()

            # Save final session summary
            self.state_manager.save_session_summary(self.stats)
            self.log_final_results()

        return self.stats

    def log_final_results(self) -> None:
        """Log comprehensive final results and statistics."""
        logging.info("\nFinal Simulation Results:")
        logging.info(f"Total Games: {self.stats.total_games}")
        logging.info(f"Player 1 Wins: {self.stats.player1_wins}")
        logging.info(f"Player 2 Wins: {self.stats.player2_wins}")
        logging.info(f"Draws: {self.stats.draws}")
        logging.info(f"Average Game Duration: {self.stats.avg_game_duration} seconds")
        logging.info(f"Average Moves per Game: {self.stats.avg_moves_per_game}")


if __name__ == "__main__":
    simulator = GameSimulator(num_games=6, visualize=True,
                              save_path=Path("game_data"))
    stats = simulator.simulate()
