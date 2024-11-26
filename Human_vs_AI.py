import sys
from Game import Game, pygame
from Board import SQUARE_SIZE, HEIGHT, CIRCLE_RADIUS, positions, adjacent_points
from Algorithms.RandomAgent import RandomAgent
from Algorithms.Minimax import MinimaxAgent

# Modified WIDTH to include space for info panel
INFO_PANEL_WIDTH = 400  # Increased from 200 to give more breathing room
BOARD_WIDTH = 600  # Original game board width
WIDTH = BOARD_WIDTH + INFO_PANEL_WIDTH  # Total window width


class GameRunner:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT+100))
        pygame.display.set_caption('Morabaraba')

        # Colors for mode selection
        self.BACKGROUND_COLOR = (230, 230, 230)
        self.BUTTON_COLOR = (100, 150, 255)
        self.BUTTON_HOVER_COLOR = (150, 200, 255)
        self.TEXT_COLOR = (0, 0, 0)

        # Add colors for selection and highlighting
        self.SELECTED_COLOR = (255, 255, 0)  # Yellow highlight for selected piece
        self.VALID_MOVE_COLOR = (0, 255, 0, 128)  # Semi-transparent green for valid moves
        self.INFO_BG_COLOR = (255, 255, 255)  # Semi-transparent white for info panel

        # Initialize game state
        self.game = None
        self.mode = None  # Will store 'human' or 'ai'
        self.clock = pygame.time.Clock()

    def draw_mode_selection(self):
        """Draw the mode selection screen"""
        self.screen.fill(self.BACKGROUND_COLOR)

        # Title
        font_title = pygame.font.Font(None, 74)
        title = font_title.render("Morabaraba", True, self.TEXT_COLOR)
        title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 4))

        # Subtitle
        font_subtitle = pygame.font.Font(None, 36)
        subtitle = font_subtitle.render("Choose Game Mode", True, self.TEXT_COLOR)
        subtitle_rect = subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 4 + 80))

        # Buttons
        font_button = pygame.font.Font(None, 48)

        human_button = font_button.render("Human vs Human", True, self.TEXT_COLOR)
        human_rect = human_button.get_rect(center=(WIDTH // 2, HEIGHT // 2))

        ai_button = font_button.render("Human vs AI", True, self.TEXT_COLOR)
        ai_rect = ai_button.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100))

        # Draw elements
        self.screen.blit(title, title_rect)
        self.screen.blit(subtitle, subtitle_rect)
        self.screen.blit(human_button, human_rect)
        self.screen.blit(ai_button, ai_rect)

        pygame.display.update()

        return human_rect, ai_rect

    def handle_mode_selection(self):
        """Handle mode selection screen events"""
        human_rect, ai_rect = self.draw_mode_selection()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if human_rect.collidepoint(mouse_pos):
                        self.mode = 'human'
                        return
                    elif ai_rect.collidepoint(mouse_pos):
                        self.mode = 'ai'
                        return

    def initialize_game(self):
        """Initialize game based on selected mode"""
        self.game = Game()

        if self.mode == 'human':
            # For human multiplayer, replace AI agents with human-controlled players
            from Algorithms.Player import Player
            self.game.player1 = Player('X')
            self.game.player2 = Player('O')
        elif self.mode == 'ai':
            # Keep existing AI setup
            # self.game.player2 = RandomAgent('O')
            # Option to uncomment and choose different AI difficulty
            self.game.player2 = MinimaxAgent('O', max_depth=2)

        self.game.current_player = self.game.player1
        self.game.player1.opp = self.game.player2
        self.game.player2.opp = self.game.player1

        self.game.screen = self.screen
        self.game.human = True

    def draw_game_info(self):
        """Draw game information panel"""
        # Position the info panel to the right of the game board
        info_surface = pygame.Surface((INFO_PANEL_WIDTH, 200))
        info_surface.fill((255, 255, 255))
        info_surface.set_alpha(180)
        self.screen.blit(info_surface, (BOARD_WIDTH, 10))
        Stage = {1: 'Placement', 2: 'Movement', 3: 'Flying/Jumping'}

        font = pygame.font.Font(None, 24)
        info_texts = [
            f"Player 1 (X): {self.game.player1.pieces} pieces left to place",
            f"Player 2 (O): {self.game.player2.pieces} pieces left to place",
            f"Player 1 (X): {self.game.player1.pieces_usable} pieces that are usable",
            f"Player 2 (O): {self.game.player2.pieces_usable} pieces that are usable",
            f"Current Stage: {Stage[self.game.current_player.stage]}",
            f"Current Player: {self.game.current_player.piece}",
            "Game Mode: Remove Piece" if self.game.removal_mode else "Game Mode: Normal"
        ]

        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (BOARD_WIDTH + 10, 20 + i * 25))

    def draw_selected_piece(self):
        """Highlight the selected piece and show valid moves"""
        if hasattr(self.game, 'selected_piece'):
            row, col = self.game.selected_piece
            center_pos = (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                          row * SQUARE_SIZE + SQUARE_SIZE // 2)

            # Draw selection circle
            pygame.draw.circle(self.screen, self.SELECTED_COLOR,
                               center_pos, CIRCLE_RADIUS + 8, 3)

            # Highlight valid moves
            if self.game.current_player.stage == 2:
                # Show adjacent points for stage 2
                for adj_row, adj_col in adjacent_points[(row, col)]:
                    if self.game.Board.is_square_empty(adj_row, adj_col):
                        pygame.draw.circle(self.screen, self.VALID_MOVE_COLOR,
                                           (adj_col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                            adj_row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                           CIRCLE_RADIUS + 5)
            elif self.game.current_player.stage == 3:
                # Show all empty positions for stage 3
                for pos_row, pos_col in positions:
                    if self.game.Board.is_square_empty(pos_row, pos_col):
                        pygame.draw.circle(self.screen, self.VALID_MOVE_COLOR,
                                           (pos_col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                            pos_row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                           CIRCLE_RADIUS + 5)

    def handle_events(self):
        """Handle pygame events and return True if game should quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            if self.mode == 'ai':

                if event.type == pygame.MOUSEBUTTONDOWN and self.game.current_player == self.game.player1:
                    mouseX, mouseY = event.pos
                    # Only process clicks within the game board area
                    if mouseX < BOARD_WIDTH:
                        clicked_row = mouseY // SQUARE_SIZE
                        clicked_col = mouseX // SQUARE_SIZE
                        self.handle_player_move(clicked_row, clicked_col)

                # Handle AI moves when it's player2's turn and in AI mode
                if self.game.current_player == self.game.player2 and not self.game.game_over:
                    self.handle_ai_move()

            else:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouseX, mouseY = event.pos
                    # Only process clicks within the game board area
                    if mouseX < BOARD_WIDTH:
                        clicked_row = mouseY // SQUARE_SIZE
                        clicked_col = mouseX // SQUARE_SIZE
                        self.handle_player_move(clicked_row, clicked_col)

        return False

    def handle_player_move(self, row, col):
        """Process a player's move"""
        try:
            self.game.handle_click(row, col)
            self.game.draw_board()
            self.draw_selected_piece()
            self.draw_game_info()
            pygame.display.update()
            print(f"Move made at: {row}, {col}")
        except Exception as e:
            print(f"Error processing move: {e}")

    def handle_ai_move(self):
        """Handle AI player's move"""
        try:
            move = self.game.player2.make_move(self.game)
            if move is not None:
                if self.game.removal_mode:
                    self.game.handle_removal(*move)
                elif self.game.player2.stage == 1:
                    self.game.handle_placement(*move)
                else:
                    from_pos, to_pos = move
                    self.game.handle_movement(*from_pos, *to_pos)

                self.game.draw_board()
                self.draw_game_info()
                pygame.display.update()
        except Exception as e:
            print(f"Error processing AI move: {e}")

    def check_game_state(self):
        """Check if game is over and handle end game state"""
        if not self.game.game_over:
            winner = self.game.check_winner()
            if winner != "In Progress":
                self.game.game_states['Winner'] = winner
                self.game.game_over = True
                return True
        return False

    def render_game_over(self):
        """Display game over screen"""
        winner = self.game.game_states['Winner']

        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT+100))
        overlay.fill((255, 255, 255))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))

        # Game Over text
        font = pygame.font.Font(None, 74)
        game_over_text = font.render("Game Over", True, (255, 0, 0))
        game_over_rect = game_over_text.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 50))

        # Winner text
        winner_text = font.render(
            f"{'Draw!' if winner == 'Draw' else f'Player {winner} Wins!'}",
            True, (0, 0, 0))
        winner_rect = winner_text.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 50))

        # Exit instruction text
        font_small = pygame.font.Font(None, 36)
        exit_text = font_small.render("Press QUIT to exit", True, (0, 0, 0))
        exit_rect = exit_text.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 150))

        # Draw all text elements
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(winner_text, winner_rect)
        self.screen.blit(exit_text, exit_rect)

    def run(self):
        """Main game flow with mode selection"""
        try:
            # First show mode selection screen
            self.handle_mode_selection()

            # Initialize game based on selected mode
            self.initialize_game()

            # Game loop
            while True:
                if self.handle_events():
                    break

                self.game.draw_board()
                self.draw_selected_piece()
                self.draw_game_info()

                if self.check_game_state():
                    self.render_game_over()
                    pygame.display.update()
                    pygame.time.wait(2000)  # Wait 2 seconds before showing exit screen
                    while True:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()

                pygame.display.update()
                self.clock.tick(30)

        except Exception as e:
            print(f"Fatal error in game loop: {e}")
        finally:
            pygame.quit()
            sys.exit()


def main():
    try:
        runner = GameRunner()
        runner.run()
    except Exception as e:
        print(f"Error initializing game: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
