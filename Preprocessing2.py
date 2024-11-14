import json

import numpy as np

from Board import mills, adjacent_points, positions


def mobility(board, piece):
    moves = 0
    for r, c in positions:
        if board[r][c] == piece:
            for adj_r, adj_c in adjacent_points[(r, c)]:
                if board[adj_r][adj_c] == ' ':
                    moves += 1
    return moves


def blockades(board, opponent_piece):
    blocked = 0
    for r, c in positions:
        if board[r][c] == opponent_piece:
            if all(board[adj_r][adj_c] != ' ' for adj_r, adj_c in adjacent_points[(r, c)]):
                blocked += 1
    return blocked


central_positions = [(1, 1), (1, 3), (1, 5), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3),
                     (4, 4), (5, 1), (5, 3), (5, 5)]


def central_control(board, piece):
    return sum(1 for r, c in central_positions if board[r][c] == piece)


def connectedness(board, piece):
    # Simple measure: count the number of adjacent friendly pieces
    connections = 0
    for r, c in positions:
        if board[r][c] == piece:
            connections += sum(1 for adj_r, adj_c in adjacent_points[(r, c)] if board[adj_r][adj_c] == piece)
    return connections


def piece_stability(board, piece):
    stable = 0
    for r, c in positions:
        if board[r][c] == piece:
            # Define stability criteria, e.g., part of multiple mills or not adjacent to opponent pieces
            if count_mills(board, piece) > 3:
                stable += 1
    return stable


def count_mills(board, piece):
    mill_count = 0
    for mill in mills:
        if all(board[r][c] == piece for r, c in mill):
            mill_count += 1
    return mill_count


def configuration_count(board, piece):
    configuration_count = 0
    for r, c in positions:
        if board[r][c] == piece:
            # Check all adjacent points for the same piece
            if all(board[adj_r][adj_c] == piece for adj_r, adj_c in adjacent_points[(r, c)]):
                configuration_count += 1
    return configuration_count


def count_potential_mills(board, piece):
    potential_mills = 0
    for mill in mills:  # Assuming mills is a class attribute
        pieces = [board[r][c] for r, c in mill]
        if pieces.count(piece) == 2 and pieces.count(' ') == 1:
            empty_spot_coords = mill[pieces.index(' ')]
            if any([board[pos[0]][pos[1]] == piece for pos in adjacent_points[empty_spot_coords]]):
                potential_mills += 1
    return potential_mills


def blocked_potential_mills(board, piece1, piece2):
    potential_mills = 0
    for mill in mills:
        pieces = [board[r][c] for r, c in mill]
        if pieces.count(piece2) == 2 and pieces.count(piece1) == 1:
            current_player_spot_coords = mill[pieces.index(piece1)]
            if any([board[pos[0]][pos[1]] == piece2 for pos in adjacent_points[current_player_spot_coords]]):
                potential_mills += 1
    return potential_mills


def calculate_win_probability(winner, move_no, moves_left, total_moves):
    if winner == 'Draw':
        return 0.5
    player = 'X' if move_no % 2 == 0 else 'O'
    if winner == player:
        return 0.5 + 0.5 * ((total_moves - moves_left) / total_moves)
    else:
        return 0.5 - 0.5 * ((total_moves - moves_left) / total_moves)


def revert(move_no):
    return 1 if np.mod(move_no, 2) == 0 else -1


def board_to_numpy(board, pieces_usable=0):
    # Piece counts
    player1_pieces = sum(1 for row, col in positions if board[row][col] == 'X')
    player2_pieces = sum(1 for row, col in positions if board[row][col] == 'O')

    # Potential mills
    player1_potential_mills = count_potential_mills(board, 'X')
    player2_potential_mills = count_potential_mills(board, 'O')

    # Closed mills
    player1_closed_mills = count_mills(board, 'X')
    player2_closed_mills = count_mills(board, 'O')

    # Blocked mills
    player1_blocked_mills = blocked_potential_mills(board, 'X', 'O')
    player2_blocked_mills = blocked_potential_mills(board, 'O', 'X')

    # Configuration counts
    player1_configuration = configuration_count(board, 'X')
    player2_configuration = configuration_count(board, 'O')

    # Additional Features
    player1_mobility = mobility(board, 'X')
    player2_mobility = mobility(board, 'O')

    player1_central_control = central_control(board, 'X')
    player2_central_control = central_control(board, 'O')

    player1_blockades = blockades(board, 'O')
    player2_blockades = blockades(board, 'X')

    # player1_stability = piece_stability(board, 'X')
    # player2_stability = piece_stability(board, 'O')

    # Difference-based features
    piece_diff = player1_pieces - player2_pieces + pieces_usable
    closed_mills_diff = player1_closed_mills - player2_closed_mills
    potential_mills_diff = player1_potential_mills - player2_potential_mills
    blocked_mills_diff = player1_blocked_mills - player2_blocked_mills
    configuration_diff = player1_configuration - player2_configuration
    mobility_diff = player1_mobility - player2_mobility
    central_control_diff = player1_central_control - player2_central_control
    blockades_diff = player1_blockades - player2_blockades
    #stability_diff = player1_stability - player2_stability

    # Combine all features into a single array
    parameters = [
        piece_diff,
        closed_mills_diff,
        potential_mills_diff,
        blocked_mills_diff,
        configuration_diff,
        mobility_diff,
        central_control_diff,
        blockades_diff
        # stability_diff
    ]
    board_representation = [1 if board[row][col] == 'X' else (-1 if board[row][col] == 'O' else 0)
                            for row, col in positions]

    # Combine all features
    return np.concatenate((parameters, board_representation))


class MorabarabaDataProcessor:
    def __init__(self, board_size=7, valid_positions=None):
        self.board_size = board_size
        self.valid_positions = valid_positions if valid_positions else [
            (0, 0), (0, 3), (0, 6),
            (1, 1), (1, 3), (1, 5),
            (2, 2), (2, 3), (2, 4),
            (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6),
            (4, 2), (4, 3), (4, 4),
            (5, 1), (5, 3), (5, 5),
            (6, 0), (6, 3), (6, 6)
        ]
        self.position_map = {pos: idx for idx, pos in enumerate(self.valid_positions)}

    def load_game_states(self, filename):
        with open(filename, 'r') as f:
            return json.load(f)

    def process_game_data(self, game_states):
        X_placement, y_placement = [], []
        X_movement, y_movement = [], []
        X_jumping, y_jumping = [], []

        winner = game_states.get('Winner', 'Draw')
        stages = game_states.get('states', {})
        placement_pieces_usable = game_states.get('pieces_usable', [])
        total_moves = sum(len(states) for states in stages.values())

        move_no = 0  # Initialize move counter
        for stage_key, states in stages.items():
            try:
                stage = int(stage_key)
            except ValueError:
                print(f"Invalid stage key: {stage_key}. Skipping.")
                continue

            processed_moves = sum(len(stages.get(str(s), [])) for s in range(1, stage))

            for i, state in enumerate(states):
                revert_factor = revert(move_no)
                try:
                    if stage == 1:
                        current_state = board_to_numpy(state,
                                                       pieces_usable=placement_pieces_usable[i]*revert_factor)
                    else:
                        current_state = board_to_numpy(state)
                except Exception as e:
                    current_state = board_to_numpy(state)
                    print(f"Error processing state {i} in stage {stage}: {e}")


                moves_left = total_moves - processed_moves - i

                if stage == 1:
                    win_prob = calculate_win_probability(winner, move_no, moves_left, total_moves)
                    X_placement.append(current_state * revert_factor)
                    y_placement.append(win_prob)
                elif stage == 2:
                    win_prob = calculate_win_probability(winner, move_no, moves_left + 1, total_moves)
                    X_movement.append(current_state * revert_factor)
                    y_movement.append(win_prob)
                elif stage == 3:
                    win_prob = calculate_win_probability(winner, move_no, moves_left - 1, total_moves)
                    X_jumping.append(current_state * revert_factor)
                    y_jumping.append(win_prob)

                move_no += 1  # Increment move counter

        return (
            np.array(X_placement),
            np.array(y_placement),
            np.array(X_movement),
            np.array(y_movement),
            np.array(X_jumping),
            np.array(y_jumping)
        )
