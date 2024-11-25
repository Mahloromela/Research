import math
import random
import time

from Research.Board import adjacent_points
from Research.Algorithms.Player import Player


class MCTSTreeNode:

    def __init__(self, game, parent=None, move=None, depth=0):
        self.game_state = game
        self.parent = parent
        self.move = move
        self.depth = depth
        self.children = []
        self.visits = 0
        self.score = 0
        self.is_terminal = game.check_winner() != 'In Progress'
        self.is_fully_expanded = False


class MCTSAgent(Player):
    def __init__(self, piece, simulation_time=1, max_iterations=300, exploration_constant=math.sqrt(2)):
        super().__init__(piece)
        self.exploration_constant = exploration_constant
        self.root = None
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.name = 'MCTS'

    def make_move(self, game):
        return self.search(game).move

    def search(self, initial_state):
        self.root = MCTSTreeNode(initial_state)
        self.expand(self.root)
        print('The initial length of root is', len(self.root.children))
        iterations = 0
        Total_time = 0

        while iterations < self.max_iterations:
            child_node = self.select(self.root)
            # leaf_node = self.expand(child_node)
            score, time_taken = self.rollout(child_node)

            self.backpropagate(child_node, score)
            Total_time += time_taken
            iterations += 1
        print(f'Iteration {iterations} Total Time taken is ', Total_time, 'seconds')

        best_child = self.select_best_child(self.root, exploration_constant=0.1)
        print('Child move is', best_child.move)
        return best_child

    def select(self, node):
        if not node.is_fully_expanded:
            self.expand(node)
        leaf_node = self.select_best_child(node, exploration_constant=0.1)
        return leaf_node

    def select_best_child(self, node, exploration_constant=math.sqrt(2)):

        return max(node.children, key=lambda c: self.uct_value(c, exploration_constant))

    def uct_value(self, node, exploration_constant):
        if node.visits == 0:
            return float('inf')
        exploitation = node.score / node.visits

        #exploitation = node.score
        exploration = exploration_constant * math.sqrt(math.log(node.parent.visits) / node.visits)
        return exploitation + exploration

    def expand(self, node):
        possible_moves = self.get_possible_moves(node.game_state.current_player, node.game_state)
        # random.shuffle(possible_moves)
        count = 0
        for move in possible_moves:
            # if not any(child.move == move for child in node.children):
            print('Expansion current player stage', node.game_state.current_player.stage)
            new_game_state = self.apply_move(node.game_state, move)
            new_node = MCTSTreeNode(new_game_state, parent=node, move=move, depth=node.depth + 1)
            node.children.append(new_node)
            if len(node.children) == len(possible_moves):
                node.is_fully_expanded = True
                break

            print('The depth of the new node is', new_node.depth, 'The count is', count)
            # return node
            count += 1
        node.is_fully_expanded = True
        return node

    def rollout(self, node):
        current_game_state = node.game_state.copy()
        max_moves = 20  # Prevent infinite loops
        move_count = 0
        start_time = time.time()
        while current_game_state.check_winner() == 'In Progress':
            possible_moves = self.get_possible_moves(current_game_state.current_player, current_game_state)
            # print('The length of possible moves is ', len(possible_moves))
            if len(possible_moves) == 0:
                break  # No more moves possible, end the rollout
            move = random.choice(possible_moves)
            current_game_state = self.apply_move(current_game_state, move)
            move_count += 1
        end_time = time.time()

        print('Time taken is ', end_time - start_time, 'seconds')
        return self.evaluate(current_game_state), end_time - start_time

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.score += result
            node = node.parent

    def evaluate(self, game):
        winner = game.check_winner()
        if winner == self.piece:
            return 1
        elif winner == "Draw":
            return 0.5
        else:
            return 0

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
                    valid_moves = list(game.free_positions)
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
