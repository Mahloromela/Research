import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import json
import numpy as np

from Research.Board import positions
from Research.Algorithms.Neural_network import NNAgent
from Research.Preprocessing import configuration_count, blockades, central_control, mobility, \
    blocked_potential_mills, count_mills, count_potential_mills, piece_stability


class MorabarabaModelAnalyzer:
    def __init__(self, model, feature_names=None):
        """
        Initialize analyzer with a trained model and feature names

        Args:
            model: Trained Keras model
            feature_names: List of feature names corresponding to input features
        """
        self.model = model
        self.feature_names = feature_names if feature_names else [
                                                                     'piece_diff', 'closed_mills_diff',
                                                                     'potential_mills_diff',
                                                                     'blocked_mills_diff', 'configuration_diff',
                                                                     'mobility_diff',
                                                                     'central_control_diff', 'blockades_diff',
                                                                     'stability_diff'
                                                                 ] + [f'pos_{i}' for i in
                                                                      range(24)]  # 24 board positions

    def analyze_input_distribution(self, input_data, output_path=None):
        """
        Analyze the distribution of input features

        Args:
            input_data: Input data array
            output_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(5, 7, figsize=(20, 15))
        axes = axes.ravel()

        for i in range(min(len(self.feature_names), input_data.shape[1])):
            ax = axes[i]
            sns.histplot(data=input_data[:, i], ax=ax, bins=30)
            ax.set_title(self.feature_names[i])
            ax.set_xlabel('')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def get_feature_importance(self, input_data):
        """
        Calculate feature importance using gradient-based approach

        Args:
            input_data: Input data array

        Returns:
            Dictionary of feature importance scores
        """
        # Create a gradient tape model
        gradient_model = Model(
            inputs=self.model.inputs,
            outputs=self.model.outputs
        )

        importance_scores = []

        # Calculate gradients for each sample
        for sample in input_data:
            sample_reshaped = np.reshape(sample, (1, -1))
            with tf.GradientTape() as tape:
                inputs = tf.convert_to_tensor(sample_reshaped, dtype=tf.float32)
                tape.watch(inputs)
                predictions = gradient_model(inputs)

            # Get gradients
            gradients = tape.gradient(predictions, inputs)
            importance = np.abs(gradients.numpy()[0])
            importance_scores.append(importance)

        # Average importance across all samples
        mean_importance = np.mean(importance_scores, axis=0)

        # Create feature importance dictionary
        importance_dict = {
            feature: score for feature, score in zip(self.feature_names, mean_importance)
        }

        return importance_dict

    def plot_feature_importance(self, importance_dict, top_k=10, output_path=None):
        """
        Plot feature importance

        Args:
            importance_dict: Dictionary of feature importance scores
            top_k: Number of top features to display
            output_path: Optional path to save the plot
        """
        # Sort features by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]

        features, scores = zip(*sorted_features)

        plt.figure(figsize=(12, 6))
        plt.bar(features, scores)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top Feature Importance')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    def analyze_decision_path(self, input_sample):
        """
        Analyze the decision path for a single input sample

        Args:
            input_sample: Single input sample to analyze

        Returns:
            Dictionary containing layer activations and their contributions
        """
        # Create models for intermediate layers
        layer_models = []
        for i, layer in enumerate(self.model.layers):
            if 'dense' in layer.name or 'multiply' in layer.name:
                layer_model = Model(
                    inputs=self.model.inputs,
                    outputs=layer.output
                )
                layer_models.append((layer.name, layer_model))

        # Get activations for each layer
        activations = {}
        input_reshaped = np.reshape(input_sample, (1, -1))

        for layer_name, layer_model in layer_models:
            activation = layer_model.predict(input_reshaped)
            activations[layer_name] = {
                'values': activation,
                'shape': activation.shape,
                'mean': np.mean(activation),
                'std': np.std(activation)
            }

        return activations


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

    def board_to_numpy_for_player_1(self, board, pieces_usable=0):
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

        player1_stability = piece_stability(board, 'X')
        player2_stability = piece_stability(board, 'O')

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

    def process_game_data(self, game_states):
        X_placement, X_movement, X_jumping = [], [], []

        # Parse the game states
        stages = game_states.get('states', {})
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
                # Only include even indices
                if i % 2 == 0:
                    revert_factor = 1  # You can adjust this based on your game logic
                    try:
                        if stage == 1:
                            current_state = self.board_to_numpy_for_player_1(state)
                        else:
                            current_state = self.board_to_numpy_for_player_1(state)
                    except Exception as e:
                        print(f"Error processing state {i} in stage {stage}: {e}")

                    moves_left = total_moves - processed_moves - i

                    if stage == 1:
                        X_placement.append(current_state)
                    elif stage == 2:
                        X_movement.append(current_state)
                    elif stage == 3:
                        X_jumping.append(current_state)

                    move_no += 1  # Increment move counter

        return np.array(X_placement), np.array(X_movement), np.array(X_jumping)


# Example usage
data_processor = MorabarabaDataProcessor()

# Load game data from the JSON file
game_data = ''
# Iterate through the files and load the first `.json` file
for file_name in os.listdir(r'C:\Users\seabi\Downloads\Research\Morabaraba\Data7'):
    if file_name.endswith(".json"):
        file_path = rf'C:\Users\seabi\Downloads\Research\Morabaraba\Data7\game_states_109.json'
        game_data = data_processor.load_game_states(file_path)
        break
# Process the game data, returning only states at even indices
X_placement, X_movement, X_jumping = data_processor.process_game_data(game_data)

print("X_placement states (even indices):", X_placement)
print("X_movement states (even indices):", X_movement)
print("X_jumping states (even indices):", X_jumping)

# # First create mock data using the preprocessor
# mock_game_states = {
#     'Winner': 'O',
#     'states': {
#         '1': [
#             [['X', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', 'O', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', 'O']],
#
#             [['X', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', 'X', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', 'O', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', 'O']]
#         ],
#         '2': [
#             [['X', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', 'X', 'O', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', 'O', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', 'O', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', 'O']],
#
#             [['X', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', 'X', ' ', 'O', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', 'O']]
#         ],
#         '3': [
#             [['X', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', 'X', ' ', 'O', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', ' '],
#              [' ', ' ', ' ', ' ', ' ', ' ', 'O']]
#         ]
#     },
#     'pieces_usable': [9, 12]
# }

# # Process the data
# processor = MorabarabaDataProcessor()
# (X_placement, v_placement,
#  X_movement, v_movement,
#  X_jumping, v_jumping) = processor.process_game_data(mock_game_states)


agent = NNAgent()
# Create analyzer
analyzer = MorabarabaModelAnalyzer(agent.placement_value_model)

# 1. Analyze input distribution for each stage
print("Analyzing input distributions...")
analyzer.analyze_input_distribution(X_placement, 'placement_distribution.png')
analyzer.analyze_input_distribution(X_movement, 'movement_distribution.png')
analyzer.analyze_input_distribution(X_jumping, 'jumping_distribution.png')

# 2. Get and plot feature importance for each stage
print("\nAnalyzing feature importance...")

# Placement stage
placement_importance = analyzer.get_feature_importance(X_placement)
analyzer.plot_feature_importance(placement_importance, output_path='placement_importance.png')
print("\nTop 20 important features for Placement stage:")
sorted_placement = sorted(placement_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
for feature, importance in sorted_placement:
    print(f"{feature}: {importance:}")

# Movement stage
movement_importance = analyzer.get_feature_importance(X_movement)
analyzer.plot_feature_importance(movement_importance, output_path='movement_importance.png')
print("\nTop 20 important features for Movement stage:")
sorted_movement = sorted(movement_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
for feature, importance in sorted_movement:
    print(f"{feature}: {importance:}")

# Jumping stage
jumping_importance = analyzer.get_feature_importance(X_jumping)
analyzer.plot_feature_importance(jumping_importance, output_path='jumping_importance.png')
print("\nTop 20 important features for Jumping stage:")
sorted_jumping = sorted(jumping_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
for feature, importance in sorted_jumping:
    print(f"{feature}: {importance:}")

# 3. Analyze decision path for a sample from each stage
print("\nAnalyzing decision paths...")

# Analyze first state from each stage
placement_path = analyzer.analyze_decision_path(X_placement[0])
movement_path = analyzer.analyze_decision_path(X_movement[0])
jumping_path = analyzer.analyze_decision_path(X_jumping[0])

print("\nPlacement Stage Decision Path:")
for layer_name, info in placement_path.items():
    print(f"\nLayer: {layer_name}")
    print(f"Mean activation: {info['mean']:}")
    print(f"Std activation: {info['std']:}")

print("\nMovement Stage Decision Path:")
for layer_name, info in movement_path.items():
    print(f"\nLayer: {layer_name}")
    print(f"Mean activation: {info['mean']:}")
    print(f"Std activation: {info['std']:}")

print("\nJumping Stage Decision Path:")
for layer_name, info in jumping_path.items():
    print(f"\nLayer: {layer_name}")
    print(f"Mean activation: {info['mean']:}")
    print(f"Std activation: {info['std']:}")


def get_feature_importance(model, X_input):
    """
    Computes feature importance scores for each feature by averaging the absolute gradients across the samples.

    Parameters:
    - model: The Keras model instance.
    - X_input: The input data for which we want to compute feature importance.

    Returns:
    - A list of feature importance scores.
    """
    X_input = tf.convert_to_tensor(X_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_input)
        predictions = model(X_input)

    # Compute the gradients of the predictions with respect to the input features
    gradients = tape.gradient(predictions, X_input)

    # Take the mean of the absolute gradients across all samples
    feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0)

    return feature_importance.numpy()


# Example usage:
# Assuming `X_placement` is your input data for placement stage

feature_importance_placement = get_feature_importance(agent.placement_value_model, X_placement)
print("Feature Importance for Placement Stage:", feature_importance_placement)

# Repeat for movement and jumping stages if needed
feature_importance_movement = get_feature_importance(agent.movement_value_model, X_movement)
feature_importance_jumping = get_feature_importance(agent.jumping_value_model, X_jumping)
print("Feature Importance for Movement Stage:", feature_importance_movement)
print("Feature Importance for Jumping Stage:", feature_importance_jumping)


def get_feature_importance(model, X_input):
    """
    Computes feature importance scores for each feature by averaging the absolute gradients across the samples.

    Parameters:
    - model: The Keras model instance.
    - X_input: The input data for which we want to compute feature importance.

    Returns:
    - A list of feature importance scores.
    """
    X_input = tf.convert_to_tensor(X_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_input)
        predictions = model(X_input)

    # Compute the gradients of the predictions with respect to the input features
    gradients = tape.gradient(predictions, X_input)

    # Take the mean of the absolute gradients across all samples
    feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0)

    return feature_importance.numpy()



def plot(filename, non_position_features, position_features):
    # Plot non-position feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(non_position_features)), non_position_features)
    plt.xticks(range(len(non_position_features)), [
        'piece_diff', 'closed_mills_diff',
        'potential_mills_diff',
        'blocked_mills_diff', 'configuration_diff',
        'mobility_diff',
        'central_control_diff', 'blockades_diff',
        'stability_diff'],
               rotation=45,
               ha='right')
    plt.title(f'Non-Position {filename} Feature Importance')
    plt.tight_layout()
    plt.savefig(f'./Features/non_position_{filename}_feature_importance.png')

    # Print position feature importance
    print("Position Feature Importance:", position_features)


feature_importance_placement = get_feature_importance(agent.placement_value_model, X_placement)
# Extract non-position feature importance
non_position_features = feature_importance_placement[:-24]
position_features = feature_importance_placement[-24:]
plot('Placement', non_position_features, position_features)

# Repeat for movement and jumping stages if needed
feature_importance_movement = get_feature_importance(agent.movement_value_model, X_movement)
# Extract non-position feature importance
non1_position_features = feature_importance_movement[:-24]
position_features1 = feature_importance_movement[-24:]
plot('Movement', non1_position_features, position_features)
feature_importance_jumping = get_feature_importance(agent.jumping_value_model, X_jumping)
# Extract non-position feature importance
non2_position_features = feature_importance_jumping[:-24]
position_features2 = feature_importance_jumping[-24:]
plot('Jumping', non2_position_features, position_features)

Overall_non_position_features = (non_position_features + non1_position_features + non2_position_features) / 3
Overall_position_features = (position_features + position_features1 + position_features2) / 3

plot('Overall Phases', Overall_non_position_features, position_features)
