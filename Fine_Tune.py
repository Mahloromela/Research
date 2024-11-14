import json
import math
import os
from datetime import datetime
from glob import glob

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.utils import to_categorical  # Correct import path
# from Board import mills, adjacent_points
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Neural_network import save_model_with_unique_name, NNAgent
from Preprocessing import MorabarabaDataProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# [Include all corrected functions and the MorabarabaDataProcessor class here]


def reshape_for_conv2d(X_data, board_size):
    return np.array(X_data).reshape(-1, 33, 1)


def load_most_recent_model(base_name):
    # Search for all model files that match the base name pattern
    model_files = glob(rf'/home/vmuser/Pictures/Code/Morabaraba/Training5/Models_to_evaluate/{base_name}*.keras')

    if not model_files:
        print(f"No models found with base name: {base_name}")
        return None

    # Sort model files by their modification time (most recent first)
    model_files.sort(key=os.path.getmtime, reverse=True)

    # The most recent model will be the first one after sorting
    most_recent_model = model_files[0]
    print(f'Loading most recent model: {most_recent_model}')

    # Load and return the most recent model
    return tf.keras.models.load_model(most_recent_model)


def evaluate_value_model(X_value, v_true, model, stage_name, board_size=33):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = rf"/home/vmuser/Pictures/Code/Morabaraba/Training5/Confusion Matrix/Prediction_loss_{timestamp}.png"
    X_value_reshaped = reshape_for_conv2d(X_value, board_size)
    v_pred = model.predict(X_value_reshaped)

    mse = mean_squared_error(v_true, v_pred)
    mae = mean_absolute_error(v_true, v_pred)
    print(f'{stage_name} Value Model MSE:', mse)
    print(f'{stage_name} Value Model MAE:', mae)

    plt.scatter(v_true, v_pred, alpha=0.5)
    plt.title(f'{stage_name} Value Model: Predicted vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.savefig(filename)


def evaluate_individual_predictions(X_test, v_test, model, stage_name):
    close_count = 0
    total_predictions = len(X_test)

    for index, test_data in enumerate(X_test):
        test_data_reshaped = test_data.reshape(1, 33, 1)
        if stage_name == 'Movement':
            predicted_value = model.movement_predict_value(test_data_reshaped)
        elif stage_name == 'Placement':
            predicted_value = model.placement_predict_value(test_data_reshaped)
        else:
            predicted_value = model.jumping_predict_value(test_data_reshaped)

        true_value = v_test[index]
        is_close = math.isclose(predicted_value, true_value, rel_tol=0.15, abs_tol=0.05)

        print(f"Prediction {index + 1}:")
        print(f"Predicted value: {predicted_value}")
        print(f"True value: {true_value}")
        print(f"The values are close: {is_close}")
        print("--------------------")

        if is_close:
            close_count += 1

    close_percentage = (close_count / total_predictions) * 100
    print(f"{stage_name} - Close predictions: {close_count}/{total_predictions} ({close_percentage:.2f}%)")

    return close_count


def plot_accuracy_loss(history, Heading):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Generate a unique filename using the current timestamp
    filename = rf"/home/vmuser/Pictures/Code/Morabaraba/Training5/Loss/LossTraining_loss_{timestamp}.png"
    # Plot training history
    plt.figure(figsize=(12, 5))

    # Loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss {Heading}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)


def main():
    all_X_placement, all_y_placement = [], []
    all_X_movement, all_y_movement = [], []
    all_X_jumping, all_y_jumping = [], []
    # Scale the features
    scaler = StandardScaler()
    processor = MorabarabaDataProcessor()
    # Load game states
    for file_name in os.listdir(path=r'/home/vmuser/Pictures/Code/Morabaraba/Data3'):
        if file_name.endswith(".json"):
            try:
                game_states = processor.load_game_states(
                rf'/home/vmuser/Pictures/Code/Morabaraba/Data3/{file_name}')
            except Exception as e:
                print(f"Error loading state : {e}")
                continue

            X_placement, y_placement, X_movement, y_movement, X_jumping, y_jumping = processor.process_game_data(
                game_states)

            all_X_placement.extend(np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten()
                                             for row in X_placement]))
            all_y_placement.extend(y_placement)

            all_X_movement.extend(np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten()
                                            for row in X_movement]))
            all_y_movement.extend(y_movement)

            all_X_jumping.extend(np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten()
                                           for row in X_jumping]))
            all_y_jumping.extend(y_jumping)

    # Split the data
    X_train_placement, X_test_placement, v_train_placement, v_test_placement = train_test_split(
        np.array(all_X_placement), np.array(all_y_placement), test_size=0.01, random_state=42,
        shuffle=True
    )

    X_train_movement, X_test_movement, v_train_movement, v_test_movement = train_test_split(
        np.array(all_X_movement), np.array(all_y_movement), test_size=0.01, random_state=42,
        shuffle=True
    )

    X_train_jumping, X_test_jumping, v_train_jumping, v_test_jumping = train_test_split(
        np.array(all_X_jumping), np.array(all_y_jumping), test_size=0.01, random_state=42,
        shuffle=True
    )

    print("Shape of X_train_placement:", np.shape(X_train_placement))
    print("Shape of v_train_placement:", np.shape(v_train_placement))
    print("Shape of X_train_movement:", np.shape(X_train_movement))
    print("Shape of v_train_movement:", np.shape(v_train_movement))
    print("Shape of X_train_jumping:", np.shape(X_train_jumping))
    print("Shape of v_train_jumping:", np.shape(v_train_jumping))

    agent = NNAgent()
    agent.placement_value_model = load_most_recent_model('placement_value_model')
    agent.movement_value_model = load_most_recent_model('movement_value_model')
    agent.jumping_value_model = load_most_recent_model('jumping_value_model')

    # fit models
    agent.train(
        X_train_placement, v_train_placement,
        X_train_movement, v_train_movement,
        X_train_jumping, v_train_jumping
    )

    agent.evaluate_placement(X_test_placement, v_test_placement)#     main()

    agent.evaluate_movement(X_test_movement, v_test_movement)
    agent.evaluate_jumping(X_test_jumping, v_test_jumping)

    # # Plot History
    # plot_accuracy_loss(agent.history_placement, 'Placement(Stage 1)')
    # plot_accuracy_loss(agent.history_movement, "Movement(Stage 2)")
    # plot_accuracy_loss(agent.history_jumping, 'Jumping(Stage 3)')

    # save_model_with_unique_name(agent.placement_value_model, 'placement_value_model')
    # save_model_with_unique_name(agent.movement_value_model, 'movement_value_model')
    # save_model_with_unique_name(agent.jumping_value_model, 'jumping_value_model')

    print("Evaluating placement value model...")
    evaluate_value_model(X_test_placement, v_test_placement, agent.placement_value_model, 'Placement', 33)
    # evaluate_individual_predictions(X_test_placement, v_test_placement, agent, 'Placement')

    print("Evaluating movement value model...")
    evaluate_value_model(X_test_movement, v_test_movement, agent.movement_value_model, 'Movement', 33)
    # evaluate_individual_predictions(X_test_movement, v_test_movement, agent, 'Movement')

    print("Evaluating jumping value model...")
    evaluate_value_model(X_test_jumping, v_test_jumping, agent.jumping_value_model, 'Jumping', 33)
    # evaluate_individual_predictions(X_test_jumping, v_test_jumping, agent, 'Jumping')


if __name__ == "__main__":
    main()


