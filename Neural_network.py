import os
from datetime import datetime
from glob import glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Multiply, Lambda, Input, BatchNormalization
import keras.backend as K
from keras.optimizers import SGD 
from keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras.regularizers import l1


class MonitorWeightsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print attention weights for each epoch
        for layer in self.model.layers:
            if 'dense' in layer.name and 'attention' in layer.name:  # Adjust as needed
                weights = layer.get_weights()[0]
                sparsity = (weights == 0).mean()  # Calculate sparsity ratio
                print(f"Epoch {epoch + 1} - Layer {layer.name} sparsity: {sparsity:.2%}")
                print(weights)  # Print weights, or you can log them for detailed analysis


class NNAgent:
    def __init__(self):
        self.history_placement = None
        self.history_movement = None
        self.history_jumping = None
        self.placement_value_model = self.create_value_model()
        self.placement_value_model.summary()
        self.movement_value_model = self.create_value_model()
        self.movement_value_model.summary()
        self.jumping_value_model = self.create_value_model()
        self.jumping_value_model.summary()

    def create_value_model(self):
        # Input layer
        inputs = Input(shape=(31,))

        # Input attention mechanism
        attention_scores = Dense(31, activation='softmax')(inputs)
        attention_output = Multiply()([inputs, attention_scores])
        attention_output = BatchNormalization()(attention_output)

        # First layer
        x = Dense(128, activation='relu', activity_regularizer=l1(1e-4))(attention_output)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        # First attention mechanism
        attention_scores = Dense(128, activation='softmax')(x)
        attention_output = Multiply()([x, attention_scores])
        attention_output = BatchNormalization()(attention_output)

        # Second layer
        x = Dense(64, activation='relu', activity_regularizer=l1(1e-4))(attention_output)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        # Second attention mechanism
        attention_scores = Dense(64, activation='softmax')(x)
        attention_output = Multiply()([x, attention_scores])
        attention_output = BatchNormalization()(attention_output)

        # Third layer
        x = Dense(32, activation='relu', activity_regularizer=l1(1e-4))(attention_output)
        # x = Dropout(0.2)(x)

        attention_scores = Dense(32, activation='softmax')(x)
        attention_output = Multiply()([x, attention_scores])
        attention_output = BatchNormalization()(attention_output)

        # Fourth layer
        x = Dense(16, activation='relu')(attention_output)
        #x = Dropout(0.2)(x)
        #x = BatchNormalization()(x)

        # Fourth attention mechanism
        attention_scores = Dense(16, activation='softmax')(x)
        attention_output = Multiply()([x, attention_scores])

        # Output layer
        outputs = Dense(1, activation='linear')(x)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])

        return model

    def train(self, X_placement, v_placement, X_movement, v_movement, X_jumping,
              v_jumping):
        monitor_weights_callback = MonitorWeightsCallback()

        # Create ModelCheckpoint callback for each model
        placement_checkpoint = ModelCheckpoint(
            filepath=save_model_with_unique_name('placement_value_model'),  # Filepath to save the model
            monitor='val_loss',  # Metric to monitor
            save_best_only=True,  # Save only the best model
            mode='min',  # 'min' for loss, 'max' for accuracy
            verbose=1
        )

        movement_checkpoint = ModelCheckpoint(
            filepath=save_model_with_unique_name('movement_value_model'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        jumping_checkpoint = ModelCheckpoint(
            filepath=save_model_with_unique_name('jumping_value_model'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        print("Training placement neural network")
        X_train_placement, val_placement, y_train_placement, y_val_placement = train_test_split(
            X_placement, v_placement, test_size=0.1, random_state=42,
            shuffle=True
        )
        # Create the optimizer with momentum
        momentum_value = 0.82 
        optimizer = SGD(learning_rate=0.001, momentum=momentum_value)  # Adjust the learning rate as needed

        # Compile the model with the new optimizer
        self.placement_value_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        self.history_placement = self.placement_value_model.fit(X_train_placement, y_train_placement, epochs=100,
                                                                validation_data=(val_placement, y_val_placement),
                                                                callbacks=[monitor_weights_callback,placement_checkpoint], batch_size=10)

        # print("Training movement neural network")
        # X_train_movement, val_movement, y_train_movement, y_val_movement = train_test_split(
        #     X_movement, v_movement, test_size=0.1, random_state=42,
        #     shuffle=True
        # )
        # self.history_movement = self.movement_value_model.fit(X_train_movement, y_train_movement, epochs=100,
        #                                                       validation_data=(val_movement, y_val_movement),
        #                                                       callbacks=[monitor_weights_callback, movement_checkpoint])

        print("Training jumping neural network")
        X_train_jumping, val_jumping, y_train_jumping, y_val_jumping = train_test_split(
            X_jumping, v_jumping, test_size=0.1, random_state=42,
            shuffle=True
        )
        self.history_jumping = self.jumping_value_model.fit(X_train_jumping, y_train_jumping, epochs=50,
                                                            validation_data=(val_jumping, y_val_jumping),
                                                            callbacks=[monitor_weights_callback, jumping_checkpoint], batch_size=10)

    def train_placement(self, X_placement, v_placement):
        print("Training placement neural network")
        self.placement_value_model.fit(X_placement, v_placement, epochs=30, validation_split=0.2)

    def train_movement(self, X_movement, v_movement):
        print("Training movement neural network")
        self.movement_value_model.fit(X_movement, v_movement, epochs=80, validation_split=0.2)

    def train_jumping(self, X_jumping, v_jumping):
        print("Training jumping neural network")
        self.jumping_value_model.fit(X_jumping, v_jumping, epochs=50, validation_split=0.2)

    def evaluate_placement(self, X_test, y_test):
        # Evaluate the model
        placement_test_loss, placement_test_accuracy = self.placement_value_model.evaluate(X_test, y_test, verbose=2)
        print(f"Placement Test Accuracy: {placement_test_accuracy:.4f}")

    def evaluate_movement(self, X_test, y_test):
        # Evaluate the model
        movement_test_loss, movement_test_accuracy = self.movement_value_model.evaluate(X_test, y_test, verbose=2)
        print(f"Movement Test Accuracy: {movement_test_accuracy:.4f}")

    def evaluate_jumping(self, X_test, y_test):
        # Evaluate the model
        jumping_test_loss, jumping_test_accuracy = self.jumping_value_model.evaluate(X_test, y_test, verbose=2)
        print(f"Jumping Test Accuracy: {jumping_test_accuracy:.4f}")

    def placement_predict_value(self, state):
        return self.placement_value_model.predict(state, verbose=0)[0][0]

    def movement_predict_value(self, state):
        return self.movement_value_model.predict(state, verbose=0)[0][0]

    def jumping_predict_value(self, state):
        return self.jumping_value_model.predict(state, verbose=0)[0][0]


def save_model_with_unique_name(base_name):
    file_name = rf'/home/vmuser/Pictures/Code/Morabaraba/Training5/Models_to_evaluate/{base_name}.keras'

    if os.path.exists(file_name):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = rf'/home/vmuser/Pictures/Code/Morabaraba/Training5/Models_to_evaluate/{base_name}_{timestamp}.keras'

    return file_name
    # print(f'Model saved as: {file_name}')


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
