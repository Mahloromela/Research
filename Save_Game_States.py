import os
import json


def save_game_states(game_states, filename):
    # Check if the filename already exists
    if os.path.exists(filename):
        # If the filename exists, append a suffix until a unique filename is found
        i = 1
        while True:
            new_filename = rf"/home/vmuser/Pictures/Code/Morabaraba/Test/{filename}_{i}.json"
            if not os.path.exists(new_filename):
                filename = new_filename
                break
            i += 1

    # Write the game states to the new filename
    with open(filename, 'w') as f:
        json.dump(game_states, f)
