import copy


class Player:
    def __init__(self, piece='X'):
        self.piece = piece
        self.pieces = 12
        self.pieces_usable = 12
        self.stage = 1
        self.opp = None
        self.fromPos = None
        self.my_pieces = set()
        self.last_move = ()
        self.move_history = []

    def update_stage(self):
        if self.pieces > 0:
            self.stage = 1
        elif self.pieces == 0 and self.pieces_usable > 3:
            self.stage = 2
        else:
            self.stage = 3

    def copy(self):
        new_player = Player(self.piece)
        new_player.stage = self.stage
        new_player.pieces_usable = self.pieces_usable
        new_player.pieces = self.pieces
        new_player.my_pieces = self.my_pieces.copy()
        new_player.move_history = copy.deepcopy(self.move_history)
        return new_player
