import numpy as np 


class TicTacToe:

    # constructor
    def __init__(self, n_rows : int, n_cols : int) -> None :
        self.n_rows = 3
        self.n_cols = 3
        self.board = np.zeros((self.n_rows,self.n_cols))
        self.action_size = self.n_rows * self.n_cols

    def get_initial_state(self):
        return np.zeros((self.n_rows,self.n_cols))

    # state + action => next_state // action is from 0 to 8 // state is the board
    def get_next_state(self, state, action , player):
        state = state.copy()
        r = action//self.n_cols
        c = action%self.n_rows
        state[r,c] = player
        return state
    
    # valid moves
    def get_valid_moves(self,state):
        return (state.reshape(-1) == 0).astype(int)
    
    # check if after taking action in state we win
    def check_win(self, state, action):
        r = action//self.n_cols
        c = action%self.n_cols
        player = state[r,c]

        return (player != 0) and (
            np.sum(state[r, :]) == player * self.n_cols
            or np.sum(state[:, c]) == player * self.n_rows
            or np.sum(np.diag(state)) == player * self.n_rows
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.n_rows
        )

    # get value if it's a leaf node 
    def get_value_and_terminated(self,action, state):
        if self.check_win(state,action):
            return 1,True
        elif np.sum(self.get_valid_moves(state)) == 0:
            return 0,True
        return 0,False
    
    # flip the player
    def get_opponent(self,player):
        return -player

    # get the opp value
    def get_opponent_value(self, value):
        return -value

    # change the perspective
    def change_perspective(self, state, player):
        return state * player
    
    # encode a state to have a (3,3,3) matrix like rgb
    def get_encoded_state(self, state):
        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(np.float32)
        if len(state.shape) == 3:
                encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
