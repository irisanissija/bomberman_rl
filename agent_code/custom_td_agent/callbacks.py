import os
import pickle
import random
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'custom_td_agent/my-saved-model_rule.pt')

random_prob = 0


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train:
        if os.path.isfile(MODEL_PATH):
            self.logger.info("Loading model from saved state.")
            with open(MODEL_PATH, "rb") as file:
                self.model = pickle.load(file)
        else:
            self.logger.info("Setting up model from scratch.")
            self.model = None
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)
        self.logger.info('model loaded')


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    currentRandomProb = -1 # increase if exploitation vs exploration
    if self.train and random.random() < currentRandomProb:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    x = state_to_features(game_state)
    response = np.ravel([model.predict([x.ravel()]) for model in self.model])

    if np.abs(np.sort(response)[-1] - np.sort(response)[-2]) < 0.01:
        index = np.random.choice(np.argsort(response)[-2:])
        return ACTIONS[index]
    self.logger.debug(f'Model chose action {ACTIONS[np.argmax(response)]}')
    action = ACTIONS[np.argmax(response)]
    return action



def getSurroundingFields(field, position):
    myX = position[0]
    myY = position[1]

    surrounding = []
    surrounding.append(field[myX-1,myY-1])
    surrounding.append(field[myX, myY-1])
    surrounding.append(field[myX+1, myY-1])
    surrounding.append(field[myX-1, myY])
    surrounding.append(field[myX+1, myY])
    surrounding.append(field[myX-1, myY+1])
    surrounding.append(field[myX, myY+1])
    surrounding.append(field[myX+1, myY+1])

    return surrounding


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None


    field = game_state['field']
    coins = game_state['coins']
    index_arrays = np.where(field == 1)
    crates = list(zip(index_arrays[0],index_arrays[1]))
    position_x, position_y = game_state['self'][3]
    if len(coins)>0:
        distance = np.zeros((len(coins), 2))
        i = 0
        for (x, y) in coins:
            dist_x = position_x - x
            dist_y = position_y - y
            distance[i, 0] = dist_x
            distance[i, 1] = dist_y
            i += 1

        assert len(np.sum(distance, axis=1)) == len(coins)
        features = distance[np.argmin(np.sum(distance**2, axis=1))]  # distance in x and y direction to closest coin
        # distance can be negative
        assert len(features) == 2
    else:
        distance = np.zeros((len(crates), 2))
        i = 0
        for (x,y) in crates:
            dist_x = position_x - x
            dist_y = position_y - y
            distance[i, 0] = dist_x
            distance[i, 1] = dist_y
            i += 1

        assert len(np.sum(distance, axis=1)) == len(crates)
        features = distance[np.argmin(np.sum(distance**2, axis=1))]  # distance in x and y direction to closest coin
        # distance can be negative
        assert len(features) == 2
    


    changedField = field
    for coin in game_state['coins']:
        changedField[coin] = 2
    for bomb in game_state['bombs']:
        bomb_coordinates = bomb[0]
        changedField[bomb_coordinates] = 3

    surrounding = getSurroundingFields(changedField, (position_x, position_y))
    
    for point in surrounding:
        features = np.append(features, point)

    

    return features
