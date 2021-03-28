import os
import pickle
import random
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'my_agent_rule/my-saved-model_rule.pt')

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


def getRandomProb(game_state: dict):
    gamesPlayed = game_state['round']
    steps = game_state['step']


    if gamesPlayed < 6000:
        return 0

    if gamesPlayed > 300 and steps < 11:
        return 0
    else:
        if gamesPlayed > 25 and gamesPlayed < 100:
            return 0.8
        if gamesPlayed > 99 and gamesPlayed < 150:
            return 0.7
        if gamesPlayed > 149 and gamesPlayed < 200:
            return 0.6
        if gamesPlayed > 199 and gamesPlayed < 500:
            return 0.5
        if gamesPlayed > 499 and gamesPlayed < 1000:
            return 0.3
        if gamesPlayed > 1249 and gamesPlayed < 2500:
            return 0.15
        if gamesPlayed > 2499:
            return 0.0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    currentRandomProb = -1
    # TODO Exploration vs exploitation
     # exploration in rule_based_agent/callbacks.py
    if self.train and random.random() < currentRandomProb:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    elif self.train and not self.model_initialised:
        self.logger.debug("Model not yet initialised, choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS)  # wait and bomb excluded for now

    self.logger.debug("Querying model for action.")
    x = state_to_features(game_state)
    response = np.ravel([model.predict([x.ravel()]) for model in self.model])
    # print(x)
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
    
    
    """ environment = np.zeros(4)  # the surrounding 4 fields (up, down, left, right)
    if field[position_x - 1, position_y] == 0:
        environment[0] = 1  # free space = 1
    if field[position_x + 1, position_y] == 0:
        environment[1] = 1
    if field[position_x, position_y - 1] == 0:
        environment[2] = 1
    if field[position_x, position_y + 1] == 0:
        environment[3] = 1
    
    features = np.append(features, environment)
    """

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
