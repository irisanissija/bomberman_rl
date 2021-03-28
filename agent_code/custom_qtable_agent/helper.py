import numpy as np
from typing import List

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


    channels = []
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
    elif len(crates)>0:
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
    else:
        features = [100,100] # set arbitrary value if there are neither coins nor crates. this should only happen if the game has ended.
    
    
    environment = np.zeros(4)  # the surrounding 4 fields (up, down, left, right)
    if field[position_x - 1, position_y] == 0:
        environment[0] = 1  # free space = 1
    if field[position_x + 1, position_y] == 0:
        environment[1] = 1
    if field[position_x, position_y - 1] == 0:
        environment[2] = 1
    if field[position_x, position_y + 1] == 0:
        environment[3] = 1
    
    features = np.append(features, environment)

    for feature in features:
        channels.append(features)

    stacked_channels = np.stack(channels)
      
    # and return them as a vector
    return tuple(stacked_channels.reshape(-1))

def action_to_numeric(actions: List[str], action):
    """
    this function maps an array string to a numeric list of actions 
    lets see if we need this
    """
    numericList = np.zeros(len(actions))
    actionIndex = actions.index(action)
    numericList[actionIndex] = 1

def getGameNumberFromState(game_state: dict):
    return game_state['round']

def getStepsFromState(game_state:dict):
    return game_state['step']

def getOwnPosition(game_state:dict):
    if game_state is not None:
        return game_state['self'][3]
    else:
        return (0,0)

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
    """     surrounding.append(field[myX-2, myY-2])
    surrounding.append(field[myX-1, myY-2])
    surrounding.append(field[myX, myY-2])
    surrounding.append(field[myX+1, myY-2])
    surrounding.append(field[myX+2, myY-2])
    surrounding.append(field[myX-2, myY-1])
    surrounding.append(field[myX+2, myY-1])
    surrounding.append(field[myX-2, myY])
    surrounding.append(field[myX+2, myY])
    surrounding.append(field[myX-2, myY+1])
    surrounding.append(field[myX+2, myY+1])
    surrounding.append(field[myX-2, myY+2])
    surrounding.append(field[myX-1, myY+2])
    surrounding.append(field[myX, myY+2])
    surrounding.append(field[myX+1, myY+2])
    surrounding.append(field[myX+2, myY+2]) """

    return surrounding

def directionWithMostCoins(position, coins):
    directions = [0,0,0,0]
    myX = position[0]
    myY = position[1]
    for coin in coins: 
        coinX = coin[0]
        coinY = coin[1]
        if coinX > myX:
            directions[1] += 1
        if coinX < myX:
            directions[3] += 1
        if coinY > myY:
            directions[0] += 1
        if coinY < myY:
            directions[2] += 1

    direction_to_go = np.argmax(directions)
    feature_vector = [0,0,0,0]
    feature_vector[direction_to_go] = 1
    return feature_vector