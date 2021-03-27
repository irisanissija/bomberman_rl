import numpy as np
from scipy.spatial import distance
from .helper import state_to_features
import time
import os
import pickle

class CustomModel:

    modelDict = {(0,0,0,0):[0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
    qDictFileName = str
    gamma: int
    alpha: int 
    lastPositions = [(np.NINF, np.NINF), (np.inf,np.inf)]

    def __init__(self):
        self.actions = np.array(['UP','DOWN','LEFT','RIGHT','BOMB'])
        self.gamma = 0.8
        self.alpha = 0.5

        self.qDictFileName = "custom_model.pt"

        if not os.path.isfile(self.qDictFileName):
            print('created new dict file')
            pickle.dump(self.modelDict, open(self.qDictFileName, "wb"))


    def predict_action(self, game_state):
        features = state_to_features(game_state)
        qTable = pickle.load(open(self.qDictFileName, "rb"))
        
        if features in qTable:
            rewards = qTable[features]
            if np.abs(np.sort(rewards)[-1] - np.sort(rewards)[-2]) < 1:
                index = np.random.choice(np.argsort(rewards)[-2:])
                action = index
            else:
                action = np.argmax(rewards)
        else:
            generalizedState = self.getNearestNeighbourState(features, qTable)
            qvalues = qTable[generalizedState]
            qvalues_sorted = np.sort(qvalues)
            action = np.argmax(qvalues)
        return action

    def updateLastPositions(self, position: tuple):
        self.lastPositions[0] = self.lastPositions[1]
        self.lastPositions[1] = position
    
    def stateExistsInTable(self, state, table):
        return next((True for elem in table if np.array_equal(elem, state)), False)
    
    def update_qtable(self, old_state, next_state, action_taken, total_reward):

        currentQTable = pickle.load(open(self.qDictFileName, "rb" ))

        """ newTable = np.load(self.qTableFileName)
        statesTable = np.load(self.stateTableFileName) """
        actionIndex = np.where(self.actions == action_taken)[0]        
        
        old_state_features = state_to_features(old_state)
        next_state_features = state_to_features(next_state)

        if type(old_state_features) is not tuple:
            return None
        
        if type(next_state_features) is not tuple:
            return None

        if old_state_features in currentQTable:
            old_state_rewards = currentQTable[old_state_features]
            old_reward = old_state_rewards[actionIndex]

            if next_state_features in currentQTable:
                next_state_rewards = currentQTable[next_state_features]
                max_value_of_next_state = np.max(next_state_rewards)
            else:
                max_value_of_next_state = 0

            # this is the essential q_learning function
            new_value = (1 - self.alpha)* old_reward + self.alpha * (total_reward + self.gamma * max_value_of_next_state)
            currentQTable[old_state_features][actionIndex] = new_value

        else:
            newRow = np.zeros([len(self.actions)])
            newRow[actionIndex] = total_reward
            currentQTable.update({old_state_features: newRow})
        
        pickle.dump(currentQTable, open(self.qDictFileName,"wb"))


    def update_qtable_after_game_ends(self, old_game_state, action_taken, total_reward):
        currentQTable = pickle.load(open(self.qDictFileName, "rb" ))

        actionIndex = np.where(self.actions == action_taken)[0]        
        old_state_features = state_to_features(old_game_state)
        if type(old_state_features) is not tuple:
            return None

        if old_state_features in currentQTable:
            old_state_rewards = currentQTable[old_state_features]
            old_reward = old_state_rewards[actionIndex]

            new_value = (1-self.alpha)*old_reward + self.alpha * total_reward
            currentQTable[old_state_features][actionIndex] = new_value
        
        else:
            newRow = np.zeros([len(self.actions)])
            newRow[actionIndex] = total_reward
            currentQTable.update({old_state_features: newRow})
        
        pickle.dump(currentQTable, open(self.qDictFileName,"wb"))

    def getNearestNeighbourState(self, state, qtable: dict):
        keys = list(qtable.keys())
        distances = []
        for key in keys:
            if key == (0,0,0,0):
                pass
            else:
                distances.append(distance.euclidean(key, state))
        return keys[np.argmin(distances)]