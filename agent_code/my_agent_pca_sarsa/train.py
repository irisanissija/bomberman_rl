import pickle
from collections import namedtuple, deque
from typing import List
from sklearn.ensemble import GradientBoostingRegressor
import os
import numpy as np

import events as e
from .callbacks import state_to_array

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_action'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/my-saved-model_rule.pt')

GAMMA = 0.95

FEATURE_HISTORY_SIZE = 500  # number of features to use for training
# train with:
# python main.py play --agents my_agent_sarsa --no-gui --train 1 --n-rounds 7000
# with random_prob = 0.7 in my_agent_sarsa/callbacks/act_rule_based


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    Assumes that a model has already been trained in train_with_saved_states.py

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # transition contains the state-action-reward-next-state-next-action tuple needed
    # to calculate the targets with SARSA
    self.transition = {'state': None, "action": None, "next_state": None, "reward": None, "next_action": None}
    self.x = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]  # features
    self.y = [deque(maxlen=FEATURE_HISTORY_SIZE) for _ in ACTIONS]  # targets


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step.

    Updates the transition, adds the old game state to the features, calculates the
    corresponding target with SARSA and adds it to the targets.

    :param self: The same object that is passed to all of your callbacks.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None and new_game_state is not None and self_action is not None:
        # transform game state into feature vector with 20 entries using kernel PCA
        old_features = self.pca_transformer.transform(state_to_array(old_game_state).reshape(1, -1))
        new_features = self.pca_transformer.transform(state_to_array(new_game_state).reshape(1, -1))

        if self.transition["state"] is None:  # first step, initialising
            self.transition["state"] = old_features
            self.transition["action"] = self_action
            self.transition["next_state"] = new_features
            self.transition["reward"] = reward_from_events(self, events)
        else:
            self.transition["next_action"] = self_action

            index = ACTION_INDEX[self.transition["action"]]
            index_next = ACTION_INDEX[self.transition["next_action"]]
            x = self.transition["state"]
            # SARSA
            y = self.transition["reward"] + GAMMA * \
                self.model[index_next].predict([self.transition["next_state"].ravel()])[0]

            self.transition["state"] = old_features
            self.transition["action"] = self_action
            self.transition["next_state"] = new_features
            self.transition["reward"] = reward_from_events(self, events)

            self.x[index].append(x)
            self.y[index].append(y)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    Updates the transition, adds the old game state to the features, calculates the
    corresponding target with SARSA and adds it to the targets.

    Fits and stores the model if enough features have been collected for each action.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: The state that was passed to the last call of `act`.
    :param last_action: The action that you took.
    :param events: The events that occurred after the last action
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    if last_action is not None:
        if self.transition["next_action"] is None:
            x = self.transition["state"]
            y = self.transition["reward"]
            index = ACTION_INDEX[self.transition["action"]]
            self.x[index].append(x)
            self.y[index].append(y)
        elif self.transition["next_action"] is not None:
            self.transition["next_action"] = last_action

            index = ACTION_INDEX[self.transition["action"]]
            index_next = ACTION_INDEX[self.transition["next_action"]]
            x1 = self.transition["state"]
            # SARSA
            y1 = self.transition["reward"] + GAMMA * \
                 self.model[index_next].predict([self.transition["next_state"].ravel()])[0]
            x2 = self.pca_transformer.transform(state_to_array(last_game_state).reshape(1, -1))
            # x2 is a terminal state, so the expected reward is 0
            y2 = reward_from_events(self, events) + 0

            self.x[index].append(x1)
            self.y[index].append(y1)
            self.x[index_next].append(x2)
            self.y[index_next].append(y2)

    # fit and store the model if enough states have been recorded for each action
    if all([(len(x) == FEATURE_HISTORY_SIZE) for x in self.x]):
        print("Fitting model")
        for i, action in enumerate(ACTIONS):
            self.model[i].fit(self.x[i], self.y[i])
            self.x[i].clear()
            self.y[i].clear()

        # Store the model
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculates the sum of rewards for the events that occurred in a step

    :param self: The same object that is passed to all of your callbacks.
    :param events: List of events that occurred in a step
    :return: int
    """
    game_rewards = {
        e.COIN_COLLECTED: 3,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -3,
        e.WAITED: -0.2,
        # e.GOT_KILLED: -5,
        e.KILLED_SELF: -5
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
