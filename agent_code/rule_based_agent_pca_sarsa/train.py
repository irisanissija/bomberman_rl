import pickle
from collections import namedtuple
from typing import List
import os
import numpy as np

import events as e
from .callbacks import state_to_array


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'custom_sarsa_pca_agent/my-saved-model_rule.pt')
STATE_PATH = os.path.join(BASE_PATH, 'custom_sarsa_pca_agent/states_1_agent_crates.pt')
NEXT_STATE_PATH = os.path.join(BASE_PATH, 'custom_sarsa_pca_agent/next_states_1_agent_crates.pt')
NEXT_ACTION_INDEX_PATH = os.path.join(BASE_PATH, 'custom_sarsa_pca_agent/next_action_index_1_agent_crates.pt')
REWARD_PATH = os.path.join(BASE_PATH, 'custom_sarsa_pca_agent/rewards_1_agent_crates.pt')

FEATURE_HISTORY_SIZE = 100000  # number of states to collect per action for training

# collect states with:
# python main.py play --agents rule_based_agent_pca_sarsa --no-gui --train 1 --n-rounds 25000

# COLLECTS TRAINING DATA (100 000 for each action)


def setup_training(self):
    """
    Initialise self.

    This is called after `setup` in callbacks.py.

    Collects states for PCA transformation, as well as the state-action-reward-next-state-next-action tuple
    needed for training a model with SARSA

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.x = np.zeros((len(ACTIONS), FEATURE_HISTORY_SIZE, 314))  # states
    self.x_next = np.zeros((len(ACTIONS), FEATURE_HISTORY_SIZE, 314))  # state after action was executed
    self.rewards = np.zeros((len(ACTIONS), FEATURE_HISTORY_SIZE))  # reward after action was executed
    self.action_index_next = np.zeros((len(ACTIONS), FEATURE_HISTORY_SIZE)).astype(int)  # index of the action executed next
    self.index = np.zeros(len(ACTIONS)).astype(int)  # number of states that have been collected for each action
    self.previous_action_index = None  # index of the previous action
    self.saved = False  # whether states have been saved


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step.

    Updates the x with the old game state, x_next with the new game states, action_index_next with the action
    and rewards with the calculated rewards.

    Saves all the states, actions and rewards in files if enough have been collected.

    :param self: The same object that is passed to all of your callbacks.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None and new_game_state is not None and self_action is not None:
        index = ACTION_INDEX[self_action]

        if self.previous_action_index is None:
            feature_array = state_to_array(old_game_state)
            self.x[index, self.index[index], :] = feature_array
            self.rewards[index, self.index[index]] = (reward_from_events(self, events))
            feature_array_next = state_to_array(new_game_state)
            self.x_next[index, self.index[index], :] = feature_array_next
            self.previous_action_index = index
            self.index[index] += 1

        elif self.index[index] < FEATURE_HISTORY_SIZE:
            feature_array = state_to_array(old_game_state)
            index = ACTION_INDEX[self_action]
            self.x[index, self.index[index], :] = feature_array
            self.rewards[index, self.index[index]] = (reward_from_events(self, events))
            feature_array_next = state_to_array(new_game_state)
            self.x_next[index, self.index[index], :] = feature_array_next

            self.action_index_next[self.previous_action_index] = index
            self.previous_action_index = index

            self.index[index] += 1

    # save all states, actions and rewards if enough have been collected for each action
    if all([i == FEATURE_HISTORY_SIZE for i in self.index]) and not self.saved:
        print("Saving states")
        with open(STATE_PATH, "wb") as file:
            pickle.dump(self.x, file)
        with open(REWARD_PATH, "wb") as file:
            pickle.dump(self.rewards, file)
        with open(NEXT_STATE_PATH, "wb") as file:
            pickle.dump(self.x_next, file)
        with open(NEXT_ACTION_INDEX_PATH, "wb") as file:
            pickle.dump(self.action_index_next, file)
        self.saved = True


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    Updates the x with the old game state, x_next with the new game states, action_index_next with the action
    and rewards with the calculated rewards.

    Saves all the states, actions and rewards in files if enough have been collected.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: The state that was passed to the last call of `act`.
    :param last_action: The action that you took.
    :param events: The events that occurred after the last action    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    if last_action is not None:

        index = ACTION_INDEX[last_action]

        if self.index[index] < FEATURE_HISTORY_SIZE:
            feature_array = state_to_array(last_game_state)
            index = ACTION_INDEX[last_action]
            self.x[index, self.index[index], :] = feature_array

            self.rewards[index, self.index[index]] = (reward_from_events(self, events))

            self.action_index_next[self.previous_action_index] = index
            self.previous_action_index = index

            self.index[index] += 1

    if all([i == FEATURE_HISTORY_SIZE for i in self.index]) and not self.saved:
        print("Saving states")
        with open(STATE_PATH, "wb") as file:
            pickle.dump(self.x, file)
        with open(REWARD_PATH, "wb") as file:
            pickle.dump(self.rewards, file)
        with open(NEXT_STATE_PATH, "wb") as file:
            pickle.dump(self.x_next, file)
        with open(NEXT_ACTION_INDEX_PATH, "wb") as file:
            pickle.dump(self.action_index_next, file)
        self.saved = True


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
        e.KILLED_SELF: -5,
        e.CRATE_DESTROYED: 1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
