import pickle
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

# discount factor
GAMMA = 0.95

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/my-saved-model_rule.pt')
STATE_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/states_1_agent_crates.pt')
NEXT_STATE_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/next_states_1_agent_crates.pt')
NEXT_ACTION_INDEX_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/next_action_index_1_agent_crates.pt')
REWARD_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/rewards_1_agent_crates.pt')

PCA_TRANSFORMER_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/pca_transformer_1_agent_crates.pt')
TRANSFORMED_STATES_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/pca_transformed_states_1_agent_crates.pt')
TRANSFORMED_NEXT_STATES_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/pca_transformed_states_1_agent_crates.pt')

# load features and rewards

with open(TRANSFORMED_STATES_PATH, "rb") as file:
    states = pickle.load(file)

with open(TRANSFORMED_NEXT_STATES_PATH, "rb") as file:
    next_states = np.array(pickle.load(file))

with open(NEXT_ACTION_INDEX_PATH, "rb") as file:
    next_action_index = pickle.load(file)

with open(REWARD_PATH, "rb") as file:
    rewards = pickle.load(file)

# print(np.shape(states), np.shape(next_states), np.shape(rewards), np.shape(next_action_index))

# initialise model (one for each action)
model = [GradientBoostingRegressor(n_estimators=1000, learning_rate=0.001, max_depth=1, random_state=0, loss='ls',
                                   warm_start=True, init='zero', verbose=True) for _ in ACTIONS]

# fit model
for i in range(len(ACTIONS)):
    model[i].fit(states[i], rewards[i])  # initial guess: Q = 0

predictions = np.zeros(np.shape(rewards))
for i in range(len(ACTIONS)):
    for j, index_next in enumerate(next_action_index[i]):
        predictions[i, j] = model[index_next].predict(next_states[i, j, :].reshape(1, -1))[0]

# targets for SARSA
targets = [rewards[i] + GAMMA * predictions[i] for i in range(len(ACTIONS))]

assert np.shape(targets) == np.shape(rewards)

# fit model again
for i in range(len(ACTIONS)):
    model[i].fit(states[i], targets[i])

# save model
with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

