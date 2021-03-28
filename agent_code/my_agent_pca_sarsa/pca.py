# PERFORM KERNEL PCA TO GET RELEVANT FEATURES FROM STATE
import pickle
import os
import numpy as np
from sklearn.decomposition import KernelPCA

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_INDEX = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/my-saved-model_rule.pt')
STATE_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/states_1_agent_crates.pt')
NEXT_STATE_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/next_states_1_agent_crates.pt')
NEXT_ACTION_INDEX_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/next_action_index_1_agent_crates.pt')
REWARD_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/rewards_1_agent_crates.pt')

PCA_TRANSFORMER_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/pca_transformer_1_agent_crates.pt')
TRANSFORMED_STATES_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/pca_transformed_states_1_agent_crates.pt')
TRANSFORMED_NEXT_STATES_PATH = os.path.join(BASE_PATH, 'my_agent_pca_sarsa/pca_transformed_states_1_agent_crates.pt')

N_COMPONENTS = 20  # dimension of the output

with open(STATE_PATH, "rb") as file:
    states = pickle.load(file)

with open(NEXT_STATE_PATH, "rb") as file:
    next_states = pickle.load(file)

# print(np.shape(states), np.shape(next_states))

pca_transformer = KernelPCA(n_components=20, kernel='linear')

n_actions, n_states, dim_states = np.shape(states)
# select a random subset of the states for the fit. Fitting all 6*100000 states would require 80GB of memory
random_indices = np.random.permutation(np.linspace(0, n_actions*n_states, n_actions*n_states + 1)).astype(int)[0:5000]
pca_transformer.fit(X=np.reshape(states, (n_actions * n_states, dim_states))[random_indices])

# to train a model using SARSA later, we need for each target the state and the following state
transformed_states = []
transformed_next_states = []

for i in range(len(ACTIONS)):
    transformed_states.append(pca_transformer.transform(X=states[i]))

for i in range(len(ACTIONS)):
    transformed_next_states.append(pca_transformer.transform(X=next_states[i]))

# print(np.shape(transformed_states), np.shape(transformed_next_states))

with open(PCA_TRANSFORMER_PATH, "wb") as file:
    pickle.dump(pca_transformer, file)

with open(TRANSFORMED_STATES_PATH, "wb") as file:
    pickle.dump(transformed_states, file)

with open(TRANSFORMED_NEXT_STATES_PATH, "wb") as file:
    pickle.dump(transformed_next_states, file)


