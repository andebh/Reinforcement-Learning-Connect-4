import numpy as np

class QLearning:
    def __init__(self, initial_state, discount=0.9, alpha = 0.01, epsilon=0.1):
        self.newboard = np.array_repr(initial_state)
        self.state = self.newboard
        self.action = 0
        self.reward = 0
        self.Q = {}
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon

    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        if actions[0] == 7:
            return 7
        elif (np.random.uniform() < self.epsilon):
            return np.random.choice(actions)
        else:
            max_action = np.random.choice(actions)
            max_value = 0
            state = np.array_repr(obs['board'])
            for a in actions:
                if (state, a) in self.Q:
                    if self.Q[(state,a)] > max_value:
                        max_action = a
                        max_value = self.Q[(state,a)]
        return max_action

    def update(self, action, obses, rewards):
        if rewards[0] == 0 and action == 7:
            next_state = np.array_repr(obses[0]['board'])
            max_future_reward = 0
            actions = obses[0]['action_mask']
            for a in np.argwhere(actions).reshape(-1):
                if a != 7:
                    if (next_state, a) in self.Q:
                        if self.Q[(next_state, a)] > max_future_reward:
                            max_future_reward = self.Q[(next_state, a)] 
            if not (self.state, self.action) in self.Q:
                self.Q[(self.state, self.action)] = 0
            self.Q[(self.state, self.action)] += self.alpha*(self.reward + self.discount*max_future_reward - self.Q[(self.state, self.action)])
            self.state = next_state

        self.reward = rewards[0]
        self.action = action
        
        if rewards[0] != 0:
            if not (self.state, self.action) in self.Q:
                self.Q[(self.state, self.action)] = 0
            self.Q[(self.state, self.action)] += self.alpha*(self.reward - self.Q[(self.state, self.action)])
            self.state = self.newboard

    def set_Q(self, filename):
        with open(filename, 'r') as reader:
            self.Q = eval(reader.read().replace('\n', ''))

