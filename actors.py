import numpy as np

class RandomActor():
    """Random actor""" 
    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        return np.random.choice(actions)

class ManualActor():
    """Manual Actor, leave render off in workflow"""
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        actions = np.argwhere(obs['action_mask']).reshape(-1)
        if actions[0] == 7:
            return 7
        else:
            self.env.render()
            action = int(input("Choose column... ")) - 1 
            print(action)
        return action

