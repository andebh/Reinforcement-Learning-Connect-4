import gym
import gym_connect4
import numpy as np
import actors
import QLearning
import matplotlib.pyplot as plt


class Train_QLearning:

    def __init__(self, env, Q_agent, opponent_agent, n_inner=100, n_outer=100):
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.env = env
        self.Q_agent = Q_agent
        self.opponent = opponent_agent
        self.data_list = []

    def create_dictionary_file(self, filename):
        actors = [self.Q_agent, self.opponent]
        for _ in range(self.n_outer):
            Q_reward = 0
            for _ in range(self.n_inner):
                obses = env.reset()
                game_over = False
                while not game_over:
                    action_dict = {}
                    for actor_id, actor in enumerate(actors):
                        action = actor.act(obses[actor_id])
                        action_dict[actor_id] = action
                    
                    obses, rewards, game_over, info = env.step(action_dict)
                    if game_over:
                        if rewards[0] == 1:
                            Q_reward += 1
                    self.Q_agent.update(action_dict[0], obses, rewards)
            self.data_list.append(Q_reward/self.n_inner)

        with open(filename, 'w') as writer:
            writer.write(str(self.Q_agent.Q))


    def create_plot_file(self, filename, show=False):
        mean_arr = [0.5 for _ in range(self.n_outer)]
        plt.plot(list(range(len(self.data_list))), self.data_list)
        plt.plot(list(range(len(self.data_list))), mean_arr)
        plt.title('QLearning training evolution')
        plt.xlabel(f'Each point corresponds to {self.n_inner} games')
        plt.ylabel('Score')
        plt.savefig(filename)
        if show:
            plt.show()


if __name__ == "__main__":
    env = gym.make('Connect4Env-v0')
    obses = env.reset()
    Q_agent = QLearning.QLearning(obses[0]['board'])
    random_agent = actors.RandomActor()
    n_inner = 100
    n_outer = 100
    trainer =  Train_QLearning(env, Q_agent, random_agent, n_inner, n_outer)
    trainer.create_dictionary_file(f'train_{n_inner*n_outer}_dictionary.txt')
    trainer.create_plot_file(f'train_{n_inner*n_outer}_iterations.pdf')
