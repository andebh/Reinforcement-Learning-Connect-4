import gym
import gym_connect4
import actors
import QLearning


env = gym.make('Connect4Env-v0')
obses = env.reset()

actor_Q = QLearning.QLearning(obses[0]['board'], epsilon=0)
actor_Q.set_Q('train_10000_dictionary.txt')
actor_random = actors.RandomActor()
actor_manual = actors.ManualActor(env)
#actors = [actor_Q, actor_random]
actors = [actor_Q, actor_manual]

game_over = False
while not game_over:
    action_dict = {}
    for actor_id, actor in enumerate(actors):
        action = actor.act(obses[actor_id])
        action_dict[actor_id] = action
    
    obses, rewards, game_over, info = env.step(action_dict)
    #env.render()

