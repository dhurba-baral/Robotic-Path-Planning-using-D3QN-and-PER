#train
from agent import Agent
from maze_environment import MazeEnvironment
from train_functions import train_agent
from plots import plot

#DQN model with dynamic maze (one wall moves every step)
env = MazeEnvironment(size=5, dynamic=False)
agent = Agent(state_size=25, action_size=4, model_type='DQN', use_per=False)
metrics = train_agent(agent, env, episodes=1500)
plot(metrics["collisions"], 'collisions', 'DQN')