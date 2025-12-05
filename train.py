#train
from agent import Agent
from maze_environment import MazeEnvironment
from train_functions import train_agent

#DQN model with dynamic maze (one wall moves every step)
env = MazeEnvironment(size=5, dynamic=True)
agent = Agent(state_size=100, action_size=4, model_type='DQN', use_per=False)
metrics = train_agent(agent, env, episodes=1500)