#test maze
from agent import Agent
from maze_environment import MazeEnvironment
from test_functions import test_agent
from plots import plot

# DQN model with static maze
env = MazeEnvironment(size=5, dynamic=False)
agent = Agent(state_size=25, action_size=4, model_type='DQN', use_per=False)
metrics = test_agent(agent, env, episodes=100)
print(metrics)

# dynamic maze
env_dynamic = MazeEnvironment(size=5, dynamic=True)
agent_dynamic = Agent(state_size=25, action_size=4, model_type='DQN', use_per=False)
metrics_dynamic = test_agent(agent_dynamic, env_dynamic, episodes=100)
print(metrics_dynamic)