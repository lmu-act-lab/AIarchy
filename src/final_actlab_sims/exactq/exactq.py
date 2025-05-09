from src.final_actlab_sims.models import exact_q_agents
from matplotlib import pyplot as plt
import pandas as pd

from src.final_actlab_sims.set_seed import set_seed
import logging
logging.disable(logging.WARNING)
set_seed()

for exact in exact_q_agents:
    print(f"agent: {exact_q_agents.index(exact)}")
    exact.train(2000)

average_rewards: list[float] = []
for iteration in range(len(exact_q_agents[0].reward_tracker)):
    total_reward = 0
    for agent in exact_q_agents:
        total_reward += agent.reward_tracker[iteration]
    average_rewards.append(total_reward / len(exact_q_agents))

reward_data = []
for agent in exact_q_agents:
    if agent.reward_tracker:
        reward_data.append(agent.reward_tracker[-1])

df = pd.DataFrame(reward_data)
df.to_csv(f"rewards.csv", index=False)
fig = plt.figure(figsize=(12, 9))
plt.plot(average_rewards)
plt.xlabel("Iteration")
plt.ylabel("Average Reward")
plt.title(f"Average Reward Across {len(exact_q_agents)} Agents")
fig.savefig(f"monte_carlo.png")
plt.close(fig)