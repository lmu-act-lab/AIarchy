from src.util import Counter
import random
import json
from collections import defaultdict
from src.cla import CausalLearningAgent
import numpy as np
from src.final_actlab_sims.set_seed import set_seed

set_seed()

class ExactQComparisonAgent:
    def __init__(self, actions, gamma, alpha, cla_agent: CausalLearningAgent) -> None:
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.agent = cla_agent
        self.iter = 0
        self.eps = 0.9
        self.qvals = Counter()
        self.reward_tracker = []

    
    def get_current_state(self):
        state = {}
        for cpt in self.agent.get_cpts():
            values_rounded = np.round(cpt.get_values(), 2)
            state[cpt.variable] = values_rounded.tolist()
        return json.dumps(state)
        
    def get_qval(self, state, action):
        return self.qvals[(state, action)]
        

    def get_best_action_qval(self, state):
        return max(
            [(action, self.get_qval(state, action)) for action in self.actions],
            key=lambda x: x[1],
        )

    def q_value_update(self, reward, prev_state, state, action):
        # print(reward)
        # print(self.get_best_action_qval(state))
        delta = (
            reward
            + (self.gamma * self.get_best_action_qval(state)[1])
            - self.get_qval(prev_state, action)
        )
        # print(prev_state)
        # print(f"q_val: {self.qvals[(prev_state, action)]} \n delta: {delta}")
        self.qvals[(prev_state, action)] += self.alpha * delta


    def choose_action(self, state):
        best_action = (
            max(
                [(action, self.get_qval(state, action)) for action in self.actions],
                key=lambda x: x[1],
            )[0]
            if (self.iter > 0) and (random.random() > self.eps)
            else random.choice(self.actions)
        )
        self.eps -= 0.001
        self.iter += 1
        # print(self.iter)
        # # print(self.qvals)
        # print(self.eps)
        # print(best_action)
        return best_action

    def sample_current_model(self):
        step = self.agent.time_step(self.agent.fixed_assignment, self.agent.weights, self.agent.sample_num)
        return sum(step.average_reward.values())
    
    def train(self, episodes):
        fixed = self.agent.fixed_assignment
        weights = self.agent.weights
        samples = self.agent.sample_num
        for _ in range(episodes):
            prev_state = self.get_current_state()
            action = self.choose_action(prev_state)
            #nudge cpt
            adjusted_cpt = self.agent.nudge_cpt_new(
                self.agent.sampling_model.get_cpds(action[0][0]),
                {action[0][0]: action[0][1]},
                1,
                0.02,
            )
            self.agent.sampling_model.remove_cpds(
                self.agent.sampling_model.get_cpds(action[0][0])
            )
            self.agent.sampling_model.add_cpds(adjusted_cpt)

            step = self.agent.time_step(fixed, weights, samples, {action[0][0] : action[0][1]})
            self.q_value_update(sum(step.average_reward.values()), prev_state, self.get_current_state(), action)
            self.reward_tracker.append(float(self.sample_current_model()))