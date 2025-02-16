from util import Counter
import random
import json
from collections import defaultdict



class ExactQComparisonAgent:
    def __init__(self, actions, gamma, alpha, cla_agent) -> None:
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.agent = cla_agent
        self.iter = 0
        self.eps = 0.2
        self.qvals: dict[tuple[json, tuple[str, int]] : float] = defaultdict(len(self.actions))
        self.state = self.curr_state_JSON()

    
    def curr_state_JSON(self):
        cpt_list = self.agent.get_cpts()
        return json.dumps(cpt_list, index=4)
        
    def get_qval(self, state, action):
        return self.qvals[(state, action)]
        

    def get_best_action_qval(self, state):
        return max(
            [(action, self.get_qval(state, action)) for action in self.actions],
            key=lambda x: x[1],
        )

    def q_value_update(self, reward, state, action):
        delta = (
            reward
            + (self.gamma * self.get_best_action_qval(state))
            - self.get_qval(state, action)
        )
        self.qvals[(self.state, action)] += self.alpha * delta


    def choose_action(self):
        best_action = (
            max(
                [(action, self.get_qval(action)) for action in self.actions],
                key=lambda x: x[1],
            )[0]
            if (self.iter > 0) or (random.random() > self.eps)
            else random.choice(self.actions)
        )
        self.iter += 1
        return best_action
    
    def train(self, episodes):
        fixed = self.agent.fixed_assignment
        weights = self.agent.weights
        samples = self.agent.samples
        for episode in range(episodes):
            # state = self.state
            done = False
            while not done:
                action = self.choose_action()
                step = self.agent.time_step(fixed, weights, samples, action)
                #nudge cpt
                self.q_value_update(step.average_reward)
                self.state = self.curr_state_JSON
            self.eps *= self.eps_decay  # Decay exploration rate
            episode -= 1
            
