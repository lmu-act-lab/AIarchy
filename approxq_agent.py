from util import Counter
import random

class ApproxQComparisonAgent:
    def __init__(self, actions, gamma, alpha, cla_agent) -> None:
        self.actions = actions
        self.weights = Counter()
        self.gamma = gamma
        self.alpha = alpha
        self.agent = cla_agent
        self.iter = 0
        self.eps = 0.2

    def get_q_value(self, state, action):
        return self.get_features(state, action) * self.weights

    def get_features(self, action):
        return Counter(self.agent.time_step({}, self.agent.weights, self.agent.sample_num, {action[0] : action[1]}).average_sample)

    def get_best_action_value(self, state):
        return max([(action, self.get_q_value(state, action)) for action in self.actions], key = lambda x : x[1])

    def q_value_update(self, reward, prev_state, state, action):
        for weight in self.weights.keys():
            delta = reward + (self.gamma * self.get_best_action_value(state)[1]) - self.get_q_value(prev_state, action)
            self.weights[weight] += self.alpha * delta * self.get_features(state, action)[weight]

    def choose_action(self, state):
        best_action = max([(action, self.get_q_value(state, action)) for action in self.actions], key = lambda x : x[1])[0] if (self.iter > 0) or (random.random > self.eps) else random.choice(self.actions)
        self.iter += 1
        return best_action




