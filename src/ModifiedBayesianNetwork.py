from pgmpy.models import BayesianNetwork  # type: ignore

class BayesianNetwork(BayesianNetwork):
    def __eq__(self, other):
      if isinstance(other, BayesianNetwork):
        return set(self.get_cpds()) == set(other.get_cpds())
      return False
    
    def __hash__(self):
      return hash(tuple(cpd.to_factor() for cpd in self.get_cpds()))