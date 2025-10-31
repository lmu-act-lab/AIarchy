from pgmpy.inference import CausalInference  # type: ignore
from pgmpy.inference.ExactInference import VariableElimination  # type: ignore
from pgmpy.inference.ExactInference import BeliefPropagation  # type: ignore
from pgmpy.inference.base import Inference  # type: ignore
from pgmpy.models import BayesianNetwork  # type: ignore
from pgmpy.models import JunctionTree  # type: ignore
from pgmpy.models import MarkovNetwork  # type: ignore
from pgmpy.models import FactorGraph  # type: ignore
from pgmpy.models import DynamicBayesianNetwork  # type: ignore
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from collections import defaultdict
from itertools import chain


class FastVariableElimination(VariableElimination):
    """VariableElimination that skips check_model()."""
    def __init__(self, model):
        # Skip check_model() - we trust the model is valid since we validate after modifications
        self.model = model
        
        # Copy the rest of Inference.__init__ logic from base.py
        if isinstance(self.model, JunctionTree):
            self.variables = set(chain(*self.model.nodes()))
        else:
            self.variables = self.model.nodes()
        
        # Initialize structures without validation
        self._initialize_structures()


class FastBeliefPropagation(BeliefPropagation):
    """BeliefPropagation that skips check_model()."""
    def __init__(self, model):
        # Skip check_model() - we trust the model is valid since we validate after modifications
        self.model = model
        
        # Copy the rest of Inference.__init__ logic from base.py
        if isinstance(self.model, JunctionTree):
            self.variables = set(chain(*self.model.nodes()))
        else:
            self.variables = self.model.nodes()
        
        # Initialize structures without validation
        self._initialize_structures()


class FastCausalInference(CausalInference):
    """
    CausalInference that uses FastInference algorithms to skip check_model().
    This provides significant performance improvements when creating many inference objects.
    """
    def query(
        self,
        variables,
        do=None,
        evidence=None,
        adjustment_set=None,
        inference_algo="ve",
        show_progress=True,
        **kwargs,
    ):
        """
        Override query to use FastInference algorithms that skip check_model().
        """
        # Copy the parent query logic but use FastInference instead
        from itertools import chain
        
        do = {} if do is None else do
        evidence = {} if evidence is None else evidence

        if inference_algo == "ve":
            inference_algo = FastVariableElimination
        elif inference_algo == "bp":
            inference_algo = FastBeliefPropagation
        elif not isinstance(inference_algo, Inference):
            raise ValueError(
                f"inference_algo must be one of: 've', 'bp', or an instance of pgmpy.inference.Inference. Got: {inference_algo}"
            )

        # Step 2: Check if adjustment set is provided, otherwise try calculating it.
        if adjustment_set is None:
            do_vars = [var for var, state in do.items()]
            adjustment_set = set(
                chain(*[self.model.predecessors(var) for var in do_vars])
            )
            if len(adjustment_set.intersection(self.model.latents)) != 0:
                raise ValueError(
                    "Not all parents of do variables are observed. Please specify an adjustment set."
                )

        infer = inference_algo(self.model)

        # Step 3.1: If no do variable specified, do a normal probabilistic inference.
        if do == {}:
            return infer.query(variables, evidence, show_progress=False)
        # Step 3.2: If no adjustment is required, do a normal probabilistic
        #           inference with do variables as the evidence.
        elif len(adjustment_set) == 0:
            evidence = {**evidence, **do}
            return infer.query(variables, evidence, show_progress=False)

        # Step 4: For other cases, compute \sum_{z} p(variables | do, z) p(z)
        values = []

        # Step 4.1: Compute p_z and states of z to iterate over.
        # For computing p_z, if evidence variables also in adjustment set,
        # manually do reduce else inference will throw error.
        evidence_adj_inter = {
            var: state
            for var, state in evidence.items()
            if var in adjustment_set.intersection(evidence.keys())
        }
        evidence_adj_only = {
            var: state
            for var, state in evidence.items()
            if var in adjustment_set and var not in evidence_adj_inter.keys()
        }
        evidence_not_adj = {
            var: state
            for var, state in evidence.items()
            if var not in adjustment_set
        }

        # Compute p(z) where z is adjustment set
        if len(evidence_adj_inter) > 0:
            p_z = infer.query(
                list(adjustment_set),
                evidence_adj_inter,
                show_progress=False,
            )
        else:
            p_z = infer.query(list(adjustment_set), show_progress=False)

        # Step 4.2: Iterate over all possible values of z and compute p(variables | do, z)
        for z_state in p_z.get_all_assignments():
            z_state_dict = dict(z_state)
            evidence_with_z = {**evidence_not_adj, **z_state_dict, **do}
            p_var_given_do_z = infer.query(
                variables, evidence_with_z, show_progress=False
            )
            p_z_val = p_z.get_value(**z_state_dict)
            values.append(p_var_given_do_z * p_z_val)

        # Step 4.3: Sum over all z states
        result = sum(values)
        return result

