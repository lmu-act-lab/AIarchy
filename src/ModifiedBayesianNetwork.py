from pgmpy.models import BayesianNetwork  # type: ignore
from pgmpy.sampling import BayesianModelSampling  # type: ignore
from pgmpy.factors.discrete import TabularCPD  # type: ignore
from pgmpy.utils import compat_fns  # type: ignore

class BayesianNetwork(BayesianNetwork):
    def __eq__(self, other):
      if isinstance(other, BayesianNetwork):
        return set(self.get_cpds()) == set(other.get_cpds())
      return False
    
    def __hash__(self):
      return hash(tuple(cpd.to_factor() for cpd in self.get_cpds()))
    
    def add_cpds(self, *cpds, validate=True):
        """
        Add CPDs and optionally validate the model once after modification.
        
        Parameters
        ----------
        *cpds : CPD objects to add
        validate : bool, default True
            If True, validate the model after adding CPDs. Set to False when
            adding multiple CPDs in sequence to validate only once at the end.
        """
        super().add_cpds(*cpds)
        # Validate once after modification - then we can skip validation later
        if validate:
            self.check_model()
        return self
    
    def check_model_without_utility_vars(self, utility_vars):
        """
        Check model but skip utility variables that don't have CPDs.
        This is needed because utility variables may be in the graph via edges
        but don't have CPDs since they're computed from reward functions.
        Validates that all non-utility nodes have CPDs.
        
        Parameters
        ----------
        utility_vars : set[str]
            Set of utility variable names to skip during validation.
        """
        from pgmpy.factors.continuous import ContinuousFactor
        from pgmpy.factors.discrete import TabularCPD
        
        # Check that all non-utility nodes have CPDs and are valid
        for node in self.nodes():
            if node in utility_vars:
                continue  # Skip utility variables
            
            cpd = self.get_cpds(node=node)
            
            # Check if a CPD is associated with every non-utility node.
            if cpd is None:
                raise ValueError(f"No CPD associated with {node}")
            
            # Check if the CPD is an instance of either TabularCPD or ContinuousFactor.
            if isinstance(cpd, (TabularCPD, ContinuousFactor)):
                evidence = cpd.get_evidence()
                parents = self.get_parents(node)
                
                # Check if the evidence set of the CPD is same as its parents.
                if set(evidence) != set(parents):
                    raise ValueError(
                        f"CPD associated with {node} doesn't have proper parents associated with it."
                    )
                
                if len(set(cpd.variables) - set(cpd.state_names.keys())) > 0:
                    raise ValueError(
                        f"CPD for {node} doesn't have state names defined for all the variables."
                    )
                
                # Check if the values of the CPD sum to 1.
                if not cpd.is_valid_cpd():
                    raise ValueError(
                        f"Sum or integral of conditional probabilities for node {node} is not equal to 1."
                    )
        
        # Check cardinality consistency for non-utility nodes
        for node in self.nodes():
            if node in utility_vars:
                continue
            cpd = self.get_cpds(node=node)
            for index, parent_node in enumerate(cpd.variables[1:]):
                # Skip utility variable parents in cardinality check
                if parent_node in utility_vars:
                    continue
                parent_cpd = self.get_cpds(node=parent_node)
                if parent_cpd.cardinality[0] != cpd.cardinality[1 + index]:
                    raise ValueError(
                        f"The cardinality of {parent_node} doesn't match in it's child nodes."
                    )
    
    def simulate(
        self,
        n_samples=10,
        do=None,
        evidence=None,
        virtual_evidence=None,
        virtual_intervention=None,
        include_latents=False,
        partial_samples=None,
        seed=None,
        show_progress=True,
    ):
        """
        Simulate with check_model() skipped since we validate after modifications.
        This is a copy of the parent simulate() method but with check_model() removed.
        """
        from pgmpy.sampling import BayesianModelSampling

        # Skip check_model() - we trust the model is valid since we validate after modifications
        state_names = self.states

        evidence = {} if evidence is None else evidence
        for var, state in evidence.items():
            if state not in state_names[var]:
                raise ValueError(f"Evidence state: {state} for {var} doesn't exist")

        do = {} if do is None else do
        for var, state in do.items():
            if state not in state_names[var]:
                raise ValueError(f"Do state: {state} for {var} doesn't exist")

        virtual_intervention = (
            [] if virtual_intervention is None else virtual_intervention
        )
        virtual_evidence = [] if virtual_evidence is None else virtual_evidence

        if set(do.keys()).intersection(set(evidence.keys())):
            raise ValueError("Variable can't be in both do and evidence")

        # Only copy the model if we need to modify it (interventions or virtual evidence)
        # This avoids expensive deepcopy operations when just sampling from the original model
        needs_modification = (do != {}) or (virtual_intervention != []) or (virtual_evidence != [])
        
        if needs_modification:
            model = self.copy()
        else:
            model = self  # Use original model directly - no copy needed!

        # Step 1: If do or virtual_intervention is specified, modify the network structure.
        if (do != {}) or (virtual_intervention != []):
            virt_nodes = [cpd.variables[0] for cpd in virtual_intervention]
            model = model.do(list(do.keys()) + virt_nodes)
            evidence = {**evidence, **do}
            virtual_evidence = [*virtual_evidence, *virtual_intervention]

        # Step 2: If virtual_evidence; modify the network structure
        if virtual_evidence != []:
            for cpd in virtual_evidence:
                var = cpd.variables[0]
                if var not in model.nodes():
                    raise ValueError(
                        "Evidence provided for variable which is not in the model"
                    )
                elif len(cpd.variables) > 1:
                    raise (
                        "Virtual evidence should be defined on individual variables. Maybe you are looking for soft evidence."
                    )
                elif self.get_cardinality(var) != cpd.get_cardinality([var])[var]:
                    raise ValueError(
                        "The number of states/cardinality for the evidence should be same as the number of states/cardinality of the variable in the model"
                    )

            for cpd in virtual_evidence:
                var = cpd.variables[0]
                new_var = "__" + var
                model.add_edge(var, new_var)
                values = compat_fns.get_compute_backend().vstack(
                    (cpd.values, 1 - cpd.values)
                )
                new_cpd = TabularCPD(
                    variable=new_var,
                    variable_card=2,
                    values=values,
                    evidence=[var],
                    evidence_card=[model.get_cardinality(var)],
                    state_names={new_var: [0, 1], var: cpd.state_names[var]},
                )
                model.add_cpds(new_cpd)
                evidence[new_var] = 0

        # Step 3: If no evidence do a forward sampling
        if len(evidence) == 0:
            samples = BayesianModelSampling(model).forward_sample(
                size=n_samples,
                include_latents=include_latents,
                seed=seed,
                show_progress=show_progress,
                partial_samples=partial_samples,
            )

        # Step 4: If evidence; do a rejection sampling
        else:
            samples = BayesianModelSampling(model).rejection_sample(
                size=n_samples,
                evidence=[(k, v) for k, v in evidence.items()],
                include_latents=include_latents,
                seed=seed,
                show_progress=show_progress,
                partial_samples=partial_samples,
            )

        # Step 5: Postprocess and return
        if include_latents:
            return samples
        else:
            return samples.loc[:, list(set(self.nodes()) - self.latents)]