from src.cla import CausalLearningAgent as CLA
from src.training_environment import TrainingEnvironment
import copy
import time
import logging
import cProfile
import pstats

# Set up logging for hierarchy operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Hierarchy:
    def __init__(
        self,
        agent_parents: dict[CLA, list[CLA]],
        pooling_vars: list[str],
        glue_vars: list[str],
        monte_carlo_samples: int = 100,
    ):
        """
        Initializes the hierarchy with a dictionary of agents and their parents.

        :param agent_parents: A dictionary where keys are parents and values are lists of children.
        """
        print(f"🏗️  Initializing Hierarchy with {monte_carlo_samples} Monte Carlo samples")
        print(f"   Pooling variables: {pooling_vars}")
        print(f"   Glue variables: {glue_vars}")
        
        init_start = time.time()
        
        for pooling_var in pooling_vars:
            if (
                pooling_var
                not in agent_parents[next(iter(agent_parents))][0].utility_vars
            ):
                raise ValueError(
                    f"Pooling variable {pooling_var} not found in utility variables of child agent."
                )
        for glue_var in glue_vars:
            if glue_var not in next(iter(agent_parents)).reflective_vars:
                raise ValueError(
                    f"Glue variable {glue_var} not found in reflective variables of parent agent."
                )
            if glue_var not in agent_parents[next(iter(agent_parents))][0].glue_vars:
                raise ValueError(
                    f"Glue variable {glue_var} not found in glue variables of child agent."
                )
        
        self.parents = []
        self.children = []
        
        print(f"   Creating Monte Carlo replicas...")
        mc_start = time.time()
        
        for parent, children in agent_parents.items():
            print(f"     Parent: Creating {monte_carlo_samples} replicas")
            self.parents.append(
                self.make_monte_carlo(parent, num_samples=monte_carlo_samples)
            )
            for i, child in enumerate(children):
                print(f"     Child {i+1}: Creating {monte_carlo_samples} replicas")
                self.children.append(
                    self.make_monte_carlo(child, num_samples=monte_carlo_samples)
                )
        
        mc_time = time.time() - mc_start
        print(f"   ✅ Monte Carlo creation completed in {mc_time:.2f}s")
        
        self.TE = TrainingEnvironment()
        self.monte_carlo_samples = monte_carlo_samples
        self.pooling_vars = pooling_vars
        self.glue_vars = glue_vars
        self.pooling_var_vals = {}
        self.glue_var_cpds = {}
        
        init_time = time.time() - init_start
        print(f"🏗️  Hierarchy initialization completed in {init_time:.2f}s")
        print(f"   Total agents: {len(self.parents[0]) + sum(len(child_group) for child_group in self.children)}")

    def make_monte_carlo(self, agent: CLA, num_samples: int = 1000):
        """
        Creates a Monte Carlo simulation for the given agent.

        :param agent: The agent to create the Monte Carlo simulation for.
        :param num_samples: The number of samples to generate.
        :return: A list of samples.
        """
        return [copy.deepcopy(agent) for _ in range(num_samples)]

    def pooling_func_update(self):
        print("🔄 Starting pooling function update...")
        pooling_start = time.time()
        
        for pooling_var in self.pooling_vars:
            print(f"   Processing pooling variable: {pooling_var}")
            var_start = time.time()
            
            aggregated_rewards = []
            for i in range(self.monte_carlo_samples):
                avg_reward = 0
                child_rewards = []
                
                for j, children in enumerate(self.children):
                    child_total = 0
                    if len(children[i].memory) > 0:
                        for timestep in children[i].memory:
                            child_total += timestep.average_reward[pooling_var]
                        child_avg = child_total / len(children[i].memory)
                        child_rewards.append(child_avg)
                        avg_reward += child_avg
                
                if len(self.children) > 0:
                    avg_reward /= len(self.children)
                aggregated_rewards.append(avg_reward)
            
            self.pooling_var_vals[pooling_var] = aggregated_rewards
            var_time = time.time() - var_start
            
            print(f"     {pooling_var}: avg={sum(aggregated_rewards)/len(aggregated_rewards):.4f}, "
                  f"range=[{min(aggregated_rewards):.4f}, {max(aggregated_rewards):.4f}] ({var_time:.2f}s)")

        print(f"   Updating parent agents with pooled rewards...")
        parent_update_start = time.time()
        
        for parent_group_idx, parent in enumerate(self.parents):
            for i in range(self.monte_carlo_samples):
                parent[i].lower_tier_pooled_reward = {}
                for pooling_var in self.pooling_vars:
                    parent[i].lower_tier_pooled_reward[pooling_var] = (
                        self.pooling_var_vals[pooling_var][i]
                    )
                # Update u_hat models to include pooling variables as context
                parent[i].update_u_hat_models_for_pooling()
        
        parent_update_time = time.time() - parent_update_start
        pooling_time = time.time() - pooling_start
        
        print(f"   ✅ Parent update completed in {parent_update_time:.2f}s")
        print(f"🔄 Pooling function update completed in {pooling_time:.2f}s")

    def glue_var_update(self):
        print("🔗 Starting glue variable update...")
        glue_start = time.time()
        
        # Collect CPDs from parents
        print("   Collecting CPDs from parent agents...")
        collect_start = time.time()
        
        for glue_var in self.glue_vars:
            print(f"     Processing glue variable: {glue_var}")
            cpts = []
            for i in range(self.monte_carlo_samples):
                for parent_group_idx, parents in enumerate(self.parents):
                    try:
                        cpd = parents[i].sampling_model.get_cpds(glue_var)
                        cpts.append(cpd)
                    except Exception as e:
                        print(f"       ⚠️  Error getting CPD for {glue_var} from parent {parent_group_idx}, agent {i}: {e}")
                        raise
            self.glue_var_cpds[glue_var] = cpts
            print(f"       Collected {len(cpts)} CPDs for {glue_var}")
        
        collect_time = time.time() - collect_start
        print(f"   ✅ CPD collection completed in {collect_time:.2f}s")

        # Update children with new CPDs
        print("   Updating child agents with new CPDs...")
        update_start = time.time()
        
        for child_group_idx, children in enumerate(self.children):
            print(f"     Updating child group {child_group_idx+1}/{len(self.children)}")
            for i in range(self.monte_carlo_samples):
                for glue_var in self.glue_vars:
                    try:
                        # Check if the variable exists before trying to remove
                        existing_cpds = children[i].sampling_model.get_cpds()
                        var_names = [cpd.variable for cpd in existing_cpds]
                        
                        if glue_var in var_names:
                            children[i].sampling_model.remove_cpds(glue_var)
                        else:
                            print(f"       ⚠️  Variable {glue_var} not found in child {child_group_idx}, agent {i}")
                            print(f"           Available variables: {var_names}")
                        
                        children[i].sampling_model.add_cpds(self.glue_var_cpds[glue_var][i])
                        
                    except Exception as e:
                        print(f"       ❌ Error updating {glue_var} in child {child_group_idx}, agent {i}: {e}")
                        print(f"          Child glue_vars: {children[i].glue_vars}")
                        print(f"          Available CPDs: {[cpd.variable for cpd in children[i].sampling_model.get_cpds()]}")
                        raise
        
        update_time = time.time() - update_start
        glue_time = time.time() - glue_start
        
        print(f"   ✅ Child update completed in {update_time:.2f}s")
        print(f"🔗 Glue variable update completed in {glue_time:.2f}s")

    def train(self, cycles, parent_iter, child_iter):
        print(f"\n🎓 Starting Hierarchical Training")
        print(f"   Cycles: {cycles}, Parent iterations: {parent_iter}, Child iterations: {child_iter}")
        print(f"   Total child groups: {len(self.children)}")
        print(f"   Total parent groups: {len(self.parents)}")
        print(f"   Monte Carlo samples per agent: {self.monte_carlo_samples}")
        
        total_start = time.time()
        
        for cycle in range(cycles):
            print(f"\n📚 === CYCLE {cycle + 1}/{cycles} ===")
            cycle_start = time.time()
            
            # Step 1: Train children
            print(f"👶 Training children ({child_iter} iterations each)...")
            child_train_start = time.time()
            
            for child_group_idx, children in enumerate(self.children):
                group_start = time.time()
                print(f"   Training child group {child_group_idx + 1}/{len(self.children)} ({len(children)} agents)")
                self.TE.train(child_iter, "SA", children)
                group_time = time.time() - group_start
                print(f"   ✅ Child group {child_group_idx + 1} completed in {group_time:.2f}s")
            
            child_train_time = time.time() - child_train_start
            print(f"👶 All children training completed in {child_train_time:.2f}s")
            
            # Step 2: Pool utilities
            self.pooling_func_update()
            
            # Step 3: Train parents
            print(f"👨‍🏫 Training parents ({parent_iter} iterations each)...")
            parent_train_start = time.time()
            
            for parent_group_idx, parents in enumerate(self.parents):
                group_start = time.time()
                print(f"   Training parent group {parent_group_idx + 1}/{len(self.parents)} ({len(parents)} agents)")
                self.TE.train(parent_iter, "SA", parents)
                group_time = time.time() - group_start
                print(f"   ✅ Parent group {parent_group_idx + 1} completed in {group_time:.2f}s")
            
            parent_train_time = time.time() - parent_train_start
            print(f"👨‍🏫 All parent training completed in {parent_train_time:.2f}s")
            
            # Step 4: Update glue variables
            self.glue_var_update()
            
            cycle_time = time.time() - cycle_start
            print(f"📚 Cycle {cycle + 1} completed in {cycle_time:.2f}s")
            print(f"   Breakdown: Children={child_train_time:.1f}s, Parents={parent_train_time:.1f}s, Updates={cycle_time - child_train_time - parent_train_time:.1f}s")
        
        total_time = time.time() - total_start
        avg_cycle_time = total_time / cycles
        
        print(f"\n🎓 === HIERARCHICAL TRAINING COMPLETED ===")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average cycle time: {avg_cycle_time:.2f}s")
        print(f"   Total training iterations: {cycles * (len(self.children) * child_iter + len(self.parents) * parent_iter)}")
        
        # Cache statistics
        if hasattr(CLA, '_shared_cdn_cache'):
            cache_size = len(CLA._shared_cdn_cache)
            print(f"   Shared cache size: {cache_size} entries")

if __name__ == "__main__":
    # Example usage with profiling
    from src.teacher import build_classroom_hierarchy
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Build a simple hierarchy for profiling
    from src.final_actlab_sims.models import hierarchy_students
    
    hier = build_classroom_hierarchy(
        list(hierarchy_students.values())[:2],  # Use 2 students for faster profiling
        pooling_vars=["grades"],
        glue_vars=["grade_leniency", "curriculum_complexity"],
        monte_carlo_samples=10,  # Reduced for faster profiling
    )
    
    # Train with reduced iterations for profiling
    hier.train(cycles=5, parent_iter=1, child_iter=2)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)
    stats.dump_stats('hierarchy_profile.prof')
    print("\nProfile saved to hierarchy_profile.prof")
    print("Visualize with: snakeviz hierarchy_profile.prof")