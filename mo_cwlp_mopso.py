import numpy as np
import random
import matplotlib.pyplot as plt
import inspyred
from inspyred.ec import emo
from inspyred.ec.variators import crossover, mutator
from inspyred.ec.terminators import generation_termination
import time

def parse_cap_file(filename, carbon_factor=0.1):
    """
    Parse problem instance file with robust handling for multi-line costs.
    """
    with open(filename, 'r') as f:
        parts = f.readline().split()
        m, n = int(parts[0]), int(parts[1])
        capacities, fixed_costs = [], []
        
        # Read warehouses
        for _ in range(m):
            c, fcost = map(float, f.readline().split())
            capacities.append(c)
            fixed_costs.append(fcost)
            
        demands = []
        cost_matrix = np.zeros((m, n))
        
        # Read customers
        for j in range(n):
            # Read demand
            demand = float(f.readline().strip())
            demands.append(demand)
            
            # Read exactly m costs (may span multiple lines)
            costs = []
            remaining = m
            while remaining > 0:
                line = f.readline().strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                take = min(len(parts), remaining)
                costs.extend(parts[:take])
                remaining -= take
            
            # Assign to cost matrix
            for i in range(m):
                cost_matrix[i, j] = costs[i]
        
        emissions = carbon_factor * cost_matrix
        
    return m, n, np.array(capacities), np.array(fixed_costs), np.array(demands), cost_matrix, emissions

def decode_position(position, m, n):
    """
    Decode particle position with repair mechanism.
    Returns valid (y, assignments) with open warehouses.
    """
    y = (np.array(position[:m]) > 0.5).astype(int)
    
    # Ensure at least one warehouse is open
    if np.sum(y) == 0:
        y[np.random.randint(0, m)] = 1
    
    # Get probabilities only for open warehouses
    probs = np.array(position[m:]).reshape((m, n))
    open_probs = probs * y[:, np.newaxis]  # Mask closed warehouses
    
    assignments = np.argmax(open_probs, axis=0)
    
    return y, assignments

class MopsoEvaluator:
    def __init__(self, capacities, fixed_costs, demands, cost_matrix, emissions, penalty_factor=1e6):
        self.capacities = capacities
        self.fixed_costs = fixed_costs
        self.demands = demands
        self.cost_matrix = cost_matrix
        self.emissions = emissions
        self.penalty_factor = penalty_factor
        self.m = capacities.shape[0]
        self.n = demands.shape[0]
    
    def evaluate(self, candidates, args):
        fitness = []
        for c in candidates:
            opt_fit, true_vals = self.evaluate_position(c)
            # Return tuple of objectives for multi-objective optimization
            fitness.append(emo.Pareto([opt_fit[0], opt_fit[1]]))
        return fitness
    
    def evaluate_position(self, position):
        """
        Evaluate solution with penalties only on cost.
        """
        y, assign = decode_position(position, self.m, self.n)
        
        # Calculate fixed costs
        total_cost = np.dot(y, self.fixed_costs)
        total_emissions = 0.0
        capacity_usage = np.zeros(self.m)
        
        # Calculate transportation costs and capacity usage
        for j, i in enumerate(assign):
            total_cost += self.cost_matrix[i, j]
            total_emissions += self.emissions[i, j]
            capacity_usage[i] += self.demands[j]
        
        # Capacity violation penalty
        capacity_violation = np.sum(np.maximum(capacity_usage - y*self.capacities, 0))
        penalty = self.penalty_factor * capacity_violation
        
        # Assignment validity check penalty
        invalid_assignments = np.sum([y[i] == 0 for i in assign])
        penalty += self.penalty_factor * invalid_assignments
        
        return (total_cost + penalty, total_emissions), (total_cost, total_emissions, capacity_violation, invalid_assignments)

def mopso_generator(random, args):
    """Generate a particle for MOPSO"""
    dim = args["dim"]
    m = args["m"]
    pos = [random.random() for _ in range(dim)]
    # Bias warehouse selection probabilities towards 0.5 threshold
    for i in range(m):
        pos[i] = random.uniform(0.4, 0.6)
    return pos

@mutator
def mopso_move(random, candidates, args):
    """Move particles according to PSO rules"""
    w = args.get("w", 0.4)
    c1 = args.get("c1", 1.0)
    c2 = args.get("c2", 1.0)
    dim = args["dim"]
    velocities = args["velocities"]
    pbests = args["pbests"]
    archive = args["_ec"].archive
    
    new_candidates = []
    for i, candidate in enumerate(candidates):
        # Select global best (leader) from archive
        if len(archive) == 0:
            gbest = pbests[i].candidate
        else:
            gbest = random.choice(archive).candidate
        
        # Calculate new velocity
        r1 = [random.random() for _ in range(dim)]
        r2 = [random.random() for _ in range(dim)]
        
        new_vel = []
        for j in range(dim):
            vel = (w * velocities[i][j] + 
                   c1 * r1[j] * (pbests[i].candidate[j] - candidate[j]) + 
                   c2 * r2[j] * (gbest[j] - candidate[j]))
            new_vel.append(vel)
        
        # Update position
        new_pos = [np.clip(candidate[j] + new_vel[j], 0, 1) for j in range(dim)]
        new_candidates.append(new_pos)
        
        # Store updated velocity
        velocities[i] = new_vel
        
    return new_candidates

def mopso_replacer(random, population, parents, offspring, args):
    """Custom replacer to handle personal best updates"""
    evaluator = args["_ec"].evaluator
    pbests = args["pbests"]
    
    # Update personal bests
    for i, off in enumerate(offspring):
        parent_idx = i % len(parents)
        
        # Check if offspring dominates its parent
        if off.fitness.values[0] <= pbests[parent_idx].fitness.values[0] and \
           off.fitness.values[1] <= pbests[parent_idx].fitness.values[1] and \
           (off.fitness.values[0] < pbests[parent_idx].fitness.values[0] or 
            off.fitness.values[1] < pbests[parent_idx].fitness.values[1]):
            pbests[parent_idx] = off
    
    # Return offspring as the new population (generational model)
    return offspring

def mopso(filename, swarm_size=100, iterations=100, carbon_factor=0.1,
          w=0.4, c1=1.0, c2=1.0, max_archive=100, visualize=False):
    # Parse problem instance
    m, n, capacities, fixed_costs, demands, cost_matrix, emissions = parse_cap_file(filename, carbon_factor)
    dim = m + m * n
    
    # Set up MOPSO using inspyred
    rand = random.Random()
    rand.seed(int(time.time()))
    
    # Create evolutionary computation object
    ea = inspyred.ec.emo.NSGA2(rand)
    ea.variator = mopso_move
    ea.replacer = mopso_replacer
    
    # Create evaluator
    evaluator = MopsoEvaluator(capacities, fixed_costs, demands, cost_matrix, emissions)
    
    # Initialize storage for personal bests and velocities
    velocities = [[0.0 for _ in range(dim)] for _ in range(swarm_size)]
    
    # Run optimization
    final_pop = ea.evolve(
        generator=mopso_generator,
        evaluator=evaluator.evaluate,
        pop_size=swarm_size,
        maximize=False,
        max_generations=iterations,
        dim=dim,
        m=m,
        n=n,
        velocities=velocities,
        pbests=[],
        w=w,
        c1=c1,
        c2=c2,
        max_archive_size=max_archive
    )
    
    # Plot final Pareto front
    plt.figure(figsize=(10, 6))
    archive = ea.archive
    costs = []
    emissions_vals = []
    
    # Decode and evaluate solutions in archive
    solutions = []
    for sol in archive:
        pos = sol.candidate
        y, assign = decode_position(pos, m, n)
        
        # Calculate actual cost directly
        true_cost = np.dot(y, fixed_costs)
        for j, i in enumerate(assign):
            true_cost += cost_matrix[i, j]
        
        true_emissions = 0.0
        for j, i in enumerate(assign):
            true_emissions += emissions[i, j]
        
        # Get violation information
        _, true_vals = evaluator.evaluate_position(pos)
        cap_violation = true_vals[2]
        invalid = true_vals[3]
        
        costs.append(true_cost)
        emissions_vals.append(true_emissions)
        
        solutions.append({
            'y': y,
            'assign': assign,
            'cost': true_cost,
            'emissions': true_emissions,
            'feasible': cap_violation == 0 and invalid == 0
        })
    
    # Plot Pareto front
    if visualize:
        plot_pareto_front(solutions, title="Pareto Front")

    return solutions

def plot_pareto_front(solutions, title="Pareto Front"):
    costs = [sol['cost'] for sol in solutions]
    emissions = [sol['emissions'] for sol in solutions]
    feasible = [sol['feasible'] for sol in solutions]
    
    if feasible:
        plt.scatter(costs, emissions, c='green', marker='o', label='Feasible')
    else:
        plt.scatter(costs, emissions, c='red', marker='o', label='Infeasible')
    plt.title(title)
    plt.xlabel('Cost')
    plt.ylabel('Emissions')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python mo_cwlp_mopso.py <input_file> [swarm_size] [iterations] [carbon_factor]")
        sys.exit(1)
    
    filename = sys.argv[1]
    swarm_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    carbon_factor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    
    solutions = mopso(filename, swarm_size=swarm_size, iterations=iterations, carbon_factor=carbon_factor)
    
    print(f"Found {len(solutions)} Pareto-optimal solutions")
    for sol in solutions[:3]:  # Print first 3 solutions
        print(f"\nCost: {sol['cost']:.2f}, Emissions: {sol['emissions']:.2f}")
        print(f"Open warehouses: {np.where(sol['y'] == 1)[0] + 1}")