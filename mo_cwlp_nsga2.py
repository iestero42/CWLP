"""
Multi-Objective Capacitated Warehouse Location Problem (MO-CWLP) solver using inspyred

This module implements the NSGA-II algorithm using the inspyred library to solve the capacitated 
warehouse location problem with two objectives:
1. Minimize total cost (fixed cost + transportation cost)
2. Minimize emissions (transportation cost * emissions factor)
"""

import random
import numpy as np
import inspyred
from typing import List, Tuple, Dict, Optional


def parse_cap_file(filename, carbon_factor=0.1):
    """
    Parse the capacitated warehouse location problem instance file.
    
    Parameters:
    - filename: path to the input file
    - carbon_factor: factor to compute emissions from transportation costs
    
    Returns:
    - m: number of warehouses
    - n: number of customers
    - capacities: array of warehouse capacities
    - fixed_costs: array of warehouse fixed opening costs
    - demands: array of customer demands
    - cost_matrix: transportation cost matrix [warehouses × customers]
    - emissions: emissions matrix derived from cost_matrix
    """
    with open(filename, 'r') as f:
        parts = f.readline().split()
        m, n = int(parts[0]), int(parts[1])
        capacities, fixed_costs = [], []
        
        # Read warehouse data
        for _ in range(m):
            c, fcost = map(float, f.readline().split())
            capacities.append(c)
            fixed_costs.append(fcost)
            
        demands = []
        cost_matrix = np.zeros((m, n))
        
        # Read customer data
        for j in range(n):
            # Read demand (on its own line)
            demand = float(f.readline().strip())
            demands.append(demand)
            
            # Read exactly m costs for the customer (may span multiple lines)
            costs = []
            remaining = m
            while remaining > 0:
                line = f.readline().strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                # Take only the needed number of elements
                take = min(len(parts), remaining)
                costs.extend(parts[:take])
                remaining -= take
            
            # Assign costs to cost matrix
            for i in range(m):
                cost_matrix[i, j] = costs[i]
                
        # Compute emissions based on transportation costs
        emissions = carbon_factor * cost_matrix
        
    return m, n, np.array(capacities), np.array(fixed_costs), np.array(demands), cost_matrix, emissions


class MOCWLPProblem:
    """Problem class for Multi-Objective Capacitated Warehouse Location Problem."""
    
    def __init__(self, instance_file, carbon_factor=0.1):
        """Initialize the problem with instance data.
        
        Args:
            instance_file: Path to .cap file
            carbon_factor: Factor for calculating emissions
        """
        # Parse instance file
        result = parse_cap_file(instance_file, carbon_factor)
        self.n_warehouses, self.n_customers, self.capacities = result[0], result[1], result[2]
        self.fixed_costs, self.demands, self.cost_matrix, self.emissions_matrix = result[3], result[4], result[5], result[6]
    
    def generator(self, random, args):
        """Generate a random individual using a GRASP-like heuristic to ensure feasibility."""
        individual = [-1] * self.n_customers
        open_warehouses = set()
        remaining_capacities = np.zeros(self.n_warehouses)
        alpha = 0.3  # GRASP RCL parameter

        # Check total feasibility
        if np.sum(self.demands) > np.sum(self.capacities):
            raise ValueError("Problem is infeasible: total demand exceeds total capacity.")

        # Random customer order
        customers = list(range(self.n_customers))
        random.shuffle(customers)

        for c in customers:
            demand = self.demands[c]
            feasible_options = []

            # Evaluate all warehouses for feasibility and cost
            for w in range(self.n_warehouses):
                # Calculate potential remaining capacity if used
                if w in open_warehouses:
                    available = remaining_capacities[w]
                else:
                    available = self.capacities[w]
                
                if available >= demand:
                    # Cost includes transportation and fixed cost (if opening)
                    if w in open_warehouses:
                        cost = self.cost_matrix[w, c]
                    else:
                        cost = self.cost_matrix[w, c] + self.fixed_costs[w]
                    feasible_options.append((w, cost, w not in open_warehouses))
            
            if not feasible_options:
                # Fallback: find warehouse with maximum capacity (even if requires opening)
                w = np.argmax(self.capacities)
                cost = self.cost_matrix[w, c] + self.fixed_costs[w]
                feasible_options.append((w, cost, True))

            # Determine RCL (Restricted Candidate List)
            min_cost = min(f[1] for f in feasible_options)
            max_cost = max(f[1] for f in feasible_options)
            threshold = min_cost + alpha * (max_cost - min_cost)
            rcl = [f for f in feasible_options if f[1] <= threshold]

            # Random selection from RCL
            selected = random.choice(rcl)
            w, cost, needs_opening = selected

            # Assign customer to warehouse w
            individual[c] = w

            if needs_opening:
                open_warehouses.add(w)
                remaining_capacities[w] = self.capacities[w] - demand
            else:
                remaining_capacities[w] -= demand

        return individual
    
    def evaluator(self, candidates, args):
        """Evaluate individuals with adjusted penalties for infeasibility."""
        fitness = []
        penalty_factor = 1e4  # Reduced penalty factor
        
        print(f"Evaluating {len(candidates)} candidates...")
        for individual in candidates:
            open_warehouses = set(individual)
            total_fixed = sum(self.fixed_costs[w] for w in open_warehouses)
            total_transport = sum(self.cost_matrix[w, c] for c, w in enumerate(individual))
            total_emissions = sum(self.emissions_matrix[w, c] for c, w in enumerate(individual))
            
            # Calculate capacity violations
            capacity_usage = np.zeros(self.n_warehouses)
            for c, w in enumerate(individual):
                capacity_usage[w] += self.demands[c]
            violation = sum(max(0, capacity_usage[w] - self.capacities[w]) for w in open_warehouses)
            
            # Apply penalty proportional to violation
            penalty = penalty_factor * violation
            total_cost = total_fixed + total_transport + penalty
            total_emis = total_emissions + (penalty * 0.1)
            
            fitness.append(inspyred.ec.emo.Pareto([total_cost, total_emis], maximize=False))
        return fitness
    
    def crossover(self, random, mom, dad, args):
        """Uniform crossover with repair to encourage feasibility."""
        child1, child2 = [], []
        for m, d in zip(mom, dad):
            if random.random() < 0.5:
                child1.append(m)
                child2.append(d)
            else:
                child1.append(d)
                child2.append(m)
        
        # Repair children to minimize capacity violations
        for child in [child1, child2]:
            self._repair(child, random)
        return child1, child2
    
    def _repair(self, individual, random):
        """Attempt to repair capacity violations by moving customers."""
        capacity_usage = np.zeros(self.n_warehouses)
        for c, w in enumerate(individual):
            capacity_usage[w] += self.demands[c]
        
        # Check for overloaded warehouses
        for w in set(individual):
            if capacity_usage[w] <= self.capacities[w]:
                continue
            
            # Customers in this warehouse, sorted by demand (descending)
            customers = [c for c, w_assign in enumerate(individual) if w_assign == w]
            customers.sort(key=lambda c: self.demands[c], reverse=True)
            
            for c in customers:
                current_demand = self.demands[c]
                # Find alternative warehouses
                possible = [alt_w for alt_w in range(self.n_warehouses)
                           if (capacity_usage[alt_w] + current_demand <= self.capacities[alt_w])]
                if possible:
                    alt_w = random.choice(possible)
                    individual[c] = alt_w
                    capacity_usage[w] -= current_demand
                    capacity_usage[alt_w] += current_demand
                    if capacity_usage[w] <= self.capacities[w]:
                        break

    def mutator(self, random, candidate, args):
        """Mutate with feasibility-aware reassignment."""
        mutant = candidate.copy()
        mutation_rate = args.get('mutation_rate', 0.1)  # Higher mutation rate
        
        for c in range(len(mutant)):
            if random.random() < mutation_rate:
                current_w = mutant[c]
                demand = self.demands[c]
                
                # Find feasible warehouses for this customer
                feasible = []
                for w in range(self.n_warehouses):
                    # Calculate current usage if assigned here
                    used = sum(self.demands[other] for other, w_assign in enumerate(mutant) if w_assign == w)
                    if w == current_w:
                        used -= demand
                    if used + demand <= self.capacities[w]:
                        feasible.append(w)
                
                if feasible:
                    new_w = random.choice(feasible)
                    mutant[c] = new_w
        
        # Repair after mutation
        self._repair(mutant, random)
        return mutant
    
    def selector(random, population, args):
        """Custom selector that prioritizes feasible solutions or those with lower penalties.
        
        Args:
            random: Random number generator
            population: List of individuals
            args: Optional arguments
            
        Returns:
            Selected individual
        """
        if len(population) == 0:
            return None
            
        # First, separate feasible from infeasible solutions
        feasible = []
        infeasible = []
        problem = args['_ec'].evaluator.args[0]  # Get problem instance
        
        for individual in population:
            # Check feasibility by calculating capacity violations
            capacity_usage = np.zeros(problem.n_warehouses)
            for c, w in enumerate(individual.candidate):
                capacity_usage[w] += problem.demands[c]
                
            is_feasible = True
            for w in set(individual.candidate):
                if capacity_usage[w] > problem.capacities[w]:
                    is_feasible = False
                    break
                    
            if is_feasible:
                feasible.append(individual)
            else:
                infeasible.append(individual)
        
        # If we have feasible solutions, select among them using tournament selection
        if feasible:
            # Use tournament selection among feasible solutions
            tournament_size = min(3, len(feasible))
            competitors = random.sample(feasible, tournament_size)
            return min(competitors, key=lambda x: x.fitness.values[0])
        else:
            # No feasible solutions, select from infeasible ones with preference
            # to those with lower penalty (implied by fitness value)
            tournament_size = min(3, len(infeasible))
            competitors = random.sample(infeasible, tournament_size)
            return min(competitors, key=lambda x: x.fitness.values[0])


def nsga2_solve(
    instance_file: str,
    pop_size: int = 100,
    generations: int = 100,
    carbon_factor: float = 0.1,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.05,
    verbose: bool = False,
    visualize: bool = True
) -> List[Tuple]:
    """Solve MO-CWLP using inspyred's NSGA-II algorithm with direct customer assignments.
    
    Args:
        instance_file: Path to .cap file
        pop_size: Population size
        generations: Number of generations
        carbon_factor: Factor for calculating emissions
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        verbose: Whether to print progress
        visualize: Whether to visualize the Pareto front
        
    Returns:
        List of non-dominated solutions in Pareto front
    """
    # Create problem instance
    problem = MOCWLPProblem(instance_file, carbon_factor)

    # Create evolutionary algorithm
    ea = inspyred.ec.emo.NSGA2(random.Random())
    
    # Set observer for verbose output
    if verbose:
        def observer(population, num_generations, num_evaluations, args):
            if num_generations % 10 == 0:
                print(f"Generation {num_generations}/{generations}")
                best = sorted(population)[0]
                objectives = best.fitness.values
                print(f"  Best solution: Cost = {objectives[0]:.2f}, Emissions = {objectives[1]:.2f}")
                
                # Compute capacity violations for the best solution
                individual = best.candidate
                problem_instance = args['_ec'].evaluator.args[0]
                capacity_usage = np.zeros(problem_instance.n_warehouses)
                for c, w in enumerate(individual):
                    capacity_usage[w] += problem_instance.demands[c]
                
                violations = 0
                open_warehouses = set(individual)
                for w in open_warehouses:
                    if capacity_usage[w] > problem_instance.capacities[w]:
                        violations += 1
                
                print(f"  Open warehouses: {len(open_warehouses)}, Capacity violations: {violations}")
        
        ea.observer = observer
    
    #ea.selector = problem.selector  # Use custom selector

    # Solve using inspyred
    final_pop = ea.evolve(
        generator=problem.generator,
        evaluator=problem.evaluator,
        pop_size=pop_size,
        maximize=False,
        num_selected=pop_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        num_elites=int(0.1 * pop_size),  # NSGA-II handles elitism internally
        max_generations=generations,
        crossover=problem.crossover,
        mutator=problem.mutator,
        args={'_ec': ea}
    )
    
    # Extract Pareto front
    pareto_front = []
    print(f"Final population size: {len(final_pop)}")
    for individual in final_pop:
        # Get solution details
        solution = individual.candidate
        objectives = individual.fitness.values
        
        # Calculate capacity violations
        capacity_usage = np.zeros(problem.n_warehouses)
        for c, w in enumerate(solution):
            capacity_usage[w] += problem.demands[c]
        
        capacity_violation = 0
        open_warehouses = set(solution)
        for w in open_warehouses:
            if capacity_usage[w] > problem.capacities[w]:
                capacity_violation += capacity_usage[w] - problem.capacities[w]
        
        # Consider solution feasible only if there's no capacity violation
        is_feasible = capacity_violation == 0
        
        # Store solution with objectives and feasibility
        pareto_front.append((solution, objectives, is_feasible))
    
    # Sort Pareto front by feasibility (feasible first) and then by first objective
    pareto_front.sort(key=lambda x: (0 if x[2] else 1, x[1][0]))

    # Visualize Pareto front if requested
    if visualize:
        visualize_pareto_front(pareto_front, instance_file)
    
    if verbose:
        print(f"Final Pareto front size: {len(pareto_front)}")
        feasible_count = sum(1 for _, _, is_feasible in pareto_front if is_feasible)
        print(f"Feasible solutions: {feasible_count}/{len(pareto_front)}")
        
    return pareto_front

def visualize_pareto_front(pareto_front, instance_name=''):
    """
    Visualize the Pareto front with different colors for feasible and infeasible solutions.
    
    Args:
        pareto_front: List of tuples (solution, objectives, is_feasible)
        instance_name: Name of the problem instance for the plot title
    """
    import matplotlib.pyplot as plt
    
    # Extract data for visualization
    feasible_solutions = []
    infeasible_solutions = []
    
    for _, objectives, is_feasible in pareto_front:
        if is_feasible:
            feasible_solutions.append(objectives)
        else:
            infeasible_solutions.append(objectives)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot feasible solutions
    if feasible_solutions:
        feasible_costs = [obj[0] for obj in feasible_solutions]
        feasible_emissions = [obj[1] for obj in feasible_solutions]
        plt.scatter(feasible_costs, feasible_emissions, c='blue', marker='o', 
                   s=50, label='Feasible Solutions')
    
    # Plot infeasible solutions
    if infeasible_solutions:
        infeasible_costs = [obj[0] for obj in infeasible_solutions]
        infeasible_emissions = [obj[1] for obj in infeasible_solutions]
        plt.scatter(infeasible_costs, infeasible_emissions, c='red', marker='x', 
                   s=50, label='Infeasible Solutions')
    
    # Add labels and title
    plt.xlabel('Total Cost')
    plt.ylabel('Total Emissions')
    
    instance_label = instance_name.split('/')[-1] if instance_name else ''
    plt.title(f'Pareto Front for {instance_label}')
    
    # Add legend
    if feasible_solutions or infeasible_solutions:
        plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) != 2:
        print("Usage: python mo_cwlp_nsga2.py <cap_file>")
        sys.exit(1)
    
    cap_file = sys.argv[1]
    pareto_front = nsga2_solve(cap_file, verbose=True)
    
    print("\nPareto front solutions:")
    for i, (solution, objectives, is_feasible) in enumerate(pareto_front[:10]):  # Show top 10
        cost, emissions = objectives
        feasible_str = "Feasible" if is_feasible else "Infeasible"
        print(f"Solution {i+1}: Cost = {cost:.2f}, Emissions = {emissions:.2f}, {feasible_str}")
    
    # Print customer assignments for the first few solutions
    for i, (solution, _, _) in enumerate(pareto_front[:3]):
        print(f"\nSolution {i+1} customer assignments:")
        if len(solution) <= 20:  # Show all assignments if reasonable
            for c, w in enumerate(solution):
                print(f"  Customer {c+1} -> Warehouse {w+1}")
        else:  # Show only first few
            for c, w in enumerate(solution[:10]):
                print(f"  Customer {c+1} -> Warehouse {w+1}")
            print("  ...")
        
        # Show which warehouses are open
        open_warehouses = sorted(set(solution))
        print(f"  Open warehouses: {[w+1 for w in open_warehouses]}")