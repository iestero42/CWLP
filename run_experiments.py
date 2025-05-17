import os
import time
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.stats import mannwhitneyu, wilcoxon

from mo_cwlp_nsga2 import nsga2_solve
from mo_cwlp_mopso import mopso

# --- CONFIGURABLE GRIDS --------------------------------------------------
GA_GRID = [
    {'pop_size': 200, 'generations': 300, 'crossover_rate': 0.9, 'mutation_rate': 0.1},
    {'pop_size': 100, 'generations': 200, 'crossover_rate': 0.8, 'mutation_rate': 0.2},
    {'pop_size': 50,  'generations': 100, 'crossover_rate': 0.7, 'mutation_rate': 0.3},
    {'pop_size': 20,  'generations': 50,  'crossover_rate': 0.6, 'mutation_rate': 0.4},
    {'pop_size': 10,  'generations': 20,  'crossover_rate': 0.5, 'mutation_rate': 0.5},
]

PSO_GRID = [
    {'swarm_size': 100, 'iterations': 300, 'w': 0.5, 'c1': 1.0, 'c2': 1.0}, 
    {'swarm_size': 50,  'iterations': 200, 'w': 0.6, 'c1': 1.2, 'c2': 1.2},
    {'swarm_size': 20,  'iterations': 100,  'w': 0.7, 'c1': 1.4, 'c2': 1.4},
    {'swarm_size': 10,  'iterations': 50,  'w': 0.8, 'c1': 1.6, 'c2': 1.6},
    {'swarm_size': 5,   'iterations': 20,  'w': 0.9, 'c1': 1.8, 'c2': 1.8},
]

# --- CONFIGURATION --------------------------------------------------------
CARBON_FACTOR = 0.1
DEFAULT_RUNS  = 30
RESULTS_DIR  = "results"
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")

# --- HELPERS -------------------------------------------------------------

def hypervolume(front, ref=(1e9, 1e9)):
    """2D hypervolume (minimization)."""
    if not front: return 0.0
    pts = sorted(front, key=lambda x: x[0])
    hv = 0.0
    prev_e = ref[1]
    for c, e in pts:
        if c>ref[0] or e>ref[1]: continue
        hv += (ref[0]-c)*(prev_e - e)
        prev_e = e
    return hv

def write_csv(filepath, rows, fieldnames):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def summary_stats(name, arr):
    return f"{name}: min={np.min(arr):.2f}, avg={np.mean(arr):.2f}, std={np.std(arr):.2f}"

def plot_all_fronts(archive, title, out_png):
    plt.figure(figsize=(6,6))
    for label, pts in archive.items():
        costs = [p[0] for p in pts]
        emis  = [p[1] for p in pts]
        plt.scatter(costs, emis, label=label, s=15, alpha=0.6)
    plt.xlabel("Cost")
    plt.ylabel("Emissions")
    plt.title(title)
    plt.legend(fontsize="small", loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# --- MAIN RUNNERS --------------------------------------------------------

def run_nsga2(instance_file, cfg, runs):
    all_metrics = []
    all_archives = []
    for i in range(runs):
        start = time.time()
        try:
            pareto = nsga2_solve(
                instance_file,
                pop_size=cfg['pop_size'],
                generations=cfg['generations'],
                carbon_factor=CARBON_FACTOR,
                crossover_rate=cfg.get('crossover_rate', 0.9),
                mutation_rate=cfg.get('mutation_rate', 0.1),
                verbose=False,
                visualize=False
            )
            
            if not pareto:
                raise ValueError("Empty Pareto front")
   
            # Extract fitness values - filter out infinity
            front = [(fit[0], fit[1]) for _, fit, is_feasible in pareto 
                if not np.isinf(fit[0]) and not np.isinf(fit[1])]
            
            if not front:
                front = [(1e9, 1e9)]  # Default large value but not infinity
                
            hv = hypervolume(front)
            best_c = min(c for c, _ in front)
            best_e = min(e for _, e in front)
            
        except Exception as ex:
            print(f"  Error in run {i+1}: {ex}")
            front = []
            hv, best_c, best_e = 0, float('inf'), float('inf')
        
        rt = time.time() - start
        all_metrics.append({
            'run': i+1, 'cost': best_c, 'emis': best_e,
            'hv': hv, 'time': rt, 'size': len(front)
        })
        all_archives.append((f"GA_p{cfg['pop_size']}_g{cfg['generations']}", front))
    
    return all_metrics, all_archives

def run_mopso(instance_file, cfg, runs):
    all_metrics = []
    all_archives = []
    for i in range(runs):
        start = time.time()
        soln = mopso(
            instance_file,
            swarm_size=cfg['swarm_size'],
            iterations=cfg['iterations'],
            carbon_factor=CARBON_FACTOR,
            w=cfg.get('w', 0.5),
            c1=cfg.get('c1', 1.0),
            c2=cfg.get('c2', 1.0),
            visualize=False
        )
        rt = time.time() - start
        front = [(s['cost'], s['emissions']) for s in soln]
        hv = hypervolume(front)
        best_c = min(c for c,_ in front)
        best_e = min(e for _,e in front)
        all_metrics.append({
            'run': i+1, 'cost': best_c, 'emis': best_e,
            'hv': hv, 'time': rt, 'size': len(front)
        })
        all_archives.append((f"PSO_s{cfg['swarm_size']}_i{cfg['iterations']}", front))
    return all_metrics, all_archives

# --- STATICAL TESTS -----------------------------------------------------
def load_all_results(results_dir):
    """Load all CSV results into GA and PSO groups."""
    ga_configs = []
    pso_configs = []
    for filename in os.listdir(results_dir):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            metrics = []
            for row in reader:
                # Convert metrics to appropriate types
                conv_row = {
                    'run': int(row['run']),
                    'cost': float(row['cost']),
                    'emis': float(row['emis']),
                    'hv': float(row['hv']),
                    'time': float(row['time']),
                    'size': int(row['size'])
                }
                metrics.append(conv_row)
            if filename.startswith('GA'):
                ga_configs.append((filename[:-4], metrics))
            elif filename.startswith('PSO'):
                pso_configs.append((filename[:-4], metrics))
    return ga_configs, pso_configs

# --- STATISTICAL TESTS ----------------------------------------------------
def pairwise_wilcoxon(data_dict, alpha=0.05):
    results = {}
    for (a, x), (b, y) in combinations(data_dict.items(), 2):
        stat, p = wilcoxon(x, y)
        results[(a, b)] = (p, p<alpha)
    return results

# --- ANALYSIS & PLOTTING -------------------------------------------------
def perform_analysis(results_dir):
    # Prepare directories
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load CSVs into GA/PSO dicts
    ga, pso = {}, {}
    for f in os.listdir(results_dir):
        if not f.endswith('.csv'): continue
        tag = f[:-4]
        with open(os.path.join(results_dir, f)) as ff:
            reader = csv.DictReader(ff)
            vals = [float(row['hv']) for row in reader]
        if tag.startswith('GA'):
            ga[tag] = vals
        else:
            pso[tag] = vals

    # Plot boxplots for GA
    if ga:
        plt.figure(figsize=(8, 6))
        labels, data = zip(*sorted(ga.items()))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha='right')
        plt.title('NSGA-II Hypervolume Comparison')
        plt.ylabel('Hypervolume')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'ga_hv_boxplot.png'), dpi=150)
        plt.close()

    # Plot boxplots for PSO
    if pso:
        plt.figure(figsize=(8, 6))
        labels, data = zip(*sorted(pso.items()))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha='right')
        plt.title('MOPSO Hypervolume Comparison')
        plt.ylabel('Hypervolume')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'pso_hv_boxplot.png'), dpi=150)
        plt.close()

    # Combined GA vs PSO
    if ga and pso:
        combined = {**ga, **pso}
        plt.figure(figsize=(10, 6))
        labels, data = zip(*sorted(combined.items()))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=90, ha='right')
        plt.title('GA vs PSO Hypervolume Comparison')
        plt.ylabel('Hypervolume')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'ga_vs_pso_hv_boxplot.png'), dpi=150)
        plt.close()

    # Heatmap of pairwise Mann-Whitney p-values
    if ga and pso:
        tags = sorted(list(ga.keys()) + list(pso.keys()))
        n = len(tags)
        pval_mat = np.ones((n, n))
        for i in range(n):
            for j in range(i+1, n):
                a, b = tags[i], tags[j]
                x = ga[a] if a in ga else pso[a]
                y = ga[b] if b in ga else pso[b]
                _, p = mannwhitneyu(x, y, alternative='two-sided')
                pval_mat[i,j] = pval_mat[j,i] = p
        # Plot
        plt.figure(figsize=(8, 6))
        im = plt.imshow(-np.log10(pval_mat+1e-16), aspect='auto')
        plt.colorbar(im, label='-log10(p-value)')
        plt.xticks(range(n), tags, rotation=90)
        plt.yticks(range(n), tags)
        plt.title('Pairwise Significance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'pvalue_heatmap.png'), dpi=150)
        plt.close()

    print(f"Plots saved to: {PLOTS_DIR}")

# --- ENTRY POINT ---------------------------------------------------------

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cap_file", help="OR-Library .cap file")
    p.add_argument("--runs",      type=int, default=DEFAULT_RUNS)
    p.add_argument("--ga-only",   action="store_true")
    p.add_argument("--pso-only",  action="store_true")
    p.add_argument("--plot",      action="store_true",
                   help="overlay Pareto fronts after all runs")
    args = p.parse_args()

    instance = args.cap_file
    runs     = args.runs
    
    base = os.path.splitext(instance)[0]

    # Prepare results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # NSGA-II experiments
    if not args.pso_only:
        for cfg in GA_GRID:
            tag = f"GA_p{cfg['pop_size']}_g{cfg['generations']}"
            print(f"\n→ Running {tag} ({runs}x)")
            metrics, archives = run_nsga2(instance, cfg, runs)
            # Write CSV
            csv_path = f"{RESULTS_DIR}/{tag}.csv"
            write_csv(csv_path, metrics,
                      ['run','cost','emis','hv','time','size'])
            # Print summary
            print("  ", summary_stats("Cost",   [m['cost'] for m in metrics]))
            print("  ", summary_stats("Emis",   [m['emis'] for m in metrics]))
            print("  ", summary_stats("HV",     [m['hv']   for m in metrics]))
            print("  ", summary_stats("Time[s]", [m['time'] for m in metrics]))

    # MOPSO experiments
    if not args.ga_only:
        for cfg in PSO_GRID:
            tag = f"PSO_s{cfg['swarm_size']}_i{cfg['iterations']}"
            print(f"\n→ Running {tag} ({runs}x)")
            metrics, archives = run_mopso(instance, cfg, runs)
            csv_path = f"{RESULTS_DIR}/{tag}.csv"
            write_csv(csv_path, metrics,
                      ['run','cost','emis','hv','time','size'])
            print("  ", summary_stats("Cost",   [m['cost'] for m in metrics]))
            print("  ", summary_stats("Emis",   [m['emis'] for m in metrics]))
            print("  ", summary_stats("HV",     [m['hv']   for m in metrics]))
            print("  ", summary_stats("Time[s]", [m['time'] for m in metrics]))

    # Overlay Pareto fronts
    if args.plot:
        combined = {}
        if not args.pso_only:
            for cfg in GA_GRID:
                tag = f"GA_p{cfg['pop_size']}_g{cfg['generations']}"
                _, archives = run_nsga2(instance, cfg, 1)  # single run for plot
                for label, front in archives:
                    combined[label] = front
        if not args.ga_only:
            for cfg in PSO_GRID:
                tag = f"PSO_s{cfg['swarm_size']}_i{cfg['iterations']}"
                _, archives = run_mopso(instance, cfg, 1)
                for label, front in archives:
                    combined[label] = front
        plot_all_fronts(combined, f"Pareto Fronts on {os.path.basename(instance)}",
                        f"{RESULTS_DIR}/overlay_{os.path.basename(instance)}.png")

    # Perform statistical analysis
    print("\n\n=== Performing statistical analysis ===")
    perform_analysis(RESULTS_DIR)
