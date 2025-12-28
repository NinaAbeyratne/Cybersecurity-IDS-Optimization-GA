"""
Genetic Algorithm for Feature Selection

Implements GA-based feature selection for intrusion detection optimization.
This is the core evolutionary computing component of the solution.

Dataset: CICIDS2017
"""

import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import time
import matplotlib.pyplot as plt


class GeneticFeatureSelector:
    """
    Genetic Algorithm for feature selection in intrusion detection.
    
    Chromosome Representation:
        Binary vector of length d (number of features)
        Gene g_i ∈ {0,1} indicates feature i is selected (1) or not (0)
    
    Fitness Function:
        fitness = α * F1_score - β * (k/d)
        where:
            k = number of selected features
            d = total number of features
            α = weight for model performance (default: 1.0)
            β = weight for feature reduction (default: 0.05)
    
    Constraints:
        - Minimum features: k >= k_min (default: 10)
        - Maximum features: k <= k_max (optional)
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, 
                 alpha=1.0, beta=0.05, k_min=10, k_max=None,
                 pop_size=50, n_generations=30, cx_prob=0.7, mut_prob=0.2,
                 tournament_size=3, random_state=42):
        """
        Initialize GA feature selector.
        
        Args:
            X_train, y_train: Training data (numpy arrays)
            X_val, y_val: Validation data (numpy arrays)
            alpha: Weight for F1 score in fitness
            beta: Weight for feature reduction penalty
            k_min: Minimum number of features to select
            k_max: Maximum number of features (None = no limit)
            pop_size: Population size
            n_generations: Number of generations
            cx_prob: Crossover probability
            mut_prob: Mutation probability
            tournament_size: Tournament selection size
            random_state: Random seed
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.d = X_train.shape[1]  # Number of features
        self.alpha = alpha
        self.beta = beta
        self.k_min = k_min
        self.k_max = k_max if k_max else self.d
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        
        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)
        
        # DEAP setup
        self._setup_deap()
        
        # Results tracking
        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = {'max': [], 'mean': [], 'min': []}
        self.generation_times = []
        
    def _setup_deap(self):
        """Setup DEAP framework for GA"""
        # Clear any existing creator classes
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_bool, self.d)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
    
    def _evaluate_individual(self, individual):
        """
        Evaluate fitness of an individual (feature subset).
        
        Fitness = α * F1_score - β * (k/d)
        
        Args:
            individual: Binary chromosome
            
        Returns:
            Tuple with fitness value
        """
        # Convert to boolean mask
        mask = np.array(individual, dtype=bool)
        k = mask.sum()
        
        # Enforce constraints
        if k < self.k_min or k > self.k_max:
            # Heavy penalty for violating constraints
            return (-1.0,)
        
        # Select features
        X_train_sel = self.X_train[:, mask]
        X_val_sel = self.X_val[:, mask]
        
        # Train lightweight classifier
        try:
            clf = LogisticRegression(max_iter=1000, n_jobs=1, 
                                    class_weight='balanced', solver='lbfgs')
            clf.fit(X_train_sel, self.y_train)
            
            # Evaluate on validation set
            y_pred = clf.predict(X_val_sel)
            f1 = f1_score(self.y_val, y_pred, zero_division=0)
            
            # Calculate fitness
            fitness = self.alpha * f1 - self.beta * (k / self.d)
            
            return (fitness,)
        
        except Exception as e:
            # Return penalty if training fails
            print(f"Warning: Evaluation failed for subset size {k}: {e}")
            return (-1.0,)
    
    def run(self, verbose=True):
        """
        Execute the genetic algorithm.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Best individual (feature mask) and its fitness
        """
        print("\n" + "="*70)
        print("GENETIC ALGORITHM FOR FEATURE SELECTION")
        print("="*70)
        print(f"Total features:       {self.d}")
        print(f"Feature range:        [{self.k_min}, {self.k_max}]")
        print(f"Population size:      {self.pop_size}")
        print(f"Generations:          {self.n_generations}")
        print(f"Crossover prob:       {self.cx_prob}")
        print(f"Mutation prob:        {self.mut_prob}")
        print(f"Fitness weights:      α={self.alpha}, β={self.beta}")
        print("="*70)
        
        # Initialize population
        pop = self.toolbox.population(n=self.pop_size)
        
        # Evaluate initial population
        print("\nEvaluating initial population...")
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Track best individual
        self.best_individual = tools.selBest(pop, 1)[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        # Evolution loop
        start_time = time.time()
        
        for gen in range(self.n_generations):
            gen_start = time.time()
            
            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population with elitism
            pop[:] = tools.selBest(pop + offspring, k=len(pop))
            
            # Track statistics
            fits = [ind.fitness.values[0] for ind in pop]
            self.fitness_history['max'].append(max(fits))
            self.fitness_history['mean'].append(np.mean(fits))
            self.fitness_history['min'].append(min(fits))
            
            # Update best
            current_best = tools.selBest(pop, 1)[0]
            if current_best.fitness.values[0] > self.best_fitness:
                self.best_individual = current_best
                self.best_fitness = current_best.fitness.values[0]
            
            gen_time = time.time() - gen_start
            self.generation_times.append(gen_time)
            
            if verbose:
                best_k = np.sum(self.best_individual)
                print(f"Gen {gen+1:3d}: Best={self.best_fitness:.4f} (k={best_k}), "
                      f"Mean={self.fitness_history['mean'][-1]:.4f}, Time={gen_time:.2f}s")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("GENETIC ALGORITHM COMPLETED")
        print("="*70)
        print(f"Total time:          {total_time:.2f} seconds")
        print(f"Time per generation: {total_time/self.n_generations:.2f} seconds")
        print(f"Best fitness:        {self.best_fitness:.4f}")
        print(f"Features selected:   {np.sum(self.best_individual)} / {self.d}")
        print("="*70)
        
        return self.best_individual, self.best_fitness
    
    def get_selected_features(self, feature_names=None):
        """
        Get list of selected feature names/indices.
        
        Args:
            feature_names: List of feature names (optional)
            
        Returns:
            List of selected features
        """
        if self.best_individual is None:
            raise ValueError("GA has not been run yet")
        
        mask = np.array(self.best_individual, dtype=bool)
        
        if feature_names is not None:
            return [feature_names[i] for i in range(len(mask)) if mask[i]]
        else:
            return np.where(mask)[0].tolist()
    
    def get_feature_mask(self):
        """
        Get boolean mask of selected features.
        
        Returns:
            Boolean numpy array
        """
        if self.best_individual is None:
            raise ValueError("GA has not been run yet")
        
        return np.array(self.best_individual, dtype=bool)
    
    def plot_fitness_evolution(self, save_path=None):
        """
        Plot fitness evolution over generations.
        
        Args:
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(12, 6))
        
        generations = range(1, len(self.fitness_history['max']) + 1)
        
        plt.plot(generations, self.fitness_history['max'], 'g-', label='Best', linewidth=2)
        plt.plot(generations, self.fitness_history['mean'], 'b--', label='Mean', linewidth=1.5)
        plt.plot(generations, self.fitness_history['min'], 'r:', label='Worst', linewidth=1)
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_generation_times(self, save_path=None):
        """
        Plot time taken per generation.
        
        Args:
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(12, 5))
        
        generations = range(1, len(self.generation_times) + 1)
        
        plt.bar(generations, self.generation_times, color='steelblue', alpha=0.7)
        plt.axhline(y=np.mean(self.generation_times), color='r', 
                   linestyle='--', label=f'Mean: {np.mean(self.generation_times):.2f}s')
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title('Generation Execution Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("Genetic Algorithm module loaded")
    print("\nChromosome representation: Binary vector [g1, g2, ..., gd]")
    print("Fitness function: α*F1 - β*(k/d)")
    print("Constraints: k_min <= k <= k_max")



# import random
# import time
# import numpy as np
# import matplotlib.pyplot as plt

# from deap import base, creator, tools

# from src.models import LightweightClassifier


# class GeneticFeatureSelector:
#     """
#     Genetic Algorithm for feature selection
#     """

#     def __init__(
#         self,
#         X_train,
#         y_train,
#         X_val,
#         y_val,
#         alpha=1.0,
#         beta=0.05,
#         k_min=10,
#         k_max=None,
#         pop_size=50,
#         n_generations=30,
#         cx_prob=0.7,
#         mut_prob=0.2,
#         tournament_size=3,
#         random_state=42
#     ):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_val = X_val
#         self.y_val = y_val

#         self.alpha = alpha
#         self.beta = beta
#         self.k_min = k_min
#         self.k_max = k_max or X_train.shape[1]

#         self.pop_size = pop_size
#         self.n_generations = n_generations
#         self.cx_prob = cx_prob
#         self.mut_prob = mut_prob
#         self.tournament_size = tournament_size

#         self.n_features = X_train.shape[1]

#         random.seed(random_state)
#         np.random.seed(random_state)

#         self.best_individual = None
#         self.best_fitness = None
#         self.fitness_history = []
#         self.generation_times = []

#         self._setup_deap()

#     def _setup_deap(self):
#         """
#         Configure DEAP toolbox
#         """
#         creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#         creator.create("Individual", list, fitness=creator.FitnessMax)

#         self.toolbox = base.Toolbox()

#         self.toolbox.register(
#             "attr_bool",
#             random.randint,
#             0,
#             1
#         )

#         self.toolbox.register(
#             "individual",
#             tools.initRepeat,
#             creator.Individual,
#             self.toolbox.attr_bool,
#             n=self.n_features
#         )

#         self.toolbox.register(
#             "population",
#             tools.initRepeat,
#             list,
#             self.toolbox.individual
#         )

#         self.toolbox.register(
#             "evaluate",
#             self._fitness_function
#         )

#         self.toolbox.register(
#             "select",
#             tools.selTournament,
#             tournsize=self.tournament_size
#         )

#         self.toolbox.register(
#             "mate",
#             tools.cxUniform,
#             indpb=0.5
#         )

#         self.toolbox.register(
#             "mutate",
#             tools.mutFlipBit,
#             indpb=0.02
#         )

#     def _fitness_function(self, individual):
#         """
#         Fitness = α·F1 − β·(k/d)
#         """
#         selected = np.array(individual, dtype=bool)
#         k = np.sum(selected)

#         if k < self.k_min or k > self.k_max:
#             return (-1.0,)

#         X_train_sel = self.X_train[:, selected]
#         X_val_sel = self.X_val[:, selected]

#         clf = LightweightClassifier()
#         clf.fit(X_train_sel, self.y_train)

#         f1 = clf.f1_score(X_val_sel, self.y_val)

#         penalty = self.beta * (k / self.n_features)
#         fitness = self.alpha * f1 - penalty

#         return (fitness,)

#     def run(self, verbose=True):
#         """
#         Execute GA
#         """
#         population = self.toolbox.population(n=self.pop_size)

#         for gen in range(self.n_generations):
#             start_time = time.time()

#             offspring = self.toolbox.select(population, len(population))
#             offspring = list(map(self.toolbox.clone, offspring))

#             for c1, c2 in zip(offspring[::2], offspring[1::2]):
#                 if random.random() < self.cx_prob:
#                     self.toolbox.mate(c1, c2)
#                     del c1.fitness.values
#                     del c2.fitness.values

#             for mutant in offspring:
#                 if random.random() < self.mut_prob:
#                     self.toolbox.mutate(mutant)
#                     del mutant.fitness.values

#             invalid = [ind for ind in offspring if not ind.fitness.valid]
#             fitnesses = map(self.toolbox.evaluate, invalid)
#             for ind, fit in zip(invalid, fitnesses):
#                 ind.fitness.values = fit

#             population[:] = offspring

#             best = tools.selBest(population, 1)[0]
#             self.best_individual = best
#             self.best_fitness = best.fitness.values[0]

#             self.fitness_history.append(self.best_fitness)
#             self.generation_times.append(time.time() - start_time)

#             if verbose:
#                 print(
#                     f"Generation {gen+1}/{self.n_generations} | "
#                     f"Best fitness: {self.best_fitness:.4f}"
#                 )

#         return self.best_individual, self.best_fitness

#     def get_feature_mask(self):
#         return np.array(self.best_individual, dtype=bool)

#     def get_selected_features(self, feature_names):
#         mask = self.get_feature_mask()
#         return [f for f, m in zip(feature_names, mask) if m]

#     def plot_fitness_evolution(self, save_path=None):
#         plt.figure()
#         plt.plot(self.fitness_history)
#         plt.xlabel("Generation")
#         plt.ylabel("Best Fitness")
#         plt.title("GA Fitness Evolution")
#         if save_path:
#             plt.savefig(save_path)
#         plt.close()

#     def plot_generation_times(self, save_path=None):
#         plt.figure()
#         plt.plot(self.generation_times)
#         plt.xlabel("Generation")
#         plt.ylabel("Time (s)")
#         plt.title("GA Generation Time")
#         if save_path:
#             plt.savefig(save_path)
#         plt.close()
