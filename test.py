import eas
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Fitness Functions
# ----------------------------------------

def maxones(individual):
    # Handles single individual
    if (individual + 1).ndim == 1:
        return np.mean(individual + 1) / 2
    return np.mean(individual + 1, axis=1) / 2


def variety(individual):
    rounded = np.round(individual, 2)

    # Single individual
    if rounded.ndim == 1:
        unique = len(np.unique(rounded))
        size = len(rounded)
        return 1.0 if unique == 1 else 1.1 - (unique / size)

    # Population
    unique_counts = np.apply_along_axis(lambda row: len(np.unique(row)), 1, rounded)
    size = rounded.shape[1]
    return np.where(unique_counts == 1, 1.0, 1.1 - (unique_counts / size))


# ----------------------------------------
# Algorithm + Problem Settings
# ----------------------------------------

population = 10
geneSize = 10
mutationProbability = 0.1
recombinationProbability = 0.5
generations = 500

choice = 1   # 1: HC | 2: PHC | 3: Microbial GA | 4: GA
problem = 1  # 1: MaxOnes | 2: Variety

# ----------------------------------------
# Algorithm Selection
# ----------------------------------------

if choice == 1:
    focus = eas.HillClimber(
        maxones if problem == 1 else variety,
        geneSize,
        mutationProbability,
        generations
    )

elif choice == 2:
    focus = eas.ParallelHillClimber(
        population,
        maxones if problem == 1 else variety,
        geneSize,
        mutationProbability,
        generations
    )

elif choice == 3:
    focus = eas.MicrobialGA(
        population,
        maxones if problem == 1 else variety,
        geneSize,
        mutationProbability,
        recombinationProbability,
        generations
    )

else:
    focus = eas.GA(
        population,
        maxones if problem == 1 else variety,
        geneSize,
        mutationProbability,
        recombinationProbability,
        generations
    )

# ----------------------------------------
# Run + Visualization
# ----------------------------------------

focus.run()

plt.plot(focus.fHistory)
plt.xlabel("Generations")
plt.ylabel("Fitness")

titles = {
    (1, 1): "Hill Climber — Max Ones",
    (1, 2): "Hill Climber — Variety",
    (2, 1): "Parallel Hill Climber — Max Ones",
    (2, 2): "Parallel Hill Climber — Variety",
    (3, 1): "Microbial Genetic Algorithm — Max Ones",
    (3, 2): "Microbial Genetic Algorithm — Variety",
    (4, 1): "Genetic Algorithm — Max Ones",
    (4, 2): "Genetic Algorithm — Variety",
}

plt.title(
    f"{titles[(choice, problem)]} "
    f"(Population={population}, GeneSize={geneSize}, "
    f"MP={mutationProbability}, RP={recombinationProbability})"
)

plt.show()

# ----------------------------------------
# Results
# ----------------------------------------

print("Final Fitness:", focus.fHistory[-1])
print("Final Individual(s):")
print(focus.i)
