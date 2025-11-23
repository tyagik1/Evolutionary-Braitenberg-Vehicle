import eas
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# History Helper Class
# ============================================================

class History:
    def __init__(self, xHistory, yHistory):
        self.xHistory = xHistory
        self.yHistory = yHistory


# ============================================================
# Vehicle Simulation
# ============================================================

def simVehicle(vehicle, duration, light):
    xHistory = np.zeros(duration + 1)
    yHistory = np.zeros(duration + 1)

    xHistory[0] = vehicle.x
    yHistory[0] = vehicle.y

    for i in range(duration):
        vehicle.sense(light, int(np.sqrt(duration)))
        vehicle.move()
        xHistory[i + 1] = vehicle.x
        yHistory[i + 1] = vehicle.y

    return History(xHistory, yHistory)

# ============================================================
# Fitness Function
# ============================================================

duration = 10000  # steps

def fitnessFunction(gene):
    light = eas.LightSource(int(np.sqrt(duration)))
    vehicle = eas.BraitenbergVehicle(gene[0], gene[1], gene[2])

    history = simVehicle(vehicle, duration, light)

    start_dist = np.sqrt((history.xHistory[0] - light.x)**2 +
                         (history.yHistory[0] - light.y)**2)

    end_dist = np.sqrt((history.xHistory[-1] - light.x)**2 +
                       (history.yHistory[-1] - light.y)**2)

    score = np.clip(1 - (end_dist / start_dist), 0, 1)

    return score, history, np.array([light.x, light.y])


# ============================================================
# Hill Climber for Braitenberg Vehicle
# ============================================================

def climberBV():
    gene = np.zeros(3)  # [turn_gain, vel_gain, sensor_gain]

    fHistory = np.zeros(generations)
    fitness, history, light = fitnessFunction(gene)
    fHistory[0] = fitness

    for i in range(1, generations):
        child = gene + np.random.normal(0.0, mutationProbability, size=3)

        child[0] = np.clip(child[0], -250, 250)
        child[1] = np.clip(child[1], 0, 20)
        child[2] = np.clip(child[2], 0, 10)

        childFitness, newHistory, newLight = fitnessFunction(child)

        if childFitness >= fitness:
            gene = child
            fitness = childFitness
            history = newHistory
            light = newLight

        fHistory[i] = fitness

    # ---- Plot Fitness Curve ----
    plt.plot(fHistory)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title(f"Hill Climber — Braitenberg Vehicle (MP={mutationProbability})")
    plt.show()

    print(
        f"Training Vehicle:\n"
        f"Start:  ({history.xHistory[0]}, {history.yHistory[0]})\n"
        f"End:    ({history.xHistory[-1]}, {history.yHistory[-1]})\n"
        f"Light:  ({light[0]}, {light[1]})\n"
        f"Score:  {fHistory[-1]}\n"
        f"Gene:   {gene}\n"
    )

    np.save("gene.npy", gene)


# ============================================================
# Fitness Utility
# ============================================================

def getFitnessScore(history, lightXY):
    start = np.sqrt((history.xHistory[0] - lightXY[0])**2 + (history.yHistory[0] - lightXY[1])**2)
    end = np.sqrt((history.xHistory[-1] - lightXY[0])**2 + (history.yHistory[-1] - lightXY[1])**2)
    return np.clip(1 - (end / start), 0, 1)


# ============================================================
# Visualization of Trained Vehicle Across Multiple Scenarios
# ============================================================

def showBVTogether(gene):
    fitnessValues = np.zeros(numLights)

    for i in range(numLights):

        light = eas.LightSource(int(np.sqrt(duration)))
        vehicle = eas.BraitenbergVehicle(*gene)

        history = simVehicle(vehicle, duration, light)
        lightXY = np.array([light.x, light.y])

        fitnessValues[i] = getFitnessScore(history, lightXY)

        x, y = history.xHistory, history.yHistory
        lightX, lightY = light.x, light.y

        plt.plot(lightX, lightY, 'mo')   # light
        plt.plot(x, y)                   # trajectory
        plt.plot(x[0], y[0], 'ro')       # start
        plt.plot(x[-1], y[-1], 'ko')     # end

        print(
            f"Vehicle {i + 1}:\n"
            f"  Start:  ({x[0]}, {y[0]})\n"
            f"  End:    ({x[-1]}, {y[-1]})\n"
            f"  Light:  ({lightX}, {lightY})\n"
            f"  Score:  {fitnessValues[i]}\n"
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Braitenberg Vehicle — Trajectories Across Lights")
    plt.show()

    print(f"Mean Fitness Across {numLights} Lights = {np.mean(fitnessValues)}")


# ============================================================
# Testing With Saved Gene
# ============================================================

def testBV():
    gene = np.load("gene.npy")
    showBVTogether(gene)


# ============================================================
# Main Execution
# ============================================================

generations = 500
mutationProbability = 0.1
numLights = 10

climberBV()

testBV()
