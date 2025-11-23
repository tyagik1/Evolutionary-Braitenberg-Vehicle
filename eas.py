import numpy as np

class HillClimber():
    def __init__(self, ff, gs, mp, g):
        self.ff = ff # Fitness Function
        self.gs = gs # Gene Size
        self.mp = mp # Mutation Probability determines mutation size
        self.g = g # Generations

        # Create a single individual with values in range [-1, 1]
        self.i = np.random.rand(self.gs) * 2 - 1  
        # Evaluate initial fitness
        self.f = self.ff(self.i)  
        # Track fitness across generations
        self.fHistory = np.zeros(self.g)  
        self.fHistory[0] = self.f  # Store initial fitness

    def run(self):
        # Loop through generations 1 → g-1
        for i in range(1, self.g):
            # Create mutated child from current best individual
            child = self.i + np.random.normal(0.0, self.mp, size=self.gs)
            # Clip gene values to allowable range [-1, 1]
            child = np.clip(child, -1, 1)

            # Compute child's fitness
            childFitness = self.ff(child)

            # Selection: child replaces parent if better or equal
            if childFitness >= self.f:
                self.i = child
                self.f = childFitness

            # Record best fitness so far
            self.fHistory[i] = self.f



class ParallelHillClimber():
    def __init__(self, p, ff, gs, mp, g):
        self.p = p # Population
        self.ff = ff # Fitness Function
        self.gs = gs # Gene Size
        self.mp = mp # Mutation Probability determines mutation size
        self.g = g # Generations

        # Create population of individuals, each in [-1,1]
        self.i = np.random.rand(self.p, self.gs) * 2 - 1
        # Compute initial fitness for every individual
        self.f = self.ff(self.i)
        # Fitness history: store fitness over time for all individuals
        self.fHistory = np.zeros((self.g, self.p))
        self.fHistory[0] = self.f

    def run(self):
        for i in range(1, self.g):
            # Mutate entire population in parallel
            children = self.i + np.random.normal(0.0, self.mp, size=(self.p, self.gs))
            children = np.clip(children, -1, 1)

            # Evaluate fitness for all children
            childrenFitness = self.ff(children)

            # Replace parents with children when better
            for j in range(self.p):
                if childrenFitness[j] >= self.f[j]:
                    self.i[j] = children[j]
                    self.f[j] = childrenFitness[j]

            # Save current generation fitness
            self.fHistory[i] = self.f



class MicrobialGA():
    def __init__(self, p, ff, gs, mp, rp, g):
        self.p = p # Population
        self.ff = ff # Fitness Function
        self.gs = gs # Gene Size
        self.mp = mp # Mutation Probability determines mutation size
        self.rp = rp # Recombination Probability
        self.g = g # Generations

        # Total number of tournaments = population × generations
        self.t = int(self.g * self.p)

        # Initial population with values in [-1,1]
        self.i = np.random.rand(self.p, self.gs) * 2 - 1
        self.f = self.ff(self.i)
        # Only track best fitness each generation
        self.fHistory = np.zeros(self.g)

    def run(self):
        for i in range(self.g):
            bestF = 0.0  # Track best fitness in this generation

            # Perform microbial tournaments
            for j in range(1, self.t):
                # Select two distinct individuals
                a = np.random.randint(0, self.p)
                b = np.random.randint(0, self.p)
                while a == b:
                    b = np.random.randint(0, self.p)

                fA = self.ff(self.i[a])
                fB = self.ff(self.i[b])

                # Determine winner and loser
                winner = a
                loser = b
                if fA < fB:
                    winner = b
                    loser = a

                # Track best fitness found so far
                if self.ff(self.i[winner]) >= bestF:
                    bestF = self.ff(self.i[winner])

                # Recombine winner into loser
                for k in range(self.gs):
                    if np.random.random() < self.rp:
                        self.i[loser][k] = self.i[winner][k]

                # Mutate loser
                self.i[loser] += np.random.normal(0.0, self.mp, size=self.gs)
                self.i[loser] = np.clip(self.i[loser], -1, 1)

            # Save best fitness
            self.fHistory[i] = bestF



class GA():
    def __init__(self, p, ff, gs, mp, rp, g):
        self.p = p # Population
        self.ff = ff # Fitness Function
        self.gs = gs # Gene Size
        self.mp = mp # Mutation Probability determines mutation size
        self.rp = rp # Recombination Probability
        self.g = g # Generations

        # Initial population
        self.i = np.random.rand(self.p, self.gs) * 2 - 1
        # Fitness vector
        self.f = np.zeros(self.p)
        # Track fitness history for all individuals
        self.fHistory = np.zeros((self.g, self.p))
        # Ranking array
        self.rank = np.zeros(self.p, dtype=int)

    def rankChoose(self, value):
        raw_value = value
        value = int(self.p * (self.p + 1) * value / 2)
        # print(f"raw={raw_value:.4f}, mapped={value}")

        rank = 0
        while (rank + 1) * (rank + 2) / 2 < value:
            rank += 1
        return rank

    def run(self):
        for i in range(self.g):
            # Evaluate fitness
            for j in range(self.p):
                self.f[j] = self.ff(self.i[j])

            self.fHistory[i] = self.f

            # Rank individuals by fitness (descending)
            fCopy = self.f.copy()
            self.rank = np.argsort(-fCopy)

            newI = np.zeros((self.p, self.gs))

            # Build next generation
            for j in range(self.p):
                # Select parents using rank-based selection
                a = self.rank[self.rankChoose(np.random.random())]
                b = self.rank[self.rankChoose(np.random.random())]

                while a == b:
                    b = self.rank[self.rankChoose(np.random.random())]

                # Recombine genes
                for k in range(self.gs):
                    if np.random.random() < self.rp:
                        newI[j][k] = self.i[a][k]
                    else:
                        newI[j][k] = self.i[b][k]

                # Mutate and clip gene values
                newI[j] += np.random.normal(0.0, self.mp, size=self.gs)
                newI[j] = np.clip(newI[j], -1, 1)

            self.i = newI.copy()



class Braitenberg():
    def __init__(self):
        self.x = 0.0 # X-position
        self.y = 0.0 # Y-position
        self.o = np.random.choice([0, 0.5, 1, 1.5]) * np.pi # Orientation
        self.v = 1.0 # Velocity
        self.r = 1.0 # Size
        self.ls = 0.0 # Left Sensor
        self.rs = 0.0 # Right Sensor
        self.lm = 0.0 # Left Motor
        self.rm = 0.0 # Right Motor
        self.a = np.pi / 2 # Angle offset of sensors

        # Sensor positions
        self.rsX = self.r * np.cos(self.o + self.a)
        self.rsY = self.r * np.sin(self.o + self.a)
        self.lsX = self.r * np.cos(self.o - self.a)
        self.lsY = self.r * np.sin(self.o - self.a)

        self.tg = 1/10.0 # Turning gain
        self.vg = 1/10.0 # Velocity gain
        self.sg = 1/10.0 # Sensor gain

        # Gene list (modifies robot behavior)
        self.g = []
        self.g.append(self.vg)
        self.g.append(self.tg)
        self.g.append(self.sg)

        self.mp = 0.1
        self.rp = 0.5

    def move(self):
        # Update orientation based on motor difference
        self.o += self.tg * (self.rm - self.lm)

        # Update velocity from motor signals
        self.v = self.vg * (self.lm + self.rm)

        # Update position
        self.x += self.v * np.cos(self.o)
        self.y += self.v * np.sin(self.o)

        # Update sensor world positions
        self.rsX = self.x + self.r*np.cos(self.o+self.a)
        self.rsY = self.y + self.r*np.sin(self.o+self.a)
        self.lsX = self.x + self.r*np.cos(self.o-self.a)
        self.lsY = self.y + self.r*np.sin(self.o-self.a)
    
    def sense(self, light, size):
        # Compute distance to light for each sensor
        self.ls = self.sg*np.sqrt((self.lsX - light.x)**2 + (self.lsY - light.y)**2)
        self.rs = self.sg*np.sqrt((self.rsX - light.x)**2 + (self.rsY - light.y)**2)

        # Clip to avoid extreme values
        self.ls = np.clip(self.ls, 0, size)
        self.rs = np.clip(self.rs, 0, size)

    def thinkHillClimber(self, duration):
        # Motor signals based on inverse distance
        self.lm = 1.0 / (self.ls)
        self.rm = 1.0 / (self.rs)
        self.lm = np.clip(self.lm, 0.0, 10.0)
        self.rm = np.clip(self.rm, 0.0, 10.0)

        # Mutate genes
        for i in range(3):
            self.g[i] += np.random.normal(0.0, self.mp, size=5)
            self.g[i] = np.clip(self.g[i], 0.0, 10.0)

    def thinkParallelHillClimber(self, duration):
        pass

    def thinkMicrobialGA(self, duration):
        pass

    def thinkGA(self, duration):
        pass



class BraitenbergVehicle():
    def __init__(self, tg, vg, sg):
        self.x = 0.0
        self.y = 0.0
        self.o = np.random.choice([0, 0.5, 1, 1.5]) * np.pi
        self.v = 1.0
        self.r = 1.0

        self.ls = 0.0 # Left Sensor
        self.rs = 0.0 # Right Sensor
        self.lm = 0.0 # Left Motor
        self.rm = 0.0 # Right Motor

        self.a = np.pi/2 # Sensor angle offset

        # Sensor coordinates
        self.rsX = self.r*np.cos(self.o+self.a)
        self.rsY = self.r*np.sin(self.o+self.a)
        self.lsX = self.r*np.cos(self.o-self.a)
        self.lsY = self.r*np.sin(self.o-self.a)

        # Gains
        self.tg = tg
        self.vg = vg
        self.sg = sg

    def move(self):
        # Turn based on motor difference
        self.o += self.tg * (self.rm - self.lm)

        # Motor values from inverse distance
        self.lm = 1.0 / (self.ls)
        self.rm = 1.0 / (self.rs)

        # Bound motor values
        self.lm = np.clip(self.lm, 0.0, 10.0)
        self.rm = np.clip(self.rm, 0.0, 10.0)

        # Forward velocity
        self.v = self.vg * (self.lm + self.rm)

        # Update position
        self.x += self.v*np.cos(self.o)
        self.y += self.v*np.sin(self.o)

        # Update sensor coordinates
        self.rsX = self.x + self.r*np.cos(self.o+self.a)
        self.rsY = self.y + self.r*np.sin(self.o+self.a)
        self.lsX = self.x + self.r*np.cos(self.o-self.a)
        self.lsY = self.y + self.r*np.sin(self.o-self.a)

        # Stop vehicle if very close to light
        if self.ls < 0.5 or self.rs < 0.5:
            self.lm = 0.0
            self.rm = 0.0
            self.vg = 0.0
            self.tg = 0.0
    
    def sense(self, light, size):
        # Distance to light for each sensor
        self.ls = self.sg*np.sqrt((self.lsX - light.x)**2 + (self.lsY - light.y)**2)
        self.rs = self.sg*np.sqrt((self.rsX - light.x)**2 + (self.rsY - light.y)**2)

        # Clip to avoid division by zero
        self.ls = np.clip(self.ls, 0.0001, size)
        self.rs = np.clip(self.rs, 0.0001, size)



class LightSource():
    def __init__(self, size):
        # Random point in square region
        self.x = np.random.randint(1 - size, size)
        self.y = np.random.randint(1 - size, size)

        # Ensure light is outside radius-1 circle around origin
        while np.sqrt(self.x**2 + self.y**2) <= 1:
            self.x = np.random.randint(1 - size, size)
            self.y = np.random.randint(1 - size, size)
