from math import log2, ceil
from typing import List, Tuple
from random import choice, choices, randint, random, uniform

input = open("fisier.in", "r")
output = open("fisier.out", "w")

# types for genetics
Chromosome = List[int]
Population = List[Chromosome]

# types for function
Bounds = Tuple[int, int]
Parameters = Tuple[int, int, int, int]


# def f(x: float, par: Parameters) -> float:
#     return par[0] * x ** 2 + par[1] * x + par[2]
def f(x: float, par: Parameters) -> float:
     return par[0] * x ** 3 + par[1] * x ** 2 + par[2] * x + par[3]


def chromosome_base(length: int) -> Chromosome:
    return choices([0, 1], k=length)


def chromosome_float(chromosome: Chromosome, bounds: Bounds, precision: int, length: int) -> float:
    bin = ''.join(str(i) for i in chromosome)
    bin = int(bin, 2)
    return (round(((bounds[1] - bounds[0]) / (2 ** length - 1) * bin + bounds[0]), precision))


def population(dimension: int, ch_length: int, bounds: Bounds, precision: int) -> Population:
    return [chromosome_base(ch_length) for _ in range(dimension)]


def probability(chromosome: Chromosome, population: Population, bounds: Bounds, precision: int, length: int,
                parameters: Parameters) -> float:
    return f(chromosome_float(chromosome, bounds, precision, length), parameters) / sum(
        [f(chromosome_float(c, bounds, precision, length), parameters) for c in population])


def search(x: int, interval: List[int]):
    left = 0
    right = len(interval)
    while left <= right:
        mid = left + (right - left) // 2
        if interval[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return left


def roulette(pop: Population, bounds: Bounds, precision: int, length: int, parameters: Parameters,
             first: bool = False) -> Population:
    suma = 0
    interval = [0]

    for i in range(len(pop)):
        suma += probability(pop[i], pop, bounds, precision, length, parameters)
        interval.append(suma)

    if first:
        output.write("\nIntervale probabilitati de selectie\n")
        for i in range(len(interval)):
            output.write(str(interval[i]) + " ")

        output.write("\n")

    interm = []
    for i in range(len(pop)):
        u = random()
        found = search(u, interval)
        if first:
            output.write("u= " + str(u) + " selectam cromozomul " + str(found) + '\n')

        interm.append(pop[found - 1])

    if first:
        output.write("\nDupa selectie:\n")
        for i in range(len(interm)):
            output.write(str(i) + ": " + ''.join(str(l) for l in interm[i]) + "\tx= " + str(
                chromosome_float(interm[i], bounds, precision, length)) + "\tf= " + str(
                f(chromosome_float(interm[i], bounds, precision, length), parameters)) + '\n')
    return interm


def flip(l) :
    new_l = []
    for i in range(len(l)-1, -1, -1):
        new_l.append(l[i])
    return new_l

def cross(chromo1: Chromosome, chromo2: Chromosome, ind1: int, ind2: int, first: bool = False):
    length = len(chromo1)
    if length < 2:
        return chromo1, chromo2

    point = randint(0, length - 1)
    if first:
        output.write("\nRecombinare dintre cromozomul {} cu cromozomul {}".format(ind1, ind2))
        output.write('\n' + ''.join(str(i) for i in chromo1) + " " + ''.join(str(i) for i in chromo2) + " punct " + str(point))
        output.write("\nRezultat " + ''.join(str(i) for i in chromo1[0:point] + chromo2[point:]) + " " + ''.join(str(i) for i in chromo2[0:point] + chromo1[point:]) + '\n')
        # output.write("\nRezultat " + ''.join(str(i) for i in chromo1[0:point] + flip(chromo2[point:])) + " " + ''.join(str(i) for i in chromo2[0:point] + flip(chromo1[point:])) + '\n')

    return chromo1[0:point] + chromo2[point:], chromo2[0:point] + chromo1[point:]


def mutation(pop: Population, mutation_pct: float, bounds: Bounds, precision: int, length: int, parameters: Parameters,
             first: bool = False):
    if first:
        output.write("\nProbabilitate de mutatie pentru fiecare gena {}\n".format(mutation_pct))
        output.write("Au fost modificati cromozomii:\n")
    for i in range(len(pop)):
        for j in range(len(pop[i])):
            u = uniform(0, 1)
            if u < mutation_pct:
                pop[i][j] = 1 if pop[i][j] == 0 else 0
                if first:
                    output.write(str(i) + '\n')
    if first:
        output.write("Dupa mutatie:\n")
        for i in range(len(pop)):
            output.write(str(i) + ": " + ''.join(str(l) for l in pop[i]) + "\tx= " + str(
                chromosome_float(pop[i], bounds, precision, length)) + "\tf= " + str(
                f(chromosome_float(pop[i], bounds, precision, length), parameters)) + '\n')


dimension: int = 20
bounds: Bounds = (-4, 2)
parameters: Parameters = (1, 3, -4, 7)
precision: int = 6
nr_generations: int = 50
crossover_probability = 0.25
mutation_probability = 0.01

length = ceil(log2((bounds[1] - bounds[0]) * 10 ** precision))
pop = population(dimension, length, bounds, precision)
elite = []
elites = []

output.write("Populatia initiala:\n")
for i in range(len(pop)):
    output.write(
        str(i) + ": " + ''.join(str(l) for l in pop[i]) + "\tx= " + str(
            chromosome_float(pop[i], bounds, precision, length)) + "\tf= " + str(
            f(chromosome_float(pop[i], bounds, precision, length), parameters)) + '\n')

output.write("\nProbabilitati selectie\n")
for i in range(len(pop)):
    output.write("cromozom " + str(i) + " probabilitate " + str(
        probability(pop[i], pop, bounds, precision, length, parameters)) + "\n")

for t in range(nr_generations):

    if t != 0:
        pop.sort(key=lambda t: f(chromosome_float(t, bounds, precision, length), parameters), reverse=True)
        next_gen = pop[0:2]  # pastram primii 2 cei mai buni
        elite.append((chromosome_float(pop[0], bounds, precision, length), f(chromosome_float(pop[0], bounds, precision, length), parameters)))
        elites.append(f(chromosome_float(pop[0], bounds, precision, length), parameters))

        next_gen += choices(population=pop, k=dimension - 2)
    else:
        next_gen = pop.copy()
    if t == 0:
        interm = roulette(next_gen, bounds, precision, length, parameters, True)
    else:
        interm = roulette(next_gen, bounds, precision, length, parameters)

    partikip = []
    if t == 0:
        output.write("\nProbabilitatea de incrucisare {}\n".format(crossover_probability))
    for i in range(len(next_gen)):
        u = uniform(0, 1)
        if t == 0:
            output.write(str(i) + ": " + ''.join(str(l) for l in next_gen[i]) + "\tu= " + str(u))
        if u < crossover_probability:
            if t == 0:
                output.write("<" + str(crossover_probability) + " participa")
            partikip.append(next_gen[i])
        if t == 0:
            output.write('\n')

    to_remove = []
    aux = len(partikip)
    for i in range(aux // 2):
        father = choice(partikip)
        ind1 = next_gen.index(father)
        partikip.remove(father)
        to_remove.append(father)

        mother = choice(partikip)
        ind2 = next_gen.index(mother)
        partikip.remove(mother)
        to_remove.append(mother)

        if t == 0:
            child1, child2 = cross(father, mother, ind1, ind2, True)
        else:
            child1, child2 = cross(father, mother, ind1, ind2)

        next_gen.extend([child1, child2])

    for tr in to_remove:
        next_gen.remove(tr)

    if t == 0:
        mutation(next_gen, mutation_probability, bounds, precision, length, parameters, True)
    else:
        mutation(next_gen, mutation_probability, bounds, precision, length, parameters)

    pop = next_gen

output.write("\nEvolutia maximului:\n")
for i in elite:
    output.write(str(i[0]) + " " + str(i[1]) + '\n')

output.write("\nValoarea medie a performantei: " + str(sum(elites)/len(elite)))
