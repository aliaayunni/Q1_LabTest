import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
GENERATIONS = 50
TARGET_ONES = 40
MUTATION_RATE = 0.01
ELITISM_COUNT = 5


def create_individual():
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]

def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def fitness(individual):
    ones_count = sum(individual)
    return CHROMOSOME_LENGTH - abs(ones_count - TARGET_ONES)

def selection(population):
    selected = random.sample(population, 2)
    return max(selected, key=fitness)

def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Genetic Algorithm Bit Pattern Generator")

if st.button("Run Genetic Algorithm"):
    population = create_population()
    best_fitness_history = []

    for gen in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)

        # Elitism
        new_population = population[:ELITISM_COUNT]

        # Generate new offspring
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:POPULATION_SIZE]
        best_fitness_history.append(fitness(population[0]))

    best_individual = max(population, key=fitness)

    st.subheader("Best Individual Found")
    st.text("".join(map(str, best_individual)))
    st.write("Number of Ones:", sum(best_individual))
    st.write("Fitness Score:", fitness(best_individual))

    st.subheader("Fitness Progress Over Generations")
    st.line_chart(best_fitness_history)
