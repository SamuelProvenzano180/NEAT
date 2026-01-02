#ifndef SPECIES_H
#define SPECIES_H

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <random>


class Network;

struct Species {
    int size = 0;
    int age = 0;
    int gens_since_improved = 0;
    float max_fitness_ever = 0.0f;
    std::vector<Network*> networks;
    std::vector<std::vector<float>> representative_genome;

    void add_member(Network* network);
    void sort_networks();
    float evaluate_compatibility(Network* network);
    std::vector<Network*> reproduce(int offspring_count, std::mt19937 &gen);
    Network* perform_crossover(Network* netA, Network* netB, std::mt19937 &gen);
};
#endif