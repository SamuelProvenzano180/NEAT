#ifndef SPECIES_H
#define SPECIES_H

#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <random>
#include "NEATAgent.h"

class Network;

class Species {

    public:
        Species();
        ~Species();

        std::vector<Network*>& get_networks(); //Creating a direct getter for simplicity reasons

        int get_age() const;
        void set_age(const int amount);

        int get_offspring_count() const;
        void set_offspring_count(const int amount);
        int get_gens_since_improved() const;
        void set_gens_since_improved(const int amount);
        int get_max_fitness_ever() const;
        void set_max_fitness_ever(const int amount);

        void set_representative_connections(const std::vector<std::vector<float>>& new_value);
        void set_representative_leaks(const std::map<int, float>& new_value);

        void add_member(Network* network);
        void sort_networks();
        float evaluate_compatibility(Network* network);
        static Network* perform_crossover(Network* netA, Network* netB);
    private:
        int size;
        int age;
        int offspring_count;
        int gens_since_improved;
        float max_fitness_ever;
        std::vector<Network*> networks;

        std::vector<std::vector<float>> representative_connections;
        std::map<int, float> representative_leaks;
};

#endif