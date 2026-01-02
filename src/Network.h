#ifndef NETWORK_H
#define NETWORK_H

#include "Neuron.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

namespace godot {
    class NEATAgent;
}

struct Network {
    godot::NEATAgent* parent_agent = nullptr;

    std::map<int, Neuron*> neurons;
    std::vector<int> ordered_by_depth; //Int is id of neuron
    std::vector<std::vector<float>> connection_data;
    std::vector<int> temporary_depth_data;

    std::string hidden_func_str;
    std::string output_func_str;

    int inputs;
    int outputs;
    float fitness = 0.0;
    float adjusted_fitness = 0.0;


    void weight_mutation(std::mt19937 &gen);
    void add_connection(std::mt19937 &gen);
    void toggle_enable(std::mt19937 &gen);
    void add_neuron(std::mt19937 &gen);

    void build_network_structure();

    Network(int inputs, int outputs, std::vector<int>* depth_data, std::vector<std::vector<float>>* connection_data, std::string h, std::string o, bool mutate, std::mt19937 &gen, godot::NEATAgent* parent_agent);
    ~Network();
    
    std::vector<int>& get_depth_data();
    std::vector<std::vector<float>>& get_connection_data();
    
    float activation_func(float x, std::string type);
    std::vector<float> guess(std::vector<float> inputs);
    void connect_neurons(std::vector<std::vector<float>> *c, int first_id, int second_id, float weight);
    int get_active_connection_count();

};

#endif