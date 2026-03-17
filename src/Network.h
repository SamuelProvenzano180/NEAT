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

class Network {
    
    public:
        Network(const int inputs, const int outputs, std::map<int, Neuron*>* neuron_data, std::vector<std::vector<float>>* connection_data, const bool mutate, godot::NEATAgent* parent_agent);
        Network(const int inputs, const int outputs, std::map<int, Neuron*>* neuron_data, std::vector<std::vector<float>>* connection_data);
        ~Network();

        std::vector<std::vector<float>>& get_connection_data(); //Creating a direct getter for simplicity reasons
        std::map<int, Neuron*>& get_neurons(); //Creating a direct getter for simplicity reasons
        Neuron* get_neuron(const int id) const;
        godot::NEATAgent* get_parent_agent() const;

        int get_inputs() const;
        int get_outputs() const;
        float get_fitness() const;
        void set_fitness(const float amount);
        float get_adjusted_fitness() const;
        void set_adjusted_fitness(const float amount);
        
        void clear_memory();
        std::vector<float> guess(std::vector<float> input_vec);
    
    private:
        godot::NEATAgent* parent_agent = nullptr;
        std::map<int, Neuron*> neurons;
        std::vector<std::vector<float>> connection_data;

        int inputs;
        int outputs;
        float fitness;
        float adjusted_fitness;

        void weight_mutation();
        void add_connection();
        void toggle_enable();
        void add_neuron();
        void leak_mutation();

        float activation_func(const float x) const;
        void connect_neurons(const int first_id, const int second_id, const float weight);
};

#endif