#ifndef NEURON_H
#define NEURON_H

#include <map>
#include <vector>

struct Neuron {
    int id;
    float depth;
    float accumulated_value;

    std::map<int, float> to_connections;

    Neuron(int id, float depth);
    void add_connection(int to, float weight);
};

#endif