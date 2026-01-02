#include "Neuron.h"

Neuron::Neuron(int id, float depth){
    this->id = id;
    this->depth = depth;
    this->accumulated_value = 0.0f;
}

void Neuron::add_connection(int to, float weight){
    this->to_connections[to] = weight;
}
