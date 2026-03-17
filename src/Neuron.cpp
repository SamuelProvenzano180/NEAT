#include "Neuron.h"


Neuron::Neuron(int id, Type type, float leak_value){
    this->id = id;
    this->type = type;
    set_leak_value(leak_value);
    current_value = 0.0f;
    previous_value = 0.0f;
    target_value = 0.0f;
}

Neuron::~Neuron(){}

void Neuron::add_connection(Neuron* to_neuron, const float weight){
    to_connections[to_neuron->id] = weight;
    to_neuron->from_connections.insert(id);
}

int Neuron::get_id() const{
    return id;
}

Neuron::Type Neuron::get_type() const{
    return type;
}

float Neuron::get_leak_value() const{
    return leak_value;
}

float Neuron::get_inverse_leak_value() const{
    return inverse_leak_value;
}

void Neuron::set_leak_value(const float amount){
    leak_value = amount;
    inverse_leak_value = 1.0f / amount;
}

float Neuron::get_current_value() const{
    return current_value;
}

void Neuron::set_current_value(const float amount){
    current_value = amount;
}

float Neuron::get_previous_value() const{
    return previous_value;
}

void Neuron::set_previous_value(const float amount){
    previous_value = amount;
}

float Neuron::get_target_value() const{
    return target_value;
}

void Neuron::set_target_value(const float amount){
    target_value = amount;
}

std::map<int, float> Neuron::get_to_connections() const{
    return to_connections;
}

std::set<int> Neuron::get_from_connections() const{
    return from_connections;
}