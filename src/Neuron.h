#ifndef NEURON_H
#define NEURON_H

#include <map>
#include <vector>
#include <set>

namespace godot {
    class NEATAgent;
}

class Neuron {

    public:
        enum Type{
            INPUT,
            HIDDEN,
            OUTPUT
        };

        int get_id() const;
        Type get_type() const;

        float get_leak_value() const;
        float get_inverse_leak_value() const; //Inversing it once and multiplying with it rather than constantly dividing using original. This is for optimization pruposes in the guess function 
        void set_leak_value(const float amount);
        float get_current_value() const;
        void set_current_value(const float amount);
        float get_previous_value() const;
        void set_previous_value(const float amount);
        float get_target_value() const;
        void set_target_value(const float amount);

        std::map<int, float> get_to_connections() const;
        std::set<int> get_from_connections() const;

        Neuron(const int id, const Type type, const float leak_value);
        ~Neuron();
        void add_connection(Neuron* to_neuron, const float weight);

    private:
        int id;
        Type type;
        
        float current_value;
        float previous_value;
        float target_value;
        float leak_value;
        float inverse_leak_value;

        std::map<int, float> to_connections; //For guessing (requires weight) and pruning's forward pass
        std::set<int> from_connections; //For pruning's backward pass
};

#endif