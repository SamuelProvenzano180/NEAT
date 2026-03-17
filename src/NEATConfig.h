#ifndef NEATCONFIG_H
#define NEATCONFIG_H

#include <godot_cpp/classes/ref_counted.hpp>

namespace godot {

    class NEATConfig : public RefCounted {
        GDCLASS(NEATConfig, RefCounted);
    
    public:
        NEATConfig();
        ~NEATConfig();
        void initialize(const int inputs, const int outputs, const int population_size, const int desired_species_count, const float initial_enabled_percent, const int max_memory_frames, const float weight_mutation_chance, const float connection_mutation_chance, const float toggle_mutation_chance, const float neuron_mutation_chance, const float leak_mutation_chance, const int connection_cap, const int neuron_cap);
        
        Array get_config_contents() const;
        bool is_valid() const;
    
    private:
        static void _bind_methods();
        
        Array config_contents;
        bool valid;
    };
};

#endif