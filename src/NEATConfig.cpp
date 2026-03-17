#include "NEATConfig.h"

using namespace godot;

void NEATConfig::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "inputs", "outputs", "population_size", "desired_species_count", "initial_enabled_percent", "max_memory_frames", "weight_mutation_chance", "connection_mutation_chance", "toggle_mutation_chance", "neuron_mutation_chance", "leak_mutation_chance", "connection_cap", "neuron_cap"), &NEATConfig::initialize);
}

NEATConfig::NEATConfig(){
    valid = false;
}

NEATConfig::~NEATConfig(){}

void NEATConfig::initialize(const int inputs, const int outputs, const int population_size, const int desired_species_count, const float initial_enabled_percent, const int max_memory_frames, const float weight_mutation_chance, const float connection_mutation_chance, const float toggle_mutation_chance, const float neuron_mutation_chance, const float leak_mutation_chance, const int connection_cap, const int neuron_cap){
    valid = false;
    config_contents.clear();
    
    ERR_FAIL_COND_MSG(inputs < 1, "NEATConfig Error: inputs must be greater than 0.");
    ERR_FAIL_COND_MSG(outputs < 1, "NEATConfig Error: outputs must be greater than 0.");
    ERR_FAIL_COND_MSG(population_size < 51, "NEATConfig Error: population_size must be greater than 50.");
    ERR_FAIL_COND_MSG(desired_species_count < 5, "NEATConfig Error: desired_species_count must be greater than 4.");
    ERR_FAIL_COND_MSG(population_size <= desired_species_count * 10, "NEATConfig Error: population_size must be greater than desired_species_count * 10.");
    ERR_FAIL_COND_MSG(initial_enabled_percent < 0.0f || initial_enabled_percent > 1.0f, "NEATConfig Error: initial_enabled_percent must be between 0.0 and 1.0 (inclusive).");
    ERR_FAIL_COND_MSG(max_memory_frames < 1 || max_memory_frames > 3600, "NEATConfig Error: max_memory_frames must be between 1 and 3600 (inclusive).");
    ERR_FAIL_COND_MSG(weight_mutation_chance < 0.0f || weight_mutation_chance > 1.0f, "NEATConfig Error: weight_mutation_chance must be between 0.0 and 1.0 (inclusive).");
    ERR_FAIL_COND_MSG(connection_mutation_chance < 0.0f || connection_mutation_chance > 1.0f, "NEATConfig Error: connection_mutation_chance must be between 0.0 and 1.0 (inclusive).");
    ERR_FAIL_COND_MSG(toggle_mutation_chance < 0.0f || toggle_mutation_chance > 1.0f, "NEATConfig Error: toggle_mutation_chance must be between 0.0 and 1.0 (inclusive).");
    ERR_FAIL_COND_MSG(neuron_mutation_chance < 0.0f || neuron_mutation_chance > 1.0f, "NEATConfig Error: neuron_mutation_chance must be between 0.0 and 1.0 (inclusive).");
    ERR_FAIL_COND_MSG(leak_mutation_chance < 0.0f || leak_mutation_chance > 1.0f, "NEATConfig Error: leak_mutation_chance must be between 0.0 and 1.0 (inclusive).");
    ERR_FAIL_COND_MSG(connection_cap < (inputs + 1) * outputs, "NEATConfig Error: connection_cap cannot be less than (inputs + 1) * outputs.");
    ERR_FAIL_COND_MSG(neuron_cap < inputs + outputs + 1, "NEATConfig Error: neuron_cap cannot be less than inputs + outputs + 1.");

    Array config_array = Array();
    config_array.append(inputs);
    config_array.append(outputs);
    config_array.append(population_size);
    config_array.append(desired_species_count);
    config_array.append(initial_enabled_percent);
    config_array.append(max_memory_frames);
    config_array.append(weight_mutation_chance);
    config_array.append(connection_mutation_chance);
    config_array.append(toggle_mutation_chance);
    config_array.append(neuron_mutation_chance);
    config_array.append(leak_mutation_chance);
    config_array.append(connection_cap);
    config_array.append(neuron_cap);

    valid = true;
    config_contents = config_array;
}

Array NEATConfig::get_config_contents() const{
    return config_contents;
}

bool NEATConfig::is_valid() const{
    return valid;
}