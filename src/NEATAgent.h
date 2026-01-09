#ifndef NEATAGENT_H
#define NEATAGENT_H

#include <vector>
#include <random>
#include <map>
#include <unordered_set>
#include <string>
#include <godot_cpp/classes/ref_counted.hpp>

class Network;
class Species;

namespace godot {

    class NEATAgent : public RefCounted {
        GDCLASS(NEATAgent, RefCounted);
    protected:
        static void _bind_methods();
    private:
        int inputs;
        int outputs;
        int population_size;
        std::string hidden_activation;
        std::string output_activation;
        
        Network* global_champion = nullptr;
        float global_highest_fitness;
        int generation_count;

        float compatibility_threshold;
        int desired_species_count;

        int generations_without_improvement;
        float last_best_fitness;
        int stagnation_limit = INT_MAX;

        static std::vector<float> packed_to_vector_float(const PackedFloat32Array &array);
        static PackedFloat32Array vector_to_packed_float(const std::vector<float> &vec);

    public:

        std::vector<Network*> population;
        std::vector<Species*> species;

        std::mt19937 rng;

        float rate_weight_mutate;
        float rate_connection_mutate;
        float rate_enable_mutate;
        float rate_node_mutate;
        int size_cap = INT_MAX;

        std::map<std::pair<int, int>, int> innovation_table;
        int neuron_counter = 0;

        NEATAgent();
        ~NEATAgent();

        void initialize_population(int inputs, int outputs, int population_size = 100, String hidden_activation = "tanh", String output_activation = "tanh", int desired_species_count = 5, float initial_enabled_percent = 0.25);
        void import_template(Array network_data, int population_size, int desired_species_count);
        void set_mutation_rates(float rate_weight_mutate = 0.95f, float rate_connection_mutate = 0.1f, float rate_enable_mutate = 0.15f, float rate_node_mutate = 0.002);

        PackedFloat32Array get_network_guess(int index, PackedFloat32Array inputs);
        PackedFloat32Array get_champion_guess(PackedFloat32Array inputs);
        void set_network_fitness(int index, float fitness);
        void next_generation();

        float get_champion_fitness();
        int get_champion_connection_count();
        void set_stagnation_limit(int limit);
        void set_connection_size_limit(int limit);
        Array extract_champion_data();
        void force_champion_reset();
        bool has_champion();
        
    };
};

#endif