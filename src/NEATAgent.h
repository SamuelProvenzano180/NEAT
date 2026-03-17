#ifndef NEATAGENT_H
#define NEATAGENT_H

#include <vector>
#include <random>
#include <map>
#include <unordered_set>
#include <string>
#include <godot_cpp/classes/ref_counted.hpp>
#include "NEATConfig.h"
#include "Network.h"
#include "Neuron.h"
#include "GenomeData.h"

class Network;
class Species;
class GenomeData;
class NEATConfig;

namespace godot {

    class NEATAgent : public RefCounted {
        GDCLASS(NEATAgent, RefCounted);
    
    public:
        NEATAgent();
        ~NEATAgent();
        void initialize_population(const Ref<NEATConfig> configuration);
        void import_template(const Ref<GenomeData> genome_data, const Ref<NEATConfig> configuration);

        PackedFloat32Array get_network_guess(const int index, const PackedFloat32Array input_array);
        void clear_network_memory(const int index);
        void set_network_fitness(const int index, const float fitness);
        float get_network_fitness(const int index);
        int get_network_connection_amount(const int index);
        int get_network_neuron_amount(const int index);
        Ref<GenomeData> extract_pruned_network_data(const int index);
        void next_generation();

        int get_max_memory_frames() const;
        float get_weight_mutation_chance() const;
        float get_connection_mutation_chance() const;
        float get_toggle_mutation_chance() const;
        float get_neuron_mutation_chance() const;
        float get_leak_mutation_chance() const;
        int get_connection_cap() const;
        int get_neuron_cap() const;

        void set_innovation_table_value(const std::pair<int, int> id_pair, const int innovation_num);
        int get_innovation_table_value(const std::pair<int, int> id_pair) const;
        int get_innovation_table_size() const;

        int get_neuron_counter() const;
        void set_neuron_counter(const int amount);

        std::mt19937& get_rng();

    private:
        static void _bind_methods();

        bool valid;

        std::mt19937 rng;
        std::vector<Network*> population;
        std::vector<Species*> species;

        int inputs;
        int outputs;
        int population_size;
        int desired_species_count;
        int max_memory_frames;
        float weight_mutation_chance;
        float connection_mutation_chance;
        float toggle_mutation_chance;
        float neuron_mutation_chance;
        float leak_mutation_chance;
        //If cap is exceeded, genes will NOT be cut, but they will stop growing (no enabling connections, no creating neurons, no creating connections).
        //Network size will gradually be diminished to cap through disabling connections.
        int connection_cap;
        int neuron_cap;
        float compatibility_threshold;
        std::map<std::pair<int, int>, int> innovation_table;
        int neuron_counter;

        void forward_traverse(Network* network, Neuron* start, std::set<std::pair<int, int> >* pass_set);
        void backwards_traverse(Network* network, Neuron* start, std::set<std::pair<int, int> >* pass_set);
        int find_new_id(int& id_counter, std::map<int, int>& map_id, int old_id);
        std::vector<Network*> reproduce(Species* s);
        
        static std::vector<float> packed_to_vector_float(const PackedFloat32Array &array);
        static PackedFloat32Array vector_to_packed_float(const std::vector<float> &vec);
    };
};

#endif

