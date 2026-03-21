#include "NEATAgent.h"
#include "Network.h"
#include "Species.h"

using namespace godot;

void NEATAgent::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize_population", "configuration"), &NEATAgent::initialize_population);
    ClassDB::bind_method(D_METHOD("import_template", "genome_data", "configuration"), &NEATAgent::import_template);
    ClassDB::bind_method(D_METHOD("get_network_guess", "index", "inputs"), &NEATAgent::get_network_guess);
    ClassDB::bind_method(D_METHOD("clear_network_memory", "index"), &NEATAgent::clear_network_memory);
    ClassDB::bind_method(D_METHOD("set_network_fitness", "index", "fitness"), &NEATAgent::set_network_fitness);
    ClassDB::bind_method(D_METHOD("get_network_fitness", "index"), &NEATAgent::get_network_fitness);
    ClassDB::bind_method(D_METHOD("get_network_connection_amount", "index"), &NEATAgent::get_network_connection_amount);
    ClassDB::bind_method(D_METHOD("get_network_neuron_amount", "index"), &NEATAgent::get_network_neuron_amount);
    ClassDB::bind_method(D_METHOD("extract_pruned_network_data", "index"), &NEATAgent::extract_pruned_network_data);
    ClassDB::bind_method(D_METHOD("next_generation"), &NEATAgent::next_generation);
}

NEATAgent::NEATAgent(){
    valid = false;
}

NEATAgent::~NEATAgent(){
    //Clear out old data
    for (Network* n : population) delete n;
    for (Species* s : species) delete s;
}

void NEATAgent::initialize_population(const Ref<NEATConfig> configuration){

    //Reset everything if it was initialized previously
    if (valid){
        //Clear memory from previously allocated networks and species
        for (Network* n : population) delete n;
        for (Species* s : species) {
            delete s;
        }

        //Clear all data
        population.clear();
        species.clear();
        innovation_table.clear();
    }
    valid = false;
    neuron_counter = 0;
    
    //Error check
    ERR_FAIL_COND_MSG(!configuration.is_valid(), "NEATAgent Error: NEATConfig not valid. Initialize NEATConfig first.");

    //Get the contents out of the config object
    Array config_contents = configuration->get_config_contents().duplicate(true);

    //Initialze fields from config content
    inputs = (int)config_contents[0] + 1; //+1 accounts for bias neuron
    outputs = (int)config_contents[1];
    population_size = (int)config_contents[2];
    desired_species_count = (int)config_contents[3];
    max_memory_frames = (int)config_contents[5];
    weight_mutation_chance = (float)config_contents[6];
    connection_mutation_chance = (float)config_contents[7];
    toggle_mutation_chance = (float)config_contents[8];
    neuron_mutation_chance = (float)config_contents[9];
    leak_mutation_chance = (float)config_contents[10];
    connection_cap = (int)config_contents[11];
    neuron_cap = (int)config_contents[12];

    compatibility_threshold = 3.0f;

    //Create the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    rng = gen;

    //Initialize neurons map
    std::map<int, Neuron*> neuron_data;

    //Create distribution from 1 to sqrt(max_memory_frames)
    //Start the leak value as a small value since most likely want a fast leak speed
    std::uniform_real_distribution<float> rand1(1.0f, sqrt(max_memory_frames));
    
    //Cycle through all inputs and outputs (all neurons in initial template). Create a new neuron object for each and add to neuron data.
    for (int i = 0; i < inputs + outputs; i++){
        Neuron::Type type;
        if (i < inputs) type = Neuron::Type::INPUT;
        else if (i < inputs + outputs) type = Neuron::Type::OUTPUT;

        Neuron* new_neuron = new Neuron(i, type, rand1(rng));
        neuron_data.insert({i, new_neuron});
    }

    //Initialize connections vector
    std::vector<std::vector<float>> connection_data;

    //Create a disabled connection between each input and output, then record in innovation table
    int innov_num = 0;
    for (int j = 0; j < inputs; j++){
        for (int k = inputs; k < inputs + outputs; k++){
            //Create the connection
            std::vector<float> connection = {(float)j, (float)k, 0.0f, 0.0f, (float)(innov_num)};
            connection_data.push_back(connection);

            //Add to the innovation table
            std::pair<int, int> id_pair = {j, k};
            innovation_table[id_pair] = innov_num;

            innov_num++;
        }
    }

    //Create distribution from -1.0 to 1.0 for weight randomization
    std::uniform_real_distribution<float> rand2(-1.0f, 1.0f);

    //For each network in the initial population...
    for (int i = 0; i < population_size; i++){
        //Create its own connection data as a copy of connection_data
        std::vector<std::vector<float>> this_connection_data(connection_data);

        //Cycle through all connections
        for (int j = 0; j < this_connection_data.size(); j++){
            //Randomize the weight value
            this_connection_data[j][2] = rand2(rng);

            //There is a chance for it to become enabled initially
            float enabled_rand_val = (rand2(rng) + 1.0f) / 2.0f;
            if (enabled_rand_val * 0.9999f < (float)config_contents[4]){ //* 0.9999 just so it can never be equal to 1.0, because 1.0 !< 1.0 and dont want to use <= because then same issue with 0
                this_connection_data[j][3] = 1.0f;
            }
        }
        //Put this network in the population
        population.push_back(new Network(inputs, outputs, &neuron_data, &this_connection_data, true, this));
    }
    
    neuron_counter = inputs + outputs;

    for (auto& [id, neuron_ptr] : neuron_data) {
        delete neuron_ptr;
    }

    valid = true;
}

void NEATAgent::import_template(const Ref<GenomeData> genome_data, const Ref<NEATConfig> configuration){

    //Reset everything if it was initialized previously
    if (valid){
        //Clear memory from previously allocated networks and species
        for (Network* n : population) delete n;
        for (Species* s : species) {
            delete s;
        }

        //Clear all data
        population.clear();
        species.clear();
        innovation_table.clear();
    }
    valid = false;
    neuron_counter = 0;
    
    //Error check
    ERR_FAIL_COND_MSG(!genome_data.is_valid(), "NEATAgent Error: GenomeData not valid. Initialize GenomeData first.");
    ERR_FAIL_COND_MSG(!configuration.is_valid(), "NEATAgent Error: NEATConfig not valid. Initialize NEATConfig first.");

    //Get the contents out of the genome and config objects
    Array new_network_data = genome_data->get_genome_contents().duplicate(true);
    Array config_contents = configuration->get_config_contents().duplicate(true);

    //Initialze fields from config content
    inputs = new_network_data.pop_front();
    outputs = new_network_data.pop_front();
    population_size = (int)config_contents[2];
    desired_species_count = (int)config_contents[3];
    max_memory_frames = (int)config_contents[5];
    weight_mutation_chance = (float)config_contents[6];
    connection_mutation_chance = (float)config_contents[7];
    toggle_mutation_chance = (float)config_contents[8];
    neuron_mutation_chance = (float)config_contents[9];
    leak_mutation_chance = (float)config_contents[10];
    connection_cap = (int)config_contents[11];
    neuron_cap = (int)config_contents[12];

    //Pop to get rid of max_memory_frames as its this element is unnecessary here
    new_network_data.pop_front();

    compatibility_threshold = 3.0f;

    //Create the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    rng = gen;

    //Create the neuron map
    std::map<int, Neuron*> neuron_data;

    //Create the connection data vector
    std::vector<std::vector<float>> connection_data;
    //Keeping a map to corrispond neuron id to leak value
    std::map<int, float> neuron_leak;

    int max_neuron_id = -1;
    int innov_num = 0;

    //Cycle through remaining network data that hasnt been popped
    for (int i = 0; i < new_network_data.size(); i++){
        //Seperate the connection data into unique variabled
        Array this_conn = new_network_data[i];
        int from = this_conn[0];
        int to = this_conn[1];
        float weight = this_conn[2];
        float from_leak = this_conn[3];
        float to_leak = this_conn[4];

        //Insert neuron leak pair into map
        neuron_leak.insert({from, from_leak});
        neuron_leak.insert({to, to_leak});

        //Create the connection
        std::vector<float> connection = {(float)from, (float)to, weight, 1.0f, (float)(innov_num)};
        connection_data.push_back(connection);

        //Update the innovation table
        std::pair<int, int> id_pair = {from, to};
        innovation_table[id_pair] = innov_num;
        innov_num++;

        //Update max_neuron_id
        if (from > max_neuron_id) max_neuron_id = from;
        if (to > max_neuron_id) max_neuron_id = to;
    }

    //Cycle through all neurons id's using max_neuron_id
    for (int i = 0; i < max_neuron_id + 1; i++){
        //Create the neuron
        Neuron::Type type;
        if (i < inputs) type = Neuron::Type::INPUT;
        else if (i < inputs + outputs) type = Neuron::Type::OUTPUT;
        else type = Neuron::Type::HIDDEN;

        Neuron* new_neuron = new Neuron(i, type, neuron_leak[i]);
        neuron_data.insert({i, new_neuron});
    }

    //Create the new population and mutate each network added
    for (int i = 0; i < population_size-1; i++){
        population.push_back(new Network(inputs, outputs, &neuron_data, &connection_data, true, this));
    }
    //Dont mutate a single network. Keep imported network active (as it was most likely champ from previous training)
    population.push_back(new Network(inputs, outputs, &neuron_data, &connection_data, false, this));

    neuron_counter = max_neuron_id + 1;

    for (auto& [id, neuron_ptr] : neuron_data) {
        delete neuron_ptr;
    }

    valid = true;
}

PackedFloat32Array NEATAgent::get_network_guess(const int index, const PackedFloat32Array input_array){
    //Error check
    ERR_FAIL_COND_V_MSG(!valid, PackedFloat32Array(), "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    ERR_FAIL_COND_V_MSG(index < 0 || index >= population_size, PackedFloat32Array(), "NEATAgent Error: index must be in range 0 to population_size-1.");
    ERR_FAIL_COND_V_MSG(input_array.size() != inputs-1, PackedFloat32Array(), "NEATAgent Error: input_array size is not equal to expected input size.");

    std::vector<float> input_vec = NEATAgent::packed_to_vector_float(input_array);
    //Add the bias input
    input_vec.push_back(1.0f);
    //Get the guess from the chosen network
    Network* chosen_network = population[index];
    std::vector<float> guess = chosen_network->guess(input_vec);
    return NEATAgent::vector_to_packed_float(guess);
}

void NEATAgent::clear_network_memory(const int index){
    //Error check
    ERR_FAIL_COND_MSG(!valid, "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    ERR_FAIL_COND_MSG(index < 0 || index >= population_size, "NEATAgent Error: index must be in range 0 to population_size-1.");
    
    //Find the network and clear its memory
    Network* network = population[index];
    network->clear_memory();
}

void NEATAgent::set_network_fitness(const int index, const float fitness){
    //Error check
    ERR_FAIL_COND_MSG(!valid, "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    ERR_FAIL_COND_MSG(index < 0 || index >= population_size, "NEATAgent Error: index must be in range 0 to population_size-1.");

    //Find network and set fitness value
    Network* chosen_network = population[index];
    chosen_network->set_fitness(fitness);
}

float NEATAgent::get_network_fitness(const int index){
    //Error check
    ERR_FAIL_COND_V_MSG(!valid, -1.0f, "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    ERR_FAIL_COND_V_MSG(index < 0 || index >= population_size, -1.0f, "NEATAgent Error: index must be in range 0 to population_size-1.");

    //Find network and get the fitness value
    Network* chosen_network = population[index];
    return chosen_network->get_fitness();
}

//This function is for putting evolutionary pressure on networks to keep a minimum size
int NEATAgent::get_network_connection_amount(const int index){
    //Error check
    ERR_FAIL_COND_V_MSG(!valid, -1, "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    ERR_FAIL_COND_V_MSG(index < 0 || index >= population_size, -1, "NEATAgent Error: index must be in range 0 to population_size-1.");

    //Find the network and get the amount of enabled connections it has
    Network* chosen_network = population[index];
    return chosen_network->get_connection_data().size();
}

//This function is for putting evolutionary pressure on networks to keep a minimum size
int NEATAgent::get_network_neuron_amount(const int index){
    //Error check
    ERR_FAIL_COND_V_MSG(!valid, -1, "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    ERR_FAIL_COND_V_MSG(index < 0 || index >= population_size, -1, "NEATAgent Error: index must be in range 0 to population_size-1.");

    //Find the network and get the amount of neuron's it has
    Network* chosen_network = population[index];
    return chosen_network->get_neurons().size();
}

Ref<godot::GenomeData> NEATAgent::extract_pruned_network_data(const int index) {

    Ref<godot::GenomeData> genome_data;
    genome_data.instantiate();

    //Error check
    ERR_FAIL_COND_V_MSG(!valid, genome_data, "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    ERR_FAIL_COND_V_MSG(index < 0 || index >= population_size, genome_data, "NEATAgent Error: index must be in range 0 to population_size-1.");

    Array network_data; //[input, output, [from, to, weight]...]

    //Add initial elements to network_data array
    network_data.append(inputs);
    network_data.append(outputs);
    network_data.append(max_memory_frames);

    Network* network = population[index];

    // Traverse from all inputs of the network and record the ones reached
    std::set<std::pair<int, int> > forward_pass;
    for (int i = 0; i < inputs; i++){
        //Start the forward traverse from each input neuron
        forward_traverse(network, network->get_neuron(i), &forward_pass);
    }

    //Traverse from all inputs of the network and record the ones reached
    std::set<std::pair<int, int> > backwards_pass;
    for (int i = inputs; i < inputs + outputs; i++){
        //Start the backward traverse from each output neuron
        backwards_traverse(network, network->get_neuron(i), &backwards_pass);
    }

    //Neuron ID's dont need to be so large so flatten them using id_counter and map
    int id_counter = inputs + outputs;
    std::map<int, int> map_id;

    //Cycle through backwards_pass id pairs
    for (auto& id_pair: backwards_pass){
        //If forwards pass contains this element from backwards pass, that means that the neuron is reached from both input to output.
        //Since the neuron can get signal from inputs and transfer signal toward output, that means its useful so keep it.
        if (forward_pass.count(id_pair) > 0){
            //Find the neuron from neuron, find its connecting neuron, then grab the weight value
            float weight = 0.0f;
            for (auto& data: network->get_neuron(id_pair.first)->get_to_connections()){
                if (data.first == id_pair.second){
                    weight = data.second;
                    break;
                }
            }

            //Run these functions to flatten the ID values
            int new_from_id = find_new_id(id_counter, map_id, id_pair.first);
            int new_to_id = find_new_id(id_counter, map_id, id_pair.second);

            //Get the leak values from both neurons
            float from_leak = network->get_neuron(id_pair.first)->get_leak_value();
            float to_leak = network->get_neuron(id_pair.second)->get_leak_value();

            //Record all necessary data in connection array
            Array connection;
            connection.append(new_from_id);
            connection.append(new_to_id);
            connection.append(weight);
            connection.append(from_leak);
            connection.append(to_leak);
            network_data.append(connection);
        }
    }

    //Initialize genome with pruned network data and return
    genome_data->initialize(network_data);
    return genome_data;
}

void NEATAgent::forward_traverse(Network* network, Neuron* start, std::set<std::pair<int, int> >* pass_set){
    //Cycle through the neurons to connections
    for (auto& data: start->get_to_connections()){
        int neuron_id = data.first;
        Neuron* current_neuron = network->get_neuron(neuron_id);

        //Record this connection
        std::pair<int, int> connection = {start->get_id(), neuron_id};
        //If connection doesnt exist, that means you can continue down this path
        if (pass_set->count(connection) == 0){
            //Add the connection to the set
            pass_set->insert(connection);
            //Traverse through that connection if not a recurrent connection
            if (connection.first != connection.second){
                forward_traverse(network, current_neuron, pass_set);
            }
        }
    }
}

void NEATAgent::backwards_traverse(Network* network, Neuron* start, std::set<std::pair<int, int> >* pass_set){
    //Cycle through the neurons from connections
    for (int neuron_id: start->get_from_connections()){
        Neuron* current_neuron = network->get_neuron(neuron_id);

        //Record this connection
        std::pair<int, int> connection = {neuron_id, start->get_id()};
        //If connection doesnt exist, that means you can continue down this path
        if (pass_set->count(connection) == 0){
            //Add the connection to the set
            pass_set->insert(connection);
            //Traverse through that connection if not a recurrent connection
            if (connection.first != connection.second){
                backwards_traverse(network, current_neuron, pass_set);
            }
        }
    }
}

int NEATAgent::find_new_id(int& id_counter, std::map<int, int>& map_id, int old_id){
    //Dont need to map input or output to a new id
    if (old_id <= inputs + outputs - 1) return old_id;

    // If the id has already been mapped, return it
    if (map_id.count(old_id) > 0) return map_id[old_id];

    // Otherwise, grab the current id, increment the counter, map it, then return it
    int new_id = id_counter;
    id_counter++;
    map_id[old_id] = new_id;
    return new_id;
}


//next_generation() is the function that controls the evolutionary process. Arguably the most important function of the plugin.
//Create species / add networks to species
//Adjust fitness for each network based on species size
//Sort networks in each species
//Delete bottom 50% of performers
//Age species and delete the ones that havent improved in long enough
//Reproduce all species with offspring count being relative to performance
void NEATAgent::next_generation(){

    //Error Check
    ERR_FAIL_COND_MSG(!valid, "NEATAgent Error: NEATAgent not valid. Initialize NEATAgent first.");
    
    //Cycle through all members of the population and assign to species
    for (int j = 0; j < population.size(); j++){
        Network* current_network = population[j];

        //Speciate
        bool found = false;
        for (Species* s : species) {
            //Compatibility check
            if (s->evaluate_compatibility(current_network) < compatibility_threshold) {
                s->add_member(current_network);
                found = true;
                break;
            }
        }

        //If a network didnt fit into any species, create a new species with this network as the representative
        if (!found) {
            Species* new_s = new Species();
            new_s->add_member(current_network);

            //Representative connections
            new_s->set_representative_connections(current_network->get_connection_data());
            
            //Representative leaks
            std::map<int, float> representative_leaks;
            for (auto& [id, neuron]: current_network->get_neurons()){
                representative_leaks.insert({id, neuron->get_leak_value()});
            }
            new_s->set_representative_leaks(representative_leaks);

            //Add the new species to the species vector
            species.push_back(new_s);
        }
    }

    //Adjust each networks fitness by the size of the species
    for (Species* s: species){
        for (Network* network: s->get_networks()){
            //Put a limit on how low the fitness can go. Fitness cannot be <= 0.0 because offspring allocation would blow up as denominator goes to 0.
            if (network->get_fitness() < 0.01f){
                WARN_PRINT("NEATAgent Error: Network fitness cannot be < 0.01 when creating next generation. Network fitness was automatically set to 0.01.");
                network->set_fitness(0.01f);
            }
            //Adjust the fitness of this network
            network->set_adjusted_fitness(network->get_fitness() / s->get_networks().size());
        }
    }

    //Delete bottom 75% of networks in all species so top 25% can reproduce.
    for (Species* s: species){
        if (s->get_networks().empty()) continue;
        s->sort_networks();

        int survivors = ceil(s->get_networks().size() * 0.25f);
        s->get_networks().resize(survivors);
    }

    for (Species* s : species) {
        //Increase age
        s->set_age(s->get_age() + 1);

        //Find best performing network from the run
        float species_best = 0.0f;
        for (Network* n : s->get_networks()) {
            if (n->get_fitness() > species_best) species_best = n->get_fitness();
        }

        //If the best performer was better than the best that the species has ever seen, reset improvement counter
        if (species_best > s->get_max_fitness_ever()) {
            s->set_max_fitness_ever(species_best);
            s->set_gens_since_improved(0);
        } else {
            s->set_gens_since_improved(s->get_gens_since_improved() + 1);
        }

        //Even if the species doesnt improve, if it holds the elite network, dont increase gens_since_improved because we dont want stagnation to eliminate elite network
        Species* top_species = nullptr;
        for (Species* s: species){
            if (top_species == nullptr) top_species = s;
            else if (s->get_max_fitness_ever() > top_species->get_max_fitness_ever()) top_species = s;
        }
        top_species->set_gens_since_improved(0);

        //Give newer species a fitness bonus so they dont die too soon
        if (s->get_age() < 15) {
            for (Network* n : s->get_networks()) { //1.6x to start and diminishes to 1.0x after 15 gens
                n->set_adjusted_fitness(n->get_adjusted_fitness() * (1.6f - 0.6f * (s->get_age() / 15.0f)));
            }
        }

        // Kill species that haven't improved in 15 generations
        if (s->get_gens_since_improved() > 15) {
            for (Network* n : s->get_networks()) {
                n->set_adjusted_fitness(0.0f);
            }
        }
    }

    //Calculate total aquired fitness
    float global_adjusted_sum = 0.0f;
    for (Species* s: species){
        for (Network* network: s->get_networks()){
            global_adjusted_sum += network->get_adjusted_fitness();
        }
    }

    //Calculate offspring count for each species
    for (Species* s: species){
        if (global_adjusted_sum == 0.0f) break;

        //Calculate the sum for this species
        float species_adj_sum = 0.0f;
        for (Network* network : s->get_networks()) species_adj_sum += network->get_adjusted_fitness();
        
        //Determine offspring count
        int offspring_count = std::floor((species_adj_sum / global_adjusted_sum) * population_size);

        s->set_offspring_count(offspring_count);
    }

    std::vector<Network*> next_generation;

    //Reproduce species to fill next generation
    for (Species* s: species){
        std::vector<Network*> offspring = reproduce(s);
        next_generation.insert(next_generation.end(), offspring.begin(), offspring.end());
    }

    //Since we could get a next_generation size less than population_size due to rounding errors (through floor func), we want to fill in remaining spaces.
    while (next_generation.size() < population_size){

        //Pick random species
        std::uniform_int_distribution<int> rand1(0, species.size()-1);
        int s_idx = rand1(rng);
        Species* s = species[s_idx];
        
        if (s->get_networks().empty()) continue;

        //Pick random network
        Network* parent = s->get_networks()[0];
        
        //Create new child
        Network* new_net = new Network(inputs, outputs, &parent->get_neurons(), &parent->get_connection_data(), true, this);
        next_generation.push_back(new_net);
    }

    //Update representative genomes
    for (Species* s : species) {
        if (!s->get_networks().empty()) {
            //Pick random network as new representative
            std::uniform_int_distribution<int> rand1(0, s->get_networks().size()-1);
            int n_idx = rand1(rng);

            //Representative connections
            s->set_representative_connections(s->get_networks()[n_idx]->get_connection_data());

            //Representative leaks
            std::map<int, float> representative_leaks;
            for (auto& [id, neuron]: s->get_networks()[n_idx]->get_neurons()){
                representative_leaks.insert({id, neuron->get_leak_value()});
            }
            s->set_representative_leaks(representative_leaks);
        }
    }

    //Delete all of the networks from the previous population (which also deletes their neurons)
    for (Network* n : population) {
        delete n;
    }
    population.clear();

    //Delete empty species object if it doesnt have any networks assigned to it (0 offspring)
    std::vector<Species*> new_species;
    std::map<int, Species*> delete_species;
    for (int i = 0; i < species.size(); i++){
        if (species[i]->get_networks().empty()){
            delete_species.insert({i, species[i]});
        }
        else{
            Species* saving_species = species[i];
            //Clear the networks out of the species as the function of speciation is over in this generation. We want it to be clean for next generation
            saving_species->get_networks().clear();
            new_species.push_back(saving_species);
        }
    }
    for (int i = 0; i < species.size(); i++){
        if (delete_species.count(i) > 0){
            delete delete_species[i];
        }
    }
    species = new_species;
    population = next_generation;

    //Adjust compatability threshold to make it easier or harder to join species depending on the amount of species
    int tolerance = std::max(1, desired_species_count / 10);

    if (species.size() < desired_species_count - tolerance) {
        compatibility_threshold -= 0.1f;
    }
    else if (species.size() > desired_species_count + tolerance) {
        compatibility_threshold += 0.1f;
    }
    if (compatibility_threshold < 0.5f) compatibility_threshold = 0.5f;
    if (compatibility_threshold > 10.0f) compatibility_threshold = 10.0f;
}

std::vector<Network*> NEATAgent::reproduce(Species* s){
    std::vector<Network*> new_networks;

    if (s->get_networks().empty()) return new_networks;

    //Add best network in species to new population
    if (s->get_offspring_count() >= 1){
        Network* species_best = s->get_networks()[0];

        new_networks.push_back(new Network(inputs, outputs, &species_best->get_neurons(), &species_best->get_connection_data(), false, this));
        s->set_offspring_count(s->get_offspring_count() - 1);
    }

    std::uniform_real_distribution<float> rand1(0.0f, 1.0f);

    //Create a lambda function which will scew the output towards 0 by squaring the random number
    auto get_biased_index = [&](int size) {
        float r = rand1(rng);
        int idx = std::floor(r * r * size * 0.9999f); //* 0.9999 so it never equals exactly 1, which would leave our index outside of range
        return idx;
    };

    //Create a child of two biased networks
    for (int i = 0; i < s->get_offspring_count(); i++){
        //Pick parent 1 biased towards the elite
        Network* rand_network_1 = s->get_networks()[get_biased_index(s->get_networks().size())];
        Network* rand_network_2 = nullptr;
        
        //2% chance to choose network from other species
        if (rand1(rng) < 0.02f){
            std::uniform_int_distribution<int> rand2(0, species.size()-1);
            Species* other_species = species[rand2(rng)];
            
            //If the selected species happens to have 0 members, just select another network from current species
            if (other_species->get_networks().empty()){
                rand_network_2 = s->get_networks()[get_biased_index(s->get_networks().size())];
            }
            //Otherwise, get a network from the other species
            else{
                rand_network_2 = other_species->get_networks()[get_biased_index(other_species->get_networks().size())];
            }
        }
        //Get a network from the same species species
        else{
            rand_network_2 = s->get_networks()[get_biased_index(s->get_networks().size())];
        }
        
        Network* child_net = Species::perform_crossover(rand_network_1, rand_network_2);
        new_networks.push_back(child_net);
    }

    s->set_offspring_count(0);
    return new_networks;
}

std::vector<float> NEATAgent::packed_to_vector_float(const PackedFloat32Array &array) {
    std::vector<float> vec(array.size());
    for (int i = 0; i < array.size(); i++) vec[i] = array[i];
    return vec;
}

PackedFloat32Array NEATAgent::vector_to_packed_float(const std::vector<float> &vec) {
    PackedFloat32Array array;
    array.resize((int)vec.size());
    for (int i = 0; i < array.size(); i++) array[i] = vec[i];
    return array;
}

int NEATAgent::get_max_memory_frames() const{
    return max_memory_frames;
}

float NEATAgent::get_weight_mutation_chance() const{
    return weight_mutation_chance;
}

float NEATAgent::get_connection_mutation_chance() const{
    return connection_mutation_chance;
}

float NEATAgent::get_toggle_mutation_chance() const{
    return toggle_mutation_chance;
}

float NEATAgent::get_neuron_mutation_chance() const{
    return neuron_mutation_chance;
}

float NEATAgent::get_leak_mutation_chance() const{
    return leak_mutation_chance;
}

int NEATAgent::get_connection_cap() const{
    return connection_cap;
}

int NEATAgent::get_neuron_cap() const{
    return neuron_cap;
}

int NEATAgent::get_neuron_counter() const{
    return neuron_counter;
}

void NEATAgent::set_neuron_counter(const int amount){
    neuron_counter = amount;
}

void NEATAgent::set_innovation_table_value(const std::pair<int, int> id_pair, const int innovation_num){
    innovation_table[id_pair] = innovation_num;
}

int NEATAgent::get_innovation_table_value(const std::pair<int, int> id_pair) const{
    if (innovation_table.count(id_pair) == 0) return -1;
    return innovation_table.at(id_pair);
}

int NEATAgent::get_innovation_table_size() const{
    return innovation_table.size();
}

std::mt19937& NEATAgent::get_rng(){
    return rng;
}