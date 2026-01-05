#include "NEATAgent.h"
#include "Network.h"
#include "Species.h"

using namespace godot;

void NEATAgent::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize_population", "inputs", "outputs", "population_size", "hidden_activation", "output_activation", "species_count"), &NEATAgent::initialize_population, DEFVAL(150), DEFVAL("tanh"), DEFVAL("tanh"), DEFVAL(8));
    ClassDB::bind_method(D_METHOD("import_template", "network_data", "population_size", "species_count"), &NEATAgent::import_template, DEFVAL(150), DEFVAL(8));
    ClassDB::bind_method(D_METHOD("set_mutation_rates", "rate_weight_mutate", "rate_connection_mutate", "rate_enable_mutate", "rate_node_mutate"), &NEATAgent::set_mutation_rates, DEFVAL(0.8), DEFVAL(0.1), DEFVAL(0.05), DEFVAL(0.03));
    ClassDB::bind_method(D_METHOD("get_network_guess", "index", "inputs"), &NEATAgent::get_network_guess);
    ClassDB::bind_method(D_METHOD("get_champion_guess", "inputs"), &NEATAgent::get_champion_guess);
    ClassDB::bind_method(D_METHOD("set_network_fitness", "index", "fitness"), &NEATAgent::set_network_fitness);
    ClassDB::bind_method(D_METHOD("next_generation"), &NEATAgent::next_generation);
    ClassDB::bind_method(D_METHOD("get_champion_fitness"), &NEATAgent::get_champion_fitness);
    ClassDB::bind_method(D_METHOD("get_champion_connection_count"), &NEATAgent::get_champion_connection_count);
    ClassDB::bind_method(D_METHOD("set_stagnation_limit", "limit"), &NEATAgent::set_stagnation_limit);
    ClassDB::bind_method(D_METHOD("set_connection_size_limit", "limit"), &NEATAgent::set_connection_size_limit);
    ClassDB::bind_method(D_METHOD("extract_champion_data"), &NEATAgent::extract_champion_data);
    ClassDB::bind_method(D_METHOD("force_champion_reset"), &NEATAgent::force_champion_reset);
}

void NEATAgent::initialize_population(int inputs, int outputs, int population_size, godot::String hidden_activation, godot::String output_activation, int desired_species_count){

    //Set fields and error check
    ERR_FAIL_COND_MSG(inputs < 1, "NEATAgent Import Error: Input size must be greater than 0");
    ERR_FAIL_COND_MSG(outputs < 1, "NEATAgent Import Error: Output size must be greater than 0");
    ERR_FAIL_COND_MSG(desired_species_count < 5, "NEATAgent Import Error: Species count must be greater than 4");
    ERR_FAIL_COND_MSG(population_size <= desired_species_count * 10, "NEATAgent Import Error: Population size must be greater than species count * 10.0");

    this->inputs = inputs + 1; //+1 accounts for bias neuron
    this->outputs = outputs;
    this->population_size = population_size;
    this->hidden_activation = hidden_activation.utf8().get_data();
    this->output_activation = output_activation.utf8().get_data();

    ERR_FAIL_COND_MSG(this->hidden_activation != "relu" && this->hidden_activation != "linear" && this->hidden_activation != "sigmoid" && this->hidden_activation != "tanh", "NEATAgent Import Error: Hidden activation function must be \"relu\", \"linear\", \"sigmoid\", or \"tanh\"");
    ERR_FAIL_COND_MSG(this->output_activation != "relu" && this->output_activation != "linear" && this->output_activation != "sigmoid" && this->output_activation != "tanh", "NEATAgent Import Error: Output activation function must be \"relu\", \"linear\", \"sigmoid\", or \"tanh\"");
    
    this->rate_connection_mutate = 0.0;
    this->rate_node_mutate = 0.0;
    this->rate_enable_mutate = 0.0;
    this->rate_weight_mutate = 0.0;

    this->population.clear();
    this->species.clear();
    this->innovation_table.clear();

    this->global_champion = nullptr;
    this->global_highest_fitness = 0.0;
    this->generation_count = 0;

    this->desired_species_count = desired_species_count;
    this->compatibility_threshold = 3.0;

    this->generations_without_improvement = 0;
    this->last_best_fitness = 0.0f;
    this->stagnation_limit = INT_MAX;
    this->size_cap = INT_MAX;

    std::random_device rd;
    std::mt19937 gen(rd());
    this->rng = gen;

    //Initialize neurons
    std::vector<int> depth_data;
    int i = 0;
    while (i < this->inputs + this->outputs){
        depth_data.push_back(i);
        i++;
    }
    this->neuron_counter = i;

    //Initialize conections
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<std::vector<float>> connection_data;
    int innov_num = 0;
    for (int j = 0; j < this->inputs; j++){
        for (int k = this->inputs; k < this->inputs + this->outputs; k++){
            std::vector<float> connection = {(float)j, (float)k, 0.0f, 0.0f, (float)(innov_num)};
            connection_data.push_back(connection);

            //Add to the innovation table
            std::pair<int, int> id_pair = {j, k};
            this->innovation_table[id_pair] = innov_num;

            innov_num++;
        }
    }

    //For every connection, there is a 25% chance that it will be enabled
    for (int i = 0; i < this->population_size; i++){
        std::vector<std::vector<float>> this_connection_data(connection_data);
        for (int j = 0; j < this_connection_data.size(); j++){
            //Also randomize the weight value
            this_connection_data[j][2] = dis(this->rng);

            if (dis(this->rng) > 0.5){
                this_connection_data[j][3] = 1.0;
            }
        }
        //Generate the initial population
        population.push_back(new Network(this->inputs, this->outputs, &depth_data, &this_connection_data, this->hidden_activation, this->output_activation, true, this->rng, this));
    }
}

void NEATAgent::import_template(Array network_data, int population_size, int desired_species_count){

    Array new_network_data = network_data.duplicate(true);

    //Set fields and error check
    ERR_FAIL_COND_MSG(new_network_data.size() < 4, "NEATAgent Import Error: Network data array size must be greater than 3");

    for (int i = 0; i < 4; i++) {
        ERR_FAIL_COND_MSG(new_network_data[i].get_type() != Variant::INT && new_network_data[i].get_type() != Variant::FLOAT, ("NEATAgent Import Error: Index " + std::to_string(i) + " is not a number").c_str());
    }

    for (int i = 4; i < new_network_data.size(); i++) {
        Variant item = new_network_data[i];

        //Must be an array
        ERR_FAIL_COND_MSG(item.get_type() != Variant::ARRAY, ("NEATAgent Import Error: Item at index " + std::to_string(i) + " is not an Array").c_str());

        Array conn = item;

        //Must be size 3
        ERR_FAIL_COND_MSG(conn.size() != 3, ("NEATAgent Import Error: Connection at index " + std::to_string(i) + " has invalid size. Expected 3").c_str());

        //Must be [int, int, float] (float can be casted to int and int can be casted to float so accept both)
        bool id1_ok = (conn[0].get_type() == Variant::INT || conn[0].get_type() == Variant::FLOAT);
        bool id2_ok = (conn[1].get_type() == Variant::INT || conn[1].get_type() == Variant::FLOAT);
        bool weight_ok = (conn[2].get_type() == Variant::INT || conn[2].get_type() == Variant::FLOAT);

        ERR_FAIL_COND_MSG(!id1_ok || !id2_ok || !weight_ok, ("NEATAgent Import Error: Connection at index " + std::to_string(i) + " has invalid types. Expected [int, int, float]").c_str());
    }

    if (desired_species_count < 5) desired_species_count = 5;
    if (population_size < desired_species_count * 10) population_size = desired_species_count * 10;

    this->inputs = new_network_data.pop_front();
    this->outputs = new_network_data.pop_front();
    this->population_size = population_size;

    float hid_fun = new_network_data.pop_front();
    float out_fun = new_network_data.pop_front();

    if (hid_fun == 0.0) this->hidden_activation = "relu";
    else if (hid_fun == 1.0) this->hidden_activation = "linear";
    else if (hid_fun == 2.0) this->hidden_activation = "sigmoid";
    else if (hid_fun == 3.0) this->hidden_activation = "tanh";
    if (out_fun == 0.0) this->output_activation = "relu";
    else if (out_fun == 1.0) this->output_activation = "linear";
    else if (out_fun == 2.0) this->output_activation = "sigmoid";
    else if (out_fun == 3.0) this->output_activation = "tanh";
    
    this->rate_connection_mutate = 0.0;
    this->rate_node_mutate = 0.0;
    this->rate_enable_mutate = 0.0;
    this->rate_weight_mutate = 0.0;

    this->population.clear();
    this->species.clear();
    this->innovation_table.clear();

    this->global_champion = nullptr;
    this->global_highest_fitness = 0.0;
    this->generation_count = 0;

    this->desired_species_count = desired_species_count;
    this->compatibility_threshold = 3.0;

    this->generations_without_improvement = 0;
    this->last_best_fitness = 0.0f;
    this->stagnation_limit = INT_MAX;
    this->size_cap = INT_MAX;

    std::random_device rd;
    std::mt19937 gen(rd());
    this->rng = gen;

    std::unordered_set<int> seen_neurons;

    //Initialize input neurons
    std::vector<int> depth_data;
    for (int i = 0; i < this->inputs; i++){
        depth_data.push_back(i);
    }
    
    //Initialze hidden neurons based on connection data
    for (int i = 0; i < new_network_data.size(); i++){
        Array this_conn = new_network_data[i];
        int from = this_conn[0];
        if (from >= this->inputs){ //If not an input (which is already added to depth data
            if (seen_neurons.count(from) == 0){ //If not yet seen
                seen_neurons.insert(from);
                depth_data.push_back(from);
            }
        }
    }

    //Initialize output neurons
    for (int i = this->inputs; i < this->inputs + this->outputs; i++){
        depth_data.push_back(i);
    }

    this->neuron_counter = depth_data.size();

    //Create the connection data with all connections enabled
    std::vector<std::vector<float>> connection_data;
    int innov_num = 0;
    for (auto conn: new_network_data){
        Array conn_array = conn;
        int from = conn_array[0];
        int to = conn_array[1];
        float weight = conn_array[2];

        std::vector<float> connection = {(float)from, (float)to, weight, 1.0f, (float)(innov_num)};
        connection_data.push_back(connection);

        std::pair<int, int> id_pair = {from, to};
        this->innovation_table[id_pair] = innov_num;

        innov_num++;
    }

    //Create population based on imported network data
    for (int i = 0; i < this->population_size; i++){
        population.push_back(new Network(this->inputs, this->outputs, &depth_data, &connection_data, this->hidden_activation, this->output_activation, true, this->rng, this));
    }
}

void NEATAgent::set_mutation_rates(float rate_weight_mutate, float rate_connection_mutate, float rate_enable_mutate, float rate_node_mutate){
    //Error check
    bool less_than_0 = rate_weight_mutate < 0.0 || rate_connection_mutate < 0.0 || rate_enable_mutate < 0.0 || rate_node_mutate < 0.0;
    bool greater_than_1 = rate_weight_mutate > 1.0 || rate_connection_mutate > 1.0 || rate_enable_mutate > 1.0 || rate_node_mutate > 1.0;

    ERR_FAIL_COND_MSG(less_than_0 || greater_than_1, "NEATAgent Mutation Error: Mutation rates must be in range 0.0 to 1.0");

    //Set mutation rates
    this->rate_weight_mutate = rate_weight_mutate;
    this->rate_connection_mutate = rate_connection_mutate;
    this->rate_enable_mutate = rate_enable_mutate;
    this->rate_node_mutate = rate_node_mutate;
}

PackedFloat32Array NEATAgent::get_network_guess(int index, PackedFloat32Array inputs){
    //Error check
    ERR_FAIL_COND_V_MSG(index < 0 || index >= this->population_size, PackedFloat32Array(), "NEATAgent Guess Error: Index must be in range 0 to population_size-1");
    ERR_FAIL_COND_V_MSG(inputs.size() != this->inputs-1, PackedFloat32Array(), "NEATAgent Guess Error: Number of inputs is not equal to expected input size");

    //Get the guess array from network at index
    std::vector<float> input_vec = NEATAgent::packed_to_vector_float(inputs);
    input_vec.push_back(1.0);
    Network* chosen_network = this->population[index];
    std::vector<float> guess = chosen_network->guess(input_vec);
    return NEATAgent::vector_to_packed_float(guess);
}

PackedFloat32Array NEATAgent::get_champion_guess(PackedFloat32Array inputs){
    //Error check
    ERR_FAIL_COND_V_MSG(inputs.size() != this->inputs-1, PackedFloat32Array(), "NEATAgent Guess Error: Number of inputs is not equal to expected input size");
    ERR_FAIL_COND_V_MSG(this->global_champion == nullptr, PackedFloat32Array(), "NEATAgent Champion Error: No champion yet");

    //Get the guess array from champ network
    std::vector<float> input_vec = NEATAgent::packed_to_vector_float(inputs);
    input_vec.push_back(1.0);
    std::vector<float> guess = this->global_champion->guess(input_vec);
    return NEATAgent::vector_to_packed_float(guess);
}

void NEATAgent::set_network_fitness(int index, float fitness){
    //Error check
    ERR_FAIL_COND_MSG(index < 0 || index >= this->population_size, "NEATAgent Set Error: Index must be in range 0 to population_size-1");
    ERR_FAIL_COND_MSG(fitness <= 0.0001, "NEATAgent Set Error: Fitness must be greater than 0.0001");

    //Set network at index's fitness to fitness value
    Network* chosen_network = this->population[index];
    chosen_network->fitness = fitness;
}

void NEATAgent::next_generation(){
    //Check if there was improvement from last generation
    if (this->global_highest_fitness > this->last_best_fitness) { 
        this->last_best_fitness = this->global_highest_fitness;
        this->generations_without_improvement = 0;
    } else {
        this->generations_without_improvement++;
    }

    //If hasnt improved in stagnation_limit amount of generations, continue...
    if (this->generations_without_improvement > stagnation_limit) {
        
        //If there is no champion, create a default network as champion
        if (this->global_champion == nullptr) {
            this->global_champion = new Network(this->inputs, this->outputs, nullptr, nullptr, this->hidden_activation, this->output_activation, false, this->rng, this);
            this->global_champion->fitness = 0.01;
        }

        //Delete current population
        for (Network* n : this->population) {
            delete n;
        }
        this->population.clear(); 

        //Clear out old species
        for (Species* s : this->species) {
            s->networks.clear();
            delete s;
        }
        this->species.clear();

        //Keep champion in the new population
        this->population.push_back(new Network(this->inputs, this->outputs, &this->global_champion->get_depth_data(), &this->global_champion->get_connection_data(), this->hidden_activation, this->output_activation, false, this->rng, this));
        
        //Repopulate from the champion
        for (int k = 1; k < this->population_size; k++) {
            Network* mutant = new Network(this->inputs, this->outputs, &this->global_champion->get_depth_data(), &this->global_champion->get_connection_data(), this->hidden_activation, this->output_activation, true, this->rng, this);
            this->population.push_back(mutant);
        }

        this->generations_without_improvement = 0;

        return;
    }

    //If improvement is occuring, clear all species members (spceies representative will still exist)
    for (Species* s : this->species) {
        s->networks.clear();
    }

    Network* best_performer = nullptr;
    Network* best_clone = nullptr;

    //Cycle through all members of the population
    for (int j = 0; j < population.size(); j++){
        Network* current_network = this->population[j];

        //Speciate
        bool found = false;
        for (Species* s : this->species) {
            //Compatibility check
            if (s->evaluate_compatibility(current_network) < this->compatibility_threshold) {
                s->add_member(current_network);
                found = true;
                break;
            }
        }

        //If a network didnt fit into any species, create a new species with this network as the representative
        if (!found) {
            Species* new_s = new Species();
            new_s->add_member(current_network);
            // Important: Set identity for the new species
            new_s->representative_genome = current_network->get_connection_data();
            this->species.push_back(new_s);
        }

        //Get best performer out of previous generation
        if (best_performer == nullptr || current_network->fitness > best_performer->fitness){
            best_performer = current_network;
            
            //Clone the best performer immediately
            if (best_clone != nullptr) delete best_clone;
            best_clone = new Network(this->inputs, this->outputs, &best_performer->get_depth_data(), &best_performer->get_connection_data(), this->hidden_activation, this->output_activation, false, this->rng, this);
        
            //If best is better than global champion, change global champion to best
            if (current_network->fitness > this->global_highest_fitness) {
                this->global_highest_fitness = current_network->fitness;
                if (this->global_champion != nullptr) delete this->global_champion;
                this->global_champion = new Network(this->inputs, this->outputs, &current_network->get_depth_data(), &current_network->get_connection_data(), this->hidden_activation, this->output_activation, false, this->rng, this);
                this->global_champion->fitness = current_network->fitness;
            }
        }
    }

    //Delete 75% of networks in all species so top 25% can reproduce
    for (Species* s: this->species){
        if (s->networks.empty()) continue;
        s->sort_networks();

        int survivors = ceil(s->networks.size() * 0.25);
        if (survivors < 1) survivors = 1;
        for (int k = survivors; k < s->networks.size(); k++) {
            delete s->networks[k];
        }
        s->networks.resize(survivors);
    }

    //Adjust each networks fitness by the size of the species
    for (Species* s: species){
        for (Network* network: s->networks){
            network->adjusted_fitness = network->fitness / s->networks.size();
        }
    }

    for (Species* s : this->species) {
        //Increase age
        s->age++;

        //Check for improvement
        float species_best = 0.0f;
        for (Network* n : s->networks) {
            if (n->fitness > species_best) species_best = n->fitness;
        }

        if (species_best > s->max_fitness_ever) {
            s->max_fitness_ever = species_best;
            s->gens_since_improved = 0;
        } else {
            s->gens_since_improved++;
        }

        //Give newer species a fitness bonus so they dont die too soon
        if (s->age < 10) {
            for (Network* n : s->networks) {
                n->adjusted_fitness *= 1.5f;
            }
        }

        //Kill species that haven't improved in 15 generations
        if (s->age > 20 && s->gens_since_improved > 15) {
            for (Network* n : s->networks) {
                n->adjusted_fitness = 0.0f;
            }
        }
    }

    float global_adjusted_sum = 0.0;
    for (Species* s: this->species){
        for (Network* network: s->networks){
            global_adjusted_sum += network->adjusted_fitness;
        }
    }

    std::vector<Network*> next_generation; 

    // Take the best network we found earlier and add to the new generation
    if (best_clone != nullptr) {
        next_generation.push_back(best_clone);
        best_clone = nullptr;
    }


    for (Species* s: this->species){
        //Calculate the sum for this species
        float species_adj_sum = 0.0;
        for (Network* network : s->networks) species_adj_sum += network->adjusted_fitness;
        
        //Determine offspring count
        if (global_adjusted_sum == 0.0) break;
        int offspring_count = (species_adj_sum / global_adjusted_sum) * this->population_size;
        
        if (offspring_count <= 0 && species_adj_sum > 0) offspring_count = 1;
        if (offspring_count <= 0) continue;

        //Perform crossover and add to next generation
        std::vector<Network*> babies = s->reproduce(offspring_count, this->rng);
        next_generation.insert(next_generation.end(), babies.begin(), babies.end());
    }

    //Since we could get a next_generation size less than population_size, we want to fill in remaining gaps
    while (next_generation.size() < this->population_size){
        if (this->species.empty()) break;
        
        //Pick random species
        int s_idx = std::uniform_int_distribution<>(0, this->species.size()-1)(this->rng);
        Species* s = this->species[s_idx];
        
        if (s->networks.empty()) continue;

        //Pick random network
        int n_idx = std::uniform_int_distribution<>(0, s->networks.size()-1)(this->rng);
        Network* parent = s->networks[n_idx];
        
        //Create new child
        Network* new_net = new Network(this->inputs, this->outputs, &parent->get_depth_data(), &parent->get_connection_data(), this->hidden_activation, this->output_activation, true, this->rng, this);
        next_generation.push_back(new_net);
    }

    //Update representative genomes
    for (Species* s : this->species) {
        if (!s->networks.empty()) {
            s->representative_genome = s->networks[0]->get_connection_data(); //[0] is best since networks are sorted
        }
    }

    //Delete old parents
    for (Species* s : this->species) {
        for (Network* parent : s->networks) {
            delete parent;
        }
        s->networks.clear();
    }

    //Delete species object
    auto it = this->species.begin();
    while (it != this->species.end()) {
        if ((*it)->networks.empty()) {
            delete *it;
            it = this->species.erase(it);
        } else {
            ++it;
        }
    }

    this->population = next_generation; 

    //Adjust compatability threshold to make it easier or harder to join species based on the amount of species
    if (this->species.size() < this->desired_species_count -  3) {
        this->compatibility_threshold -= 0.3;
    } 
    else if (this->species.size() > this->desired_species_count + 3) {
        this->compatibility_threshold += 0.3;
    }
    if (this->compatibility_threshold < 0.3) this->compatibility_threshold = 0.3;

}

float NEATAgent::get_champion_fitness(){
    ERR_FAIL_COND_V_MSG(this->global_champion == nullptr, -1.0, "NEATAgent Champion Error: No champion yet");
    return this->global_highest_fitness;
}

int NEATAgent::get_champion_connection_count(){
    ERR_FAIL_COND_V_MSG(this->global_champion == nullptr, -1, "NEATAgent Champion Error: No champion yet");
    return this->global_champion->get_active_connection_count();
}

void NEATAgent::set_stagnation_limit(int limit){
    ERR_FAIL_COND_MSG(limit < 3, "NEATAgent Set Error: limit must be greater than 2");
    this->stagnation_limit = limit;
}

void NEATAgent::set_connection_size_limit(int limit){ //NOTE: Wont add connections past limit. If this value is changed and connection amount exceeds, it will remain but not add connections any more
    ERR_FAIL_COND_MSG(limit < 3, "NEATAgent Set Error: limit must be greater than 2");
    this->size_cap = limit;
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

Array NEATAgent::extract_champion_data() {
    //Set the initial fields and error check
    ERR_FAIL_COND_V_MSG(this->global_champion == nullptr, Array(), "NEATAgent Champion Error: No champion yet");

    Array network_data;
    network_data.append(this->inputs);
    network_data.append(this->outputs);
    if (this->hidden_activation == "relu") network_data.append(0);
    else if (this->hidden_activation == "linear") network_data.append(1);
    else if (this->hidden_activation == "sigmoid") network_data.append(2);
    else if (this->hidden_activation == "tanh") network_data.append(3);
    if (this->output_activation == "relu") network_data.append(0);
    else if (this->output_activation == "linear") network_data.append(1);
    else if (this->output_activation == "sigmoid") network_data.append(2);
    else if (this->output_activation == "tanh") network_data.append(3);

    //Prune network so that a connection route that doesnt have a path to an output neuron are culled
    std::unordered_set<int> useful_nodes;
    for (int i = 0; i < this->outputs; i++) {
        useful_nodes.insert(this->inputs + i);
    }

    const std::vector<int>& depth_list = this->global_champion->ordered_by_depth;

    for (int i = depth_list.size() - 1; i >= 0; i--) {
        int neuron_id = depth_list[i];
        
        //Skip if this neuron is an output (already handled above)
        if (useful_nodes.count(neuron_id)) continue;

        Neuron* neuron = this->global_champion->neurons[neuron_id];

        //Check if this neuron feeds into any useful neurons
        for (auto& conn : neuron->to_connections) {
            int target_id = conn.first;
            
            //If target is useful, this neuron is useful too
            if (useful_nodes.count(target_id)) {
                useful_nodes.insert(neuron_id);
                break;
            }
        }
    }

    std::unordered_map<int, int> id_map;
    int current_mapped_id = 0;

    //Flatten the connection data so that neuron ID's wont be extremely large but instead relative

    //Add inputs to map
    for (int i = 0; i < this->inputs; i++) {
        id_map[i] = current_mapped_id++;
    }

    //Add outputs to map
    for (int i = 0; i < this->outputs; i++) {
        id_map[this->inputs + i] = current_mapped_id++;
    }

    //Add hiddens to map only if they are useful
    for (int old_id : depth_list) {
        //Skip inputs and outputs
        if (id_map.count(old_id)) continue;

        if (useful_nodes.count(old_id)) {
            id_map[old_id] = current_mapped_id++;
        }
    }

    for (int neuron_id : depth_list) {
        // Skip nodes we decided were useless
        if (id_map.count(neuron_id) == 0) continue;

        Neuron* neuron = this->global_champion->neurons[neuron_id];
        
        for (auto& connection : neuron->to_connections) {
            
            // Create connections. Skip ones that connect to useless nodes (dead end)
            if (id_map.count(neuron->id) && id_map.count(connection.first)) {
                
                int new_from = id_map[neuron->id];
                int new_to   = id_map[connection.first];
                float weight = connection.second;

                Array connection_array;
                connection_array.append(new_from);
                connection_array.append(new_to);
                connection_array.append(weight);
                
                network_data.append(connection_array);
            }
        }
    }
    return network_data;
}

void NEATAgent::force_champion_reset(){
    this->global_champion = nullptr;
    this->global_highest_fitness = 0.0;
}

NEATAgent::NEATAgent(){}
NEATAgent::~NEATAgent(){}