#include "Network.h"
#include "NEATAgent.h"

float Network::activation_func(float x, std::string type){
    if (type == "relu"){
        return (x > 0) ? x : 0.01f * x;
    }
    else if (type == "linear"){
        return x;
    }
    else if (type == "sigmoid"){
        return 1.0 / (1.0 + exp(-x));
    }
    else if (type == "tanh"){
        return tanh(x);
    }
    return 0.0;
}

std::vector<float> Network::guess(std::vector<float> inputs){
    std::vector<float> outputs;
    int inputs_passed = 0;

    //Cycle through every single neuron
    for (int i = 0; i < ordered_by_depth.size(); i++){
        Neuron* current_neuron = this->neurons[ordered_by_depth[i]];

        //If input neuron, set accumulated value to the input
        if (current_neuron->depth == 0.0){
            current_neuron->accumulated_value = inputs[inputs_passed++];
        }
        //If not first layer, apply the hidden func
        else{
            if (current_neuron->depth != 1.0){
                current_neuron->accumulated_value = activation_func(current_neuron->accumulated_value, this->hidden_func_str);
            }
        }

        //If last layer, apply the output func
        if (current_neuron->depth == 1.0){
            current_neuron->accumulated_value = activation_func(current_neuron->accumulated_value, this->output_func_str);
            outputs.push_back(current_neuron->accumulated_value);
        }
        //Multiply accumulated value (input) by weights
        else{
            for (auto const& [to, weight] : current_neuron->to_connections){
                neurons[to]->accumulated_value += current_neuron->accumulated_value * weight;
            }
        }
    }

    //Once the neuron is done with, set accumulated value to 0 for next guess
    for (auto& neuron: neurons){
        Neuron* current_neuron = neuron.second;
        current_neuron->accumulated_value = 0.0;
    }

    return outputs;
}

void Network::weight_mutation(std::mt19937 &gen){
    std::uniform_real_distribution<float> prob(0.0, 1.0);
    std::uniform_real_distribution<float> random_weight(-5.0, 5.0);
    std::normal_distribution<float> nudge(0.0f, 0.13f);

    for (int i = 0; i < this->connection_data.size(); i++){
        //10% chance to leave this weight exactly as is
        if (prob(gen) > 0.90f) continue; 

        //10% chance to completely rerandomize this weights value
        if (prob(gen) < 0.10f) {
            this->connection_data[i][2] = random_weight(gen);
        }
        //80% chance to nudge it
        else {
            //Multiplied by weight chance so decreaing the chance decreases the nudge. This is useful for late stage fine tune training
            this->connection_data[i][2] += nudge(gen) * (this->parent_agent->rate_weight_mutate);
        }

        //Clamp weights to prevent them from drifting too far
        float cap = 100.0f;
        if (this->connection_data[i][2] > cap) this->connection_data[i][2] = cap;
        if (this->connection_data[i][2] < -cap) this->connection_data[i][2] = -cap;
    }
}

void Network::add_connection(std::mt19937 &gen){
    int end_hidden = this->temporary_depth_data.size()-this->outputs-1;
    //First neuron chosen must not be one of the output layer
    std::uniform_int_distribution<> dist1(0, end_hidden);

    for (int i = 0; i < 20; i++){ //Try 20 times before terminating
        int first_neuron_index = dist1(gen);

        //Shouldnt connect 2 input neurons
        std::uniform_int_distribution<> dist2(first_neuron_index+1, this->temporary_depth_data.size()-1);
        int second_neuron_index = dist2(gen);
        if (second_neuron_index < this->inputs){
            continue;
        }

        int first_neuron_id = (this->temporary_depth_data)[first_neuron_index];
        int second_neuron_id = (this->temporary_depth_data)[second_neuron_index];

        //Check if the connection exists already between the 2 chosen neurons
        bool does_connection_exist = false;
        for (int i = 0; i < this->connection_data.size(); i++){
            int check_first_id = this->connection_data[i][0];
            int check_second_id = this->connection_data[i][1];

            if (check_first_id == first_neuron_id && check_second_id == second_neuron_id){
                does_connection_exist = true;
                break;
            }
        }

        //If connection doesnt exist, add it
        if (!does_connection_exist){
            std::uniform_real_distribution<float> dis(-1.0, 1.0);
            connect_neurons(&connection_data, first_neuron_id, second_neuron_id, dis(gen));
            break;
        }
    }
}

void Network::toggle_enable(std::mt19937 &gen){
    if (this->connection_data.empty()) return;

    //Choose a random connection
    std::uniform_int_distribution<> distr(0.0, this->connection_data.size()-1);
    int idx = distr(gen);
    int current_size = get_active_connection_count();

    //Only toggle enable if size cap not reached (if on, turn off. If off, turn on)
    if (this->connection_data[idx][3] == 0.0f && current_size < parent_agent->size_cap) {
        this->connection_data[idx][3] = 1.0f;
    } else {
        this->connection_data[idx][3] = 0.0f;
    }
}

void Network::add_neuron(std::mt19937 &gen){
    std::uniform_int_distribution<> distr(0, this->connection_data.size()-1);

    //Choosen connection
    int chosen_connection = distr(gen);

    //Extract necessary data
    int from_neuron_id = this->connection_data[chosen_connection][0];
    int to_neuron_id = this->connection_data[chosen_connection][1];
    int new_neuron_id = this->parent_agent->neuron_counter++;

    int from_neuron_index = -1;
    int to_neuron_index = -1;

    //Find from and to neuron indicies in depth array
    for (int i = 0; i < this->temporary_depth_data.size(); i++){
        if (this->temporary_depth_data[i] == from_neuron_id) from_neuron_index = i;
        if (this->temporary_depth_data[i] == to_neuron_id) to_neuron_index = i;
    }

    if (from_neuron_index == -1 || to_neuron_index == -1) {
        return; 
    }

    //Disable the current connection from A->B
    this->connection_data[chosen_connection][3] = 1.0;

    //Add connection A->C and C->B
    int new_neuron_index = ceil(from_neuron_index + (to_neuron_index - from_neuron_index) / 2.0);
    this->temporary_depth_data.insert(this->temporary_depth_data.begin() + new_neuron_index, new_neuron_id);

    connect_neurons(&this->connection_data, this->connection_data[chosen_connection][0], new_neuron_id, 1.0);
    connect_neurons(&this->connection_data, new_neuron_id, this->connection_data[chosen_connection][1], this->connection_data[chosen_connection][2]);
}

void Network::build_network_structure(){
    int hidden_count = 0;
    for (int id : this->temporary_depth_data) {
        bool is_input = (id < inputs);
        bool is_output = (id >= inputs && id < inputs + outputs);
        if (!is_input && !is_output) hidden_count++;
    }

    int hiddens_passed = 0;

    for (int i = 0; i < this->temporary_depth_data.size(); i++) {
        int neuron_id = this->temporary_depth_data[i];
        float new_depth = 0.5f;

        //Its an input neuron so depth is 0
        if (neuron_id < inputs) {
            new_depth = 0.0f;
        }
        //Its an output neuron so depth is 1
        else if (neuron_id >= inputs && neuron_id < inputs + outputs) {
            new_depth = 1.0f;
        }
        else {
            //Hidden neuron so calculate depth based on ordering
            if (hidden_count > 0) {
                float sep_amount = 1.0f / (hidden_count + 1);
                hiddens_passed++;
                new_depth = sep_amount * hiddens_passed;
            } else {
                new_depth = 0.5f;
            }
        }

        //Create new neuron
        Neuron* new_neuron = new Neuron(neuron_id, new_depth);
        neurons.insert({neuron_id, new_neuron});
    }

    // Connecting all neurons based on connection data
    for (std::vector<float> connection: this->connection_data){
        int from_neuron = (int)connection[0];
        int to_neuron = (int)connection[1];
        float weight = connection[2];
        bool enabled = (bool)connection[3];
        
        //Ensure neuron exists
        if (neurons.count(from_neuron) && neurons.count(to_neuron)) {
            //Only add the connection if it is enabled
            if (enabled == 1.0f) { 
                neurons[from_neuron]->add_connection(to_neuron, weight);
            }
        }
    }

    // Building the ordered_by_depth vector
    ordered_by_depth.clear();
    for (auto& neuron : neurons) {
        ordered_by_depth.push_back(neuron.second->id);
    }
    std::sort(ordered_by_depth.begin(), ordered_by_depth.end(), [this](int a, int b) {
        return this->neurons[a]->depth < this->neurons[b]->depth;
    });
}

std::vector<int>& Network::get_depth_data(){
    return this->ordered_by_depth;
}

std::vector<std::vector<float>>& Network::get_connection_data(){
    return this->connection_data;
}

Network::Network(int inputs, int outputs, std::vector<int>* depth_data, std::vector<std::vector<float>>* connection_data, std::string h, std::string o, bool mutate, std::mt19937 &gen, godot::NEATAgent* parent_agent){
    //Initialize fields
    this->parent_agent = parent_agent;
    
    this->hidden_func_str = h;
    this->output_func_str = o;
    this->inputs = inputs;
    this->outputs = outputs;

    this->temporary_depth_data = *depth_data;
    this->connection_data = *connection_data;

    std::uniform_real_distribution<float> rate(0.0, 1.0);

    //Random chance for mutations
    if (mutate){
        std::uniform_real_distribution<float> dist(0.0, 1.0);
        int current_size = get_active_connection_count();

        // Only allow growth if the network is small
        if (dist(gen) < parent_agent->rate_node_mutate) {
            if (current_size < parent_agent->size_cap) add_neuron(gen);
        }
        if (dist(gen) < parent_agent->rate_connection_mutate) {
            if (current_size < parent_agent->size_cap) add_connection(gen);
        }

        if (dist(gen) < parent_agent->rate_enable_mutate) toggle_enable(gen);
        if (dist(gen) < parent_agent->rate_weight_mutate) weight_mutation(gen);
    }

    build_network_structure();
}

Network::~Network() {
    //Delete all neuron objects
    for (auto const& [id, neuron_ptr] : this->neurons) {
        delete neuron_ptr;
    }
}

void Network::connect_neurons(std::vector<std::vector<float>>* connection_data, int first_id, int second_id, float weight){
    //Look at global table for connection pair
    std::pair<int, int> id_pair = {first_id, second_id};
    int innov_num;

    //Didnt find so add the table connection
    if (this->parent_agent->innovation_table.find(id_pair) == this->parent_agent->innovation_table.end()){
        innov_num = this->parent_agent->innovation_table.size();
        this->parent_agent->innovation_table[id_pair] = innov_num;
    }
    //Found so take that table slots innov number
    else{
        innov_num = this->parent_agent->innovation_table.at(id_pair);
    }

    //Add the connection with the innov number
    std::vector<float> new_connection = {(float)first_id, (float)second_id, weight, 0.0, (float)innov_num};
    connection_data->push_back(new_connection);
}

int Network::get_active_connection_count(){
    int count = 0;
    //Only gets ennabled connections
    for (std::vector<float> conn : this->connection_data) {
        if (conn[3] > 0.5) {
            count++;
        }
    }
    return count;
}