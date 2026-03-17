#include "Network.h"
#include "NEATAgent.h"

Network::Network(const int inputs, const int outputs, std::map<int, Neuron*>* neuron_data, std::vector<std::vector<float>>* connection_data, const bool mutate, godot::NEATAgent* parent_agent){
    //Initialize fields
    this->inputs = inputs;
    this->outputs = outputs;
    this->parent_agent = parent_agent;
    fitness = 0.0f;
    adjusted_fitness = 0.0f;

    //Deep copy all neurons from inputted data to make new neuron data
    neurons.clear();
    for (auto& [id, neuron_ptr] : *neuron_data){
        Neuron* new_neuron = new Neuron(id, neuron_ptr->get_type(), neuron_ptr->get_leak_value());
        neurons.insert({id, new_neuron});
    }
    this->connection_data = *connection_data;

    //Create distribution from 0.0 to 1.0
    std::uniform_real_distribution<float> rand1(0.0f, 1.0f);

    //Random chance for mutations to occur
    if (mutate){
        if (rand1(parent_agent->get_rng()) < parent_agent->get_weight_mutation_chance()) weight_mutation();
        //Wont connect if connection cap reached
        if (rand1(parent_agent->get_rng()) < parent_agent->get_connection_mutation_chance() && this->connection_data.size() < parent_agent->get_connection_cap()) add_connection();
        //Will not enable if connection cap reached
        if (rand1(parent_agent->get_rng()) < parent_agent->get_toggle_mutation_chance()) toggle_enable();
        if (rand1(parent_agent->get_rng()) < parent_agent->get_leak_mutation_chance()) leak_mutation();
        //Will only add neuron if neuron cap not reached and connection cap not reached (adding neuron always adds 2 connections, so im using get_connection_cap()-1)
        if (rand1(parent_agent->get_rng()) < parent_agent->get_neuron_mutation_chance() && neurons.size() < parent_agent->get_neuron_cap() && this->connection_data.size() < parent_agent->get_connection_cap() - 1) add_neuron();
    }

    // Connecting all neurons based on connection data
    for (std::vector<float> connection: this->connection_data){
        int from_neuron = (int)connection[0];
        int to_neuron = (int)connection[1];
        float weight = connection[2];
        bool enabled = (bool)connection[3];
        
        //Ensure neuron exists
        if (neurons.count(from_neuron) && neurons.count(to_neuron)) {
            //Only add the connection to neuron if the connection gene is enabled
            if (enabled) neurons[from_neuron]->add_connection(neurons[to_neuron], weight);
        }
    }
}

Network::Network(const int inputs, const int outputs, std::map<int, Neuron*>* neuron_data, std::vector<std::vector<float>>* connection_data){
    //Initialize fields
    this->inputs = inputs;
    this->outputs = outputs;
    parent_agent = nullptr;
    fitness = 0.0f;
    adjusted_fitness = 0.0f;

    this->connection_data = *connection_data;

    //Deep copy all neurons from inputted data to make new neuron data
    neurons.clear();
    for (auto& [id, neuron_ptr] : *neuron_data){
        Neuron* new_neuron = new Neuron(id, neuron_ptr->get_type(), neuron_ptr->get_leak_value());
        neurons.insert({id, new_neuron});
    }

    //Connecting all neurons based on connection data
    for (std::vector<float> connection: this->connection_data){
        int from_neuron = (int)connection[0];
        int to_neuron = (int)connection[1];
        float weight = connection[2];
        bool enabled = (bool)connection[3];
        
        //Ensure neuron exists
        if (neurons.count(from_neuron) && neurons.count(to_neuron)) {
            //Only add the connection to neuron if the connection gene is enabled
            if (enabled) neurons[from_neuron]->add_connection(neurons[to_neuron], weight);
        }
    }
}

Network::~Network() {
    //Delete all neuron objects
    for (auto const& [id, neuron_ptr] : neurons) delete neuron_ptr;
}

float Network::activation_func(float x) const{
    return tanh(x);
}

std::vector<float> Network::guess(std::vector<float> input_vec){

    //Cycle through every single neuron and move their last runs value to previous_value
    for (auto& [id, current_neuron]: neurons){
        current_neuron->set_previous_value(current_neuron->get_current_value());
        current_neuron->set_target_value(0.0f);
    }

    //Load inputs in
    for (int i = 0; i < inputs; i++){
        Neuron* current_neuron = neurons[i];
        current_neuron->set_current_value(input_vec[i]);
        //Immediately load into previous_value so there is no delay
        current_neuron->set_previous_value(input_vec[i]);
    }

    //Propogate signal through network
    for (auto& [id, current_neuron] : neurons){
        //Only apply activation function if casting out of a neuron that isnt an inputx
        float signal = current_neuron->get_previous_value();
        if (current_neuron->get_type() != Neuron::Type::INPUT) signal = activation_func(signal);

        for (auto const& [to, weight] : current_neuron->get_to_connections()){
            neurons[to]->set_target_value(neurons[to]->get_target_value() + signal * weight);
        }
    }

    //Apply the leak
    for (auto& [id, current_neuron] : neurons){
        if (current_neuron->get_type() == Neuron::Type::INPUT) continue;

        current_neuron->set_current_value(current_neuron->get_previous_value() + (current_neuron->get_target_value() - current_neuron->get_previous_value()) * current_neuron->get_inverse_leak_value());
    }

    //Collect the outputs.
    std::vector<float> outputs;
    for (auto& [id, current_neuron] : neurons){
        if (current_neuron->get_type() == Neuron::Type::OUTPUT) {
            float output_value = activation_func(current_neuron->get_current_value());
            outputs.push_back(output_value);
        }
    }

    return outputs;
}

void Network::weight_mutation(){
    std::uniform_real_distribution<float> rand1(0.0f, 1.0f); //prob 0-1
    std::uniform_real_distribution<float> rand2(-1.0f, 1.0f); //random weight
    std::normal_distribution<float> rand3(0.0f, 0.1f); //random weight nudge

    for (int i = 0; i < connection_data.size(); i++){
        //10% chance to leave this weight exactly as is
        if (rand1(parent_agent->get_rng()) > 0.90f) continue; 

        //10% chance to completely rerandomize this weights value
        if (rand1(parent_agent->get_rng()) < 0.10f) {
            connection_data[i][2] = rand2(parent_agent->get_rng());
        }
        //80% chance to nudge it
        else {
            connection_data[i][2] += rand3(parent_agent->get_rng());
        }

        //Clamp weights to prevent them from drifting too far
        float cap = 5.0f;
        if (connection_data[i][2] > cap) connection_data[i][2] = cap;
        if (connection_data[i][2] < -cap) connection_data[i][2] = -cap;
    }
}

void Network::add_connection(){
    std::uniform_int_distribution<int> rand1(0, neurons.size()-1);

    for (int i = 0; i < 20; i++){ //Try 20 times before terminating
        //Pick first and second neuron indexes
        int first_neuron_index = rand1(parent_agent->get_rng());
        int second_neuron_index = rand1(parent_agent->get_rng());

        auto it_second = std::begin(neurons);
        std::advance(it_second, second_neuron_index);
        //Dont connect to an input
        if (it_second->second->get_type() == Neuron::Type::INPUT) continue;

        auto it_first = std::begin(neurons);
        std::advance(it_first, first_neuron_index);

        //Get the neurons from the map
        int first_neuron_id = it_first->first;
        int second_neuron_id = it_second->first;

        //Check if the connection exists already between the 2 chosen neurons
        bool does_connection_exist = false;
        for (int i = 0; i < connection_data.size(); i++){
            int check_first_id = connection_data[i][0];
            int check_second_id = connection_data[i][1];

            if (check_first_id == first_neuron_id && check_second_id == second_neuron_id){
                does_connection_exist = true;
                break;
            }
        }

        //If connection doesnt exist, add it
        if (!does_connection_exist){
            std::uniform_real_distribution<float> rand2(-1.0f, 1.0f);
            connect_neurons(first_neuron_id, second_neuron_id, rand2(parent_agent->get_rng()));
            break;
        }
    }
}

void Network::toggle_enable(){
    if (connection_data.empty()) return;

    //Choose a random connection
    std::uniform_int_distribution<int> rand1(0.0f, connection_data.size()-1);
    int idx = rand1(parent_agent->get_rng());

    //Only toggle enable if size cap not reached (if on, turn off. If off, turn on)
    if (connection_data[idx][3] == 0.0f && connection_data.size() < parent_agent->get_connection_cap()) connection_data[idx][3] = 1.0f;
    else connection_data[idx][3] = 0.0f;
}

void Network::add_neuron(){
    std::uniform_int_distribution<int> rand1(0, connection_data.size()-1);

    //Choosen connection
    int chosen_connection = rand1(parent_agent->get_rng());

    int new_neuron_id = parent_agent->get_neuron_counter();
    parent_agent->set_neuron_counter(new_neuron_id + 1);

    //Disable the current connection from A->B
    connection_data[chosen_connection][3] = 0.0f;

    //Create distribution for random leak value
    std::uniform_real_distribution<float> rand2(1.0f, sqrt(parent_agent->get_max_memory_frames()));

    Neuron* new_neuron = new Neuron(new_neuron_id, Neuron::Type::HIDDEN, rand2(parent_agent->get_rng()));
    neurons.insert({new_neuron_id, new_neuron});

    //Add connection A->C and C->B
    connect_neurons(connection_data[chosen_connection][0], new_neuron_id, 1.0f);
    connect_neurons(new_neuron_id, connection_data[chosen_connection][1], connection_data[chosen_connection][2]);
}

void Network::leak_mutation(){
    std::uniform_real_distribution<float> rand1(0.0f, 1.0f); //prob 0-1
    std::uniform_real_distribution<float> rand2(1.0f, sqrt(parent_agent->get_max_memory_frames())); //random leak
    std::normal_distribution<float> rand3(1.0f, 0.2f); //random leak nudge

    for (auto& data : neurons){
        Neuron* current_neuron = data.second;
        //10% chance to leave this neurons leak value exactly as is
        if (rand1(parent_agent->get_rng()) > 0.90f) continue;

        //10% chance to completely rerandomize this neurons leak value
        if (rand1(parent_agent->get_rng()) < 0.10f) {
            current_neuron->set_leak_value(rand2(parent_agent->get_rng()));
        }
        //80% chance to nudge it
        else {
            current_neuron->set_leak_value(current_neuron->get_leak_value() * rand3(parent_agent->get_rng()));
        }

        //Clamp the leak value from 1 to max_memory_frames
        if (current_neuron->get_leak_value() > parent_agent->get_max_memory_frames()) current_neuron->set_leak_value(parent_agent->get_max_memory_frames());
        if (current_neuron->get_leak_value() < 1.0f) current_neuron->set_leak_value(1.0f);
    }
}

void Network::clear_memory(){
    for (auto& data : neurons){
        Neuron* current_neuron = data.second;
        current_neuron->set_current_value(0.0f);
        current_neuron->set_previous_value(0.0f);
        current_neuron->set_target_value(0.0f);
    }
}

void Network::connect_neurons(int first_id, int second_id, float weight){
    //Look at global table for connection pair
    std::pair<int, int> id_pair = {first_id, second_id};
    int innov_num = -1;

    if (parent_agent != nullptr) {
        //If the element doesnt exists, create new innov element
        if (parent_agent->get_innovation_table_value(id_pair) == -1){
            innov_num = parent_agent->get_innovation_table_size();
            parent_agent->set_innovation_table_value(id_pair, innov_num);
        }
        //Found so take that table slots innov number
        else{
            innov_num = parent_agent->get_innovation_table_value(id_pair);
        }
    }

    //Add the connection with the innov number
    std::vector<float> new_connection = {(float)first_id, (float)second_id, weight, 1.0f, (float)innov_num};
    connection_data.push_back(new_connection);
}

godot::NEATAgent* Network::get_parent_agent() const{
    return parent_agent;
}

int Network::get_inputs() const{
    return inputs;
}

int Network::get_outputs() const{
    return outputs;
}

std::vector<std::vector<float>>& Network::get_connection_data(){
    return connection_data;
}

std::map<int, Neuron*>& Network::get_neurons(){
    return neurons;
}

Neuron* Network::get_neuron(const int id) const{
    if (neurons.count(id) == 0) return nullptr;
    return neurons.at(id);
}

float Network::get_fitness() const{
    return fitness;
}

void Network::set_fitness(float amount){
    fitness = amount;
}

float Network::get_adjusted_fitness() const{
    return adjusted_fitness;
}

void Network::set_adjusted_fitness(float amount){
    adjusted_fitness = amount;
}