#include "NetworkAgent.h"
#include "GenomeData.h"

using namespace godot;

void NetworkAgent::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize_agent", "genome_data"), &NetworkAgent::initialize_agent);
    ClassDB::bind_method(D_METHOD("guess", "inputs"), &NetworkAgent::guess);
    ClassDB::bind_method(D_METHOD("clear_memory"), &NetworkAgent::clear_memory);
}

NetworkAgent::NetworkAgent(){
    valid = false;
}

NetworkAgent::~NetworkAgent(){
    if (network) delete network;
}

void NetworkAgent::initialize_agent(const Ref<GenomeData> genome_data){
    valid = false;

    //Error check
    ERR_FAIL_COND_MSG(!genome_data.is_valid(), "NetworkAgent Error: GenomeData not valid.");

    //Grab the genome contents from genome_data object
    Array new_network_data = genome_data->get_genome_contents().duplicate(true);

    //Initialze fields from genome content
    inputs = new_network_data.pop_front();
    outputs = new_network_data.pop_front();

    new_network_data.pop_front();

    std::map<int, float> neuron_leak;
    std::map<int, Neuron*> neuron_data;

    
    int max_neuron_id = -1;
    //Cycle through network data from genome
    for (int i = 0; i < new_network_data.size(); i++){
        //Seperate the connection data into unique variabled
        Array this_conn = new_network_data[i];
        int from = this_conn[0];
        int to = this_conn[1];
        float from_leak = this_conn[3];
        float to_leak = this_conn[4];

        //Store the leaks
        neuron_leak.insert({from, from_leak});
        neuron_leak.insert({to, to_leak});

        //Find the max_neuron_id
        if (from > max_neuron_id) max_neuron_id = from;
        if (to > max_neuron_id) max_neuron_id = to;
    }

    //Cycle through every neuron ID and create the neuron
    for (int i = 0; i < max_neuron_id + 1; i++){
        Neuron::Type type;
        if (i < inputs) type = Neuron::Type::INPUT;
        else if (i < inputs + outputs) type = Neuron::Type::OUTPUT;
        else type = Neuron::Type::HIDDEN;

        Neuron* new_neuron = new Neuron(i, type, neuron_leak[i]);
        neuron_data.insert({i, new_neuron});
    }

    //Create the connection data with all connections enabled
    std::vector<std::vector<float>> connection_data;
    int innov_num = 0;
    for (auto conn: new_network_data){
        //Seperate the connection data into unique variabled
        Array conn_array = conn;
        int from = conn_array[0];
        int to = conn_array[1];
        float weight = conn_array[2];

        std::vector<float> connection = {(float)from, (float)to, weight, 1.0f, (float)(innov_num)};
        connection_data.push_back(connection);

        innov_num++;
    }

    if (network) delete network;

    valid = true;
    network = new Network(inputs, outputs, &neuron_data, &connection_data);
}

PackedFloat32Array NetworkAgent::guess(const PackedFloat32Array input_array){
    //Error check
    ERR_FAIL_COND_V_MSG(!valid, PackedFloat32Array(), "NetworkAgent Error: NetworkAgent not valid. Initialize NetworkAgent first.");
    ERR_FAIL_COND_V_MSG(input_array.size() != inputs-1, PackedFloat32Array(), "NetworkAgent Error: input_array size is not equal to expected input size.");

    std::vector<float> input_vec = NetworkAgent::packed_to_vector_float(input_array);
    //Add the bias input
    input_vec.push_back(1.0f);
    //Get the guess
    std::vector<float> guess = network->guess(input_vec);
    return NetworkAgent::vector_to_packed_float(guess);
}

void NetworkAgent::clear_memory(){
    //Error check
    ERR_FAIL_COND_MSG(!valid, "NetworkAgent Error: NetworkAgent not valid. Initialize NetworkAgent first.");
    //Clear the networks memory
    network->clear_memory();
}

float NetworkAgent::activation_func(const float x) const{
    return tanh(x);
}

std::vector<float> NetworkAgent::packed_to_vector_float(const PackedFloat32Array &array) {
    std::vector<float> vec(array.size());
    for (int i = 0; i < array.size(); i++) vec[i] = array[i];
    return vec;
}

PackedFloat32Array NetworkAgent::vector_to_packed_float(const std::vector<float> &vec) {
    PackedFloat32Array array;
    array.resize((int)vec.size());
    for (int i = 0; i < array.size(); i++) array[i] = vec[i];
    return array;
}