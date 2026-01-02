#include "NetworkAgent.h"

using namespace godot;

void NetworkAgent::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize_agent", "network_data"), &NetworkAgent::initialize_agent);
    ClassDB::bind_method(D_METHOD("guess", "inputs"), &NetworkAgent::guess);
}

void NetworkAgent::initialize_agent(Array network_data){

    Array new_network_data = network_data.duplicate(true);

    //Set fields and error check
    ERR_FAIL_COND_MSG(new_network_data.size() < 4, "NetworkAgent Import Error: Network data array size must be greater than 3");

    for (int i = 0; i < 4; i++) {
        ERR_FAIL_COND_MSG(new_network_data[i].get_type() != Variant::INT && new_network_data[i].get_type() != Variant::FLOAT, ("NetworkAgent Import Error: Index " + std::to_string(i) + " is not a number").c_str());
    }

    for (int i = 4; i < new_network_data.size(); i++) {
        Variant item = new_network_data[i];

        //Must be an array
        ERR_FAIL_COND_MSG(item.get_type() != Variant::ARRAY, ("NetworkAgent Import Error: Item at index " + std::to_string(i) + " is not an Array").c_str());

        Array conn = item;

        //Must be size 3
        ERR_FAIL_COND_MSG(conn.size() != 3, ("NetworkAgent Import Error: Connection at index " + std::to_string(i) + " has invalid size. Expected 3").c_str());

        //Must be [int, int, float] (float can be casted to int and int can be casted to float so accept both)
        bool id1_ok = (conn[0].get_type() == Variant::INT || conn[0].get_type() == Variant::FLOAT);
        bool id2_ok = (conn[1].get_type() == Variant::INT || conn[1].get_type() == Variant::FLOAT);
        bool weight_ok = (conn[2].get_type() == Variant::INT || conn[2].get_type() == Variant::FLOAT);

        ERR_FAIL_COND_MSG(!id1_ok || !id2_ok || !weight_ok, ("NetworkAgent Import Error: Connection at index " + std::to_string(i) + " has invalid types. Expected [int, int, float]").c_str());
    }

    this->connections.clear();
    this->inputs = new_network_data.pop_front();
    this->outputs = new_network_data.pop_front();
    this->hidden_function = new_network_data.pop_front();
    this->output_function = new_network_data.pop_front();

    //Determine network size
    int max_id = this->inputs + this->outputs - 1;

    //Load connections
    for (int i = 0; i < new_network_data.size(); i++){
        Array connection_data = new_network_data[i];
        int from_id = connection_data[0];
        int to_id = connection_data[1];
        float weight = connection_data[2];
        
        //Store connection
        std::pair<int, int> link = {from_id, to_id};
        this->connections.push_back({link, weight});

        //Track max_id to resize vector later
        if (from_id > max_id) max_id = from_id;
        if (to_id > max_id) max_id = to_id;
    }
    
    //Resize the values vector based on max id
    this->values.resize(max_id + 1, 0.0f);
}

PackedFloat32Array NetworkAgent::guess(PackedFloat32Array input_array){
    //Error check
    ERR_FAIL_COND_V_MSG(input_array.size() != this->inputs-1, PackedFloat32Array(), "NetworkAgent Guess Error: Number of inputs is not equal to expected input size");

    //Append bias input into the input array
    input_array.push_back(1.0);

    //Reset all values in the vector
    std::fill(this->values.begin(), this->values.end(), 0.0f);

    //Load inputs into the input neurons
    for (int i = 0; i < this->inputs && i < input_array.size(); i++){
        this->values[i] = input_array[i];
    }

    //Forward loop
    std::set<int> neuron_visited;
    
    for (const auto& conn : this->connections){
        int from = conn.first.first;
        int to = conn.first.second;
        float weight = conn.second;

        //Activation function
        if (from >= this->inputs){ 
            if (neuron_visited.find(from) == neuron_visited.end()){
                this->values[from] = activation_func(this->values[from], this->hidden_function);
                neuron_visited.insert(from);
            }
        }
        
        //Multiply weight by input
        this->values[to] += this->values[from] * weight;
    }

    //Collect outputs
    std::vector<float> outputs;
    for (int i = this->inputs; i < this->inputs + this->outputs; i++){
        // Apply Output Activation
        float output_val = activation_func(this->values[i], this->output_function);
        outputs.push_back(output_val);
    }

    return vector_to_packed_float(outputs);
}

float NetworkAgent::activation_func(float x, int type){
    if (type == 0){
        return (x > 0) ? x : 0.01f * x;
    }
    else if (type == 1){
        return x;
    }
    else if (type == 2){
        return 1.0 / (1.0 + exp(-x));
    }
    else if (type == 3){
        return tanh(x);
    }
    return 0.0;
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