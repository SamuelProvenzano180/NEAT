#include "GenomeData.h"

using namespace godot;

void GenomeData::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "genome_data"), &GenomeData::initialize);
    ClassDB::bind_method(D_METHOD("get_genome_contents"), &GenomeData::get_genome_contents);
    ClassDB::bind_method(D_METHOD("print"), &GenomeData::print);
}

GenomeData::GenomeData(){
    valid = false;
}

GenomeData::~GenomeData(){}

void GenomeData::initialize(const Array& genome_data){
    valid = false;
    genome_contents.clear();

    //Set fields and error check
    ERR_FAIL_COND_MSG(genome_data.size() < 3, "GenomeData Error: Network data array size too small.");

    for (int i = 0; i < 3; i++) {
        ERR_FAIL_COND_MSG(genome_data[i].get_type() != Variant::INT && genome_data[i].get_type() != Variant::FLOAT, ("GenomeData Error: Index " + std::to_string(i) + " is not a number.").c_str());
    }

    for (int i = 3; i < genome_data.size(); i++) {
        Variant item = genome_data[i];

        //Must be an array
        ERR_FAIL_COND_MSG(item.get_type() != Variant::ARRAY, ("GenomeData Error: Item at index " + std::to_string(i) + " is not an array.").c_str());

        Array conn = item;

        //Must be size 5
        ERR_FAIL_COND_MSG(conn.size() != 5, ("GenomeData Error: Connection array at index " + std::to_string(i) + " must have 5 elements.").c_str());

        //Must be [int, int, float, float, float] (float can be casted to int and int can be casted to float so accept both)
        bool id1_ok = (conn[0].get_type() == Variant::INT || conn[0].get_type() == Variant::FLOAT);
        bool id2_ok = (conn[1].get_type() == Variant::INT || conn[1].get_type() == Variant::FLOAT);
        bool weight_ok = (conn[2].get_type() == Variant::INT || conn[2].get_type() == Variant::FLOAT);
        bool from_leak_ok = (conn[3].get_type() == Variant::INT || conn[3].get_type() == Variant::FLOAT);
        bool to_leak_ok = (conn[4].get_type() == Variant::INT || conn[4].get_type() == Variant::FLOAT);

        ERR_FAIL_COND_MSG(!id1_ok || !id2_ok || !weight_ok || !from_leak_ok || !to_leak_ok, ("GenomeData Error: Connection array at index " + std::to_string(i) + " must have types [int, int, float, float, float].").c_str());
    }

    valid = true;
    genome_contents = genome_data;
}

Array GenomeData::get_genome_contents() const{
    ERR_FAIL_COND_V_MSG(!valid, Array(), "GenomeData Error: GenomeData not valid. Initialize GenomeData first.");
    return genome_contents;
}

void GenomeData::print() const{

    ERR_FAIL_COND_MSG(!valid, "GenomeData Error: GenomeData not valid. Initialize GenomeData first.");

    int inputs = (int)genome_contents[0];
    int outputs = (int)genome_contents[1];
    int max_memory_frames = (int)genome_contents[2];

    int max_neuron_id = -1;
    std::map<int, float> neuron_data;
    for (int i = 3; i < genome_contents.size(); i++){
        Array this_connection = genome_contents[i];
        int from = this_connection[0];
        int to = this_connection[1];
        float from_leak = this_connection[3];
        float to_leak = this_connection[4];
        neuron_data.insert({from, from_leak});
        neuron_data.insert({to, to_leak});

        if (max_neuron_id == -1 || from > max_neuron_id) max_neuron_id = from;
        if (to > max_neuron_id) max_neuron_id = to;
    }

    UtilityFunctions::print("neuron Data:");
    for (int i = 0; i < max_neuron_id+1; i++){
        String type;
        if (i == inputs-1) type = "Bias";
        else if (i < inputs) type = "Input";
        else if (i >= inputs && i < inputs + outputs) type = "Output";
        else type = "Hidden";
        String leak;
        if (neuron_data.count(i) > 0){
            float leak_decimal = 1.0f - (neuron_data[i] - 1.0f) / (max_memory_frames - 1.0f);
            leak = String::num_real(leak_decimal * 100.0f) + "%";
        } else {
            leak = "No Connections";
        }
        UtilityFunctions::print("ID #" + String::num_int64(i) + " | Type: " + type + " | Leak Amount: " + leak);
    }

    UtilityFunctions::print("Connection Data:");
    if (genome_contents.size() == 0){
        UtilityFunctions::print("No Connection Data!");
    }
    for (int i = 3; i < genome_contents.size(); i++){
        Array this_connection = genome_contents[i];

        String from = this_connection[0];
        String to = this_connection[1];
        String weight = this_connection[2];

        UtilityFunctions::print("Connection #" + String::num_int64(i-2) + " | From: " + from + " - To: " + to, " - Weight: " + weight);
    }
}

bool GenomeData::is_valid() const{
    return valid;
}