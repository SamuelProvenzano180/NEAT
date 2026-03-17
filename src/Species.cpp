#include "Species.h"
#include "Network.h"

Species::Species(){
    size = 0;
    age = 0;
    offspring_count = 0;
    gens_since_improved = 0;
    max_fitness_ever = 0.0f;
}

Species::~Species(){}

void Species::add_member(Network* network){
    size++;
    networks.push_back(network);
}

void Species::sort_networks() {
    std::sort(networks.begin(), networks.end(), [](Network* a, Network* b) {
        return a->get_fitness() > b->get_fitness();
    });
}

float Species::evaluate_compatibility(Network* candidate){
    //Set coefficient values
    float c1 = 1.0f;
    float c2 = 1.0f;
    float c3 = 0.4f;

    //Get both genes to compare
    const std::vector<std::vector<float>>& genes1 = candidate->get_connection_data();
    const std::vector<std::vector<float>>& genes2 = representative_connections;

    int matching_connections = 0;
    int disjoint = 0;
    int excess = 0;
    float weight_diff_sum = 0.0f;

    auto it1 = genes1.begin();
    auto it2 = genes2.begin();

    //Determine if specific genes are matching, disjoint or excess
    while (it1 != genes1.end() || it2 != genes2.end()){ //Using an iterator here for simplicity sake

        // check if we reached end of one list (Excess genes)
        if (it1 == genes1.end()) {
            excess++;
            it2++;
            continue;
        }
        if (it2 == genes2.end()) {
            excess++;
            it1++;
            continue;
        }

        int innov1 = (int)(*it1)[4];
        int innov2 = (int)(*it2)[4];

        //If innov numbers are same, matching gene
        if (innov1 == innov2) {
            matching_connections++;
            weight_diff_sum += std::abs((*it1)[2] - (*it2)[2]);
            it1++;
            it2++;
        }
        //Otherwise disjoint
        else if (innov1 < innov2) {
            disjoint++;
            it1++;
        }
        else {
            disjoint++;
            it2++;
        }
    }

    float average_leak_diff = 0.0f;
    int matching_neurons = 0;
    float max_memory_frames = candidate->get_parent_agent()->get_max_memory_frames();

    //Cycle through all neurons
    for (auto& [id, neuron]: candidate->get_neurons()){
        //If neuron exists in the representative leak, there is a match
        if (representative_leaks.count(id) > 0){
            //Get the difference in the log of the leaks
            float first_leak = neuron->get_leak_value();
            float second_leak = representative_leaks[id];
            // leak_diff_sum += std::abs(log(second_leak) - log(first_leak)) / log(max_memory_frames); This line becomes the line below
            average_leak_diff += std::abs(log(second_leak / first_leak)) / log(max_memory_frames);
            matching_neurons++;
        }
    }

    //Determine compatability value with parameters
    float term1 = c1 * excess;
    float term2 = c2 * disjoint;
    float term3 = 0.0f;
    
    if (matching_connections > 0) {
        term3 += weight_diff_sum / matching_connections;
    }
    if (matching_neurons > 0){
        term3 += average_leak_diff / matching_neurons;
    }
    term3 *= c3;

    return term1 + term2 + term3;
}

Network* Species::perform_crossover(Network* netA, Network* netB){
    std::vector<std::vector<float>> new_connection_data;

    std::uniform_real_distribution<float> rand1(0.0f, 1.0f);

    //Determine more and less fit parent
    Network* more_fit = nullptr;
    Network* less_fit = nullptr;
    if (netA->get_fitness() > netB->get_fitness()){
        more_fit = netA;
        less_fit = netB;
    }
    else{
        more_fit = netB;
        less_fit = netA;
    }

    //Innov number, weight
    std::map<int, float> less_fit_data;

    //Fill less fit parent data with the innov num and weight pair
    for (std::vector<float> connection: less_fit->get_connection_data()){
        float innov_num = connection[4];
        float weight = connection[2];
        less_fit_data.insert({(int)innov_num, weight});
    }
    //Cycle through more fit parent connectoin data
    for (std::vector<float> connection: more_fit->get_connection_data()){
        float innov_num = connection[4];
        float weight = connection[2];
        //Exists in prev_data so matching gene
        if (less_fit_data.count((int)innov_num) > 0){
            //80% chance to take weight from higher genome
            if (rand1(netA->get_parent_agent()->get_rng()) > 0.2f) {
                new_connection_data.push_back(connection);
            }
            //20% chance to average both
            else {
                float avg_weight = (connection[2] + less_fit_data[innov_num]) / 2.0f;
                std::vector<float> averaged_gene = {connection[0], connection[1], avg_weight, connection[3], innov_num};
                new_connection_data.push_back(averaged_gene);
            }
        }
        //Disjoint/excess
        else{
            new_connection_data.push_back(connection);
        }
    }

    //Create and return the new child network
    return new Network(netA->get_inputs(), netA->get_outputs(), &more_fit->get_neurons(), &new_connection_data, true, netA->get_parent_agent());
}

std::vector<Network*>& Species::get_networks(){
    return networks;
}

int Species::get_age() const{
    return age;
}

void Species::set_age(const int amount){
    age = amount;
}

int Species::get_offspring_count() const{
    return offspring_count;
}

void Species::set_offspring_count(const int amount){
    offspring_count = amount;
}

int Species::get_gens_since_improved() const{
    return gens_since_improved;
}

void Species::set_gens_since_improved(const int amount){
    gens_since_improved = amount;
}

int Species::get_max_fitness_ever() const{
    return max_fitness_ever;
}

void Species::set_max_fitness_ever(const int amount){
    max_fitness_ever = amount;
}

void Species::set_representative_connections(const std::vector<std::vector<float>>& new_value){
    representative_connections = new_value;
}

void Species::set_representative_leaks(const std::map<int, float>& new_value){
    representative_leaks = new_value;
}