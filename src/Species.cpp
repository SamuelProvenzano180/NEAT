#include "Species.h"
#include "Network.h"

void Species::add_member(Network* network){
    this->size++;
    this->networks.push_back(network);
}

void Species::sort_networks() {
    std::sort(this->networks.begin(), this->networks.end(), [](Network* a, Network* b) {
        return a->fitness > b->fitness;
    });
}

float Species::evaluate_compatibility(Network* candidate){
    //Set coefficient values
    float c1 = 1.0;
    float c2 = 1.0;
    float c3 = 0.4;

    //Get both genes to compare
    const std::vector<std::vector<float>>& genes1 = candidate->get_connection_data();
    const std::vector<std::vector<float>>& genes2 = this->representative_genome;

    auto it1 = genes1.begin();
    auto it2 = genes2.begin();

    int matching = 0;
    int disjoint = 0;
    int excess = 0;
    float weight_diff_sum = 0.0f;

    //Determine if specific genes are matching, disjoint or excess
    while (it1 != genes1.end() || it2 != genes2.end()) {

        // heck if we reached end of one list (Excess genes)
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
            matching++;
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

    //Determine compatability value
    float max_size = std::max(genes1.size(), genes2.size());
    float N = (max_size < 20.0f) ? 1.0f : max_size;

    float term1 = (c1 * excess) / N;
    float term2 = (c2 * disjoint) / N;
    float term3 = 0.0f;
    
    if (matching > 0) {
        term3 = c3 * (weight_diff_sum / matching);
    }

    return term1 + term2 + term3;
}

Network* Species::perform_crossover(Network* netA, Network* netB, std::mt19937 &gen){
    std::vector<std::vector<float>> new_connection_data;

    std::uniform_real_distribution<float> dis(0.0, 1.0);

    //Determine more and less fit parent
    Network* more_fit = nullptr;
    Network* less_fit = nullptr;
    if (netA->fitness > netB->fitness){
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
    for (std::vector<float> connection: less_fit->connection_data){
        float innov_num = connection[4];
        float weight = connection[2];
        less_fit_data.insert({(int)innov_num, weight});
    }
    //Cycle through more fit parent connectoin data
    for (std::vector<float> connection: more_fit->connection_data){
        float innov_num = connection[4];
        float weight = connection[2];
        //Exists in prev_data so matching gene
        if (less_fit_data.count((int)innov_num) > 0){
            //Random choice from this weight and other weight
            if (dis(gen) > 0.5){
                new_connection_data.push_back(connection);
            }
            else{
                std::vector<float> new_genome = {connection[0], connection[1], less_fit_data[innov_num], connection[3], innov_num};
                new_connection_data.push_back(new_genome);
            }
        }
        //Disjoint/excess
        else{
            new_connection_data.push_back(connection);
        }
    }

    //Create and return the new child network
    return new Network(netA->inputs, netA->outputs, &more_fit->get_depth_data(), &new_connection_data, netA->hidden_func_str, netA->output_func_str, true, gen, netA->parent_agent);
}