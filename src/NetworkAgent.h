#ifndef NETWORKAGENT_H
#define NETWORKAGENT_H

#include <vector>
#include <map>
#include <set>
#include <string>
#include <godot_cpp/classes/ref_counted.hpp>
#include "GenomeData.h"
#include "Neuron.h"
#include "Network.h"

namespace godot {

    class NetworkAgent : public RefCounted {
        GDCLASS(NetworkAgent, RefCounted);

    public:
        NetworkAgent();
        ~NetworkAgent();

        void initialize_agent(const Ref<GenomeData> genome_data);
        PackedFloat32Array guess(const PackedFloat32Array input_array);
        void clear_memory();
    
    private:
        static void _bind_methods();

        bool valid;
        int inputs;
        int outputs;
        Network* network = nullptr;

        float activation_func(float x) const;
        std::vector<float> packed_to_vector_float(const PackedFloat32Array &array);
        PackedFloat32Array vector_to_packed_float(const std::vector<float> &vec);
    };
};

#endif