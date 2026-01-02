#ifndef NETWORKAGENT_H
#define NETWORKAGENT_H

#include <vector>
#include <map>
#include <set>
#include <godot_cpp/classes/ref_counted.hpp>

namespace godot {
    class NetworkAgent : public RefCounted {
        GDCLASS(NetworkAgent, RefCounted);
    protected:
        static void _bind_methods();
    private:
        int inputs = -1;
        int outputs = -1;
        int hidden_function = -1;
        int output_function = -1;
        std::vector<float> values;
        std::vector<std::pair<std::pair<int, int>, float>> connections;

        void initialize_agent(Array network_data);
        PackedFloat32Array guess(PackedFloat32Array inputs);
        std::vector<float> packed_to_vector_float(const PackedFloat32Array &array);
        PackedFloat32Array vector_to_packed_float(const std::vector<float> &vec);
        float activation_func(float x, int type);
    };
};

#endif