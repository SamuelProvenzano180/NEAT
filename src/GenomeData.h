#ifndef GENOMEDATA_H
#define GENOMEDATA_H

#include <map>
#include <godot_cpp/classes/ref_counted.hpp>

namespace godot {

    class GenomeData : public RefCounted {
        GDCLASS(GenomeData, RefCounted);
    
    public:
        GenomeData();
        ~GenomeData();
        void initialize(const Array& genome_data);

        Array get_genome_contents() const;
        bool is_valid() const;
        void print() const;

    private:
        static void _bind_methods();

        Array genome_contents;
        bool valid;
    };
};

#endif