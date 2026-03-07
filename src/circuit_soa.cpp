/**
 * CPU-side SoA flatten: build CircuitSoA from AIG-style graph
 */

#include "axsim/circuit_soa.hpp"
#include <algorithm>
#include <cassert>

namespace axsim {

bool CircuitSoA::valid() const {
    if (num_pis < 0 || num_ands < 0 || num_nodes != num_pis + num_ands) return false;
    if ((int)fanin0_indices.size() != num_ands) return false;
    if ((int)fanin1_indices.size() != num_ands) return false;
    if ((int)is_compl0.size() != num_ands) return false;
    if ((int)is_compl1.size() != num_ands) return false;
    if ((int)output_node_ids.size() != num_outputs) return false;
    return true;
}

CircuitSoA flatten_from_aig(
    int num_pis,
    const std::vector<std::tuple<int, int, bool, bool>>& and_nodes,
    const std::vector<int>& out_ids)
{
    CircuitSoA soa;
    soa.num_pis = num_pis;
    soa.num_ands = (int)and_nodes.size();
    soa.num_nodes = soa.num_pis + soa.num_ands;
    soa.num_outputs = (int)out_ids.size();

    soa.fanin0_indices.resize(soa.num_ands);
    soa.fanin1_indices.resize(soa.num_ands);
    soa.is_compl0.resize(soa.num_ands);
    soa.is_compl1.resize(soa.num_ands);
    soa.output_node_ids = out_ids;

    for (int k = 0; k < soa.num_ands; ++k) {
        const auto& [f0, f1, c0, c1] = and_nodes[k];
        soa.fanin0_indices[k] = f0;
        soa.fanin1_indices[k] = f1;
        soa.is_compl0[k] = c0 ? 1 : 0;
        soa.is_compl1[k] = c1 ? 1 : 0;
    }

    return soa;
}

} // namespace axsim
