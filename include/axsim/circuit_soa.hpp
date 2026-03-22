/**
 * Circuit SoA (Structure of Arrays) for GPU-friendly simulation
 *
 * Assumes AIG: nodes [0, num_pis) are PIs, [num_pis, num_nodes) are AND nodes
 * in topological order. AND: out = (c0?~L:L) & (c1?~R:R).
 */

#ifndef AXSIM_CIRCUIT_SOA_HPP
#define AXSIM_CIRCUIT_SOA_HPP

#include <cstdint>
#include <tuple>
#include <vector>

namespace axsim {

struct CircuitSoA {
    int num_pis{0};       ///< number of primary inputs
    int num_nodes{0};     ///< total nodes (PIs + ANDs)
    int num_ands{0};      ///< num_nodes - num_pis
    int num_outputs{0};   ///< number of circuit outputs

    /// SoA: one entry per AND node (index 0 = first AND = node id num_pis)
    std::vector<int> fanin0_indices;   ///< left fanin node id
    std::vector<int> fanin1_indices;   ///< right fanin node id
    std::vector<uint8_t> is_compl0;    ///< invert left fanin
    std::vector<uint8_t> is_compl1;    ///< invert right fanin

    /// Which node ids are primary outputs (length num_outputs)
    std::vector<int> output_node_ids;

    /// If non-empty, is_output_complemented[o] means PO o is logically NOT of output_node_ids[o] (length num_outputs)
    std::vector<uint8_t> is_output_complemented;

    bool valid() const;
};

/**
 * Flatten external graph (e.g. from ABC) into SoA.
 * Implement in circuit_soa.cpp using your parser/library.
 *
 * @param num_pis     number of primary inputs
 * @param and_nodes   list of (fanin0_id, fanin1_id, compl0, compl1) per AND in topo order
 * @param out_ids     node ids that are circuit outputs
 * @return SoA ready for cudaMemcpy
 */
CircuitSoA flatten_from_aig(
    int num_pis,
    const std::vector<std::tuple<int, int, bool, bool>>& and_nodes,
    const std::vector<int>& out_ids
);

} // namespace axsim

#endif
