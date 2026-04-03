/**
 * Shared primary-input pattern file (AXPI010) for CPU Verilog eval and GPU axsim.
 * Same packed layout as GPU scatter: pi_packed[pi * num_blocks_64 + block_idx] (uint64).
 */

#ifndef AXSIM_PATTERN_FILE_HPP
#define AXSIM_PATTERN_FILE_HPP

#include <cstdint>
#include <cstdio>
#include <vector>

namespace axsim {

/// Load AXPI010; builds GPU scatter buffer. Returns false on I/O / magic mismatch.
bool load_axpi010_file(
    const char* path,
    int expected_num_pis,
    std::size_t expected_num_patterns,
    std::vector<std::uint64_t>& out_pi_packed
);

} // namespace axsim

#endif
