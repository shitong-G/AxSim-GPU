/**
 * ABC netlist reading → CircuitSoA
 *
 * Reads AIG/BLIF/Verilog etc. via ABC and fills SoA for GPU simulation.
 * Requires linking with ABC library and ABC include path (e.g. -I path/to/abc/src).
 */

#ifndef AXSIM_ABC_INTERFACE_HPP
#define AXSIM_ABC_INTERFACE_HPP

#include "circuit_soa.hpp"
#include <string>

namespace axsim {

/**
 * Read netlist from file and convert to SoA (AIG form).
 *
 * Supported formats: AIGER, BLIF, Verilog, BENCH, etc. (ABC's Io_ReadFileType).
 * The network is strashed to AIG; AND(x, Const1) is folded to x.
 *
 * @param filename  Path to .aig, .blif, .v, etc.
 * @return SoA; use valid() to check. On failure returns empty/invalid SoA.
 */
CircuitSoA soa_from_abc_file(const char* filename);

inline CircuitSoA soa_from_abc_file(const std::string& filename) {
    return soa_from_abc_file(filename.c_str());
}

} // namespace axsim

#endif
