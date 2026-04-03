#include "axsim/pattern_file.hpp"

#include <cstring>
#include <fstream>

namespace axsim {

namespace {

constexpr char kMagic[8] = {'A', 'X', 'P', 'I', '0', '1', '0', '\0'};

} // namespace

bool load_axpi010_file(
    const char* path,
    int expected_num_pis,
    std::size_t expected_num_patterns,
    std::vector<std::uint64_t>& out_pi_packed)
{
    out_pi_packed.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[axsim] cannot open patterns file: %s\n", path);
        return false;
    }
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, kMagic, 8) != 0) {
        std::fprintf(stderr, "[axsim] patterns file: bad magic (expected AXPI010)\n");
        return false;
    }
    std::uint32_t num_pi_bits = 0;
    std::uint64_t num_patterns = 0;
    f.read(reinterpret_cast<char*>(&num_pi_bits), sizeof(num_pi_bits));
    f.read(reinterpret_cast<char*>(&num_patterns), sizeof(num_patterns));
    if (!f) {
        std::fprintf(stderr, "[axsim] patterns file: truncated header\n");
        return false;
    }
    if (static_cast<int>(num_pi_bits) != expected_num_pis) {
        std::fprintf(stderr,
            "[axsim] patterns file: num_pi_bits=%u but circuit has num_pis=%d\n",
            static_cast<unsigned>(num_pi_bits), expected_num_pis);
        return false;
    }
    if (num_patterns != static_cast<std::uint64_t>(expected_num_patterns)) {
        std::fprintf(stderr,
            "[axsim] patterns file: num_patterns=%llu but config has %zu\n",
            static_cast<unsigned long long>(num_patterns),
            expected_num_patterns);
        return false;
    }
    std::size_t num_blocks_64 = (static_cast<std::size_t>(num_patterns) + 63u) / 64u;
    const std::size_t n = static_cast<std::size_t>(num_pi_bits) * num_blocks_64;
    out_pi_packed.resize(n);
    f.read(reinterpret_cast<char*>(out_pi_packed.data()), static_cast<std::streamsize>(n * sizeof(std::uint64_t)));
    if (!f) {
        std::fprintf(stderr, "[axsim] patterns file: truncated payload\n");
        out_pi_packed.clear();
        return false;
    }
    return true;
}

} // namespace axsim
