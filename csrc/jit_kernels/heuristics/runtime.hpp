#pragma once

#include <deep_jit/utils/lazy.hpp>
#include <deep_jit/utils/env.hpp>

#include "../../runtime/jit.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

class HeuristicsRuntime {
public:
    static constexpr int kLegacyMKAlignmentForContiguousLayout = 128;

    bool ignore_compile_dims = false;
    bool deterministic_algorithms = false;
    int block_m_multiple_of = 1;
    int block_n_multiple_of = 1;
    int mk_alignment_for_contiguous_layout = kLegacyMKAlignmentForContiguousLayout;

    void use_deterministic_algorithms(const bool enabled) {
        deterministic_algorithms = enabled;
    }

    bool get_deterministic_algorithms() const {
        return deterministic_algorithms;
    }

    void set_ignore_compile_dims(const bool& new_value) {
        ignore_compile_dims = new_value;
    }

    bool get_ignore_compile_dims() const {
        return ignore_compile_dims;
    }

    void set_block_size_multiple_of(const int& new_block_m_multiple_of, const int& new_block_n_multiple_of) {
        block_m_multiple_of = new_block_m_multiple_of;
        block_n_multiple_of = new_block_n_multiple_of;
    }

    int get_block_m_multiple_of() const {
        return block_m_multiple_of;
    }

    int get_block_n_multiple_of() const {
        return block_n_multiple_of;
    }

    void set_mk_alignment_for_contiguous_layout(const int& new_value) {
        mk_alignment_for_contiguous_layout = new_value;
    }

    int get_mk_alignment_for_contiguous_layout() const {
        return mk_alignment_for_contiguous_layout;
    }

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
        const int arch_major = jit->device.get_arch_major();
        if (arch_major != 10 and arch_major != 11)
            return kLegacyMKAlignmentForContiguousLayout;

        // Thor opt-in (round 13): restore the pre-sync per-call shrinking alignment
        // {max 240, min 32, step 16} instead of upstream's fixed 256. Default off.
        if (arch_major == 11 and deep_jit::get_env<int>("DG_THOR_SHRINK_ALIGN", 0)) {
            int block_m = 240;
            if (expected_m.has_value()) {
                const int per_group_m = expected_m.value();
                for (; block_m > 32 and block_m - 16 >= per_group_m; block_m -= 16);
            }
            return block_m;
        }

        // The newly supported UMMA_N=256 allows a fixed alignment of 256, which is friendlier to MoE cast, M-grouped, and K-grouped operators.
        // NOTES: `expected_m` is ignored, so small values may incur performance loss; use Mega MoE directly for such workloads.
        return 256;
    }
};

inline auto heuristics_runtime = deep_jit::LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
