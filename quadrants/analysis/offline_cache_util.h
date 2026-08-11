#pragma once

#include <string>

#include "quadrants/rhi/arch.h"

namespace quadrants::lang {

struct CompileConfig;
struct DeviceCapabilityConfig;
class Program;
class IRNode;
class SNode;
class Kernel;
class OffloadedStmt;

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel);

// Per-offloaded-task cache key. See perso_hugh/doc/quadrants_per_task_ir_key_design_2026jul22.md for the soundness
// contract. Folds the task's CHI IR, compile_config, device caps, the layout signature of every SNode tree the task
// touches, the owning kernel's argument/return ABI (the context struct the task's compiled code reads/writes against),
// and autodiff_mode into one deterministic, collision-safe key. `task` must be the post-`re_id` single-task IR (as
// produced per task in `KernelCodeGen::compile_kernel_to_module`).
std::string get_hashed_per_task_cache_key(const CompileConfig &config,
                                          const DeviceCapabilityConfig &caps,
                                          OffloadedStmt *task,
                                          const Kernel *kernel);

// Per-construct cache key. Keys the pre-offload, isolated, re-id'd construct sub-IR (the input to the per-construct
// frontend split) so an unchanged construct's frontend output (its OffloadedStmt tasks) can be reused on a warm
// compile instead of re-running simplify/merge_global_ptrs/offload. Folds the same wideners as the per-task key --
// compile config, touched-SNode tree ids, kernel argument/return ABI, autodiff mode, and the construct body IR --
// EXCEPT device caps: the construct cache is in-memory and program-scoped, so caps are invariant within it (a fresh
// Program from `qd.init` starts with an empty cache). `construct` is the isolated construct block AFTER lower_ast +
// backward-slice isolation + die, and must be re-id'd by the caller for determinism.
std::string get_hashed_per_construct_cache_key(const CompileConfig &config,
                                               IRNode *construct,
                                               const Kernel *kernel);

void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace quadrants::lang
