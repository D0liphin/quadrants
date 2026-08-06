#pragma once

#include <string>

#include "quadrants/inc/constants.h"  // AutodiffMode
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

// Per-construct cache key (S2 / §9.B). Keys the pre-offload, isolated, re-id'd construct sub-IR (the input to the
// per-construct frontend split) so an unchanged construct's frontend output (its OffloadedStmt tasks) can be reused on
// a warm compile instead of re-running simplify/merge_global_ptrs/offload. Folds the same wideners as the per-task key
// -- compile config, touched-SNode layout+tree-id, kernel argument/return ABI, autodiff mode, and the construct body
// IR -- EXCEPT device caps: the construct cache is in-memory and program-scoped, so caps are invariant within it (a
// fresh Program from `qd.init` starts with an empty cache). `construct` is the isolated construct block AFTER lower_ast
// + backward-slice isolation + die, and must be re-id'd by the caller for determinism.
std::string get_hashed_per_construct_cache_key(const CompileConfig &config,
                                               IRNode *construct,
                                               const Kernel *kernel);

// CROSS-PROCESS construct key (§9.D Part A2). Same basis as `get_hashed_per_construct_cache_key` but restores the two
// wideners that the in-memory key deliberately drops because they are invariant within one `Program`: the full
// SNode-layout hash (struct-access arithmetic is inlined into the compiled tasks, so two identical constructs over
// differently-laid-out trees must not share an entry) and the device capabilities. Layout hashing is O(tree_size), so
// it is memoized per snode-tree within a single call -- computing it once per construct was a >20x cold-compile
// blowup on the genesis kernel.
std::string get_hashed_per_construct_disk_key(const CompileConfig &config,
                                              const DeviceCapabilityConfig &caps,
                                              IRNode *construct,
                                              const Kernel *kernel);

void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace quadrants::lang
