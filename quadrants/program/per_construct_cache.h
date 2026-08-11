#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "quadrants/ir/ir.h"  // Stmt (the cached OffloadedStmt tasks)

namespace quadrants::lang {

// Program-scoped in-memory cache of the per-construct FRONTEND output. Each entry is the vector of OffloadedStmt
// tasks produced by running the isolated-construct frontend (simplify -> merge_global_ptrs -> offload -> ...) on ONE
// construct, keyed by the per-construct IR key (`get_hashed_per_construct_cache_key`). On a warm compile the split
// path recomputes only the changed construct's frontend and clones the rest out of here, skipping the dominant
// whole-kernel simplify/mgp/offload cost.
//
// Stores quadrants IR only (no LLVM), so it is arch-agnostic and hangs off the base `Program` (unlike the LLVM-module
// `PerTaskModuleCache`, which lives on `LlvmProgramImpl`). Content-keyed: identical constructs in different kernels
// (with the same ABI/config) share an entry -- that is what makes "edit one construct" reuse the other N-1. Cloned in
// and out under `mu`; the expensive frontend on a miss runs OUTSIDE `mu` so it does not serialize compilation.
//
// SNode tree-ids and compile config are invariant within one Program, so the key omits device caps (a fresh Program
// from `qd.init` starts with an empty cache).
struct PerConstructCache {
  // Per-kernel frontend-split cache stats, recorded by the split driver and read back by the codegen driver into
  // `PerTaskCacheStats` so Python (`PerOffloadCacheObservations`) can assert a warm one-construct edit
  // (recompiled == 1, hit == total - 1). Keyed by kernel name; overwritten on each frontend split of that kernel.
  struct Stats {
    int total = 0;
    int hit = 0;
    int recompiled = 0;
  };

  std::mutex mu;
  std::unordered_map<std::string, std::vector<std::unique_ptr<Stmt>>> entries;  // ckey -> cloned OffloadedStmt tasks
  std::unordered_map<std::string, Stats> last_stats;                            // kernel name -> most-recent split
};

}  // namespace quadrants::lang
