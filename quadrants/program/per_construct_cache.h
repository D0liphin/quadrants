#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "quadrants/common/serialization.h"
#include "quadrants/ir/ir.h"  // Stmt (the cached OffloadedStmt tasks)

namespace quadrants::lang {

// What one top-level construct contributed to the kernel, recorded so a warm process can reproduce it WITHOUT running
// that construct's frontend (merge_global_ptrs / full_simplify / offload) at all. Just the ordered per-task artifact
// keys: each names a `PerTaskArtifact` that already holds the task's cubin *and* its launch metadata, so the
// construct is fully described by this list of strings.
//
// Keyed by the CROSS-PROCESS construct key (`get_hashed_per_construct_disk_key`); the in-memory construct key is not
// sufficient, as it deliberately drops the full SNode-layout hash and the device caps.
//
// The keys embed each task's index within the kernel (`...#<index>`), and that index is baked into the compiled
// entry-function / `shared_array_t{id}` / `adstack_row_counters[id]` names. So a manifest is only reusable if the
// construct lands at the same task offset; the loader verifies this and treats a shift as a miss.
struct ConstructManifest {
  std::vector<std::string> task_keys;
  QD_IO_DEF(task_keys);
};

class ConstructManifestCache {
 public:
  explicit ConstructManifestCache(std::string dir) : dir_(std::move(dir)) {
  }

  bool try_load(const std::string &ckey, ConstructManifest *out) const {
    if (dir_.empty() || out == nullptr) {
      return false;
    }
    const auto p = path_for(ckey);
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
      return false;
    }
    return read_from_binary_file(*out, p);
  }

  void store(const std::string &ckey, const ConstructManifest &m) const {
    if (dir_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    const auto p = path_for(ckey);
    // temp+rename, and the temp must keep the `.qdb` suffix the serializer requires
    const auto tmp =
        p + ".tmp" +
        std::to_string((unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()) + ".qdb";
    write_to_binary_file(m, tmp);
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
      std::filesystem::remove(tmp, ec);
    }
  }

 private:
  std::string path_for(const std::string &ckey) const {
    std::string safe = ckey;
    for (char &c : safe) {
      if (c == '#' || c == '/') {
        c = '_';
      }
    }
    return (std::filesystem::path(dir_) / (safe + ".qdb")).string();
  }

  std::string dir_;
};

// Point `offline_cache_file_path` somewhere fresh to get a cold read of this tier.
inline std::string construct_manifest_dir_for(const std::string &offline_cache_file_path) {
  if (offline_cache_file_path.empty()) {
    return std::string("/tmp/qd_construct_manifests");
  }
  return offline_cache_file_path + "/construct_manifests";
}

// Side channel between the frontend split and the codegen driver, per kernel.
//
// The split decides, per construct, whether that construct's frontend can be skipped entirely -- but the per-task
// keys needed to *record* a manifest are only formed later, in codegen, because they embed the task's index within
// the whole reassembled kernel. So the two phases communicate through this table (indexed by task position) instead
// of threading new fields through the IR, which would perturb the printed-IR-derived per-task key.
//
// `artifact_key_by_task[i]` non-empty  => task i is a PLACEHOLDER: a manifest hit already named its compiled
//                                         artifact, so codegen must not compile it, only load that artifact.
// `construct_key_by_task[i]` non-empty => task i came from that construct; codegen groups the per-task keys it
//                                         computes by this and writes the construct's manifest for next time.
struct ConstructDiskPlan {
  std::vector<std::string> construct_key_by_task;
  std::vector<std::string> artifact_key_by_task;
};

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
  std::unordered_map<std::string, ConstructDiskPlan> disk_plans;                // kernel name -> cross-process plan
};

}  // namespace quadrants::lang
