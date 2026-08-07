#pragma once

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "quadrants/codegen/llvm/llvm_compiled_data.h"
#include "quadrants/common/serialization.h"

namespace quadrants::lang {

// One offloaded task's fully-compiled artifact, persisted so a *fresh process* can launch that task without re-running
// any of its compilation: no CHI->LLVM codegen, no link, no optimize, no PTX, no ptxas. Keyed by the per-task IR key
// (`get_hashed_per_task_cache_key` + "#index"), which is derived from the task IR *before* codegen -- that is what
// makes it hittable on a warm one-task edit (§9.D Part B).
//
// The cubin alone is NOT sufficient to launch. The kernel launcher and the CUDA graph builder consume `OffloadedTask`
// metadata -- entry-function name, block/grid dim, `graph_do_while_level_id`, `stream_parallel_group_id`,
// `graph_parallel_region_id`, `checkpoint_id`, adstack sizing, and the snode/arg read-write sets used for cache
// invalidation -- and the runtime needs `used_tree_ids`. All of that is `QD_IO_DEF`-serializable, so the whole record
// is written as a single binary blob next to the code.
//
// `used_tree_ids` / `struct_for_tls_sizes` are `unordered_set<int>` on `LLVMCompiledTask`; they are stored as sorted
// vectors here so the on-disk bytes are deterministic (an unordered_set's iteration order is not).
struct PerTaskArtifact {
  std::vector<OffloadedTask> tasks;
  std::vector<int> used_tree_ids;
  std::vector<int> struct_for_tls_sizes;
  std::vector<char> cubin;  // relocatable cubin (`ptxas -c`), device-linked with the sibling tasks via cuLink
  QD_IO_DEF(tasks, used_tree_ids, struct_for_tls_sizes, cubin);
};

// Content-addressed on-disk store of `PerTaskArtifact`, shared by the codegen driver (which *probes* it before
// deciding to compile a task) and the CUDA JIT (which *fills* it once it has produced the cubin). Both sides derive
// the path from the same (dir, ir_key) pair, so they must agree on `dir` -- it comes from the compile config.
//
// Note the IR key already folds the compile config and device capabilities, so entries cannot collide across
// architectures or option sets; the directory is namespaced by arch only to keep the tree browsable.
class PerTaskArtifactCache {
 public:
  explicit PerTaskArtifactCache(std::string dir) : dir_(std::move(dir)) {
  }

  const std::string &dir() const {
    return dir_;
  }

  bool exists(const std::string &ir_key) const {
    std::error_code ec;
    return std::filesystem::exists(path_for(ir_key), ec);
  }

  bool try_load(const std::string &ir_key, PerTaskArtifact *out) const {
    if (dir_.empty() || out == nullptr) {
      return false;
    }
    const auto p = path_for(ir_key);
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
      return false;
    }
    // A truncated/corrupt record (e.g. a crash mid-write before the rename landed) must degrade to a miss, never a
    // hard failure -- the caller just recompiles the task.
    return read_from_binary_file(*out, p);
  }

  void store(const std::string &ir_key, const PerTaskArtifact &rec) const {
    if (dir_.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    const auto p = path_for(ir_key);
    // Write to a unique temp then rename: concurrent compile workers (and concurrent processes sharing the cache dir)
    // must never observe a partially written record. Same discipline as the PTX/cubin caches. The temp name must
    // itself keep the `.qdb` suffix -- `write_data_to_file` rejects any other extension.
    const auto tmp =
        p + ".tmp" +
        std::to_string((unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()) + ".qdb";
    write_to_binary_file(rec, tmp);
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
      std::filesystem::remove(tmp, ec);
    }
  }

 private:
  // The IR key is already a fixed-length hex digest plus a "#<index>" suffix; '#' is legal in POSIX filenames but is
  // awkward in shells, so swap it for '_'. The `.qdb` extension is mandatory -- the binary serializer refuses to
  // write any other suffix.
  std::string path_for(const std::string &ir_key) const {
    std::string safe = ir_key;
    for (char &c : safe) {
      if (c == '#' || c == '/') {
        c = '_';
      }
    }
    return (std::filesystem::path(dir_) / (safe + ".qdb")).string();
  }

  std::string dir_;
};

// Single source of truth for where per-task artifacts live. The codegen driver (probe side) and the CUDA JIT (store
// side) run in different layers and must resolve the same directory, so both call this rather than each building a
// path. `QD_PERTASK_ARTIFACT_DIR` overrides, mainly so tests can point at a scratch dir.
inline std::string pertask_artifact_dir_for(const std::string &offline_cache_file_path) {
  if (const char *e = std::getenv("QD_PERTASK_ARTIFACT_DIR")) {
    return std::string(e);
  }
  if (offline_cache_file_path.empty()) {
    return std::string("/tmp/qd_pertask_artifacts");
  }
  return offline_cache_file_path + "/pertask_artifacts";
}

// `CompiledKernelData::load_impl` has to find the artifact cache but sits far from any `CompileConfig`. The artifacts
// always live beside the `.qdc` files, so resolve the directory once when the LLVM program is constructed (which is
// before any kernel is loaded or compiled) instead of threading the config through the whole load path.
inline std::string &pertask_artifact_dir_ref() {
  static std::string dir;
  return dir;
}

inline void set_pertask_artifact_dir_from_offline_cache(const std::string &offline_cache_file_path) {
  pertask_artifact_dir_ref() = pertask_artifact_dir_for(offline_cache_file_path);
}

inline std::string resolved_pertask_artifact_dir() {
  if (const char *e = std::getenv("QD_PERTASK_ARTIFACT_DIR")) {
    return std::string(e);
  }
  const auto &d = pertask_artifact_dir_ref();
  return d.empty() ? pertask_artifact_dir_for(std::string()) : d;
}

}  // namespace quadrants::lang
