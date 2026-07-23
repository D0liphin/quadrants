// Driver class for kernel codegen

#include <chrono>

#include "codegen.h"

#if defined(QD_WITH_LLVM)
#include "quadrants/codegen/cpu/codegen_cpu.h"
#include "quadrants/runtime/program_impls/llvm/llvm_program.h"
#endif
#if defined(QD_WITH_CUDA)
#include "quadrants/codegen/cuda/codegen_cuda.h"
#endif
#if defined(QD_WITH_AMDGPU)
#include "quadrants/codegen/amdgpu/codegen_amdgpu.h"
#endif
#include "quadrants/system/timer.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/offloaded_task_type.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/rhi/device_capability.h"

#if defined(QD_WITH_LLVM)
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <string>

namespace quadrants::lang {

KernelCodeGen::KernelCodeGen(const CompileConfig &compile_config,
                             const Kernel *kernel,
                             IRNode *ir,
                             QuadrantsLLVMContext &tlctx)
    : prog(kernel->program), kernel(kernel), ir(ir), compile_config_(compile_config), tlctx_(tlctx) {
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(const CompileConfig &compile_config,
                                                     const Kernel *kernel,
                                                     IRNode *ir,
                                                     QuadrantsLLVMContext &tlctx) {
#ifdef QD_WITH_LLVM
  const auto arch = compile_config.arch;
  if (arch_is_cpu(arch)) {
    return std::make_unique<KernelCodeGenCPU>(compile_config, kernel, ir, tlctx);
  } else if (arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    return std::make_unique<KernelCodeGenCUDA>(compile_config, kernel, ir, tlctx);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else if (arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    return std::make_unique<KernelCodeGenAMDGPU>(compile_config, kernel, ir, tlctx);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else {
    QD_NOT_IMPLEMENTED
  }
#else
  QD_ERROR("Llvm disabled");
#endif
}
#ifdef QD_WITH_LLVM

LLVMCompiledKernel KernelCodeGen::compile_kernel_to_module() {
  auto block = dynamic_cast<Block *>(ir);
  auto &worker = get_llvm_program(kernel->program)->compilation_workers;
  QD_ASSERT(block);

  // Prototype (A1) per-task cache key: with QD_PERTASK_KEY_LOG=1, compute and log each task's key. Compute-and-log
  // only -- it does not gate caching yet (see perso_hugh/doc/quadrants_per_task_ir_key_design_2026jul22.md). Keys are
  // built in the worker threads into an indexed vector and printed in order after flush() to avoid log interleaving.
  static const bool log_pertask_key = []() {
    const char *e = std::getenv("QD_PERTASK_KEY_LOG");
    return e != nullptr && std::string(e) == "1";
  }();

  // Env-gated (QD_PHASE_TIME=1) compile phase timing: per-task codegen / link / whole-module optimize.
  static const bool phase_time = []() {
    const char *e = std::getenv("QD_PHASE_TIME");
    return e != nullptr && std::string(e) == "1";
  }();
  auto _pt_now = []() { return std::chrono::steady_clock::now(); };
  auto _pt_ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  auto t_pertask0 = _pt_now();

  auto &offloads = block->statements;
  std::vector<std::unique_ptr<LLVMCompiledTask>> data(offloads.size());
  std::vector<std::string> pertask_keys;
  DeviceCapabilityConfig pertask_caps;
  if (log_pertask_key) {
    pertask_keys.resize(offloads.size());
    pertask_caps = prog->get_device_caps();
  }
  for (int i = 0; i < offloads.size(); i++) {
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());

      if (log_pertask_key) {
        pertask_keys[i] =
            get_hashed_per_task_cache_key(compile_config_, pertask_caps, offload->as<OffloadedStmt>(),
                                          kernel->autodiff_mode);
      }

      Block blk;
      blk.insert(std::move(offload));
      auto new_data = this->compile_task(i, compile_config_, nullptr, &blk);
      data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
    };
    worker.enqueue(compile_func);
  }
  worker.flush();

  // Env-gated (QD_TASK_SYMBOL_CENSUS=1) census of per-task module symbols, to assess cuLink device-link feasibility
  // (migration D/E). Classifies each symbol as: local definition (safe, no collision), non-local definition
  // (collision risk if the same name is defined by >1 task module), or declaration (a cross-module reference a
  // runtime/struct cubin must resolve at cuLink). Reports the true strong-collision set and the external-dep set.
  static const bool task_symbol_census = []() {
    const char *e = std::getenv("QD_TASK_SYMBOL_CENSUS");
    return e != nullptr && std::string(e) == "1";
  }();
  if (task_symbol_census) {
    std::map<std::string, int> nonlocal_def_count;  // name -> #task modules with a non-local definition
    std::map<std::string, int> strong_def_count;    // name -> #task modules with a strong (external) definition
    std::set<std::string> any_def;                  // names defined (any linkage) in some task module
    std::set<std::string> decl_refs;                // names declared (referenced) in some task module
    std::vector<int> per_task_nonlocal(data.size(), 0), per_task_local(data.size(), 0), per_task_decl(data.size(), 0);
    auto scan = [&](int i, llvm::GlobalValue &gv) {
      const std::string name = gv.getName().str();
      if (gv.isDeclaration()) {
        per_task_decl[i]++;
        decl_refs.insert(name);
        return;
      }
      any_def.insert(name);
      if (gv.hasLocalLinkage()) {
        per_task_local[i]++;
      } else {
        per_task_nonlocal[i]++;
        nonlocal_def_count[name]++;
        if (!gv.isWeakForLinker())
          strong_def_count[name]++;
      }
    };
    for (int i = 0; i < (int)data.size(); i++) {
      if (!data[i] || !data[i]->module)
        continue;
      for (auto &F : data[i]->module->functions())
        scan(i, F);
      for (auto &G : data[i]->module->globals())
        scan(i, G);
    }
    int collide_names = 0, strong_collide = 0, external_deps = 0, max_nonlocal = 0;
    for (auto &kv : nonlocal_def_count)
      if (kv.second > 1)
        collide_names++;
    for (auto &kv : strong_def_count)
      if (kv.second > 1)
        strong_collide++;
    for (auto &n : decl_refs)
      if (!any_def.count(n))
        external_deps++;
    for (int i = 0; i < (int)data.size(); i++)
      max_nonlocal = std::max(max_nonlocal, per_task_nonlocal[i]);
    QD_INFO(
        "[task-symbol-census] kernel={} n_tasks={} nonlocal_def_names={} collide_names(nonlocal,>1mod)={} "
        "strong_collide_names={} external_dep_symbols(decl-not-defined-in-any-task)={} max_nonlocal_defs_per_task={}",
        kernel->get_name(), (int)data.size(), (int)nonlocal_def_count.size(), collide_names, strong_collide,
        external_deps, max_nonlocal);
    int printed = 0;
    for (auto &kv : strong_def_count) {
      if (kv.second > 1 && printed < 25) {
        QD_INFO("[task-symbol-census]   STRONG-COLLIDE {} defined-in {} task modules", kv.first, kv.second);
        printed++;
      }
    }
    printed = 0;
    for (auto &n : decl_refs) {
      if (!any_def.count(n) && printed < 15) {
        QD_INFO("[task-symbol-census]   EXTERNAL-DEP {}", n);
        printed++;
      }
    }
  }

  // Env-gated (QD_PERTASK_SELFCONTAINED=1) probe of option A (self-contained per-construct cubins). For the single
  // largest task module, run the *normal* link+optimize on a 1-task list (this pulls in runtime + libdevice via
  // LinkOnlyNeeded and inlines under -O3), then census the RESULT module: how many external declarations remain (=
  // symbols a shared cubin would still need to resolve; 0 => already self-contained) and how many non-local
  // definitions remain besides the entry function (= would collide across cubins => need an internalize pass). Also
  // times link+optimize for one task (the per-construct codegen cost paid only by the changed construct on an edit).
  static const bool selfcontained_probe = []() {
    const char *e = std::getenv("QD_PERTASK_SELFCONTAINED");
    return e != nullptr && std::string(e) == "1";
  }();
  if (selfcontained_probe) {
    int imax = -1;
    long long best = -1;
    for (int i = 0; i < (int)data.size(); i++) {
      if (!data[i] || !data[i]->module)
        continue;
      long long ic = 0;
      for (auto &F : data[i]->module->functions())
        ic += F.getInstructionCount();
      if (ic > best) {
        best = ic;
        imax = i;
      }
    }
    if (imax >= 0) {
      std::vector<std::unique_ptr<LLVMCompiledTask>> one;
      one.push_back(std::make_unique<LLVMCompiledTask>(data[imax]->clone()));
      auto _t0 = std::chrono::steady_clock::now();
      auto linked_one = tlctx_.link_compiled_tasks(std::move(one));
      auto _t1 = std::chrono::steady_clock::now();
      optimize_module(linked_one.module.get());
      auto _t2 = std::chrono::steady_clock::now();
      int remaining_decls = 0, nonlocal_defs = 0, entry_defs = 0;
      std::set<std::string> entry_names;
      for (auto &t : linked_one.tasks)
        entry_names.insert(t.name);
      auto scan2 = [&](llvm::GlobalValue &gv, bool is_intrinsic) {
        if (gv.isDeclaration()) {
          if (!is_intrinsic)
            remaining_decls++;
          return;
        }
        if (!gv.hasLocalLinkage()) {
          if (entry_names.count(gv.getName().str()))
            entry_defs++;
          else
            nonlocal_defs++;
        }
      };
      for (auto &F : linked_one.module->functions())
        scan2(F, F.isIntrinsic());
      for (auto &G : linked_one.module->globals())
        scan2(G, false);
      auto _ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
      QD_INFO(
          "[selfcontained-probe] kernel={} largest_task_idx={} pretask_instrs={} link_ms={:.1f} optimize_ms={:.1f} "
          "post_link: remaining_external_decls(non-intrinsic)={} nonlocal_defs_besides_entry={} entry_defs={}",
          kernel->get_name(), imax, best, _ms(_t0, _t1), _ms(_t1, _t2), remaining_decls, nonlocal_defs, entry_defs);
    }
  }

  if (log_pertask_key) {
    for (int i = 0; i < (int)pertask_keys.size(); i++) {
      QD_INFO("[pertask-key] kernel={} task={} type={} key={}", kernel->get_name(), i,
              offloaded_task_type_name(offloads[i]->as<OffloadedStmt>()->task_type), pertask_keys[i]);
    }
  }

  auto t_link0 = _pt_now();
  auto llvm_compiled_kernel = tlctx_.link_compiled_tasks(std::move(data));
  auto t_opt0 = _pt_now();
  optimize_module(llvm_compiled_kernel.module.get());
  auto t_end = _pt_now();
  if (phase_time) {
    QD_INFO(
        "[phase-time] kernel={} n_tasks={} pertask_compile={:.1f}ms link={:.1f}ms optimize={:.1f}ms",
        kernel->get_name(), (int)offloads.size(), _pt_ms(t_pertask0, t_link0), _pt_ms(t_link0, t_opt0),
        _pt_ms(t_opt0, t_end));
  }
  return llvm_compiled_kernel;
}

#endif
}  // namespace quadrants::lang
