#include "quadrants/ir/ir.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/pass.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/program/extension.h"
#include "quadrants/program/function.h"
#include "quadrants/program/kernel.h"
#include "quadrants/program/program.h"
#include "quadrants/program/per_construct_cache.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/codegen/ir_dump.h"
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <algorithm>
#include <functional>
#include <typeinfo>
#include <string>
#include <cstdint>

namespace quadrants::lang {

namespace {
// S0 per-construct dependency census (env QD_CONSTRUCT_CENSUS=1). Diagnostic only. Treats each top-level statement of
// the kernel body as one "construct" and counts operand edges that cross construct boundaries -- i.e. a statement in
// construct B whose operand is defined in construct A. These edges are the data-flow that a per-construct split would
// have to duplicate (const/arg sources are trivially recomputable; alloca/other sources are real coupling). Reports
// construct counts by kind and the largest construct, so we can size the "changed construct" recompile cost.
void run_construct_census(IRNode *ir, const std::string &kname, const char *where) {
  auto *block = ir->cast<Block>();
  if (block == nullptr) {
    QD_INFO("[construct-census] kernel={} where={} SKIP(root-not-block)", kname, where);
    return;
  }
  const int n = (int)block->statements.size();
  std::unordered_map<int, int> owner;  // instance_id -> owning top-level construct index
  std::vector<std::unordered_set<Stmt *>> members(n);
  for (int c = 0; c < n; c++) {
    Stmt *root = block->statements[c].get();
    owner[root->instance_id] = c;
    members[c].insert(root);
    auto sub = irpass::analysis::gather_statements(root, [](Stmt *) { return true; });
    for (auto *s : sub) {
      owner[s->instance_id] = c;
      members[c].insert(s);
    }
  }
  auto kind_of = [](Stmt *r) -> const char * {
    if (r->is<RangeForStmt>())
      return "RangeFor";
    if (r->is<StructForStmt>())
      return "StructFor";
    if (r->is<WhileStmt>())
      return "While";
    if (r->is<IfStmt>())
      return "If";
    if (r->is<AllocaStmt>())
      return "Alloca";
    if (r->is<ConstStmt>())
      return "Const";
    return typeid(*r).name();
  };
  // count inner loops nested inside a construct (to tell "one big loop" from "a loop nest region").
  auto inner_loops = [](const std::unordered_set<Stmt *> &mem) {
    int k = 0;
    for (auto *s : mem)
      if (s->is<RangeForStmt>() || s->is<StructForStmt>() || s->is<WhileStmt>())
        k++;
    return k;
  };
  int loops = 0, ifs = 0, allocas = 0, consts = 0, other = 0;
  size_t max_members = 0;
  std::vector<std::pair<size_t, int>> sizes;  // (member_count, construct index)
  for (int c = 0; c < n; c++) {
    Stmt *r = block->statements[c].get();
    if (r->is<RangeForStmt>() || r->is<StructForStmt>() || r->is<WhileStmt>())
      loops++;
    else if (r->is<IfStmt>())
      ifs++;
    else if (r->is<AllocaStmt>())
      allocas++;
    else if (r->is<ConstStmt>())
      consts++;
    else
      other++;
    max_members = std::max(max_members, members[c].size());
    sizes.emplace_back(members[c].size(), c);
  }
  std::sort(sizes.begin(), sizes.end(), std::greater<>());
  std::string top;
  for (size_t i = 0; i < std::min((size_t)3, sizes.size()); i++) {
    int c = sizes[i].second;
    top += "[#" + std::to_string(c) + " kind=" + kind_of(block->statements[c].get()) +
           " stmts=" + std::to_string(sizes[i].first) + " inner_loops=" + std::to_string(inner_loops(members[c])) +
           "] ";
  }
  long long cross_edges = 0, src_const = 0, src_alloca = 0, src_arg = 0, src_other = 0;
  std::set<std::pair<int, int>> cross_pairs;
  std::map<std::string, int> nontrivial_src_types;  // type-name histogram of non-const/non-arg cross-edge sources
  for (int c = 0; c < n; c++) {
    for (auto *s : members[c]) {
      for (auto *op : s->get_operands()) {
        if (op == nullptr)
          continue;
        auto it = owner.find(op->instance_id);
        if (it == owner.end() || it->second == c)
          continue;
        cross_edges++;
        cross_pairs.insert({it->second, c});
        if (op->is<ConstStmt>())
          src_const++;
        else if (op->is<AllocaStmt>())
          src_alloca++;
        else if (op->is<ArgLoadStmt>())
          src_arg++;
        else {
          src_other++;
          nontrivial_src_types[typeid(*op).name()]++;
        }
      }
    }
  }
  std::vector<std::pair<int, std::string>> nt;
  for (auto &kv : nontrivial_src_types)
    nt.emplace_back(kv.second, kv.first);
  std::sort(nt.begin(), nt.end(), std::greater<>());
  std::string nt_str;
  for (size_t i = 0; i < std::min((size_t)6, nt.size()); i++)
    nt_str += nt[i].second + "=" + std::to_string(nt[i].first) + " ";
  QD_INFO(
      "[construct-census] kernel={} where={} n_constructs={} (loops={} ifs={} allocas={} consts={} other={}) "
      "max_construct_stmts={} | cross_operand_edges={} cross_pairs={} | cross_src: const={} alloca={} arg={} "
      "nontrivial={} | top3: {}| nontrivial_src_types: {}",
      kname, where, n, loops, ifs, allocas, consts, other, (long long)max_members, cross_edges,
      (long long)cross_pairs.size(), src_const, src_alloca, src_arg, src_other, top, nt_str);
}

// S0b per-construct cost probe (env QD_CONSTRUCT_TIMING=1). Diagnostic only, does NOT mutate the real IR: for each
// top-level construct it clones the construct into a throwaway wrapper block, runs merge_global_ptrs (the dominant
// pre-offload pass, ~42% of cold compile) on the clone, and times it. Sum over constructs approximates the whole-
// kernel merge_global_ptrs cost minus cross-construct merges; the per-construct max/min bound the "edit one loop"
// recompile cost for this pass. Meant to be placed right before the real merge_global_ptrs call so the IR state
// (post simplify_I/II) matches what the real pass sees.
void run_construct_timing(IRNode *ir, const std::string &kname, const char *where) {
  auto *block = ir->cast<Block>();
  if (block == nullptr) {
    QD_INFO("[construct-timing] kernel={} where={} SKIP(root-not-block)", kname, where);
    return;
  }
  const int n = (int)block->statements.size();
  Callable *callable = block->parent_callable();
  std::vector<double> ms(n, 0.0);
  double total = 0.0;
  long long modified_constructs = 0, total_stmts = 0;
  for (int c = 0; c < n; c++) {
    auto clone = irpass::analysis::clone(block->statements[c].get());
    Block wrapper;
    wrapper.set_parent_callable(callable);
    wrapper.insert(std::move(clone));
    auto sub = irpass::analysis::gather_statements(&wrapper, [](Stmt *) { return true; });
    total_stmts += (long long)sub.size();
    auto t0 = std::chrono::steady_clock::now();
    bool modified = irpass::merge_global_ptrs(&wrapper);
    auto t1 = std::chrono::steady_clock::now();
    if (modified)
      modified_constructs++;
    ms[c] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total += ms[c];
  }
  std::vector<std::pair<double, int>> order;
  for (int c = 0; c < n; c++)
    order.emplace_back(ms[c], c);
  std::sort(order.begin(), order.end(), std::greater<>());
  int under_100 = 0, under_500 = 0, under_2000 = 0;
  for (double v : ms) {
    if (v < 100.0)
      under_100++;
    if (v < 500.0)
      under_500++;
    if (v < 2000.0)
      under_2000++;
  }
  std::string top;
  for (size_t i = 0; i < std::min((size_t)5, order.size()); i++)
    top += "[#" + std::to_string(order[i].second) + " " + std::to_string((long long)order[i].first) + "ms] ";
  QD_INFO(
      "[construct-timing] kernel={} where={} n={} total_stmts={} modified_constructs={} mgp_sum={:.0f}ms "
      "mgp_max={:.0f}ms | constructs_under: 100ms={} 500ms={} 2000ms={} | top5: {}",
      kname, where, n, total_stmts, modified_constructs, total, order.empty() ? 0.0 : order[0].first, under_100,
      under_500, under_2000, top);
}

// S2a post-offload GlobalTemporary cross-construct census (env QD_CONSTRUCT_GT_CENSUS=1). Diagnostic only. This is the
// catch-1 feasibility gate for the per-construct split: after offload, all cross-*task* data flow is carried through
// global-temp slots (offload::PromoteIntermediateToGlobalTmp). For each global-temp offset it records which tasks
// write vs read it, maps every task back to its originating top-level construct via the pre-offload owner map
// (instance_id -> construct), and unions constructs joined by a shared slot. If the union-find collapses the kernel
// into a few giant "super-constructs", per-construct caching wins evaporate; if groups stay small/many, the split is
// viable. `owner`/`n_constructs` are captured on the flat top-level block just before offload.
void run_offload_gt_census(IRNode *ir, const std::unordered_map<int, int> &owner, int n_constructs,
                           const std::vector<char> &c_is_loop, const std::vector<int> &c_stmts,
                           const std::string &kname) {
  auto *root = ir->cast<Block>();
  if (root == nullptr) {
    QD_INFO("[gt-census] kernel={} SKIP(root-not-block)", kname);
    return;
  }
  std::vector<OffloadedStmt *> tasks;
  for (auto &s : root->statements)
    if (auto *o = s->cast<OffloadedStmt>())
      tasks.push_back(o);
  const int T = (int)tasks.size();

  // task -> originating construct: majority vote over body stmts carrying a known pre-offload owner. -1 => no owned
  // stmt (a pure promotion prologue / runtime-synthesized task, e.g. a serial slot-init).
  std::vector<int> task_c(T, -1);
  int prologue_tasks = 0;
  std::vector<std::vector<Stmt *>> body(T);
  for (int t = 0; t < T; t++) {
    body[t] = irpass::analysis::gather_statements(tasks[t], [](Stmt *) { return true; });
    std::map<int, int> votes;
    for (auto *s : body[t]) {
      auto it = owner.find(s->instance_id);
      if (it != owner.end())
        votes[it->second]++;
    }
    int best = -1, bestv = 0;
    for (auto &kv : votes)
      if (kv.second > bestv) {
        bestv = kv.second;
        best = kv.first;
      }
    task_c[t] = best;
    if (best < 0)
      prologue_tasks++;
  }

  // global-temp offset -> writer tasks / reader tasks (GlobalStore=write, GlobalLoad=read, AtomicOp=both), plus the
  // slot's data type and the set of access kinds so we can characterize each hub concretely (reduction vs flag vs bound).
  std::map<std::size_t, std::set<int>> writers, readers;
  std::map<std::size_t, std::string> off_dtype;
  std::map<std::size_t, std::set<std::string>> off_access;
  auto note = [&](Stmt *ptr, int t, bool is_write, bool is_read, const std::string &access) {
    if (auto *g = ptr->cast<GlobalTemporaryStmt>()) {
      if (is_write)
        writers[g->offset].insert(t);
      if (is_read)
        readers[g->offset].insert(t);
      off_dtype[g->offset] = g->ret_type.to_string();
      off_access[g->offset].insert(access);
    }
  };
  for (int t = 0; t < T; t++)
    for (auto *s : body[t]) {
      if (auto *st = s->cast<GlobalStoreStmt>())
        note(st->dest, t, true, false, "store");
      else if (auto *ld = s->cast<GlobalLoadStmt>())
        note(ld->src, t, false, true, "load");
      else if (auto *at = s->cast<AtomicOpStmt>())
        note(at->dest, t, true, true, "atomic:" + atomic_op_type_name(at->op_type));
    }

  // union-find over constructs joined by a shared global-temp offset (cross-construct edge).
  std::vector<int> uf(std::max(n_constructs, 0));
  for (int i = 0; i < n_constructs; i++)
    uf[i] = i;
  std::function<int(int)> find = [&](int x) { return uf[x] == x ? x : uf[x] = find(uf[x]); };

  std::set<std::size_t> all_offsets;
  for (auto &kv : writers)
    all_offsets.insert(kv.first);
  for (auto &kv : readers)
    all_offsets.insert(kv.first);
  // Full sorted offset list + a checksum, to verify the offload offset assignment is deterministic across runs and to
  // compare the default (traversal-order) vs QD_STABLE_GTMP=1 (content-keyed) policies (S2a').
  {
    std::string offs;
    std::uint64_t chk = 1469598103934665603ULL;
    for (auto o : all_offsets) {
      offs += std::to_string(o) + ",";
      chk = (chk ^ (std::uint64_t)o) * 1099511628211ULL;
    }
    QD_INFO("[gtmp-map] kernel={} n_offsets={} checksum={} offsets=[{}]", kname, (int)all_offsets.size(), chk, offs);
  }
  long long cross_task = 0, cross_construct = 0, prologue_sourced = 0;
  std::vector<std::pair<int, std::size_t>> fanout;  // (#distinct constructs, offset) for each cross-construct slot
  for (auto off : all_offsets) {
    std::set<int> ts(writers[off].begin(), writers[off].end());
    ts.insert(readers[off].begin(), readers[off].end());
    if (ts.size() > 1)
      cross_task++;
    std::set<int> cs;
    bool has_prologue = false;
    for (int t : ts) {
      if (task_c[t] < 0)
        has_prologue = true;
      else
        cs.insert(task_c[t]);
    }
    if (has_prologue)
      prologue_sourced++;
    if (cs.size() > 1) {
      cross_construct++;
      fanout.emplace_back((int)cs.size(), off);
      int base = *cs.begin();
      for (int c : cs)
        uf[find(base)] = find(c);
    }
  }
  std::sort(fanout.begin(), fanout.end(), std::greater<>());
  int fanout_gt2 = 0;
  for (auto &f : fanout)
    if (f.first > 2)
      fanout_gt2++;
  std::string ftop;
  for (size_t i = 0; i < std::min((size_t)6, fanout.size()); i++)
    ftop += std::to_string(fanout[i].first) + " ";
  // Concrete identity of the top hub slots: offset, data type, access kinds, #writer/#reader tasks, and a
  // classification of the *producer* (the value stored into the slot). S2a' decision input: walk the operand cone of
  // the writer's stored value. If the cone bottoms out only in ArgLoad/Const/ExternalPtr (+ optional memory
  // GlobalLoads) and references NO other global-temp slot, the hub is independently recomputable inside each
  // construct (catch-2 dissolves it, no stable-offset ABI needed). A cone that reads another temp slot is a genuine
  // cross-construct chain that needs a stable offset.
  for (size_t i = 0; i < std::min((size_t)8, fanout.size()); i++) {
    auto off = fanout[i].second;
    std::string acc;
    for (auto &a : off_access[off])
      acc += a + ",";
    // find the (single) writer's stored value
    Stmt *producer = nullptr;
    for (int t : writers[off])
      for (auto *s : body[t])
        if (auto *st = s->cast<GlobalStoreStmt>())
          if (auto *g = st->dest->cast<GlobalTemporaryStmt>())
            if (g->offset == off)
              producer = st->val;
    int n_arg = 0, n_const = 0, n_extptr = 0, n_gload = 0, n_other_gtmp = 0, n_nodes = 0;
    if (producer) {
      std::set<Stmt *> seen;
      std::vector<Stmt *> stk{producer};
      while (!stk.empty()) {
        Stmt *s = stk.back();
        stk.pop_back();
        if (s == nullptr || seen.count(s))
          continue;
        seen.insert(s);
        n_nodes++;
        if (s->is<ArgLoadStmt>())
          n_arg++;
        else if (s->is<ConstStmt>())
          n_const++;
        else if (s->is<ExternalPtrStmt>())
          n_extptr++;
        else if (s->is<GlobalLoadStmt>())
          n_gload++;
        else if (auto *g = s->cast<GlobalTemporaryStmt>()) {
          if (g->offset != off)
            n_other_gtmp++;
        }
        for (auto *o : s->get_operands())
          stk.push_back(o);
      }
    }
    QD_INFO(
        "[gt-census]   HUB off={} dtype={} fanout={} writers={} readers={} access={} | producer_cone: nodes={} arg={} "
        "const={} extptr={} memload={} other_temp={} => {}",
        off, off_dtype.count(off) ? off_dtype[off] : "?", fanout[i].first, (int)writers[off].size(),
        (int)readers[off].size(), acc, n_nodes, n_arg, n_const, n_extptr, n_gload, n_other_gtmp,
        (producer == nullptr ? "NO-STORE" : (n_other_gtmp == 0 ? "RECOMPUTABLE" : "CHAINED")));
  }

  // Per-group stats, weighting by real recompile cost: count loop-constructs and sum their statements per group. The
  // group with the most loops (and the one with the most statements) bounds the worst-case single-edit recompile.
  std::map<int, int> group_size, group_loops, group_stmts;
  for (int i = 0; i < n_constructs; i++) {
    int r = find(i);
    group_size[r]++;
    if (!c_is_loop.empty() && c_is_loop[i])
      group_loops[r]++;
    if (!c_stmts.empty())
      group_stmts[r] += c_stmts[i];
  }
  int largest = 0, grouped_constructs = 0, total_loops = 0;
  for (auto &kv : group_size) {
    largest = std::max(largest, kv.second);
    if (kv.second > 1)
      grouped_constructs += kv.second;
  }
  for (int i = 0; i < n_constructs; i++)
    if (!c_is_loop.empty() && c_is_loop[i])
      total_loops++;
  int max_group_loops = 0, loops_in_multi = 0, max_group_stmts = 0;
  for (auto &kv : group_loops) {
    max_group_loops = std::max(max_group_loops, kv.second);
    if (group_size[kv.first] > 1)
      loops_in_multi += kv.second;
  }
  for (auto &kv : group_stmts)
    if (group_size[kv.first] > 1)
      max_group_stmts = std::max(max_group_stmts, kv.second);
  QD_INFO(
      "[gt-census] kernel={} n_constructs={} (loops={}) n_tasks={} prologue_tasks={} | gt_offsets={} cross_task={} "
      "cross_construct={} prologue_sourced={} | super_groups={} largest_group={} constructs_in_multi_groups={} | "
      "loops_in_multi_groups={} max_loops_in_one_group={} max_stmts_in_multi_group={} | "
      "cross_offset_fanout(>2 constructs)={} top_fanouts=[{}]",
      kname, n_constructs, total_loops, T, prologue_tasks, (long long)all_offsets.size(), cross_task, cross_construct,
      prologue_sourced, (int)group_size.size(), largest, grouped_constructs, loops_in_multi, max_group_loops,
      max_group_stmts, fanout_gt2, ftop);
}
// S2b split-offload prototype (env QD_SPLIT_OFFLOAD=1). Instead of one whole-kernel offload, compile each top-level
// construct independently and reassemble the tasks, to validate the split + relink path (correctness only; caching and
// per-construct GT-offset coordination are separate). For each container / side-effecting top-level construct: clone
// the whole block, drop the *other* constructs (containers / side-effecting stmts), DIE the now-unused shared scalar
// defs (catch-2: recompute per construct), run the normal `offload` on the isolated construct, and collect its
// OffloadedStmts. Cross-construct memory ordering is preserved by keeping tasks in original construct order; pure value
// coupling is preserved by duplicating the defs into each clone. (Cross-construct global-temp slots are NOT yet
// coordinated here -- kernels that need them require S2a' shared offsets; this prototype targets constructs whose
// offload allocates only construct-private / no global-temp slots.)
void split_offload_per_construct(IRNode *ir, const CompileConfig &config) {
  auto *block = ir->cast<Block>();
  QD_ASSERT(block != nullptr);
  const int n = (int)block->statements.size();
  std::vector<std::unique_ptr<Stmt>> tasks;
  for (int i = 0; i < n; i++) {
    Stmt *target = block->statements[i].get();
    if (!target->is_container_statement() && !target->has_global_side_effect())
      continue;  // pure value defs produce no task; they are duplicated into each consuming construct below
    auto cloned = irpass::analysis::clone(block);
    auto *cb = cloned->cast<Block>();
    cb->set_parent_callable(block->parent_callable());
    Stmt *ctarget = cb->statements[i].get();
    std::vector<Stmt *> to_remove;
    for (auto &s : cb->statements) {
      Stmt *sj = s.get();
      if (sj != ctarget && (sj->is_container_statement() || sj->has_global_side_effect()))
        to_remove.push_back(sj);
    }
    for (Stmt *sj : to_remove)
      cb->extract(sj);
    irpass::die(cb);
    irpass::offload(cb, config);
    while (!cb->statements.empty())
      tasks.push_back(cb->extract(0));
  }
  int n_constructs = 0;
  for (int i = 0; i < n; i++) {
    Stmt *s = block->statements[i].get();
    if (s->is_container_statement() || s->has_global_side_effect())
      n_constructs++;
  }
  while (!block->statements.empty())
    block->extract(0);
  for (auto &t : tasks)
    block->insert(std::move(t));
  static const bool split_log = []() {
    const char *e = std::getenv("QD_SPLIT_LOG");
    return e != nullptr && std::string(e) == "1";
  }();
  if (split_log)
    QD_INFO("[split-offload] top_level_stmts={} constructs_compiled={} tasks_produced={}", n, n_constructs,
            (int)block->statements.size());
}

// S2 step A (design §9.A): run the whole pre-offload + offload frontend on ONE isolated top-level construct instead of
// once over the whole kernel. Mirrors the kNone / non-mesh portion of the whole-kernel sequence in
// `compile_to_offloads` between simplify_I and simplify_III, in the same order, so a per-construct compile produces the
// same tasks the whole-kernel path would for that construct. `cb` is a construct already isolated + recomputed (other
// constructs dropped, shared defs DIE'd) by `split_frontend_per_construct`.
static bool split_trace() {
  static const bool e = []() {
    const char *v = std::getenv("QD_SPLIT_TRACE");
    return v != nullptr && std::string(v) == "1";
  }();
  return e;
}

void run_construct_frontend(IRNode *cb, const CompileConfig &config, const Kernel *kernel, bool verbose) {
  const std::string &name = kernel->get_name();
#define QD_SPLIT_STEP(msg)                                        \
  do {                                                            \
    if (split_trace())                                            \
      QD_INFO("[split-trace] {}: {}", name, msg);                 \
  } while (0)
  QD_SPLIT_STEP("simplify_I begin");
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_I"});
  QD_SPLIT_STEP("handle_external_ptr_boundary");
  irpass::handle_external_ptr_boundary(cb, config);
  if (config.check_out_of_bound) {
    QD_SPLIT_STEP("check_out_of_bound");
    irpass::check_out_of_bound(cb, config, {name});
  }
  QD_SPLIT_STEP("merge_global_ptrs");
  irpass::merge_global_ptrs(cb);
  QD_SPLIT_STEP("flag_access.1");
  irpass::flag_access(cb);
  QD_SPLIT_STEP("simplify_II");
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_II"});
  QD_SPLIT_STEP("offload begin");
  irpass::offload(cb, config);
  QD_SPLIT_STEP("offload done");
  if (config.opt_level > 0) {
    QD_SPLIT_STEP("cse_offloaded_tasks");
    irpass::cse_offloaded_tasks(cb);
  }
  QD_SPLIT_STEP("flag_access.2");
  irpass::flag_access(cb);
  QD_SPLIT_STEP("simplify_III");
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_III"});
  QD_SPLIT_STEP("run_construct_frontend done");
#undef QD_SPLIT_STEP
}

bool block_has_mesh_for(Block *block) {
  if (block == nullptr)
    return false;
  auto sub = irpass::analysis::gather_statements(block, [](Stmt *s) { return s->is<MeshForStmt>(); });
  return !sub.empty();
}

// Resolve a local pointer (an AllocaStmt, or a MatrixPtrStmt into one) to its base AllocaStmt. Returns nullptr when the
// pointer is not alloca-based (e.g. a global pointer / global temporary), in which case it is not a local variable.
Stmt *resolve_local_alloca(Stmt *ptr) {
  while (ptr != nullptr) {
    if (ptr->is<AllocaStmt>())
      return ptr;
    if (auto *mp = ptr->cast<MatrixPtrStmt>()) {
      ptr = mp->origin;
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

// Find the top-level statement (direct child of `top`) that transitively contains `s`, and report via
// `*inside_container` whether the path from `s` up to `top` crosses any container statement (loop/while). Returns
// nullptr when `s` is not under `top`.
Stmt *top_level_owner(Stmt *s, Block *top, bool *inside_container) {
  *inside_container = false;
  Stmt *cur = s;
  Block *b = s->parent;
  while (b != nullptr && b != top) {
    Stmt *owner = b->parent_stmt();
    if (owner == nullptr)
      return nullptr;
    if (owner->is_container_statement())
      *inside_container = true;
    cur = owner;
    b = owner->parent;
  }
  return (b == top) ? cur : nullptr;
}

// Split-safety gate for the per-construct frontend split (design §7.3 catch-1/catch-2, lines 155-169). A kernel is NOT
// safe to split when a top-level local variable's value is *produced inside one construct* (written by a store/atomic
// nested inside a top-level container / loop) and *consumed by a different construct* (read outside that producing
// loop). Whole-kernel offload carries such values across tasks via a cross-construct GlobalTemporary; isolating the
// consuming construct would drop the producing loop, so the value cannot be reconstructed (it is not recomputable from
// pure top-level defs). These are the "mutable shared allocas" the S0 census measured at zero in the target workload,
// so falling back to the whole-kernel path here costs ~nothing on the target while keeping correctness for the general
// case (recomputable cross-construct SSA — consts/args/field-loads/top-level pure defs — is fine: clone+die duplicates
// it into each construct).
bool split_is_recompute_safe(Block *block) {
  if (block == nullptr)
    return false;
  auto accesses = irpass::analysis::gather_statements(block, [](Stmt *s) {
    return s->is<LocalLoadStmt>() || s->is<LocalStoreStmt>() || s->is<AtomicOpStmt>();
  });
  std::unordered_map<Stmt *, std::set<Stmt *>> read_owners;        // alloca -> top-level constructs that read it
  std::unordered_map<Stmt *, std::set<Stmt *>> loop_write_owners;  // alloca -> constructs that write it inside a loop
  for (Stmt *acc : accesses) {
    Stmt *read_ptr = nullptr;
    Stmt *write_ptr = nullptr;
    if (auto *ld = acc->cast<LocalLoadStmt>()) {
      read_ptr = ld->src;
    } else if (auto *st = acc->cast<LocalStoreStmt>()) {
      write_ptr = st->dest;
    } else if (auto *at = acc->cast<AtomicOpStmt>()) {
      read_ptr = at->dest;  // atomic read-modify-write counts as both a read and a write of dest
      write_ptr = at->dest;
    }
    bool inside_container = false;
    Stmt *owner = top_level_owner(acc, block, &inside_container);
    if (owner == nullptr)
      continue;
    if (Stmt *a = resolve_local_alloca(read_ptr))
      read_owners[a].insert(owner);
    if (Stmt *a = resolve_local_alloca(write_ptr)) {
      if (inside_container)
        loop_write_owners[a].insert(owner);
    }
  }
  for (auto &kv : loop_write_owners) {
    auto rit = read_owners.find(kv.first);
    if (rit == read_owners.end())
      continue;
    for (Stmt *r : rit->second) {
      if (kv.second.find(r) == kv.second.end())
        return false;  // a loop-produced local is read outside its producing loop -> cross-construct, not recomputable
    }
  }
  return true;
}

// Whether a top-level statement is a *real* observable effect that forces its segment to become a task. Pointer/address
// computations (GlobalPtr/ExternalPtr/MatrixPtr) report has_global_side_effect()==true because of sparse activation,
// but they are not standalone effects — they are the address half of a load/store and get recomputed into whichever
// construct consumes them (e.g. a dynamic loop bound `range(lo[None], hi[None])` lowers to GlobalPtr+GlobalLoad in a
// serial segment that must NOT become its own task; it is recomputed into the loop construct via the backward slice).
bool stmt_is_task_effect(Stmt *s) {
  if (s->is<GlobalPtrStmt>() || s->is<ExternalPtrStmt>() || s->is<MatrixPtrStmt>())
    return false;
  return s->has_global_side_effect();
}

// S2 step A driver (env QD_SPLIT_FRONTEND=1): split the flat top-level block into constructs right after lower_ast +
// the structural prefix, run the full per-construct frontend on each (with recompute of shared top-level defs,
// catch-2), and reassemble the produced OffloadedStmts in source order. This is the correctness backbone for the
// per-construct frontend cache (§9): each construct compiles independently so its simplify/mgp buckets are tiny (S0b)
// and its output can later be keyed + cached + skipped (§9.B/C). Correctness for recompute-safe kernels rests on S2b:
// cross-construct global-temp hubs dissolve via recompute, and cross-construct memory ordering is preserved by keeping
// tasks in original construct order. Restricted to autodiff_mode==kNone / non-mesh / recompute-safe kernels by the
// caller; anything else falls back to the whole-kernel path.
void split_frontend_per_construct(IRNode *ir, const CompileConfig &config, const Kernel *kernel, bool verbose) {
  auto *block = ir->cast<Block>();
  QD_ASSERT(block != nullptr);
  const int n = (int)block->statements.size();
  if (split_trace())
    QD_INFO("[split-trace] {}: split_frontend_per_construct begin, top_level_stmts={}", kernel->get_name(), n);

  // Segment the top-level block exactly the way `offload` will chunk it into tasks: every container statement
  // (RangeFor/StructFor/While/...) is its own segment, and every maximal run of consecutive non-container (serial)
  // statements is one serial segment. A construct = a segment that emits a task: containers always do; a serial run
  // does iff it contains a real effect (stmt_is_task_effect). A serial run of only pure value defs / pointer chains
  // (e.g. dynamic loop bounds) emits no task and is recomputed into whichever construct consumes it.
  std::vector<int> seg_id(n, -1);
  std::vector<bool> seg_emits_task;  // per segment
  int cur_seg = -1;
  bool in_serial_run = false;
  for (int j = 0; j < n; j++) {
    Stmt *s = block->statements[j].get();
    if (s->is_container_statement()) {
      cur_seg = (int)seg_emits_task.size();
      seg_emits_task.push_back(true);
      in_serial_run = false;
    } else {
      if (!in_serial_run) {
        cur_seg = (int)seg_emits_task.size();
        seg_emits_task.push_back(false);
        in_serial_run = true;
      }
      if (stmt_is_task_effect(s))
        seg_emits_task[cur_seg] = true;
    }
    seg_id[j] = cur_seg;
  }
  const int n_segs = (int)seg_emits_task.size();

  // S2 step C (§9.C): program-scoped per-construct FRONTEND cache. On a warm compile only the changed construct's
  // frontend re-runs; unchanged constructs are cloned out of the cache, skipping simplify/mgp/offload. Content-keyed,
  // so identical constructs across kernels (same ABI/config) share entries. Default on whenever the split runs;
  // QD_CONSTRUCT_CACHE=0 disables it for A/B equivalence checks.
  static const bool construct_cache_on = []() {
    const char *e = std::getenv("QD_CONSTRUCT_CACHE");
    return e == nullptr || std::string(e) != "0";
  }();
  PerConstructCache *cc =
      (construct_cache_on && kernel->program != nullptr) ? &kernel->program->per_construct_cache() : nullptr;

  std::vector<std::unique_ptr<Stmt>> tasks;
  int n_constructs = 0, n_hit = 0, n_recompiled = 0;
  double key_ms = 0.0, fe_ms = 0.0;  // diagnostic: time in construct-key vs the (cached-out) frontend
  auto _now = []() { return std::chrono::steady_clock::now(); };
  auto _ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  for (int k = 0; k < n_segs; k++) {
    if (!seg_emits_task[k])
      continue;  // pure-def-only serial run: no standalone task, recomputed into consuming constructs below
    n_constructs++;
    auto cloned = irpass::analysis::clone(block);
    auto *cb = cloned->cast<Block>();
    cb->set_parent_callable(block->parent_callable());
    // Isolate construct k by its BACKWARD SLICE: keep segment k's whole subtree plus the transitive operand-def chain
    // it reads (const/arg/binop/pointer/field-load chains from earlier segments — recomputed into this construct), and
    // drop every other top-level statement (other constructs' loops and stores). The slice is closed under operands, so
    // no kept statement can reference a dropped one. This both keeps a store together with its own pointer operand and
    // recomputes cross-construct recomputable values (e.g. dynamic loop bounds), without the has_global_side_effect
    // heuristic that mis-stripped pointer chains.
    std::unordered_set<Stmt *> needed;
    std::vector<Stmt *> worklist;
    for (int j = 0; j < n; j++) {
      if (seg_id[j] != k)
        continue;
      Stmt *tlj = cb->statements[j].get();
      if (needed.insert(tlj).second)
        worklist.push_back(tlj);
      for (Stmt *sub : irpass::analysis::gather_statements(tlj, [](Stmt *) { return true; }))
        if (needed.insert(sub).second)
          worklist.push_back(sub);
    }
    while (!worklist.empty()) {
      Stmt *s = worklist.back();
      worklist.pop_back();
      for (Stmt *op : s->get_operands())
        if (op != nullptr && needed.insert(op).second)
          worklist.push_back(op);
    }
    std::vector<Stmt *> to_remove;
    for (int j = 0; j < n; j++) {
      Stmt *tlj = cb->statements[j].get();
      if (needed.find(tlj) == needed.end())
        to_remove.push_back(tlj);
    }
    for (Stmt *sj : to_remove)
      cb->extract(sj);
    if (split_trace())
      QD_INFO("[split-trace] {}: construct #{} (seg {}) pre-die stmts={}", kernel->get_name(), n_constructs, k,
              (int)cb->statements.size());
    irpass::die(cb);  // clean up anything left dead after slicing (recompute per construct)
    if (split_trace())
      QD_INFO("[split-trace] {}: construct #{} (seg {}) post-die stmts={}", kernel->get_name(), n_constructs, k,
              (int)cb->statements.size());

    // Key the isolated, re-id'd construct and consult the cache. Key + clone are cheap; the expensive
    // run_construct_frontend on a miss runs OUTSIDE the cache lock so parallel kernel compiles never serialize on it.
    bool hit = false;
    std::string ckey;
    if (cc != nullptr) {
      irpass::re_id(cb);
      auto _k0 = _now();
      ckey = get_hashed_per_construct_cache_key(config, cb, kernel);
      key_ms += _ms(_k0, _now());
      std::lock_guard<std::mutex> g(cc->mu);
      auto it = cc->entries.find(ckey);
      if (it != cc->entries.end()) {
        for (auto &t : it->second)
          tasks.push_back(irpass::analysis::clone(t.get()));
        hit = true;
      }
    }
    if (hit) {
      n_hit++;
      if (split_trace())
        QD_INFO("[split-trace] {}: construct #{} (seg {}) cache HIT ckey={}", kernel->get_name(), n_constructs, k,
                ckey);
      continue;
    }

    auto _f0 = _now();
    run_construct_frontend(cb, config, kernel, verbose);
    fe_ms += _ms(_f0, _now());
    n_recompiled++;
    std::vector<std::unique_ptr<Stmt>> produced;
    while (!cb->statements.empty())
      produced.push_back(cb->extract(0));
    if (cc != nullptr) {
      std::vector<std::unique_ptr<Stmt>> stored;
      stored.reserve(produced.size());
      for (auto &t : produced)
        stored.push_back(irpass::analysis::clone(t.get()));
      std::lock_guard<std::mutex> g(cc->mu);
      cc->entries.emplace(ckey, std::move(stored));  // emplace: a racing store of the same key wins harmlessly
    }
    for (auto &t : produced)
      tasks.push_back(std::move(t));
  }
  while (!block->statements.empty())
    block->extract(0);
  for (auto &t : tasks)
    block->insert(std::move(t));
  if (cc != nullptr) {
    std::lock_guard<std::mutex> g(cc->mu);
    cc->last_stats[kernel->get_name()] = {n_constructs, n_hit, n_recompiled};
  }
  static const bool split_log = []() {
    const char *e = std::getenv("QD_SPLIT_LOG");
    return e != nullptr && std::string(e) == "1";
  }();
  if (split_log)
    QD_INFO(
        "[split-frontend] kernel={} top_level_stmts={} constructs_compiled={} constructs_cache_hit={} "
        "constructs_recompiled={} tasks_produced={} key_ms={:.1f} frontend_ms={:.1f}",
        kernel->get_name(), n, n_constructs, n_hit, n_recompiled, (int)block->statements.size(), key_ms, fe_ms);
}
}  // namespace

namespace irpass {

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast) {
  QD_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, kernel->get_name(), ir);
  print("Initial IR");

  if (!verbose && config.print_preprocessed_ir && start_from_ast) {
    QD_INFO("[{}] {}:", kernel->get_name(), "Preprocessed IR");
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  }

  if (autodiff_mode == AutodiffMode::kReverse) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  const char *dump_ir_env = std::getenv(DUMP_IR_ENV.data());
  std::filesystem::path ir_dump_dir = config.debug_dump_path;
  bool should_dump = (dump_ir_env != nullptr && std::string(dump_ir_env) == "1");

  auto dump_ir = [&](const std::string &stage_name) {
    if (!should_dump)
      return;
    std::filesystem::create_directories(ir_dump_dir);
    std::filesystem::path filename = ir_dump_dir / (kernel->name + "_" + stage_name + ".ll");
    std::string ir_str;
    irpass::print(ir, &ir_str);
    std::ofstream ofs(filename.string());
    if (ofs.good()) {
      ofs << ir_str;
    }
  };

  dump_ir("from_ast");

  if (start_from_ast) {
    irpass::frontend_type_check(ir);
    irpass::lower_ast(ir);
  }

  static const bool construct_census = []() {
    const char *e = std::getenv("QD_CONSTRUCT_CENSUS");
    return e != nullptr && std::string(e) == "1";
  }();
  static const bool construct_timing = []() {
    const char *e = std::getenv("QD_CONSTRUCT_TIMING");
    return e != nullptr && std::string(e) == "1";
  }();
  static const bool mgp_per_construct = []() {
    const char *e = std::getenv("QD_MGP_PER_CONSTRUCT");
    return e != nullptr && std::string(e) == "1";
  }();
  static const bool split_frontend = []() {
    const char *e = std::getenv("QD_SPLIT_FRONTEND");
    return e != nullptr && std::string(e) == "1";
  }();
  if (construct_census)
    run_construct_census(ir, kernel->get_name(), "after_lower_ast");

  dump_ir("quadrants1");
  irpass::compile_quadrants_functions(ir, config, Function::IRStage::BeforeLowerAccess);
  irpass::analysis::gather_func_store_dests(ir);
  irpass::compile_quadrants_functions(ir, config, Function::IRStage::OptimizedIR);
  irpass::analysis::gather_func_store_dests(ir);

  irpass::eliminate_immutable_local_vars(ir);

  irpass::type_check(ir, config);
  irpass::analysis::verify_if_debug(ir, config);

  // TODO: strictly enforce bit vectorization for x86 cpu and CUDA now
  //       create a separate CompileConfig flag for the new pass
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda || config.arch == Arch::amdgpu) {
    irpass::bit_loop_vectorize(ir);
    irpass::type_check(ir, config);
    irpass::analysis::verify_if_debug(ir, config);
  }

  // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
  irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
    irpass::die(ir);
  }

  dump_ir("before_simplify_I");

  // S2 step A (design §9.A, env QD_SPLIT_FRONTEND=1): for recompute-safe kernels (forward-only, non-mesh), run the
  // remaining pre-offload + offload frontend PER top-level construct and reassemble, instead of once over the whole
  // kernel. The seam is here — right after lower_ast + the structural prefix (function inlining, matrix-ptr lowering,
  // bit-loop vectorize), before the expensive simplify/merge_global_ptrs/offload passes (§7.3). Anything not
  // recompute-safe (autodiff, mesh-for) falls through to the whole-kernel path below.
  if (split_frontend && autodiff_mode == AutodiffMode::kNone && !block_has_mesh_for(ir->cast<Block>()) &&
      split_is_recompute_safe(ir->cast<Block>())) {
    split_frontend_per_construct(ir, config, kernel, verbose);
    dump_ir("after_offload");
    return;
  }
  if (split_frontend && split_trace() && autodiff_mode == AutodiffMode::kNone &&
      !block_has_mesh_for(ir->cast<Block>()) && !split_is_recompute_safe(ir->cast<Block>()))
    QD_INFO("[split-trace] {}: NOT recompute-safe -> whole-kernel fallback", kernel->get_name());

  irpass::full_simplify(
      ir, config,
      {false, /*autodiff_enabled*/ autodiff_mode != AutodiffMode::kNone, kernel->get_name(), verbose, "simplify_I"});
  irpass::analysis::verify_if_debug(ir, config);
  dump_ir("after_simplify_I");

  irpass::handle_external_ptr_boundary(ir, config);

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::analysis::gather_meshfor_relation_types(ir);
  }

  if (config.debug && autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    // Check whether the kernel obeys the autodiff limitation e.g., gloabl data
    // access rule
    // This check should be performed in the forward kernel i.e., autodiff_mode
    // == AutodiffMode::kCheckAutodiffValid
    irpass::demote_atomics(ir, config);
    irpass::differentiation_validation_check(ir, config, kernel->get_name());
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (autodiff_mode == AutodiffMode::kReverse || autodiff_mode == AutodiffMode::kForward) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir, config);

    irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ true, kernel->get_name(), verbose, "pre_autodiff"});
    irpass::auto_diff(ir, config, autodiff_mode, ad_use_stack);
    // TODO: Be carefull with the full_simplify when do high-order autodiff
    irpass::full_simplify(ir, config,
                          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "post_autodiff"});
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir, config, {kernel->get_name()});
    irpass::analysis::verify_if_debug(ir, config);
  }

  // Merge a global's separate read/write GlobalPtrStmts (same address) into one shared, activate=true pointer BEFORE
  // this first flag_access, so flag_access cannot stamp a read-only (activate=false) copy that the CSE eliminability
  // rule then refuses to re-merge with the in-loop write. Without it, cache_loop_invariant_global_vars sees a split
  // read/write and cannot cache conditional/in-if stores -> the -88% solver break-flag bug + the lost duck_in_box
  // optimization. On main this fell out of whole_kernel_cse running inside every full_simplify fixpoint; per-task CSE
  // does no pre-offload whole-kernel CSE, so we do this one cheap, pointers-only pass here instead (arithmetic is
  // already canonical after simplify_I, so a single call is enough; running it in the fixpoint was a +12-22s
  // compile regression for no extra benefit).
  if (construct_timing)
    run_construct_timing(ir, kernel->get_name(), "pre_merge_global_ptrs");
  if (mgp_per_construct) {
    // Experiment (QD_MGP_PER_CONSTRUCT=1): run the same-address pointer merge scoped to each top-level construct
    // instead of once over the whole kernel. Extracting each construct into a standalone wrapper keeps the pass's
    // per-elimination replace/mark-undone walk inside the construct rather than rewalking the whole top-level block,
    // which S0b measured ~580x cheaper in sum (20ms vs 11.6s). Cross-construct same-address merges are dropped; the
    // load-bearing within-loop read/write pointer unification is per-construct and preserved. Extract+reinsert at the
    // same index keeps the vector stable.
    if (auto *b = ir->cast<Block>()) {
      for (int i = 0; i < (int)b->statements.size(); i++) {
        Stmt *construct = b->statements[i].get();
        Block wrapper;
        wrapper.set_parent_callable(b->parent_callable());
        wrapper.insert(b->extract(construct));
        irpass::merge_global_ptrs(&wrapper);
        b->insert(wrapper.extract(construct), i);
      }
    }
  } else {
    irpass::merge_global_ptrs(ir);
  }
  irpass::analysis::verify_if_debug(ir, config);

  irpass::flag_access(ir);
  irpass::analysis::verify_if_debug(ir, config);

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_II"});
  irpass::analysis::verify_if_debug(ir, config);

  if (construct_census)
    run_construct_census(ir, kernel->get_name(), "before_offload");

  // S2a catch-1 gate: capture the flat top-level construct ownership (instance_id -> construct) just before offload,
  // so the post-offload census can attribute each task/global-temp slot back to its originating construct.
  static const bool gt_census = []() {
    const char *e = std::getenv("QD_CONSTRUCT_GT_CENSUS");
    return e != nullptr && std::string(e) == "1";
  }();
  std::unordered_map<int, int> gt_owner;
  int gt_nconstructs = 0;
  std::vector<char> gt_is_loop;
  std::vector<int> gt_stmts;
  if (gt_census) {
    if (auto *b = ir->cast<Block>()) {
      gt_nconstructs = (int)b->statements.size();
      gt_is_loop.assign(gt_nconstructs, 0);
      gt_stmts.assign(gt_nconstructs, 0);
      for (int c = 0; c < gt_nconstructs; c++) {
        Stmt *rootc = b->statements[c].get();
        gt_owner[rootc->instance_id] = c;
        gt_is_loop[c] =
            (rootc->is<RangeForStmt>() || rootc->is<StructForStmt>() || rootc->is<WhileStmt>()) ? 1 : 0;
        auto sub = irpass::analysis::gather_statements(rootc, [](Stmt *) { return true; });
        gt_stmts[c] = (int)sub.size();
        for (auto *s : sub)
          gt_owner[s->instance_id] = c;
      }
    }
  }

  static const bool split_offload = []() {
    const char *e = std::getenv("QD_SPLIT_OFFLOAD");
    return e != nullptr && std::string(e) == "1";
  }();
  if (split_offload)
    split_offload_per_construct(ir, config);
  else
    irpass::offload(ir, config);
  irpass::analysis::verify_if_debug(ir, config);

  if (gt_census)
    run_offload_gt_census(ir, gt_owner, gt_nconstructs, gt_is_loop, gt_stmts, kernel->get_name());

  dump_ir("after_offload");

  // Full per-task CSE now, before flag_access #2 splits a global's read/write pointers by access flag and before
  // simplify_III's LICM hoists the read pointer out of the loop. This restores the pointer-unification that main
  // gets from whole_kernel_cse running inside the post-offload full_simplify (per-task CSE otherwise defers to the
  // codegen workers, which run after cache_loop_invariant_global_vars). Needed for ndarrays, which only become
  // ExternalPtrStmts during offload and so cannot be reached by the pre-offload merge_global_ptrs. See the pass.
  // Gated on opt_level like all other CSE (per_task_cse / upstream whole_kernel_cse): at opt_level 0 there is no CSE
  // to require pointer unification, matching upstream behaviour.
  if (config.opt_level > 0) {
    irpass::cse_offloaded_tasks(ir);
  }

  // NOTE: There was an additional CFG pass here, removed in
  // https://github.com/taichi-dev/taichi/pull/8691
  irpass::flag_access(ir);

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_III"});
  irpass::analysis::verify_if_debug(ir, config);

  dump_ir("after_simplify_III");

  // Run the adstack-size pre-pass here, before the per-task split in `KernelCodeGen::compile_kernel_to_module`
  // and before `make_cpu_multithreaded_range_for` in `offload_to_executable` rewrites user ranges into chunk
  // wrappers. The kernel IR still has every `OffloadedStmt` as a sibling in the top-level block, so the pre-
  // pass can resolve a `GlobalLoadStmt(GlobalTemporaryStmt)` source by walking across tasks: prep serial tasks
  // that store a dynamic range bound (e.g. `arr.shape[0]` lowered via `offload::PromoteIntermediateToGlobalTmp`)
  // are still visible alongside the consuming range-for task. Gated on the same reverse+ad_use_stack predicate
  // the per-task call used so compile behaviour is unchanged for forward-only kernels.
  if (autodiff_mode == AutodiffMode::kReverse && ad_use_stack) {
    irpass::determine_ad_stack_size(ir, config);
    print("Autodiff stack size determined");
  }
}

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  QD_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, kernel->get_name(), ir);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.

  auto amgr = std::make_unique<AnalysisManager>();

  print("Start offload_to_executable");
  irpass::analysis::verify_if_debug(ir, config);

  if (config.detect_read_only) {
    irpass::detect_read_only(ir);
    print("Detect read-only accesses");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted I");
  irpass::analysis::verify_if_debug(ir, config);

  if (config.cache_loop_invariant_global_vars) {
    irpass::cache_loop_invariant_global_vars(ir, config);
    print("Cache loop-invariant global vars");
  }

  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::type_check(ir, config);
    print("Dense struct-for demoted");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (config.make_cpu_multithreading_loop && arch_is_cpu(config.arch)) {
    irpass::make_cpu_multithreaded_range_for(ir, config);
    irpass::type_check(ir, config);
    print("Make CPU multithreaded range-for");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (is_extension_supported(config.arch, Extension::mesh) && config.demote_no_access_mesh_fors) {
    irpass::demote_no_access_mesh_fors(ir);
    irpass::type_check(ir, config);
    print("No-access mesh-for demoted");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (make_thread_local) {
    irpass::make_thread_local(ir, config);
    print("Make thread local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::make_mesh_thread_local(ir, config, {kernel->get_name()});
    print("Make mesh thread local");
    if (config.make_mesh_block_local && config.arch == Arch::cuda) {
      irpass::make_mesh_block_local(ir, config, {kernel->get_name()});
      print("Make mesh block local");
      irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_X"});
      print("Simplified X");
    }
  }

  if (make_block_local) {
    irpass::make_block_local(ir, config, {kernel->get_name(), verbose});
    print("Make block local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::demote_mesh_statements(ir, config, {kernel->get_name()});
    print("Demote mesh statements");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted II");
  irpass::analysis::verify_if_debug(ir, config);

  if (is_extension_supported(config.arch, Extension::quant) && config.quant_opt_atomic_demotion) {
    irpass::analysis::gather_uniquely_accessed_bit_structs(ir, amgr.get());
  }

  irpass::remove_range_assumption(ir);
  print("Remove range assumption");

  irpass::remove_loop_unique(ir);
  print("Remove loop_unique");
  irpass::analysis::verify_if_debug(ir, config);

  if (lower_global_access) {
    irpass::full_simplify(ir, config,
                          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "before_lower_access"});
    print("Simplified before lower access");
    irpass::lower_access(ir, config, {kernel->no_activate, true});
    print("Access lowered");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify_if_debug(ir, config);
  }

  irpass::demote_operations(ir, config);
  print("Operations demoted");

  irpass::full_simplify(ir, config,
                        {lower_global_access, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_IV"});
  print("Simplified IV");

  // `determine_ad_stack_size` used to run here, but the pre-pass needs the full kernel IR (all offloaded
  // tasks as siblings) so cross-task `GlobalTemporaryStmt` sources can be resolved. It now runs at the end
  // of `compile_to_offloads`, before the per-task split in `KernelCodeGen::compile_kernel_to_module`. The
  // `determine_ad_stack_size` parameter is kept in the signature for API stability but is no longer used.
  (void)determine_ad_stack_size;

  if (is_extension_supported(config.arch, Extension::quant)) {
    irpass::optimize_bit_struct_stores(ir, config, amgr.get());
    print("Bit struct stores optimized");
  }

  bool half2_optimization_enabled =
      (config.arch == Arch::cuda && config.half2_vectorization && !get_custom_cuda_library_path().empty());
  if (config.real_matrix_scalarize) {
    if (irpass::scalarize(ir, half2_optimization_enabled)) {
      irpass::die(ir);
      print("DIE");

      // Remove redundant MatrixInitStmt inserted during scalarization
      irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "scalarize"});
      print("Scalarized");
    }
  }

  // Final field registration correctness & type checking
  irpass::type_check(ir, config);
  irpass::analysis::verify_if_debug(ir, config);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local,
                           bool start_from_ast) {
  QD_AUTO_PROF;

  compile_to_offloads(ir, config, kernel, verbose, autodiff_mode, ad_use_stack, start_from_ast);

  offload_to_executable(ir, config, kernel, verbose,
                        /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse && ad_use_stack,
                        lower_global_access, make_thread_local, make_block_local);
}

void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      Function::IRStage target_stage) {
  QD_AUTO_PROF;

  auto current_stage = func->ir_stage();
  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, func->get_name(), ir);
  print("Initial IR");

  if (target_stage >= Function::IRStage::BeforeLowerAccess && current_stage < Function::IRStage::BeforeLowerAccess) {
    if (autodiff_mode == AutodiffMode::kReverse) {
      irpass::reverse_segments(ir);
      print("Segment reversed (for autodiff)");
    }

    if (current_stage < Function::IRStage::InitialIR) {
      irpass::frontend_type_check(ir);
      irpass::lower_ast(ir);
      print("Lowered");
    }

    // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
    irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);
    print("Matrix ptr lowered");

    irpass::demote_atomics(ir, config);
    print("Atomics demoted");
    irpass::associate_continue_scope(ir, config);
    print("Associated continue scope");
    func->set_ir_stage(Function::IRStage::BeforeLowerAccess);
  }

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
  }

  if (target_stage >= Function::IRStage::OptimizedIR && current_stage < Function::IRStage::OptimizedIR) {
    irpass::lower_access(ir, config, {{}, true});
    print("Access lowered");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::type_check(ir, config);
    print("Typechecked");

    irpass::demote_operations(ir, config);
    print("Operations demoted");

    if (config.real_matrix_scalarize) {
      if (irpass::scalarize(ir)) {
        // Remove redundant MatrixInitStmt inserted during scalarization
        irpass::die(ir);
        print("Scalarized");
      }
    }

    irpass::full_simplify(ir, config, {true, autodiff_mode != AutodiffMode::kNone, func->get_name(), verbose, "final"});
    print("Simplified");
    irpass::analysis::verify_if_debug(ir, config);
    func->set_ir_stage(Function::IRStage::OptimizedIR);
  }
}

}  // namespace irpass

}  // namespace quadrants::lang
