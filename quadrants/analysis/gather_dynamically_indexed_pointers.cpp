#include "quadrants/analysis/gather_uniquely_accessed_pointers.h"
#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/visitors.h"
#include <algorithm>

namespace quadrants::lang {

bool is_leaf_nodes_on_same_branch(SNode *snode0, SNode *snode1) {
  // Verify: place snode
  if (!snode0->is_place() || !snode1->is_place()) {
    return false;
  }

  // Check parent snode
  if (snode0->parent != snode1->parent) {
    return false;
  }

  return true;
}

static bool indices_have_const(const std::vector<Stmt *> &indices) {
  for (auto *index_stmt : indices) {
    if (index_stmt->is<ConstStmt>()) {
      return true;
    }
  }
  return false;
}

static bool indices_have_loop_index(const std::vector<Stmt *> &indices) {
  for (auto *index_stmt : indices) {
    if (index_stmt->is<LoopIndexStmt>()) {
      return true;
    }
  }
  return false;
}

class DynamicIndexingAnalyzer : public BasicStmtVisitor {
  void record_dynamic_indexed_ptr(ExternalPtrStmt *extern_ptr) {
    dynamically_indexed_ptrs_.insert(extern_ptr);
    // Find aliased ExternPtrStmt
    for (auto *other_extern_ptr : extern_ptrs_) {
      if (other_extern_ptr != extern_ptr && other_extern_ptr->base_ptr == extern_ptr->base_ptr) {
        // Aliased ExternalPtrStmt, with same base_ptr and outter index
        dynamically_indexed_ptrs_.insert(other_extern_ptr);
      }
    }
  }

  void record_dynamic_indexed_ptr(GlobalPtrStmt *global_ptr) {
    dynamically_indexed_ptrs_.insert(global_ptr);
    // Find aliased GlobalPtrStmt
    for (auto *other_global_ptr : global_ptrs_) {
      if (other_global_ptr != global_ptr && is_leaf_nodes_on_same_branch(other_global_ptr->snode, global_ptr->snode)) {
        dynamically_indexed_ptrs_.insert(other_global_ptr);
      }
    }
  }

  // A literal (const) index and a loop-variable index into the same memory can refer to the same element (the loop
  // variable may equal the constant), so caching either one to an independent local slot would drop the aliasing
  // store/load. The parallel-loop uniqueness analysis rejects such a pair on its own, but a serialized loop bypasses
  // that check, so both accesses are flagged here to keep them out of the loop-invariant caching pass.
  void record_const_loop_alias(ExternalPtrStmt *extern_ptr) {
    bool has_const = indices_have_const(extern_ptr->indices);
    bool has_loop_index = indices_have_loop_index(extern_ptr->indices);
    if (!has_const && !has_loop_index) {
      return;
    }
    for (auto *other_extern_ptr : extern_ptrs_) {
      if (other_extern_ptr == extern_ptr || other_extern_ptr->base_ptr != extern_ptr->base_ptr) {
        continue;
      }
      if ((has_const && indices_have_loop_index(other_extern_ptr->indices)) ||
          (has_loop_index && indices_have_const(other_extern_ptr->indices))) {
        record_dynamic_indexed_ptr(extern_ptr);
        return;
      }
    }
  }

  void record_const_loop_alias(GlobalPtrStmt *global_ptr) {
    bool has_const = indices_have_const(global_ptr->indices);
    bool has_loop_index = indices_have_loop_index(global_ptr->indices);
    if (!has_const && !has_loop_index) {
      return;
    }
    for (auto *other_global_ptr : global_ptrs_) {
      if (other_global_ptr == global_ptr || !is_leaf_nodes_on_same_branch(other_global_ptr->snode, global_ptr->snode)) {
        continue;
      }
      if ((has_const && indices_have_loop_index(other_global_ptr->indices)) ||
          (has_loop_index && indices_have_const(other_global_ptr->indices))) {
        record_dynamic_indexed_ptr(global_ptr);
        return;
      }
    }
  }

 public:
  explicit DynamicIndexingAnalyzer(IRNode *node) {
  }

  void visit(GlobalPtrStmt *stmt) override {
    for (auto *index_stmt : stmt->indices) {
      if (!index_stmt->is<ConstStmt>() && !index_stmt->is<LoopIndexStmt>()) {
        record_dynamic_indexed_ptr(stmt);
      }
    }
    record_const_loop_alias(stmt);

    global_ptrs_.insert(stmt);
  }

  void visit(ExternalPtrStmt *stmt) override {
    for (auto *index_stmt : stmt->indices) {
      if (!index_stmt->is<ConstStmt>() && !index_stmt->is<LoopIndexStmt>()) {
        record_dynamic_indexed_ptr(stmt);
      }
    }
    record_const_loop_alias(stmt);

    extern_ptrs_.insert(stmt);
  }

  void visit(MatrixPtrStmt *stmt) override {
    GlobalPtrStmt *global_ptr = nullptr;
    ExternalPtrStmt *extern_ptr = nullptr;

    if (stmt->origin->is<GlobalPtrStmt>()) {
      global_ptr = stmt->origin->as<GlobalPtrStmt>();
    } else if (stmt->origin->is<ExternalPtrStmt>()) {
      extern_ptr = stmt->origin->as<ExternalPtrStmt>();
    } else {
      return;
    }

    // Is dynamic index
    if (stmt->offset->is<ConstStmt>()) {
      return;
    }

    if (global_ptr) {
      record_dynamic_indexed_ptr(global_ptr);
    }

    if (extern_ptr) {
      record_dynamic_indexed_ptr(extern_ptr);
    }
  }

  std::unordered_set<Stmt *> get_dynamically_indexed_ptrs() {
    return dynamically_indexed_ptrs_;
  }

 private:
  using BasicStmtVisitor::visit;
  std::unordered_set<Stmt *> dynamically_indexed_ptrs_;
  std::unordered_set<GlobalPtrStmt *> global_ptrs_;
  std::unordered_set<ExternalPtrStmt *> extern_ptrs_;
};

namespace irpass::analysis {

std::unordered_set<Stmt *> gather_dynamically_indexed_pointers(IRNode *root) {
  DynamicIndexingAnalyzer pass(root);

  // This pass is intended to run twice
  root->accept(&pass);
  root->accept(&pass);

  auto dynamically_indexed_ptrs = pass.get_dynamically_indexed_ptrs();
  return dynamically_indexed_ptrs;
}

}  // namespace irpass::analysis
}  // namespace quadrants::lang
