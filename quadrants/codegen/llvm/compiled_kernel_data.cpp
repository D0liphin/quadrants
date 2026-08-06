#include "quadrants/codegen/llvm/compiled_kernel_data.h"

#include "llvm/IR/Verifier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"

namespace quadrants::lang {

static std::unique_ptr<CompiledKernelData> new_llvm_compiled_kernel_data() {
  return std::make_unique<LLVM::CompiledKernelData>();
}

CompiledKernelData::Creator *const CompiledKernelData::llvm_creator = new_llvm_compiled_kernel_data;

namespace LLVM {

CompiledKernelData::CompiledKernelData(Arch arch, InternalData data) : arch_(arch), data_(std::move(data)) {
}

Arch CompiledKernelData::arch() const {
  return arch_;
}

std::unique_ptr<lang::CompiledKernelData> CompiledKernelData::clone() const {
  return std::make_unique<CompiledKernelData>(arch_, data_);
}

CompiledKernelData::Err CompiledKernelData::check() const {
  const auto &compiled_data = data_.compiled_data;
  const auto &tasks = compiled_data.tasks;
  // Per-task cuLink path (§9.D): when one or more tasks were served from the on-disk artifact cache there is no
  // whole-kernel LLVM module in this process -- the device code is a set of per-task cubins that the CUDA JIT
  // device-links. The module-based checks below are meaningless (and would null-deref), so validate what this
  // representation actually guarantees: every task must have code, i.e. an artifact carrying it.
  if (!compiled_data.module) {
    if (compiled_data.per_construct_artifacts.empty() || tasks.empty()) {
      return Err::kCompiledKernelDataBroken;
    }
    return Err::kNoError;
  }
  if (llvm::verifyModule(*compiled_data.module, &llvm::errs())) {
    return Err::kCompiledKernelDataBroken;
  }
  for (const auto &t : tasks) {
    if (compiled_data.module->getFunction(t.name) == nullptr) {
      return Err::kCompiledKernelDataBroken;
    }
  }
  return Err::kNoError;
}

std::string CompiledKernelData::debug_dump_to_string() const {
  auto &data = this->get_internal_data().compiled_data;
  auto *module = data.module.get();
  if (module == nullptr) {
    // Per-task cuLink path: no whole-kernel module to print (device code is per-task cubins).
    return fmt::format("<no whole-kernel module: {} per-task cubin artifact(s)>", data.per_construct_artifacts.size());
  }
  std::string result;
  llvm::raw_string_ostream oss(result);
  module->print(oss, /*AAW=*/nullptr);
  return oss.str();
}

CompiledKernelData::Err CompiledKernelData::load_impl(const CompiledKernelDataFile &file) {
  arch_ = file.arch();
  if (!arch_uses_llvm(arch_)) {
    return Err::kArchNotMatched;
  }
  try {
    liong::json::deserialize(liong::json::parse(file.metadata()), data_, true);
  } catch (const liong::json::JsonException &) {
    return Err::kParseMetadataFailed;
  }
  llvm::SMDiagnostic err;
  auto ret = llvm::parseAssemblyString(file.src_code(), err, llvm_ctx_);
  if (!ret) {  // File not found or Parse failed
    QD_DEBUG("Fail to parse llvm::Module from string: {}", err.getMessage().str());
    return Err::kParseSrcCodeFailed;
  }
  data_.compiled_data.module = std::move(ret);
  llvm::Module *mod = data_.compiled_data.module.get();
  mod->setModuleIdentifier("kernel");
  return Err::kNoError;
}

CompiledKernelData::Err CompiledKernelData::dump_impl(CompiledKernelDataFile &file) const {
  file.set_arch(arch_);
  try {
    file.set_metadata(liong::json::print(liong::json::serialize(data_)));
  } catch (const liong::json::JsonException &) {
    return Err::kSerMetadataFailed;
  }
  // On the per-task cuLink path (§9.D) a kernel whose tasks all came from the on-disk artifact cache has no
  // whole-kernel LLVM module -- the device code lives as per-task cubins instead. There is nothing to write into the
  // `.qdc` src_code field, so report a dump failure: `KernelCompilationManager::dump` treats that as "skip this
  // entry" (debug log, no `.qdc` written), which is exactly right because the artifact cache is already this
  // kernel's cross-process persistence.
  if (!data_.compiled_data.module) {
    return Err::kSerSrcCodeFailed;
  }
  std::string str;
  llvm::raw_string_ostream oss(str);
  data_.compiled_data.module->print(oss, /*AAW=*/nullptr);
  file.set_src_code(std::move(str));
  return Err::kNoError;
}

}  // namespace LLVM
}  // namespace quadrants::lang
