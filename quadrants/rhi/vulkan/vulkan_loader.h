#pragma once

#include <thread>
#include <mutex>

#include "quadrants/rhi/vulkan/vulkan_common.h"
#include "quadrants/common/dynamic_loader.h"

namespace quadrants::lang {
namespace vulkan {

class QD_DLL_EXPORT VulkanLoader {
 public:
  static VulkanLoader &instance() {
    static VulkanLoader instance;
    return instance;
  }

 public:
  VulkanLoader(VulkanLoader const &) = delete;
  void operator=(VulkanLoader const &) = delete;

  bool check_vulkan_device();

  void load_instance(VkInstance instance_);
  void load_device(VkDevice device_);
  bool init(PFN_vkGetInstanceProcAddr get_proc_addr = nullptr);
  PFN_vkVoidFunction load_function(const char *name);
  VkInstance get_instance() {
    return vulkan_instance_;
  }
  // The VkInstance is normally destroyed and recreated on every qd.init()/qd.reset() cycle. We keep it
  // alive for the whole process ONLY on NVIDIA, where repeated vkDestroyInstance/vkCreateInstance
  // triggers a driver bug (SubgroupLocalInvocationId corruption after ~11 cycles). On every other
  // vendor -- notably MoltenVK, where reusing one VkInstance/MTLDevice across thousands of per-cycle
  // VkDevice create/destroy cycles leaks Metal state until vkQueueSubmit returns VK_ERROR_DEVICE_LOST
  // (~2900 cycles) -- the instance is torn down each cycle so the accumulation cannot build up.
  // Set from the selected physical device's vendorID (see VulkanDeviceCreator::create_logical_device).
  void set_keep_instance_alive(bool v) {
    keep_instance_alive_ = v;
  }
  bool keep_instance_alive() const {
    return keep_instance_alive_;
  }
  // Forget the cached instance handle so the next qd.init() creates a fresh one. The caller
  // (VulkanDeviceCreator dtor) is responsible for actually vkDestroyInstance-ing it first.
  void clear_instance() {
    vulkan_instance_ = VK_NULL_HANDLE;
  }
  std::string visible_device_id;

 private:
  std::once_flag init_flag_;
  bool initialized_{false};
  // Default true = conservative (don't tear down an instance we haven't classified yet, e.g. if
  // device creation throws before the vendor is read). Flipped to the correct value every cycle.
  bool keep_instance_alive_{true};

  VulkanLoader();

#if defined(__APPLE__)
  std::unique_ptr<DynamicLoader> vulkan_rt_{nullptr};
#endif

  VkInstance vulkan_instance_{VK_NULL_HANDLE};
  VkDevice vulkan_device_{VK_NULL_HANDLE};
};

QD_DLL_EXPORT bool is_vulkan_api_available();

QD_DLL_EXPORT void set_vulkan_visible_device(std::string id);

}  // namespace vulkan
}  // namespace quadrants::lang
