// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "runtime/runtime/rt.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/backends/npu_arg_buffer.h"
#include "triton_jit/backends/npu_types.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"

namespace triton_jit {

struct NpuBackend {
  using StreamType = aclrtStream;
  using ContextType = aclrtContext;
  using KernelHandle = void*;

  // NPU does not use warp concept, but we need a non-zero value for block size calculation
  static constexpr unsigned int WARP_SIZE = 1;
  static constexpr bool cache_launch_device = true;

  static bool env_truthy(const char* value) {
    if (value == nullptr) return false;
    if (value[0] == '1' && value[1] == '\0') return true;
    const char* expected = "true";
    for (size_t i = 0; i < 4; ++i) {
      if (value[i] == '\0' ||
          std::tolower(static_cast<unsigned char>(value[i])) != expected[i]) {
        return false;
      }
    }
    return value[4] == '\0';
  }

  static bool prepared_launch_enabled() {
    const char* value = std::getenv("TRITON_JIT_NPU_PREPARED_LAUNCH");
    return value != nullptr && std::string_view(value) == "1";
  }

  struct LaunchOptions {
    unsigned int shared_memory = 0;
    std::string signature;
    const std::vector<NpuArgInfo>* arg_layout = nullptr;
    size_t num_args = 0;
    size_t workspace_size = 0;
    bool enable_auto_blockify = false;
    bool vector_kernel = true;
  };

  struct ModuleData {
    void* bin_handle;
    void* fn_handle;
    NpuKernelMetadata metadata;
  };

  static inline std::unordered_map<std::string, ModuleData> module_cache_;
  static inline std::mutex cache_mutex_;
  // Static storage for function stubs
  static inline std::unordered_map<std::string, size_t> registered_names_;
  static inline std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs_;

  static LaunchOptions prepare_launch(const std::string& dir,
                                      const std::string& name,
                                      unsigned int shared_mem,
                                      const std::string& sig,
                                      size_t num_args) {
    const char* prepared = std::getenv("TRITON_JIT_NPU_PREPARED_LAUNCH");
    if (prepared != nullptr && std::string_view(prepared) == "1") {
      return prepare_cached_launch(dir, name, shared_mem, sig, num_args);
    }
    return prepare_legacy_launch(dir, name, shared_mem, sig, num_args);
  }

  static LaunchOptions prepare_legacy_launch(const std::string& dir,
                                             const std::string& name,
                                             unsigned int shared_mem,
                                             const std::string& sig,
                                             size_t num_args) {
    const auto* metadata = get_kernel_metadata(dir, name);
    const auto* layout = (metadata && metadata->has_arg_layout()) ? &(metadata->arg_layout) : nullptr;
    size_t ws_size = metadata ? metadata->workspace_size : 0;
    // Match FlagTree Ascend's launcher: per-kernel metadata takes precedence
    // over TRITON_ALL_BLOCKS_PARALLEL. Ordinary kernels need the full logical
    // grid; only auto-blockified kernels loop over the remaining programs.
    bool auto_blockify = env_truthy(std::getenv("TRITON_ALL_BLOCKS_PARALLEL"));
    if (metadata && metadata->enable_auto_blockify.has_value()) {
      auto_blockify = *metadata->enable_auto_blockify;
    }

    VLOG(1) << fmt::format("NpuBackend::prepare_launch: kernel={}, metadata={}, workspace_size={}",
                           name,
                           metadata ? "found" : "null",
                           ws_size);

    return {
        .shared_memory = shared_mem,
        .signature = sig,
        .arg_layout = layout,
        .num_args = num_args,
        .workspace_size = ws_size,
        .enable_auto_blockify = auto_blockify,
        .vector_kernel = metadata && metadata->mix_mode == "aiv",
    };
  }

  static void launch_kernel(aclrtStream stream,
                            void* kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    uint32_t blockNum = grid_x * grid_y * grid_z;

    // The C2C control address is constant for a device/context. Cache it per
    // stream under the prepared-launch path; the stream identifies the
    // context. Default/null streams always refresh because they are shared
    // across devices.
    struct FftsCache {
      aclrtStream stream = nullptr;
      void* addr = nullptr;
      bool ready = false;
    };
    thread_local static FftsCache ffts_cache;
    const bool default_stream = stream == nullptr;
    void* ffts_addr = nullptr;
    if (prepared_launch_enabled() && !default_stream && ffts_cache.ready &&
        ffts_cache.stream == stream) {
      ffts_addr = ffts_cache.addr;
    } else {
      uint32_t ffts_len;
      rtError_t ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
      if (ret != RT_ERROR_NONE) {
        throw std::runtime_error(fmt::format("rtGetC2cCtrlAddr failed: {}", static_cast<int>(ret)));
      }
      if (prepared_launch_enabled()) {
        ffts_cache.stream = stream;
        ffts_cache.addr = ffts_addr;
        ffts_cache.ready = true;
      }
    }

    // Determine argument layout
    std::vector<NpuArgInfo> parsed_layout;
    const std::vector<NpuArgInfo>* layout_ptr = nullptr;
    if (!opts.signature.empty()) {
      parsed_layout = parse_signature(opts.signature);
      layout_ptr = &parsed_layout;
      VLOG(1) << fmt::format("Parsed signature '{}' -> {} runtime args", opts.signature, parsed_layout.size());
    } else if (opts.arg_layout != nullptr) {
      layout_ptr = opts.arg_layout;
      VLOG(1) << fmt::format("Using prepared arg_layout with {} args", layout_ptr->size());
    } else {
      throw std::runtime_error("launch_kernel: no signature or arg_layout provided");
    }
    const auto& layout = *layout_ptr;

    if (opts.num_args != 0 && opts.num_args != layout.size()) {
      throw std::runtime_error(fmt::format("launch_kernel: arg count mismatch (layout={}, args={})",
                                           layout.size(),
                                           opts.num_args));
    }

    // Build argument buffer dynamically
    NpuArgBuffer arg_buffer(layout.size() * 8 + 16);

    // 1. Allocate workspace if needed
    void* workspace_addr = nullptr;
    if (opts.workspace_size > 0) {
      size_t total_workspace = opts.workspace_size * blockNum;
      rtError_t ws_ret = rtMalloc(&workspace_addr, total_workspace, RT_MEMORY_HBM, 0);
      if (ws_ret != RT_ERROR_NONE) {
        throw std::runtime_error(fmt::format("rtMalloc workspace failed: {}, requested {} bytes",
                                             static_cast<int>(ws_ret),
                                             total_workspace));
      }
      VLOG(1) << fmt::format("NPU workspace allocated: {} bytes ({} per block x {} blocks)",
                             total_workspace,
                             opts.workspace_size,
                             blockNum);
    }

    // 2. Set system arguments (ffts, sync_lock, workspace)
    arg_buffer.set_system_args(ffts_addr, nullptr, workspace_addr);

    // 3. Add user arguments based on layout
    if (args != nullptr) {
      VLOG(1) << fmt::format("NPU args debug: num_args={}", layout.size());
      for (size_t i = 0; i < std::min(layout.size(), size_t(6)); ++i) {
        if (args[i] != nullptr) {
          if (layout[i].type == NpuArgType::POINTER) {
            void* ptr_val = *reinterpret_cast<void**>(args[i]);
            VLOG(1) << fmt::format("  arg[{}]: POINTER = {}", i, ptr_val);
          } else if (layout[i].type == NpuArgType::I64) {
            int64_t val = *reinterpret_cast<int64_t*>(args[i]);
            VLOG(1) << fmt::format("  arg[{}]: I64 = {}", i, val);
          }
        }
      }
      arg_buffer.push_args_from_layout(args, layout);
    } else {
      LOG(WARNING) << "launch_kernel: args is nullptr!";
    }

    // 4. Add grid dimensions
    arg_buffer.set_grid(static_cast<int32_t>(grid_x),
                        static_cast<int32_t>(grid_y),
                        static_cast<int32_t>(grid_z));

    VLOG(1) << fmt::format(
        "NPU launch_kernel: blockNum={}, arg_buffer_size={}, grid=({},{},{}), workspace={}",
        blockNum,
        arg_buffer.size(),
        grid_x,
        grid_y,
        grid_z,
        workspace_addr ? fmt::format("{}B", opts.workspace_size * blockNum) : "none");

    // A physical-core clamp without compiler-generated auto-blockify loops
    // silently drops all logical programs beyond the clamp.
    if (opts.enable_auto_blockify) {
      uint32_t ai_core_cnt = 0;
      if (rtGetAiCoreCount(&ai_core_cnt) != RT_ERROR_NONE || ai_core_cnt == 0) {
        if (workspace_addr) rtFree(workspace_addr);
        throw std::runtime_error("Cannot determine physical block count for auto-blockified kernel");
      }
      blockNum = std::min(blockNum, ai_core_cnt * (opts.vector_kernel ? 2u : 1u));
    }

    // Launch kernel
    rtError_t rt_err = rtKernelLaunch(kernel,
                                      blockNum,
                                      arg_buffer.data(),
                                      static_cast<uint32_t>(arg_buffer.size()),
                                      nullptr,
                                      stream);

    if (rt_err != RT_ERROR_NONE) {
      if (workspace_addr) {
        rtFree(workspace_addr);
      }
      throw std::runtime_error(fmt::format("rtKernelLaunch failed: {}", static_cast<int>(rt_err)));
    }

    // Free workspace after kernel completes via stream callback
    if (workspace_addr) {
      aclError cb_err = aclrtLaunchCallback([](void* userData) { rtFree(userData); },
                                            workspace_addr,
                                            ACL_CALLBACK_BLOCK,
                                            stream);
      if (cb_err != ACL_ERROR_NONE) {
        LOG(WARNING) << fmt::format(
            "Failed to register workspace cleanup callback (err={}), "
            "workspace may leak",
            static_cast<int>(cb_err));
      }
    }
  }

  static void ensure_context() {
    aclrtContext ctx;
    aclError ret = aclrtGetCurrentContext(&ctx);

    if (ret != ACL_ERROR_NONE || ctx == nullptr) {
      LOG(WARNING) << "No ACL context found. Creating default context.";
      int deviceId = 0;
      aclError err = aclrtSetDevice(deviceId);
      if (err != ACL_ERROR_NONE) {
        throw std::runtime_error(fmt::format("aclrtSetDevice failed: {}", static_cast<int>(err)));
      }
      err = aclrtCreateContext(&ctx, deviceId);
      if (err != ACL_ERROR_NONE) {
        throw std::runtime_error(fmt::format("aclrtCreateContext failed: {}", static_cast<int>(err)));
      }
      err = aclrtSetCurrentContext(ctx);
      if (err != ACL_ERROR_NONE) {
        throw std::runtime_error(fmt::format("aclrtSetCurrentContext failed: {}", static_cast<int>(err)));
      }
    }
  }

  static int get_device_index() {
    int device_id = -1;
    aclError err = aclrtGetDevice(&device_id);

    if (err != ACL_ERROR_NONE) {
      throw std::runtime_error(fmt::format("Failed to get NPU device: {}", static_cast<int>(err)));
    }

    return device_id;
  }

  static void* load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check cache first
    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.fn_handle;
    }

    // Load metadata via centralized loader (no JSON dependency in header)
    NpuKernelMetadata metadata = load_npu_metadata(dir, kernel_name);

    VLOG(1) << fmt::format("Loading NPU kernel {} with mix_mode={}, shared={}",
                           kernel_name,
                           metadata.mix_mode,
                           metadata.shared);

    // Find kernel binary file (try .npubin, .o, .ttadapter, .bin)
    std::string rt_bin_path = fmt::format("{}/{}.npubin", dir, kernel_name);
    std::ifstream bin_file(rt_bin_path, std::ios::binary | std::ios::ate);

    if (!bin_file.good()) {
      std::vector<std::string> fallback_exts = {".o", ".ttadapter", ".bin"};
      bool file_found = false;

      for (const auto& ext : fallback_exts) {
        rt_bin_path = fmt::format("{}/{}{}", dir, kernel_name, ext);
        bin_file.open(rt_bin_path, std::ios::binary | std::ios::ate);
        if (bin_file.good()) {
          file_found = true;
          break;
        }
        bin_file.close();
        bin_file.clear();
      }

      if (!file_found) {
        throw std::runtime_error(fmt::format("Kernel binary not found: {}/{}", dir, kernel_name));
      }
    }

    // Read binary file
    std::streamsize size = bin_file.tellg();
    if (size <= 0) {
      throw std::runtime_error(fmt::format("Invalid binary size: {}", rt_bin_path));
    }

    bin_file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));

    if (!bin_file.read(buffer.data(), size)) {
      throw std::runtime_error(fmt::format("Failed to read binary: {}", rt_bin_path));
    }
    bin_file.close();

    VLOG(1) << fmt::format("Loading NPU binary from {}, size={}", rt_bin_path, size);

    // Get current device ID
    int device_id = -1;
    aclError err = aclrtGetDevice(&device_id);
    if (err != ACL_SUCCESS) {
      device_id = 0;  // fallback
    }

    // Set device
    rtError_t rt_err = rtSetDevice(device_id);
    if (rt_err != RT_ERROR_NONE) {
      throw std::runtime_error(
          fmt::format("rtSetDevice failed for device {}, error: {}", device_id, static_cast<int>(rt_err)));
    }

    // Register binary with RT API
    rtDevBinary_t binary;
    binary.data = buffer.data();
    binary.length = static_cast<uint32_t>(size);

    binary.magic = (metadata.mix_mode == "aiv") ? RT_DEV_BINARY_MAGIC_ELF_AIVEC : RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;

    void* rt_bin_handle = nullptr;
    rt_err = rtDevBinaryRegister(&binary, &rt_bin_handle);
    if (rt_err != RT_ERROR_NONE) {
      throw std::runtime_error(fmt::format("rtDevBinaryRegister failed: {}", static_cast<int>(rt_err)));
    }

    // Create function stub with unique name
    std::string stubName = kernel_name;
    stubName += "_" + std::to_string(registered_names_[kernel_name]);
    registered_names_[kernel_name]++;

    auto registered = func_stubs_.emplace(stubName, std::make_unique<size_t>(0));
    void* func_stub_handle = registered.first->second.get();

    // Register function
    rt_err =
        rtFunctionRegister(rt_bin_handle, func_stub_handle, stubName.c_str(), (void*)kernel_name.c_str(), 0);
    if (rt_err != RT_ERROR_NONE) {
      throw std::runtime_error(fmt::format("rtFunctionRegister failed: {}", static_cast<int>(rt_err)));
    }

    // Cache the module
    module_cache_[key] = ModuleData {rt_bin_handle, func_stub_handle, metadata};

    return func_stub_handle;
  }

  static unsigned int get_shared_memory(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.metadata.shared;
    }

    return load_shared_memory(dir, kernel_name);
  }

  // Kernel metadata is immutable after load_kernel() publishes ModuleData;
  // there is no module-unload API. Cache the resolved ABI per kernel object
  // and launching thread, keeping runtime argument values out of the cache.
  static LaunchOptions prepare_cached_launch(const std::string& dir,
                                             const std::string& name,
                                             unsigned int shared_mem,
                                             const std::string& sig,
                                             size_t num_args) {
    struct Prepared {
      std::string dir;
      std::string name;
      std::string signature;
      std::vector<NpuArgInfo> layout;
      size_t workspace_size = 0;
      std::optional<bool> auto_blockify;
      bool vector_kernel = false;
      bool valid = false;
    };
    // TritonKernel owns dir. Validate values as well as its address so a
    // destroyed/moved kernel object cannot reuse another kernel's options.
    // Retain each signature's layout separately: an enter hook may prepare
    // another signature recursively while its caller still owns LaunchOptions.
    thread_local static std::unordered_map<
        const std::string*, std::unordered_map<std::string, Prepared>> cache;
    auto& state = cache[&dir][sig];
    if (!state.valid || state.dir != dir || state.name != name || state.signature != sig) {
      const auto* metadata = get_kernel_metadata(dir, name);
      if (metadata == nullptr || (sig.empty() && !metadata->has_arg_layout())) {
        return prepare_legacy_launch(dir, name, shared_mem, sig, num_args);
      }
      Prepared next;
      next.dir = dir;
      next.name = name;
      next.signature = sig;
      // Keep the established signature-first precedence, including tuple
      // flattening and constexpr removal. Metadata may describe another ABI.
      next.layout = sig.empty() ? metadata->arg_layout : parse_signature(sig);
      next.workspace_size = metadata->workspace_size;
      next.auto_blockify = metadata->enable_auto_blockify;
      next.vector_kernel = metadata->mix_mode == "aiv";
      next.valid = true;
      state = std::move(next);
    }
    bool auto_blockify = false;
    if (state.auto_blockify.has_value()) {
      auto_blockify = *state.auto_blockify;
    } else {
      // This environment setting is deliberately dynamic. Per-kernel
      // metadata still takes precedence, exactly as in the legacy launcher.
      auto_blockify = env_truthy(std::getenv("TRITON_ALL_BLOCKS_PARALLEL"));
    }
    return {
        .shared_memory = shared_mem,
        .signature = {},
        .arg_layout = &state.layout,
        .num_args = num_args,
        .workspace_size = state.workspace_size,
        .enable_auto_blockify = auto_blockify,
        .vector_kernel = state.vector_kernel,
    };
  }

  /**
   * @brief Get kernel metadata including arg_layout
   */
  static const NpuKernelMetadata* get_kernel_metadata(const std::string& dir,
                                                      const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return &(it->second.metadata);
    }

    return nullptr;
  }
};

static_assert(BackendPolicy<NpuBackend>, "NpuBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
