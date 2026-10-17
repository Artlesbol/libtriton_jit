// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// CPU-only ABI and dispatch-preparation checks. No NPU context or kernel launch.
#include "triton_jit/backends/npu_backend.h"

#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void publish(const std::string& dir, const std::string& name,
             std::optional<bool> auto_blockify) {
  using namespace triton_jit;
  NpuBackend::ModuleData data{};
  data.metadata.workspace_size = 48;
  data.metadata.mix_mode = "aiv";
  data.metadata.enable_auto_blockify = auto_blockify;
  // Deliberately different from the runtime signature: signature wins.
  data.metadata.arg_layout = {{NpuArgType::I64}};
  std::lock_guard lock(NpuBackend::cache_mutex_);
  NpuBackend::module_cache_.emplace(dir + "::" + name, std::move(data));
}

void check_preparation() {
  using namespace triton_jit;
  std::string dir = "/cpu-test/prepared", name = "tuple";
  const std::string signature = "*fp32:16,(fp32,i32),64";
  publish(dir, name, std::nullopt);
  setenv("TRITON_ALL_BLOCKS_PARALLEL", "false", 1);
  setenv("TRITON_JIT_NPU_PREPARED_LAUNCH", "0", 1);
  auto legacy = NpuBackend::prepare_launch(dir, name, 128, signature, 3);
  require(legacy.signature == signature, "legacy path must remain selectable");
  setenv("TRITON_JIT_NPU_PREPARED_LAUNCH", "1", 1);
  auto prepared = NpuBackend::prepare_launch(dir, name, 128, signature, 3);
  require(prepared.signature.empty(), "prepared path must reuse the parsed ABI");
  require(prepared.arg_layout && prepared.arg_layout->size() == 3, "tuple ABI must flatten");
  require((*prepared.arg_layout)[0].type == NpuArgType::POINTER, "pointer ABI changed");
  require((*prepared.arg_layout)[1].type == NpuArgType::F32, "float ABI changed");
  require((*prepared.arg_layout)[2].type == NpuArgType::I32, "integer ABI changed");
  require(prepared.workspace_size == legacy.workspace_size && prepared.workspace_size == 48,
          "workspace bytes changed");
  require(prepared.vector_kernel && prepared.shared_memory == 128, "kernel options changed");
  const auto* layout = prepared.arg_layout;
  auto repeated = NpuBackend::prepare_launch(dir, name, 256, signature, 5);
  require(repeated.arg_layout == layout, "unchanged signature must reuse its parsed layout");
  require(repeated.num_args == 5 && repeated.shared_memory == 256,
          "dynamic call options must not be cached");
  setenv("TRITON_ALL_BLOCKS_PARALLEL", "TrUe", 1);
  require(NpuBackend::prepare_launch(dir, name, 0, signature, 3).enable_auto_blockify,
          "environment changes must remain visible");
  setenv("TRITON_ALL_BLOCKS_PARALLEL", "0", 1);
  require(!NpuBackend::prepare_launch(dir, name, 0, signature, 3).enable_auto_blockify,
          "environment disable must remain visible");
  auto changed = NpuBackend::prepare_launch(dir, name, 0, "i64,*fp32", 2);
  require(changed.arg_layout->size() == 2 && (*changed.arg_layout)[0].type == NpuArgType::I64,
          "signature change reused stale options");
  require(prepared.arg_layout->size() == 3 && (*prepared.arg_layout)[0].type == NpuArgType::POINTER,
          "recursive signature preparation invalidated an in-flight layout");
  auto constants = NpuBackend::prepare_launch(dir, name, 0, "64,true,nullopt", 0);
  require(constants.arg_layout && constants.arg_layout->empty(), "constexpr-only ABI must be empty");
  dir = "/cpu-test/reused-object";
  publish(dir, name, false);
  setenv("TRITON_ALL_BLOCKS_PARALLEL", "true", 1);
  require(!NpuBackend::prepare_launch(dir, name, 0, signature, 3).enable_auto_blockify,
          "metadata must override environment after object-address reuse");
  name = "new-name";
  publish(dir, name, true);
  setenv("TRITON_ALL_BLOCKS_PARALLEL", "false", 1);
  require(NpuBackend::prepare_launch(dir, name, 0, signature, 3).enable_auto_blockify,
          "kernel name must be part of the cache validation");
  auto metadata_only = NpuBackend::prepare_launch(dir, name, 0, "", 1);
  require(metadata_only.arg_layout->size() == 1, "metadata ABI fallback changed");

  std::atomic<bool> failed{false};
  std::vector<std::thread> workers;
  for (int thread = 0; thread < 8; ++thread) {
    workers.emplace_back([&] {
      try {
        for (int i = 0; i < 1000; ++i) {
          auto opts = NpuBackend::prepare_launch(dir, name, 0, signature, i % 5);
          require(opts.num_args == static_cast<size_t>(i % 5) && opts.arg_layout->size() == 3,
                  "launching threads must have independent call options");
        }
      } catch (...) { failed = true; }
    });
  }
  for (auto& worker : workers) worker.join();
  require(!failed, "concurrent launch preparation failed");
}

void check_argument_buffers() {
  using namespace triton_jit;
  for (size_t estimate : {size_t(0), size_t(64), size_t(1024)}) {
    for (size_t count : {size_t(0), size_t(1), size_t(24), size_t(40), size_t(1000)}) {
      NpuArgBuffer buffer(estimate);
      void* ffts = reinterpret_cast<void*>(0x12345678);
      void* lock = reinterpret_cast<void*>(0x87654321);
      void* workspace = reinterpret_cast<void*>(0x24681357);
      buffer.set_system_args(ffts, lock, workspace);
      std::vector<std::byte> expected(24);
      std::memcpy(expected.data(), &ffts, 8);
      std::memcpy(expected.data() + 8, &lock, 8);
      std::memcpy(expected.data() + 16, &workspace, 8);
      auto append = [&]<typename T>(T value) {
        const size_t offset = (expected.size() + alignof(T) - 1) & ~(alignof(T) - 1);
        expected.resize(offset + sizeof(T));
        std::memcpy(expected.data() + offset, &value, sizeof(T));
        buffer.push_arg(value);
      };
      for (size_t i = 0; i < count; ++i) {
        if (i % 4 == 0) append(static_cast<int32_t>(i * 17));
        else if (i % 4 == 1) append(static_cast<int64_t>(i * 987654321));
        else if (i % 4 == 2) append(static_cast<float>(i + 0.125));
        else append(static_cast<double>(i + 0.25));
      }
      const size_t grid_offset = (expected.size() + 3) & ~size_t(3);
      expected.resize(grid_offset + 12);
      const int32_t grid[]{5, 7, 11};
      std::memcpy(expected.data() + grid_offset, grid, sizeof(grid));
      buffer.set_grid(5, 7, 11);
      require(buffer.size() == expected.size(), "packed ABI byte count changed");
      require(std::memcmp(buffer.data(), expected.data(), expected.size()) == 0,
              "inline/overflow buffer changed ABI bytes or padding");
    }
  }
}

}  // namespace

int main() {
  try {
    check_argument_buffers();
    check_preparation();
    std::cout << "NPU launch preparation and argument ABI checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
