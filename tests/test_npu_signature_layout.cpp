// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

#include <barrier>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "triton_jit/backends/npu_backend.h"

namespace {
void require(bool value) {
  if (!value) throw std::runtime_error("NPU signature layout check failed");
}

void check_layout(const std::string& signature) {
  const auto expected = triton_jit::parse_signature(signature);
  const auto& cached = triton_jit::NpuBackend::signature_layout(signature);
  require(expected.size() == cached.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    require(expected[i].type == cached[i].type);
  }
  require(&cached == &triton_jit::NpuBackend::signature_layout(signature));
}
}  // namespace

int main() {
  const std::vector<std::string> signatures{
      "*fp32:16,*fp32:16,*i32:16,i32",
      "*fp64:16,i64,u32,u64,fp32,fp64",
      "*fp32,(i32,fp64),1024,nullopt,true,false,@jit:test",
      "*fp32:16,i32:1",
      "*fp32:16,i64",
      "", "1024,nullopt"};
  for (const auto& signature : signatures) check_layout(signature);

  const auto* first = &triton_jit::NpuBackend::signature_layout(signatures.front());
  for (size_t i = 0; i < 2048; ++i) {
    check_layout("*fp32,i32," + std::to_string(i));
  }
  require(first == &triton_jit::NpuBackend::signature_layout(signatures.front()));
  require(first->size() == 4);

  for (const std::string invalid : {"i32,", "(i32", "((i32))", "()", "i32,,i64"}) {
    for (int repeat = 0; repeat < 2; ++repeat) {
      bool rejected = false;
      try {
        triton_jit::NpuBackend::signature_layout(invalid);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected);
    }
  }

  std::barrier ready(3);
  auto check_thread = [&] {
    const auto* local = &triton_jit::NpuBackend::signature_layout(signatures.front());
    ready.arrive_and_wait();
    require(local != first);
    for (int repeat = 0; repeat < 32; ++repeat) {
      for (const auto& signature : signatures) check_layout(signature);
    }
    ready.arrive_and_wait();
  };
  auto a = std::async(std::launch::async, check_thread);
  auto b = std::async(std::launch::async, check_thread);
  ready.arrive_and_wait();
  check_layout(signatures.front());
  ready.arrive_and_wait();
  a.get();
  b.get();
  std::cout << "NPU signature layouts: parser equivalence, reuse, rehash, "
               "invalid inputs and thread isolation passed\n";
}
