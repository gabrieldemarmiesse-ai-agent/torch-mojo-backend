// The one piece of C++ in the profiler integration: an adapter from PyTorch's
// C++-only PrivateUse1 profiler hook (`ProfilerStubs`) to a C function table
// that Python / Mojo fill in. It is JIT-compiled against whatever torch is
// installed, so it never pins a torch version, and it is kept to this one
// translation unit with one torch header so the compile stays in seconds.
#include <torch/csrc/profiler/stubs/base.h>

#include <chrono>
#include <cstdint>
#include <memory>

extern "C" {
struct TmbProfilerHooks {
  // Returns an opaque event recorded on the current device's current stream,
  // and the device index it was recorded on.
  void* (*record)(int32_t* device_index);
  // Microseconds between two events (synchronizes as needed).
  float (*elapsed)(void* event, void* event2);
  void (*release)(void* event);
  void (*mark)(const char* name);
  void (*range_push)(const char* name);
  void (*range_pop)();
  int32_t (*device_count)();
  void (*synchronize)();
};
}

namespace {

TmbProfilerHooks g_hooks{};

struct Stubs final : torch::profiler::impl::ProfilerStubs {
  void record(
      c10::DeviceIndex* device,
      torch::profiler::impl::ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    if (cpu_ns) {
      *cpu_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
    }
    int32_t index = -1;
    void* ev = g_hooks.record ? g_hooks.record(&index) : nullptr;
    if (device) {
      *device = static_cast<c10::DeviceIndex>(index);
    }
    if (event) {
      *event = torch::profiler::impl::ProfilerVoidEventStub(ev, [](void* p) {
        if (p && g_hooks.release) {
          g_hooks.release(p);
        }
      });
    }
  }
  float elapsed(
      const torch::profiler::impl::ProfilerVoidEventStub* a,
      const torch::profiler::impl::ProfilerVoidEventStub* b) const override {
    if (!a || !b || !*a || !*b || !g_hooks.elapsed) {
      return 0.0f;
    }
    return g_hooks.elapsed(a->get(), b->get());
  }
  void mark(const char* name) const override {
    if (g_hooks.mark) g_hooks.mark(name);
  }
  void rangePush(const char* name) const override {
    if (g_hooks.range_push) g_hooks.range_push(name);
  }
  void rangePop() const override {
    if (g_hooks.range_pop) g_hooks.range_pop();
  }
  bool enabled() const override {
    return true;
  }
  void onEachDevice(std::function<void(int)> op) const override {
    const int32_t n = g_hooks.device_count ? g_hooks.device_count() : 0;
    for (int32_t i = 0; i < n; ++i) {
      op(i);
    }
  }
  void synchronize() const override {
    if (g_hooks.synchronize) g_hooks.synchronize();
  }
};

Stubs g_stubs;

} // namespace

extern "C" int32_t tmb_profiler_register(const TmbProfilerHooks* hooks) {
  if (!hooks) {
    return -1;
  }
  g_hooks = *hooks;
  torch::profiler::impl::registerPrivateUse1Methods(&g_stubs);
  return 0;
}
