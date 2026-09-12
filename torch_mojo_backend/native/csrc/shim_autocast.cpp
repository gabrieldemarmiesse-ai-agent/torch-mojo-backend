// AutocastPrivateUse1 as ONE boxed fallback with a policy table, instead of
// torch's per-op unboxed WrapFunction templates (which take tens of seconds to
// compile). The policies and the op lists are torch's own CUDA ones.
#include "tmb_internal.h"

#include <ATen/autocast_mode.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKeySet.h>
#include <torch/library.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace {

enum Policy : int32_t { NONE = 0, LOWER_PRECISION_FP = 1, FP32 = 2, FP32_SET_OPT_DTYPE = 3, PROMOTE = 4 };

std::mutex g_policy_mutex;
std::unordered_map<std::string, int32_t> g_policies;

int32_t lookup_policy(const c10::FunctionSchema& schema) {
  std::lock_guard<std::mutex> g(g_policy_mutex);
  const auto& name = schema.name();
  const auto& overload = schema.overload_name();
  if (!overload.empty()) {
    auto it = g_policies.find(name + "." + overload);
    if (it != g_policies.end()) return it->second;
  }
  auto it = g_policies.find(name);
  return it == g_policies.end() ? NONE : it->second;
}

constexpr auto kDevice = c10::DeviceType::PrivateUse1;

c10::IValue cast_value(const c10::IValue& v, c10::ScalarType to) {
  if (v.isTensor()) return at::autocast::cached_cast(to, v.toTensor(), kDevice);
  if (v.isTensorList()) {
    auto in = v.toTensorList();
    c10::List<at::Tensor> out;
    out.reserve(in.size());
    for (const at::Tensor t : in) out.push_back(at::autocast::cached_cast(to, t, kDevice));
    return c10::IValue(std::move(out));
  }
  if (v.isOptionalTensorList()) {
    auto in = v.toOptionalTensorList();
    c10::List<std::optional<at::Tensor>> out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      std::optional<at::Tensor> t = in.get(i);
      out.push_back(t ? std::optional<at::Tensor>(at::autocast::cached_cast(to, *t, kDevice)) : std::nullopt);
    }
    return c10::IValue(std::move(out));
  }
  return v;
}

void autocast_fallback(const c10::OperatorHandle& op, c10::DispatchKeySet ks, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const int32_t policy = lookup_policy(schema);
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastPrivateUse1);
  if (policy != NONE) {
    const size_t n = schema.arguments().size();
    c10::IValue* args = stack->data() + (stack->size() - n);
    if (policy == LOWER_PRECISION_FP || policy == FP32) {
      const auto to = policy == FP32 ? at::kFloat : at::autocast::get_lower_precision_fp_from_device_type(kDevice);
      for (size_t i = 0; i < n; ++i) args[i] = cast_value(args[i], to);
    } else if (policy == PROMOTE) {
      auto widest = at::ScalarType::Undefined;
      for (size_t i = 0; i < n; ++i) {
        if (args[i].isTensor()) widest = at::autocast::prioritize(widest, args[i].toTensor(), kDevice);
      }
      if (widest != at::ScalarType::Undefined) {
        for (size_t i = 0; i < n; ++i) args[i] = cast_value(args[i], widest);
      }
    } else if (policy == FP32_SET_OPT_DTYPE) {
      for (size_t i = 0; i < n; ++i) {
        if (schema.arguments()[i].name() == "dtype" && args[i].isNone()) args[i] = c10::IValue(at::kFloat);
      }
    }
  }
  (void)ks;
  op.callBoxed(stack);  // recomputed key set skips AutocastPrivateUse1 via the TLS exclude above
}

#define TMB_Q1(op) "aten::" #op
#define TMB_Q2(op, ov) "aten::" #op "." #ov
#define TMB_QSEL(_1, _2, NAME, ...) NAME
#define TMB_QNAME(...) TMB_QSEL(__VA_ARGS__, TMB_Q2, TMB_Q1, )(__VA_ARGS__)
#define TMB_LOW(...) g_policies[TMB_QNAME(__VA_ARGS__)] = LOWER_PRECISION_FP;
#define TMB_FP32(...) g_policies[TMB_QNAME(__VA_ARGS__)] = FP32;
#define TMB_SET(...) g_policies[TMB_QNAME(__VA_ARGS__)] = FP32_SET_OPT_DTYPE;
#define TMB_PROMOTE(...) g_policies[TMB_QNAME(__VA_ARGS__)] = PROMOTE;

}  // namespace

extern "C" {

int32_t tmb_autocast_policy(const char* qualified_name, int32_t policy) {
  std::lock_guard<std::mutex> g(g_policy_mutex);
  g_policies[qualified_name] = policy;
  return 0;
}

int32_t tmb_autocast_install_cuda_policies(void) {
  std::lock_guard<std::mutex> g(g_policy_mutex);
  AT_FORALL_LOWER_PRECISION_FP(TMB_LOW)
  AT_FORALL_FP32(TMB_FP32)
  AT_FORALL_FP32_SET_OPT_DTYPE(TMB_SET)
  AT_FORALL_PROMOTE(TMB_PROMOTE)
  return 0;
}

}  // extern "C"

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&autocast_fallback>());
}
