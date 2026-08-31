# Streams and events on the mojo device

The mojo device supports the device-generic accelerator stream API:

```python
s = torch.Stream(device=torch.accelerator.current_accelerator())
cur = torch.accelerator.current_stream()
s.wait_stream(cur)
with s:
    ...                      # torch.accelerator.current_stream() is now s
e = s.record_event()         # a torch.Event
cur.wait_event(e)
e.synchronize()
```

`torch.Stream(device="mojo")`, `torch.Event(device="mojo", enable_timing=True)`,
`torch.accelerator.current_stream()` / `set_stream()`, and the `torch.mojo`
module equivalents (`Stream`, `Event`, `current_stream`, `default_stream`,
`set_stream`, `stream`) all work, with `query` / `synchronize` /
`wait_event` / `wait_stream` / `record_event` / `elapsed_time` backed by
real CUDA driver events on real streams. `isinstance(s, torch.Stream)`
holds in both directions. Not supported: interprocess events
(`from_ipc_handle`), stream priorities (accepted, always 0), and graph
capture (`is_capturing()` is always `False`).

## Why registration patches torch.Stream

PyTorch's generic `torch.Stream`/`torch.Event` route through a C++ device
guard that is a stub for Python-backed PrivateUse1 devices — every stream
it mints is stream id 0 and every wait/record is a silent no-op. So
`register_mojo_devices()` dispatches construction on mojo devices to the
implementations in `mojo_device/streams.py` (built on the channel layer in
`mojo_device/channels.py`, which also carries the NCCL comm stream for
distributed training). This mirrors the existing
`torch.accelerator.synchronize` patch and disappears if upstream ever
forwards the guard hooks to Python device modules.

## Execution semantics (read this before pipelining)

Eager mojo kernels currently always **execute on the device's default
stream**, regardless of the current stream. `with s:` tracks the current
stream per thread and every ordering primitive acts on the real streams,
so device-agnostic pipelining code runs with exactly the semantics it
would have on CUDA with a single stream: correct, but without extra
compute concurrency. (NCCL collectives are the exception — they really do
run on their own channel and overlap compute; see `docs/distributed.md`.)
Redirecting kernel launches through per-stream MAX DeviceContexts is the
known follow-up that would make `with s:` fully concurrent.

Two rules carried over from CUDA apply unchanged: work you enqueue is only
on a stream once the kernel-call queue has launched it (host reads and
`torch.mojo.synchronize()` drain it for you), and a tensor produced on one
stream must be ordered (event or `wait_stream`) before another stream —
including external consumers — touches it.
