"""Vendor driver calls on MAX's raw streams.

MAX's DeviceEvent can be recorded, waited on and synchronized but not queried
or timed, so the two things torch.Event needs beyond ordering come from the
CUDA / HIP driver on the CUstream / hipStream_t behind each MAX stream. Metal
and the CPU device have no driver here; device.mojo answers those from host
clocks instead.
"""
from std.ffi import OwnedDLHandle, external_call
from std.sys.info import has_amd_gpu_accelerator

from max.gpu.host import DeviceContext

comptime AMD = has_amd_gpu_accelerator()
comptime DRIVER_LIB = "libamdhip64.so" if AMD else "libcuda.so.1"
comptime FN_EVENT_CREATE = "hipEventCreateWithFlags" if AMD else "cuEventCreate"
comptime FN_EVENT_RECORD = "hipEventRecord" if AMD else "cuEventRecord"
comptime FN_EVENT_QUERY = "hipEventQuery" if AMD else "cuEventQuery"
comptime FN_EVENT_SYNC = "hipEventSynchronize" if AMD else "cuEventSynchronize"
comptime FN_EVENT_ELAPSED = "hipEventElapsedTime" if AMD else "cuEventElapsedTime"
comptime FN_EVENT_DESTROY = "hipEventDestroy" if AMD else "cuEventDestroy_v2"
comptime FN_STREAM_WAIT_EVENT = "hipStreamWaitEvent" if AMD else "cuStreamWaitEvent"
comptime FN_STREAM_QUERY = "hipStreamQuery" if AMD else "cuStreamQuery"
comptime FN_RAW_STREAM = (
    "AsyncRT_DeviceStream_hip_stream" if AMD else "AsyncRT_DeviceStream_cuda_stream"
)
comptime NOT_READY = Int32(600)  # CUDA_ERROR_NOT_READY == hipErrorNotReady
comptime DISABLE_TIMING = UInt32(
    2
)  # CU_EVENT_DISABLE_TIMING == hipEventDisableTiming


struct Vendor(Movable):
    var lib: OwnedDLHandle

    def __init__(out self) raises:
        self.lib = OwnedDLHandle(DRIVER_LIB)

    def _check(self, rc: Int32, what: String) raises:
        if rc != 0:
            raise Error(what, " failed with driver error ", rc)

    def event_create(self, ctx: DeviceContext, timing: Bool) raises -> Int:
        var ev: Int = 0
        with ctx.push_context():  # cuEventCreate needs the device's context current
            self._check(
                self.lib.get_function[Int32](FN_EVENT_CREATE)(
                    Pointer(to=ev), UInt32(0) if timing else DISABLE_TIMING
                ),
                FN_EVENT_CREATE,
            )
        return ev

    def event_record(self, ev: Int, raw_stream: Int) raises:
        self._check(
            self.lib.get_function[Int32](FN_EVENT_RECORD)(ev, raw_stream),
            FN_EVENT_RECORD,
        )

    def event_query(self, ev: Int) raises -> Bool:
        var rc = self.lib.get_function[Int32](FN_EVENT_QUERY)(ev)
        if rc == 0:
            return True
        if rc == NOT_READY:
            return False
        self._check(rc, FN_EVENT_QUERY)
        return True

    def event_synchronize(self, ev: Int) raises:
        self._check(
            self.lib.get_function[Int32](FN_EVENT_SYNC)(ev), FN_EVENT_SYNC
        )

    def event_elapsed_ms(self, start: Int, end: Int) raises -> Float64:
        var ms: Float32 = 0
        self._check(
            self.lib.get_function[Int32](FN_EVENT_ELAPSED)(
                Pointer(to=ms), start, end
            ),
            FN_EVENT_ELAPSED,
        )
        return Float64(ms)

    def event_destroy(self, ev: Int) raises:
        _ = self.lib.get_function[Int32](FN_EVENT_DESTROY)(ev)

    def stream_wait_event(self, raw_stream: Int, ev: Int) raises:
        self._check(
            self.lib.get_function[Int32](FN_STREAM_WAIT_EVENT)(
                raw_stream, ev, UInt32(0)
            ),
            FN_STREAM_WAIT_EVENT,
        )

    def stream_query(self, raw_stream: Int) raises -> Bool:
        var rc = self.lib.get_function[Int32](FN_STREAM_QUERY)(raw_stream)
        if rc == 0:
            return True
        if rc == NOT_READY:
            return False
        self._check(rc, FN_STREAM_QUERY)
        return True


def raw_stream(ctx: DeviceContext) raises -> Int:
    """The CUstream / hipStream_t behind a context view's stream."""
    var raw: Int = 0
    var err = external_call[FN_RAW_STREAM, OpaquePointer[MutUntrackedOrigin]](
        Pointer(to=raw), ctx.stream()._handle
    )
    if Int(err) != 0:
        raise Error("raw stream handle unavailable")
    return raw
