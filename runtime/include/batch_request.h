#ifndef __TUTTI_RUNTIME_BATCH_REQUEST_H__
#define __TUTTI_RUNTIME_BATCH_REQUEST_H__

/**
 * batch_request.h -- runtime-level batched IO submission unit.
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - BatchRequest is the high-level submission shape application code
 *     hands to the Runtime. It's the natural pair to BatchCompletion:
 *     "submit one BatchRequest, get one BatchCompletion."
 *   - Internally, the Runtime translates a BatchRequest into the
 *     SPI-level pair (BufferDescriptorBatch, IORequestBatch) before
 *     calling IBackendProvider::submit_*. Adapters (LMCache,
 *     Mooncake) construct BatchRequests; backends never see them.
 *
 * Why separate from io_engine's IORequestBatch:
 *   - IORequestBatch lives in the SPI; it carries already-resolved
 *     BufferDescriptor pointers and a MemoryRegion handle. By contrast
 *     BatchRequest is application-shaped: it references IOBuffers and
 *     StorageTargets, leaving address translation, DMA mapping, and
 *     descriptor preparation to the Runtime's "lower the request"
 *     pipeline.
 *
 * Direction policy:
 *   - One BatchRequest carries one direction (`is_read`). Mixed-direction
 *     batches are not supported in v0.1 (matches SPI signature).
 *
 * Submit-mode policy:
 *   - The caller picks the IOSubmitMode at submit time (passed into
 *     Runtime::submit_batch). The same BatchRequest may be submitted
 *     multiple times, possibly under different modes (e.g. retry
 *     BATCH_GPU_STREAM as BATCH_CPU_SYNC on GPU error).
 *
 * Lifetime:
 *   - BatchRequest is value-typed. Caller-owned. The runtime captures
 *     what it needs at submit time and does not retain a pointer to
 *     the BatchRequest after submission returns / future fires.
 */

#include <cstdint>
#include <vector>

namespace tutti {

struct StorageTarget;          // storage_target.h
struct IOBuffer;               // io_buffer.h
class  ICompletionSink;        // declared below

/**
 * One IO inside a BatchRequest. Pairs a target (where) with a buffer
 * (with what) plus the byte window inside that buffer. The runtime
 * translates this into one or more SPI-level IORequest entries
 * (a single application IO may fan out to multiple device IOs when
 * the buffer crosses a granularity slice).
 */
struct BatchEntry {
    const StorageTarget* target;       // where to read from / write to
    const IOBuffer*      buffer;       // application data buffer
    uint64_t             buffer_offset; // bytes into buffer.byte_length
    uint64_t             length;        // transfer size in bytes
    uint64_t             target_offset; // bytes into the target (file offset, LBA*block_size, ...)
};

/**
 * Per-entry result the runtime fills in via the completion sink.
 *
 * `entry_index` matches the position of the BatchEntry inside the
 * submitted BatchRequest::entries vector. Mirrors the SPI's
 * IOCompletion shape but stays at the runtime/object-model level
 * (no SPI-internal request_index).
 */
struct BatchCompletion {
    uint32_t entry_index;
    uint32_t status;       // 0 = success; non-zero = backend-defined
    uint64_t bytes_done;
};

/**
 * Sink the Runtime drains completions into. The application implements
 * this; the Runtime calls on_complete() for each entry as it lands.
 *
 * For BATCH_CPU_SYNC the calls happen before submit_batch() returns.
 * For BATCH_GPU_STREAM / BATCH_CPU_ASYNC / COOP they happen later
 * (CUDA stream callback, async future fire, or proxy-thread drain).
 */
class ICompletionSink {
public:
    virtual ~ICompletionSink() = default;
    virtual void on_complete(const BatchCompletion& c) = 0;

    /// Optional: called once after every entry has been completed
    /// (status reported via on_complete) so the sink knows to release
    /// any per-batch resources. Default no-op.
    virtual void on_batch_finished() {}
};

/**
 * The full submission unit.
 *
 * `entries` is value-stored so the caller can build it on the stack
 * for small batches without allocating; for large batches the caller
 * uses move semantics.
 */
struct BatchRequest {
    std::vector<BatchEntry> entries;
    bool                    is_read;     // direction (uniform)

    // Optional completion sink. May be nullptr for fire-and-forget
    // patterns (the runtime then logs failures and drops). Most
    // callers populate it.
    ICompletionSink*        sink;

    // Optional lease the runtime checks before submitting. Lets the
    // application bind a batch to a specific lease so a reaper-event
    // mid-flight can fail submissions early. May be empty.
    std::string             lease_id;
};

} // namespace tutti

#endif // __TUTTI_RUNTIME_BATCH_REQUEST_H__
