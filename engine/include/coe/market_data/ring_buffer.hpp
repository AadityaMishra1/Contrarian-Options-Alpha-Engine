#pragma once

#include <array>
#include <atomic>
#include <cstddef>

namespace coe::md {

/// Single-Producer / Single-Consumer lock-free ring buffer.
///
/// Template parameters
/// -------------------
/// T  – element type; must be trivially copyable for strongest perf guarantees
///      (non-trivial types are supported but copy/move may be expensive).
/// N  – buffer capacity; MUST be a power of two.
///
/// Memory-ordering contract
/// ------------------------
/// try_push
///   • head load  : relaxed  (only producer reads head)
///   • tail load  : acquire  (synchronise with consumer's tail release)
///   • head store : release  (publish the new element to the consumer)
///
/// try_pop
///   • tail load  : relaxed  (only consumer reads tail)
///   • head load  : acquire  (synchronise with producer's head release)
///   • tail store : release  (return the slot to the producer)
///
/// The head and tail counters are placed on separate cache lines (64-byte
/// aligned) to eliminate false sharing between the two threads.
template <typename T, std::size_t N>
class SPSCRingBuffer {
    static_assert((N & (N - 1)) == 0,
                  "SPSCRingBuffer: N must be a power of two");

public:
    SPSCRingBuffer()  = default;
    ~SPSCRingBuffer() = default;

    // Non-copyable, non-movable – the atomics cannot be relocated safely.
    SPSCRingBuffer(const SPSCRingBuffer&)            = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer(SPSCRingBuffer&&)                 = delete;
    SPSCRingBuffer& operator=(SPSCRingBuffer&&)      = delete;

    // ── Producer interface ─────────────────────────────────────────────────

    /// Attempt to enqueue a copy of @p item.
    /// Returns true on success, false when the buffer is full.
    [[nodiscard]] bool try_push(const T& item) noexcept(noexcept(T(item))) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t next_h = (h + 1) & kMask;

        // Full when the next write position would overwrite the tail slot.
        if (next_h == tail_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[h] = item;
        head_.store(next_h, std::memory_order_release);
        return true;
    }

    // ── Consumer interface ─────────────────────────────────────────────────

    /// Attempt to dequeue into @p item.
    /// Returns true on success, false when the buffer is empty.
    [[nodiscard]] bool try_pop(T& item) noexcept(noexcept(T(std::declval<T>()))) {
        const std::size_t t = tail_.load(std::memory_order_relaxed);

        // Empty when tail has caught up to head.
        if (t == head_.load(std::memory_order_acquire)) {
            return false;
        }

        item = buffer_[t];
        tail_.store((t + 1) & kMask, std::memory_order_release);
        return true;
    }

    // ── Observers ─────────────────────────────────────────────────────────

    /// Approximate number of elements currently enqueued.
    /// The value may be stale by the time it is used; treat as a hint.
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        // Works for both the normal case and the wrapped case because N is a
        // power of two and we are working with unsigned arithmetic.
        return (h - t) & kMask;
    }

    /// Returns true when the ring buffer contains no elements.
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    /// Compile-time capacity.
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }

private:
    static constexpr std::size_t kMask = N - 1;

    // Each counter lives on its own 64-byte cache line.
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};

    // Storage lives between the two counters in address space but on a fresh
    // cache line thanks to the alignas on head_.
    std::array<T, N> buffer_{};
};

} // namespace coe::md
