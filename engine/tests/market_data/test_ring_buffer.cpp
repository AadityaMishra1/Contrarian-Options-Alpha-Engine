#include <gtest/gtest.h>
#include <coe/market_data/ring_buffer.hpp>

#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

using coe::md::SPSCRingBuffer;

// ── Capacity ──────────────────────────────────────────────────────────────────

TEST(SPSCRingBuffer, CompileTimeCapacity) {
    static_assert(SPSCRingBuffer<int, 1024>::capacity() == 1024u);
}

// ── Empty buffer ──────────────────────────────────────────────────────────────

TEST(SPSCRingBuffer, DefaultConstructedIsEmpty) {
    SPSCRingBuffer<int, 1024> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
}

TEST(SPSCRingBuffer, PopFromEmptyReturnsFalse) {
    SPSCRingBuffer<int, 1024> buf;
    int value{};
    EXPECT_FALSE(buf.try_pop(value));
}

// ── Single push/pop ───────────────────────────────────────────────────────────

TEST(SPSCRingBuffer, PushSingleElement) {
    SPSCRingBuffer<int, 1024> buf;
    EXPECT_TRUE(buf.try_push(42));
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.size(), 1u);
}

TEST(SPSCRingBuffer, PopSingleElement) {
    SPSCRingBuffer<int, 1024> buf;
    buf.try_push(99);
    int out{};
    EXPECT_TRUE(buf.try_pop(out));
    EXPECT_EQ(out, 99);
    EXPECT_TRUE(buf.empty());
}

TEST(SPSCRingBuffer, PopRestoresEmpty) {
    SPSCRingBuffer<int, 1024> buf;
    buf.try_push(1);
    int out{};
    buf.try_pop(out);
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
}

// ── FIFO ordering ─────────────────────────────────────────────────────────────

TEST(SPSCRingBuffer, FifoOrdering) {
    SPSCRingBuffer<int, 16> buf;
    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(buf.try_push(i));
    }
    for (int i = 0; i < 10; ++i) {
        int out{};
        ASSERT_TRUE(buf.try_pop(out));
        EXPECT_EQ(out, i);
    }
}

// ── Capacity boundary: N-1 usable slots ──────────────────────────────────────
// The SPSC ring buffer uses N-1 slots to distinguish full from empty.

TEST(SPSCRingBuffer, FillToCapacityMinus1) {
    // N=16 means 15 usable slots.
    SPSCRingBuffer<int, 16> buf;
    int filled = 0;
    for (int i = 0; i < 16; ++i) {
        if (buf.try_push(i)) {
            ++filled;
        } else {
            break;
        }
    }
    // Should have pushed exactly N-1 = 15 items.
    EXPECT_EQ(filled, 15);
    // Next push must fail — buffer is full.
    EXPECT_FALSE(buf.try_push(999));
}

TEST(SPSCRingBuffer, AfterFillPopAllThenPushAgain) {
    SPSCRingBuffer<int, 16> buf;
    // Fill.
    for (int i = 0; i < 15; ++i) { buf.try_push(i); }
    // Drain.
    int out{};
    for (int i = 0; i < 15; ++i) { buf.try_pop(out); }
    EXPECT_TRUE(buf.empty());
    // Should be pushable again.
    EXPECT_TRUE(buf.try_push(42));
}

// ── Interleaved push/pop ──────────────────────────────────────────────────────

TEST(SPSCRingBuffer, InterleavedPushPop) {
    SPSCRingBuffer<int, 8> buf;
    // Push 3, pop 2, push 3, pop 4 — verify FIFO across wrap boundary.
    buf.try_push(10);
    buf.try_push(20);
    buf.try_push(30);
    int a{}, b{};
    buf.try_pop(a);
    buf.try_pop(b);
    EXPECT_EQ(a, 10);
    EXPECT_EQ(b, 20);
    buf.try_push(40);
    buf.try_push(50);
    buf.try_push(60);
    int vals[4]{};
    for (auto& v : vals) { buf.try_pop(v); }
    EXPECT_EQ(vals[0], 30);
    EXPECT_EQ(vals[1], 40);
    EXPECT_EQ(vals[2], 50);
    EXPECT_EQ(vals[3], 60);
}

// ── Multi-threaded: producer pushes N items, consumer pops N items ─────────────

TEST(SPSCRingBuffer, MultiThreadedProducerConsumer) {
    constexpr int kCount = 100'000;
    SPSCRingBuffer<int, 4096> buf;

    // Producer.
    std::thread producer([&] {
        for (int i = 0; i < kCount; ++i) {
            while (!buf.try_push(i)) { /* spin */ }
        }
    });

    // Consumer collects into a vector.
    std::vector<int> received;
    received.reserve(kCount);

    std::thread consumer([&] {
        int out{};
        while (static_cast<int>(received.size()) < kCount) {
            if (buf.try_pop(out)) {
                received.push_back(out);
            }
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(static_cast<int>(received.size()), kCount);

    // Verify all received in order: 0, 1, 2, ..., kCount-1.
    for (int i = 0; i < kCount; ++i) {
        EXPECT_EQ(received[i], i) << "Mismatch at index " << i;
    }
}
