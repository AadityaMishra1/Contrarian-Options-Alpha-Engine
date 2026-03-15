#include <coe/market_data/ring_buffer.hpp>
#include <coe/market_data/message_types.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace coe::md;
using Clock = std::chrono::high_resolution_clock;

template<size_t N>
void bench_throughput(size_t total_messages) {
    SPSCRingBuffer<int, N> buf;
    
    auto start = Clock::now();
    
    std::jthread producer([&](std::stop_token) {
        for (size_t i = 0; i < total_messages; ++i) {
            while (!buf.try_push(static_cast<int>(i))) {
                // spin
            }
        }
    });
    
    size_t received = 0;
    int val = 0;
    while (received < total_messages) {
        if (buf.try_pop(val)) {
            ++received;
        }
    }
    
    producer.join();
    auto end = Clock::now();
    
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double elapsed_s = static_cast<double>(elapsed_ns) / 1e9;
    double msgs_per_sec = static_cast<double>(total_messages) / elapsed_s;
    
    std::cout << "  Buffer size: " << N 
              << " | Messages: " << total_messages
              << " | Time: " << elapsed_s << "s"
              << " | Throughput: " << msgs_per_sec / 1e6 << "M msg/s\n";
}

void bench_latency() {
    constexpr size_t N = 65536;
    constexpr size_t SAMPLES = 100000;
    SPSCRingBuffer<int64_t, N> buf;
    std::vector<int64_t> latencies;
    latencies.reserve(SAMPLES);
    
    std::jthread producer([&](std::stop_token) {
        for (size_t i = 0; i < SAMPLES; ++i) {
            auto now = Clock::now().time_since_epoch().count();
            while (!buf.try_push(now)) {}
        }
    });
    
    size_t received = 0;
    int64_t send_time = 0;
    while (received < SAMPLES) {
        if (buf.try_pop(send_time)) {
            auto now = Clock::now().time_since_epoch().count();
            latencies.push_back(now - send_time);
            ++received;
        }
    }
    
    producer.join();
    
    std::sort(latencies.begin(), latencies.end());
    
    auto percentile = [&](double p) -> int64_t {
        size_t idx = static_cast<size_t>(p * static_cast<double>(latencies.size()));
        if (idx >= latencies.size()) idx = latencies.size() - 1;
        return latencies[idx];
    };
    
    double mean = static_cast<double>(std::accumulate(latencies.begin(), latencies.end(), 0LL)) 
                  / static_cast<double>(latencies.size());
    
    std::cout << "  Latency (ns) - Mean: " << mean
              << " | p50: " << percentile(0.50)
              << " | p95: " << percentile(0.95)
              << " | p99: " << percentile(0.99)
              << " | p99.9: " << percentile(0.999) << "\n";
}

int main() {
    std::cout << "=== SPSC Ring Buffer Benchmark ===\n\n";
    
    std::cout << "Throughput:\n";
    bench_throughput<1024>(10'000'000);
    bench_throughput<4096>(10'000'000);
    bench_throughput<65536>(10'000'000);
    
    std::cout << "\nLatency:\n";
    bench_latency();
    
    std::cout << "\nDone.\n";
    return 0;
}
