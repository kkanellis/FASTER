
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <mutex>
#include <vector>

#include <immintrin.h>

#include "../common/log.h"

class BlockedBloomFilter {
 public:
  BlockedBloomFilter()
    : ready_{ false }
    , build_time_{ 0 } {
  }

  void Init(uint32_t num_elements, double epsilon = 1e-2) {

    ready_.store(false);
    num_elements_ = num_elements;
    epsilon_ = epsilon;

    num_bits_ = ComputeNumBits();
    num_blocks_ = ComputeNumBlocks();
    num_hash_fns_ = ComputeNumHashFns();

    log_debug("Initializing filter for %'lu elements [FPR=%.1lf%%]",
              num_elements_, epsilon_ * 100);
    log_debug("[# Bits: %lu, # Blocks: %lu, # Hash Fns: %lu]",
              num_bits_, num_blocks_, num_hash_fns_);

    //bv.resize(num_blocks_);
    bv.resize((num_bits_ + 7) / 8);
    std::fill(bv.begin(), bv.end(), 0);
  }

  void Insert(uint64_t h) {
    std::lock_guard<std::mutex> lk{ mtx };

    uint32_t h1 = h & ((1ull << 32) - 1);
    uint32_t h2 = h >> 32;

    for(int i = 0; i < num_hash_fns_; i++) {
      uint32_t hash = (h1 + i * h2) % num_bits_;
      uint64_t bit_idx = hash % 8;
      uint64_t byte_idx = hash / 8;
      bv[byte_idx] |= (1 << bit_idx);
    }
  }

  bool Query(uint64_t h) {
    uint32_t h1 = h & ((1ull << 32) - 1);
    uint32_t h2 = h >> 32;

    bool result = true;
    for(int i = 0; i < num_hash_fns_; i++) {
      uint32_t hash = (h1 + i * h2) % num_bits_;
      uint64_t bit_idx = hash % 8;
      uint64_t byte_idx = hash / 8;
      result &= (bv[byte_idx] >> bit_idx) & 1;
    }
    return result;
  }

  inline uint64_t Size() {
    return bv.size() * sizeof(uint64_t);
  }

  inline bool IsReady() {
    return ready_.load();
  }

  inline void Seal() {
    //log_debug("Filter build time: %lu ms", std::chrono::duration_cast<std::chrono::milliseconds>(
    //  std::chrono::high_resolution_clock::now() - build_start));
    log_debug("Filter build time: %lu ms", std::chrono::duration_cast<std::chrono::milliseconds>(build_time_));
    ready_.store(true);
  }

 private:
  uint64_t ComputeNumBits() {
    double bits_per_val = -1.44 * std::log2(epsilon_);
    return static_cast<uint64_t>(bits_per_val * num_elements_ + 0.5);
  }

  uint32_t ComputeNumHashFns() {
    return static_cast<uint32_t>(-std::log2(epsilon_) + 0.5);
  }

  uint32_t ComputeNumBlocks() {
    uint32_t log_num_blocks = 32 - __builtin_clz(num_bits_) - 6;
    return (1 << log_num_blocks);
  }

  void GetBlockIdx(__m256i &vecBlockIdx, __m256i &vecH1, __m256i &vecH2) {
    __m256i vecNumBlocksMask = _mm256_set1_epi64x(num_blocks_ - 1);
    vecBlockIdx = _mm256_and_si256(vecH1, vecNumBlocksMask);
  }

  void ConstructMask(__m256i &vecMask, __m256i &vecH1, __m256i &vecH2) {
    __m256i vecShiftMask = _mm256_set1_epi64x((1 << 6) - 1);
    __m256i vecOnes = _mm256_set1_epi64x(1);
    for (int i = 1; i < num_hash_fns_; i++) {
      __m256i vecI = _mm256_set1_epi64x(i);
      __m256i vecMulH2 = _mm256_mul_epi32(vecI, vecH2);
      __m256i vecHash = _mm256_add_epi64(vecH1, vecMulH2);
      __m256i vecShift = _mm256_and_si256(vecHash, vecShiftMask);
      __m256i vecPartial = _mm256_sllv_epi64(vecOnes, vecShift);
      vecMask = _mm256_or_si256(vecMask, vecPartial);
    }
  }

 private:
  std::atomic<bool> ready_;
  uint32_t num_elements_;
  double epsilon_;
  std::chrono::nanoseconds build_time_;
  std::mutex mtx;

  uint64_t num_bits_;
  uint64_t num_blocks_;
  uint32_t num_hash_fns_;
  //std::vector<uint64_t> bv;
  std::vector<uint8_t> bv;
};