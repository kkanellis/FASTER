#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "histogram.h"
#include "filter.h"

#include "../core/address.h"
#include "../core/light_epoch.h"
#include "../core/status.h"
#include "../core/thread.h"

namespace FASTER {
namespace agent {

using namespace FASTER::core;

enum class MemoryComponent : uint8_t {
	HLOG_WRITE = 0,
  HLOG_READ = 1,
	READ_CACHE = 2,

	NUM
};

struct MemoryAllocation {
	// TODO: add hash index
	uint64_t hlog_budget;
	uint64_t rc_budget;

  void Print() {
    log_rep("Memory Allocation: [hlog: %.3lf GiB, rc: %.3lf GiB]",
             static_cast<double>(hlog_budget) / (1 << 30ULL),
             static_cast<double>(rc_budget) / (1 << 30ULL));
  }
  int64_t Dist(const MemoryAllocation& other) {
    return std::abs(static_cast<int64_t>(hlog_budget) - static_cast<int64_t>(other.hlog_budget)) +
           std::abs(static_cast<int64_t>(rc_budget) - static_cast<int64_t>(other.rc_budget));
  }
  bool operator==(const MemoryAllocation& other) {
    return (hlog_budget == other.hlog_budget) && (rc_budget == other.rc_budget);
  }
};

struct CandidateMemoryAllocation {
  MemoryAllocation alloc;
  uint64_t total_io_ops;
};


class MemoryAgent {
 public:
	constexpr static uint64_t kSamplingInterval = 1024;
  constexpr static uint32_t kMaxNumInMemPages = 4096; // (4096 pages -> 128 GiB)
  constexpr static uint32_t kMaxNumFilters = 100;
  constexpr static uint8_t kNumMemoryComponents = static_cast<uint8_t>(MemoryComponent::NUM);

  constexpr static uint8_t kHlogWriteComponent = static_cast<uint8_t>(MemoryComponent::HLOG_WRITE);
  constexpr static uint8_t kHlogReadComponent = static_cast<uint8_t>(MemoryComponent::HLOG_READ);
  constexpr static uint8_t kReadCacheComponent = static_cast<uint8_t>(MemoryComponent::READ_CACHE);

  constexpr static uint64_t kMinAddress = Constants::kCacheLineBytes;

  // Candidate allocations better than current, with expected improvement (%)
  // *less than* this, would not be applied
  constexpr static uint8_t kMinExpectedPctImprovement = 2;

  MemoryAgent(LightEpoch& epoch, MemoryAllocation& curr_allocation, uint16_t num_allocations = 2)
    : epoch_{ epoch }
		, curr_allocation_{ curr_allocation }
    , prev_tail_address_page_{ 0 } {

		std::memset(op_counter_, 0, kNumMemoryComponents * Thread::kMaxNumThreads * sizeof(uint64_t));
    rc_filters_ = std::make_unique<BlockedBloomFilter[]>(kMaxNumFilters);
	}

  inline bool UpdateAllocation(uint32_t tail_address_page) {
    curr_allocation_.Print();
    log_info("Checking for better memory allocation...");

    // Seal histograms
    for (uint8_t component = 0; component < kNumMemoryComponents; component++) {
      histograms_[component].SealAndBump();
    }

    // Compute expected hit-rate for hlog-reads based on hlog page insertion rate
    std::vector<uint64_t> hlog_read_cdf = histograms_[kHlogReadComponent].ComputeCDF();
    u_int64_t hlog_expected_new_inserted_pages = tail_address_page - prev_tail_address_page_;
    prev_tail_address_page_ = tail_address_page; // update prev_tail_address_page_ for next iteration
    histograms_[kHlogReadComponent].PrintCDF("HLOG Read CDF");

    // Compute expected hit-rate for hlog-writes and read-cache
    std::vector<uint64_t> hlog_write_cdf = histograms_[kHlogWriteComponent].ComputeCDF();
    std::vector<uint64_t> read_cache_cdf = histograms_[kReadCacheComponent].ComputeCDF();
    histograms_[kHlogWriteComponent].PrintCDF("HLOG Write CDF");
    histograms_[kReadCacheComponent].PrintCDF("Read Cache CDF");

    // Compute best allocation
    uint64_t mem_budget = curr_allocation_.hlog_budget + curr_allocation_.rc_budget;
    uint32_t max_pages = mem_budget / Histogram::kPageSize;

    MemoryAllocation best_allocation;
    uint64_t best_total_hits = 0;

    for (uint32_t hlog_num_pages = 0; hlog_num_pages < max_pages; hlog_num_pages++) {
      uint32_t rc_num_pages = max_pages - hlog_num_pages - 1;

      uint64_t hlog_expected_hits = hlog_write_cdf[hlog_num_pages];
      uint64_t rc_expected_hits = read_cache_cdf[rc_num_pages];

      uint64_t candidate_total_hits = hlog_expected_hits + rc_expected_hits;
      candidate_total_hits += (hlog_num_pages < hlog_expected_new_inserted_pages)
                                ? hlog_read_cdf[hlog_num_pages]
                                : hlog_read_cdf[hlog_expected_new_inserted_pages];

      log_rep("HLOG: %lu | RC: %lu | Total Hits: %lu", hlog_num_pages, rc_num_pages, candidate_total_hits);

      MemoryAllocation candidate_allocation = MemoryAllocation{ hlog_num_pages * Histogram::kPageSize,
                                                                rc_num_pages * Histogram::kPageSize };
      bool found_better = (candidate_total_hits > best_total_hits) ||
                          ((candidate_total_hits == best_total_hits) // break ties by choosing closest to current allocation
                            && (candidate_allocation.Dist(curr_allocation_) < best_allocation.Dist(curr_allocation_)));

      if (candidate_total_hits > best_total_hits) {
        log_rep("Found better allocation: HLOG: %lu | RC: %lu | Total Hits: %lu",
                hlog_num_pages, rc_num_pages, candidate_total_hits);
        best_total_hits = candidate_total_hits;
        best_allocation = candidate_allocation;
      }
    }
    log_rep("Best Allocation:");
    best_allocation.Print();

    curr_allocation_ = best_allocation;

    return true;
  }

	inline void ReportUpsertHlogOp(const uint32_t record_address_page, const uint32_t tail_address_page) {
		auto& counter = op_counter_[kHlogWriteComponent][Thread::id()];
		if (++counter % kSamplingInterval != 0) {
			return;
		}
    assert(record_address_page <= tail_address_page);
    histograms_[kHlogWriteComponent].Add(tail_address_page - record_address_page);
	}

	inline void ReportReadHlogOp(const uint32_t record_address_page, const uint32_t tail_address_page) {
		auto& counter = op_counter_[kHlogReadComponent][Thread::id()];
		if (++counter % kSamplingInterval != 0) {
			return;
		}
    assert(record_address_page <= tail_address_page);
    histograms_[kHlogReadComponent].Add(tail_address_page - record_address_page);
	}

  inline BlockedBloomFilter& GetFilter(uint32_t page) {
    return rc_filters_[page % kMaxNumFilters];
  }

	inline void ReportReadCacheOp(const int32_t record_address_page,
                                const uint32_t tail_address_page,
                                uint64_t key_hash, bool from_cold_log) {
		auto& counter = op_counter_[kReadCacheComponent][Thread::id()];
		if (++counter % kSamplingInterval != 0) {
			return;
		}
    // record_address_page is either:
    //  >=0: an in-memory record with a matching key (i.e., address >= head_address), or
    // 	 -1: if no such record exists in read-cache (i.e., cache-miss)

    // TODO: Do something with cold-log flag!
    //       Maybe instead of countring mem-access, count I/O ops?

    if (record_address_page >= 0) {
      // cache-hit
      uint64_t page_dist = tail_address_page - record_address_page;
      histograms_[kReadCacheComponent].Add(page_dist);
      return;
    }

    // cache-miss
    uint32_t head_address_page = tail_address_page - (curr_allocation_.rc_budget / Histogram::kPageSize);
    uint32_t from_page = std::max(static_cast<uint32_t>(0), head_address_page - kMaxNumFilters);

    for (uint32_t evicted_page = head_address_page; evicted_page >= from_page; evicted_page--) {
      auto& f = GetFilter(evicted_page);
      if (!f.IsReady()) {
        continue;
      }

      if (f.Query(key_hash)) {
        // cache-hit (expected)
        uint64_t page_dist = tail_address_page - evicted_page;
        histograms_[kReadCacheComponent].Add(page_dist);
        return;
      }
    }
	}

 private:
  class OnVersionBumpedContext : public IAsyncContext {
   public:
    OnVersionBumpedContext(MemoryAgent* agent_, uint8_t old_version_)
      : agent{ agent_ }
      , old_version{ old_version_ } {
    }
    /// The deep-copy constructor.
    OnVersionBumpedContext(const OnVersionBumpedContext& other)
      : agent{ other.agent }
      , old_version{ other.old_version } {
    }

   protected:
    Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   public:
    MemoryAgent* agent;
    uint8_t old_version;
  };

 private:

  LightEpoch& epoch_;
  MemoryAllocation& curr_allocation_;
  uint32_t prev_tail_address_page_;

  Histogram histograms_[kNumMemoryComponents];

  uint64_t op_counter_[kNumMemoryComponents][Thread::kMaxNumThreads];
  std::unique_ptr<BlockedBloomFilter[]> rc_filters_;
};

}
}
