#pragma once

#include <atomic>
#include <vector>

#include "../common/log.h"
#include "../core/address.h"

namespace FASTER {
namespace agent {

class Histogram {
 public:
  static constexpr uint64_t kPageSize = Address::kMaxOffset + 1; // 32 MiB
  static constexpr uint64_t kMaxMemorySize = 1ULL << 37; // 128 GiB

  Histogram(uint64_t max_memory_size = kMaxMemorySize)
    : max_memory_size_{ max_memory_size }
    , num_pages_{ max_memory_size / kPageSize }
    , version_{ 0 }
  {
    for (uint8_t i = 0; i < 2; i++) {
      histogram_[i].resize(num_pages_);
    }
    cdf_.resize(num_pages_);
    SealAndBump();
  }

  // Add a page distance to the histogram
  void Add(uint64_t page_dist) {
    uint8_t version = version_.load();

    if (page_dist >= num_pages_) {
      ++histogram_[version][num_pages_ - 1];
      log_warn("Histogram::Add(): page_dist (%lu) >= num_pages_ (%lu)", page_dist, num_pages_);
    } else {
      ++histogram_[version][page_dist];
    }
  }

  void SealAndBump() {
    uint8_t next_version = 1 - version_.load();
    for (uint64_t i = 0; i < num_pages_; i++) {
      histogram_[next_version][i] = 0;
    }
    // Bump version
    version_ = 1 - version_.load();
  }

  std::vector<uint64_t>& ComputeCDF() {
    uint8_t old_version = 1 - version_.load();
    cdf_[0] = histogram_[old_version][0];
    for (uint64_t i = 1; i < num_pages_; i++) {
      cdf_[i] = cdf_[i - 1] + histogram_[old_version][i];
    }

    return cdf_;
  }

  void PrintCDF(const std::string& name = "") {
    uint8_t old_version = 1 - version_.load();
    log_rep("CDF Histogram: %s", name.c_str());
    log_rep("==============");
    for (uint64_t i = 0; i < 128; i++) {
      log_rep("[%lu]: %lu", i, cdf_[i]);
    }
  }

 private:
  const uint64_t max_memory_size_;
  const uint64_t num_pages_;

  std::atomic<uint8_t> version_;
  std::vector<uint64_t> histogram_[2];
  std::vector<uint64_t> cdf_;
};

}
} // namespace FASTER::agent
