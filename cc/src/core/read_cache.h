// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>

#include "address.h"
#include "internal_contexts.h"
#include "light_epoch.h"
#include "persistent_memory_malloc.h"
#include "read_cache_utils.h"
#include "record.h"
#include "status.h"

#include "agents/memory.h"
#include "device/null_disk.h"

using namespace FASTER::agent;

namespace FASTER {
namespace core {

// Read Cache that stores read-hot records in memory
// NOTE: Current implementation stores *at most one* record per hash index entry
template <class K, class V, class D, class H>
class ReadCache {
 public:
  typedef K key_t;
  typedef V value_t;
  typedef Record<key_t, value_t> record_t;

  typedef D faster_disk_t;
  typedef typename D::file_t file_t;
  typedef PersistentMemoryMalloc<faster_disk_t> faster_hlog_t;

  typedef H hash_index_t;
  typedef typename H::hash_bucket_t hash_bucket_t;
  typedef typename H::hash_bucket_entry_t hash_bucket_entry_t;

  // Read cache allocator
  typedef FASTER::device::NullDisk disk_t;
  typedef ReadCachePersistentMemoryMalloc<disk_t> hlog_t;

  constexpr static uint32_t kEvictChunkSize = 32;
  static_assert((kEvictChunkSize * sizeof(Address)) % Constants::kCacheLineBytes == 0,
                "kEvictChunkSize * sizeof(Address) must be a multiple of kCacheLineBytes");

  constexpr static bool CopyToTail = true;

  ReadCache(LightEpoch& epoch, hash_index_t& hash_index, faster_hlog_t& faster_hlog,
            ReadCacheConfig& config, MemoryAgent* memory_agent)
    : epoch_{ epoch }
    , hash_index_{ &hash_index }
    , disk_{ "", epoch, "" }
    , faster_hlog_{ &faster_hlog }
    , memory_agent_{ memory_agent }
    , read_cache_{ true, config.mem_size, epoch, disk_, disk_.log(),
                   config.mutable_fraction, config.pre_allocate, DoEvictCallback}
    , evicted_to_{ Constants::kCacheLineBytes }     // First page starts with kCacheLineBytes offset
    , evict_to_req_{ Constants::kCacheLineBytes } // First page starts with kCacheLineBytes offset
    , evicting_to_{ 0 }
    , evict_reordering_threads_{ 0 }
    , evict_record_addrs_idx_{ 0 } {

    // hash index should be entirely in memory
    assert(hash_index_->IsSync());
    // required when evict callback is called
    read_cache_.SetReadCacheInstance(static_cast<void*>(this));
  }

  template <class C>
  Status Read(C& pending_context, Address& address);

  template <class C>
  Address Skip(C& pending_context) const;

  Address Skip(Address address);

  template <class C>
  Address Skip(C& pending_context);

  template <class C>
  Address SkipAndInvalidate(const C& pending_context);

  template <class C>
  Status TryInsert(ExecutionContext& exec_context, C& pending_context,
                   record_t* record, bool is_cold_log_record = false);

  // Eviction-related methods
  // Called from ReadCachePersistentMemoryMalloc, when head address is shifted
  static Address DoEvictCallback(void* readcache, Address from_head_address, Address to_head_address) {
    typedef ReadCache<K, V, D, H> readcache_t;
    readcache_t* self = static_cast<readcache_t*>(readcache);
    return self->EvictMT(from_head_address, to_head_address);
  }

  // Multi-threaded eviction
  Address EvictMT(Address from_head_address, Address to_head_address);
  // Single-threaded eviction
  void EvictST(Address from_head_address, Address to_head_address);

  // Checkpoint-related methods
  Status Checkpoint(CheckpointState<file_t>& checkpoint);
  void SkipBucket(hash_bucket_t* const bucket) const;

 private:
  // Eviction-related methods
  void PrepareEvictPage(Address from_address, Address to_address);
  void EvictRecord(const Address address, record_t* record, const bool mt);

 private:
  LightEpoch& epoch_;
  hash_index_t* hash_index_;

  disk_t disk_;
  faster_hlog_t* faster_hlog_;

  MemoryAgent* memory_agent_;

 public:
  hlog_t read_cache_;

 private:
  /// Evicted-related members
  enum class PageEvictStatus : uint8_t {
    Ready = 0,
    Preparing,
    Reordering,
    Finalizing,
  };

  class AtomicPageEvictStatus {
   public:
    AtomicPageEvictStatus()
      : control_{ 0 } {
    }

    inline void store(PageEvictStatus status) {
      control_.store(static_cast<uint8_t>(status));
    }

    inline PageEvictStatus load() const {
      return static_cast<PageEvictStatus>(control_.load());
    }

    inline bool compare_exchange_strong(PageEvictStatus& expected, PageEvictStatus value) {
      uint8_t expected_control = static_cast<uint8_t>(expected);
      bool result = control_.compare_exchange_strong(expected_control, static_cast<uint8_t>(value));
      expected = static_cast<PageEvictStatus>(expected_control);
      return result;
    }

 private:
    union {
      std::atomic<uint8_t> control_;
    };
  };
  static_assert(sizeof(AtomicPageEvictStatus) == 1, "sizeof(AtomicPageEvictStatus) != 1");

  // Eviction status
  AtomicPageEvictStatus eviction_status_;
  // Used to measure page eviction time
  std::chrono::time_point<std::chrono::high_resolution_clock> evict_start_time_;

  // Largest address for which record eviction has finished
  AtomicAddress evicted_to_{ Constants::kCacheLineBytes };
  // Until address, for which eviction is currently performed
  AtomicAddress evicting_to_{ 0 };
  // Largest address for which *any* thread has requested eviction
  AtomicAddress evict_to_req_{ Constants::kCacheLineBytes };

  // Keeps track of how many threads are participating in the reordering process
  std::atomic<int> evict_reordering_threads_{ 0 };

  // Vector of valid records residing in soon-to-be evicted page
  std::vector<Address> evict_record_addrs_;
  // Points to next-to-be-checked for eviction record
  std::atomic<uint32_t> evict_record_addrs_idx_{ 0 };

#ifdef STATISTICS
 public:
  void EnableStatsCollection() {
    collect_stats_ = true;
  }
  void DisableStatsCollection() {
    collect_stats_ = false;
  }
  // implementation at the end of this file
  void PrintStats() const;

 private:
  bool collect_stats_{ true };
  // Read
  std::atomic<uint64_t> read_calls_{ 0 };
  std::atomic<uint64_t> read_success_count_{ 0 };
  std::atomic<uint64_t> read_copy_to_tail_calls_{ 0 };
  std::atomic<uint64_t> read_copy_to_tail_success_count_{ 0 };
  // TryInsert
  std::atomic<uint64_t> try_insert_calls_{ 0 };
  std::atomic<uint64_t> try_insert_success_count_{ 0 };
  // Evict
  std::atomic<uint64_t> evicted_records_count_{ 0 };
  std::atomic<uint64_t> evicted_records_invalid_{ 0 };
#endif
};

template <class K, class V, class D, class H>
template <class C>
inline Status ReadCache<K, V, D, H>::Read(C& pending_context, Address& address) {
  if (pending_context.skip_read_cache) {
    address = Skip(pending_context);
    return Status::NotFound;
  }

  #ifdef STATISTICS
  if (collect_stats_) {
    ++read_calls_;
  }
  #endif

  uint32_t tail_address_page = read_cache_.GetTailAddress().page();
  int32_t address_page = (address.in_readcache()) ? address.readcache_address().page() : -1;
  uint64_t key_hash = pending_context.get_key_hash().control_;

  if (memory_agent_ != nullptr) {
    // TODO: how to decide on when do we say this was a hot/cold read?
    memory_agent_->ReportReadCacheOp(address_page, tail_address_page, key_hash, false);
  }

  if (address.in_readcache()) {
    Address rc_address = address.readcache_address();
    record_t* record = reinterpret_cast<record_t*>(read_cache_.Get(rc_address));
    ReadCacheRecordInfo rc_record_info{ record->header };

    assert(!rc_record_info.tombstone); // read cache does not store tombstones

    if (!rc_record_info.invalid
        && rc_address >= read_cache_.safe_head_address.load()
        && pending_context.is_key_equal(record->key())
    ) {
      pending_context.Get(record);

      #ifdef STATISTICS
      if (collect_stats_) {
        ++read_success_count_;
      }
      #endif

      if (CopyToTail && rc_address < read_cache_.read_only_address.load()) {
        #ifdef STATISTICS
        if (collect_stats_) {
          ++read_copy_to_tail_calls_;
        }
        #endif

        ExecutionContext exec_context; // dummy context; not actually used
        Status status = TryInsert(exec_context, pending_context, record,
                                  ReadCacheRecordInfo{ record->header }.in_cold_hlog);
        if (status == Status::Ok) {
          // Invalidate this record, since we copied it to the tail
          record->header.invalid = true;

          #ifdef STATISTICS
          if (collect_stats_) {
            ++read_copy_to_tail_success_count_;
          }
          #endif
        }
      }
      return Status::Ok;
    }

    address = rc_record_info.previous_address();
    assert(!address.in_readcache());
  }
  // not handled by read cache
  return Status::NotFound;
}

template <class K, class V, class D, class H>
template <class C>
inline Address ReadCache<K, V, D, H>::Skip(C& pending_context) const {
  Address address = pending_context.entry.address();
  const record_t* record;

  if (address.in_readcache()) {
    assert(address.readcache_address() >= read_cache_.safe_head_address.load());
    record = reinterpret_cast<const record_t*>(read_cache_.Get(address.readcache_address()));
    address = ReadCacheRecordInfo{ record->header }.previous_address();
    assert(!address.in_readcache());
  }
  return address;
}

template <class K, class V, class D, class H>
inline Address ReadCache<K, V, D, H>::Skip(Address address) {
  const record_t* record;

  if (address.in_readcache()) {
    assert(address.readcache_address() >= read_cache_.safe_head_address.load());
    record = reinterpret_cast<const record_t*>(read_cache_.Get(address.readcache_address()));
    address = ReadCacheRecordInfo{ record->header }.previous_address();
    assert(!address.in_readcache());
  }
  return address;
}

template <class K, class V, class D, class H>
template <class C>
inline Address ReadCache<K, V, D, H>::Skip(C& pending_context) {
  Address address = pending_context.entry.address();
  record_t* record;

  if (address.in_readcache()) {
    assert(address.readcache_address() >= read_cache_.safe_head_address.load());
    record = reinterpret_cast<record_t*>(read_cache_.Get(address.readcache_address()));
    address = ReadCacheRecordInfo{ record->header }.previous_address();
    assert(!address.in_readcache());
  }
  return address;
}

template <class K, class V, class D, class H>
template <class C>
inline Address ReadCache<K, V, D, H>::SkipAndInvalidate(const C& pending_context) {
  Address address = pending_context.entry.address();
  record_t* record;

  if (address.in_readcache()) {
    assert(address.readcache_address() >= read_cache_.safe_head_address.load());
    record = reinterpret_cast<record_t*>(read_cache_.Get(address.readcache_address()));
    if (pending_context.is_key_equal(record->key())) {
      // invalidate record if keys match
      record->header.invalid = true;
    }
    address = record->header.previous_address();
    assert(!address.in_readcache());
  }
  return address;
}

template <class K, class V, class D, class H>
template <class C>
inline Status ReadCache<K, V, D, H>::TryInsert(ExecutionContext& exec_context, C& pending_context,
                                               record_t* record, bool is_cold_log_record) {
  #ifdef STATISTICS
  if (collect_stats_) {
    ++try_insert_calls_;
  }
  #endif

  // Store previous info wrt expected hash bucket entry
  HashBucketEntry orig_expected_entry{ pending_context.entry };
  bool invalid_hot_index_entry = (orig_expected_entry == HashBucketEntry::kInvalidEntry);
  // Expected hot index entry cannot be invalid, unless record is cold-resident,
  // otherwise, how did we even ended-up in this record? :)
  assert(is_cold_log_record || !invalid_hot_index_entry);

  Status index_status = hash_index_->FindOrCreateEntry(exec_context, pending_context);
  assert(pending_context.atomic_entry != nullptr);
  assert(index_status == Status::Ok);

  bool created_index_entry = (pending_context.entry.address() == Address::kInvalidAddress);
  // Address should not be invalid, unless record is cold-log-resident OR some deletion op took place
  if (!is_cold_log_record && created_index_entry) {
    return Status::Aborted;
  }

  // Find first non-read cache address, to be used as previous address for this RC record
  Address hlog_address = Skip(pending_context);
  assert(!hlog_address.in_readcache());
  // Again, address should not be invalid, unless record is cold-log-resident OR some deletion op took place
  if (!is_cold_log_record && (hlog_address == Address::kInvalidAddress)) {
    return Status::Aborted;
  }

  // Typically, we use as expected_entry, the original hot index entry we retrieved at the start of FASTER/F2's Read.
  // -----
  // Yet for F2, if this record is cold-resident AND new hot index entry was created, it means that there is no record
  // with the same key in hot log. Thus, we can safely keep this newly-created entry, as our expected entry, because
  // we can guarrantee that no newer record for this key has been inserted in the hot/cold log in the meantime.
  // (i.e., if they were, they would have been found by the cold_store.Read op that called this method).
  // NOTE: There is a slim case where we cannot guarrantee that (i.e., see F2's Read op). If so, we skip inserting to RC.
  if (!is_cold_log_record || !created_index_entry) {
    pending_context.entry = orig_expected_entry;
  }
  // Proactive check on expected entry, before actually allocating space in read-cache log
  if (pending_context.atomic_entry->load() != pending_context.entry) {
    return Status::Aborted;
  }

  // Try Allocate space in the RC log (best-effort)
  uint32_t page;
  Address new_address = read_cache_.Allocate(record->size(), page);
  if (new_address < read_cache_.read_only_address.load()) {
    // No space in this page -- ask for a new one and abort
    assert(new_address == Address::kInvalidAddress);
    // NOTE: Even if we wanted to retry multiple times here, it would be tricky to guarrantee correctness!
    //       Consider the case where we copy a read-cache-resident record that resides in RC's RO region.
    //       If we Allocate() after a successful NewPage(), then an RC eviction process might have been
    //       triggered, potentially invalidating our source record; we would have to duplicate the record...
    read_cache_.NewPage(page);
    return Status::Aborted;
  }

  // Populate new record
  record_t* new_record = reinterpret_cast<record_t*>(read_cache_.Get(new_address));
  memcpy(reinterpret_cast<void*>(new_record), reinterpret_cast<void*>(record), record->size());
  // Overwrite record header
  ReadCacheRecordInfo record_info {
    static_cast<uint16_t>(exec_context.version),
    is_cold_log_record, false, false, hlog_address.control()
  };
  new(new_record) record_t{ record_info.ToRecordInfo() };
  // Increase record count in the page
  read_cache_.PageSize(new_address.page())++;

  // Try to update hash index
  index_status = hash_index_->TryUpdateEntry(exec_context, pending_context, new_address, true);
  assert(index_status == Status::Ok || index_status == Status::Aborted);
  if (index_status == Status::Aborted) {
    new_record->header.invalid = true;
  }

  #ifdef STATISTICS
  if (index_status == Status::Ok) {
    if (collect_stats_) {
      ++try_insert_success_count_;
    }
  }
  #endif

  return index_status;
}

template <class K, class V, class D, class H>
inline void ReadCache<K, V, D, H>::PrepareEvictPage(Address from_address, Address to_address) {
  const uint32_t page_idx = from_address.page();
  log_debug("[PAGE: %u]{tid=%u} Read-Cache START {PREPARE-EVICT}: [%u] -> [%u]",
            page_idx, Thread::id(), evicted_to_.page(), evicting_to_.page());

  const uint32_t records_in_page = read_cache_.PageSize(page_idx).load();
  evict_record_addrs_.clear(); // do not deallocate memory
  evict_record_addrs_.reserve(records_in_page);

  if (records_in_page == 0) {
    log_warn("[PAGE: %u]{tid=%u} No records found! :/", page_idx, Thread::id());
  } else if (memory_agent_ != nullptr) {
    // Initializing filter for this page
    log_debug("[PAGE: %u]{tid=%u} There exist %lu records in page", page_idx, Thread::id(), records_in_page);
    log_debug("[PAGE: %u]{tid=%u} Initializing filter: [%lu] -> [%lu]", page_idx, Thread::id(),
              Address{ page_idx, 0 }.control(), Address{ page_idx, Address::kMaxOffset }.control());
    memory_agent_->GetFilter(page_idx).Init(records_in_page);
  }

  Address address{ from_address };
  uint32_t all_records = 0, valid_records_count = 0;

  auto start = std::chrono::high_resolution_clock::now();
  while (address < to_address) {
    const record_t* record = reinterpret_cast<const record_t*>(read_cache_.Get(address));
    const ReadCacheRecordInfo record_info{ record->header };
    if (record_info.IsNull()) {
      break; // no more records in this page!
    }

    all_records++;
    valid_records_count += (!record_info.invalid);
    evict_record_addrs_.push_back(address);

    if (address.offset() + record->size() > Address::kMaxOffset) {
      break; // no more records in this page!
    }
    address += record->size();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  log_debug("[PAGE: %u]{tid=%u} Found %u valid records in %.3lf ms", page_idx, Thread::id(),
             valid_records_count, static_cast<double>(elapsed.count()) / 1e6);

  #ifdef STATISTICS
  if (collect_stats_) {
    evicted_records_count_ += all_records;
    evicted_records_invalid_ += (all_records - valid_records_count);
  }
  #endif

  if (all_records != records_in_page) {
    log_warn("[PAGE: %u]{tid=%u} Found %u records by iterating, but page size is %u! :/",
             page_idx, Thread::id(), all_records, records_in_page);
  }
  log_debug("[PAGE: %u]{tid=%u} Read-Cache END {PREPARE-EVICT}: [%u] -> [%u]",
            page_idx, Thread::id(), evicted_to_.page(), evicting_to_.page());
}

template <class K, class V, class D, class H>
inline void ReadCache<K, V, D, H>::EvictRecord(const Address address, record_t* record, const bool mt) {
  typedef ReadCacheEvictContext<K, V> rc_evict_context_t;

  const Address faster_hlog_begin_address{ faster_hlog_->begin_address.load() };
  const uint32_t page_idx = address.page();

  const ReadCacheRecordInfo rc_record_info{ record->header };
  // Assume (for now) that only a single entry per hash bucket lies in read cache
  assert(!rc_record_info.previous_address().in_readcache());

  rc_evict_context_t context{ record };

  if (memory_agent_ != nullptr && !rc_record_info.invalid) {
    memory_agent_->GetFilter(page_idx).Insert(context.get_key_hash().control_);
  }

  ExecutionContext exec_context; // dummy context; not actually used
  Status index_status = hash_index_->FindEntry(exec_context, context);
  assert(index_status == Status::Ok || index_status == Status::NotFound);

  // Index status can be NotFound if:
  //  (1) entry was GCed after log trimming,
  //  (2) during insert two threads tried to insert the same record in RC, but
  //      only the first (i.e., later) one succeded, and this was invalidated
  if (index_status == Status::NotFound) {
    // nothing to update -- continue to next record
    assert(context.entry == HashBucketEntry::kInvalidEntry);
    return;
  }
  assert(context.entry != HashBucketEntry::kInvalidEntry);
  assert(context.atomic_entry != nullptr);

  assert(
      // HI entry does not point to RC
      !context.entry.rc_.readcache_
      // HI points to this RC entry, or some later one
      || (!mt && context.entry.address().readcache_address() >= address)
      // HI points to an RC entry, which is not that one
      // MT: it could point to some earlier address that the other thread didn't evict yet
      || (mt)
  );
  if (
    // HI entry does not point to RC
    !context.entry.rc_.readcache_
    // HI points to an RC entry, which is in some larger address
    || (!mt && context.entry.address().readcache_address() > address)
    // HI points to an RC entry, which is not that one
    // MT: it could point to some earlier address that the other thread didn't evict yet
    || (mt && context.entry.address().readcache_address() != address)
  ) {
    return;
  }
  assert(context.entry.address().readcache_address() == address);

  // Hash index entry points to the entry -- need to CAS pointer to actual record in FASTER log
  while (context.atomic_entry
         && context.entry.rc_.readcache_
         && (context.entry.address().readcache_address() == address)) {

    // Minor optimization when RC Evict() runs concurrent to HashIndex GC
    // If a to-be-evicted read cache entry points to a trimmed hlog address, try to elide the bucket entry
    // NOTE: there is still a chance that read-cache entry points to an invalid hlog address (race condition)
    //       this is handled by FASTER's Internal* methods
    Address updated_address = (rc_record_info.previous_address() >= faster_hlog_begin_address)
                                ? rc_record_info.previous_address()
                                : HashBucketEntry::kInvalidEntry;
    assert(!updated_address.in_readcache());

    index_status = hash_index_->TryUpdateEntry(exec_context, context, updated_address);
    if (index_status == Status::Ok) {
      break; // successfully updated hash index entry
    }
    assert(index_status == Status::Aborted);
    // context.entry should reflect current hash index entry -- retry!
  }
  {
    // new entry should point to either HybridLog, or to some later RC entry
    HashBucketEntry new_entry{ context.atomic_entry->load() };
    assert(!new_entry.rc_.readcache_ || new_entry.address().readcache_address() > address);
  }
}

template <class K, class V, class D, class H>
inline Address ReadCache<K, V, D, H>::EvictMT(Address from_head_address, Address to_head_address) {
  // Keep in the largest address to evict until, from all threads
  Address evicted_to_req;
  do {
    evicted_to_req = evict_to_req_.load();
    if (evicted_to_req.control() >= to_head_address.control()) {
      break;
    }
  } while(!evict_to_req_.compare_exchange_strong(evicted_to_req, to_head_address));

  // Find the next page to evict
  assert(from_head_address <= evicted_to_.load() );
  from_head_address = evicted_to_.load();
  to_head_address = evict_to_req_.load();
  assert(from_head_address <= to_head_address);
  if (from_head_address == to_head_address) {
    return Address::kInvalidAddress; // nothing to evict
  }

  PageEvictStatus expected_status = PageEvictStatus::Ready;
  bool should_prepare = eviction_status_.compare_exchange_strong(expected_status, PageEvictStatus::Preparing);
  if (should_prepare) {
    evict_start_time_ = std::chrono::high_resolution_clock::now();

    // Prepare eviction for this page
    Address from = evicted_to_.load();
    uint32_t page_idx = from.page();
    Address to{ page_idx + 1, 0 };
    assert(to <= evict_to_req_.load());
    evicting_to_.store(to);

    PrepareEvictPage(from, to);

    // init multi-threaded record eviction process
    evict_record_addrs_idx_.store(0);
    assert(eviction_status_.load() == PageEvictStatus::Preparing);
    eviction_status_.store(PageEvictStatus::Reordering);

    log_debug("[PAGE: %u]{tid=%u} Read-Cache START {REORDER}: [%u] -> [%u]",
              page_idx, Thread::id(), evicted_to_.page(), evicting_to_.page());
  }
  else if (expected_status != PageEvictStatus::Reordering) {
    // Another thread is either preparing or finalizing eviction -- abort
    return Address::kInvalidAddress;
  }

  // Participate in record eviction
  ++evict_reordering_threads_;

  // Each thread processes a single chunk of records (i.e., kEvictChunkSize records)
  uint32_t start_idx = evict_record_addrs_idx_.fetch_add(kEvictChunkSize);
  uint32_t until_idx = start_idx + kEvictChunkSize;
  for (uint32_t idx = start_idx; idx < until_idx; ++idx) {
    if (idx >= evict_record_addrs_.size()) {
      // No more records to evict -- finalize eviction
      PageEvictStatus expected_status = PageEvictStatus::Reordering;
      bool won_cas = eviction_status_.compare_exchange_strong(expected_status, PageEvictStatus::Finalizing);
      if (won_cas) {
        log_debug("[PAGE: %lu]{tid=%u} Read-Cache END {REORDER}: [%u] -> [%u]",
                   evicted_to_.page(), Thread::id(), evicted_to_.page(), evicting_to_.page());
      }
      break;
    }

    Address address{ evict_record_addrs_[idx] };
    record_t* record = reinterpret_cast<record_t*>(read_cache_.Get(address));
    EvictRecord(address, record, true);
  }

  int remaining_threads = --evict_reordering_threads_;
  if (remaining_threads > 0 || eviction_status_.load() != PageEvictStatus::Finalizing) {
    // Wait for other threads to finish eviction
    return Address::kInvalidAddress;
  }

  // Finalizing phase (only one thread should reach this point)
  uint32_t page_idx = evicted_to_.page();
  log_debug("[PAGE: %u]{tid=%u} Read-Cache START {FINALIZE}: [%u] -> [%u]",
            page_idx, Thread::id(), evicted_to_.page(), evicting_to_.page());

  uint64_t records_in_page = read_cache_.PageSize(page_idx).load();
  if (records_in_page == 0) {
    log_warn("[PAGE: %u]{tid=%u} No records found! :/", page_idx, Thread::id());
  } else if (memory_agent_ != nullptr) {
    // Seal filters
    log_debug("[PAGE: %u]{tid=%u} Sealing filter...", page_idx, Thread::id());
    memory_agent_->GetFilter(page_idx).Seal();
  }

  log_debug("[PAGE: %u]{tid=%u} Read-Cache END {FINALIZE}: [%u] -> [%u]",
            page_idx, Thread::id(), from_head_address.page(), evicting_to_.load().page());

  // Compute page eviction time
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - evict_start_time_);
  log_debug("[PAGE: %u] ReadCache EVICT: DONE! Total time: %.3lf ms",
            page_idx, static_cast<double>(elapsed.count()) / 1e6);

  // Update evicted-to address
  Address new_evicted_to{ evicting_to_.load() };
  evicted_to_.store(new_evicted_to);
  log_debug("[PAGE: %u]{tid=%u} Evicted to: [%u]", page_idx, Thread::id(), new_evicted_to.page());
  assert(eviction_status_.load() == PageEvictStatus::Finalizing);
  eviction_status_.store(PageEvictStatus::Ready);

  return new_evicted_to; // return the new evicted-to address
}

/*
template <class K, class V, class D, class H>
inline void ReadCache<K, V, D, H>::EvictST(Address from_head_address, Address to_head_address) {
  auto start_total = std::chrono::high_resolution_clock::now();

  assert(to_head_address.offset() == 0);
  for (uint32_t page_idx = from_head_address.page(); page_idx < to_head_address.page(); ++page_idx) {
      auto start_page = std::chrono::high_resolution_clock::now();
      std::chrono::nanoseconds elapsed_filter{ 0 };

    log_rep("[PAGE: %u]{tid=%u} Evicting records...", page_idx, Thread::id());

    uint64_t address_ = Address{ page_idx, 0 }.control();
    address_ = std::min(std::max(from_head_address.control(), address_), to_head_address.control());
    Address address{ address_ };

    uint64_t until_address_ = Address{page_idx + 1, 0}.control();
    until_address_ = std::min(to_head_address.control(), until_address_);
    const Address until_address{ until_address_ };

    log_debug("[PAGE: %u] from: %lu [offset: %lu]\t| to: %lu [offset: %lu]", page_idx,
              address_, address.offset(), until_address_, until_address.offset());

    uint32_t records_in_page = read_cache_.PageSize(page_idx).load();
    if (records_in_page == 0) {
      log_warn("[PAGE: %lu] No records found! :/", page_idx);
      continue;
    } else if (memory_agent_ != nullptr) {
      // Initializing filter for this page
      log_debug("[PAGE: %lu]: Found %lu records in page", page_idx, records_in_page);
      log_debug("[PAGE: %lu]: Initializing filter: [%lu] -> [%lu]", page_idx,
                Address{ page_idx, 0 }.control(), Address{ page_idx, Address::kMaxOffset }.control());
      memory_agent_->GetFilter(page_idx).Init(records_in_page);
    }

    uint32_t invalid_records_count = 0;
    while (address < until_address) {
      record_t* record = reinterpret_cast<record_t*>(read_cache_.Get(address));
      const ReadCacheRecordInfo rc_record_info{ record->header };
      if (rc_record_info.IsNull()) {
        // no more records in this page!
        break;
      }
      invalid_records_count += rc_record_info.invalid;

      EvictRecord(address, record);

      if (address.offset() + record->size() > Address::kMaxOffset) {
        break; // no more records in this page!
      }
      address += record->size();
    }

    #ifdef STATISTICS
    if (collect_stats_) {
      evicted_records_count_ += records_in_page;
      evicted_records_invalid_ += invalid_records_count;
    }
    #endif

    if (memory_agent_ != nullptr) {
      // Seal filter for this page
      if (records_in_page > 0) {
        log_debug("[PAGE: %lu] Sealing filter [%lu] -> [%lu]", page_idx,
                  Address{ page_idx, 0 }.control(), Address{ page_idx, Address::kMaxOffset }.control());
        log_debug("[PAGE: %lu] Found %u *valid* records [%.2lf%%]", page_idx, records_in_page - invalid_records_count,
                  100.0 * (records_in_page - invalid_records_count) / records_in_page);

        memory_agent_->GetFilter(page_idx).Seal();
      } else {
        log_warn("[PAGE: %lu] No records found! :/", page_idx);
      }
    }

    auto end_page = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_page - start_page);
    log_rep("[PAGE: %lu]{tid=%u} Eviction time: %.2lf ms", page_idx, Thread::id(),
              static_cast<double>(elapsed.count()) / 1e6);
    log_rep("[PAGE: %lu]{tid=%u} Filter insertion time: %.2lf ms", page_idx, Thread::id(),
              static_cast<double>(elapsed_filter.count()) / 1e6);
    log_rep("[PAGE: %lu]{tid=%u} ReadCache EVICT: DONE!", page_idx, Thread::id());
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds elapsed_total = std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total);
  log_debug("ReadCache EVICT: DONE! Total time: %.2lf ms", static_cast<double>(elapsed_total.count()) / 1e6);
}
*/

template <class K, class V, class D, class H>
inline void ReadCache<K, V, D, H>::SkipBucket(hash_bucket_t* const bucket) const {
  Address head_address = read_cache_.head_address.load();
  assert(bucket != nullptr);

  for (uint32_t idx = 0; idx < hash_bucket_t::kNumEntries; idx++) {
    do {
      auto* atomic_entry = reinterpret_cast<AtomicHashBucketEntry*>(&(bucket->entries[idx]));
      hash_bucket_entry_t entry{ atomic_entry->load() };

      if (!entry.address().in_readcache()) {
        break;
      }

      // Retrieve hlog address, and replace entry with it
      Address rc_address{ entry.address().readcache_address() };
      assert(rc_address >= head_address);
      const record_t* record = reinterpret_cast<const record_t*>(read_cache_.Get(rc_address));

      hash_bucket_entry_t new_entry{ ReadCacheRecordInfo{ record->header }.previous_address(),
                                     entry.tag(), entry.tentative() };
      assert(!new_entry.address().in_readcache());

      HashBucketEntry expected_entry{ entry };
      if(atomic_entry->compare_exchange_strong(expected_entry, new_entry)) {
        break;
      }
    } while (true);
  }
}

#ifdef STATISTICS
template <class K, class V, class D, class H>
inline void ReadCache<K, V, D, H>::PrintStats() const {
  // Read
  fprintf(stderr, "Read Calls\t: %lu\n", read_calls_.load());
  double read_success_pct = (read_calls_.load() > 0)
      ? (static_cast<double>(read_success_count_.load()) / read_calls_.load()) * 100.0
      : std::numeric_limits<double>::quiet_NaN();
  fprintf(stderr, "Status::Ok (%%): %.2lf\n", read_success_pct);

  // Read [CopyToTail]
  fprintf(stderr, "\nRead [CopyToTail] Calls: %lu\n", read_copy_to_tail_calls_.load());
  double read_copy_to_tail_success_pct = (read_copy_to_tail_calls_.load() > 0)
      ? (static_cast<double>(read_copy_to_tail_success_count_.load()) / read_copy_to_tail_calls_.load()) * 100.0
      : std::numeric_limits<double>::quiet_NaN();
  fprintf(stderr, "Status::Ok (%%): %.2lf\n", read_copy_to_tail_success_pct);

  // TryInsert
  fprintf(stderr, "\nTryInsert Calls\t: %lu\n", try_insert_calls_.load());
  double try_insert_calls_success = (try_insert_calls_.load() > 0)
      ? (static_cast<double>(try_insert_success_count_.load()) / try_insert_calls_.load()) * 100.0
      : std::numeric_limits<double>::quiet_NaN();
  fprintf(stderr, "Status::Ok (%%): %.2lf\n", try_insert_calls_success);

  // Evicted Records
  fprintf(stderr, "\nEvicted Records\t: %lu\n", evicted_records_count_.load());
  double evicted_records_invalid_pct = (evicted_records_count_ > 0)
    ? (static_cast<double>(evicted_records_invalid_.load()) / evicted_records_count_.load()) * 100.0
    : std::numeric_limits<double>::quiet_NaN();
  fprintf(stderr, "Invalid (%%): %.2lf\n", evicted_records_invalid_pct);
}
#endif

}
} // namespace FASTER::core
