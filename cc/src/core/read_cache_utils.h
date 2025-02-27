// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "address.h"
#include "internal_contexts.h"
#include "persistent_memory_malloc.h"
#include "record.h"
#include "status.h"

namespace FASTER {
namespace core {

typedef Address(*ReadCacheBlockAllocateCallback)(void* faster, uint32_t record_size);

/// Read Cache Record header, internal to FASTER.
class ReadCacheRecordInfo {
 public:
  ReadCacheRecordInfo(uint16_t checkpoint_version_, bool in_cold_hlog_, bool tombstone_,
                      bool invalid_, Address previous_address)
    : previous_address_{ previous_address.control() }
    , checkpoint_version{ checkpoint_version_ }
    , invalid{ invalid_ }
    , tombstone{ tombstone_ }
    , in_cold_hlog{ in_cold_hlog_ } {
  }
  ReadCacheRecordInfo(const ReadCacheRecordInfo& other)
    : control_{ other.control_ } {
  }
  ReadCacheRecordInfo(const RecordInfo& other)
    : control_{ other.control_ } {
  }

  inline RecordInfo ToRecordInfo() {
    return RecordInfo{ control_ };
  }

  inline bool IsNull() const {
    return control_ == 0;
  }
  inline Address previous_address() const {
    return Address{ previous_address_ };
  }

  union {
      struct {
        uint64_t previous_address_  : 48;
        uint64_t checkpoint_version : 13;
        uint64_t invalid            :  1;
        uint64_t tombstone          :  1;
        uint64_t in_cold_hlog       :  1;
      };
      struct {
        uint64_t previous_address_  : 47;
        uint64_t readcache          :  1;
        uint64_t checkpoint_version : 13;
        uint64_t invalid            :  1;
        uint64_t tombstone          :  1;
        uint64_t in_cold_hlog       :  1;
      } rc_;

      uint64_t control_;
    };
};
static_assert(sizeof(ReadCacheRecordInfo) == 8, "sizeof(RecordInfo) != 8");


template <class K, class V>
class ReadCacheEvictContext : public IAsyncContext {
 public:
  typedef K key_t;
  typedef V value_t;
  typedef Record<key_t, value_t> record_t;

  /// Constructs and returns a context given a pointer to a record.
  ReadCacheEvictContext(record_t* record)
   : record_{ record }
   , entry{ HashBucketEntry::kInvalidEntry }
   , atomic_entry{ nullptr }
   // the following two are not used, since RC is only available in hot hlog
   , index_op_type{ IndexOperationType::None }
   , index_op_result{ Status::Corruption } {
  }

  /// Copy constructor deleted -- operation never goes async
  ReadCacheEvictContext(const ReadCacheEvictContext& from) = delete;

  /// Invoked from within FASTER.
  inline const key_t& key() const {
    return record_->key();
  }
  inline uint32_t key_size() const {
    return key().size();
  }
  inline KeyHash get_key_hash() const {
    return key().GetHash();
  }

  inline void set_index_entry(HashBucketEntry entry_, AtomicHashBucketEntry* atomic_entry_) {
    entry = entry_;
    atomic_entry = atomic_entry_;
  }
  inline void clear_index_op() { }

 protected:
  /// Copies this context into a passed-in pointer if the operation goes
  /// asynchronous inside FASTER
  Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    throw std::runtime_error{ "ReadCache Evict Context should *not* go async!" };
  }

 private:
  /// Pointer to the record
  record_t* record_;

 public:
  /// Hash table entry that (indirectly) leads to the record being read or modified.
  HashBucketEntry entry;
  /// Pointer to the atomic hash bucket entry
  AtomicHashBucketEntry* atomic_entry;
  /// Not used -- kept only for compilation purposes.
  IndexOperationType index_op_type;
  Status index_op_result;

};

typedef Address(*ReadCacheEvictCallback)(void* readcache, Address from_head_address, Address to_head_address);

template <class D>
class ReadCachePersistentMemoryMalloc : public PersistentMemoryMalloc<D> {
 public:
  typedef D disk_t;
  typedef typename D::log_file_t log_file_t;

  ReadCachePersistentMemoryMalloc(bool has_no_backing_storage, uint64_t log_size, LightEpoch& epoch, disk_t& disk_, log_file_t& file_,
                                double log_mutable_fraction, bool pre_allocate_log,
                                ReadCacheEvictCallback evict_callback) // read-cache specific
    : PersistentMemoryMalloc<disk_t>(has_no_backing_storage, log_size, epoch, disk_, file_,
                                    log_mutable_fraction, pre_allocate_log)
    , evict_callback_{ evict_callback } {
      // evict callback should be provided
      assert(evict_callback_ != nullptr);
  }

  ~ReadCachePersistentMemoryMalloc() { }

  void SetReadCacheInstance(void* readcache) {
    readcache_ = readcache;
  }

 private:
  inline void PageAlignedShiftHeadAddress(uint32_t tail_page) final {
    static constexpr uint32_t kNumHeadPages = PersistentMemoryMalloc<D>::kNumHeadPages;

    // obtain local values of variables that can change
    Address current_head_address = this->head_address.load();
    Address current_flushed_until_address = this->flushed_until_address.load();
    uint32_t num_pages = this->buffer_.num_pages;

    if(tail_page <= (num_pages - kNumHeadPages)) {
      // Desired head address is <= 0.
      return;
    }

    Address desired_head_address{ tail_page - (num_pages - kNumHeadPages), 0 };
    if(current_flushed_until_address < desired_head_address) {
      desired_head_address = Address{ current_flushed_until_address.page(), 0 };
    }
    if (desired_head_address <= current_head_address) {
      // Current head address is already ahead of desired head address.
      return;
    }

    Address evicted_to = evict_callback_(readcache_, current_head_address, desired_head_address);
    if (evicted_to == Address::kInvalidAddress) {
      return;
    }
    assert(evicted_to > current_head_address);

    Address old_head_address;
    if(this->MonotonicUpdate(this->head_address, evicted_to, old_head_address)) {
      typename PersistentMemoryMalloc<D>::OnPagesClosed_Context context{ this, evicted_to, false };
      IAsyncContext* context_copy;
      Status result = context.DeepCopy(context_copy);
      assert(result == Status::Ok);
      this->epoch_->BumpCurrentEpoch(this->OnPagesClosed, context_copy);
    } else {
      log_error("MonotonicUpdate failed: %lu -> %lu [%lu]", current_head_address.control(),
                evicted_to.control(), old_head_address.control());
      throw std::runtime_error{ "MonotonicUpdate for head_address not successful" };
    }
  }


 private:
  void* readcache_;
  ReadCacheEvictCallback evict_callback_;
};


}
} // namespace FASTER::core