// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <string>
#include "address.h"
#include "guid.h"
#include "hash_bucket.h"
#include "native_buffer_pool.h"
#include "record.h"
#include "state_transitions.h"
#include "thread.h"
#include "key_hash.h"

namespace FASTER {
namespace core {

/// Internal contexts, used by FASTER.

enum class OperationType : uint8_t {
  Read,
  RMW,
  Upsert,
  Insert,
  Delete,
  ConditionalInsert,
};

enum class OperationStatus : uint8_t {
  SUCCESS,
  NOT_FOUND,
  RETRY_NOW,
  RETRY_LATER,
  RECORD_ON_DISK,
  SUCCESS_UNMARK,
  NOT_FOUND_UNMARK,
  CPR_SHIFT_DETECTED,
  ABORTED,
  ABORTED_UNMARK,
};

/// Internal FASTER context.
template <class K>
class PendingContext : public IAsyncContext {
 public:
  typedef K key_t;

 protected:
  PendingContext(OperationType type_, IAsyncContext& caller_context_,
                 AsyncCallback caller_callback_)
    : type{ type_ }
    , caller_context{ &caller_context_ }
    , caller_callback{ caller_callback_ }
    , version{ UINT32_MAX }
    , phase{ Phase::INVALID }
    , result{ Status::Pending }
    , address{ Address::kInvalidAddress }
    , entry{ HashBucketEntry::kInvalidEntry } {
  }

 public:
  /// The deep-copy constructor.
  PendingContext(const PendingContext& other, IAsyncContext* caller_context_)
    : type{ other.type }
    , caller_context{ caller_context_ }
    , caller_callback{ other.caller_callback }
    , version{ other.version }
    , phase{ other.phase }
    , result{ other.result }
    , address{ other.address }
    , entry{ other.entry } {
  }

 public:
  /// Go async, for the first time.
  void go_async(Phase phase_, uint32_t version_, Address address_, HashBucketEntry entry_) {
    phase = phase_;
    version = version_;
    address = address_;
    entry = entry_;
  }

  /// Go async, again.
  void continue_async(Address address_, HashBucketEntry entry_) {
    address = address_;
    entry = entry_;
  }

  virtual uint32_t key_size() const = 0;
  virtual void write_deep_key_at(key_t* dst) const = 0;
  virtual KeyHash get_key_hash() const = 0;
  virtual bool is_key_equal(const key_t& other) const = 0;

  /// Caller context.
  IAsyncContext* caller_context;
  /// Caller callback.
  AsyncCallback caller_callback;
  /// Checkpoint version.
  uint32_t version;
  /// Checkpoint phase.
  Phase phase;
  /// Type of operation (Read, Upsert, RMW, etc.).
  OperationType type;
  /// Result of operation.
  Status result;
  /// Address of the record being read or modified.
  Address address;
  /// Hash table entry that (indirectly) leads to the record being read or modified.
  HashBucketEntry entry;
};

// A helper class to copy the key into FASTER log.
// In old API, the Key provided is just the Key type, and we use in-place-new and copy constructor
// to copy the key into the log. In new API, the user provides a ShallowKey, and we call the
// ShallowKey's write_deep_key_at() method to write the key content into the log.
// New API case (user provides ShallowKey)
//
template<bool isShallowKey>
struct write_deep_key_at_helper
{
  template<class ShallowKey, class Key>
  static inline void execute(const ShallowKey& key, Key* dst) {
    key.write_deep_key_at(dst);
  }
};

// Old API case (user provides Key)
//
template<>
struct write_deep_key_at_helper<false>
{
  template<class Key>
  static inline void execute(const Key& key, Key* dst) {
    new (dst) Key(key);
  }
};

/// FASTER's internal Read() context.

/// An internal Read() context that has gone async and lost its type information.
template <class K>
class AsyncPendingReadContext : public PendingContext<K> {
 public:
  typedef K key_t;
 protected:
  AsyncPendingReadContext(IAsyncContext& caller_context_, AsyncCallback caller_callback_, bool abort_if_tombstone_)
    : PendingContext<key_t>(OperationType::Read, caller_context_, caller_callback_)
    , abort_if_tombstone{ abort_if_tombstone_ } {
  }
  /// The deep copy constructor.
  AsyncPendingReadContext(AsyncPendingReadContext& other, IAsyncContext* caller_context)
    : PendingContext<key_t>(other, caller_context)
    , abort_if_tombstone{ other.abort_if_tombstone } {
  }
 public:
  virtual void Get(const void* rec) = 0;
  virtual void GetAtomic(const void* rec) = 0;

  // If true, Read will return ABORT (instead of NOT_FOUND), if record is tombstone
  bool abort_if_tombstone;
};

/// A synchronous Read() context preserves its type information.
template <class RC>
class PendingReadContext : public AsyncPendingReadContext<typename RC::key_t> {
 public:
  typedef RC read_context_t;
  typedef typename read_context_t::key_t key_t;
  typedef typename read_context_t::value_t value_t;
  using key_or_shallow_key_t = std::remove_const_t<std::remove_reference_t<std::result_of_t<decltype(&RC::key)(RC)>>>;
  typedef Record<key_t, value_t> record_t;
  constexpr static const bool kIsShallowKey = !std::is_same<key_or_shallow_key_t, key_t>::value;

  PendingReadContext(read_context_t& caller_context_, AsyncCallback caller_callback_, bool abort_if_tombstone_)
    : AsyncPendingReadContext<key_t>(caller_context_, caller_callback_, abort_if_tombstone_) {
  }
  /// The deep copy constructor.
  PendingReadContext(PendingReadContext& other, IAsyncContext* caller_context_)
    : AsyncPendingReadContext<key_t>(other, caller_context_) {
  }
 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    return IAsyncContext::DeepCopy_Internal(*this, PendingContext<key_t>::caller_context,
                                            context_copy);
  }
 private:
  inline const read_context_t& read_context() const {
    return *static_cast<const read_context_t*>(PendingContext<key_t>::caller_context);
  }
  inline read_context_t& read_context() {
    return *static_cast<read_context_t*>(PendingContext<key_t>::caller_context);
  }
 public:
  /// Accessors.
  inline const key_or_shallow_key_t& get_key_or_shallow_key() const {
    return read_context().key();
  }
  inline uint32_t key_size() const final {
    return read_context().key().size();
  }
  inline void write_deep_key_at(key_t* dst) const final {
    // this should never be called
    assert(false);
  }
  inline KeyHash get_key_hash() const final {
    return read_context().key().GetHash();
  }
  inline bool is_key_equal(const key_t& other) const final {
    return read_context().key() == other;
  }
  inline void Get(const void* rec) final {
    const record_t* record = reinterpret_cast<const record_t*>(rec);
    read_context().Get(record->value());
  }
  inline void GetAtomic(const void* rec) final {
    const record_t* record = reinterpret_cast<const record_t*>(rec);
    read_context().GetAtomic(record->value());
  }
};

/// FASTER's internal Upsert() context.

/// An internal Upsert() context that has gone async and lost its type information.
template <class K>
class AsyncPendingUpsertContext : public PendingContext<K> {
 public:
  typedef K key_t;
 protected:
  AsyncPendingUpsertContext(IAsyncContext& caller_context_, AsyncCallback caller_callback_)
    : PendingContext<key_t>(OperationType::Upsert, caller_context_, caller_callback_) {
  }
  /// The deep copy constructor.
  AsyncPendingUpsertContext(AsyncPendingUpsertContext& other, IAsyncContext* caller_context)
    : PendingContext<key_t>(other, caller_context) {
  }
 public:
  virtual void Put(void* rec) = 0;
  virtual bool PutAtomic(void* rec) = 0;
  virtual uint32_t value_size() const = 0;
};

/// A synchronous Upsert() context preserves its type information.
template <class UC>
class PendingUpsertContext : public AsyncPendingUpsertContext<typename UC::key_t> {
 public:
  typedef UC upsert_context_t;
  typedef typename upsert_context_t::key_t key_t;
  typedef typename upsert_context_t::value_t value_t;
  using key_or_shallow_key_t = std::remove_const_t<std::remove_reference_t<std::result_of_t<decltype(&UC::key)(UC)>>>;
  typedef Record<key_t, value_t> record_t;
  constexpr static const bool kIsShallowKey = !std::is_same<key_or_shallow_key_t, key_t>::value;

  PendingUpsertContext(upsert_context_t& caller_context_, AsyncCallback caller_callback_)
    : AsyncPendingUpsertContext<key_t>(caller_context_, caller_callback_) {
  }
  /// The deep copy constructor.
  PendingUpsertContext(PendingUpsertContext& other, IAsyncContext* caller_context_)
    : AsyncPendingUpsertContext<key_t>(other, caller_context_) {
  }
 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    return IAsyncContext::DeepCopy_Internal(*this, PendingContext<key_t>::caller_context,
                                            context_copy);
  }
 private:
  inline const upsert_context_t& upsert_context() const {
    return *static_cast<const upsert_context_t*>(PendingContext<key_t>::caller_context);
  }
  inline upsert_context_t& upsert_context() {
    return *static_cast<upsert_context_t*>(PendingContext<key_t>::caller_context);
  }

 public:
  /// Accessors.
  inline const key_or_shallow_key_t& get_key_or_shallow_key() const {
     return upsert_context().key();
  }
  inline uint32_t key_size() const final {
    return upsert_context().key().size();
  }
  inline void write_deep_key_at(key_t* dst) const final {
    write_deep_key_at_helper<kIsShallowKey>::execute(upsert_context().key(), dst);
  }
  inline KeyHash get_key_hash() const final {
    return upsert_context().key().GetHash();
  }
  inline bool is_key_equal(const key_t& other) const final {
    return upsert_context().key() == other;
  }
  inline void Put(void* rec) final {
    record_t* record = reinterpret_cast<record_t*>(rec);
    upsert_context().Put(record->value());
  }
  inline bool PutAtomic(void* rec) final {
    record_t* record = reinterpret_cast<record_t*>(rec);
    return upsert_context().PutAtomic(record->value());
  }
  inline constexpr uint32_t value_size() const final {
    return upsert_context().value_size();
  }
};

/// FASTER's internal Rmw() context.
/// An internal Rmw() context that has gone async and lost its type information.
template <class K>
class AsyncPendingRmwContext : public PendingContext<K> {
 public:
  typedef K key_t;
 protected:
  AsyncPendingRmwContext(IAsyncContext& caller_context_, AsyncCallback caller_callback_, bool create_if_not_exists_)
    : PendingContext<key_t>(OperationType::RMW, caller_context_, caller_callback_)
    , create_if_not_exists{create_if_not_exists_} {
  }
  /// The deep copy constructor.
  AsyncPendingRmwContext(AsyncPendingRmwContext& other, IAsyncContext* caller_context)
    : PendingContext<key_t>(other, caller_context)
    , create_if_not_exists{other.create_if_not_exists} {
  }
 public:
  /// Set initial value.
  virtual void RmwInitial(void* rec) = 0;
  /// RCU.
  virtual void RmwCopy(const void* old_rec, void* rec) = 0;
  /// in-place update.
  virtual bool RmwAtomic(void* rec) = 0;
  /// Get value size for initial value or in-place update
  virtual uint32_t value_size() const = 0;
  /// Get value size for RCU
  virtual uint32_t value_size(const void* old_rec) const = 0;

  /// If false, it will return NOT_FOUND instead of creating a new record
  bool create_if_not_exists;
};

/// A synchronous Rmw() context preserves its type information.
template <class MC>
class PendingRmwContext : public AsyncPendingRmwContext<typename MC::key_t> {
 public:
  typedef MC rmw_context_t;
  typedef typename rmw_context_t::key_t key_t;
  typedef typename rmw_context_t::value_t value_t;
  using key_or_shallow_key_t = std::remove_const_t<std::remove_reference_t<std::result_of_t<decltype(&MC::key)(MC)>>>;
  typedef Record<key_t, value_t> record_t;
  constexpr static const bool kIsShallowKey = !std::is_same<key_or_shallow_key_t, key_t>::value;

  PendingRmwContext(rmw_context_t& caller_context_, AsyncCallback caller_callback_, bool create_if_not_exists_)
    : AsyncPendingRmwContext<key_t>(caller_context_, caller_callback_, create_if_not_exists_) {
  }
  /// The deep copy constructor.
  PendingRmwContext(PendingRmwContext& other, IAsyncContext* caller_context_)
    : AsyncPendingRmwContext<key_t>(other, caller_context_) {
  }
 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    return IAsyncContext::DeepCopy_Internal(*this, PendingContext<key_t>::caller_context,
                                            context_copy);
  }
 private:
  const rmw_context_t& rmw_context() const {
    return *static_cast<const rmw_context_t*>(PendingContext<key_t>::caller_context);
  }
  rmw_context_t& rmw_context() {
    return *static_cast<rmw_context_t*>(PendingContext<key_t>::caller_context);
  }
 public:
  /// Accessors.
  inline const key_or_shallow_key_t& get_key_or_shallow_key() const {
    return rmw_context().key();
  }
  inline uint32_t key_size() const final {
    return rmw_context().key().size();
  }
  inline void write_deep_key_at(key_t* dst) const final {
    write_deep_key_at_helper<kIsShallowKey>::execute(rmw_context().key(), dst);
  }
  inline KeyHash get_key_hash() const final {
    return rmw_context().key().GetHash();
  }
  inline bool is_key_equal(const key_t& other) const final {
    return rmw_context().key() == other;
  }
  /// Set initial value.
  inline void RmwInitial(void* rec) final {
    record_t* record = reinterpret_cast<record_t*>(rec);
    rmw_context().RmwInitial(record->value());
  }
  /// RCU.
  inline void RmwCopy(const void* old_rec, void* rec) final {
    const record_t* old_record = reinterpret_cast<const record_t*>(old_rec);
    record_t* record = reinterpret_cast<record_t*>(rec);
    rmw_context().RmwCopy(old_record->value(), record->value());
  }
  /// in-place update.
  inline bool RmwAtomic(void* rec) final {
    record_t* record = reinterpret_cast<record_t*>(rec);
    return rmw_context().RmwAtomic(record->value());
  }
  /// Get value size for initial value or in-place update
  inline constexpr uint32_t value_size() const final {
    return rmw_context().value_size();
  }
  /// Get value size for RCU
  inline constexpr uint32_t value_size(const void* old_rec) const final {
    const record_t* old_record = reinterpret_cast<const record_t*>(old_rec);
    return rmw_context().value_size(old_record->value());
  }
};

/// FASTER's internal Delete() context.

/// An internal Delete() context that has gone async and lost its type information.
template <class K>
class AsyncPendingDeleteContext : public PendingContext<K> {
 public:
  typedef K key_t;
 protected:
  AsyncPendingDeleteContext(IAsyncContext& caller_context_, AsyncCallback caller_callback_)
    : PendingContext<key_t>(OperationType::Delete, caller_context_, caller_callback_) {
  }
  /// The deep copy constructor.
  AsyncPendingDeleteContext(AsyncPendingDeleteContext& other, IAsyncContext* caller_context)
    : PendingContext<key_t>(other, caller_context) {
  }
 public:
  /// Get value size for initial value
  virtual uint32_t value_size() const = 0;
};

/// A synchronous Delete() context preserves its type information.
template <class MC>
class PendingDeleteContext : public AsyncPendingDeleteContext<typename MC::key_t> {
 public:
  typedef MC delete_context_t;
  typedef typename delete_context_t::key_t key_t;
  typedef typename delete_context_t::value_t value_t;
  using key_or_shallow_key_t = std::remove_const_t<std::remove_reference_t<std::result_of_t<decltype(&MC::key)(MC)>>>;
  typedef Record<key_t, value_t> record_t;
  constexpr static const bool kIsShallowKey = !std::is_same<key_or_shallow_key_t, key_t>::value;

  PendingDeleteContext(delete_context_t& caller_context_, AsyncCallback caller_callback_)
    : AsyncPendingDeleteContext<key_t>(caller_context_, caller_callback_) {
  }
  /// The deep copy constructor.
  PendingDeleteContext(PendingDeleteContext& other, IAsyncContext* caller_context_)
    : AsyncPendingDeleteContext<key_t>(other, caller_context_) {
  }
 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    return IAsyncContext::DeepCopy_Internal(*this, PendingContext<key_t>::caller_context,
                                            context_copy);
  }
 private:
  const delete_context_t& delete_context() const {
    return *static_cast<const delete_context_t*>(PendingContext<key_t>::caller_context);
  }
  delete_context_t& delete_context() {
    return *static_cast<delete_context_t*>(PendingContext<key_t>::caller_context);
  }
 public:
  /// Accessors.
  inline const key_or_shallow_key_t& get_key_or_shallow_key() const {
    return delete_context().key();
  }
  inline uint32_t key_size() const final {
    return delete_context().key().size();
  }
  inline void write_deep_key_at(key_t* dst) const final {
    write_deep_key_at_helper<kIsShallowKey>::execute(delete_context().key(), dst);
  }
  inline KeyHash get_key_hash() const final {
    return delete_context().key().GetHash();
  }
  inline bool is_key_equal(const key_t& other) const final {
    return delete_context().key() == other;
  }
  /// Get value size for initial value
  inline uint32_t value_size() const final {
    return delete_context().value_size();
  }
};

// FASTER's internal ConditionalInsert

/// An internal ConditionalInsert() context that has gone async and lost its type information.
template <class K>
class AsyncPendingConditionalInsertContext : public PendingContext<K> {
 public:
  typedef K key_t;
 protected:
  AsyncPendingConditionalInsertContext(IAsyncContext& caller_context_, AsyncCallback caller_callback_,
                                        HashBucketEntry start_search_entry_, Address min_search_offset_, void* dest_store_)
    : PendingContext<key_t>(OperationType::ConditionalInsert, caller_context_, caller_callback_)
    , start_search_entry{ start_search_entry_ }
    , min_search_offset{ min_search_offset_ }
    , dest_store{ dest_store_ } {
  }
  /// The deep copy constructor.
  AsyncPendingConditionalInsertContext(AsyncPendingConditionalInsertContext& other, IAsyncContext* caller_context)
    : PendingContext<key_t>(other, caller_context)
    , start_search_entry{ other.start_search_entry }
    , min_search_offset{ other.min_search_offset }
    , dest_store{ other.dest_store } {
  }
 public:
  virtual uint32_t value_size() const = 0;
  virtual bool is_tombstone() const = 0;

  // Called when writing to same store
  virtual bool Insert(void* dest, uint32_t alloc_size) const = 0;
  // Called when writing to different store (i.e. hot-cold compaction)
  // NOTE: these methods are called by FASTER InternalUpsert method
  virtual void Put(void *rec) = 0;
  virtual bool PutAtomic(void *rec) = 0;
 public:
  HashBucketEntry start_search_entry;
  Address min_search_offset;
  void* dest_store;
};

/// A synchronous ConditionalInsert() context preserves its type information.
template<class CIC>
class PendingConditionalInsertContext : public AsyncPendingConditionalInsertContext<typename CIC::key_t> {
 public:
  typedef CIC conditional_insert_context_t;
  typedef typename conditional_insert_context_t::key_t key_t;
  typedef typename conditional_insert_context_t::value_t value_t;
  using key_or_shallow_key_t = std::remove_const_t<std::remove_reference_t<std::result_of_t<decltype(&CIC::key)(CIC)>>>;
  typedef Record<key_t, value_t> record_t;

  PendingConditionalInsertContext(conditional_insert_context_t& caller_context_, AsyncCallback caller_callback_,
                                  HashBucketEntry start_search_entry_, Address min_search_offset_, void* dest_store_)
    : AsyncPendingConditionalInsertContext<key_t>(caller_context_, caller_callback_,
                                                start_search_entry_, min_search_offset_, dest_store_) {
  }
  /// The deep copy constructor.
  PendingConditionalInsertContext(PendingConditionalInsertContext& other, IAsyncContext* caller_context_)
    : AsyncPendingConditionalInsertContext<key_t>(other, caller_context_) {
  }
 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    return IAsyncContext::DeepCopy_Internal(*this, PendingContext<key_t>::caller_context,
                                            context_copy);
  }
 private:
  const conditional_insert_context_t& conditional_insert_context() const {
    return *static_cast<const conditional_insert_context_t*>(PendingContext<key_t>::caller_context);
  }
  conditional_insert_context_t& conditional_insert_context() {
    return *static_cast<conditional_insert_context_t*>(PendingContext<key_t>::caller_context);
  }
 public:
  /// Accessors.
  inline const key_or_shallow_key_t& get_key_or_shallow_key() const {
    return conditional_insert_context().key();
  }
  inline uint32_t key_size() const final {
    return conditional_insert_context().key().size();
  }
  inline void write_deep_key_at(key_t* dst) const final {
    // Only called with doing hot-cold compaction, by the FASTER InternalUpsert method
    // Here, we cannot utilize the `write_deep_key_at_helper`, because the
    // user does NOT provides us with a context that contains the ShallowKey
    // Instead, we manually memcpy the key contents (hot log) to the new destination (cold log)
    memcpy(dst, &conditional_insert_context().key(), conditional_insert_context().key().size());
  }
  inline KeyHash get_key_hash() const final {
    return conditional_insert_context().key().GetHash();
  }
  inline bool is_key_equal(const key_t& other) const final {
    return conditional_insert_context().key() == other;
  }
  inline uint32_t value_size() const final {
    return conditional_insert_context().value_size();
  }

  // ConditionalInsert contexts contains these additional methods
  inline bool is_tombstone() const final {
    return conditional_insert_context().is_tombstone();
  }
  inline bool Insert(void* dest, uint32_t alloc_size) const final {
    return conditional_insert_context().Insert(dest, alloc_size);
  }
  inline void Put(void* rec) final {
    conditional_insert_context().Put(rec);
  }
  inline bool PutAtomic(void* rec) final {
    return conditional_insert_context().PutAtomic(rec);
  }
};

class AsyncIOContext;

/// Per-thread execution context. (Just the stuff that's checkpointed to disk.)
struct PersistentExecContext {
  PersistentExecContext()
    : serial_num{ 0 }
    , version{ 0 }
    , guid{} {
  }

  void Initialize(uint32_t version_, const Guid& guid_, uint64_t serial_num_) {
    serial_num = serial_num_;
    version = version_;
    guid = guid_;
  }

  uint64_t serial_num;
  uint32_t version;
  /// Unique identifier for this session.
  Guid guid;
};
static_assert(sizeof(PersistentExecContext) == 32, "sizeof(PersistentExecContext) != 32");

/// Per-thread execution context. (Also includes state kept in-memory-only.)
struct ExecutionContext : public PersistentExecContext {
  /// Default constructor.
  ExecutionContext()
    : phase{ Phase::INVALID }
    , io_id{ 0 } {
  }

  void Initialize(Phase phase_, uint32_t version_, const Guid& guid_, uint64_t serial_num_) {
    assert(retry_requests.empty());
    assert(pending_ios.empty());
    assert(io_responses.empty());

    PersistentExecContext::Initialize(version_, guid_, serial_num_);
    phase = phase_;
    retry_requests.clear();
    io_id = 0;
    pending_ios.clear();
    io_responses.clear();
  }

  Phase phase;

  /// Retry request contexts are stored inside the deque.
  std::deque<IAsyncContext*> retry_requests;
  /// Assign a unique ID to every I/O request.
  uint64_t io_id;
  /// For each pending I/O, maps io_id to the hash of the key being retrieved.
  std::unordered_map<uint64_t, KeyHash> pending_ios;

  /// The I/O completion thread hands the PendingContext back to the thread that issued the
  /// request.
  concurrent_queue<AsyncIOContext*> io_responses;
};

}
} // namespace FASTER::core
