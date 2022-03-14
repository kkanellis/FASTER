
#pragma once

#include "address.h"
#include "async.h"
#include "hash_bucket.h"
#include "key_hash.h"
#include "guid.h"
#include "record.h"

namespace FASTER {
namespace core {

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


template<bool isHotColdContext>
struct hc_context_helper
{
  /// Used by all FASTER pending contexts
  template<class RC>
  static inline uint32_t key_size(RC& context) {
    return context.key_size();
  }
  template<class RC>
  static inline KeyHash get_key_hash(RC& context) {
    return context.get_key_hash();
  }
  template<class RC, class Key>
  static inline bool is_key_equal(RC& context, const Key& other) {
    return context.is_key_equal(other);
  }
  template<class MC, class Key>
  static void write_deep_key_at(MC& context, Key* dst) {
    context.write_deep_key_at(dst);
  }

  /// Used by FASTER's Read pending context
  template<class RC, class Record>
  static inline void Get(RC& context, const void* rec) {
    context.Get(rec);
  }
  template<class RC, class Record>
  static inline void GetAtomic(RC& context, const void* rec) {
    context.GetAtomic(rec);
  }
};

template<>
struct hc_context_helper<false>
{
  /// Used by all FASTER pending contexts
  template<class RC>
  static inline uint32_t key_size(RC& context) {
    return context.key().size();
  }
  template<class RC>
  static inline KeyHash get_key_hash(RC& context) {
    return context.key().GetHash();
  }
  template<class RC, class Key>
  static inline bool is_key_equal(RC& context, const Key& other) {
    return context.key() == other;
  }
  template<class MC, class Key>
  static inline void write_deep_key_at(MC& context, Key* dst) {
    using key_or_shallow_key_t = std::remove_const_t<std::remove_reference_t<std::result_of_t<decltype(&MC::key)(MC)>>>;
    constexpr static const bool kIsShallowKey = !std::is_same<key_or_shallow_key_t, Key>::value;

    write_deep_key_at_helper<kIsShallowKey>::execute(context.key(), dst);
  }

  /// Used by FASTER's Read pending context
  template<class RC, class Record>
  static inline void Get(RC& context, const void* rec) {
    const Record* record = reinterpret_cast<const Record*>(rec);
    context.Get(record->value());
  }
  template<class RC, class Record>
  static inline void GetAtomic(RC& context, const void* rec) {
    const Record* record = reinterpret_cast<const Record*>(rec);
    context.GetAtomic(record->value());
  }
};

/// Internal contexts, used by FASTER HC
enum class ReadOperationStage {
  HOT_LOG_READ = 1,
  COLD_LOG_READ = 2,
};

enum class RmwOperationStage {
  HOT_LOG_RMW = 1,
  COLD_LOG_READ = 2,
  HOT_LOG_CONDITIONAL_INSERT = 3,
  WAIT_FOR_RETRY = 4,
};

template <class K>
class HotColdContext : public IAsyncContext {
 public:
  typedef K key_t;

  HotColdContext(void* faster_hc_, IAsyncContext& caller_context_,
                  AsyncCallback caller_callback_, uint64_t monotonic_serial_num_)
    : faster_hc{ faster_hc_ }
    , caller_context{ &caller_context_ }
    , caller_callback{ caller_callback_ }
    , serial_num{ monotonic_serial_num_ }
  {}
  /// No copy constructor.
  HotColdContext(const HotColdContext& other) = delete;
  /// The deep-copy constructor.
  HotColdContext(HotColdContext& other, IAsyncContext* caller_context_)
    : faster_hc{ other.faster_hc }
    , caller_context{ caller_context_ }
    , caller_callback{ other.caller_callback }
    , serial_num{ other.serial_num }
  {}

  /// Points to FasterHC
  void* faster_hc;
  /// User context
  IAsyncContext* caller_context;
  /// User-provided callback
  AsyncCallback caller_callback;
  /// Request serial num
  uint64_t serial_num;
};

/// Context that holds user context for Read request
template <class K, class V>
class AsyncHotColdReadContext : public HotColdContext<K> {
 public:
  typedef K key_t;
  typedef V value_t;

 protected:
  AsyncHotColdReadContext(void* faster_hc_, ReadOperationStage stage_, IAsyncContext& caller_context_,
                        AsyncCallback caller_callback_, uint64_t monotonic_serial_num_)
    : HotColdContext<key_t>(faster_hc_, caller_context_, caller_callback_, monotonic_serial_num_)
    , stage{ stage_ }
  {}
  /// The deep-copy constructor.
  AsyncHotColdReadContext(AsyncHotColdReadContext& other_, IAsyncContext* caller_context_)
    : HotColdContext<key_t>(other_, caller_context_)
    , stage{ other_.stage }
  {}
 public:
  virtual uint32_t key_size() const = 0;
  virtual KeyHash get_key_hash() const = 0;
  virtual bool is_key_equal(const key_t& other) const = 0;

  virtual void Get(const void* rec) = 0;
  virtual void GetAtomic(const void* rec) = 0;

  ReadOperationStage stage;
};

/// Context that holds user context for Read request
template <class RC>
class HotColdReadContext : public AsyncHotColdReadContext <typename RC::key_t, typename RC::value_t> {
 public:
  typedef RC read_context_t;
  typedef typename read_context_t::key_t key_t;
  typedef typename read_context_t::value_t value_t;
  typedef Record<key_t, value_t> record_t;

  HotColdReadContext(void* faster_hc_, ReadOperationStage stage_, read_context_t& caller_context_,
                    AsyncCallback caller_callback_, uint64_t monotonic_serial_num_)
    : AsyncHotColdReadContext<key_t, value_t>(faster_hc_, stage_, caller_context_,
                                            caller_callback_, monotonic_serial_num_)
    {}

  /// The deep-copy constructor.
  HotColdReadContext(HotColdReadContext& other_, IAsyncContext* caller_context_)
    : AsyncHotColdReadContext<key_t, value_t>(other_, caller_context_)
    {}

 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    return IAsyncContext::DeepCopy_Internal(*this, HotColdContext<key_t>::caller_context,
                                            context_copy);
  }
 private:
  inline const read_context_t& read_context() const {
    return *static_cast<const read_context_t*>(HotColdContext<key_t>::caller_context);
  }
  inline read_context_t& read_context() {
    return *static_cast<read_context_t*>(HotColdContext<key_t>::caller_context);
  }
 public:
  /// Propagates calls to caller context
  inline uint32_t key_size() const final {
    return hc_context_helper<false>::key_size(read_context());
  }
  inline KeyHash get_key_hash() const final {
    return hc_context_helper<false>::get_key_hash(read_context());
  }
  inline bool is_key_equal(const key_t& other) const final {
    return hc_context_helper<false>::is_key_equal(read_context(), other);
  }
  inline void Get(const void* rec) {
    return hc_context_helper<false>::template Get<read_context_t, record_t>(read_context(), rec);
  }
  inline void GetAtomic(const void* rec) {
    return hc_context_helper<false>::template GetAtomic<read_context_t, record_t>(read_context(), rec);
  }
};

/*
template<class K, class V>
class AsyncHotColdRmwReadContext : public AsyncHotColdReadContext<K, V> {
 public:
  typedef K key_t;
  typedef V value_t;
  typedef Record<key_t, value_t> record_t;

 protected:
  AsyncHotColdRmwReadContext(void* faster_hc_, IAsyncContext* hc_rmw_context_, IAsyncContext& caller_context_,
                            AsyncCallback caller_callback_, uint64_t monotonic_serial_num_)
    : AsyncHotColdReadContext<key_t, value_t>(faster_hc_, ReadOperationStage::COLD_LOG_READ,
                                        caller_context_, caller_callback_, monotonic_serial_num_)
    , hc_rmw_context{ hc_rmw_context_ }
    , record{ nullptr }
  {}
  /// The deep-copy constructor.
  AsyncHotColdRmwReadContext(AsyncHotColdRmwReadContext& other_, IAsyncContext* caller_context_)
    : AsyncHotColdReadContext<key_t, value_t>(other_, caller_context_)
    , hc_rmw_context{ other_.hc_rmw_context }
    , record{ other_.record }
  {}

 public:
  virtual const key_t* key() const = 0;
  virtual const value_t* value() const = 0;

  IAsyncContext* hc_rmw_context; // HotColdRmwContext
  record_t* record;
};

template<class MC>
class HotColdRmwReadContext : public AsyncHotColdRmwReadContext<typename MC::key_t, typename MC::value_t> {
 public:
  typedef MC rmw_context_t;
  typedef typename rmw_context_t::key_t key_t;
  typedef typename rmw_context_t::value_t value_t;
  typedef Record<key_t, value_t> record_t;

  HotColdRmwReadContext(void* faster_hc_, IAsyncContext* rmw_context_, rmw_context_t& caller_context_,
                    AsyncCallback caller_callback_, uint64_t monotonic_serial_num_)
    : AsyncHotColdRmwReadContext<key_t, value_t>(faster_hc_, rmw_context_, caller_context_,
                                                  caller_callback_, monotonic_serial_num_)
    {}

  /// The deep-copy constructor.
  HotColdRmwReadContext(HotColdRmwReadContext& other, IAsyncContext* caller_context_)
    : AsyncHotColdRmwReadContext<key_t, value_t>(other, caller_context_)
    {}

 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    // need to deep copy rmw context, if didn't went async
    Status rmw_deep_copy_status = this->hc_rmw_context->DeepCopy(this->hc_rmw_context);
    if (rmw_deep_copy_status != Status::Ok) {
      return rmw_deep_copy_status;
    }
    return IAsyncContext::DeepCopy_Internal(*this, HotColdContext<key_t>::caller_context,
                                            context_copy);
  }

 private:
  inline const rmw_context_t& rmw_context() const {
    return *static_cast<const rmw_context_t*>(HotColdContext<key_t>::caller_context);
  }
  inline rmw_context_t& rmw_context() {
    return *static_cast<rmw_context_t*>(HotColdContext<key_t>::caller_context);
  }
 public:
  /// Propagates calls to caller context
  inline uint32_t key_size() const final {
    return hc_context_helper<false>::key_size(rmw_context());
  }
  inline KeyHash get_key_hash() const final {
    return hc_context_helper<false>::get_key_hash(rmw_context());
  }
  inline bool is_key_equal(const key_t& other) const final {
    return hc_context_helper<false>::is_key_equal(rmw_context(), other);
  }

 public:
  inline const key_t* key() const final {
    assert(this->record != nullptr);
    return &this->record->key();
  }

  inline const value_t* value() const final {
    assert(this->record != nullptr);
    return &this->record->value();
  }

  inline void Get(const void* rec) final {
    this->record = const_cast<record_t*>(reinterpret_cast<const record_t*>(rec));
    assert(this->record != nullptr);
    // TODO: check if we need to create a copy instead
  }
  inline void GetAtomic(const void* rec) final {
    // This should not be called, since we are doing a request on the cold log
    // TODO: fix
    this->record = const_cast<record_t*>(reinterpret_cast<const record_t*>(rec));
    assert(this->record != nullptr);
  }
};
*/


/// Context that holds user context for RMW request

template <class K, class V>
class AsyncHotColdRmwContext : public HotColdContext<K> {
 public:
  typedef K key_t;
  typedef V value_t;
  //typedef AsyncHotColdRmwReadContext<key_t, value_t> rmw_read_context_t;

 protected:
  AsyncHotColdRmwContext(void* faster_hc_, RmwOperationStage stage_, HashBucketEntry& expected_entry_,
                      IAsyncContext& caller_context_, AsyncCallback caller_callback_, uint64_t monotonic_serial_num_)
    : HotColdContext<key_t>(faster_hc_, caller_context_, caller_callback_, monotonic_serial_num_)
    , stage{ stage_ }
    , expected_entry{ expected_entry_ }
    , read_context{ nullptr }
  {}
  /// The deep copy constructor.
  AsyncHotColdRmwContext(AsyncHotColdRmwContext& other, IAsyncContext* caller_context)
    : HotColdContext<key_t>(other, caller_context)
    , stage{ other.stage }
    , expected_entry{ other.expected_entry }
    , read_context{ other.read_context }
  {}
 public:
  //virtual const key_t& key() const = 0;
  virtual uint32_t key_size() const = 0;
  virtual void write_deep_key_at(key_t* dst) const = 0;
  virtual KeyHash get_key_hash() const = 0;
  virtual bool is_key_equal(const key_t& other) const = 0;

  /// Set initial value.
  virtual void RmwInitial(value_t& value) = 0;
  /// RCU.
  virtual void RmwCopy(const value_t& old_value, value_t& value) = 0;
  /// in-place update.
  virtual bool RmwAtomic(value_t& value) = 0;
  /// Get value size for initial value or in-place update
  virtual uint32_t value_size() const = 0;
  /// Get value size for RCU
  virtual uint32_t value_size(const value_t& value) const = 0;

  RmwOperationStage stage;
  HashBucketEntry expected_entry;
  IAsyncContext* read_context; // HotColdRmwRead context
};

template <class MC>
class HotColdRmwContext : public AsyncHotColdRmwContext<typename MC::key_t, typename MC::value_t> {
 public:
  typedef MC rmw_context_t;
  typedef typename rmw_context_t::key_t key_t;
  typedef typename rmw_context_t::value_t value_t;
  typedef Record<key_t, value_t> record_t;

  HotColdRmwContext(void* faster_hc_, RmwOperationStage stage_, HashBucketEntry& expected_entry_,
                  rmw_context_t& caller_context_, AsyncCallback caller_callback_, uint64_t monotonic_serial_num_)
    : AsyncHotColdRmwContext<key_t, value_t>(faster_hc_, stage_, expected_entry_,
                                            caller_context_, caller_callback_, monotonic_serial_num_)
    {}
  /// The deep-copy constructor.
  HotColdRmwContext(HotColdRmwContext& other_, IAsyncContext* caller_context_)
    : AsyncHotColdRmwContext<key_t, value_t>(other_, caller_context_)
    {}

 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    //typedef HotColdRmwReadContext<typename MC::key_t, typename MC::value_t> hc_rmw_read_context_t;
    //assert(this->read_context == nullptr);

    return IAsyncContext::DeepCopy_Internal(
        *this, HotColdContext<key_t>::caller_context, context_copy);

    /*
    if (this->read_context != nullptr && !this->read_context->from_deep_copy()) {
      // need to deep copy HC-RMW-READ context, if didn't went async
      auto read_context = static_cast<hc_rmw_read_context_t*>(this->read_context);
      read_context->hc_rmw_context = context_copy;
      RETURN_NOT_OK(this->read_context->DeepCopy(this->read_context));
    }
    return Status::Ok;
    */
  }
 private:
  inline const rmw_context_t& rmw_context() const {
    return *static_cast<const rmw_context_t*>(HotColdContext<key_t>::caller_context);
  }
  inline rmw_context_t& rmw_context() {
    return *static_cast<rmw_context_t*>(HotColdContext<key_t>::caller_context);
  }
 public:
  inline uint32_t key_size() const final {
    return hc_context_helper<false>::key_size(rmw_context());
  }
  inline KeyHash get_key_hash() const final {
    return hc_context_helper<false>::get_key_hash(rmw_context());
  }
  inline bool is_key_equal(const key_t& other) const final {
    return hc_context_helper<false>::is_key_equal(rmw_context(), other);
  }
  inline void write_deep_key_at(key_t* dst) const {
    hc_context_helper<false>::write_deep_key_at(rmw_context(), dst);
  }

  /// Set initial value.
  inline void RmwInitial(value_t& value) final {
    rmw_context().RmwInitial(value);
  }
  /// RCU.
  inline void RmwCopy(const value_t& old_value, value_t& value) final {
    rmw_context().RmwCopy(old_value, value);
  }
  /// in-place update.
  inline bool RmwAtomic(value_t& value) final {
    return rmw_context().RmwAtomic(value);
  }
  /// Get value size for initial value or in-place update
  inline constexpr uint32_t value_size() const final {
    return rmw_context().value_size();
  }
  /// Get value size for RCU
  inline constexpr uint32_t value_size(const value_t& value) const final {
    return rmw_context().value_size(value);
  }
};

template<class K, class V>
class HotColdRmwReadContext : public IAsyncContext {
 public:
  //typedef MC rmw_context_t;
  //typedef typename rmw_context_t::key_t key_t;
  //typedef typename rmw_context_t::value_t value_t;
  typedef K key_t;
  typedef V value_t;
  typedef Record<key_t, value_t> record_t;
  typedef AsyncHotColdRmwContext<K, V> hc_rmw_context_t;

  HotColdRmwReadContext(IAsyncContext* hc_rmw_context_)
    : hc_rmw_context{ hc_rmw_context_ }
    , record{ nullptr }
    , deep_copied_{ false }
    {}

  /// The deep-copy constructor.
  HotColdRmwReadContext(HotColdRmwReadContext& other)
    : hc_rmw_context{ other.hc_rmw_context }
    , record{ other.record }
    , deep_copied_{ false }
    {}

  ~HotColdRmwReadContext() {
    if (record != nullptr && (!deep_copied_ || this->from_deep_copy())) {
      delete record;
    }
  }

 protected:
  Status DeepCopy_Internal(IAsyncContext*& context_copy) final {
    typedef AsyncHotColdRmwContext<key_t, value_t> async_hc_rmw_context_t;

    assert(this->hc_rmw_context->from_deep_copy());
    //RETURN_NOT_OK(
    assert(Status::Ok ==
      IAsyncContext::DeepCopy_Internal(*this, context_copy));
    deep_copied_ = true;

    // Update reference of this, to rmw context
    auto hc_rmw_context = static_cast<async_hc_rmw_context_t*>(this->hc_rmw_context);
    hc_rmw_context->read_context = context_copy;

    return Status::Ok;
    //RETURN_NOT_OK(
    //  IAsyncContext::DeepCopy_Internal(*this, context_copy));

    /*
    if (!this->hc_rmw_context->from_deep_copy()) {
      // need to deep copy HC-RMW context, if didn't go async
      hc_rmw_context->read_context = context_copy;
      RETURN_NOT_OK(this->hc_rmw_context->DeepCopy(this->hc_rmw_context));
    }
    return Status::Ok;
    */
  }

 private:
  inline const hc_rmw_context_t& rmw_context() const {
    return *static_cast<const hc_rmw_context_t*>(hc_rmw_context);
  }
  inline hc_rmw_context_t& rmw_context() {
    return *static_cast<hc_rmw_context_t*>(hc_rmw_context);
  }
 public:
  /// Propagates calls to caller context
  inline uint32_t key_size() const {
    return hc_context_helper<true>::key_size(rmw_context());
  }
  inline KeyHash get_key_hash() const {
    return hc_context_helper<true>::get_key_hash(rmw_context());
  }
  inline bool is_key_equal(const key_t& other) const {
    return hc_context_helper<true>::is_key_equal(rmw_context(), other);
  }

 public:
  inline const value_t* value() const {
    assert(this->record != nullptr);
    record_t* record_ = reinterpret_cast<record_t*>(record);
    return &record_->value();
  }

  inline void Get(const void* rec) {
    record_t* record_ = const_cast<record_t*>(reinterpret_cast<const record_t*>(rec));
    assert(record_ != nullptr);
    // TODO: avoid copying
    this->record = new uint8_t[record_->size()];
    memcpy(this->record, record_, record_->size());
  }
  inline void GetAtomic(const void* rec) {
    // This should not be called, since we are doing a request on the cold log
    // TODO: fix
    this->Get(rec);
  }

  IAsyncContext* hc_rmw_context;
  uint8_t* record;

 private:
  bool deep_copied_;
};


template <class RC>
constexpr bool is_hc_read_context = std::integral_constant<bool,
                                  std::is_base_of<AsyncHotColdReadContext<typename RC::key_t, typename RC::value_t>, RC>::value ||
                                  std::is_base_of<HotColdRmwReadContext<typename RC::key_t, typename RC::value_t>, RC>::value>::value;

template <class MC>
constexpr bool is_hc_rmw_context = std::is_base_of<AsyncHotColdRmwContext<typename MC::key_t, typename MC::value_t>, MC>::value;


class HotColdConditionalInsertContext : public IAsyncContext {
  // dummy class to differentiate HotCold ConditionalInsert from compaction one
};
template <class CIC>
constexpr bool is_hc_ci_context = std::is_base_of<HotColdConditionalInsertContext, CIC>::value;

}
} // namespace FASTER::core
