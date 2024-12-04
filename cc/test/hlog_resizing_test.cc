// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <random>

#include "gtest/gtest.h"

#include "core/faster.h"
#include "device/null_disk.h"

#include "test_types.h"
#include "utils.h"

using namespace FASTER::core;
using FASTER::test::FixedSizeKey;
using FASTER::test::VariableSizeKey;
using FASTER::test::VariableSizeShallowKey;

using FASTER::test::SimpleAtomicValue;
using FASTER::test::SimpleAtomicMediumValue;
using FASTER::test::SimpleAtomicLargeValue;

using FASTER::test::GenLock;
using FASTER::test::AtomicGenLock;

using Key = FixedSizeKey<uint64_t>;
using MediumValue = SimpleAtomicMediumValue<uint64_t>;
using LargeValue = SimpleAtomicLargeValue<uint64_t>;

/// Key-value store, specialized to our key and value types.
#ifdef _WIN32
typedef FASTER::environment::ThreadPoolIoHandler handler_t;
#else
typedef FASTER::environment::QueueIoHandler handler_t;
#endif

// Parameterized test definition for in-memory tests
// <init_log_size, new_log_size, resize_at_pct, n_threads>
class HlogResizingInMemTestFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, double, int>> {
};
INSTANTIATE_TEST_CASE_P(
  HlogResizingInMemTests,
  HlogResizingInMemTestFixture,
  ::testing::Values(
    // Grow log
    std::make_tuple(512_MiB, 768_MiB, 0.5, 1),
    std::make_tuple(512_MiB, 768_MiB, 0.5, 4),
    // Shrink log
    std::make_tuple(1024_MiB, 768_MiB, 0.5, 1),
    std::make_tuple(1024_MiB, 768_MiB, 0.5, 4)
  )
);

// Parameterized test definition for on-disk tests
// <init_log_size, new_log_size, resize_at_pct, n_threads>
class HlogResizingOnDiskTestFixture : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, double, int>> {
};
INSTANTIATE_TEST_CASE_P(
  HlogResizingOnDiskTests,
  HlogResizingOnDiskTestFixture,
  ::testing::Values(
    //// Grow log
    /// ===============================================
    // In-mem log not full when start growing even more
    std::make_tuple(768_MiB, 1024_MiB, 0.5, 1),
    std::make_tuple(768_MiB, 1024_MiB, 0.5, 4),
    // In-mem log almost full when start growing even more
    std::make_tuple(512_MiB, 768_MiB, 0.5, 1),
    std::make_tuple(512_MiB, 768_MiB, 0.5, 4),
    // In-mem log already full -- growing it a bit more
    std::make_tuple(384_MiB, 512_MiB, 0.5, 1),
    std::make_tuple(384_MiB, 512_MiB, 0.5, 4),

    /// Shrink log
    /// ===============================================
    // In-mem log not full when starting shrinking
    std::make_tuple(768_MiB, 512_MiB, 0.5, 1),
    std::make_tuple(768_MiB, 512_MiB, 0.5, 4),
    // In-mem log almost full when start shrinking
    std::make_tuple(512_MiB, 384_MiB, 0.5, 1),
    std::make_tuple(512_MiB, 384_MiB, 0.5, 4),
    // In-mem log already full -- shrinking even more
    std::make_tuple(384_MiB, 256_MiB, 0.5, 1),
    std::make_tuple(384_MiB, 256_MiB, 0.5, 4)
  )
);

/// Upsert context required to insert data for unit testing.
template <class K, class V>
class UpsertContext : public IAsyncContext {
 public:
  typedef K key_t;
  typedef V value_t;

  UpsertContext(K key, V value)
    : key_(key)
    , value_(value)
  {}

  /// Copy (and deep-copy) constructor.
  UpsertContext(const UpsertContext& other)
    : key_(other.key_)
    , value_(other.value_)
  {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const K& key() const {
    return key_;
  }
  inline static constexpr uint32_t value_size() {
    return V::size();
  }
  /// Non-atomic and atomic Put() methods.
  inline void Put(V& value) {
    value.value = value_.value;
  }
  inline bool PutAtomic(V& value) {
    value.atomic_value.store(value_.value);
    return true;
  }

 protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

 private:
  K key_;
  V value_;
};

/// Context to read a key when unit testing.
template <class K, class V>
class ReadContext : public IAsyncContext {
 public:
  typedef K key_t;
  typedef V value_t;

  ReadContext(K key)
    : key_(key)
  {}

  /// Copy (and deep-copy) constructor.
  ReadContext(const ReadContext& other)
    : key_(other.key_)
  {}

  /// The implicit and explicit interfaces require a key() accessor.
  inline const K& key() const {
    return key_;
  }

  inline void Get(const V& value) {
    output = value.value;
  }
  inline void GetAtomic(const V& value) {
    output = value.atomic_value.load();
  }

 protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

 private:
  K key_;
 public:
  V output;
};

/// Context to RMW a key when unit testing.
template<class K, class V>
class RmwContext : public IAsyncContext {
 public:
  typedef K key_t;
  typedef V value_t;

  RmwContext(key_t key, value_t incr)
    : key_{ key }
    , incr_{ incr } {
  }

  /// Copy (and deep-copy) constructor.
  RmwContext(const RmwContext& other)
    : key_{ other.key_ }
    , incr_{ other.incr_ } {
  }

  /// The implicit and explicit interfaces require a key() accessor.
  inline const key_t& key() const {
    return key_;
  }
  inline static constexpr uint32_t value_size() {
    return sizeof(value_t);
  }
  inline static constexpr uint32_t value_size(const value_t& old_value) {
    return sizeof(value_t);
  }
  inline void RmwInitial(value_t& value) {
    value.value = incr_.value;
  }
  inline void RmwCopy(const value_t& old_value, value_t& value) {
    value.value = old_value.value + incr_.value;
  }
  inline bool RmwAtomic(value_t& value) {
    value.atomic_value.fetch_add(incr_.value);
    return true;
  }

  protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }

 private:
  key_t key_;
  value_t incr_;
};

/// Context to delete a key when unit testing.
template<class K, class V>
class DeleteContext : public IAsyncContext {
 private:
  K key_;

 public:
  typedef K key_t;
  typedef V value_t;

  explicit DeleteContext(const K& key)
    : key_(key)
  {}

  inline const K& key() const {
    return key_;
  }

  inline static constexpr uint32_t value_size() {
    return V::size();
  }

 protected:
  /// The explicit interface requires a DeepCopy_Internal() implementation.
  Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
  }
};

class KeyRange {
 public:
  static constexpr uint32_t kDefaultChunkSize = 320;

  KeyRange(uint64_t min, uint64_t max, uint32_t chunk_size = kDefaultChunkSize)
    : min_{ min }
    , max_{ max }
    , chunk_size_{ chunk_size }
  {
      pos_.store(min_);
  }

  bool GetNext(uint64_t& from, uint64_t& to) {
    from = pos_.fetch_add(chunk_size_);
    to = std::min(from + chunk_size_, max_);
    return (from < max_);
  }

 private:
  uint64_t min_;
  uint64_t max_;
  uint32_t chunk_size_;

  std::atomic<uint64_t> pos_;
};

// ****************************************************************************
// IN-MEMORY TESTS
// ****************************************************************************

/// Inserts a bunch of records into a FASTER instance and invokes the
/// log resizing mechanism. Since all records are still live, checks if
/// they remain so after the algorithm completes/returns.
TEST_P(HlogResizingInMemTestFixture, InMemAllLive) {
  // In memory hybrid log
  typedef FasterKv<Key, MediumValue, FASTER::device::NullDisk> faster_t;
  constexpr size_t num_records = 600'000; // ~615 MiB

  auto& params = GetParam();
  uint64_t init_log_size = std::get<0>(params);
  uint64_t new_log_size = std::get<1>(params);
  double resize_at_pct = std::get<2>(params);
  int num_threads = std::get<3>(params);

  faster_t store { 262144, init_log_size, "tmp_store", 0.6 };

  KeyRange range{ 1, num_records + 1};
  uint64_t resize_at_index = static_cast<uint64_t>(num_records * resize_at_pct);

  log_info("Initial Log Size: %.2lf MiB", static_cast<double>(init_log_size / (1 << 20)));
  log_info("--> New Log Size: %.2lf MiB", static_cast<double>(new_log_size / (1 << 20)));
  log_info("--> Resize at   : %.2lf%% [index: %lu]", resize_at_pct * 100, resize_at_index);
  log_info("--> Num Threads : %d", num_threads);

  auto upsert_func = [&]() {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };

    uint64_t from, to;
    uint64_t num_reqs = 0;

    store.StartSession();
    while (range.GetNext(from, to)) {
      for (uint64_t idx = from; idx < to; ++idx) {
        UpsertContext<Key, MediumValue> context{ Key(idx), MediumValue(idx) };
        Status result = store.Upsert(context, callback, 1);
        ASSERT_EQ(Status::Ok, result);

        if (++num_reqs % 128 == 0) {
          store.CompletePending(false);
        }
        if (idx == resize_at_index) {
          log_info("Initiating hlog resizing!");
          store.hlog.Resize(new_log_size);
        }
      }
    }
    store.CompletePending(true);
    store.StopSession();
  };

  std::deque<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(upsert_func);
  }
  for (auto& t : threads) {
    t.join();
  }

  // Validate
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback should not be called
      ASSERT_TRUE(false);
    };
    ReadContext<Key, MediumValue> context{ idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    ASSERT_EQ(idx, context.output.value);
  }
  store.CompletePending(true);
  store.StopSession();
}

/// Inserts a bunch of records into a FASTER instance, updates half of them
/// with new values and invokes the log resizing mechanism. Checks that the
/// updated ones have the new value, and the others the old one.
TEST_P(HlogResizingInMemTestFixture, InMemAllLiveNewEntries) {
  // In memory hybrid log
  typedef FasterKv<Key, MediumValue, FASTER::device::NullDisk> faster_t;
  typedef uint64_t (*value_func_t)(uint64_t);

  constexpr size_t num_records = 400'000; // ~615 MiB (2x)
  assert(num_records % 2 == 0); // required for new entries resizing code to work

  auto& params = GetParam();
  uint64_t init_log_size = std::get<0>(params);
  uint64_t new_log_size = std::get<1>(params);
  double resize_at_pct = std::get<2>(params);
  int num_threads = std::get<3>(params);

  faster_t store { 262144, init_log_size, "tmp_store", 0.0 };

  uint64_t resize_at_index = (num_records + 1) - static_cast<uint64_t>(2 * num_records * resize_at_pct);
  resize_at_index += (resize_at_index % 2);
  assert(resize_at_index % 2 == 0);

  log_info("Initial Log Size: %.2lf MiB", static_cast<double>(init_log_size / (1 << 20)));
  log_info("--> New Log Size: %.2lf MiB", static_cast<double>(new_log_size / (1 << 20)));
  log_info("--> Resize at   : %.2lf%% [index: %lu]", resize_at_pct * 100, resize_at_index);
  log_info("--> Num Threads : %d", num_threads);

  auto upsert_func = [&](KeyRange& range, value_func_t value_func, bool update_keys) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };

    uint64_t from, to;
    uint64_t num_reqs = 0;
    const uint64_t inc = update_keys ? 2 : 1;

    store.StartSession();
    while (range.GetNext(from, to)) {
      if (update_keys) {
        from += (from % 2); // update only even keys
      }
      for (uint64_t idx = from; idx < to; idx += inc) {
        UpsertContext<Key, MediumValue> context{ Key(idx), MediumValue( value_func(idx) ) };
        Status result = store.Upsert(context, callback, 1);
        ASSERT_EQ(Status::Ok, result);

        if (++num_reqs % 128 == 0) {
          store.CompletePending(false);
        }
        if (update_keys && idx == resize_at_index) {
          log_info("Initiating hlog resizing!");
          store.hlog.Resize(new_log_size);
        }
      }
    }
    store.CompletePending(true);
    store.StopSession();
  };

  // Insert all keys with initial values
  {
    auto value_func = [](uint64_t idx) { return idx; };
    KeyRange range{ 1, num_records + 1 };

    std::deque<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
      threads.emplace_back(upsert_func, std::ref(range), value_func, false);
    }
    for (auto& t : threads) {
      t.join();
    }
  }
  // Update all keys with new values
  {
    auto value_func = [](uint64_t idx) { return 2 * idx; };
    KeyRange range{ 1, num_records + 1 };

    std::deque<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
      threads.emplace_back(upsert_func, std::ref(range), value_func, true);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  // After compaction, reads should return newer values
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };
    ReadContext<Key, MediumValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    ASSERT_EQ(result, Status::Ok);
    Key expected_key{ (idx % 2 == 1)
                        ? context.output.value
                        : context.output.value / 2};
    ASSERT_EQ(idx, expected_key.key);
  }
  store.CompletePending(true);
  store.StopSession();
}

/// Inserts a bunch of records into a FASTER instance, and invokes the
/// resizing algorithm. Concurrent to the resizing, upserts and deletes
/// are performed in 1/3 of the keys, respectively. After compaction, it
/// checks that updated keys have the new value, while deleted keys do not exist.
TEST_P(HlogResizingInMemTestFixture, InMemConcurrentOps) {
  // In memory hybrid log
  typedef FASTER::device::NullDisk disk_t;
  typedef FasterKv<Key, MediumValue, disk_t> faster_t;
  constexpr size_t num_records = 400'000; // ~410 MiB

  auto& params = GetParam();
  uint64_t init_log_size = std::get<0>(params);
  uint64_t new_log_size = std::get<1>(params);
  double resize_at_pct = std::get<2>(params);
  int num_threads = std::get<3>(params);

  faster_t store { 262144, init_log_size, "tmp_store", 0.0 };

  KeyRange range{ 1, num_records + 1 };
  uint64_t resize_at_index = (num_records + 1) - static_cast<uint64_t>(2 * num_records * resize_at_pct);

  log_info("Initial Log Size: %.2lf MiB", static_cast<double>(init_log_size / (1 << 20)));
  log_info("--> New Log Size: %.2lf MiB", static_cast<double>(new_log_size / (1 << 20)));
  log_info("--> Resize at   : %.2lf%% [index: %lu]", resize_at_pct * 100, resize_at_index);
  log_info("--> Num Threads : %d", num_threads);

  auto worker_func = [&]() {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };

    uint64_t from, to;
    uint64_t num_reqs = 0;

    store.StartSession();
    while (range.GetNext(from, to)) {
      for (uint64_t idx = from; idx < to; idx ++) {
        if (idx % 3 == 0) { // Update value
          UpsertContext<Key, MediumValue> context{ Key(idx), MediumValue(2*idx) };
          Status result = store.Upsert(context, callback, 1);
          ASSERT_EQ(Status::Ok, result);
        } else if (idx % 3 == 1) { // Delete key
          DeleteContext<Key, MediumValue> context{ Key(idx) };
          Status result = store.Delete(context, callback, idx);
          ASSERT_EQ(Status::Ok, result);
        }

        if (++num_reqs % 128 == 0) {
          store.CompletePending(false);
        }
        if (idx == resize_at_index) {
          log_info("Initiating hlog resizing!");
          store.hlog.Resize(new_log_size);
        }
      }
    }
    store.CompletePending(true);
    store.StopSession();
  };

  // Populate initial keys
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, MediumValue> context{Key(idx), MediumValue(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }
  store.CompletePending(true);
  store.StopSession();

  // Launch threads
  std::deque<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker_func);
  }
  for (auto& t : threads) {
    t.join();
  }

  // Reads should return newer values for non-deleted entries
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };
    ReadContext<Key, MediumValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);

    if (idx % 3 == 0) {
      ASSERT_EQ(result, Status::Ok);
      ASSERT_EQ(idx, context.output.value / 2);
    } else if (idx % 3 == 1) {
      ASSERT_EQ(result, Status::NotFound);
    } else { // idx % 3 == 2
      ASSERT_EQ(result, Status::Ok);
      ASSERT_EQ(idx, context.output.value);
    }
  }
  store.CompletePending(true);
  store.StopSession();
}

/*
TEST_P(HlogResizingInMemTestFixture, InMemVariableLengthKey) {
  using Key = VariableSizeKey;
  using ShallowKey = VariableSizeShallowKey;
  using Value = MediumValue;

  class UpsertContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    UpsertContext(uint32_t* key, uint32_t key_length, value_t value)
      : key_{ key, key_length }
      , value_{ value } {
    }

    /// Copy (and deep-copy) constructor.
    UpsertContext(const UpsertContext& other)
      : key_{ other.key_ }
      , value_{ other.value_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }
    inline static constexpr uint32_t value_size() {
      return sizeof(value_t);
    }
    /// Non-atomic and atomic Put() methods.
    inline void Put(value_t& value) {
      value.value = value_.value;
    }
    inline bool PutAtomic(value_t& value) {
      value.atomic_value.store(value_.value);
      return true;
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
    return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    ShallowKey key_;
    value_t value_;
  };

  class ReadContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    ReadContext(uint32_t* key, uint32_t key_length)
            : key_{ key, key_length } {
    }

    /// Copy (and deep-copy) constructor.
    ReadContext(const ReadContext& other)
            : key_{ other.key_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }

    inline void Get(const value_t& value) {
      output.value = value.value;
    }
    inline void GetAtomic(const value_t& value) {
      output.value = value.atomic_value.load();
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    ShallowKey key_;
   public:
    value_t output;
  };

  class RmwContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    RmwContext(uint32_t* key, uint32_t key_length, value_t incr)
      : key_{ key, key_length }
      , incr_{ incr } {
    }

    /// Copy (and deep-copy) constructor.
    RmwContext(const RmwContext& other)
      : key_{ other.key_ }
      , incr_{ other.incr_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }
    inline static constexpr uint32_t value_size() {
      return sizeof(value_t);
    }
    inline static constexpr uint32_t value_size(const value_t& old_value) {
      return sizeof(value_t);
    }
    inline void RmwInitial(value_t& value) {
      value.value = incr_.value;
    }
    inline void RmwCopy(const value_t& old_value, value_t& value) {
      value.value = old_value.value + incr_.value;
    }
    inline bool RmwAtomic(value_t& value) {
      value.atomic_value.fetch_add(incr_.value);
      return true;
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    ShallowKey key_;
    Value incr_;
  };

  class DeleteContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    explicit DeleteContext(uint32_t* key, uint32_t key_length)
      : key_{ key, key_length }
    {}

    /// Copy (and deep-copy) constructor.
    DeleteContext(const DeleteContext& other)
            : key_{ other.key_ } {
    }
    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }
    inline static constexpr uint32_t value_size() {
      return sizeof(value_t);
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }
   private:
    ShallowKey key_;
  };


  typedef FasterKv<Key, Value, FASTER::device::NullDisk> faster_t;
  faster_t store { 2048, (1 << 30), "", 0.0625 }; // 64 MB of mutable region
  uint32_t numRecords = 12500; // will occupy ~512 MB space in store

  auto& params = GetParam();
  int delta = std::get<0>(params);
  int n_threads = std::get<1>(params);

  store.StartSession();
  // Insert.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false); // upserts do not go pending
    };
    // Create the key as a variable length array
    uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      key[j] = j;
    }

    UpsertContext context{ key, idx, idx};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    free(key);
  }
  // Read.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false); // In-memory test.
    };
    // Create the key as a variable length array
    uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      key[j] = j;
    }

    ReadContext context{ key, idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    ASSERT_EQ(idx, context.output.value);
    free(key);
  }
  // Update one fourth of the entries
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 4 == 0) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false); // upserts do not go pending
      };
      // Create the key as a variable length array
      uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        key[j] = j;
      }

      UpsertContext context{ key, idx, 2*idx };
      Status result = store.Upsert(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
      free(key);
    }
  }
  // Delete another one fourth
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 4 == 1) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false); // deletes do no go pending
      };
      // Create the key as a variable length array
      uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        key[j] = j;
      }

      DeleteContext context{ key, idx };
      Status result = store.Delete(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
      free(key);
    }
  }
  // RMW another one fourth
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 4 == 2) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_EQ(result, Status::Ok);
        // free memory
        CallbackContext<RmwContext> context{ ctxt };
        free(context->key().key_data_);
      };
      // Create the key as a variable length array
      uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        key[j] = j;
      }

      RmwContext context{ key, idx, idx };
      Status result = store.Rmw(context, callback, 1);
      ASSERT_TRUE(result == Status::Ok || result == Status::Pending);
      if (result == Status::Ok) {
        free(key);
      }
    }
  }
  store.CompletePending(true);

  // perform resizing
  store.hlog.Resize(delta > 0);
  store.CompletePending(true);

  // Read again.
  for(uint32_t idx = 1; idx <= numRecords ; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false); // In-memory test.
    };
    // Create the key as a variable length array
    uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      key[j] = j;
    }

    ReadContext context{ key, idx };
    Status result = store.Read(context, callback, 1);
    // All upserts should have updates (atomic).
    if (idx % 4 == 0) {
      ASSERT_EQ(Status::Ok, result);
      ASSERT_EQ(2*idx, context.output.value);
    } else if (idx % 4 == 1) {
      ASSERT_EQ(Status::NotFound, result);
    } else if (idx % 4 == 2) {
      ASSERT_EQ(Status::Ok, result);
      ASSERT_EQ(2*idx, context.output.value);
    } else { // idx % 4 == 3
      ASSERT_EQ(Status::Ok, result);
      ASSERT_EQ(idx, context.output.value);
    }
    free(key);
  }
  store.StopSession();
}

TEST_P(HlogResizingInMemTestFixture, InMemVariableLengthValue) {
  using Key = FixedSizeKey<uint32_t>;

  class UpsertContextVLV;
  class ReadContextVLV;

  class Value {
   public:
    Value()
      : gen_lock_{ 0 }
      , size_{ 0 }
      , length_{ 0 } {
    }

    inline uint32_t size() const {
      return size_;
    }

    friend class UpsertContextVLV;
    friend class ReadContextVLV;

   private:
    AtomicGenLock gen_lock_;
    uint32_t size_;
    uint32_t length_;

    inline const uint8_t* buffer() const {
      return reinterpret_cast<const uint8_t*>(this + 1);
    }
    inline uint8_t* buffer() {
      return reinterpret_cast<uint8_t*>(this + 1);
    }
  };

  class UpsertContextVLV : public IAsyncContext {
   public:
    typedef Key key_t;
    typedef Value value_t;

    UpsertContextVLV(uint32_t key, uint32_t *value, uint32_t value_length)
      : key_{ key }
      , value_{ value }
      , value_length_{ value_length } {
    }
    /// Copy (and deep-copy) constructor.
    UpsertContextVLV(const UpsertContextVLV& other)
      : key_{ other.key_ }
      , value_{ other.value_ }
      , value_length_{ other.value_length_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const key_t& key() const {
      return key_;
    }
    inline uint32_t value_size() const {
      return sizeof(value_t) + value_length_ * sizeof(uint32_t);
    }
    /// Non-atomic and atomic Put() methods.
    inline void Put(value_t& value) {
      value.gen_lock_.store(0);
      value.size_ = value_size();
      value.length_ = value_length_;
      std::memcpy(value.buffer(), value_, value_length_ * sizeof(uint32_t));
    }
    inline bool PutAtomic(value_t& value) {
      bool replaced;
      while(!value.gen_lock_.try_lock(replaced) && !replaced) {
        std::this_thread::yield();
      }
      if(replaced) {
        // Some other thread replaced this record.
        return false;
      }
      if(value.size_ < value_size()) {
        // Current value is too small for in-place update.
        value.gen_lock_.unlock(true);
        return false;
      }
      // In-place update overwrites length and buffer, but not size.
      value.length_ = value_length_;
      std::memcpy(value.buffer(), value_, value_length_ * sizeof(uint32_t));
      value.gen_lock_.unlock(false);
      return true;
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    key_t key_;
    uint32_t* value_;
    uint32_t value_length_;
  };

  class ReadContextVLV : public IAsyncContext {
   public:
    typedef Key key_t;
    typedef Value value_t;

    ReadContextVLV(uint32_t key)
      : key_{ key }
      , output{ nullptr }
      , output_length{ 0 } {
    }

    /// Copy (and deep-copy) constructor.
    ReadContextVLV(const ReadContextVLV& other)
      : key_{ other.key_ }
      , output{ other.output }
      , output_length{ other.output_length }
    { }

    ~ReadContextVLV() {
      if (output != nullptr) {
        free(output);
      }
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const key_t& key() const {
      return key_;
    }
    inline void Get(const value_t& value) {
      output_length = value.length_;
      if (output == nullptr) {
        output = (uint32_t*) malloc(output_length * sizeof(uint32_t));
      }
      std::memcpy(output, value.buffer(), output_length * sizeof(uint32_t));
    }
    inline void GetAtomic(const value_t& value) {
      GenLock before, after;
      do {
        before = value.gen_lock_.load();
        output_length = value.length_;
        if (output == nullptr) {
          output = (uint32_t*) malloc(output_length * sizeof(uint32_t));
        }
        std::memcpy(output, value.buffer(), output_length * sizeof(uint32_t));
        after = value.gen_lock_.load();
      } while(before.gen_number != after.gen_number);
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    key_t key_;
   public:
    uint32_t *output;
    uint32_t output_length;
  };

  using UpsertContext = UpsertContextVLV;
  using ReadContext = ReadContextVLV;

  typedef FasterKv<Key, Value, FASTER::device::NullDisk> faster_t;
  faster_t store { 2048, (1 << 30), "", 0.0625 }; // 64 MB of mutable region
  uint32_t numRecords = 12500; // will occupy ~512 MB space in store

  auto& params = GetParam();
  int delta = std::get<0>(params);
  int n_threads = std::get<1>(params);

  store.StartSession();

  // Insert.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false); // upserts do not go pending
    };
    // Create the value as a variable length array
    uint32_t* value = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      value[j] = idx;
    }

    UpsertContext context{ idx, value, idx};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    free(value);
  }
  // Read.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false); // In-memory test.
    };

    ReadContext context{ idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    // check each position of the var-len value
    for (uint32_t j = 0; j < idx; ++j) {
      ASSERT_EQ(context.output[j], idx);
    }
  }
  // Update half
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 2 == 0) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false); // upserts do not go pending
      };
      // Create the value as a variable length array
      uint32_t* value = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        value[j] = 2 * idx;
      }

      UpsertContext context{ idx, value, idx }; // double the value_id
      Status result = store.Upsert(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
      free(value);
    }
  }

  // perform resizing
  store.hlog.Resize(delta > 0);
  store.CompletePending(true);

  // Read again.
  for(uint32_t idx = 1; idx <= numRecords ; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false); // In-memory test.
    };

    ReadContext context{ idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_EQ(result, Status::Ok);
    // check each position of the var-len value
    uint32_t value_id = (idx % 2 == 0) ? 2*idx : idx;
    for (uint32_t j = 0; j < idx; ++j) {
      ASSERT_EQ(context.output[j], value_id);
    }
  }
  store.StopSession();
}
*/

// ****************************************************************************
// PERSISTENCE STORAGE TESTS
// ****************************************************************************

/// Inserts a bunch of records into a FASTER instance and invokes the
/// hlog resizing method. Since all records are still live, checks if
/// they remain so after the algorithm completes.
TEST_P(HlogResizingOnDiskTestFixture, OnDiskAllLive) {
  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t; // 1GB file segments
  typedef FasterKv<Key, LargeValue, disk_t> faster_t;
  constexpr size_t num_records = 100'000; // ~820 MiB

  auto& params = GetParam();
  uint64_t init_log_size = std::get<0>(params);
  uint64_t new_log_size = std::get<1>(params);
  double resize_at_pct = std::get<2>(params);
  int num_threads = std::get<3>(params);

  std::experimental::filesystem::create_directories("tmp_store");
  faster_t store { 262144, init_log_size, "tmp_store", 0.6 };

  KeyRange range{ 1, num_records + 1};
  uint64_t resize_at_index = static_cast<uint64_t>(num_records * resize_at_pct);

  log_info("Initial Log Size: %.2lf MiB", static_cast<double>(init_log_size / (1 << 20)));
  log_info("--> New Log Size: %.2lf MiB", static_cast<double>(new_log_size / (1 << 20)));
  log_info("--> Resize at   : %.2lf%% [index: %lu]", resize_at_pct * 100, resize_at_index);
  log_info("--> Num Threads : %d", num_threads);

  auto upsert_func = [&]() {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback should not be called
      ASSERT_TRUE(false);
    };
    uint64_t from, to;
    uint64_t num_reqs = 0;

    store.StartSession();
    while (range.GetNext(from, to)) {
      for (uint64_t idx = from; idx < to; ++idx) {
        UpsertContext<Key, LargeValue> context{ Key(idx), LargeValue(idx) };
        Status result = store.Upsert(context, callback, 1);
        ASSERT_EQ(Status::Ok, result);

        if (++num_reqs % 128 == 0) {
          store.CompletePending(false);
        }
        if (idx == resize_at_index) {
          log_info("Initiating HLOG resizing! Store size: %.3lf MiB",
                  static_cast<double>(store.Size()) / (1 << 20UL));
          store.hlog.Resize(new_log_size);
        }
      }
    }
    store.CompletePending(true);
  };

  std::deque<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(upsert_func);
  }
  for (auto& t : threads) {
    t.join();
  }
  log_info("Finished populating all keys! Store size: %.3lf MiB",
           static_cast<double>(store.Size()) / (1 << 20UL));

  // Check that all entries are present
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_EQ(Status::Ok, result);

      CallbackContext<ReadContext<Key, LargeValue>> context(ctxt);
      ASSERT_TRUE(context->key().key > 0);
      ASSERT_EQ(context->key(), context->output.value);
    };
    ReadContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    EXPECT_TRUE(result == Status::Ok || result == Status::Pending);
    if (result == Status::Ok) {
      ASSERT_EQ(idx, context.output.value);
    }

    if (idx % 128 == 0) {
      // occasionally complete pending I/O requests
      store.CompletePending(false);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

/// Inserts a bunch of records into a FASTER instance, deletes half of them,
/// re-inserts them with new values, while invoking the log resizing algorithm.
/// Checks that the updated ones have the new value, and the rest remain unchanged.
TEST_P(HlogResizingOnDiskTestFixture, OnDiskAllLiveDeleteAndReInsert) {
  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t; // 1GB file segments
  typedef FasterKv<Key, LargeValue, disk_t> faster_t;
  constexpr size_t num_records = 50'000; // ~410 MiB

  auto& params = GetParam();
  uint64_t init_log_size = std::get<0>(params);
  uint64_t new_log_size = std::get<1>(params);
  double resize_at_pct = std::get<2>(params);
  int num_threads = std::get<3>(params);

  std::experimental::filesystem::create_directories("tmp_store");
  faster_t store { 262144, init_log_size, "tmp_store", 0.6 };

  uint64_t resize_at_index = static_cast<uint64_t>(num_records * resize_at_pct);

  log_info("Initial Log Size: %.2lf MiB", static_cast<double>(init_log_size / (1 << 20)));
  log_info("--> New Log Size: %.2lf MiB", static_cast<double>(new_log_size / (1 << 20)));
  log_info("--> Resize at   : %.2lf%% [index: %lu]", resize_at_pct * 100, resize_at_index);
  log_info("--> Num Threads : %d", num_threads);

  auto delete_worker_func = [&](KeyRange& range) {
    uint64_t from, to;
    uint64_t num_reqs = 0;

    store.StartSession();
    while (range.GetNext(from, to)) {
      from += (from % 2);
      for (size_t idx = from; idx < to; idx += 2) {
        auto callback = [](IAsyncContext* ctxt, Status result) {
          ASSERT_TRUE(false);
        };
        DeleteContext<Key, LargeValue> context{ Key(idx) };
        Status result = store.Delete(context, callback, 1);
        ASSERT_EQ(Status::Ok, result);

        if (++num_reqs % 128 == 0) {
          store.CompletePending(false);
        }
        if (idx == resize_at_index) {
          log_info("Initiating HLOG resizing! Store size: %.3lf MiB",
                  static_cast<double>(store.Size()) / (1 << 20UL));
          store.hlog.Resize(new_log_size);
        }
      }
    }
    store.CompletePending(true);
    store.StopSession();
  };

  auto update_worker_func = [&](KeyRange& range) {
    uint64_t from, to;
    uint64_t num_reqs = 0;

    store.StartSession();
    while (range.GetNext(from, to)) {
      from += (from % 2);
      for (size_t idx = from; idx < to; idx += 2) {
        auto callback = [](IAsyncContext* ctxt, Status result) {
          ASSERT_TRUE(false);
        };
        UpsertContext<Key, LargeValue> context{ Key(idx), LargeValue(2 * idx) };
        Status result = store.Upsert(context, callback, 1);
        ASSERT_EQ(Status::Ok, result);
      }
      if (++num_reqs % 128 == 0) {
        store.CompletePending(false);
      }
    }
    store.CompletePending(true);
    store.StopSession();
  };

  // Initially populate with all keys
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, LargeValue> context{Key(idx), LargeValue(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }
  store.CompletePending(true);
  store.StopSession();
  log_info("Finished populating all keys! Store size: %.3lf MiB",
           static_cast<double>(store.Size()) / (1 << 20UL));

  {
    // Delete every alternate key here.
    KeyRange range{ 1, num_records + 1};
    std::deque<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
      threads.emplace_back(delete_worker_func, std::ref(range));
    }
    for (auto& t : threads) {
      t.join();
    }
    log_info("Finished deleting alternative keys! Store size: %.3lf MiB",
             static_cast<double>(store.Size()) / (1 << 20UL));
  }

  {
    // Insert fresh entries for the alternate keys
    KeyRange range{ 1, num_records + 1};
    std::deque<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
      threads.emplace_back(update_worker_func, std::ref(range));
    }
    for (auto& t : threads) {
      t.join();
    }
    log_info("Finished updating alternative keys! Store size: %.3lf MiB",
             static_cast<double>(store.Size()) / (1 << 20UL));
  }

  // After compaction, all entries should exist
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_EQ(Status::Ok, result);

      CallbackContext<ReadContext<Key, LargeValue>> context(ctxt);
      ASSERT_TRUE(context->key().key > 0);
      Key expected_key {(context->key().key % 2 == 1)
                            ? context->output.value
                            : context->output.value / 2};
      ASSERT_EQ(context->key().key, expected_key.key);
    };

    ReadContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    EXPECT_TRUE(result == Status::Ok || result == Status::Pending);
    if (result == Status::Ok) {
      Key expected_key {(idx % 2 == 1)
                          ? context.output.value
                          : context.output.value / 2};
      ASSERT_EQ(idx, expected_key.key);
    }

    if (idx % 128 == 0) {
      store.CompletePending(false);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

/// Inserts a bunch of records into a FASTER instance, and invokes the
/// log resizing algorithm. Concurrent to the compaction, upserts and deletes
/// are performed in 1/3 of the keys, respectively. After log resizing, it
/// checks that updated keys have the new value, while deleted keys do not exist.
TEST_P(HlogResizingOnDiskTestFixture, OnDiskConcurrentOps) {
  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t; // 1GB file segments
  typedef FasterKv<Key, LargeValue, disk_t> faster_t;
  static constexpr size_t num_records = 50'000; // ~410 MiB

  std::experimental::filesystem::create_directories("tmp_store");

  auto& params = GetParam();
  uint64_t init_log_size = std::get<0>(params);
  uint64_t new_log_size = std::get<1>(params);
  double resize_at_pct = std::get<2>(params);
  int num_threads = std::get<3>(params);

  faster_t store { 262144, init_log_size, "tmp_store", 0.0 };

  KeyRange range{ 1, num_records + 1 };
  uint64_t resize_at_index = (num_records + 1) - static_cast<uint64_t>(2 * num_records * resize_at_pct);

  log_info("Initial Log Size: %.2lf MiB", static_cast<double>(init_log_size / (1 << 20)));
  log_info("--> New Log Size: %.2lf MiB", static_cast<double>(new_log_size / (1 << 20)));
  log_info("--> Resize at   : %.2lf%% [index: %lu]", resize_at_pct * 100, resize_at_index);
  log_info("--> Num Threads : %d", num_threads);

  auto worker_func = [&]() {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };

    uint64_t from, to;
    uint64_t num_reqs = 0;

    store.StartSession();
    while (range.GetNext(from, to)) {
      for (uint64_t idx = from; idx < to; idx ++) {
        if (idx % 3 == 0) { // Update value
          UpsertContext<Key, LargeValue> context{ Key(idx), LargeValue(2*idx) };
          Status result = store.Upsert(context, callback, 1);
          ASSERT_EQ(Status::Ok, result);
        } else if (idx % 3 == 1) { // Delete key
          DeleteContext<Key, LargeValue> context{ Key(idx) };
          Status result = store.Delete(context, callback, idx);
          ASSERT_EQ(Status::Ok, result);
        }

        if (++num_reqs % 128 == 0) {
          store.CompletePending(false);
        }
        if (idx == resize_at_index) {
          log_info("Initiating HLOG resizing! Store size: %.3lf MiB",
                  static_cast<double>(store.Size()) / (1 << 20UL));
          store.hlog.Resize(new_log_size);
        }
      }
    }
    store.CompletePending(true);
    store.StopSession();
  };

  // Populate initial keys
  store.StartSession();
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, LargeValue> context{Key(idx), LargeValue(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }
  store.CompletePending(true);
  store.StopSession();
  log_info("Finished populating all keys! Store size: %.3lf MiB",
           static_cast<double>(store.Size()) / (1 << 20UL));

  // Launch threads
  std::deque<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker_func);
  }
  for (auto& t : threads) {
    t.join();
  }
  log_info("Finished concurrent ops! Store size: %.3lf MiB",
           static_cast<double>(store.Size()) / (1 << 20UL));

  store.StartSession();
  // Reads should return newer values for non-deleted entries
  for (size_t idx = 1; idx <= num_records; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      CallbackContext<ReadContext<Key, LargeValue>> context(ctxt);
      ASSERT_TRUE(context->key().key > 0);
      if (context->key().key % 3 == 0) {
        ASSERT_EQ(Status::Ok, result);
        ASSERT_EQ(context->key().key, context->output.value / 2);
      } else if (context->key().key % 3 == 1) {
        ASSERT_EQ(Status::NotFound, result);
      } else {
        ASSERT_EQ(Status::Ok, result);
        ASSERT_EQ(context->key().key, context->output.value);
      }
    };
    ReadContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    EXPECT_TRUE(result == Status::Ok || result == Status::NotFound || result == Status::Pending);

    if (result == Status::Ok) {
      if (idx % 3 == 0) { // (up)inserted
        ASSERT_EQ(idx, context.output.value / 2);
      } else if (idx % 3 == 2) { // unmodified
        ASSERT_EQ(idx, context.output.value);
      } else {
        ASSERT_TRUE(false);
      }
    } else if (result == Status::NotFound) {
      ASSERT_TRUE(idx % 3 == 1); // deleted
    }

    if (idx % 20 == 0) {
      store.CompletePending(false);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

/*
TEST_P(CompactLookupParameterizedOnDiskTestFixture, OnDiskVariableLengthKey) {
  using Key = VariableSizeKey;
  using ShallowKey = VariableSizeShallowKey;
  using Value = MediumValue;

  class UpsertContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    UpsertContext(uint32_t* key, uint32_t key_length, Value value)
      : key_{ key, key_length }
      , value_{ value } {
    }

    /// Copy (and deep-copy) constructor.
    UpsertContext(const UpsertContext& other)
      : key_{ other.key_ }
      , value_{ other.value_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }
    inline static constexpr uint32_t value_size() {
      return sizeof(value_t);
    }
    /// Non-atomic and atomic Put() methods.
    inline void Put(Value& value) {
      value.value = value_.value;
    }
    inline bool PutAtomic(Value& value) {
      value.atomic_value.store(value_.value);
      return true;
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    ShallowKey key_;
    Value value_;
  };

  class ReadContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    ReadContext(uint32_t* key, uint32_t key_length)
            : key_{ key, key_length } {
    }

    /// Copy (and deep-copy) constructor.
    ReadContext(const ReadContext& other)
            : key_{ other.key_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }

    inline void Get(const value_t& value) {
      output.value = value.value;
    }
    inline void GetAtomic(const value_t& value) {
      output.value = value.atomic_value.load();
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    ShallowKey key_;
   public:
    value_t output;
  };

  class RmwContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    RmwContext(uint32_t* key, uint32_t key_length, value_t incr)
      : key_{ key, key_length }
      , incr_{ incr } {
    }

    /// Copy (and deep-copy) constructor.
    RmwContext(const RmwContext& other)
      : key_{ other.key_ }
      , incr_{ other.incr_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }
    inline static constexpr uint32_t value_size() {
      return sizeof(value_t);
    }
    inline static constexpr uint32_t value_size(const value_t& old_value) {
      return sizeof(value_t);
    }
    inline void RmwInitial(value_t& value) {
      value.value = incr_.value;
    }
    inline void RmwCopy(const value_t& old_value, value_t& value) {
      value.value = old_value.value + incr_.value;
    }
    inline bool RmwAtomic(value_t& value) {
      value.atomic_value.fetch_add(incr_.value);
      return true;
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    ShallowKey key_;
    Value incr_;
  };

  class DeleteContext : public IAsyncContext {
   public:
    // Typedef required for *PendingContext instances
    // but compiler throws warnings
    [[maybe_unused]] typedef Key key_t;
    typedef Value value_t;

    explicit DeleteContext(uint32_t* key, uint32_t key_length)
      : key_{ key, key_length }
    {}

    /// Copy (and deep-copy) constructor.
    DeleteContext(const DeleteContext& other)
            : key_{ other.key_ } {
    }
    /// The implicit and explicit interfaces require a key() accessor.
    inline const ShallowKey& key() const {
      return key_;
    }
    inline static constexpr uint32_t value_size() {
      return sizeof(value_t);
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }
   private:
    ShallowKey key_;
  };

  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t;
  typedef FasterKv<Key, Value, disk_t> faster_t;

  std::experimental::filesystem::create_directories("tmp_store");

  faster_t store { (1 << 20), (1 << 20) * 192, "tmp_store", 0.4 };
  uint32_t numRecords = 12500; // will occupy ~512 MB space in store

  bool shift_begin_address = std::get<0>(GetParam());
  bool checkpoint = std::get<1>(GetParam());
  int n_threads = std::get<2>(GetParam());

  log_debug("Compaction Threads:  %d", n_threads);
  log_debug("Shift Begin Address: %s", shift_begin_address ? "ENABLED" : "DISABLED");
  log_debug("Checkpoint:          %s", checkpoint ? "ENABLED" : "DISABLED");

  store.StartSession();
  // Insert.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // Writes do not go pending in normal operation
      ASSERT_TRUE(false);
    };
    // Create the key as a variable length array
    uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      key[j] = j;
    }

    UpsertContext context{ key, idx, idx};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    free(key);
  }
  // Read.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_EQ(Status::Ok, result);
      CallbackContext<ReadContext> context{ ctxt };

      ASSERT_EQ(context->output.value, context->key().key_length_);
      for (size_t j = 0; j < context->key().key_length_; ++j) {
        ASSERT_EQ(context->key().key_data_[j], j);
      }
      free(context->key().key_data_);
    };
    // Create the key as a variable length array
    uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      key[j] = j;
    }

    ReadContext context{ key, idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_TRUE(result == Status::Ok || result == Status::Pending);
    if (result == Status::Ok) {
      ASSERT_EQ(idx, context.output.value);
      for (uint32_t j = 0; j < context.output.value; ++j) {
        ASSERT_EQ(context.key().key_data_[j], j);
      }
      free(key);
    }
  }
  // Update one fourth of the entries
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 4 == 0) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        // Writes do not go pending in normal operation
        ASSERT_TRUE(false);
      };
      // Create the key as a variable length array
      uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        key[j] = j;
      }

      UpsertContext context{ key, idx, 2*idx };
      Status result = store.Upsert(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
      free(key);
    }
  }
  // Delete another one fourth
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 4 == 1) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false); // deletes do not go pending
      };
      // Create the key as a variable length array
      uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        key[j] = j;
      }

      DeleteContext context{ key, idx };
      Status result = store.Delete(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
      free(key);
    }
  }
  // RMW another one fourth
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 4 == 2) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_EQ(result, Status::Ok);
        // free memory
        CallbackContext<RmwContext> context{ ctxt };
        free(context->key().key_data_);
      };
      // Create the key as a variable length array
      uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        key[j] = j;
      }

      RmwContext context{ key, idx, idx };
      Status result = store.Rmw(context, callback, 1);
      ASSERT_TRUE(result == Status::Ok || result == Status::Pending);
      if (result == Status::Ok) {
        free(key);
      }
    }
  }
  store.CompletePending(true);

  // perform compaction (with or without shift begin address)
  uint64_t until_address = store.hlog.safe_read_only_address.control();
  ASSERT_TRUE(
    store.CompactWithLookup(
      until_address, shift_begin_address, n_threads, false, checkpoint));
  if (shift_begin_address) {
    ASSERT_EQ(until_address, store.hlog.begin_address.control());
  }

  // Read again.
  for(uint32_t idx = 1; idx <= numRecords ; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      CallbackContext<ReadContext> context{ ctxt };
      // check request result & value
      if (context->key().key_length_ % 4 == 0) { // Upsert
        ASSERT_EQ(Status::Ok, result);
        ASSERT_EQ(context->output.value, 2 * context->key().key_length_);
      } else if (context->key().key_length_ % 4 == 1) { // Delete
        ASSERT_EQ(Status::NotFound, result);
      } else if (context->key().key_length_ % 4 == 2) { // Rmw
        ASSERT_EQ(Status::Ok, result);
        ASSERT_EQ(context->output.value, 2 * context->key().key_length_);
      } else { // key_length_ % 4 == 3 (Intact)
        ASSERT_EQ(Status::Ok, result);
        ASSERT_EQ(context->output.value, context->key().key_length_);
      }
      // verify that key match the requested key
      for (uint32_t j = 0; j < context->key().key_length_; ++j) {
        ASSERT_EQ(context->key().key_data_[j], j);
      }
      free(context->key().key_data_);
    };
    // Create the key as a variable length array
    uint32_t* key = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      key[j] = j;
    }

    ReadContext context{ key, idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_TRUE(result == Status::Ok || result == Status::Pending ||
                result == Status::NotFound);
    if (result == Status::Ok) {
      if (idx % 4 == 0) { // Upsert
        ASSERT_EQ(2 * idx, context.output.value);
      } else if (idx % 4 == 2) { // RMW
        ASSERT_EQ(2 * idx, context.output.value);
      } else if (idx % 4 == 3) { // Intact
        ASSERT_EQ(idx, context.output.value);
      }
      else ASSERT_TRUE(false);
      for (uint32_t j = 0; j < context.key().key_length_; ++j) {
        ASSERT_EQ(context.key().key_data_[j], j);
      }
      free(key);
    }
    else if (result == Status::NotFound) { // Deleted
      ASSERT_TRUE(idx % 4 == 1);
      free(key);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

TEST_P(CompactLookupParameterizedOnDiskTestFixture, OnDiskVariableLengthValue) {
  using Key = FixedSizeKey<uint32_t>;

  class UpsertContextVLV;
  class ReadContextVLV;

  class Value {
   public:
    Value()
      : gen_lock_{ 0 }
      , size_{ 0 }
      , length_{ 0 } {
    }

    inline uint32_t size() const {
      return size_;
    }

    friend class UpsertContextVLV;
    friend class ReadContextVLV;

   private:
    AtomicGenLock gen_lock_;
    uint32_t size_;
    uint32_t length_;

    inline const uint8_t* buffer() const {
      return reinterpret_cast<const uint8_t*>(this + 1);
    }
    inline uint8_t* buffer() {
      return reinterpret_cast<uint8_t*>(this + 1);
    }
  };

  class UpsertContextVLV : public IAsyncContext {
   public:
    typedef Key key_t;
    typedef Value value_t;

    UpsertContextVLV(uint32_t key, uint32_t *value, uint32_t value_length)
      : key_{ key }
      , value_{ value }
      , value_length_{ value_length } {
    }
    /// Copy (and deep-copy) constructor.
    UpsertContextVLV(const UpsertContextVLV& other)
      : key_{ other.key_ }
      , value_{ other.value_ }
      , value_length_{ other.value_length_ } {
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const Key& key() const {
      return key_;
    }
    inline uint32_t value_size() const {
      return sizeof(value_t) + value_length_ * sizeof(uint32_t);
    }
    /// Non-atomic and atomic Put() methods.
    inline void Put(value_t& value) {
      value.gen_lock_.store(0);
      value.size_ = value_size();
      value.length_ = value_length_;
      std::memcpy(value.buffer(), value_, value_length_ * sizeof(uint32_t));
    }
    inline bool PutAtomic(value_t& value) {
      bool replaced;
      while(!value.gen_lock_.try_lock(replaced) && !replaced) {
        std::this_thread::yield();
      }
      if(replaced) {
        // Some other thread replaced this record.
        return false;
      }
      if(value.size_ < value_size()) {
        // Current value is too small for in-place update.
        value.gen_lock_.unlock(true);
        return false;
      }
      // In-place update overwrites length and buffer, but not size.
      value.length_ = value_length_;
      std::memcpy(value.buffer(), value_, value_length_ * sizeof(uint32_t));
      value.gen_lock_.unlock(false);
      return true;
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    key_t key_;
    uint32_t* value_;
    uint32_t value_length_;
  };

  class ReadContextVLV : public IAsyncContext {
   public:
    typedef Key key_t;
    typedef Value value_t;

    ReadContextVLV(uint32_t key)
      : key_{ key }
      , output{ nullptr }
      , output_length{ 0 } {
    }

    /// Copy (and deep-copy) constructor.
    ReadContextVLV(const ReadContextVLV& other)
      : key_{ other.key_ }
      , output{ other.output }
      , output_length{ other.output_length }
    { }

    ~ReadContextVLV() {
      if (output != nullptr) {
        free(output);
      }
    }

    /// The implicit and explicit interfaces require a key() accessor.
    inline const Key& key() const {
      return key_;
    }
    inline void Get(const value_t& value) {
      output_length = value.length_;
      if (output == nullptr) {
        output = (uint32_t*) malloc(output_length * sizeof(uint32_t));
      }
      std::memcpy(output, value.buffer(), output_length * sizeof(uint32_t));
    }
    inline void GetAtomic(const value_t& value) {
      GenLock before, after;
      do {
        before = value.gen_lock_.load();
        output_length = value.length_;
        if (output == nullptr) {
          output = (uint32_t*) malloc(output_length * sizeof(uint32_t));
        }
        std::memcpy(output, value.buffer(), output_length * sizeof(uint32_t));
        after = value.gen_lock_.load();
      } while(before.gen_number != after.gen_number);
    }

   protected:
    /// The explicit interface requires a DeepCopy_Internal() implementation.
    Status DeepCopy_Internal(IAsyncContext*& context_copy) {
      return IAsyncContext::DeepCopy_Internal(*this, context_copy);
    }

   private:
    key_t key_;
   public:
    uint32_t *output;
    uint32_t output_length;
  };

  using UpsertContext = UpsertContextVLV;
  using ReadContext = ReadContextVLV;

  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t;
  typedef FasterKv<Key, Value, disk_t> faster_t;

  std::experimental::filesystem::create_directories("tmp_store");

  faster_t store { 2048, (1 << 20) * 192, "tmp_store", 0.4 };
  uint32_t numRecords = 12500; // will occupy ~512 MB space in store

  bool shift_begin_address = std::get<0>(GetParam());
  bool checkpoint = std::get<1>(GetParam());
  int n_threads = std::get<2>(GetParam());

  log_debug("Compaction Threads:  %d", n_threads);
  log_debug("Shift Begin Address: %s", shift_begin_address ? "ENABLED" : "DISABLED");
  log_debug("Checkpoint:          %s", checkpoint ? "ENABLED" : "DISABLED");

  store.StartSession();
  // Insert.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false); // upserts do not go pending
    };
    // Create the value as a variable length array
    uint32_t* value = (uint32_t*) malloc(idx * sizeof(uint32_t));
    for (uint32_t j = 0; j < idx; ++j) {
      value[j] = idx;
    }

    UpsertContext context{ idx, value, idx};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    free(value);
  }
  // Read.
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_EQ(Status::Ok, result);
      CallbackContext<ReadContext> context{ ctxt };

      ASSERT_EQ(context->output_length, context->key().key);
      for (uint32_t j = 0; j < context->output_length; ++j) {
        ASSERT_EQ(context->output[j], context->key().key);
      }
    };

    ReadContext context{ idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_TRUE(result == Status::Ok || result == Status::Pending);
    if (result == Status::Ok) {
      for (uint32_t j = 0; j < idx; ++j) {
        ASSERT_EQ(context.output[j], idx);
      }
    }
  }
  // Update half
  for(uint32_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 2 == 0) {
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false); // upserts do not go pending
      };
      // Create the value as a variable length array
      uint32_t* value = (uint32_t*) malloc(idx * sizeof(uint32_t));
      for (uint32_t j = 0; j < idx; ++j) {
        value[j] = 2 * idx;
      }

      UpsertContext context{ idx, value, idx }; // double the value_id
      Status result = store.Upsert(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
      free(value);
    }
  }
  store.CompletePending(true);

  // perform compaction (with or without shift begin address)
  uint64_t until_address = store.hlog.safe_read_only_address.control();
  ASSERT_TRUE(
    store.CompactWithLookup(
      until_address, shift_begin_address, n_threads, false, checkpoint));
  if (shift_begin_address) {
    ASSERT_EQ(until_address, store.hlog.begin_address.control());
  }

  // Read again.
  for(uint32_t idx = 1; idx <= numRecords ; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_EQ(Status::Ok, result);
      CallbackContext<ReadContext> context{ ctxt };

      ASSERT_EQ(context->output_length, context->key().key);
      uint32_t value_id = (context->key().key % 2 == 0)
                              ? 2 * context->key().key
                              : context->key().key;
      for (uint32_t j = 0; j < context->output_length; ++j) {
        ASSERT_EQ(context->output[j], value_id);
      }
    };

    ReadContext context{ idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_TRUE(result == Status::Ok || result == Status::Pending);
    if (result == Status::Ok) {
      uint32_t value_id = (idx % 2 == 0) ? 2*idx : idx;
      for (uint32_t j = 0; j < idx; ++j) {
        ASSERT_EQ(context.output[j], value_id);
      }
    }
  }
  store.StopSession();
  std::experimental::filesystem::remove_all("tmp_store");
}
*/

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
