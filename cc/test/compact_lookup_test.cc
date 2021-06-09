// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "gtest/gtest.h"

#include "core/faster.h"

#include "device/null_disk.h"

#include "test_types.h"

using namespace FASTER::core;
using FASTER::test::FixedSizeKey;
using FASTER::test::SimpleAtomicValue;
using FASTER::test::SimpleAtomicLargeValue;

using Key = FixedSizeKey<uint64_t>;
using Value = SimpleAtomicValue<uint64_t>;
using LargeValue = SimpleAtomicLargeValue<uint64_t>;

/// Key-value store, specialized to our key and value types.
#ifdef _WIN32
typedef FASTER::environment::ThreadPoolIoHandler handler_t;
#else
typedef FASTER::environment::QueueIoHandler handler_t;
#endif

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

// ****************************************************************************
// IN-MEMORY TESTS
// ****************************************************************************

/// Inserts a bunch of records into a FASTER instance and invokes the
/// compaction algorithm. Since all records are still live, checks if
/// they remain so after the algorithm completes/returns.
TEST(CompactLookup, InMemAllLive) {
  // In memory hybrid log
  typedef FasterKv<Key, Value, FASTER::device::NullDisk> faster_t;
  // 1GB log size
  faster_t store { 1024, (1 << 20) * 1024, "", 0.4 };
  int numRecords = 100000;

  store.StartSession();
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, Value> context{ Key(idx), Value(idx) };
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback should won't be called
      ASSERT_TRUE(false);
    };
    ReadContext<Key, Value> context{ idx };
    Status result = store.Read(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
    ASSERT_EQ(idx, context.output.value);
  }

  store.StopSession();
}

/// Inserts a bunch of records into a FASTER instance, deletes half of them
/// and invokes the compaction algorithm. Checks that the ones that should
/// be alive are alive and the ones that should be dead stay dead.
TEST(CompactLookup, InMemHalfLive) {
  // In memory hybrid log
  typedef FasterKv<Key, Value, FASTER::device::NullDisk> faster_t;
  // 1GB log size
  faster_t store { 1024, (1 << 20) * 1024, "", 0.4 };
  int numRecords = 100000;

  store.StartSession();
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, Value> context{ Key(idx), Value(idx) };
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  // Delete every alternate key here.
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 2 == 0) continue;
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    DeleteContext<Key, Value> context{ Key(idx) };
    Status result = store.Delete(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  // After compaction, deleted keys stay deleted.
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback should won't be called
      ASSERT_TRUE(false);
    };
    ReadContext<Key, Value> context{ idx };
    Status result = store.Read(context, callback, 1);
    Status expect = idx % 2 == 0 ? Status::Ok : Status::NotFound;
    ASSERT_EQ(expect, result);
    if (idx % 2 == 0) ASSERT_EQ(idx, context.output.value);
  }

  store.StopSession();
}

/// Inserts a bunch of records into a FASTER instance, updates half of them
/// with new values and invokes the compaction algorithm. Checks that the
/// updated ones have the new value, and the others the old one.
TEST(CompactLookup, InMemAllLiveNewEntries) {
  // In memory hybrid log
  typedef FasterKv<Key, Value, FASTER::device::NullDisk> faster_t;
  // 1GB log size
  faster_t store { 1024, (1 << 20) * 1024, "", 0.4 };
  int numRecords = 100000;

  store.StartSession();
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, Value> context{Key(idx), Value(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  // Insert fresh entries for half the records
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 2 == 0) continue;
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, Value> context{ Key(idx), Value(2 * idx) };
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  // After compaction, reads should return newer values
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };
    ReadContext<Key, Value> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    ASSERT_EQ(result, Status::Ok);
    Key expected_key {(idx % 2 == 0)
                          ? context.output.value
                          : context.output.value / 2};
    ASSERT_EQ(idx, expected_key.key);
  }

  store.StopSession();
}

/// Inserts a bunch of records into a FASTER instance, and invokes the
/// compaction algorithm. Concurrent to the compaction, upserts and deletes
/// are performed in alternate keys. After compaction checks that updated
/// keys have the new value, while deleted keys do not exist.
TEST(CompactLookup, InMemConcurrentOps) {
  // In memory hybrid log
  typedef FASTER::device::NullDisk disk_t;
  typedef FasterKv<Key, Value, disk_t> faster_t;
  // 1GB log size
  faster_t store { 128, (1 << 20) * 1024, "", 0.4 };
  constexpr int numRecords = 100000;

  store.StartSession();
  // Populate initial keys
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, Value> context{Key(idx), Value(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  auto upsert_worker_func = [](FasterKv<Key, Value, disk_t>* store_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // Insert fresh entries for half the records
    for (size_t idx = 1; idx <= numRecords; ++idx) {
      if (idx % 2 == 0) continue;
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false);
      };
      UpsertContext<Key, Value> context{ Key(idx), Value(2 * idx) };
      Status result = store_->Upsert(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
    }
  };

  auto delete_worker_func = [](FasterKv<Key, Value, disk_t>* store_) {
    // Delete every alternate key here.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (size_t idx = 1; idx <= numRecords; ++idx) {
      if (idx % 2 == 1) continue;
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false);
      };
      DeleteContext<Key, Value> context{ Key(idx) };
      Status result = store_->Delete(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
    }
  };

  std::thread upset_worker (upsert_worker_func, &store);
  std::thread delete_worker (delete_worker_func, &store);

  // perform compaction concurrently
  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  upset_worker.join();
  delete_worker.join();

  // Reads should return newer values for non-deleted entries
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };
    ReadContext<Key, Value> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);

    if (idx % 2 == 1) {
      ASSERT_EQ(result, Status::Ok);
      ASSERT_EQ(idx, context.output.value / 2);
    }
    else {
      ASSERT_EQ(result, Status::NotFound);
    }
  }

  store.StopSession();
}

// ****************************************************************************
// PERSISTENCE STORAGE TESTS
// ****************************************************************************

/// Inserts a bunch of records into a FASTER instance and invokes the
/// compaction algorithm. Since all records are still live, checks if
/// they remain so after the algorithm completes/returns.
TEST(CompactLookup, AllLive) {
  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t; // 1GB file segments
  typedef FasterKv<Key, LargeValue, disk_t> faster_t;

  std::experimental::filesystem::create_directories("tmp_store");
  // NOTE: deliberatly keeping the hash index small to test hash-chain chasing correctness
  faster_t store { 1024, (1 << 20) * 192, "tmp_store", 0.4 };
  int numRecords = 50000;

  store.StartSession();
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      // request will be sync -- callback won't be called
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, LargeValue> context{Key(idx), LargeValue(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_EQ(Status::Ok, result);

      CallbackContext<ReadContext<Key, LargeValue>> context(ctxt);
      ASSERT_EQ(context->key(), context->output.value);
    };
    ReadContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    EXPECT_TRUE(result == Status::Ok || result == Status::Pending);
    if (result == Status::Ok) {
      ASSERT_EQ(idx, context.output.value);
    }

    if (idx % 20 == 0) {
      // occasionally complete pending I/O requests
      store.CompletePending(false);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

/// Inserts a bunch of records into a FASTER instance, deletes half of them
/// and invokes the compaction algorithm. Checks that the ones that should
/// be alive are alive and the ones that should be dead stay dead.
TEST(CompactLookup, HalfLive) {
  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t; // 1GB file segments
  typedef FasterKv<Key, LargeValue, disk_t> faster_t;

  std::experimental::filesystem::create_directories("tmp_store");
  // NOTE: deliberatly keeping the hash index small to test hash-chain chasing correctness
  faster_t store { 1024, (1 << 20) * 192, "tmp_store", 0.4 };
  int numRecords = 50000;

  store.StartSession();
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, LargeValue> context{Key(idx), LargeValue(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  // Delete every alternate key here.
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 2 == 0) continue;
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    DeleteContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Delete(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  // After compaction, deleted keys stay deleted.
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      CallbackContext<ReadContext<Key, LargeValue>> context(ctxt);
      Status expected_status = (context->key().key % 2 == 0) ? Status::Ok
                                                             : Status::NotFound;
      ASSERT_EQ(expected_status, result);
      if (expected_status == Status::Ok) {
        ASSERT_EQ(context->key().key, context->output.value);
      }
    };
    ReadContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    if (idx % 2 == 0) {
      EXPECT_TRUE(result == Status::Ok || result == Status::Pending);
    }
    else {
      EXPECT_TRUE(result == Status::NotFound || result == Status::Pending);
    }
    if (result == Status::Ok) {
      ASSERT_EQ(idx, context.output.value);
    }

    if (idx % 20 == 0) {
      store.CompletePending(false);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

/// Inserts a bunch of records into a FASTER instance, updates half of them
/// with new values, deletes the other half, and invokes the compaction algorithm.
/// Checks that the updated ones have the new value, and the rest remain deleted.
TEST(CompactLookup, AllLiveDeleteAndReInsert) {
  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t; // 1GB file segments
  typedef FasterKv<Key, LargeValue, disk_t> faster_t;

  std::experimental::filesystem::create_directories("tmp_store");
  // NOTE: deliberatly keeping the hash index small to test hash-chain chasing correctness
  faster_t store { 1024, (1 << 20) * 192, "tmp_store", 0.4 };
  int numRecords = 50000;

  store.StartSession();
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, LargeValue> context{Key(idx), LargeValue(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  // Delete every alternate key here.
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 2 == 0) continue;
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    DeleteContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Delete(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  // Insert fresh entries for the alternate keys
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    if (idx % 2 == 0) continue;
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, LargeValue> context{ Key(idx), LargeValue(2 * idx) };
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  // After compaction, all entries should exist
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_EQ(Status::Ok, result);

      CallbackContext<ReadContext<Key, LargeValue>> context(ctxt);
      Key expected_key {(context->key().key % 2 == 0)
                            ? context->output.value
                            : context->output.value / 2};
      ASSERT_EQ(context->key().key, expected_key.key);
    };
    ReadContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    EXPECT_TRUE(result == Status::Ok || result == Status::Pending);
    if (result == Status::Ok) {
      Key expected_key {(idx % 2 == 0)
                            ? context.output.value
                            : context.output.value / 2};
      ASSERT_EQ(idx, expected_key.key);
    }

    if (idx % 20 == 0) {
      store.CompletePending(false);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

/// Inserts a bunch of records into a FASTER instance, and invokes the
/// compaction algorithm. Concurrent to the compaction, upserts and deletes
/// are performed in alternate keys. After compaction checks that updated
/// keys have the new value, while deleted keys do not exist.
TEST(CompactLookup, ConcurrentOps) {
  typedef FASTER::device::FileSystemDisk<handler_t, (1 << 30)> disk_t; // 1GB file segments
  typedef FasterKv<Key, LargeValue, disk_t> faster_t;

  std::experimental::filesystem::create_directories("tmp_store");
  // NOTE: deliberatly keeping the hash index small to test hash-chain chasing correctness
  faster_t store { 1024, (1 << 20) * 192, "tmp_store", 0.4 };
  constexpr int numRecords = 50000;

  store.StartSession();
  // Populate initial keys
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      ASSERT_TRUE(false);
    };
    UpsertContext<Key, LargeValue> context{Key(idx), LargeValue(idx)};
    Status result = store.Upsert(context, callback, 1);
    ASSERT_EQ(Status::Ok, result);
  }

  auto upsert_worker_func = [](FasterKv<Key, LargeValue, disk_t>* store_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // Insert fresh entries for half the records
    for (size_t idx = 1; idx <= numRecords; ++idx) {
      if (idx % 2 == 0) continue;
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false);
      };
      UpsertContext<Key, LargeValue> context{ Key(idx), LargeValue(2 * idx) };
      Status result = store_->Upsert(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
    }
  };

  auto delete_worker_func = [](FasterKv<Key, LargeValue, disk_t>* store_) {
    // Delete every alternate key here.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (size_t idx = 1; idx <= numRecords; ++idx) {
      if (idx % 2 == 1) continue;
      auto callback = [](IAsyncContext* ctxt, Status result) {
        ASSERT_TRUE(false);
      };
      DeleteContext<Key, LargeValue> context{ Key(idx) };
      Status result = store_->Delete(context, callback, 1);
      ASSERT_EQ(Status::Ok, result);
    }
  };
  // launch threads
  std::thread upset_worker (upsert_worker_func, &store);
  std::thread delete_worker (delete_worker_func, &store);

  // perform compaction concurrently
  store.CompactWithLookup(store.hlog.GetTailAddress().control());

  upset_worker.join();
  delete_worker.join();

  // Reads should return newer values for non-deleted entries
  for (size_t idx = 1; idx <= numRecords; ++idx) {
    auto callback = [](IAsyncContext* ctxt, Status result) {
      CallbackContext<ReadContext<Key, LargeValue>> context(ctxt);
      if (context->key().key % 2 == 0) {
        ASSERT_EQ(Status::NotFound, result);
      } else {
        ASSERT_EQ(Status::Ok, result);
        ASSERT_EQ(context->key().key, context->output.value / 2);
      }
    };
    ReadContext<Key, LargeValue> context{ Key(idx) };
    Status result = store.Read(context, callback, 1);
    EXPECT_TRUE(result == Status::Ok || result == Status::NotFound || result == Status::Pending);

    if (result == Status::Ok) {
      ASSERT_TRUE(idx % 2 == 1);
      ASSERT_EQ(idx, context.output.value / 2);
    } else if (result == Status::NotFound) {
      ASSERT_TRUE(idx % 2 == 0);
    }

    if (idx % 20 == 0) {
      store.CompletePending(false);
    }
  }
  store.CompletePending(true);
  store.StopSession();

  std::experimental::filesystem::remove_all("tmp_store");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}