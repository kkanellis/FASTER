# Build the FASTER library.
set (FASTER_HEADERS
  core/address.h
  core/alloc.h
  core/async.h
  core/async_result_types.h
  core/auto_ptr.h
  core/checkpoint_locks.h
  core/checkpoint_state.h
  core/checkpoint_state_f2.h
  core/config.h
  core/constants.h
  core/f2.h
  core/faster.h
  core/gc_state.h
  core/grow_state.h
  core/guid.h
  core/internal_contexts.h
  core/internal_contexts_f2.h
  core/key_hash.h
  core/light_epoch.h
  core/lss_allocator.h
  core/malloc_fixed_page_size.h
  core/native_buffer_pool.h
  core/persistent_memory_malloc.h
  core/phase.h
  core/read_cache.h
  core/read_cache_utils.h
  core/record.h
  core/recovery_status.h
  core/state_transitions.h
  core/status.h
  core/thread.h
  core/utility.h
  device/file_system_disk.h
  device/null_disk.h
  environment/file.h
  environment/file_common.h
  index/cold_index.h
  index/cold_index_contexts.h
  index/hash_bucket.h
  index/hash_table.h
  index/index.h
  index/mem_index.h
)

if (MSVC)
set (FASTER_HEADERS ${FASTER_HEADERS}
  environment/file_windows.h
)
else()
set (FASTER_HEADERS ${FASTER_HEADERS}
  environment/file_linux.h
)
endif()

set (FASTER_SOURCES
  core/address.cc
  core/lss_allocator.cc
  core/thread.cc
)

if (MSVC)
set (FASTER_SOURCES ${FASTER_SOURCES}
  environment/file_windows.cc
)
else()
set (FASTER_SOURCES ${FASTER_SOURCES}
  environment/file_linux.cc
)
endif()

add_library(faster STATIC ${FASTER_SOURCES} ${FASTER_HEADERS})
