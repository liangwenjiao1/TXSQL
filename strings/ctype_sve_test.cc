/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is designed to work with certain software (including
   but not limited to OpenSSL) that is licensed under separate terms,
   as designated in a particular file or component or in included license
   documentation.  The authors of MySQL hereby grant you an additional
   permission to link the program and your derivative works with the
   separately licensed software that they have either included with
   the program or referenced in the documentation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include "ctype_sve_test.h"

#include <errno.h>
#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

namespace {

constexpr uint32_t kSharedStateMagic = 0x53564554U;
constexpr size_t kSharedStateSlots = 1024;

struct my_sve_test_slot {
  uint64_t tid;
  int32_t mode;
  uint32_t reserved;
  struct my_sve_test_metrics metrics;
};

struct my_sve_test_shared_state {
  uint32_t magic;
  uint32_t reserved;
  uint64_t process_token;
  uint32_t attach_count;
  uint32_t reserved2;
  my_sve_test_slot slots[kSharedStateSlots];
};

struct my_sve_test_fallback_state {
  int mode;
  struct my_sve_test_metrics metrics;
};

struct my_sve_test_state_refs {
  int *mode;
  struct my_sve_test_metrics *metrics;
};

bool runtime_available_impl() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#if defined(__linux__)
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
  static const bool sve_available = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
  return sve_available;
#else
  return true;
#endif
#else
  return false;
#endif
}

uint64_t process_token_impl() {
#if defined(__linux__)
  FILE *stat = fopen("/proc/self/stat", "r");
  if (stat == nullptr) return 0;

  char buffer[4096];
  size_t bytes = fread(buffer, 1, sizeof(buffer) - 1, stat);
  fclose(stat);
  if (bytes == 0) return 0;
  buffer[bytes] = '\0';

  char *tail = strrchr(buffer, ')');
  if (tail == nullptr || tail[1] == '\0') return 0;

  char *cursor = tail + 2;
  for (int field = 3; field <= 22; ++field) {
    char *next = (field == 22) ? nullptr : strchr(cursor, ' ');
    if (field == 22) {
      errno = 0;
      unsigned long long token = strtoull(cursor, nullptr, 10);
      return errno == 0 ? static_cast<uint64_t>(token) : 0;
    }
    if (next == nullptr) return 0;
    cursor = next + 1;
  }
#endif
  return 0;
}

size_t lane_bytes_impl() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
  return runtime_available_impl() ? static_cast<size_t>(svcntb()) : 0;
#else
  return 0;
#endif
}

uint64_t current_tid_impl() {
#if defined(__linux__)
  return static_cast<uint64_t>(syscall(SYS_gettid));
#else
  return static_cast<uint64_t>(getpid());
#endif
}

struct my_sve_test_shared_handle {
  my_sve_test_shared_state *state;
  char shm_name[64];
};

void shared_state_cleanup_impl();

my_sve_test_shared_handle &shared_handle_storage() {
  static my_sve_test_shared_handle handle = {nullptr, {0}};
  return handle;
}

bool &shared_handle_initialized() {
  static bool initialized = false;
  return initialized;
}

my_sve_test_shared_handle &shared_handle_impl() {
  my_sve_test_shared_handle &handle = shared_handle_storage();
  bool &initialized = shared_handle_initialized();
  if (initialized) return handle;
  initialized = true;

  char shm_name[64];
  snprintf(shm_name, sizeof(shm_name), "/mysql_sve_test_%ld",
           static_cast<long>(getpid()));

  int fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
  bool created = false;
  if (fd >= 0) {
    created = true;
  } else if (errno == EEXIST) {
    fd = shm_open(shm_name, O_RDWR, 0600);
  }

  if (fd < 0) return handle;

  if (created && ftruncate(fd, sizeof(my_sve_test_shared_state)) != 0) {
    close(fd);
    return handle;
  }

  auto *state_mem = static_cast<my_sve_test_shared_state *>(mmap(
      nullptr, sizeof(my_sve_test_shared_state), PROT_READ | PROT_WRITE,
      MAP_SHARED, fd, 0));
  close(fd);

  if (state_mem == MAP_FAILED) return handle;

  const uint64_t process_token = process_token_impl();
  if (created || state_mem->magic != kSharedStateMagic ||
      state_mem->process_token != process_token) {
    memset(state_mem, 0, sizeof(*state_mem));
    state_mem->magic = kSharedStateMagic;
    state_mem->process_token = process_token;
  }

  __sync_add_and_fetch(&state_mem->attach_count, 1U);
  handle.state = state_mem;
  memcpy(handle.shm_name, shm_name, sizeof(shm_name));

  return handle;
}

my_sve_test_shared_state *shared_state_impl() {
  return shared_handle_impl().state;
}

void shared_state_cleanup_impl() {
  if (!shared_handle_initialized()) return;

  my_sve_test_shared_handle &handle = shared_handle_impl();
  if (handle.state == nullptr) return;

  my_sve_test_shared_state *state = handle.state;
  handle.state = nullptr;

  if (__sync_sub_and_fetch(&state->attach_count, 1U) == 0) {
    shm_unlink(handle.shm_name);
  }
  munmap(state, sizeof(*state));
}

struct my_sve_test_cleanup_guard {
  ~my_sve_test_cleanup_guard() { shared_state_cleanup_impl(); }
};

my_sve_test_cleanup_guard g_cleanup_guard;

void ensure_cleanup_guard_linked() { (void)g_cleanup_guard; }

my_sve_test_shared_state *ensure_shared_state_impl() {
  ensure_cleanup_guard_linked();
  return shared_state_impl();
}

my_sve_test_slot *slot_for_current_thread() {
  my_sve_test_shared_state *state = ensure_shared_state_impl();
  if (state == nullptr) return nullptr;

  const uint64_t tid = current_tid_impl();
  size_t idx = tid % kSharedStateSlots;

  for (size_t probe = 0; probe < kSharedStateSlots; ++probe) {
    my_sve_test_slot *slot = &state->slots[(idx + probe) % kSharedStateSlots];
    if (slot->tid == tid) return slot;
    if (slot->tid == 0 &&
        __sync_bool_compare_and_swap(&slot->tid, 0ULL, tid)) {
      slot->mode = MY_SVE_TEST_MODE_AUTO;
      memset(&slot->metrics, 0, sizeof(slot->metrics));
      return slot;
    }
  }

  return nullptr;
}

my_sve_test_fallback_state &fallback_state_impl() {
  static thread_local my_sve_test_fallback_state state = {
      MY_SVE_TEST_MODE_AUTO, {}};
  return state;
}

my_sve_test_state_refs current_state_refs() {
  if (my_sve_test_slot *slot = slot_for_current_thread()) {
    return {reinterpret_cast<int *>(&slot->mode), &slot->metrics};
  }

  my_sve_test_fallback_state &fallback = fallback_state_impl();
  return {&fallback.mode, &fallback.metrics};
}

}  // namespace

extern "C" {

void my_sve_test_set_mode(int mode) {
  my_sve_test_state_refs refs = current_state_refs();
  *refs.mode = (mode == MY_SVE_TEST_MODE_FORCE_SCALAR)
                   ? MY_SVE_TEST_MODE_FORCE_SCALAR
                   : MY_SVE_TEST_MODE_AUTO;
}

int my_sve_test_get_mode(void) {
  my_sve_test_state_refs refs = current_state_refs();
  return *refs.mode;
}

bool my_sve_test_force_scalar(void) {
  my_sve_test_state_refs refs = current_state_refs();
  return *refs.mode == MY_SVE_TEST_MODE_FORCE_SCALAR;
}

void my_sve_test_reset_metrics(void) {
  my_sve_test_state_refs refs = current_state_refs();
  *refs.metrics = {};
}

void my_sve_test_get_metrics(struct my_sve_test_metrics *metrics) {
  if (metrics == nullptr) return;

  my_sve_test_state_refs refs = current_state_refs();
  *metrics = *refs.metrics;
  metrics->sve_available = runtime_available_impl() ? 1 : 0;
  metrics->lane_bytes = lane_bytes_impl();
}

void my_sve_test_cleanup(void) { shared_state_cleanup_impl(); }

bool my_sve_test_runtime_available(void) { return runtime_available_impl(); }

size_t my_sve_test_lane_bytes(void) { return lane_bytes_impl(); }

bool my_sve_test_instrumentation_enabled(void) {
#if MY_SVE_TEST_INSTRUMENTATION_LEVEL == 1
  return true;
#else
  return false;
#endif
}

void my_sve_test_note_fast_path(size_t bytes) {
  my_sve_test_state_refs refs = current_state_refs();
  ++refs.metrics->fast_calls;
  refs.metrics->ascii_prefix_bytes += bytes;
}

void my_sve_test_note_scalar_fallback(size_t bytes) {
  my_sve_test_state_refs refs = current_state_refs();
  ++refs.metrics->scalar_fallback_calls;
  refs.metrics->tail_bytes += bytes;
}

void my_sve_test_note_invalid_fallback(void) {
  my_sve_test_state_refs refs = current_state_refs();
  ++refs.metrics->invalid_fallback_calls;
}

void my_sve_test_note_tailoring_disabled(void) {
  my_sve_test_state_refs refs = current_state_refs();
  ++refs.metrics->tailoring_disabled_calls;
}

void my_sve_test_note_contraction_disabled(void) {
  my_sve_test_state_refs refs = current_state_refs();
  ++refs.metrics->contraction_disabled_calls;
}

}  // extern "C"
