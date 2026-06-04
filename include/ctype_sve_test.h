#ifndef CTYPE_SVE_TEST_INCLUDED
#define CTYPE_SVE_TEST_INCLUDED

#include <stddef.h>
#include <stdint.h>

#include "my_compiler.h"

enum my_sve_test_mode {
  MY_SVE_TEST_MODE_AUTO = 0,
  MY_SVE_TEST_MODE_FORCE_SCALAR = 1
};

struct my_sve_test_metrics {
  uint64_t sve_available;
  uint64_t lane_bytes;
  uint64_t instrumentation_enabled;
  uint64_t fast_calls;
  uint64_t scalar_fallback_calls;
  uint64_t ascii_prefix_bytes;
  uint64_t tail_bytes;
  uint64_t invalid_fallback_calls;
  uint64_t tailoring_disabled_calls;
  uint64_t contraction_disabled_calls;
};

#ifdef __cplusplus
extern "C" {
#endif

#define MY_SVE_TEST_EXPORT MY_ATTRIBUTE((visibility("default"), externally_visible))

void my_sve_test_set_mode(int mode) MY_SVE_TEST_EXPORT;
int my_sve_test_get_mode(void) MY_SVE_TEST_EXPORT;
bool my_sve_test_force_scalar(void) MY_SVE_TEST_EXPORT;

void my_sve_test_reset_metrics(void) MY_SVE_TEST_EXPORT;
void my_sve_test_get_metrics(struct my_sve_test_metrics *metrics)
    MY_SVE_TEST_EXPORT;
void my_sve_test_cleanup(void) MY_SVE_TEST_EXPORT;

bool my_sve_test_runtime_available(void) MY_SVE_TEST_EXPORT;
size_t my_sve_test_lane_bytes(void) MY_SVE_TEST_EXPORT;
bool my_sve_test_instrumentation_enabled(void) MY_SVE_TEST_EXPORT;

void my_sve_test_note_fast_path(size_t bytes) MY_SVE_TEST_EXPORT;
void my_sve_test_note_scalar_fallback(size_t bytes) MY_SVE_TEST_EXPORT;
void my_sve_test_note_invalid_fallback(void) MY_SVE_TEST_EXPORT;
void my_sve_test_note_tailoring_disabled(void) MY_SVE_TEST_EXPORT;
void my_sve_test_note_contraction_disabled(void) MY_SVE_TEST_EXPORT;

#undef MY_SVE_TEST_EXPORT

#ifdef __cplusplus
}
#endif

#ifndef MY_SVE_TEST_INSTRUMENTATION_LEVEL
#define MY_SVE_TEST_INSTRUMENTATION_LEVEL 0
#endif

#if MY_SVE_TEST_INSTRUMENTATION_LEVEL != 0 && \
    MY_SVE_TEST_INSTRUMENTATION_LEVEL != 1
#error "MY_SVE_TEST_INSTRUMENTATION_LEVEL must be 0 or 1"
#endif

static inline bool my_sve_test_force_scalar_if_enabled(void) {
#if MY_SVE_TEST_INSTRUMENTATION_LEVEL == 1
  return my_sve_test_force_scalar();
#else
  return false;
#endif
}

static inline void my_sve_test_note_fast_path_if_enabled(size_t bytes) {
#if MY_SVE_TEST_INSTRUMENTATION_LEVEL == 1
  my_sve_test_note_fast_path(bytes);
#else
  (void)bytes;
#endif
}

static inline void my_sve_test_note_scalar_fallback_if_enabled(size_t bytes) {
#if MY_SVE_TEST_INSTRUMENTATION_LEVEL == 1
  my_sve_test_note_scalar_fallback(bytes);
#else
  (void)bytes;
#endif
}

static inline void my_sve_test_note_invalid_fallback_if_enabled(void) {
#if MY_SVE_TEST_INSTRUMENTATION_LEVEL == 1
  my_sve_test_note_invalid_fallback();
#endif
}

static inline void my_sve_test_note_tailoring_disabled_if_enabled(void) {
#if MY_SVE_TEST_INSTRUMENTATION_LEVEL == 1
  my_sve_test_note_tailoring_disabled();
#endif
}

static inline void my_sve_test_note_contraction_disabled_if_enabled(void) {
#if MY_SVE_TEST_INSTRUMENTATION_LEVEL == 1
  my_sve_test_note_contraction_disabled();
#endif
}

#endif
