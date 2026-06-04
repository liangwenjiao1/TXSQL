/* Copyright (c) 2026, Huawei and/or its affiliates.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public
   License as published by the Free Software Foundation; version 2
   of the License.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#include <arm_sve.h>

#ifdef CTYPE_UTF8
#include "ctype-arm-simd.h"
#endif

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include "m_string.h"
#include <errno.h>

#include "ctype_sve_test.h"

#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif

/*
  SVE 架构最大向量长度为 2048 bit。
  在使用依赖 lane 数的 store 偏移前，先用 assert 约束运行时长度。
*/
#define SVE_MAX_VECTOR_LENGTH 256
#define MAX_ASCII 0x7F
constexpr uchar kPrintableAsciiFirst = 0x20;
constexpr uchar kPrintableAsciiLast = 0x7E;
constexpr size_t kPrintableAsciiCount =
    kPrintableAsciiLast - kPrintableAsciiFirst + 1;
#define CTYPE_UTF8_OPTIMIZED
#define CTYPE_UCA_OPTIMIZED
#define CTYPE_MB_OPTIMIZED

using sve_u8_vector_t = svuint8_t;

#ifdef CTYPE_CONVERT
#define CTYPE_CONVERT_OPTIMIZED

size_t my_convert(char *to, size_t to_length, const CHARSET_INFO *to_cs,
                  const char *from, size_t from_length,
                  const CHARSET_INFO *from_cs, uint *errors) {
  size_t length, length2;

  if ((to_cs->state | from_cs->state) & MY_CS_NONASCII)
    return my_convert_internal(to, to_length, to_cs, from, from_length, from_cs,
                               errors);

  length = length2 = std::min(to_length, from_length);

  {
    const svbool_t pg = svptrue_b8();
    const size_t vector_bytes = svcntb();

    while (vector_bytes != 0 && length >= vector_bytes) {
      const svuint8_t vec = svld1_u8(pg, pointer_cast<const uint8_t *>(from));
      if (svptest_any(pg, svcmpgt_n_u8(pg, vec, 0x7F))) {
        size_t i = 0;
        for (; i < vector_bytes; ++i) {
          if (static_cast<uchar>(from[i]) > 0x7F) break;
          to[i] = from[i];
        }
        from += i;
        to += i;
        length -= i;
        break;
      }

      svst1_u8(pg, pointer_cast<uint8_t *>(to), vec);
      from += vector_bytes;
      to += vector_bytes;
      length -= vector_bytes;
    }
  }

#if defined(__i386__) || defined(_WIN32) || defined(__x86_64__) || \
    defined(__aarch64__)
  for (; length >= 4; length -= 4, from += 4, to += 4) {
    if (uint4korr(from) & 0x80808080) break;
    int4store(to, uint4korr(from));
  }
#endif /* __i386__ */

  for (;; *to++ = *from++, length--) {
    if (!length) {
      *errors = 0;
      return length2;
    }
    if ((static_cast<uchar>(*from)) > 0x7F) {
      size_t copied_length = length2 - length;
      to_length -= copied_length;
      from_length -= copied_length;
      return copied_length + my_convert_internal(to, to_length, to_cs, from,
                                                 from_length, from_cs, errors);
    }
  }

  assert(false);
  return 0;
}
#endif

static inline bool my_simd_runtime_available()
{
  return my_sve_test_runtime_available();
}

/*
  所有直接执行 SVE 指令的路径均应先通过此检查，
  以统一约束运行时可用性和测试中的强制标量路径。
*/
static inline bool my_simd_execution_allowed()
{
  return !my_sve_test_force_scalar_if_enabled() && my_simd_runtime_available();
}

static inline size_t my_simd_lanes_u8()
{
  return my_sve_test_lane_bytes();
}

static inline void my_sve_test_note_tail_if_needed(size_t bytes)
{
  if (bytes != 0)
  {
    my_sve_test_note_scalar_fallback_if_enabled(bytes);
  }
}

#ifdef CTYPE_UCA
static inline size_t my_uca_skip_dual_space_sve(const uchar **sp,
                                                const uchar **tp,
                                                const uchar *se,
                                                const uchar *te)
{
  if (!my_simd_execution_allowed())
  {
    return 0;
  }

  size_t skipped = 0;
  const svbool_t pg = svptrue_b8();
  const size_t vector_bytes = my_simd_lanes_u8();

  while (*sp + vector_bytes <= se && *tp + vector_bytes <= te)
  {
    const svuint8_t vec_s = svld1_u8(pg, *sp);
    const svuint8_t vec_t = svld1_u8(pg, *tp);
    const svbool_t both_space =
        svand_z(pg, svcmpeq_n_u8(pg, vec_s, ' '),
                svcmpeq_n_u8(pg, vec_t, ' '));

    if (svcntp_b8(pg, both_space) != vector_bytes)
    {
      break;
    }

    *sp += vector_bytes;
    *tp += vector_bytes;
    skipped += vector_bytes;
    my_sve_test_note_fast_path_if_enabled(vector_bytes);
  }

  while (*sp < se && *tp < te && **sp == ' ' && **tp == ' ')
  {
    ++(*sp);
    ++(*tp);
    ++skipped;
  }

  return skipped;
}
#endif

/*
  仅在默认 Unicode 大小写映射且不存在非 ASCII 特殊语义时启用 SIMD 路径，
  以保证 ASCII 快速路径与既有比较和转换语义保持一致。
*/
static inline bool my_simd_support_check(const CHARSET_INFO *cs)
{
  if (!my_simd_execution_allowed())
  {
    return false;
  }

  if ((cs->state & MY_CS_UNICODE) && !(cs->state & MY_CS_NONASCII) &&
      (cs->caseinfo == &my_unicase_default))
  {
    return true;
  }

  return false;
}

static inline bool my_simd_simple_ascii_case_mapping(const CHARSET_INFO *cs)
{
  if (cs->caseinfo == nullptr || cs->caseinfo->page[0] == nullptr)
  {
    return false;
  }

  const MY_UNICASE_CHARACTER *ascii_page = cs->caseinfo->page[0];
  for (uchar ch = 0; ch < 128; ++ch)
  {
    uchar upper = ch;
    uchar lower = ch;
    if (ch >= 'a' && ch <= 'z')
    {
      upper = ch - ('a' - 'A');
    }
    if (ch >= 'A' && ch <= 'Z')
    {
      lower = ch + ('a' - 'A');
    }

    if (ascii_page[ch].toupper != upper || ascii_page[ch].tolower != lower)
    {
      return false;
    }
  }
  return true;
}

static inline bool my_simd_case_conversion_support_check(const CHARSET_INFO *cs)
{
  if (my_simd_support_check(cs))
  {
    return true;
  }

  return my_simd_execution_allowed() && (cs->state & MY_CS_UNICODE) &&
         !(cs->state & MY_CS_NONASCII) &&
         my_simd_simple_ascii_case_mapping(cs);
}

static inline bool my_simd_load_ascii_vector(const uchar *src,
                                             sve_u8_vector_t *vec_src)
{
  const svbool_t pg = svptrue_b8();
  *vec_src = svld1_u8(pg, src);
  return !svptest_any(pg, svcmpgt_n_u8(pg, *vec_src, MAX_ASCII));
}

static inline bool my_simd_load_printable_ascii_vector(const uchar *src,
                                                       sve_u8_vector_t *vec_src)
{
  const svbool_t pg = svptrue_b8();
  *vec_src = svld1_u8(pg, src);
  return !svptest_any(pg, svcmplt_n_u8(pg, *vec_src, kPrintableAsciiFirst)) &&
         !svptest_any(pg, svcmpgt_n_u8(pg, *vec_src, kPrintableAsciiLast));
}

static inline bool my_simd_case_load_ascii_vector(const CHARSET_INFO *cs,
                                                  const uchar *src,
                                                  sve_u8_vector_t *vec_src)
{
  if (cs->caseinfo == &my_unicase_default)
  {
    return my_simd_load_ascii_vector(src, vec_src);
  }
  return my_simd_load_printable_ascii_vector(src, vec_src);
}

static inline void my_simd_store_u8(uchar *dst, sve_u8_vector_t vec_src)
{
  svst1_u8(svptrue_b8(), dst, vec_src);
}

static inline void my_simd_assert_vector_bytes(size_t vector_bytes)
{
  assert(vector_bytes <= SVE_MAX_VECTOR_LENGTH);
  (void)vector_bytes;
}

static inline svbool_t my_simd_ascii_in_range(sve_u8_vector_t vec_src,
                                              uchar lower, uchar upper)
{
  const svbool_t pg = svptrue_b8();

  return svand_z(pg, svcmpge_n_u8(pg, vec_src, lower),
                 svcmple_n_u8(pg, vec_src, upper));
}

static inline void my_simd_toupper_ascii(sve_u8_vector_t *vec_src)
{
  const svbool_t lower_alpha = my_simd_ascii_in_range(*vec_src, 'a', 'z');

  *vec_src = svsub_n_u8_m(lower_alpha, *vec_src, 'a' - 'A');
}

static inline void my_simd_tolower_ascii(sve_u8_vector_t *vec_src)
{
  const svbool_t upper_alpha = my_simd_ascii_in_range(*vec_src, 'A', 'Z');

  *vec_src = svadd_n_u8_m(upper_alpha, *vec_src, 'a' - 'A');
}

static inline void my_simd_tosort_ascii(const CHARSET_INFO *cs,
                                        sve_u8_vector_t *vec_src)
{
  if (cs->state & MY_CS_LOWER_SORT)
  {
    my_simd_tolower_ascii(vec_src);
  }
  else
  {
    my_simd_toupper_ascii(vec_src);
  }
}

static inline void my_simd_ascii_to_unicode_2byte_weight(
    const sve_u8_vector_t *vec_src, uchar *dst, size_t vector_bytes)
{
  my_simd_assert_vector_bytes(vector_bytes);

  const sve_u8_vector_t zero = svdup_n_u8(0);
#ifdef WORDS_BIGENDIAN
  const sve_u8_vector_t hi = svzip1_u8(*vec_src, zero);
  const sve_u8_vector_t lo = svzip2_u8(*vec_src, zero);
#else
  const sve_u8_vector_t hi = svzip1_u8(zero, *vec_src);
  const sve_u8_vector_t lo = svzip2_u8(zero, *vec_src);
#endif
  my_simd_store_u8(dst, hi);
  my_simd_store_u8(dst + vector_bytes, lo);
}

struct my_ascii_sort_map_cache_t
{
  bool upper_ready;
  bool lower_ready;
  uchar upper[256];
  uchar lower[256];
};

static inline void my_init_ascii_sort_map(uchar *dst, bool lower_sort)
{
  for (size_t i = 0; i < 256; ++i)
  {
    uchar ch = static_cast<uchar>(i);
    if (lower_sort)
    {
      dst[i] = (ch >= 'A' && ch <= 'Z') ? ch + ('a' - 'A') : ch;
    }
    else
    {
      dst[i] = (ch >= 'a' && ch <= 'z') ? ch - ('a' - 'A') : ch;
    }
  }
}

static inline const uchar *my_get_ascii_sort_map(const CHARSET_INFO *cs)
{
  static thread_local my_ascii_sort_map_cache_t cache = {
      false, false, {0}, {0}};

  if (cs->state & MY_CS_LOWER_SORT)
  {
    if (!cache.lower_ready)
    {
      my_init_ascii_sort_map(cache.lower, true);
      cache.lower_ready = true;
    }
    return cache.lower;
  }

  if (!cache.upper_ready)
  {
    my_init_ascii_sort_map(cache.upper, false);
    cache.upper_ready = true;
  }
  return cache.upper;
}

static inline void my_simd_hash_cal(const CHARSET_INFO *cs, const uchar *src,
                                    size_t vector_bytes,
                                    ulong *hash_val1, ulong *hash_val2)
{
  my_simd_assert_vector_bytes(vector_bytes);

  sve_u8_vector_t vec_src;
  uchar mapped[SVE_MAX_VECTOR_LENGTH];

  if (my_simd_case_load_ascii_vector(cs, src, &vec_src))
  {
    my_simd_tosort_ascii(cs, &vec_src);
    my_simd_store_u8(mapped, vec_src);
  }
  else
  {
    const uchar *sort_map = my_get_ascii_sort_map(cs);
    for (size_t i = 0; i < vector_bytes; ++i)
    {
      mapped[i] = sort_map[src[i]];
    }
  }

  for (size_t i = 0; i < vector_bytes; ++i)
  {
    const uchar ch = mapped[i];
    *hash_val1 ^= (((*hash_val1 & 63) + *hash_val2) * ch) + (*hash_val1 << 8);
    *hash_val1 ^= (*hash_val1 << 8);
    *hash_val2 += 6;
  }
}

static inline int my_simd_vector_cmp(sve_u8_vector_t *vec_first,
                                     sve_u8_vector_t *vec_second)
{
  const svbool_t pg = svptrue_b8();
  const svbool_t neq = svcmpne_u8(pg, *vec_first, *vec_second);

  if (!svptest_any(pg, neq))
  {
    return 0;
  }

  const size_t first_diff =
      svcntp_b8(pg, svbrkb_b_z(pg, neq));
  const svbool_t diff_prefix =
      svwhilelt_b8((uint64_t)0, (uint64_t)(first_diff + 1));
  const int first = (int)svlastb_u8(diff_prefix, *vec_first);
  const int second = (int)svlastb_u8(diff_prefix, *vec_second);

  return first - second;
}

static inline size_t my_simd_first_mismatch_u8(sve_u8_vector_t vec_first,
                                               sve_u8_vector_t vec_second)
{
  const svbool_t pg = svptrue_b8();
  const svbool_t neq = svcmpne_u8(pg, vec_first, vec_second);

  assert(svptest_any(pg, neq));
  return svcntp_b8(pg, svbrkb_b_z(pg, neq));
}

#ifdef CTYPE_UTF8
static int
my_mb_wc_utf8mb3_no_range_adapter(const CHARSET_INFO *cs MY_ATTRIBUTE((unused)),
                                  my_wc_t *pwc, const uchar *s);
static int
my_uni_utf8mb3_no_range_adapter(const CHARSET_INFO *cs MY_ATTRIBUTE((unused)),
                                my_wc_t wc, uchar *r);
static int my_valid_mbcharlen_utf8mb3_adapter(
    const CHARSET_INFO *cs MY_ATTRIBUTE((unused)), const uchar *s,
    const uchar *e);
static inline void my_sve_test_note_noop(size_t bytes MY_ATTRIBUTE((unused)));
static inline void my_sve_test_note_noop_invalid();

static inline const uchar* skip_space(const uchar *ptr, const uchar *end)
{
  while (end - ptr >= (long)sizeof(uint64_t))
  {
    if (uint8korr(ptr) != 0x2020202020202020uLL)
    {
      break;
    }
    ptr += sizeof(uint64_t);
  }

  while (end > ptr && *ptr == 0x20)
  {
    ptr++;
  }

  return (ptr);
}

/**
  功能：
    用空格字符的排序权重填充缓冲区。

  参数：
    str：缓冲区起始位置。
    strend：缓冲区结束位置。

  返回值：
    实际写入的字节数。

  说明：
    当缓冲区长度为奇数时，允许仅写入半个排序权重。
*/
static size_t my_strxfrm_pad_unicode(uchar *str, uchar *strend) {
  uchar *str0 = str;
  assert(str && str <= strend);

  if (my_simd_execution_allowed()) {
    const size_t vector_bytes = my_simd_lanes_u8();
    const sve_u8_vector_t vec_space = svdup_n_u8(0x20);
    while ((size_t)(strend - str) >= vector_bytes * 2) {
      my_simd_ascii_to_unicode_2byte_weight(&vec_space, str, vector_bytes);
      str += vector_bytes * 2;
    }
  }

  for (; str < strend;) {
    *str++ = 0x00;
    if (str < strend) *str++ = 0x20;
  }
  return str - str0;
}

template <class Mb_wc>
static inline int my_strnxfrm_read_wc_tmpl(Mb_wc mb_wc, const uchar **src,
                                           const uchar *se, my_wc_t *wc,
                                           bool ascii_single_byte_shortcut) {
  if (ascii_single_byte_shortcut && *src < se && **src < 0x80) {
    *wc = **src;
    ++(*src);
    return 1;
  }

  const int res = mb_wc(wc, *src, se);
  if (res > 0) {
    *src += res;
  }
  return res;
}

static inline void my_strnxfrm_store_high_byte(uchar **dst, my_wc_t wc) {
  *(*dst)++ = (uchar)(wc >> 8);
}

template <class Mb_wc, class PrepareWc>
static inline void my_strnxfrm_write_leftover_high_byte_tmpl(
    Mb_wc mb_wc, const uchar **src, const uchar *se, uchar **dst,
    bool ascii_single_byte_shortcut, PrepareWc prepare_wc) {
  my_wc_t wc;

  if (my_strnxfrm_read_wc_tmpl(mb_wc, src, se, &wc,
                               ascii_single_byte_shortcut) > 0) {
    prepare_wc(&wc);
    my_strnxfrm_store_high_byte(dst, wc);
  }
}

/*
  my_strnxfrm_unicode 的模板实现。
  通过 mb_wc 特化多字节到宽字符的解码逻辑，并保持内联以降低
  高频调用 mb_wc() 的额外开销。
*/
template <class Mb_wc>
static inline size_t my_strnxfrm_unicode_tmpl(const CHARSET_INFO *cs,
                                              Mb_wc mb_wc, uchar *dst,
                                              size_t dstlen, uint nweights,
                                              const uchar *src, size_t srclen,
                                              uint flags) {
  uchar *dst0 = dst;
  uchar *de = dst + dstlen;
  const uchar *se = src + srclen;
  assert(src || srclen == 0);

  if (!(cs->state & MY_CS_BINSORT) && my_simd_support_check(cs)) {
    sve_u8_vector_t vec_src;
    const size_t vector_bytes = my_simd_lanes_u8();
    for (;
         (size_t)(de - dst) >= vector_bytes * 2 &&
             (size_t)(se - src) >= vector_bytes &&
             nweights >= vector_bytes;
         nweights -= vector_bytes) {
      if (!my_simd_load_ascii_vector(src, &vec_src)) {
        break;
      }

      my_simd_tosort_ascii(cs, &vec_src);
      my_simd_ascii_to_unicode_2byte_weight(&vec_src, dst, vector_bytes);
      my_sve_test_note_fast_path_if_enabled(vector_bytes);

      src += vector_bytes;
      dst += vector_bytes * 2;
    }
    my_sve_test_note_tail_if_needed((size_t)(se - src));
  }

  // 将该条件判断外提到循环之外，以减少循环体内的分支开销。
  // GCC（至少 6.1.1）尚不能稳定完成该优化。
  if (cs->state & MY_CS_BINSORT) {
    const bool ascii_single_byte_shortcut = (cs->mbminlen == 1);

    // 尽量处理完整字符。
    const size_t nweights_fast_path =
        std::min<size_t>((de - dst) / 2, nweights);
    for (size_t i = 0; i < nweights_fast_path; ++i, --nweights) {
      my_wc_t wc;
      int res = my_strnxfrm_read_wc_tmpl(mb_wc, &src, se, &wc,
                                         ascii_single_byte_shortcut);
      if (res <= 0)  // 源串结束，或遇到非法字符。
        goto pad;
      dst = store16be(dst, wc);
    }

    // 若输出缓冲区仅剩 1 字节，则在此写入剩余高字节。
    if (dst < de && nweights) {
      my_strnxfrm_write_leftover_high_byte_tmpl(
          mb_wc, &src, se, &dst, ascii_single_byte_shortcut,
          [](my_wc_t *wc MY_ATTRIBUTE((unused))) {});
    }
  } else {
    const MY_UNICASE_INFO *uni_plane = cs->caseinfo;

    // 尽量处理完整字符。
    const size_t nweights_fast_path =
        std::min<size_t>((de - dst) / 2, nweights);
    for (size_t i = 0; i < nweights_fast_path; ++i, --nweights) {
      my_wc_t wc;
      int res = mb_wc(&wc, src, se);
      if (res <= 0)  // 源串结束，或遇到非法字符。
        goto pad;
      src += res;

      my_tosort_unicode(uni_plane, &wc, cs->state);

      dst = store16be(dst, wc);
    }

    // 若输出缓冲区仅剩 1 字节，则在此写入剩余高字节。
    if (dst < de && nweights) {
      my_strnxfrm_write_leftover_high_byte_tmpl(
          mb_wc, &src, se, &dst, false,
          [uni_plane, cs](my_wc_t *wc) {
            my_tosort_unicode(uni_plane, wc, cs->state);
          });
    }
  }

pad:
  if (dst < de && nweights)  // 按 PAD SPACE 语义补齐剩余权重。
    dst += my_strxfrm_pad_nweights_unicode(dst, de, nweights);

  if ((flags & MY_STRXFRM_PAD_TO_MAXLEN) && dst < de)
    dst += my_strxfrm_pad_unicode(dst, de);
  return dst - dst0;
}

static int
my_mb_wc_utf8mb3_no_range_adapter(const CHARSET_INFO *cs MY_ATTRIBUTE((unused)),
                                  my_wc_t *pwc, const uchar *s) {
  return my_mb_wc_utf8mb3_no_range(pwc, s);
}

static int
my_uni_utf8mb3_no_range_adapter(const CHARSET_INFO *cs MY_ATTRIBUTE((unused)),
                                my_wc_t wc, uchar *r) {
  return my_uni_utf8mb3_no_range(cs, wc, r);
}

static int my_valid_mbcharlen_utf8mb3_adapter(
    const CHARSET_INFO *cs MY_ATTRIBUTE((unused)), const uchar *s,
    const uchar *e) {
  return my_valid_mbcharlen_utf8mb3(s, e);
}

static inline void my_sve_test_note_noop(size_t bytes MY_ATTRIBUTE((unused))) {}

static inline void my_sve_test_note_noop_invalid() {}

struct CaseUpUtf8mb3Traits {
  static size_t multiply(const CHARSET_INFO *cs) { return cs->caseup_multiply; }
  static void simd_op(sve_u8_vector_t *vec) { my_simd_toupper_ascii(vec); }
  static void case_op(const MY_UNICASE_INFO *uni_plane, my_wc_t *wc) {
    my_toupper_utf8mb3(uni_plane, wc);
  }
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb3(wc, src, srcend);
  }
  static int encode(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst,
                    uchar *dstend) {
    return my_uni_utf8mb3(cs, wc, dst, dstend);
  }
  static int decode_no_range(const CHARSET_INFO *cs, my_wc_t *wc,
                             const uchar *src) {
    return my_mb_wc_utf8mb3_no_range_adapter(cs, wc, src);
  }
  static int encode_no_range(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst) {
    return my_uni_utf8mb3_no_range_adapter(cs, wc, dst);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_fast_path_if_enabled(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_tail_if_needed(bytes); }
};

struct CaseDnUtf8mb3Traits {
  static size_t multiply(const CHARSET_INFO *cs) { return cs->casedn_multiply; }
  static void simd_op(sve_u8_vector_t *vec) { my_simd_tolower_ascii(vec); }
  static void case_op(const MY_UNICASE_INFO *uni_plane, my_wc_t *wc) {
    my_tolower_utf8mb3(uni_plane, wc);
  }
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb3(wc, src, srcend);
  }
  static int encode(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst,
                    uchar *dstend) {
    return my_uni_utf8mb3(cs, wc, dst, dstend);
  }
  static int decode_no_range(const CHARSET_INFO *cs, my_wc_t *wc,
                             const uchar *src) {
    return my_mb_wc_utf8mb3_no_range_adapter(cs, wc, src);
  }
  static int encode_no_range(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst) {
    return my_uni_utf8mb3_no_range_adapter(cs, wc, dst);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_fast_path_if_enabled(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_tail_if_needed(bytes); }
};

struct CaseUpUtf8mb4Traits {
  static size_t multiply(const CHARSET_INFO *cs) { return cs->caseup_multiply; }
  static void simd_op(sve_u8_vector_t *vec) { my_simd_toupper_ascii(vec); }
  static void case_op(const MY_UNICASE_INFO *uni_plane, my_wc_t *wc) {
    my_toupper_utf8mb4(uni_plane, wc);
  }
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb4(wc, src, srcend);
  }
  static int encode(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst,
                    uchar *dstend) {
    return my_wc_mb_utf8mb4(cs, wc, dst, dstend);
  }
  static int decode_no_range(const CHARSET_INFO *cs, my_wc_t *wc,
                             const uchar *src) {
    return my_mb_wc_utf8mb4_no_range(cs, wc, src);
  }
  static int encode_no_range(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst) {
    return my_wc_mb_utf8mb4_no_range(cs, wc, dst);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_fast_path_if_enabled(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_tail_if_needed(bytes); }
};

struct CaseDnUtf8mb4Traits {
  static size_t multiply(const CHARSET_INFO *cs) { return cs->casedn_multiply; }
  static void simd_op(sve_u8_vector_t *vec) { my_simd_tolower_ascii(vec); }
  static void case_op(const MY_UNICASE_INFO *uni_plane, my_wc_t *wc) {
    my_tolower_utf8mb4(uni_plane, wc);
  }
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb4(wc, src, srcend);
  }
  static int encode(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst,
                    uchar *dstend) {
    return my_wc_mb_utf8mb4(cs, wc, dst, dstend);
  }
  static int decode_no_range(const CHARSET_INFO *cs, my_wc_t *wc,
                             const uchar *src) {
    return my_mb_wc_utf8mb4_no_range(cs, wc, src);
  }
  static int encode_no_range(const CHARSET_INFO *cs, my_wc_t wc, uchar *dst) {
    return my_wc_mb_utf8mb4_no_range(cs, wc, dst);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_fast_path_if_enabled(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_tail_if_needed(bytes); }
};

struct StrcasecmpUtf8mb3Traits {
  static int decode(const CHARSET_INFO *cs MY_ATTRIBUTE((unused)),
                    my_wc_t *wc, const uchar *src,
                    const uchar *src_end) {
    return my_mb_wc_utf8mb3(wc, src, src_end);
  }
  static void case_op(const MY_UNICASE_INFO *uni_plane, my_wc_t *wc) {
    my_tolower_utf8mb3(uni_plane, wc);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_noop(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_noop(bytes); }
  static void on_invalid() { my_sve_test_note_noop_invalid(); }
};

struct StrcasecmpUtf8mb4Traits {
  static int decode(const CHARSET_INFO *cs MY_ATTRIBUTE((unused)), my_wc_t *wc,
                    const uchar *src, const uchar *src_end) {
    return my_mb_wc_utf8mb4(wc, src, src_end);
  }
  static void case_op(const MY_UNICASE_INFO *uni_plane, my_wc_t *wc) {
    my_tolower_utf8mb4(uni_plane, wc);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_fast_path_if_enabled(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_tail_if_needed(bytes); }
  static void on_invalid() { my_sve_test_note_invalid_fallback_if_enabled(); }
};

struct WellFormedLenUtf8mb3Traits {
  static int valid_mbcharlen(const CHARSET_INFO *cs, const uchar *s,
                             const uchar *e) {
    return my_valid_mbcharlen_utf8mb3_adapter(cs, s, e);
  }
};

struct WellFormedLenUtf8mb4Traits {
  static int valid_mbcharlen(const CHARSET_INFO *cs, const uchar *s,
                             const uchar *e) {
    return my_valid_mbcharlen_utf8mb4(cs, s, e);
  }
};

struct StrnncollUtf8mb3Traits {
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb3(wc, src, srcend);
  }
  static int bincmp_fallback(const uchar *s, const uchar *se, const uchar *t,
                             const uchar *te) {
    return bincmp(s, se, t, te);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_noop(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_noop(bytes); }
  static void on_invalid() { my_sve_test_note_noop_invalid(); }
};

struct StrnncollUtf8mb4Traits {
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb4(wc, src, srcend);
  }
  static int bincmp_fallback(const uchar *s, const uchar *se, const uchar *t,
                             const uchar *te) {
    return bincmp_utf8mb4(s, se, t, te);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_fast_path_if_enabled(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_tail_if_needed(bytes); }
  static void on_invalid() { my_sve_test_note_invalid_fallback_if_enabled(); }
};

struct HashSortUtf8mb3Traits {
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb3(wc, src, srcend);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_noop(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_noop(bytes); }
  static void emit_hash_bytes(my_wc_t wc, uint64 *tmp1, uint64 *tmp2);
};

struct HashSortUtf8mb4Traits {
  static int decode(my_wc_t *wc, const uchar *src, const uchar *srcend) {
    return my_mb_wc_utf8mb4(wc, src, srcend);
  }
  static void on_fast(size_t bytes) { my_sve_test_note_fast_path_if_enabled(bytes); }
  static void on_tail(size_t bytes) { my_sve_test_note_tail_if_needed(bytes); }
  static void emit_hash_bytes(my_wc_t wc, uint64 *tmp1, uint64 *tmp2);
};

struct Utf8StrnncollInput {
  const uchar *s;
  size_t slen;
  const uchar *t;
  size_t tlen;
};

struct Utf8StrnncollState {
  const uchar *s;
  const uchar *se;
  const uchar *t;
  const uchar *te;
};

struct Utf8CompareDecision {
  bool decided;
  int value;
};

template <class Traits>
static inline size_t skip_dual_space(const uchar **sp, const uchar **tp,
                                     const uchar *se, const uchar *te) {
  size_t skipped = 0;

  if (my_simd_execution_allowed()) {
    const svbool_t pg = svptrue_b8();
    const size_t vector_bytes = my_simd_lanes_u8();

    while ((size_t)(se - *sp) >= vector_bytes &&
           (size_t)(te - *tp) >= vector_bytes) {
      const sve_u8_vector_t vec_s = svld1_u8(pg, *sp);
      const sve_u8_vector_t vec_t = svld1_u8(pg, *tp);
      const svbool_t s_space = svcmpeq_n_u8(pg, vec_s, ' ');
      const svbool_t t_space = svcmpeq_n_u8(pg, vec_t, ' ');
      const svbool_t both_space = svand_z(pg, s_space, t_space);

      if (svcntp_b8(pg, both_space) != vector_bytes) {
        break;
      }

      *sp += vector_bytes;
      *tp += vector_bytes;
      skipped += vector_bytes;
      Traits::on_fast(vector_bytes);
    }
  }

  while (*sp < se && *tp < te && **sp == ' ' && **tp == ' ') {
    ++(*sp);
    ++(*tp);
    ++skipped;
  }

  return skipped;
}

static inline void hash_sort_mix_byte(uint ch, uint64 *tmp1, uint64 *tmp2) {
  *tmp1 ^= (((*tmp1 & 63) + *tmp2) * ch) + (*tmp1 << 8);
  *tmp2 += 3;
}

inline void HashSortUtf8mb3Traits::emit_hash_bytes(my_wc_t wc, uint64 *tmp1,
                                                   uint64 *tmp2) {
  hash_sort_mix_byte((uint)(wc & 0xFF), tmp1, tmp2);
  hash_sort_mix_byte((uint)(wc >> 8), tmp1, tmp2);
}

inline void HashSortUtf8mb4Traits::emit_hash_bytes(my_wc_t wc, uint64 *tmp1,
                                                   uint64 *tmp2) {
  hash_sort_mix_byte((uint)(wc & 0xFF), tmp1, tmp2);
  hash_sort_mix_byte((uint)((wc >> 8) & 0xFF), tmp1, tmp2);

  if (wc > 0xFFFF) {
    /*
      仅在最高字节非 0 时才写入，以保持 utf8mb3 和 utf8mb4 在 BMP
      字符上的哈希行为兼容。这有助于维持测试结果中的记录顺序稳定，
      例如 "SHOW GRANTS" 的输出。
    */
    hash_sort_mix_byte((uint)((wc >> 16) & 0xFF), tmp1, tmp2);
  }
}

template <class Traits>
static inline Utf8CompareDecision strnncoll_utf8_core_tmpl(
    const CHARSET_INFO *cs, Utf8StrnncollState *state,
    bool skip_dual_spaces = false) {
  my_wc_t s_wc = 0, t_wc = 0;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  const MY_UNICASE_CHARACTER *ascii_page = uni_plane->page[0];
  const bool lower_sort = (cs->state & MY_CS_LOWER_SORT) != 0;

  if (my_simd_support_check(cs)) {
    sve_u8_vector_t vec_first, vec_second;
    int vec_cmp = 0;
    const size_t vector_bytes = my_simd_lanes_u8();

    while ((size_t)(state->se - state->s) >= vector_bytes &&
           (size_t)(state->te - state->t) >= vector_bytes) {
      if (skip_dual_spaces && *state->s == ' ' && *state->t == ' ' &&
          skip_dual_space<Traits>(&state->s, &state->t, state->se,
                                  state->te) != 0) {
        continue;
      }

      if (!my_simd_load_ascii_vector(state->s, &vec_first) ||
          !my_simd_load_ascii_vector(state->t, &vec_second)) {
        break;
      }

      my_simd_tosort_ascii(cs, &vec_first);
      my_simd_tosort_ascii(cs, &vec_second);
      Traits::on_fast(vector_bytes);

      vec_cmp = my_simd_vector_cmp(&vec_first, &vec_second);
      if (vec_cmp != 0) {
        return {true, vec_cmp};
      }

      state->s += vector_bytes;
      state->t += vector_bytes;
    }
    Traits::on_tail((size_t)std::max(state->se - state->s, state->te - state->t));
  }

  while (state->s < state->se && state->t < state->te) {
    if (skip_dual_spaces && state->s[0] == ' ' && state->t[0] == ' ' &&
        skip_dual_space<Traits>(&state->s, &state->t, state->se, state->te) !=
            0) {
      continue;
    }

    if ((uchar)state->s[0] < 128 && (uchar)state->t[0] < 128 &&
        ascii_page != nullptr) {
      s_wc = lower_sort ? ascii_page[(uchar)state->s[0]].tolower
                        : ascii_page[(uchar)state->s[0]].sort;
      t_wc = lower_sort ? ascii_page[(uchar)state->t[0]].tolower
                        : ascii_page[(uchar)state->t[0]].sort;
      state->s++;
      state->t++;
    } else {
      int s_res = Traits::decode(&s_wc, state->s, state->se);
      int t_res = Traits::decode(&t_wc, state->t, state->te);

      if (s_res <= 0 || t_res <= 0) {
        Traits::on_invalid();
        return {true, Traits::bincmp_fallback(state->s, state->se, state->t,
                                              state->te)};
      }

      my_tosort_unicode(uni_plane, &s_wc, cs->state);
      my_tosort_unicode(uni_plane, &t_wc, cs->state);

      state->s += s_res;
      state->t += t_res;
    }

    if (s_wc != t_wc) {
      return {true, s_wc > t_wc ? 1 : -1};
    }
  }

  return {false, 0};
}

template <class Traits>
static inline int strnncoll_utf8_tmpl(const CHARSET_INFO *cs,
                                      Utf8StrnncollInput input,
                                      bool t_is_prefix) {
  Utf8StrnncollState state{input.s, input.s + input.slen, input.t,
                           input.t + input.tlen};
  Utf8CompareDecision decision = strnncoll_utf8_core_tmpl<Traits>(cs, &state);

  if (decision.decided) {
    return decision.value;
  }

  return (int)(t_is_prefix ? state.t - state.te
                           : ((state.se - state.s) - (state.te - state.t)));
}

template <class Traits>
static inline int strnncollsp_utf8_tmpl(const CHARSET_INFO *cs,
                                        Utf8StrnncollInput input) {
  Utf8StrnncollState state{input.s, input.s + input.slen, input.t,
                           input.t + input.tlen};
  Utf8CompareDecision decision =
      strnncoll_utf8_core_tmpl<Traits>(cs, &state, true);
  const uchar *s = state.s;
  const uchar *se = state.se;
  size_t slen = (size_t)(state.se - state.s);
  size_t tlen = (size_t)(state.te - state.t);
  int res = 0;

  if (decision.decided) {
    return decision.value;
  }

  if (slen != tlen) {
    int swap = 1;
    if (slen < tlen) {
      slen = tlen;
      s = state.t;
      se = state.te;
      swap = -1;
      res = -res;
    }
    /*
      该循环利用 UTF-8 的编码性质：所有多字节字符及其首字节都大于
      空格字符。因此在补齐尾部空格的比较场景中，一旦较长字符串中出现
      大于空格的字节，即可判定其更大，无需完整解析多字节序列。
    */
    s = skip_space(s, se);
    for (; s < se; s++) {
      if (*s != ' ') return (*s < ' ') ? -swap : swap;
    }
  }
  return res;
}

template <class Traits>
static inline void hash_sort_utf8_tmpl(const CHARSET_INFO *cs, const uchar *s,
                                       size_t slen, uint64 *n1, uint64 *n2) {
  my_wc_t wc;
  int res;
  const uchar *e = skip_trailing_space(s, slen);
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  uint64 tmp1 = *n1;
  uint64 tmp2 = *n2;

  if (my_simd_support_check(cs)) {
    sve_u8_vector_t vec_src;
    const size_t vector_bytes = my_simd_lanes_u8();

    while ((size_t)(e - s) >= vector_bytes) {
      if (!my_simd_load_ascii_vector(s, &vec_src)) {
        break;
      }

      my_simd_hash_cal(cs, s, vector_bytes, &tmp1, &tmp2);
      Traits::on_fast(vector_bytes);

      s += vector_bytes;
    }
    Traits::on_tail((size_t)(e - s));
  }

  while ((res = Traits::decode(&wc, s, e)) > 0) {
    my_tosort_unicode(uni_plane, &wc, cs->state);
    Traits::emit_hash_bytes(wc, &tmp1, &tmp2);
    s += res;
  }

  *n1 = tmp1;
  *n2 = tmp2;
}

template <class Traits>
static inline size_t case_convert_buffer_tmpl(const CHARSET_INFO *cs, char *src,
                                              size_t srclen, char *dst,
                                              size_t dstlen) {
  my_wc_t wc;
  int srcres, dstres;
  char *srcend = src + srclen, *dstend = dst + dstlen, *dst0 = dst;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;

  assert(src != dst || Traits::multiply(cs) == 1);

  if (my_simd_case_conversion_support_check(cs)) {
    sve_u8_vector_t vec_src;
    const size_t vector_bytes = my_simd_lanes_u8();

    while ((size_t)(srcend - src) >= vector_bytes &&
           (size_t)(dstend - dst) >= vector_bytes) {
      if (!my_simd_case_load_ascii_vector(cs, (const uchar *)src, &vec_src)) {
        break;
      }

      Traits::simd_op(&vec_src);
      my_simd_store_u8((uchar *)dst, vec_src);
      Traits::on_fast(vector_bytes);

      src += vector_bytes;
      dst += vector_bytes;
    }
    Traits::on_tail((size_t)(srcend - src));
  }

  while ((src < srcend) &&
         (srcres = Traits::decode(&wc, (uchar *)src, (uchar *)srcend)) > 0) {
    Traits::case_op(uni_plane, &wc);
    if ((dstres = Traits::encode(cs, wc, (uchar *)dst, (uchar *)dstend)) <= 0)
      break;
    src += srcres;
    dst += dstres;
  }

  return (size_t)(dst - dst0);
}

template <class Traits>
static inline size_t case_convert_cstr_tmpl(const CHARSET_INFO *cs, char *src) {
  my_wc_t wc;
  int srcres, dstres;
  char *dst = src, *dst0 = src;
  char *srcend = src + strlen(src);
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;

  assert(Traits::multiply(cs) == 1);

  if (my_simd_case_conversion_support_check(cs)) {
    sve_u8_vector_t vec_src;
    size_t length = (size_t)(srcend - src);
    const size_t vector_bytes = my_simd_lanes_u8();

    while (length >= vector_bytes) {
      if (!my_simd_case_load_ascii_vector(cs, (const uchar *)src, &vec_src)) {
        break;
      }

      Traits::simd_op(&vec_src);
      my_simd_store_u8((uchar *)dst, vec_src);
      Traits::on_fast(vector_bytes);

      src += vector_bytes;
      dst += vector_bytes;
      length -= vector_bytes;
    }
    Traits::on_tail(length);
  }

  while (src < srcend &&
         (srcres = Traits::decode(&wc, pointer_cast<uchar *>(src),
                                  pointer_cast<uchar *>(srcend))) > 0) {
    Traits::case_op(uni_plane, &wc);
    if ((dstres = Traits::encode_no_range(cs, wc, (uchar *)dst)) <= 0) break;
    src += srcres;
    dst += dstres;
  }

  /*
    转换后的字符串可能短于原字符串，例如 U+0130 在小写化后会缩短。
    因此无论大小写方向如何，都需要显式补写 '\0' 终止符。
  */
  *dst = '\0';
  return (size_t)(dst - dst0);
}

template <class Traits>
static inline int strcasecmp_utf8_tmpl(const CHARSET_INFO *cs, const char *s,
                                       const char *t) {
  const char *s0 = s;
  const char *t0 = t;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  const size_t slen = strlen(s0);
  const size_t tlen = strlen(t0);
  const uchar *s_end = pointer_cast<const uchar *>(s0 + slen);
  const uchar *t_end = pointer_cast<const uchar *>(t0 + tlen);

  if (my_simd_support_check(cs)) {
    sve_u8_vector_t vec_first, vec_second;
    int vec_cmp = 0;
    size_t simd_slen = slen;
    size_t simd_tlen = tlen;
    const size_t vector_bytes = my_simd_lanes_u8();

    while (simd_slen >= vector_bytes && simd_tlen >= vector_bytes) {
      if (!my_simd_load_ascii_vector((const uchar *)s, &vec_first) ||
          !my_simd_load_ascii_vector((const uchar *)t, &vec_second)) {
        break;
      }

      my_simd_tolower_ascii(&vec_first);
      my_simd_tolower_ascii(&vec_second);
      Traits::on_fast(vector_bytes);

      vec_cmp = my_simd_vector_cmp(&vec_first, &vec_second);
      if (vec_cmp != 0) {
        return vec_cmp;
      }

      s += vector_bytes;
      t += vector_bytes;
      simd_slen -= vector_bytes;
      simd_tlen -= vector_bytes;
    }
    size_t tail_bytes = std::max(simd_slen, simd_tlen);
    Traits::on_tail(tail_bytes);
  }

  while (s[0] && t[0]) {
    my_wc_t s_wc, t_wc;

    if ((uchar)s[0] < 128) {
      s_wc = plane00[(uchar)s[0]].tolower;
      s++;
    } else {
      int res = Traits::decode(cs, &s_wc, pointer_cast<const uchar *>(s), s_end);
      if (res <= 0) {
        Traits::on_invalid();
        return strcmp(s, t);
      }
      s += res;
      Traits::case_op(uni_plane, &s_wc);
    }

    if ((uchar)t[0] < 128) {
      t_wc = plane00[(uchar)t[0]].tolower;
      t++;
    } else {
      int res = Traits::decode(cs, &t_wc, pointer_cast<const uchar *>(t), t_end);
      if (res <= 0) {
        Traits::on_invalid();
        return strcmp(s, t);
      }
      t += res;
      Traits::case_op(uni_plane, &t_wc);
    }

    if (s_wc != t_wc) return ((int)s_wc) - ((int)t_wc);
  }

  return ((int)(uchar)s[0]) - ((int)(uchar)t[0]);
}

template <class Traits>
static inline size_t well_formed_len_utf8_tmpl(const CHARSET_INFO *cs,
                                               const char *b, const char *e,
                                               size_t pos, int *error) {
  const char *b_start = b;
  size_t len = e - b;
  *error = 0;

  if (my_simd_execution_allowed()) {
    sve_u8_vector_t vec_b;
    const size_t vector_bytes = my_simd_lanes_u8();
    while (pos >= vector_bytes && len >= vector_bytes) {
      if (!my_simd_load_ascii_vector(pointer_cast<const uchar *>(b), &vec_b)) {
        break;
      }

      b += vector_bytes;
      pos -= vector_bytes;
      len -= vector_bytes;
    }
  }

  while (pos) {
    int mb_len;

    if ((mb_len = Traits::valid_mbcharlen(cs, pointer_cast<const uchar *>(b),
                                          pointer_cast<const uchar *>(e))) <=
        0) {
      *error = b < e ? 1 : 0;
      break;
    }
    b += mb_len;
    pos--;
  }

  return (size_t)(b - b_start);
}

extern "C" {
static size_t my_caseup_utf8mb3(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  return case_convert_buffer_tmpl<CaseUpUtf8mb3Traits>(cs, src, srclen, dst,
                                                       dstlen);
}

static void my_hash_sort_utf8mb3(const CHARSET_INFO *cs, const uchar *s,
                                 size_t slen, uint64 *n1, uint64 *n2) {
  hash_sort_utf8_tmpl<HashSortUtf8mb3Traits>(cs, s, slen, n1, n2);
}

static size_t my_caseup_str_utf8mb3(const CHARSET_INFO *cs, char *src) {
  return case_convert_cstr_tmpl<CaseUpUtf8mb3Traits>(cs, src);
}

static size_t my_casedn_utf8mb3(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  return case_convert_buffer_tmpl<CaseDnUtf8mb3Traits>(cs, src, srclen, dst,
                                                       dstlen);
}

static size_t my_casedn_str_utf8mb3(const CHARSET_INFO *cs, char *src) {
  return case_convert_cstr_tmpl<CaseDnUtf8mb3Traits>(cs, src);
}

static int my_strnncoll_utf8mb3(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix) {
  Utf8StrnncollInput input{s, slen, t, tlen};
  return strnncoll_utf8_tmpl<StrnncollUtf8mb3Traits>(cs, input, t_is_prefix);
}

/**
  功能：
    比较两个字符串，并忽略尾部空格。

  参数：
    cs：字符集处理器。
    s：第一个待比较字符串。
    slen：`s` 的长度。
    t：第二个待比较字符串。
    tlen：`t` 的长度。

  返回值：
    负数表示 `s < t`；
    0 表示 `s == t`；
    正数表示 `s > t`。

  说明：
    较短一方按空格扩展后再比较，以保持尾部空格语义。
    内嵌 `'\0'` 仍按原始字节值参与比较。
*/
static int my_strnncollsp_utf8mb3(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen) {
  Utf8StrnncollInput input{s, slen, t, tlen};
  return strnncollsp_utf8_tmpl<StrnncollUtf8mb3Traits>(cs, input);
}

/**
  功能：
    比较两个以 `'\0'` 结尾的 UTF-8 字符串。

  参数：
    cs：字符集处理器。
    s：第一个待比较字符串。
    t：第二个待比较字符串。

  返回值：
    负数表示 `s < t`；
    0 表示两者相等；
    正数表示 `s > t`。

  说明：
    非法多字节序列时保持既有回退路径语义。
*/
static int my_strcasecmp_utf8mb3(const CHARSET_INFO *cs, const char *s,
                                 const char *t) {
  return strcasecmp_utf8_tmpl<StrcasecmpUtf8mb3Traits>(cs, s, t);
}

static size_t my_well_formed_len_utf8mb3(const CHARSET_INFO *, const char *b,
                                         const char *e, size_t pos,
                                         int *error) {
  return well_formed_len_utf8_tmpl<WellFormedLenUtf8mb3Traits>(nullptr, b, e,
                                                               pos, error);
}

static size_t my_caseup_utf8mb4(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  return case_convert_buffer_tmpl<CaseUpUtf8mb4Traits>(cs, src, srclen, dst,
                                                       dstlen);
}

static void my_hash_sort_utf8mb4(const CHARSET_INFO *cs, const uchar *s,
                                 size_t slen, uint64 *n1, uint64 *n2) {
  hash_sort_utf8_tmpl<HashSortUtf8mb4Traits>(cs, s, slen, n1, n2);
}

static size_t my_caseup_str_utf8mb4(const CHARSET_INFO *cs, char *src) {
  return case_convert_cstr_tmpl<CaseUpUtf8mb4Traits>(cs, src);
}

static size_t my_casedn_utf8mb4(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  return case_convert_buffer_tmpl<CaseDnUtf8mb4Traits>(cs, src, srclen, dst,
                                                       dstlen);
}

static size_t my_casedn_str_utf8mb4(const CHARSET_INFO *cs, char *src) {
  return case_convert_cstr_tmpl<CaseDnUtf8mb4Traits>(cs, src);
}

static int my_strnncoll_utf8mb4(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix) {
  Utf8StrnncollInput input{s, slen, t, tlen};
  return strnncoll_utf8_tmpl<StrnncollUtf8mb4Traits>(cs, input, t_is_prefix);
}

static int my_strnncollsp_utf8mb4(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen) {
  Utf8StrnncollInput input{s, slen, t, tlen};
  return strnncollsp_utf8_tmpl<StrnncollUtf8mb4Traits>(cs, input);
}

static int my_strcasecmp_utf8mb4(const CHARSET_INFO *cs, const char *s,
                                 const char *t) {
  return strcasecmp_utf8_tmpl<StrcasecmpUtf8mb4Traits>(cs, s, t);
}

static size_t my_well_formed_len_utf8mb4(const CHARSET_INFO *cs, const char *b,
                                         const char *e, size_t pos,
                                         int *error) {
  return well_formed_len_utf8_tmpl<WellFormedLenUtf8mb4Traits>(cs, b, e, pos,
                                                               error);
}
} // extern "C"
#endif /* CTYPE_UTF8 */

#ifdef CTYPE_UCA
#include "ctype-arm-simd.h"

static inline bool my_simd_legacy_uca_unicode_ci(const CHARSET_INFO *cs)
{
  return cs->m_coll_name != nullptr &&
         (strcmp(cs->m_coll_name, "utf8mb3_unicode_ci") == 0 ||
          strcmp(cs->m_coll_name, "utf8mb4_unicode_ci") == 0);
}

/*
  旧版 UCA SIMD 路径仅覆盖 utf8_unicode_ci/utf8mb4_unicode_ci 的可打印
  ASCII 前缀。
*/
static inline bool my_simd_support_check_uca_any(const CHARSET_INFO *cs)
{
  if (!my_simd_execution_allowed() || cs->uca == nullptr ||
      cs->uca->version == UCA_V900 ||
      cs->mbminlen != 1 || (cs->state & MY_CS_NONASCII))
  {
    return false;
  }

  if ((cs->tailoring != nullptr && cs->tailoring[0] != '\0') ||
      cs->coll_param != nullptr)
  {
    my_sve_test_note_tailoring_disabled_if_enabled();
    return false;
  }

  if (my_uca_have_contractions(cs->uca))
  {
    my_sve_test_note_contraction_disabled_if_enabled();
    return false;
  }

  return my_simd_legacy_uca_unicode_ci(cs);
}

/*
  UCA 9.0.0 的 SIMD 路径沿用相同约束，并限定为基础权重页的快速路径。
*/
static inline bool my_simd_support_check_uca_900(const CHARSET_INFO *cs)
{
  if (!my_simd_execution_allowed() || cs->uca == nullptr ||
      cs->uca->version != UCA_V900 ||
      cs->mbminlen != 1 || (cs->state & MY_CS_NONASCII))
  {
    return false;
  }

  if ((cs->tailoring != nullptr && cs->tailoring[0] != '\0') ||
      cs->coll_param != nullptr)
  {
    my_sve_test_note_tailoring_disabled_if_enabled();
    return false;
  }

  if (my_uca_have_contractions(cs->uca))
  {
    my_sve_test_note_contraction_disabled_if_enabled();
    return false;
  }

  return true;
}

static inline const uint16 *my_uca900_printable_ascii_primary_weights(
    const CHARSET_INFO *cs)
{
  if (cs->uca == nullptr || cs->uca->weights[0] == nullptr)
  {
    return nullptr;
  }
  return UCA900_WEIGHT_ADDR(cs->uca->weights[0], /*level=*/0, /*subcode=*/0);
}

static inline bool my_uca900_printable_ascii_primary_safe(
    const CHARSET_INFO *cs)
{
  const uint16 *wpage = cs->uca != nullptr ? cs->uca->weights[0] : nullptr;
  if (wpage == nullptr)
  {
    return false;
  }

  struct uca900_printable_ascii_primary_cache
  {
    const uint16 *wpage;
    bool safe;
  };
  static thread_local uca900_printable_ascii_primary_cache cache = {
      nullptr, false};

  if (cache.wpage == wpage)
  {
    return cache.safe;
  }

  bool safe = true;
  for (size_t i = 0; i < kPrintableAsciiCount; ++i)
  {
    const uchar ch = static_cast<uchar>(kPrintableAsciiFirst + i);
    if (UCA900_NUM_OF_CE(wpage, ch) != 1 ||
        UCA900_WEIGHT(wpage, /*level=*/0, ch) == 0)
    {
      safe = false;
      break;
    }
  }

  cache.wpage = wpage;
  cache.safe = safe;
  return safe;
}

static inline bool my_simd_support_check_uca_900_primary(
    const CHARSET_INFO *cs)
{
  return cs->levels_for_compare == 1 && my_simd_support_check_uca_900(cs) &&
         my_uca900_printable_ascii_primary_safe(cs);
}

static inline void my_uca_build_printable_ascii_lookup(const uint16 *wpage,
                                                       uchar wlength,
                                                       uint16 *ascii_weights,
                                                       bool *compare_safe)
{
  bool safe = true;
  for (size_t i = 0; i < kPrintableAsciiCount; ++i)
  {
    const uint16 *weight = wpage + (kPrintableAsciiFirst + i) * wlength;
    ascii_weights[i] = weight[0];
    if (weight[0] == 0)
    {
      safe = false;
      continue;
    }
    for (uchar j = 1; j < wlength; ++j)
    {
      if (weight[j] != 0)
      {
        safe = false;
        break;
      }
    }
  }
  *compare_safe = safe;
}

static inline uint16 my_uca_lookup_printable_ascii_weight(
    const uint16 *ascii_weights, uchar ch)
{
  return ascii_weights[ch - kPrintableAsciiFirst];
}

struct my_uca_printable_ascii_cache_t
{
  const uint16 *wpage;
  uchar wlength;
  bool compare_safe;
  uint16 ascii_weights[kPrintableAsciiCount];
};

static inline const my_uca_printable_ascii_cache_t *my_uca_get_printable_ascii_cache(
    const uint16 *wpage, uchar wlength)
{
  static thread_local my_uca_printable_ascii_cache_t cache = {
      nullptr, 0, false, {0}};

  if (cache.wpage != wpage || cache.wlength != wlength)
  {
    my_uca_build_printable_ascii_lookup(wpage, wlength, cache.ascii_weights,
                                        &cache.compare_safe);
    cache.wpage = wpage;
    cache.wlength = wlength;
  }

  return &cache;
}

static inline const uint16 *my_uca_get_printable_ascii_lookup(
    const uint16 *wpage, uchar wlength)
{
  return my_uca_get_printable_ascii_cache(wpage, wlength)->ascii_weights;
}

static inline bool my_uca_printable_ascii_compare_safe(const uint16 *wpage,
                                                       uchar wlength)
{
  return my_uca_get_printable_ascii_cache(wpage, wlength)->compare_safe;
}

static inline bool my_simd_support_check_uca_compare(const CHARSET_INFO *cs,
                                                     bool t_is_prefix)
{
  if (t_is_prefix || cs->levels_for_compare != 1)
  {
    return false;
  }

  if (!my_simd_support_check_uca_any(cs) || cs->uca->version != UCA_V400)
  {
    return false;
  }

  if (cs->tailoring != nullptr && cs->tailoring[0] != '\0')
  {
    my_sve_test_note_tailoring_disabled_if_enabled();
    return false;
  }

  const uint16 *wpage = cs->uca->weights[0];
  const uchar wlength = cs->uca->lengths[0];
  return wpage != nullptr && wlength != 0 &&
         my_uca_printable_ascii_compare_safe(wpage, wlength);
}

/*
  UCA 比较的保守 SIMD 前缀路径。仅当双方完整向量块均为 printable
  ASCII，且该范围内字符都可证明为单一非零排序权重时，才按权重比较
  或跳过相等前缀；其余情况交还既有 scanner 处理。
*/
static inline my_uca_ascii_compare_prefix_result my_strnncoll_uca_ascii_prefix(
    const CHARSET_INFO *cs, const uchar *s, size_t slen, const uchar *t,
    size_t tlen, bool t_is_prefix)
{
  my_uca_ascii_compare_prefix_result result = {s, slen, t, tlen, false, 0};

  if (!my_simd_support_check_uca_compare(cs, t_is_prefix))
  {
    return result;
  }

  const uint16 *wpage = cs->uca->weights[0];
  const uchar wlength = cs->uca->lengths[0];
  const uint16 *ascii_weights =
      my_uca_get_printable_ascii_lookup(wpage, wlength);
  const size_t vector_bytes = my_simd_lanes_u8();
  if (vector_bytes == 0)
  {
    return result;
  }

  const svbool_t pg = svptrue_b8();
  sve_u8_vector_t s_vec;
  sve_u8_vector_t t_vec;

  while (slen >= vector_bytes && tlen >= vector_bytes)
  {
    if (!my_simd_load_printable_ascii_vector(s, &s_vec) ||
        !my_simd_load_printable_ascii_vector(t, &t_vec))
    {
      break;
    }

    my_sve_test_note_fast_path_if_enabled(vector_bytes);
    if (!svptest_any(pg, svcmpne_u8(pg, s_vec, t_vec)))
    {
      s += vector_bytes;
      t += vector_bytes;
      slen -= vector_bytes;
      tlen -= vector_bytes;
      result.s = s;
      result.t = t;
      result.slen = slen;
      result.tlen = tlen;
      continue;
    }

    const size_t first_diff = my_simd_first_mismatch_u8(s_vec, t_vec);
    for (size_t i = first_diff; i < vector_bytes; ++i)
    {
      const uint16 s_weight =
          my_uca_lookup_printable_ascii_weight(ascii_weights, s[i]);
      const uint16 t_weight =
          my_uca_lookup_printable_ascii_weight(ascii_weights, t[i]);
      if (s_weight != t_weight)
      {
        result.decided = true;
        result.result =
            static_cast<int>(s_weight) - static_cast<int>(t_weight);
        return result;
      }
    }

    s += vector_bytes;
    t += vector_bytes;
    slen -= vector_bytes;
    tlen -= vector_bytes;
    result.s = s;
    result.t = t;
    result.slen = slen;
    result.tlen = tlen;
  }

  my_sve_test_note_tail_if_needed(std::max(slen, tlen));
  return result;
}

/*
  UCA 9.0 primary-level 比较使用的权重布局不同于 UCA 4.0。
  因此该 SIMD 前缀路径单独保留，并且只消费完整的 printable ASCII
  向量块；块内字符必须都具有单一且非零的 primary 权重。
*/
static inline my_uca_ascii_compare_prefix_result
my_strnncoll_uca_900_ascii_prefix(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen,
                                  bool t_is_prefix)
{
  my_uca_ascii_compare_prefix_result result = {s, slen, t, tlen, false, 0};

  if (t_is_prefix || !my_simd_support_check_uca_900_primary(cs))
  {
    return result;
  }

  const uint16 *ascii_weights = my_uca900_printable_ascii_primary_weights(cs);
  const size_t vector_bytes = my_simd_lanes_u8();
  if (ascii_weights == nullptr || vector_bytes == 0)
  {
    return result;
  }

  const svbool_t pg = svptrue_b8();
  sve_u8_vector_t s_vec;
  sve_u8_vector_t t_vec;

  while (slen >= vector_bytes && tlen >= vector_bytes)
  {
    if (!my_simd_load_printable_ascii_vector(s, &s_vec) ||
        !my_simd_load_printable_ascii_vector(t, &t_vec))
    {
      break;
    }

    my_sve_test_note_fast_path_if_enabled(vector_bytes);
    if (!svptest_any(pg, svcmpne_u8(pg, s_vec, t_vec)))
    {
      s += vector_bytes;
      t += vector_bytes;
      slen -= vector_bytes;
      tlen -= vector_bytes;
      result.s = s;
      result.t = t;
      result.slen = slen;
      result.tlen = tlen;
      continue;
    }

    const size_t first_diff = my_simd_first_mismatch_u8(s_vec, t_vec);
    for (size_t i = first_diff; i < vector_bytes; ++i)
    {
      const uint16 s_weight = ascii_weights[s[i]];
      const uint16 t_weight = ascii_weights[t[i]];
      if (s_weight != t_weight)
      {
        result.decided = true;
        result.result =
            static_cast<int>(s_weight) - static_cast<int>(t_weight);
        return result;
      }
    }

    s += vector_bytes;
    t += vector_bytes;
    slen -= vector_bytes;
    tlen -= vector_bytes;
    result.s = s;
    result.t = t;
    result.slen = slen;
    result.tlen = tlen;
  }

  my_sve_test_note_tail_if_needed(std::max(slen, tlen));
  return result;
}

static int my_strnncoll_any_uca(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix)
{
  return my_strnncoll_any_uca_opt<my_uca_skip_dual_space_sve>(
      cs, my_strnncoll_uca_ascii_prefix, s, slen, t, tlen, t_is_prefix);
}

static int my_strnncollsp_any_uca(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen)
{
  return my_strnncollsp_any_uca_opt<my_uca_skip_dual_space_sve>(cs, s, slen, t,
                                                                tlen);
}

static int my_strnncoll_uca_900(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix)
{
  return my_strnncoll_uca_900_opt<my_uca_skip_dual_space_sve>(
      cs, my_strnncoll_uca_900_ascii_prefix, s, slen, t, tlen, t_is_prefix);
}

static int my_strnncollsp_uca_900(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen)
{
  return my_strnncollsp_uca_900_opt<my_uca_skip_dual_space_sve>(
      cs, my_strnncoll_uca_900_ascii_prefix, s, slen, t, tlen);
}

static inline void my_uca900_hash_sort_mix(uint16 weight, uint64 *h)
{
  *h ^= weight;
  *h *= 1099511628211ULL;
}

static inline void my_hash_sort_uca_900_ascii_prefix(const CHARSET_INFO *cs,
                                                     const uchar **s,
                                                     size_t *slen, uint64 *h)
{
  if (!my_simd_support_check_uca_900_primary(cs))
  {
    return;
  }

  const uint16 *ascii_weights = my_uca900_printable_ascii_primary_weights(cs);
  const size_t vector_bytes = my_simd_lanes_u8();
  if (ascii_weights == nullptr || vector_bytes == 0)
  {
    return;
  }

  sve_u8_vector_t vec_src;
  while (*slen >= vector_bytes)
  {
    if (!my_simd_load_printable_ascii_vector(*s, &vec_src))
    {
      break;
    }

    for (size_t i = 0; i < vector_bytes; ++i)
    {
      my_uca900_hash_sort_mix(ascii_weights[(*s)[i]], h);
    }
    my_sve_test_note_fast_path_if_enabled(vector_bytes);
    *s += vector_bytes;
    *slen -= vector_bytes;
  }
  my_sve_test_note_tail_if_needed(*slen);
}

template <class Mb_wc, int LEVELS_FOR_COMPARE>
static void my_hash_sort_uca_900_tmpl(const CHARSET_INFO *cs, const Mb_wc mb_wc,
                                      const uchar *s, size_t slen,
                                      uint64 *n1) {
  uint64 h = *n1;
  h ^= 14695981039346656037ULL;

  if (LEVELS_FOR_COMPARE == 1) {
    my_hash_sort_uca_900_ascii_prefix(cs, &s, &slen, &h);
  }

  uca_scanner_900<Mb_wc, LEVELS_FOR_COMPARE> scanner(mb_wc, cs, s, slen);

  scanner.for_each_weight(
      [&](int s_res, bool) -> bool {
        h ^= s_res;
        h *= 1099511628211ULL;
        return true;
      },
      [](int) { return true; });

  *n1 = h;
}

/**
  功能：
    根据 UCA 校对规则计算字符串哈希，并忽略尾部空格。

  参数：
    cs：字符集信息。
    mb_wc：多字节解码函数。
    s：输入串。
    slen：输入串长度。
    n1：第一个哈希参数。
    n2：第二个哈希参数。

  返回值：
    无。

  说明：
    按排序权重顺序更新 `n1` 和 `n2`。
    在大小写不敏感的校对规则下，同一字母的大小写形式会得到相同哈希值。
*/
template <class Mb_wc>
static void my_hash_sort_uca(const CHARSET_INFO *cs, Mb_wc mb_wc,
                             const uchar *s, size_t slen, uint64 *n1,
                             uint64 *n2) {
  int s_res;
  uint64 tmp1;
  uint64 tmp2;

  slen = cs->cset->lengthsp(cs, pointer_cast<const char *>(s), slen);

  tmp1 = *n1;
  tmp2 = *n2;

  if (my_simd_support_check_uca_any(cs)) {
    sve_u8_vector_t vec_src;
    const size_t vector_bytes = my_simd_lanes_u8();
    const uint16 *wpage = cs->uca->weights[0];
    const uchar wlength = cs->uca->lengths[0];
    const uint16 *ascii_weights =
        my_uca_get_printable_ascii_lookup(wpage, wlength);

    while (slen >= vector_bytes) {
      if (!my_simd_load_printable_ascii_vector(s, &vec_src)) {
        break;
      }

      for (size_t i = 0; i < vector_bytes; i++) {
        const uint16 weight =
            my_uca_lookup_printable_ascii_weight(ascii_weights, s[i]);
        if (weight == 0) {
          continue;
        }
        tmp1 ^= (((tmp1 & 63) + tmp2) * (weight >> 8)) + (tmp1 << 8);
        tmp2 += 3;
        tmp1 ^= (((tmp1 & 63) + tmp2) * (weight & 0xFF)) + (tmp1 << 8);
        tmp2 += 3;
      }
      my_sve_test_note_fast_path_if_enabled(vector_bytes);
      s += vector_bytes;
      slen -= vector_bytes;
    }
    my_sve_test_note_tail_if_needed(slen);
  }

  uca_scanner_any<Mb_wc> scanner(mb_wc, cs, s, slen);
  while ((s_res = scanner.next()) > 0) {
    tmp1 ^= (((tmp1 & 63) + tmp2) * (s_res >> 8)) + (tmp1 << 8);
    tmp2 += 3;
    tmp1 ^= (((tmp1 & 63) + tmp2) * (s_res & 0xFF)) + (tmp1 << 8);
    tmp2 += 3;
  }

  *n1 = tmp1;
  *n2 = tmp2;
}

static void my_hash_sort_any_uca(const CHARSET_INFO *cs, const uchar *s,
                                 size_t slen, uint64 *n1, uint64 *n2) {
  if (cs->cset->mb_wc == my_mb_wc_utf8mb4_thunk) {
    my_hash_sort_uca(cs, Mb_wc_utf8mb4(), s, slen, n1, n2);
  } else {
    Mb_wc_through_function_pointer mb_wc(cs);
    my_hash_sort_uca(cs, mb_wc, s, slen, n1, n2);
  }
}

/**
  功能：
    生成可直接用于 `memcmp()` 的 UCA 排序键。

  参数：
    cs：字符集信息。
    mb_wc：多字节解码函数。
    dst：输出缓冲区起始位置。
    dstlen：输出缓冲区可用空间。
    num_codepoints：请求写入的码点数。
    src：输入串。
    srclen：输入串长度。
    flags：转换标志。

  返回值：
    已写入的字节数。

  说明：
    输出结果是排序键的二进制表示，仅用于比较，无法还原原始字符串。
    当输入串结束或遇到非法多字节序列时，生成过程停止。
*/
template <class Mb_wc>
static size_t my_strnxfrm_uca(const CHARSET_INFO *cs, Mb_wc mb_wc, uchar *dst,
                              size_t dstlen, uint num_codepoints,
                              const uchar *src, size_t srclen, uint flags) {
  uchar *d0 = dst;
  uchar *de = dst + dstlen;
  int s_res;

  if (my_simd_support_check_uca_any(cs)) {
    sve_u8_vector_t vec_src;
    const size_t vector_bytes = my_simd_lanes_u8();
    const uint16 *wpage = cs->uca->weights[0];
    const uchar wlength = cs->uca->lengths[0];
    const uint16 *ascii_weights =
        my_uca_get_printable_ascii_lookup(wpage, wlength);

    while (srclen >= vector_bytes &&
           (size_t)(de - dst) >= vector_bytes * 2) {
      if (!my_simd_load_printable_ascii_vector(src, &vec_src)) {
        break;
      }

      for (size_t i = 0; i < vector_bytes; i++) {
        const uint16 weight =
            my_uca_lookup_printable_ascii_weight(ascii_weights, src[i]);
        if (weight == 0) {
          continue;
        }
        *dst++ = weight >> 8;
        *dst++ = weight & 0xFF;
      }
      my_sve_test_note_fast_path_if_enabled(vector_bytes);
      src += vector_bytes;
      srclen -= vector_bytes;
    }
    my_sve_test_note_tail_if_needed(srclen);
  }

  uca_scanner_any<Mb_wc> scanner(mb_wc, cs, src, srclen);
  while (dst < de && (s_res = scanner.next()) > 0) {
    *dst++ = s_res >> 8;
    if (dst < de) *dst++ = s_res & 0xFF;
  }

  if (dst < de) {
    /*
      PAD SPACE 语义要求：当扫描器已经到达最后一级末尾而输出缓冲区
      仍有剩余空间时，需要根据尚未写入的码点数补齐空格权重。
    */
    assert(num_codepoints >= scanner.get_char_index());
    num_codepoints -= scanner.get_char_index();

    if (num_codepoints) {
      uint space_count = std::min<uint>((de - dst) / 2, num_codepoints);
      s_res = my_space_weight(cs);
      for (; space_count; space_count--) {
        dst = store16be(dst, s_res);
      }
    }
  }
  if ((flags & MY_STRXFRM_PAD_TO_MAXLEN) && dst < de) {
    s_res = my_space_weight(cs);
    for (; dst < de;) {
      *dst++ = s_res >> 8;
      if (dst < de) *dst++ = s_res & 0xFF;
    }
  }
  return dst - d0;
}

static size_t my_strnxfrm_any_uca(const CHARSET_INFO *cs, uchar *dst,
                                  size_t dstlen, uint num_codepoints,
                                  const uchar *src, size_t srclen,
                                  uint flags) {
  if (cs->cset->mb_wc == my_mb_wc_utf8mb4_thunk) {
    return my_strnxfrm_uca(cs, Mb_wc_utf8mb4(), dst, dstlen, num_codepoints,
                           src, srclen, flags);
  }

  Mb_wc_through_function_pointer mb_wc(cs);
  return my_strnxfrm_uca(cs, mb_wc, dst, dstlen, num_codepoints, src, srclen,
                         flags);
}

#endif /* CTYPE_UCA */

#ifdef CTYPE_MB
size_t my_numchars_mb(const CHARSET_INFO *cs, const char *pos,
                      const char *end) {
  size_t count = 0;

  if (my_simd_support_check(cs)) {
    sve_u8_vector_t vec_src;
    const size_t vector_bytes = my_simd_lanes_u8();
    while ((size_t)(end - pos) >= vector_bytes) {
      if (!my_simd_load_ascii_vector((const uchar *)pos, &vec_src)) {
        break;
      }

      my_sve_test_note_fast_path_if_enabled(vector_bytes);
      count += vector_bytes;
      pos += vector_bytes;
    }
    my_sve_test_note_tail_if_needed((size_t)(end - pos));
  }

  while (pos < end) {
    uint mb_len;

    while (pos < end && (uchar)*pos < 0x80) {
      pos++;
      count++;
    }
    if (pos >= end) {
      break;
    }

    pos += (mb_len = my_ismbchar(cs, pos, end)) ? mb_len : 1;
    count++;
  }
  return count;
}

size_t my_charpos_mb3(const CHARSET_INFO *cs, const char *pos, const char *end,
                      size_t length) {
  const char *start = pos;

  if (my_simd_support_check(cs)) {
    sve_u8_vector_t vec_src;
    const size_t vector_bytes = my_simd_lanes_u8();
    while (length >= vector_bytes && (size_t)(end - pos) >= vector_bytes) {
      if (!my_simd_load_ascii_vector((const uchar *)pos, &vec_src)) {
        break;
      }

      my_sve_test_note_fast_path_if_enabled(vector_bytes);
      length -= vector_bytes;
      pos += vector_bytes;
    }
    my_sve_test_note_tail_if_needed(length ? (size_t)(end - pos) : 0);
  }

  while (length && pos < end) {
    uint mb_len;

    if ((uchar)*pos < 0x80) {
      pos++;
    } else {
      pos += (mb_len = my_ismbchar(cs, pos, end)) ? mb_len : 1;
    }
    length--;
  }
  return (size_t)(length ? end + 2 - start : pos - start);
}
#endif /* CTYPE_MB */
