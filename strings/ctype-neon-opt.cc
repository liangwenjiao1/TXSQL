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

#include <arm_neon.h>

#ifdef CTYPE_UTF8
#include "ctype-arm-simd.h"
#endif

#include "m_string.h"

#define NEON_VECTOR_LENGTH 16
#define MAX_ASCII 0x7F
#define CTYPE_UTF8_OPTIMIZED
#define CTYPE_UCA_OPTIMIZED

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
    while (length >= NEON_VECTOR_LENGTH) {
      const uint8x16_t vec = vld1q_u8(pointer_cast<const uint8_t *>(from));
      if (vmaxvq_u8(vec) > MAX_ASCII) {
        size_t i = 0;
        for (; i < NEON_VECTOR_LENGTH; ++i) {
          if (static_cast<uchar>(from[i]) > MAX_ASCII) break;
          to[i] = from[i];
        }
        from += i;
        to += i;
        length -= i;
        break;
      }

      vst1q_u8(pointer_cast<uint8_t *>(to), vec);
      from += NEON_VECTOR_LENGTH;
      to += NEON_VECTOR_LENGTH;
      length -= NEON_VECTOR_LENGTH;
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

static inline bool my_neon_unicode_ascii_compatible(const CHARSET_INFO *cs) {
  return (cs->state & MY_CS_UNICODE) && !(cs->state & MY_CS_NONASCII);
}

static inline bool my_neon_all_ascii_u8x16(const uchar *src) {
  const uint8x16_t vec_src = vld1q_u8(src);
  const uint8x16_t result = vcleq_u8(vec_src, vdupq_n_u8(MAX_ASCII));
  return vminvq_u8(result) != 0;
}

static inline void my_neon_tosort_ascii_u8x16(const CHARSET_INFO *cs,
                                              uint8x16_t *vec_src) {
  const uint8x16_t diff = vdupq_n_u8(0x20);

  if (cs->state & MY_CS_LOWER_SORT) {
    const uint8x16_t mask = vcgeq_u8(*vec_src, vdupq_n_u8('A')) &
                            vcleq_u8(*vec_src, vdupq_n_u8('Z'));
    *vec_src = vbslq_u8(mask, vaddq_u8(*vec_src, diff), *vec_src);
  } else {
    const uint8x16_t mask = vcgeq_u8(*vec_src, vdupq_n_u8('a')) &
                            vcleq_u8(*vec_src, vdupq_n_u8('z'));
    *vec_src = vbslq_u8(mask, vsubq_u8(*vec_src, diff), *vec_src);
  }
}

#ifdef CTYPE_UCA
#include "ctype-arm-simd.h"

static inline size_t my_uca_skip_dual_space_neon(const uchar **sp,
                                                 const uchar **tp,
                                                 const uchar *se,
                                                 const uchar *te) {
  size_t skipped = 0;
  const uint8x16_t space = vdupq_n_u8(' ');

  while (*sp + NEON_VECTOR_LENGTH <= se && *tp + NEON_VECTOR_LENGTH <= te) {
    const uint8x16_t vec_s = vld1q_u8(*sp);
    const uint8x16_t vec_t = vld1q_u8(*tp);
    const uint8x16_t both_space =
        vandq_u8(vceqq_u8(vec_s, space), vceqq_u8(vec_t, space));

    if (vminvq_u8(both_space) == 0) {
      break;
    }

    *sp += NEON_VECTOR_LENGTH;
    *tp += NEON_VECTOR_LENGTH;
    skipped += NEON_VECTOR_LENGTH;
  }

  while (*sp < se && *tp < te && **sp == ' ' && **tp == ' ') {
    ++(*sp);
    ++(*tp);
    ++skipped;
  }

  return skipped;
}
#endif

#ifdef CTYPE_MB
#define CTYPE_MB_OPTIMIZED

size_t my_numchars_mb(const CHARSET_INFO *cs, const char *pos,
                      const char *end) {
  size_t count = 0;

  if (my_neon_unicode_ascii_compatible(cs)) {
    while (pos + NEON_VECTOR_LENGTH <= end) {
      if (!my_neon_all_ascii_u8x16(pointer_cast<const uchar *>(pos))) {
        break;
      }

      count += NEON_VECTOR_LENGTH;
      pos += NEON_VECTOR_LENGTH;
    }
  }

  while (pos < end) {
    uint mb_len;
    pos += (mb_len = my_ismbchar(cs, pos, end)) ? mb_len : 1;
    count++;
  }
  return count;
}

size_t my_charpos_mb3(const CHARSET_INFO *cs, const char *pos, const char *end,
                      size_t length) {
  const char *start = pos;

  if (my_neon_unicode_ascii_compatible(cs)) {
    while (length >= NEON_VECTOR_LENGTH && pos + NEON_VECTOR_LENGTH <= end) {
      if (!my_neon_all_ascii_u8x16(pointer_cast<const uchar *>(pos))) {
        break;
      }

      length -= NEON_VECTOR_LENGTH;
      pos += NEON_VECTOR_LENGTH;
    }
  }

  while (length && pos < end) {
    uint mb_len;
    pos += (mb_len = my_ismbchar(cs, pos, end)) ? mb_len : 1;
    length--;
  }
  return (size_t)(length ? end + 2 - start : pos - start);
}
#endif

#ifdef CTYPE_UCA

static inline bool my_neon_all_printable_ascii_u8x16(const uchar *src) {
  const uint8x16_t vec_src = vld1q_u8(src);
  const uint8x16_t ge_space = vcgeq_u8(vec_src, vdupq_n_u8(0x20));
  const uint8x16_t le_tilde = vcleq_u8(vec_src, vdupq_n_u8(0x7E));
  return vminvq_u8(vandq_u8(ge_space, le_tilde)) != 0;
}

static inline bool my_neon_legacy_uca_unicode_ci(const CHARSET_INFO *cs) {
  return cs->m_coll_name != nullptr &&
         (strcmp(cs->m_coll_name, "utf8mb3_unicode_ci") == 0 ||
          strcmp(cs->m_coll_name, "utf8mb4_unicode_ci") == 0);
}

static inline bool my_neon_support_check_uca_any(const CHARSET_INFO *cs) {
  if (cs->uca == nullptr || cs->uca->version == UCA_V900 ||
      cs->mbminlen != 1 || (cs->state & MY_CS_NONASCII)) {
    return false;
  }

  if ((cs->tailoring != nullptr && cs->tailoring[0] != '\0') ||
      cs->coll_param != nullptr || my_uca_have_contractions(cs->uca)) {
    return false;
  }

  if (!my_neon_legacy_uca_unicode_ci(cs)) {
    return false;
  }

  const uint16 *wpage = cs->uca->weights[0];
  const uchar wlength = cs->uca->lengths[0];
  if (wpage == nullptr || wlength == 0) {
    return false;
  }

  for (uchar ch = 0x20; ch <= 0x7E; ++ch) {
    const uint16 *weight = wpage + ch * wlength;
    if (weight[0] == 0) {
      return false;
    }
    for (uchar i = 1; i < wlength; ++i) {
      if (weight[i] != 0) {
        return false;
      }
    }
  }

  return true;
}

static inline my_uca_ascii_compare_prefix_result my_strnncoll_uca_ascii_prefix(
    const CHARSET_INFO *cs, const uchar *s, size_t slen, const uchar *t,
    size_t tlen, bool t_is_prefix) {
  my_uca_ascii_compare_prefix_result result = {s, slen, t, tlen, false, 0};

  if (t_is_prefix || !my_neon_support_check_uca_any(cs)) {
    return result;
  }

  const uint16 *wpage = cs->uca->weights[0];
  const uchar wlength = cs->uca->lengths[0];

  while (slen >= NEON_VECTOR_LENGTH && tlen >= NEON_VECTOR_LENGTH) {
    if (!my_neon_all_printable_ascii_u8x16(s) ||
        !my_neon_all_printable_ascii_u8x16(t)) {
      break;
    }

    if (memcmp(s, t, NEON_VECTOR_LENGTH) == 0) {
      s += NEON_VECTOR_LENGTH;
      t += NEON_VECTOR_LENGTH;
      slen -= NEON_VECTOR_LENGTH;
      tlen -= NEON_VECTOR_LENGTH;
      result.s = s;
      result.t = t;
      result.slen = slen;
      result.tlen = tlen;
      continue;
    }

    for (size_t i = 0; i < NEON_VECTOR_LENGTH; ++i) {
      if (s[i] == t[i]) {
        continue;
      }

      const uint16 s_weight = wpage[s[i] * wlength];
      const uint16 t_weight = wpage[t[i] * wlength];
      if (s_weight != t_weight) {
        result.decided = true;
        result.result = static_cast<int>(s_weight) - static_cast<int>(t_weight);
        return result;
      }
    }

    s += NEON_VECTOR_LENGTH;
    t += NEON_VECTOR_LENGTH;
    slen -= NEON_VECTOR_LENGTH;
    tlen -= NEON_VECTOR_LENGTH;
    result.s = s;
    result.t = t;
    result.slen = slen;
    result.tlen = tlen;
  }

  return result;
}

static inline my_uca_ascii_compare_prefix_result my_strnncoll_uca_900_ascii_prefix(
    const CHARSET_INFO *cs, const uchar *s, size_t slen, const uchar *t,
    size_t tlen, bool t_is_prefix) {
  (void)cs;
  (void)t_is_prefix;
  return {s, slen, t, tlen, false, 0};
}

static int my_strnncoll_any_uca(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix) {
  return my_strnncoll_any_uca_opt<my_uca_skip_dual_space_neon>(
      cs, my_strnncoll_uca_ascii_prefix, s, slen, t, tlen, t_is_prefix);
}

static int my_strnncollsp_any_uca(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen) {
  return my_strnncollsp_any_uca_opt<my_uca_skip_dual_space_neon>(cs, s, slen,
                                                                 t, tlen);
}

static int my_strnncoll_uca_900(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix) {
  return my_strnncoll_uca_900_opt<my_uca_skip_dual_space_neon>(
      cs, my_strnncoll_uca_900_ascii_prefix, s, slen, t, tlen, t_is_prefix);
}

static int my_strnncollsp_uca_900(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen) {
  return my_strnncollsp_uca_900_opt<my_uca_skip_dual_space_neon>(
      cs, my_strnncoll_uca_900_ascii_prefix, s, slen, t, tlen);
}

static inline void my_neon_hash_sort_uca_prefix(const CHARSET_INFO *cs,
                                                const uchar **s,
                                                size_t *slen, uint64 *n1,
                                                uint64 *n2) {
  if (!my_neon_support_check_uca_any(cs)) {
    return;
  }

  const uint16 *wpage = cs->uca->weights[0];
  const uchar wlength = cs->uca->lengths[0];

  while (*slen >= NEON_VECTOR_LENGTH) {
    if (!my_neon_all_printable_ascii_u8x16(*s)) {
      break;
    }

    for (size_t i = 0; i < NEON_VECTOR_LENGTH; ++i) {
      const uint16 weight = wpage[(*s)[i] * wlength];
      *n1 ^= (((*n1 & 63) + *n2) * (weight >> 8)) + (*n1 << 8);
      *n2 += 3;
      *n1 ^= (((*n1 & 63) + *n2) * (weight & 0xFF)) + (*n1 << 8);
      *n2 += 3;
    }

    *s += NEON_VECTOR_LENGTH;
    *slen -= NEON_VECTOR_LENGTH;
  }
}

static inline void my_neon_strnxfrm_uca_prefix(
    const CHARSET_INFO *cs, uchar **dst, uchar *de, uint *num_codepoints,
    const uchar **src, size_t *srclen) {
  if (!my_neon_support_check_uca_any(cs)) {
    return;
  }

  const uint16 *wpage = cs->uca->weights[0];
  const uchar wlength = cs->uca->lengths[0];

  while (*srclen >= NEON_VECTOR_LENGTH &&
         static_cast<size_t>(de - *dst) >= NEON_VECTOR_LENGTH * 2 &&
         *num_codepoints >= NEON_VECTOR_LENGTH) {
    if (!my_neon_all_printable_ascii_u8x16(*src)) {
      break;
    }

    for (size_t i = 0; i < NEON_VECTOR_LENGTH; ++i) {
      const uint16 weight = wpage[(*src)[i] * wlength];
      *(*dst)++ = weight >> 8;
      *(*dst)++ = weight & 0xFF;
    }

    *src += NEON_VECTOR_LENGTH;
    *srclen -= NEON_VECTOR_LENGTH;
    *num_codepoints -= NEON_VECTOR_LENGTH;
  }
}

static inline void my_hash_sort_uca_900_ascii_prefix(const CHARSET_INFO *cs,
                                                     const uchar **s,
                                                     size_t *slen,
                                                     uint64 *h) {
  (void)cs;
  (void)s;
  (void)slen;
  (void)h;
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
  my_neon_hash_sort_uca_prefix(cs, &s, &slen, &tmp1, &tmp2);

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

template <class Mb_wc>
static size_t my_strnxfrm_uca(const CHARSET_INFO *cs, Mb_wc mb_wc, uchar *dst,
                              size_t dstlen, uint num_codepoints,
                              const uchar *src, size_t srclen, uint flags) {
  uchar *d0 = dst;
  uchar *de = dst + dstlen;
  int s_res;

  my_neon_strnxfrm_uca_prefix(cs, &dst, de, &num_codepoints, &src, &srclen);

  uca_scanner_any<Mb_wc> scanner(mb_wc, cs, src, srclen);
  while (dst < de && (s_res = scanner.next()) > 0) {
    *dst++ = s_res >> 8;
    if (dst < de) *dst++ = s_res & 0xFF;
  }

  if (dst < de) {
    assert(num_codepoints >= scanner.get_char_index());
    num_codepoints -= scanner.get_char_index();

    if (num_codepoints) {
      uint space_count = std::min<uint>((de - dst) / 2, num_codepoints);
      s_res = my_space_weight(cs);
      for (; space_count; --space_count) {
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

#endif

#ifdef CTYPE_UTF8
static inline void my_neon_strnxfrm_unicode_ascii_prefix(
    const CHARSET_INFO *cs, uchar **dst, uchar *de, uint *nweights,
    const uchar **src, const uchar *se) {
  if (cs->state & MY_CS_NONASCII) {
    return;
  }

  const MY_UNICASE_INFO *uni_plane =
      (cs->state & MY_CS_BINSORT) ? nullptr : cs->caseinfo;

  for (; *dst + 16 <= de && *src + 16 <= se && *nweights >= 8;
       *nweights -= 8) {
    if (uint8korr(*src) & 0x8080808080808080ULL) {
      break;
    }

    uint8x8_t vec_src = vld1_u8(*src);
    if (uni_plane != nullptr) {
      const uint8x8_t diff = vdup_n_u8(0x20);
      if (cs->state & MY_CS_LOWER_SORT) {
        const uint8x8_t mask = vcge_u8(vec_src, vdup_n_u8('A')) &
                               vcle_u8(vec_src, vdup_n_u8('Z'));
        vec_src = vbsl_u8(mask, vadd_u8(vec_src, diff), vec_src);
      } else {
        const uint8x8_t mask = vcge_u8(vec_src, vdup_n_u8('a')) &
                               vcle_u8(vec_src, vdup_n_u8('z'));
        vec_src = vbsl_u8(mask, vsub_u8(vec_src, diff), vec_src);
      }
    }

    uint8x16_t vec_dst = vreinterpretq_u8_u16(vmovl_u8(vec_src));
    vec_dst = vrev16q_u8(vec_dst);
    vst1q_u8(*dst, vec_dst);

    *src += 8;
    *dst += 16;
  }
}

static inline void my_neon_hash_sort_utf8mb3_prefix(const CHARSET_INFO *cs,
                                                    const uchar **s,
                                                    const uchar *e,
                                                    uint64 *tmp1,
                                                    uint64 *tmp2) {
  if (!my_neon_unicode_ascii_compatible(cs)) {
    return;
  }

  uchar vec_dst[NEON_VECTOR_LENGTH];
  while (*s + NEON_VECTOR_LENGTH <= e) {
    uint8x16_t vec_src = vld1q_u8(*s);
    if (vminvq_u8(vcleq_u8(vec_src, vdupq_n_u8(MAX_ASCII))) == 0) {
      break;
    }

    my_neon_tosort_ascii_u8x16(cs, &vec_src);
    vst1q_u8(vec_dst, vec_src);
    for (size_t i = 0; i < NEON_VECTOR_LENGTH; ++i) {
      *tmp1 ^= (((*tmp1 & 63) + *tmp2) * vec_dst[i]) + (*tmp1 << 8);
      *tmp1 ^= (*tmp1 << 8);
      *tmp2 += 6;
    }

    *s += NEON_VECTOR_LENGTH;
  }
}

static inline void my_neon_hash_sort_utf8mb4_prefix(const CHARSET_INFO *cs,
                                                    const uchar **s,
                                                    const uchar *e,
                                                    uint64 *tmp1,
                                                    uint64 *tmp2) {
  my_neon_hash_sort_utf8mb3_prefix(cs, s, e, tmp1, tmp2);
}

static inline void my_neon_skip_space_pair(const uchar **sp, const uchar **tp,
                                           const uchar *se,
                                           const uchar *te) {
  const uint8x16_t space_ascii = vdupq_n_u8(0x20);

  while (*sp + NEON_VECTOR_LENGTH <= se && *tp + NEON_VECTOR_LENGTH <= te) {
    const uint8x16_t vec_sp = vld1q_u8(*sp);
    const uint8x16_t vec_tp = vld1q_u8(*tp);
    const uint8x16_t res_sp = vceqq_u8(vec_sp, space_ascii);
    const uint8x16_t res_tp = vceqq_u8(vec_tp, space_ascii);

    if (vminvq_u8(res_sp) == 0 || vminvq_u8(res_tp) == 0) {
      break;
    }

    *sp += NEON_VECTOR_LENGTH;
    *tp += NEON_VECTOR_LENGTH;
  }

  while (*sp + 8 <= se && *tp + 8 <= te) {
    uint64_t s_chunk;
    uint64_t t_chunk;
    memcpy(&s_chunk, *sp, sizeof(s_chunk));
    memcpy(&t_chunk, *tp, sizeof(t_chunk));
    if (s_chunk != 0x2020202020202020ULL ||
        t_chunk != 0x2020202020202020ULL) {
      break;
    }

    *sp += sizeof(s_chunk);
    *tp += sizeof(t_chunk);
  }

  while (*sp < se && *tp < te && **sp == 0x20 && **tp == 0x20) {
    ++(*sp);
    ++(*tp);
  }
}

static size_t my_strxfrm_pad_unicode(uchar *str, uchar *strend) {
  uchar *str0 = str;
  assert(str && str <= strend);
  for (; str < strend;) {
    *str++ = 0x00;
    if (str < strend) *str++ = 0x20;
  }
  return str - str0;
}

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

  my_neon_strnxfrm_unicode_ascii_prefix(cs, &dst, de, &nweights, &src, se);

  if (cs->state & MY_CS_BINSORT) {
    const size_t nweights_fast_path = std::min<size_t>((de - dst) / 2, nweights);
    for (size_t i = 0; i < nweights_fast_path; ++i, --nweights) {
      my_wc_t wc;
      int res = mb_wc(&wc, src, se);
      if (res <= 0) goto pad;
      src += res;
      dst = store16be(dst, wc);
    }

    if (dst < de && nweights) {
      my_wc_t wc;
      int res = mb_wc(&wc, src, se);
      if (res > 0) *dst++ = wc >> 8;
    }
  } else {
    const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
    const size_t nweights_fast_path = std::min<size_t>((de - dst) / 2, nweights);
    for (size_t i = 0; i < nweights_fast_path; ++i, --nweights) {
      my_wc_t wc;
      int res = mb_wc(&wc, src, se);
      if (res <= 0) goto pad;
      src += res;
      my_tosort_unicode(uni_plane, &wc, cs->state);
      dst = store16be(dst, wc);
    }

    if (dst < de && nweights) {
      my_wc_t wc;
      int res = mb_wc(&wc, src, se);
      if (res > 0) {
        my_tosort_unicode(uni_plane, &wc, cs->state);
        *dst++ = wc >> 8;
      }
    }
  }

pad:
  if (dst < de && nweights)  // PAD SPACE behavior: pad only remaining weights.
    dst += my_strxfrm_pad_nweights_unicode(dst, de, nweights);

  if ((flags & MY_STRXFRM_PAD_TO_MAXLEN) && dst < de)
    dst += my_strxfrm_pad_unicode(dst, de);
  return dst - dst0;
}

static size_t my_caseup_utf8mb3(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  my_wc_t wc;
  int srcres, dstres;
  char *srcend = src + srclen, *dstend = dst + dstlen, *dst0 = dst;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(src != dst || cs->caseup_multiply == 1);

  while ((src < srcend) &&
         (srcres = my_mb_wc_utf8mb3(&wc, (uchar *)src, (uchar *)srcend)) > 0) {
    my_toupper_utf8mb3(uni_plane, &wc);
    if ((dstres = my_uni_utf8mb3(cs, wc, (uchar *)dst, (uchar *)dstend)) <= 0)
      break;
    src += srcres;
    dst += dstres;
  }
  return (size_t)(dst - dst0);
}

static void my_hash_sort_utf8mb3(const CHARSET_INFO *cs, const uchar *s,
                                 size_t slen, uint64 *n1, uint64 *n2) {
  my_wc_t wc;
  int res;
  const uchar *e = skip_trailing_space(s, slen);
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  uint64 tmp1 = *n1;
  uint64 tmp2 = *n2;

  my_neon_hash_sort_utf8mb3_prefix(cs, &s, e, &tmp1, &tmp2);

  while ((s < e) && (res = my_mb_wc_utf8mb3(&wc, s, e)) > 0) {
    my_tosort_unicode(uni_plane, &wc, cs->state);
    tmp1 ^= (((tmp1 & 63) + tmp2) * (wc & 0xFF)) + (tmp1 << 8);
    tmp2 += 3;
    tmp1 ^= (((tmp1 & 63) + tmp2) * (wc >> 8)) + (tmp1 << 8);
    tmp2 += 3;
    s += res;
  }

  *n1 = tmp1;
  *n2 = tmp2;
}

static size_t my_caseup_str_utf8mb3(const CHARSET_INFO *cs, char *src) {
  my_wc_t wc;
  int srcres, dstres;
  char *dst = src, *dst0 = src;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(cs->caseup_multiply == 1);

  while (*src && (srcres = my_mb_wc_utf8mb3_no_range(&wc, (uchar *)src)) > 0) {
    my_toupper_utf8mb3(uni_plane, &wc);
    if ((dstres = my_uni_utf8mb3_no_range(cs, wc, (uchar *)dst)) <= 0) break;
    src += srcres;
    dst += dstres;
  }
  *dst = '\0';
  return (size_t)(dst - dst0);
}

static size_t my_casedn_utf8mb3(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  my_wc_t wc;
  int srcres, dstres;
  char *srcend = src + srclen, *dstend = dst + dstlen, *dst0 = dst;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(src != dst || cs->casedn_multiply == 1);

  while ((src < srcend) &&
         (srcres = my_mb_wc_utf8mb3(&wc, (uchar *)src, (uchar *)srcend)) > 0) {
    my_tolower_utf8mb3(uni_plane, &wc);
    if ((dstres = my_uni_utf8mb3(cs, wc, (uchar *)dst, (uchar *)dstend)) <= 0)
      break;
    src += srcres;
    dst += dstres;
  }
  return (size_t)(dst - dst0);
}

static size_t my_casedn_str_utf8mb3(const CHARSET_INFO *cs, char *src) {
  my_wc_t wc;
  int srcres, dstres;
  char *dst = src, *dst0 = src;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(cs->casedn_multiply == 1);

  while (*src && (srcres = my_mb_wc_utf8mb3_no_range(&wc, (uchar *)src)) > 0) {
    my_tolower_utf8mb3(uni_plane, &wc);
    if ((dstres = my_uni_utf8mb3_no_range(cs, wc, (uchar *)dst)) <= 0) break;
    src += srcres;
    dst += dstres;
  }

  *dst = '\0';
  return (size_t)(dst - dst0);
}

static int my_strnncoll_utf8mb3(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix) {
  int s_res, t_res;
  my_wc_t s_wc = 0, t_wc = 0;
  const uchar *se = s + slen;
  const uchar *te = t + tlen;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;

  while (s < se && t < te) {
    s_res = my_mb_wc_utf8mb3(&s_wc, s, se);
    t_res = my_mb_wc_utf8mb3(&t_wc, t, te);

    if (s_res <= 0 || t_res <= 0) return bincmp(s, se, t, te);

    my_tosort_unicode(uni_plane, &s_wc, cs->state);
    my_tosort_unicode(uni_plane, &t_wc, cs->state);

    if (s_wc != t_wc) return s_wc > t_wc ? 1 : -1;

    s += s_res;
    t += t_res;
  }
  return (int)(t_is_prefix ? t - te : ((se - s) - (te - t)));
}

static int my_strnncollsp_utf8mb3(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen) {
  int s_res, t_res, res;
  my_wc_t s_wc = 0, t_wc = 0;
  const uchar *se = s + slen, *te = t + tlen;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;

  while (s < se && t < te) {
    if (*s == ' ' && *t == ' ') {
      my_neon_skip_space_pair(&s, &t, se, te);
      continue;
    }

    s_res = my_mb_wc_utf8mb3(&s_wc, s, se);
    t_res = my_mb_wc_utf8mb3(&t_wc, t, te);

    if (s_res <= 0 || t_res <= 0) return bincmp(s, se, t, te);

    my_tosort_unicode(uni_plane, &s_wc, cs->state);
    my_tosort_unicode(uni_plane, &t_wc, cs->state);

    if (s_wc != t_wc) return s_wc > t_wc ? 1 : -1;

    s += s_res;
    t += t_res;
  }

  slen = (size_t)(se - s);
  tlen = (size_t)(te - t);
  res = 0;

  if (slen != tlen) {
    int swap = 1;
    if (slen < tlen) {
      slen = tlen;
      s = t;
      se = te;
      swap = -1;
      res = -res;
    }
    for (; s < se; ++s) {
      if (*s != ' ') return (*s < ' ') ? -swap : swap;
    }
  }
  return res;
}

static int my_strcasecmp_utf8mb3(const CHARSET_INFO *cs, const char *s,
                                 const char *t) {
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  while (s[0] && t[0]) {
    my_wc_t s_wc, t_wc;

    if ((uchar)s[0] < 128) {
      s_wc = plane00[(uchar)s[0]].tolower;
      ++s;
    } else {
      int res = my_mb_wc_utf8mb3(&s_wc, (const uchar *)s, (const uchar *)s + 3);
      if (res <= 0) return strcmp(s, t);
      s += res;
      my_tolower_utf8mb3(uni_plane, &s_wc);
    }

    if ((uchar)t[0] < 128) {
      t_wc = plane00[(uchar)t[0]].tolower;
      ++t;
    } else {
      int res = my_mb_wc_utf8mb3(&t_wc, (const uchar *)t, (const uchar *)t + 3);
      if (res <= 0) return strcmp(s, t);
      t += res;
      my_tolower_utf8mb3(uni_plane, &t_wc);
    }

    if (s_wc != t_wc) return ((int)s_wc) - ((int)t_wc);
  }
  return ((int)(uchar)s[0]) - ((int)(uchar)t[0]);
}

static size_t my_well_formed_len_utf8mb3(const CHARSET_INFO *, const char *b,
                                         const char *e, size_t pos,
                                         int *error) {
  const char *b_start = b;
  *error = 0;
  while (pos) {
    int mb_len;

    if ((mb_len = my_valid_mbcharlen_utf8mb3(pointer_cast<const uchar *>(b),
                                             pointer_cast<const uchar *>(e))) <=
        0) {
      *error = b < e ? 1 : 0;
      break;
    }
    b += mb_len;
    --pos;
  }
  return (size_t)(b - b_start);
}

static size_t my_caseup_utf8mb4(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  my_wc_t wc;
  int srcres, dstres;
  char *srcend = src + srclen, *dstend = dst + dstlen, *dst0 = dst;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(src != dst || cs->caseup_multiply == 1);

  while ((src < srcend) &&
         (srcres = my_mb_wc_utf8mb4(&wc, (uchar *)src, (uchar *)srcend)) > 0) {
    my_toupper_utf8mb4(uni_plane, &wc);
    if ((dstres = my_wc_mb_utf8mb4(cs, wc, (uchar *)dst, (uchar *)dstend)) <= 0)
      break;
    src += srcres;
    dst += dstres;
  }
  return (size_t)(dst - dst0);
}

static void my_hash_sort_utf8mb4(const CHARSET_INFO *cs, const uchar *s,
                                 size_t slen, uint64 *n1, uint64 *n2) {
  my_wc_t wc;
  int res;
  const uchar *e = skip_trailing_space(s, slen);
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  uint64 tmp1 = *n1;
  uint64 tmp2 = *n2;
  uint ch;

  my_neon_hash_sort_utf8mb4_prefix(cs, &s, e, &tmp1, &tmp2);

  while ((res = my_mb_wc_utf8mb4(&wc, s, e)) > 0) {
    my_tosort_unicode(uni_plane, &wc, cs->state);

    ch = (wc & 0xFF);
    tmp1 ^= (((tmp1 & 63) + tmp2) * ch) + (tmp1 << 8);
    tmp2 += 3;

    ch = (wc >> 8) & 0xFF;
    tmp1 ^= (((tmp1 & 63) + tmp2) * ch) + (tmp1 << 8);
    tmp2 += 3;

    if (wc > 0xFFFF) {
      ch = (wc >> 16) & 0xFF;
      tmp1 ^= (((tmp1 & 63) + tmp2) * ch) + (tmp1 << 8);
      tmp2 += 3;
    }
    s += res;
  }

  *n1 = tmp1;
  *n2 = tmp2;
}

static size_t my_caseup_str_utf8mb4(const CHARSET_INFO *cs, char *src) {
  my_wc_t wc;
  int srcres, dstres;
  char *dst = src, *dst0 = src;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(cs->caseup_multiply == 1);

  while (*src &&
         (srcres = my_mb_wc_utf8mb4_no_range(cs, &wc, (uchar *)src)) > 0) {
    my_toupper_utf8mb4(uni_plane, &wc);
    if ((dstres = my_wc_mb_utf8mb4_no_range(cs, wc, (uchar *)dst)) <= 0) break;
    src += srcres;
    dst += dstres;
  }
  *dst = '\0';
  return (size_t)(dst - dst0);
}

static size_t my_casedn_utf8mb4(const CHARSET_INFO *cs, char *src,
                                size_t srclen, char *dst, size_t dstlen) {
  my_wc_t wc;
  int srcres, dstres;
  char *srcend = src + srclen, *dstend = dst + dstlen, *dst0 = dst;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(src != dst || cs->casedn_multiply == 1);

  while ((src < srcend) &&
         (srcres = my_mb_wc_utf8mb4(&wc, (uchar *)src, (uchar *)srcend)) > 0) {
    my_tolower_utf8mb4(uni_plane, &wc);
    if ((dstres = my_wc_mb_utf8mb4(cs, wc, (uchar *)dst, (uchar *)dstend)) <= 0)
      break;
    src += srcres;
    dst += dstres;
  }
  return (size_t)(dst - dst0);
}

static size_t my_casedn_str_utf8mb4(const CHARSET_INFO *cs, char *src) {
  my_wc_t wc;
  int srcres, dstres;
  char *dst = src, *dst0 = src;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  assert(cs->casedn_multiply == 1);

  while (*src &&
         (srcres = my_mb_wc_utf8mb4_no_range(cs, &wc, (uchar *)src)) > 0) {
    my_tolower_utf8mb4(uni_plane, &wc);
    if ((dstres = my_wc_mb_utf8mb4_no_range(cs, wc, (uchar *)dst)) <= 0) break;
    src += srcres;
    dst += dstres;
  }

  *dst = '\0';
  return (size_t)(dst - dst0);
}

static int my_strnncoll_utf8mb4(const CHARSET_INFO *cs, const uchar *s,
                                size_t slen, const uchar *t, size_t tlen,
                                bool t_is_prefix) {
  my_wc_t s_wc = 0;
  my_wc_t t_wc = 0;
  const uchar *se = s + slen;
  const uchar *te = t + tlen;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;

  while (s < se && t < te) {
    int s_res = my_mb_wc_utf8mb4(&s_wc, s, se);
    int t_res = my_mb_wc_utf8mb4(&t_wc, t, te);

    if (s_res <= 0 || t_res <= 0) return bincmp_utf8mb4(s, se, t, te);

    my_tosort_unicode(uni_plane, &s_wc, cs->state);
    my_tosort_unicode(uni_plane, &t_wc, cs->state);

    if (s_wc != t_wc) return s_wc > t_wc ? 1 : -1;

    s += s_res;
    t += t_res;
  }
  return (int)(t_is_prefix ? (t - te) : ((se - s) - (te - t)));
}

static int my_strnncollsp_utf8mb4(const CHARSET_INFO *cs, const uchar *s,
                                  size_t slen, const uchar *t, size_t tlen) {
  int res;
  my_wc_t s_wc = 0;
  my_wc_t t_wc = 0;
  const uchar *se = s + slen, *te = t + tlen;
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;

  while (s < se && t < te) {
    if (*s == ' ' && *t == ' ') {
      my_neon_skip_space_pair(&s, &t, se, te);
      continue;
    }

    int s_res = my_mb_wc_utf8mb4(&s_wc, s, se);
    int t_res = my_mb_wc_utf8mb4(&t_wc, t, te);

    if (s_res <= 0 || t_res <= 0) return bincmp_utf8mb4(s, se, t, te);

    my_tosort_unicode(uni_plane, &s_wc, cs->state);
    my_tosort_unicode(uni_plane, &t_wc, cs->state);

    if (s_wc != t_wc) return s_wc > t_wc ? 1 : -1;

    s += s_res;
    t += t_res;
  }

  slen = (size_t)(se - s);
  tlen = (size_t)(te - t);
  res = 0;

  if (slen != tlen) {
    int swap = 1;
    if (slen < tlen) {
      slen = tlen;
      s = t;
      se = te;
      swap = -1;
      res = -res;
    }
    for (; s < se; ++s) {
      if (*s != ' ') return (*s < ' ') ? -swap : swap;
    }
  }
  return res;
}

static int my_strcasecmp_utf8mb4(const CHARSET_INFO *cs, const char *s,
                                 const char *t) {
  const MY_UNICASE_INFO *uni_plane = cs->caseinfo;
  while (s[0] && t[0]) {
    my_wc_t s_wc, t_wc;

    if ((uchar)s[0] < 128) {
      s_wc = plane00[(uchar)s[0]].tolower;
      ++s;
    } else {
      int res = my_mb_wc_utf8mb4_no_range(cs, &s_wc, (const uchar *)s);
      if (res <= 0) return strcmp(s, t);
      s += res;
      my_tolower_utf8mb4(uni_plane, &s_wc);
    }

    if ((uchar)t[0] < 128) {
      t_wc = plane00[(uchar)t[0]].tolower;
      ++t;
    } else {
      int res = my_mb_wc_utf8mb4_no_range(cs, &t_wc, (const uchar *)t);
      if (res <= 0) return strcmp(s, t);
      t += res;
      my_tolower_utf8mb4(uni_plane, &t_wc);
    }

    if (s_wc != t_wc) return ((int)s_wc) - ((int)t_wc);
  }
  return ((int)(uchar)s[0]) - ((int)(uchar)t[0]);
}

static size_t my_well_formed_len_utf8mb4(const CHARSET_INFO *cs, const char *b,
                                         const char *e, size_t pos,
                                         int *error) {
  const char *b_start = b;
  *error = 0;
  while (pos) {
    int mb_len;

    if ((mb_len = my_valid_mbcharlen_utf8mb4(cs, pointer_cast<const uchar *>(b),
                                             pointer_cast<const uchar *>(e))) <=
        0) {
      *error = b < e ? 1 : 0;
      break;
    }
    b += mb_len;
    --pos;
  }
  return (size_t)(b - b_start);
}
#endif
