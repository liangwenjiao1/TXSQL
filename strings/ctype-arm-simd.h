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

#ifndef STRINGS_CTYPE_ARM_OPT_INCLUDED
#define STRINGS_CTYPE_ARM_OPT_INCLUDED

#include <string.h>

#include "m_ctype.h"

#ifdef CTYPE_UTF8
#ifdef __cplusplus
extern "C" {
#endif
static inline int bincmp(const uchar *s, const uchar *se, const uchar *t,
                         const uchar *te);
static int my_mb_wc_utf8mb3_no_range(my_wc_t *pwc, const uchar *s);
static int my_uni_utf8mb3(const CHARSET_INFO *cs, my_wc_t wc, uchar *r,
                          uchar *e);
static int my_uni_utf8mb3_no_range(const CHARSET_INFO *cs, my_wc_t wc,
                                   uchar *r);
static inline void my_tolower_utf8mb3(const MY_UNICASE_INFO *uni_plane,
                                      my_wc_t *wc);
static inline void my_toupper_utf8mb3(const MY_UNICASE_INFO *uni_plane,
                                      my_wc_t *wc);
static inline int bincmp_utf8mb4(const uchar *s, const uchar *se,
                                 const uchar *t, const uchar *te);
static int my_mb_wc_utf8mb4_no_range(const CHARSET_INFO *cs, my_wc_t *pwc,
                                     const uchar *s);
static int my_wc_mb_utf8mb4(const CHARSET_INFO *cs, my_wc_t wc, uchar *r,
                            uchar *e);
static int my_wc_mb_utf8mb4_no_range(const CHARSET_INFO *cs, my_wc_t wc,
                                     uchar *r);
static inline void my_tolower_utf8mb4(const MY_UNICASE_INFO *uni_plane,
                                      my_wc_t *wc);
static inline void my_toupper_utf8mb4(const MY_UNICASE_INFO *uni_plane,
                                      my_wc_t *wc);
static int my_valid_mbcharlen_utf8mb4(const CHARSET_INFO *cs, const uchar *s,
                                      const uchar *e);
#ifdef __cplusplus
}
#endif
static size_t my_strxfrm_pad_nweights_unicode(uchar *str, uchar *strend,
                                              size_t nweights);
#endif

#ifdef CTYPE_UCA
struct my_uca_ascii_compare_prefix_result {
  const uchar *s;
  size_t slen;
  const uchar *t;
  size_t tlen;
  bool decided;
  int result;
};

static inline bool my_uca_dual_space_simd_allowed(const CHARSET_INFO *cs) {
  return cs->m_coll_name != nullptr &&
    (strcmp(cs->m_coll_name, "utf8mb3_unicode_ci") == 0 ||
     strcmp(cs->m_coll_name, "utf8mb4_unicode_ci") == 0 ||
     strcmp(cs->m_coll_name, "utf8mb4_0900_ai_ci") == 0);
  }

template <class BaseScanner,
          size_t (*SkipDualSpaceFn)(const uchar **, const uchar **,
                                    const uchar *, const uchar *)>
struct uca_scanner_opt : public BaseScanner {
  using BaseScanner::BaseScanner;

  size_t skip_dual_spaces_with(uca_scanner_opt *other) {
    if (!my_uca_dual_space_simd_allowed(this->cs) || this->weight_lv != 0 ||
        other->weight_lv != 0 || this->num_of_ce_left != 0 ||
        other->num_of_ce_left != 0 || *this->wbeg != 0 || *other->wbeg != 0 ||
        this->sbeg >= this->send || other->sbeg >= other->send ||
        *this->sbeg != ' ' || *other->sbeg != ' ') {
      return 0;
    }

    size_t skipped = SkipDualSpaceFn(&this->sbeg, &other->sbeg, this->send,
                                     other->send);
    if (skipped != 0) {
      this->prev_char = ' ';
      other->prev_char = ' ';
    }
    return skipped;
  }
};

template <class Scanner, int LEVELS_FOR_COMPARE, class Mb_wc>
static int my_strnncoll_uca_opt(const CHARSET_INFO *cs, const Mb_wc mb_wc,
                                const uchar *s, size_t slen, const uchar *t,
                                size_t tlen, bool t_is_prefix) {
  Scanner sscanner(mb_wc, cs, s, slen);
  Scanner tscanner(mb_wc, cs, t, tlen);
  int s_res = 0;
  int t_res = 0;

  for (uint current_lv = 0; current_lv < LEVELS_FOR_COMPARE; ++current_lv) {
    do {
      if (current_lv == 0) {
        sscanner.skip_dual_spaces_with(&tscanner);
      }
      s_res = sscanner.next();
      t_res = tscanner.next();
    } while (s_res == t_res && s_res >= 0 &&
             sscanner.get_weight_level() == current_lv &&
             tscanner.get_weight_level() == current_lv);

    if (sscanner.get_weight_level() == tscanner.get_weight_level()) {
      if (s_res == t_res && s_res >= 0) continue;
      break;
    }

    if (tscanner.get_weight_level() > current_lv) {
      if (t_is_prefix) {
        do {
          s_res = sscanner.next();
        } while (s_res >= 0 && sscanner.get_weight_level() == current_lv);

        if (s_res < 0) break;
        continue;
      } else {
        return 1;
      }
    }

    if (sscanner.get_weight_level() > current_lv) {
      return -1;
    }

    break;
  }

  return (s_res - t_res);
}

template <class Scanner, class Mb_wc>
static int my_strnncollsp_uca_opt(const CHARSET_INFO *cs, Mb_wc mb_wc,
                                  const uchar *s, size_t slen, const uchar *t,
                                  size_t tlen) {
  int s_res, t_res;

  Scanner sscanner(mb_wc, cs, s, slen);
  Scanner tscanner(mb_wc, cs, t, tlen);

  do {
    sscanner.skip_dual_spaces_with(&tscanner);
    s_res = sscanner.next();
    t_res = tscanner.next();
  } while (s_res == t_res && s_res > 0);

  if (s_res > 0 && t_res < 0) {
    t_res = my_space_weight(cs);

    do {
      if (s_res != t_res) return (s_res - t_res);
      s_res = sscanner.next();
    } while (s_res > 0);
    return 0;
  }

  if (s_res < 0 && t_res > 0) {
    s_res = my_space_weight(cs);

    do {
      if (s_res != t_res) return (s_res - t_res);
      t_res = tscanner.next();
    } while (t_res > 0);
    return 0;
  }

  return (s_res - t_res);
}

template <size_t (*SkipDualSpaceFn)(const uchar **, const uchar **,
                                    const uchar *, const uchar *),
          class PrefixFn>
static int my_strnncoll_any_uca_opt(const CHARSET_INFO *cs, PrefixFn prefix_fn,
                                    const uchar *s, size_t slen,
                                    const uchar *t, size_t tlen,
                                    bool t_is_prefix) {
  my_uca_ascii_compare_prefix_result prefix =
      prefix_fn(cs, s, slen, t, tlen, t_is_prefix);
  if (prefix.decided) return prefix.result;
  s = prefix.s;
  slen = prefix.slen;
  t = prefix.t;
  tlen = prefix.tlen;

  if (cs->cset->mb_wc == my_mb_wc_utf8mb4_thunk) {
    return my_strnncoll_uca_opt<
        uca_scanner_opt<uca_scanner_any<Mb_wc_utf8mb4>, SkipDualSpaceFn>, 1>(
        cs, Mb_wc_utf8mb4(), s, slen, t, tlen, t_is_prefix);
  }

  Mb_wc_through_function_pointer mb_wc(cs);
  return my_strnncoll_uca_opt<
      uca_scanner_opt<uca_scanner_any<decltype(mb_wc)>, SkipDualSpaceFn>, 1>(
      cs, mb_wc, s, slen, t, tlen, t_is_prefix);
}

template <size_t (*SkipDualSpaceFn)(const uchar **, const uchar **,
                                    const uchar *, const uchar *)>
static int my_strnncollsp_any_uca_opt(const CHARSET_INFO *cs, const uchar *s,
                                      size_t slen, const uchar *t,
                                      size_t tlen) {
  if (cs->cset->mb_wc == my_mb_wc_utf8mb4_thunk) {
    return my_strnncollsp_uca_opt<
        uca_scanner_opt<uca_scanner_any<Mb_wc_utf8mb4>, SkipDualSpaceFn>>(
        cs, Mb_wc_utf8mb4(), s, slen, t, tlen);
  }

  Mb_wc_through_function_pointer mb_wc(cs);
  return my_strnncollsp_uca_opt<
      uca_scanner_opt<uca_scanner_any<decltype(mb_wc)>, SkipDualSpaceFn>>(
      cs, mb_wc, s, slen, t, tlen);
}

template <size_t (*SkipDualSpaceFn)(const uchar **, const uchar **,
                                    const uchar *, const uchar *),
          class PrefixFn>
static int my_strnncoll_uca_900_opt(const CHARSET_INFO *cs, PrefixFn prefix_fn,
                                    const uchar *s, size_t slen,
                                    const uchar *t, size_t tlen,
                                    bool t_is_prefix) {
  my_uca_ascii_compare_prefix_result prefix =
      prefix_fn(cs, s, slen, t, tlen, t_is_prefix);
  if (prefix.decided) return prefix.result;
  s = prefix.s;
  slen = prefix.slen;
  t = prefix.t;
  tlen = prefix.tlen;

  if (cs->cset->mb_wc == my_mb_wc_utf8mb4_thunk) {
    switch (cs->levels_for_compare) {
      case 1:
        return my_strnncoll_uca_opt<
            uca_scanner_opt<uca_scanner_900<Mb_wc_utf8mb4, 1>,
                            SkipDualSpaceFn>,
            1>(cs, Mb_wc_utf8mb4(), s, slen, t, tlen, t_is_prefix);
      case 2:
        return my_strnncoll_uca_opt<
            uca_scanner_opt<uca_scanner_900<Mb_wc_utf8mb4, 2>,
                            SkipDualSpaceFn>,
            2>(cs, Mb_wc_utf8mb4(), s, slen, t, tlen, t_is_prefix);
      default:
        assert(false);
      case 3:
        return my_strnncoll_uca_opt<
            uca_scanner_opt<uca_scanner_900<Mb_wc_utf8mb4, 3>,
                            SkipDualSpaceFn>,
            3>(cs, Mb_wc_utf8mb4(), s, slen, t, tlen, t_is_prefix);
      case 4:
        return my_strnncoll_uca_opt<
            uca_scanner_opt<uca_scanner_900<Mb_wc_utf8mb4, 4>,
                            SkipDualSpaceFn>,
            4>(cs, Mb_wc_utf8mb4(), s, slen, t, tlen, t_is_prefix);
    }
  }

  Mb_wc_through_function_pointer mb_wc(cs);
  switch (cs->levels_for_compare) {
    case 1:
      return my_strnncoll_uca_opt<
          uca_scanner_opt<uca_scanner_900<decltype(mb_wc), 1>,
                          SkipDualSpaceFn>,
          1>(cs, mb_wc, s, slen, t, tlen, t_is_prefix);
    case 2:
      return my_strnncoll_uca_opt<
          uca_scanner_opt<uca_scanner_900<decltype(mb_wc), 2>,
                          SkipDualSpaceFn>,
          2>(cs, mb_wc, s, slen, t, tlen, t_is_prefix);
    default:
      assert(false);
    case 3:
      return my_strnncoll_uca_opt<
          uca_scanner_opt<uca_scanner_900<decltype(mb_wc), 3>,
                          SkipDualSpaceFn>,
          3>(cs, mb_wc, s, slen, t, tlen, t_is_prefix);
    case 4:
      return my_strnncoll_uca_opt<
          uca_scanner_opt<uca_scanner_900<decltype(mb_wc), 4>,
                          SkipDualSpaceFn>,
          4>(cs, mb_wc, s, slen, t, tlen, t_is_prefix);
  }
}

template <size_t (*SkipDualSpaceFn)(const uchar **, const uchar **,
                                    const uchar *, const uchar *),
          class PrefixFn>
static int my_strnncollsp_uca_900_opt(const CHARSET_INFO *cs,
                                      PrefixFn prefix_fn, const uchar *s,
                                      size_t slen, const uchar *t,
                                      size_t tlen) {
  return my_strnncoll_uca_900_opt<SkipDualSpaceFn>(cs, prefix_fn, s, slen, t,
                                                   tlen, false);
}
#endif

#endif
