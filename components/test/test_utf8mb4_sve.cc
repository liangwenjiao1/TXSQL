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

#include <mysql/components/component_implementation.h>
#include <mysql/components/services/mysql_string.h>
#include <mysql/components/services/udf_registration.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <list>
#include <string>
#include <vector>

#include "ctype_sve_test.h"
#include "m_ctype.h"

REQUIRES_SERVICE_PLACEHOLDER(udf_registration);
REQUIRES_SERVICE_PLACEHOLDER(mysql_charset);

namespace {

constexpr size_t kMaxStringResult = 65536;
constexpr unsigned char kGuardBefore = 0xA5;
constexpr unsigned char kGuardAfter = 0x5A;
constexpr size_t kGuardSize = 8;

class udf_list {
  using udf_list_t = std::list<std::string>;

 public:
  ~udf_list() { unregister(); }

  bool add_scalar(const char *func_name, enum Item_result return_type,
                  Udf_func_any func, Udf_func_init init_func = nullptr,
                  Udf_func_deinit deinit_func = nullptr) {
    if (!mysql_service_udf_registration->udf_register(
            func_name, return_type, func, init_func, deinit_func)) {
      set.push_back(func_name);
      return false;
    }
    return true;
  }

  bool unregister() {
    udf_list_t delete_set;
    for (const auto &udf : set) {
      int was_present = 0;
      if (!mysql_service_udf_registration->udf_unregister(udf.c_str(),
                                                          &was_present) ||
          !was_present) {
        delete_set.push_back(udf);
      }
    }

    for (const auto &udf : delete_set) set.remove(udf);
    return !set.empty();
  }

 private:
  udf_list_t set;
};

udf_list *g_udfs = nullptr;

const CHARSET_INFO *from_api(CHARSET_INFO_h api) {
  return reinterpret_cast<const CHARSET_INFO *>(api);
}

bool copy_arg(UDF_ARGS *args, unsigned int index, std::string *out) {
  if (index >= args->arg_count || args->args[index] == nullptr) return false;
  out->assign(args->args[index], args->lengths[index]);
  return true;
}

bool parse_integer_arg(UDF_ARGS *args, unsigned int index, long long *out) {
  if (index >= args->arg_count || args->args[index] == nullptr) return false;
  if (args->arg_type[index] == INT_RESULT) {
    *out = *reinterpret_cast<long long *>(args->args[index]);
    return true;
  }

  std::string value;
  if (!copy_arg(args, index, &value)) return false;
  char *end = nullptr;
  *out = std::strtoll(value.c_str(), &end, 10);
  return end != nullptr && *end == '\0';
}

bool build_utf8_collation_alias(const std::string &name, std::string *alias) {
  std::string lower_name(name);
  std::transform(
      lower_name.begin(), lower_name.end(), lower_name.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  if (lower_name.rfind("utf8mb3_", 0) == 0) {
    *alias = "utf8_";
    alias->append(name, 8, std::string::npos);
    return true;
  }

  if (lower_name.rfind("utf8_", 0) == 0) {
    *alias = "utf8mb3_";
    alias->append(name, 5, std::string::npos);
    return true;
  }

  return false;
}

bool get_charset_by_name(UDF_ARGS *args, const CHARSET_INFO **cs) {
  std::string name;
  if (!copy_arg(args, 0, &name)) return false;
  CHARSET_INFO_h handle = mysql_service_mysql_charset->get(name.c_str());
  if (handle == nullptr) {
    std::string alias;
    if (build_utf8_collation_alias(name, &alias)) {
      handle = mysql_service_mysql_charset->get(alias.c_str());
    }
  }
  if (handle == nullptr) return false;
  *cs = from_api(handle);
  return true;
}

int hex_digit(unsigned char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  return -1;
}

bool decode_hex(const std::string &hex, std::vector<unsigned char> *out) {
  if ((hex.size() % 2) != 0) return false;
  out->clear();
  out->reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    int hi = hex_digit(static_cast<unsigned char>(hex[i]));
    int lo = hex_digit(static_cast<unsigned char>(hex[i + 1]));
    if (hi < 0 || lo < 0) return false;
    out->push_back(static_cast<unsigned char>((hi << 4) | lo));
  }
  return true;
}

bool decode_hex_arg(UDF_ARGS *args, unsigned int index,
                    std::vector<unsigned char> *out) {
  std::string hex;
  return copy_arg(args, index, &hex) && decode_hex(hex, out);
}

std::string encode_hex(const unsigned char *data, size_t length) {
  static const char digits[] = "0123456789ABCDEF";
  std::string out;
  out.resize(length * 2);
  for (size_t i = 0; i < length; ++i) {
    out[i * 2] = digits[data[i] >> 4];
    out[i * 2 + 1] = digits[data[i] & 0x0F];
  }
  return out;
}

char *format_result(UDF_INIT *initid, const std::string &value,
                    unsigned long *length, unsigned char *is_null,
                    unsigned char *error) {
  if (value.size() >= kMaxStringResult) {
    *is_null = 1;
    *error = 1;
    return nullptr;
  }

  auto *buffer = reinterpret_cast<char *>(initid->ptr);
  std::memcpy(buffer, value.data(), value.size());
  buffer[value.size()] = '\0';
  *length = static_cast<unsigned long>(value.size());
  *is_null = 0;
  *error = 0;
  return buffer;
}

bool string_udf_init(UDF_INIT *initid, UDF_ARGS *, char *) {
  initid->ptr = new char[kMaxStringResult];
  initid->maybe_null = true;
  initid->max_length = kMaxStringResult - 1;
  return false;
}

void string_udf_deinit(UDF_INIT *initid) {
  delete[] reinterpret_cast<char *>(initid->ptr);
  initid->ptr = nullptr;
}

long long mode_from_string(const std::string &mode) {
  if (mode == "force_scalar") return MY_SVE_TEST_MODE_FORCE_SCALAR;
  return MY_SVE_TEST_MODE_AUTO;
}

std::string metrics_to_string() {
  my_sve_test_metrics metrics = {};
  my_sve_test_get_metrics(&metrics);

  char buffer[512];
  std::snprintf(
      buffer, sizeof(buffer),
      "sve_available=%llu;lane_bytes=%llu;instrumentation_enabled=%llu;"
      "fast_calls=%llu;"
      "scalar_fallback_calls=%llu;ascii_prefix_bytes=%llu;tail_bytes=%llu;"
      "invalid_fallback_calls=%llu;tailoring_disabled_calls=%llu;"
      "contraction_disabled_calls=%llu",
      static_cast<unsigned long long>(metrics.sve_available),
      static_cast<unsigned long long>(metrics.lane_bytes),
      static_cast<unsigned long long>(
          my_sve_test_instrumentation_enabled() ? 1 : 0),
      static_cast<unsigned long long>(metrics.fast_calls),
      static_cast<unsigned long long>(metrics.scalar_fallback_calls),
      static_cast<unsigned long long>(metrics.ascii_prefix_bytes),
      static_cast<unsigned long long>(metrics.tail_bytes),
      static_cast<unsigned long long>(metrics.invalid_fallback_calls),
      static_cast<unsigned long long>(metrics.tailoring_disabled_calls),
      static_cast<unsigned long long>(metrics.contraction_disabled_calls));
  return std::string(buffer);
}

template <typename Fn>
long long with_charset_and_two_hex_args(UDF_ARGS *args, Fn fn, bool *ok) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> lhs;
  std::vector<unsigned char> rhs;
  *ok = get_charset_by_name(args, &cs) && decode_hex_arg(args, 1, &lhs) &&
        decode_hex_arg(args, 2, &rhs);
  if (!*ok) return 0;
  return fn(cs, lhs, rhs);
}

long long udf_test_sve_mode(UDF_INIT *, UDF_ARGS *args, unsigned char *is_null,
                            unsigned char *error) {
  std::string mode;
  if (!copy_arg(args, 0, &mode)) {
    *is_null = 1;
    *error = 1;
    return 0;
  }

  my_sve_test_set_mode(static_cast<int>(mode_from_string(mode)));
  *is_null = 0;
  *error = 0;
  return my_sve_test_get_mode();
}

long long udf_test_sve_reset(UDF_INIT *, UDF_ARGS *, unsigned char *is_null,
                             unsigned char *error) {
  my_sve_test_reset_metrics();
  *is_null = 0;
  *error = 0;
  return 0;
}

char *udf_test_sve_metrics(UDF_INIT *initid, UDF_ARGS *, char *,
                           unsigned long *length, unsigned char *is_null,
                           unsigned char *error) {
  return format_result(initid, metrics_to_string(), length, is_null, error);
}

long long udf_test_sve_strnncoll(UDF_INIT *, UDF_ARGS *args,
                                 unsigned char *is_null,
                                 unsigned char *error) {
  bool ok = false;
  long long prefix = 0;
  if (!parse_integer_arg(args, 3, &prefix)) {
    *is_null = 1;
    *error = 1;
    return 0;
  }
  long long result = with_charset_and_two_hex_args(
      args,
      [prefix](const CHARSET_INFO *cs, const std::vector<unsigned char> &lhs,
               const std::vector<unsigned char> &rhs) {
        return static_cast<long long>(cs->coll->strnncoll(
            cs, lhs.data(), lhs.size(), rhs.data(), rhs.size(), prefix != 0));
      },
      &ok);
  *is_null = ok ? 0 : 1;
  *error = ok ? 0 : 1;
  return result;
}

long long udf_test_sve_strnncollsp(UDF_INIT *, UDF_ARGS *args,
                                   unsigned char *is_null,
                                   unsigned char *error) {
  bool ok = false;
  long long result = with_charset_and_two_hex_args(
      args,
      [](const CHARSET_INFO *cs, const std::vector<unsigned char> &lhs,
         const std::vector<unsigned char> &rhs) {
        return static_cast<long long>(cs->coll->strnncollsp(
            cs, lhs.data(), lhs.size(), rhs.data(), rhs.size()));
      },
      &ok);
  *is_null = ok ? 0 : 1;
  *error = ok ? 0 : 1;
  return result;
}

long long udf_test_sve_strcasecmp(UDF_INIT *, UDF_ARGS *args,
                                  unsigned char *is_null,
                                  unsigned char *error) {
  bool ok = false;
  long long result = with_charset_and_two_hex_args(
      args,
      [](const CHARSET_INFO *cs, const std::vector<unsigned char> &lhs_raw,
         const std::vector<unsigned char> &rhs_raw) {
        std::string lhs(lhs_raw.begin(), lhs_raw.end());
        std::string rhs(rhs_raw.begin(), rhs_raw.end());
        lhs.push_back('\0');
        rhs.push_back('\0');
        return static_cast<long long>(
            cs->coll->strcasecmp(cs, lhs.c_str(), rhs.c_str()));
      },
      &ok);
  *is_null = ok ? 0 : 1;
  *error = ok ? 0 : 1;
  return result;
}

char *udf_test_sve_strnxfrm_hex(UDF_INIT *initid, UDF_ARGS *args, char *,
                                unsigned long *length, unsigned char *is_null,
                                unsigned char *error) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> src;
  long long dstlen_ll = 0;
  long long flags_ll = 0;
  if (!get_charset_by_name(args, &cs) || !decode_hex_arg(args, 1, &src) ||
      !parse_integer_arg(args, 2, &dstlen_ll) ||
      !parse_integer_arg(args, 3, &flags_ll) || dstlen_ll < 0) {
    *is_null = 1;
    *error = 1;
    return nullptr;
  }

  size_t dstlen = static_cast<size_t>(dstlen_ll);
  std::vector<unsigned char> guarded(kGuardSize + dstlen + kGuardSize,
                                     kGuardBefore);
  std::fill(guarded.begin() + kGuardSize + dstlen, guarded.end(), kGuardAfter);
  unsigned char *dst = guarded.data() + kGuardSize;
  if (dstlen != 0) std::memset(dst, 0, dstlen);

  size_t written = cs->coll->strnxfrm(cs, dst, dstlen, static_cast<uint>(dstlen),
                                      src.data(), src.size(),
                                      static_cast<uint>(flags_ll));

  bool guard_ok =
      std::all_of(guarded.begin(), guarded.begin() + kGuardSize,
                  [](unsigned char value) { return value == kGuardBefore; }) &&
      std::all_of(guarded.end() - kGuardSize, guarded.end(),
                  [](unsigned char value) { return value == kGuardAfter; });

  std::string result = "len=" + std::to_string(written) + ";hex=" +
                       encode_hex(dst, std::min(written, dstlen)) +
                       ";guard=" + (guard_ok ? "1" : "0");
  return format_result(initid, result, length, is_null, error);
}

char *udf_test_sve_hash_sort(UDF_INIT *initid, UDF_ARGS *args, char *,
                             unsigned long *length, unsigned char *is_null,
                             unsigned char *error) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> src;
  if (!get_charset_by_name(args, &cs) || !decode_hex_arg(args, 1, &src)) {
    *is_null = 1;
    *error = 1;
    return nullptr;
  }

  uint64 nr1 = 1;
  uint64 nr2 = 4;
  cs->coll->hash_sort(cs, src.data(), src.size(), &nr1, &nr2);
  std::string result = "nr1=" + std::to_string(nr1) + ";nr2=" +
                       std::to_string(nr2);
  return format_result(initid, result, length, is_null, error);
}

long long udf_test_sve_numchars(UDF_INIT *, UDF_ARGS *args,
                                unsigned char *is_null,
                                unsigned char *error) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> src;
  if (!get_charset_by_name(args, &cs) || !decode_hex_arg(args, 1, &src)) {
    *is_null = 1;
    *error = 1;
    return 0;
  }
  *is_null = 0;
  *error = 0;
  return static_cast<long long>(
      cs->cset->numchars(cs, reinterpret_cast<const char *>(src.data()),
                         reinterpret_cast<const char *>(src.data() + src.size())));
}

long long udf_test_sve_charpos(UDF_INIT *, UDF_ARGS *args,
                               unsigned char *is_null,
                               unsigned char *error) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> src;
  long long chars = 0;
  if (!get_charset_by_name(args, &cs) || !decode_hex_arg(args, 1, &src) ||
      !parse_integer_arg(args, 2, &chars) || chars < 0) {
    *is_null = 1;
    *error = 1;
    return 0;
  }
  *is_null = 0;
  *error = 0;
  return static_cast<long long>(cs->cset->charpos(
      cs, reinterpret_cast<const char *>(src.data()),
      reinterpret_cast<const char *>(src.data() + src.size()),
      static_cast<size_t>(chars)));
}

char *udf_test_sve_caseup(UDF_INIT *initid, UDF_ARGS *args, char *,
                          unsigned long *length, unsigned char *is_null,
                          unsigned char *error) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> src;
  if (!get_charset_by_name(args, &cs) || !decode_hex_arg(args, 1, &src)) {
    *is_null = 1;
    *error = 1;
    return nullptr;
  }

  std::vector<char> buffer(src.begin(), src.end());
  buffer.push_back('\0');
  size_t out_len = my_caseup_str(cs, buffer.data());
  bool term_ok = out_len < buffer.size() && buffer[out_len] == '\0';
  std::string result = "len=" + std::to_string(out_len) + ";hex=" +
                       encode_hex(reinterpret_cast<unsigned char *>(buffer.data()),
                                  out_len) +
                       ";term=" + (term_ok ? "1" : "0");
  return format_result(initid, result, length, is_null, error);
}

char *udf_test_sve_casedn(UDF_INIT *initid, UDF_ARGS *args, char *,
                          unsigned long *length, unsigned char *is_null,
                          unsigned char *error) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> src;
  if (!get_charset_by_name(args, &cs) || !decode_hex_arg(args, 1, &src)) {
    *is_null = 1;
    *error = 1;
    return nullptr;
  }

  std::vector<char> buffer(src.begin(), src.end());
  buffer.push_back('\0');
  size_t out_len = my_casedn_str(cs, buffer.data());
  bool term_ok = out_len < buffer.size() && buffer[out_len] == '\0';
  std::string result = "len=" + std::to_string(out_len) + ";hex=" +
                       encode_hex(reinterpret_cast<unsigned char *>(buffer.data()),
                                  out_len) +
                       ";term=" + (term_ok ? "1" : "0");
  return format_result(initid, result, length, is_null, error);
}

long long udf_test_sve_wildcmp(UDF_INIT *, UDF_ARGS *args,
                               unsigned char *is_null,
                               unsigned char *error) {
  bool ok = false;
  long long result = with_charset_and_two_hex_args(
      args,
      [](const CHARSET_INFO *cs, const std::vector<unsigned char> &src,
         const std::vector<unsigned char> &pattern) {
        return static_cast<long long>(my_wildcmp(
            cs, reinterpret_cast<const char *>(src.data()),
            reinterpret_cast<const char *>(src.data() + src.size()),
            reinterpret_cast<const char *>(pattern.data()),
            reinterpret_cast<const char *>(pattern.data() + pattern.size()),
            '\\', '_', '%'));
      },
      &ok);
  *is_null = ok ? 0 : 1;
  *error = ok ? 0 : 1;
  return result;
}

long long udf_test_sve_bench_numchars(UDF_INIT *, UDF_ARGS *args,
                                      unsigned char *is_null,
                                      unsigned char *error) {
  const CHARSET_INFO *cs = nullptr;
  std::vector<unsigned char> src;
  long long loops = 0;
  if (!get_charset_by_name(args, &cs) || !decode_hex_arg(args, 1, &src) ||
      !parse_integer_arg(args, 2, &loops) || loops <= 0) {
    *is_null = 1;
    *error = 1;
    return 0;
  }

  volatile size_t sink = 0;
  const char *begin = reinterpret_cast<const char *>(src.data());
  const char *end = reinterpret_cast<const char *>(src.data() + src.size());
  auto start = std::chrono::steady_clock::now();
  for (long long i = 0; i < loops; ++i) {
    sink += cs->cset->numchars(cs, begin, end);
  }
  auto finish = std::chrono::steady_clock::now();
  (void)sink;

  *is_null = 0;
  *error = 0;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
      .count();
}

}  // namespace

static mysql_service_status_t init() {
  g_udfs = new udf_list();

  bool failed =
      g_udfs->add_scalar("test_sve_mode", INT_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_mode)) ||
      g_udfs->add_scalar("test_sve_reset", INT_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_reset)) ||
      g_udfs->add_scalar("test_sve_metrics", STRING_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_metrics),
                         string_udf_init, string_udf_deinit) ||
      g_udfs->add_scalar("test_sve_strnncoll", INT_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_strnncoll)) ||
      g_udfs->add_scalar(
          "test_sve_strnncollsp", INT_RESULT,
          reinterpret_cast<Udf_func_any>(udf_test_sve_strnncollsp)) ||
      g_udfs->add_scalar(
          "test_sve_strcasecmp", INT_RESULT,
          reinterpret_cast<Udf_func_any>(udf_test_sve_strcasecmp)) ||
      g_udfs->add_scalar(
          "test_sve_strnxfrm_hex", STRING_RESULT,
          reinterpret_cast<Udf_func_any>(udf_test_sve_strnxfrm_hex),
          string_udf_init, string_udf_deinit) ||
      g_udfs->add_scalar("test_sve_hash_sort", STRING_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_hash_sort),
                         string_udf_init, string_udf_deinit) ||
      g_udfs->add_scalar("test_sve_numchars", INT_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_numchars)) ||
      g_udfs->add_scalar("test_sve_charpos", INT_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_charpos)) ||
      g_udfs->add_scalar("test_sve_caseup", STRING_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_caseup),
                         string_udf_init, string_udf_deinit) ||
      g_udfs->add_scalar("test_sve_casedn", STRING_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_casedn),
                         string_udf_init, string_udf_deinit) ||
      g_udfs->add_scalar("test_sve_wildcmp", INT_RESULT,
                         reinterpret_cast<Udf_func_any>(udf_test_sve_wildcmp)) ||
      g_udfs->add_scalar(
          "test_sve_bench_numchars", INT_RESULT,
          reinterpret_cast<Udf_func_any>(udf_test_sve_bench_numchars));

  if (failed) {
    delete g_udfs;
    g_udfs = nullptr;
    return 1;
  }

  return 0;
}

static mysql_service_status_t deinit() {
  if (g_udfs != nullptr) {
    if (g_udfs->unregister()) return 1;
    delete g_udfs;
    g_udfs = nullptr;
  }
  my_sve_test_cleanup();
  return 0;
}

BEGIN_COMPONENT_PROVIDES(test_utf8mb4_sve)
END_COMPONENT_PROVIDES();

BEGIN_COMPONENT_REQUIRES(test_utf8mb4_sve)
REQUIRES_SERVICE(udf_registration), REQUIRES_SERVICE(mysql_charset),
    END_COMPONENT_REQUIRES();

BEGIN_COMPONENT_METADATA(test_utf8mb4_sve)
METADATA("mysql.author", "Oracle Corporation"),
    METADATA("mysql.license", "GPL"), METADATA("test_property", "1"),
    END_COMPONENT_METADATA();

DECLARE_COMPONENT(test_utf8mb4_sve, "mysql:test_utf8mb4_sve")
init, deinit END_DECLARE_COMPONENT();

DECLARE_LIBRARY_COMPONENTS &COMPONENT_REF(test_utf8mb4_sve)
    END_DECLARE_LIBRARY_COMPONENTS
