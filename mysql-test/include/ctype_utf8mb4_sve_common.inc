INSTALL COMPONENT "file://component_test_utf8mb4_sve";

SET NAMES utf8mb4;

SET @sve_metrics := test_sve_metrics();
SET @sve_available := @sve_metrics REGEXP '(^|;)sve_available=1(;|$)';
SET @sve_instrumented := @sve_metrics REGEXP '(^|;)instrumentation_enabled=1(;|$)';
SET @sve_metric_checks := @sve_available AND @sve_instrumented;
SET @sve_lane := CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(@sve_metrics, 'lane_bytes=', -1), ';', 1) AS UNSIGNED);
SET @sve_len_unit := IF(@sve_lane > 0, @sve_lane, 16);
let $sve_available= `SELECT @sve_available`;
let $sve_instrumented= `SELECT @sve_instrumented`;
let $sve_metric_checks= `SELECT @sve_metric_checks`;

if (!$sve_available)
{
  UNINSTALL COMPONENT "file://component_test_utf8mb4_sve";
  skip Requires SVE runtime support.;
}

if (!$sve_instrumented)
{
  UNINSTALL COMPONENT "file://component_test_utf8mb4_sve";
  skip Requires SVE test instrumentation.;
}

--disable_query_log
SELECT test_sve_reset();
SELECT test_sve_mode('auto');
--enable_query_log
