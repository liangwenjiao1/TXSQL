--disable_query_log
SELECT test_sve_reset();
SELECT test_sve_mode('auto');
eval SET @sve_auto_result := ($sve_expr);
SET @sve_auto_metrics := CONVERT(test_sve_metrics() USING utf8mb4);
SELECT test_sve_reset();
SELECT test_sve_mode('force_scalar');
eval SET @sve_scalar_result := ($sve_expr);
SET @sve_scalar_metrics := CONVERT(test_sve_metrics() USING utf8mb4);
--enable_query_log
let $sve_auto_result= `SELECT IFNULL(CONVERT(@sve_auto_result USING utf8mb4), 'NULL')`;
let $sve_auto_metrics= `SELECT @sve_auto_metrics`;
let $sve_scalar_result= `SELECT IFNULL(CONVERT(@sve_scalar_result USING utf8mb4), 'NULL')`;
let $sve_scalar_metrics= `SELECT @sve_scalar_metrics`;

SELECT @sve_auto_result <=> @sve_scalar_result AS same_result;
SELECT @sve_scalar_metrics LIKE '%fast_calls=0%' AS scalar_no_fast;
