--disable_warnings
DROP TABLE IF EXISTS t_sve_sql;
DROP TABLE IF EXISTS t_sve_mem;
DROP TABLE IF EXISTS t_sve_part;
--enable_warnings

eval CREATE TABLE t_sve_sql ( id INT PRIMARY KEY AUTO_INCREMENT, v VARCHAR(128) CHARACTER SET $sve_sql_charset ) ENGINE=InnoDB;
eval CREATE TABLE t_sve_mem ( v VARCHAR(128) CHARACTER SET $sve_sql_charset COLLATE $sve_sql_mem_collation, KEY(v) ) ENGINE=MEMORY;
eval CREATE TABLE t_sve_part ( v VARCHAR(32) CHARACTER SET $sve_sql_charset COLLATE $sve_sql_part_collation ) PARTITION BY KEY(v) PARTITIONS 4;
