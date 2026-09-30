# N1 allowed-prefix fingerprint contract

V2 may read only:

```python
pandas.read_csv(N1_TABLE, usecols=SAFE_N1_COLUMNS, nrows=3007)
```

The `ALLOWED_LOGICAL_PREFIX_FINGERPRINT` is SHA256 of a canonical UTF-8 CSV
serialization of precisely that in-memory DataFrame: fixed `SAFE_N1_COLUMNS`
order, index `0..3006`, newline `\n`, `na_rep="<NA>"`, and `%.17g` floating
format.  It is not a full-source-file SHA256.

No V2 function may call `sha256_file(N1_TABLE)`, `N1_TABLE.read_bytes()`, or
otherwise access data after the permitted logical prefix.  A regression test
modifies a synthetic row 3,008 and verifies that the fingerprint is unchanged.
