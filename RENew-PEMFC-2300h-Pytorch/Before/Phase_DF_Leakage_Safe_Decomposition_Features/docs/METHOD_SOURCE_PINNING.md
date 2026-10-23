# Method source pinning

Phase DF uses the PyPI distributions below; no alternative VMD/EMD, EEMD, CEEMDAN, or ICEEMDAN implementation is permitted. A wheel does not carry an upstream source commit, so its installed source-file SHA256, distribution version, import path, signature and license metadata are pinned instead.

## VMD

- `distribution`: `vmdpy`
- `version`: `0.2`
- `license`: `UNKNOWN`
- `home_page`: `http://github.com/vrcarva/vmdpy`
- `source_commit`: `NOT_AVAILABLE_IN_PYPI_WHEEL`
- `source_file`: `D:\ruanjian\python3.11.9\Lib\site-packages\vmdpy\vmdpy.py`
- `source_sha256`: `4d1b294bb7e699e2c0b481a358254aa042f51394996048196b5f9f3953fda9ee`
- `import_path`: `vmdpy.vmdpy.VMD`
- `signature`: `(f, alpha, tau, K, DC, init, tol)`

## EMD

- `distribution`: `EMD-signal`
- `version`: `1.10.0`
- `license`: `Apache-2.0`
- `home_page`: `NOT_DECLARED`
- `source_commit`: `NOT_AVAILABLE_IN_PYPI_WHEEL`
- `source_file`: `D:\ruanjian\python3.11.9\Lib\site-packages\PyEMD\EMD.py`
- `source_sha256`: `4d2e761d8fc7934780e1a5bb6a48d8bb6f5249a3e27211f765d6e07f23a9eade`
- `import_path`: `PyEMD.EMD.EMD`
- `signature`: `(spline_kind: str = 'cubic', nbsym: int = 2, **kwargs)`

`EMD-signal` imports as `PyEMD`; it is not the Earth Mover's Distance package.
