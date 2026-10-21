# OMP schema differential fixtures

Captured from oh-my-pi commit `d5cd750e41fede277f9eb933455da92a0738915e` using the real `streamAnthropic` request pipeline. No model inference or credentials were used. Its injected fetch redirects every request to a loopback SSE dummy server. The API key is a non-secret placeholder.

`input.tools[].input_schema` is the same raw JSON object supplied as OMP's `parameters`. Both implementations therefore receive identical inputs, including legacy schema preprocessing. Expected output comes directly from the captured request, not a reimplementation of OMP normalization. Only the transport wrapper fields `eager_input_streaming` and `cache_control` are removed. Schema fields, array ordering, descriptions, and strict selection are not normalized. Tests compare JSON values (object member order is not significant); constraint descriptions retain the source ordering verbatim.

The 18 cases cover the strict name allowlist and opt-out, optional-property budgets at 23/24/25, union budgets at 15/16/17, tool-count budgets at 19/20/21, rollback after a failed candidate, nested and open objects, combinators and nullable schemas, mixed enums and const, local and percent-encoded references, numeric property names and unsupported constraints with existing descriptions, legacy tuples and empty schemas, and incompatible keywords.

`generate.ts` is the capture program with portable source and output paths. It requires Bun and the pinned OMP checkout with its dependencies installed. It writes next to the script. To recapture deliberately from this plugin directory:

```sh
OMP_SOURCE=/path/to/oh-my-pi bun tests/fixtures/omp-schema/generate.ts
```

Offline regression tests require neither Bun nor the OMP checkout:

```sh
cargo nextest run --locked --profile local -p claude-cloak schema::differential
```

Do not refresh expected output merely to hide implementation differences.
