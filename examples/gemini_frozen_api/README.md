# gemini_frozen_api

The frozen programs of `docs/design/0002-gemini-first-class.md` (P1 to P6 and
P8), each built unchanged as a binary. They call the live Gemini API and need
`GEMINI_API_KEY`:

```console
cargo run -p gemini_frozen_api --bin p1
```

P7 is a test and lives in `crates/rig-cassette/tests/providers/gemini/`.
