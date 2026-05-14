# Provider Tests

Provider tests are split by whether they can run from committed HTTP cassettes.

- `cassette/` tests run by default in replay mode and do not require provider API keys.
- `live/` tests still hit real provider APIs and remain ignored unless run explicitly.

Cassette files live under `tests/cassettes/<provider>/...`.

## Replay

Replay is the default mode. These commands should run offline:

```bash
cargo test -p rig --all-features --test openai openai::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test anthropic anthropic::cassette -- --nocapture --test-threads=1
cargo test -p rig --all-features --test gemini gemini::cassette -- --nocapture --test-threads=1
```

## Record

Record mode requires the relevant provider API key in the environment and overwrites existing
cassette files.

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test openai openai::cassette -- --nocapture --test-threads=1
```

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test anthropic anthropic::cassette -- --nocapture --test-threads=1
```

```bash
RIG_PROVIDER_TEST_MODE=record \
cargo test -p rig --all-features --test gemini gemini::cassette -- --nocapture --test-threads=1
```

## Safety Checks

After recording or reviewing cassette changes, run the provider safety checks:

```bash
cargo test -p rig --test openai cassette_safety -- --nocapture
cargo test -p rig --test anthropic cassette_safety -- --nocapture
cargo test -p rig --test gemini cassette_safety -- --nocapture
```

Review cassette diffs for:

- no API keys, bearer tokens, cookies, or provider account identifiers;
- expected request paths, methods, and bodies;
- expected provider responses for the scenario;
- no unrelated cassette churn.
