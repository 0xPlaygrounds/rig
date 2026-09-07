# rig-cassette

Native test support for provider cassette recording, replay, scrubbing and
safety checks. The engine is shared by Rig's provider suites and downstream
application verification. It has no dependency on either agent runtime, the
Rig facade, a consumer registry or a repository's fixture inventory.

Pass a fixture root containing provider directories to `ProviderCassette::start`,
`start_direct_recording`, `cassette_path` and every `recorded_*` reader:

```text
<fixture-root>/anthropic/completion.yaml
<fixture-root>/openai/nested/scenario.yaml
```

`CassetteSpec::new` preserves strict interaction order; `.unordered()` permits
matching any unused interaction. `finish` checks complete consumption and shuts
down the replay server. Invalid fixtures and failed assertions panic, preserving
the original test-support behavior.

`RIG_PROVIDER_TEST_MODE` defaults to `replay`. `record` contacts the configured
upstream and overwrites the selected fixture after scrubbing. Controlled offline
workflows should use `ProviderCassette::start_at` with `CassetteMode::Replay` and
an exact path; that method never derives a mode from the environment. Recording
can target a separate candidate path, then be validated before promotion.

`DirectRecorder`, its request/response types and `DirectRecordingHttpClient`
preserve binary bodies that a text proxy cannot record. SSE and ordinary binary
responses work with no default features. Enable `bedrock` for Smithy event-stream
decoding and scrubbing; its Smithy dependencies are absent otherwise.

The engine retains secret and generated-identifier scrubbing, strict request
matching, and safety validation. Repository-specific source scans and fixture
censuses remain with each caller. Rig's adapter in `tests/common/cassettes.rs`
supplies its own root; a downstream can supply `fixtures/cassettes` instead.

```sh
RIG_PROVIDER_TEST_MODE=replay cargo test --locked -p rig-cassette --no-default-features
RIG_PROVIDER_TEST_MODE=replay cargo test --locked -p rig-cassette --all-features
```
