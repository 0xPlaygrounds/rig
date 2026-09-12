# rig-cassette

Native test support for provider cassette recording, replay, scrubbing and
safety checks. The engine is shared by Rig's provider suites and downstream
application verification. It has no dependency on either agent runtime, the
Rig facade, a consumer registry or a repository's fixture inventory.

Pass a fixture root containing provider directories to `ProviderCassette::start`,
`start_via(Transport::Direct, ..)`, `cassette_path` and every `recorded_*` reader:

```text
<fixture-root>/anthropic/completion.yaml
<fixture-root>/openai/nested/scenario.yaml
```

`CassetteSpec::new` preserves strict interaction order; `.unordered()` permits
matching any unused interaction. `finish` checks complete consumption and shuts
down the replay server. A replay session dropped without `finish` panics when
it still holds an unplayed interaction or a refused request, so a test that
returns early cannot pass on a recording it never played to the end. The guard
stays silent while the thread is already panicking, for a fully played session,
and after `finish_after_test_result` returns a test's own error. Invalid
fixtures and failed assertions panic, preserving the original test-support
behavior.

`RIG_PROVIDER_TEST_MODE` defaults to `replay`. `record` contacts the configured
upstream and overwrites the selected fixture after scrubbing; the mode is read from
the environment by every constructor, and a recording reaches the provider through
the proxy (`start`) or directly (`start_via(Transport::Direct, ..)`).

`DirectRecorder`, its request/response types and `DirectRecordingHttpClient`
preserve binary bodies that a text proxy cannot record. SSE and ordinary binary
responses work with no default features. Enable `bedrock` for Smithy event-stream
decoding and scrubbing; its Smithy dependencies are absent otherwise.

The engine retains secret and generated-identifier scrubbing, strict request
matching, and safety validation. Repository-specific source scans and fixture
censuses remain with each caller. Rig's adapter in `tests/common/cassettes.rs`
supplies its own root; a downstream can supply `fixtures/cassettes` instead.
`Retry-After` response headers retain canonical seconds or HTTP dates for
replay diagnostics; malformed values are discarded instead of persisting
arbitrary server text. Generated request IDs remain placeholdered.

```sh
RIG_PROVIDER_TEST_MODE=replay cargo test --locked -p rig-cassette --no-default-features
RIG_PROVIDER_TEST_MODE=replay cargo test --locked -p rig-cassette --all-features
```
