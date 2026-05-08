# rig-core test utility migration inventory

This inventory records the `mod tests` pass for reusable custom test
implementations. Test-local data shapes that only exist to exercise serde,
provider wire formats, or one specific parser should stay with their tests.

## Migrated to `crate::test_utils`

- `client/model_listing.rs`: `MockModelLister`
- `embeddings/builder.rs`: `MockEmbeddingModel`
- `pipeline/agent_ops.rs` and `pipeline/mod.rs`: `MockPromptModel`,
  `MockVectorStoreIndex`, `Foo`
- `providers/internal/openai_chat_completions_compatible.rs`: final response and
  usage stand-ins now use `MockResponse`
- `tool/mod.rs`: math tools and output-shape tools
- `tool/server.rs`: math tools, dynamic-tool indices, and concurrency probe tools
- `providers/openai/responses_api/streaming.rs`: `example_tool`
- `providers/gemini/completion.rs`: red-pixel image generator tool
- `http_client::mock::MockStreamingClient`: local provider `Default`/`Debug`
  impls folded into the shared mock type

## Intentionally left local

- `json_utils.rs` and `one_or_many.rs`: `Dummy*` serde shapes are tightly
  coupled to field attributes in those tests.
- Provider `mod tests`: request/response structs and enum variants model
  provider-specific wire JSON and are not shared test doubles.
- `providers/deepseek.rs`: `RecordingHttpClient`, captured request metadata, and
  its local response enum are specific to list-model endpoint assertions.
- `http_client::mock::MockStreamingClient`: kept in the HTTP mock module rather
  than `test_utils` because provider streaming tests depend on HTTP behavior
  directly.

## Follow-up rule

When adding a new local implementation of a Rig trait in `rig-core` tests
(`CompletionModel`, `EmbeddingModel`, `Prompt`, `Tool`, `VectorStoreIndex`,
`ModelLister`, or token-usage response stand-ins), prefer extending
`crate::test_utils` unless the type exists only to exercise one provider's wire
format or one serde attribute.
