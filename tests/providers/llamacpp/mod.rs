//! The llama.cpp provider suite: one provider, one corpus, one build.
//!
//! Every fixture under `tests/cassettes/llamacpp/` was recorded against
//! `llama-server` **b10499 (commit 6d05498)** built from source, with
//! generation pinned (`--seed 42 --temp 0`) and one model per server
//! configuration. `GET /props` reports the build as `build_info`, and
//! `unmapped_surface::props_states_which_model_and_modalities_produced_this_corpus`
//! asserts it — so the corpus states its own provenance rather than relying on
//! this comment.
//!
//! # Server configurations
//!
//! Recording is batched by **server**, not by test file: each row below is one
//! `llama-server` process, and restarting between individual cells is where
//! the wall-clock goes. The wrapper in `cassette_support.rs` names the
//! environment variable that overrides each port.
//!
//! | Port | Wrapper | Invocation |
//! | --- | --- | --- |
//! | 8080 | `with_llamacpp_cassette` | `-m Qwen3-1.7B-Q4_K_M.gguf --jinja --seed 42 --temp 0 -c 4096` |
//! | 8081 | `with_llamacpp_embeddings_cassette` | `-m Qwen3-Embedding-0.6B-Q8_0.gguf --seed 42 --temp 0 -c 2048 --embeddings --pooling mean` |
//! | 8082 | `with_llamacpp_vision_cassette` | `-m Qwen3-VL-2B-Instruct-Q8_0.gguf --mmproj mmproj-… --jinja --seed 42 --temp 0 -c 4096` |
//! | 8083 | `with_llamacpp_small_context_cassette` | `-m Qwen3-1.7B-Q4_K_M.gguf --jinja --seed 42 --temp 0 -c 512` |
//! | 8084 | `with_llamacpp_no_jinja_cassette` | `-m Qwen3-1.7B-Q4_K_M.gguf --no-jinja --seed 42 --temp 0 -c 4096` |
//! | 8085 | `with_llamacpp_rerank_cassette` | `-m bge-reranker-v2-m3-Q4_K_M.gguf --seed 42 --temp 0 -c 2048 --embeddings --pooling rank --reranking` |
//! | 8086 | `with_llamacpp_pooling_none_cassette` | `-m Qwen3-Embedding-0.6B-Q8_0.gguf --seed 42 --temp 0 -c 2048 --embeddings --pooling none` |
//! | 8087 | `with_llamacpp_causal_embeddings_cassette` | `-m Qwen3-1.7B-Q4_K_M.gguf --seed 42 --temp 0 -c 2048 --embeddings --pooling mean` |
//! | 8088 | `with_llamacpp_competent_cassette` | `-m Qwen3-8B-Q4_K_M.gguf --jinja --seed 42 --temp 0 -c 8192` |
//! | 8089 | `with_llamacpp_api_key_cassette`, `with_llamacpp_missing_api_key_cassette` | `-m Qwen3-1.7B-Q4_K_M.gguf --jinja --seed 42 --temp 0 -c 4096 --api-key llamacpp-local-test-key` |
//! | 8090 | `with_llamacpp_llama_family_cassette` | `-m Llama-3.2-3B-Instruct-Q4_K_M.gguf --jinja --seed 42 --temp 0 -c 4096` |
//! | 8091 | `with_llamacpp_mistral_family_cassette` | `-m Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf --jinja --seed 42 --temp 0 -c 4096` |
//! | 8092 | `with_llamacpp_gemma_family_cassette` | `-m gemma-3-12b-it-Q4_K_M.gguf --jinja --seed 42 --temp 0 -c 4096` |
//! | 8093 | `with_llamacpp_large_vision_cassette` | `-m Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj mmproj-… --jinja --seed 42 --temp 0 -c 8192` |
//!
//! `with_llamacpp_prompt_caching_cassette` and
//! `with_llamacpp_bare_openai_cassette` both use port 8080; they exist as
//! separate names so the cassette-safety registry can attribute their fixtures
//! to their own concern.
//!
//! # The dimensions, and where each lives
//!
//! Each module below carries its own table with per-cell status. This is the
//! index.
//!
//! ## Surface
//!
//! | Surface | Module |
//! | --- | --- |
//! | unary | `agent`, `tools`, `extractor`, `structured_output`, and every matrix's blocking cells |
//! | streaming | `streaming`, `streaming_tools`, `permission_control` |
//! | `raw_completion` | `raw_capture_matrix` |
//! | `raw_stream` | `raw_stream_capture_matrix`, `streaming_tools`'s `raw_*` cells |
//!
//! ## Everything else
//!
//! | Dimension | Module | Server(s) |
//! | --- | --- | --- |
//! | errors (10 classes) | `error_matrix` | 8080, 8083, 8084, 8085, 8086, 8087, 8089 |
//! | sampling — temperature, `max_tokens`, `stop`, `seed` | `sampling_matrix` | 8080, 8083 |
//! | tools — arity, parallel, failure, `tool_choice`, result payloads | `tool_matrix` | 8088 |
//! | structured output — `json_object`, `json_schema`, GBNF, conflicts | `structured_output_matrix` | 8080, 8088 |
//! | content — empty, same-role, unicode, long payloads, history | `content_matrix` | 8080 |
//! | model family — Qwen / Llama / Mistral / Gemma | `model_family_matrix` | 8090, 8091, 8092 |
//! | caching — cold/warm/grown, `cache_prompt`, agent loop | `prompt_caching` | 8080 |
//! | provider-only fields — `timings` | `raw_capture_matrix`, `raw_stream_capture_matrix` | 8080 |
//! | reranking | `rerank_matrix` | 8085 |
//! | multimodal | `multimodal_matrix`, `image_tool_result` | 8080, 8082, 8093 |
//! | unmapped surface — decisions | `unmapped_surface` | 8080, 8082 |
//! | the bare `openai::Client` path | `bare_openai_client` | 8080 |
//!
//! # Model tiers, and the rule for escalating
//!
//! The **smoke tier** (`unsloth/Qwen3-1.7B-GGUF` Q4_K_M) is the default and
//! carries most cells. A cell escalates only when its claim needs a model that
//! can actually do the thing, and says why in its own docs:
//!
//! * **competent** (`unsloth/Qwen3-8B-GGUF` Q4_K_M) — the tool matrix, and the
//!   structured-output cells whose claim is that a model *chose* sensible
//!   values rather than that the server enforced a shape.
//! * **family** (Llama 3.2 3B, Mistral Small 3.2 24B, Gemma 3 12B) — one
//!   blocking and one streaming tool cell each, which is where a chat-template
//!   difference shows.
//! * **vision** (Qwen3-VL-2B Q8_0) and **large vision** (Qwen2.5-VL-7B Q4_K_M)
//!   — chosen per cell by measurement, not by size; see `multimodal_matrix`.
//! * **embedding** (`Qwen/Qwen3-Embedding-0.6B-GGUF` Q8_0) and **reranker**
//!   (`gpustack/bge-reranker-v2-m3-GGUF` Q4_K_M) — a causal LM cannot stand in
//!   for either, and `error_matrix` records what happens when one tries.
//!
//! # Dimensions deliberately not recorded
//!
//! | Dropped | Reason |
//! | --- | --- |
//! | a syntactically malformed request body | rig cannot produce one; every body it emits is serialized from typed values. `error_matrix` records the reachable neighbour (a mistyped field) and states this. |
//! | a *wrong* API key, as distinct from a missing one | llama.cpp compares for equality and answers the same 401 either way; a second cell would record identical bytes. |
//! | Gemma streaming tool calls | its chat template declares `supports_tool_calls: false`, so there is no tool-call stream to observe. |
//! | the full cross-product of family × every dimension | one tool cell and one streaming-tool cell per family is what tests the template claim; recording the whole matrix per family produces a corpus nobody can re-record. |
//! | the generation matrix on the bare `openai::Client` path | that duplication is what produced the 19 colliding fixtures this PR merged away. `bare_openai_client` covers only what differs. |

mod cassette_support;

mod cassette {
    mod agent;
    mod bare_openai_client;
    mod content_matrix;
    mod context;
    mod embeddings;
    mod error_matrix;
    mod extractor;
    mod extractor_usage;
    mod image_tool_result;
    mod loaders;
    mod model_family_matrix;
    mod models;
    mod multi_extract;
    mod multimodal_matrix;
    mod permission_control;
    mod prompt_caching;
    mod raw_capture_matrix;
    mod raw_stream_capture_matrix;
    mod request_hook;
    mod rerank_matrix;
    mod sampling_matrix;
    mod streaming;
    mod streaming_tools;
    mod structured_output;
    mod structured_output_matrix;
    mod tool_matrix;
    mod tools;
    mod typed_prompt_tools;
    mod unmapped_surface;
}
