//! The llama.cpp provider suite: one provider, one corpus, one build.
//!
//! Every fixture under `crates/rig-cassette/fixtures/cassettes/llamacpp/` was recorded against
//! `llama-server` b10964 (commit b29c606e2) built from source, with generation pinned
//! (`--seed 42 --temp 0`) and one model per server configuration. Recording is batched by
//! server, and `cassette_support.rs` documents each configuration, its port and the variable
//! that overrides it. `unmapped_surface::props_states_which_model_and_modalities_produced_this_corpus`
//! asserts the build from `GET /props`, so the corpus states its own provenance.
//!
//! The matrices each own one dimension: `error_matrix`, `embedding_matrix`, `sampling_matrix`,
//! `tool_matrix`, `structured_output_matrix`, `content_matrix`, `model_family_matrix`,
//! `prompt_caching`, `response_shape_matrix`, `truncation_matrix`, `turn_termination_matrix`,
//! `rerank_matrix`, `multimodal_matrix` and `unmapped_surface`, which also covers the plain
//! `OPENAI`-dialect path. The smoke cells in `agent`, `embeddings`, `image_tool_result`,
//! `models`, `multi_extract`, `permission_control`, `streaming`, `streaming_tools`,
//! `structured_output` and `typed_prompt_tools` are indexed here instead of tabulated per
//! module, and `matrix_index` enforces both lists. Most cells run on the smallest model; a
//! cell escalates to a more capable model only when its claim needs one, and says why in its
//! own docs.

mod cassette_support;

mod cassette {
    mod agent;
    mod content_matrix;
    mod embedding_matrix;
    mod embeddings;
    mod error_matrix;
    mod image_tool_result;
    mod matrix_index;
    mod model_family_matrix;
    mod models;
    mod multi_extract;
    mod multimodal_matrix;
    mod permission_control;
    mod prompt_caching;
    mod rerank_matrix;
    mod response_shape_matrix;
    mod sampling_matrix;
    mod streaming;
    mod streaming_tools;
    mod structured_output;
    mod structured_output_matrix;
    mod tool_matrix;
    mod truncation_matrix;
    mod turn_termination_matrix;
    mod typed_prompt_tools;
    mod unmapped_surface;
}
