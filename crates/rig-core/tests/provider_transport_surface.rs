//! The two jobs the deleted `rig-reqwest::providers` alias tree used to do,
//! now guarded in the crate that owns the types.
//!
//! That tree mirrored every transport-generic provider type with `H` defaulted
//! to the bundled transport, *and* hoisted types declared in submodules up to
//! the provider root. `tests/providers_complete.rs` kept it complete. Folding
//! the transport into rig-core removed the mirror; without these tests nothing
//! would notice a provider type that quietly stops defaulting (`openai::Client`
//! needing an explicit `H` again) or one that is only reachable at
//! `huggingface::completion::CompletionModel` instead of the provider root.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};

fn providers_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/providers")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The identifier a `pub type`/`pub struct` declaration names, plus what
/// follows it, for a line already known to start with one of those keywords.
fn declared_name(rest: &str) -> (String, &str) {
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    let after = &rest[name.len()..];
    (name, after)
}

/// Every transport-generic provider type, as `(provider module, type name,
/// file, declares the default)`.
fn transport_generic_types() -> Vec<(String, String, PathBuf, bool)> {
    let root = providers_dir();
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    files.sort();

    let mut out = Vec::new();
    for file in files {
        let rel = file.strip_prefix(&root).expect("under providers/");
        let module = rel
            .components()
            .next()
            .map(|c| {
                c.as_os_str()
                    .to_string_lossy()
                    .trim_end_matches(".rs")
                    .to_string()
            })
            .unwrap_or_default();
        if module == "internal" || module == "mod" {
            continue;
        }
        let source = std::fs::read_to_string(&file).expect("readable source");
        for line in source.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            for prefix in ["pub type ", "pub struct "] {
                let Some(rest) = trimmed.strip_prefix(prefix) else {
                    continue;
                };
                let (name, after) = declared_name(rest);
                if name.is_empty() {
                    continue;
                }
                // Only the transport slot matters: `<H`/`<T` as the FIRST
                // parameter is the convention across the provider tree.
                if !(after.starts_with("<H") || after.starts_with("<T")) {
                    continue;
                }
                let defaulted = after.starts_with("<H = ") || after.starts_with("<T = ");
                // A builder's transport slot deliberately defaults to
                // `Missing` — it is filled by `.http_client(..)` or by the
                // `reqwest` feature's inherent `build`, so it is not part of
                // the bundled-transport surface.
                if after.contains("markers::Missing") {
                    continue;
                }
                out.push((module.clone(), name, file.clone(), defaulted));
            }
        }
    }
    out
}

/// Every transport-generic provider type defaults its transport, so
/// `openai::CompletionModel` keeps meaning `…<reqwest::Client>` with the
/// `reqwest` feature and callers never spell `H` for the bundled path.
#[test]
fn every_transport_generic_provider_type_defaults_its_transport() {
    let missing: Vec<String> = transport_generic_types()
        .into_iter()
        .filter(|(_, _, _, defaulted)| !defaulted)
        .map(|(module, name, file, _)| {
            format!(
                "{module}::{name} ({})",
                file.file_name().unwrap_or_default().to_string_lossy()
            )
        })
        .collect();
    assert!(
        missing.is_empty(),
        "transport-generic provider types without `H = crate::http_client::DefaultHttp`: \
         {missing:?}\nAdd the default so the type is nameable without a transport."
    );
}

/// Types declared in a provider's submodules stay reachable at the provider
/// root: `huggingface::CompletionModel`, not only
/// `huggingface::completion::CompletionModel`.
///
/// The check is that each name RESOLVES, not that `mod.rs` mentions it: a
/// substring scan passes on `pub use responses_api::ResponsesCompletionModel;`
/// when asked about `CompletionModel`, which is exactly how the root-surface
/// regression this file exists to prevent slipped through in the first place.
/// So the list below is compiled — every entry is a real path resolution, and
/// `root_hoists_are_exhaustive` keeps the list honest against the tree.
mod root_paths {
    #![allow(dead_code)]
    use rig_core::providers as p;

    macro_rules! pin {
        ($($alias:ident => $path:ty),* $(,)?) => {
            $( pub type $alias = $path; )*
        };
    }

    pin! {
        AnthropicAnthropicModelLister => p::anthropic::AnthropicModelLister,
        AnthropicClient => p::anthropic::Client,
        AnthropicCompletionModel => p::anthropic::CompletionModel,
        CohereClient => p::cohere::Client,
        CohereCompletionModel => p::cohere::CompletionModel,
        CohereEmbeddingModel => p::cohere::EmbeddingModel,
        CohereImageEmbeddingModel => p::cohere::ImageEmbeddingModel,
        DoublewordClient => p::doubleword::Client,
        DoublewordCompletionModel => p::doubleword::CompletionModel,
        DoublewordEmbeddingModel => p::doubleword::EmbeddingModel,
        GeminiCachedContentClient => p::gemini::CachedContentClient,
        GeminiClient => p::gemini::Client,
        GeminiCompletionModel => p::gemini::CompletionModel,
        GeminiEmbeddingModel => p::gemini::EmbeddingModel,
        GeminiInteractionsCompletionModel => p::gemini::InteractionsCompletionModel,
        GeminiGeminiInteractionsModelLister => p::gemini::GeminiInteractionsModelLister,
        GeminiGeminiModelLister => p::gemini::GeminiModelLister,
        GeminiInteractionsClient => p::gemini::InteractionsClient,
        GeminiTranscriptionModel => p::gemini::TranscriptionModel,
        HuggingfaceClient => p::huggingface::Client,
        HuggingfaceCompletionModel => p::huggingface::CompletionModel,
        HuggingfaceTranscriptionModel => p::huggingface::TranscriptionModel,
        LlamacppClient => p::llamacpp::Client,
        LlamacppCompletionModel => p::llamacpp::CompletionModel,
        LlamacppEmbeddingModel => p::llamacpp::EmbeddingModel,
        LlamacppRerankModel => p::llamacpp::RerankModel,
        MistralClient => p::mistral::Client,
        MistralCompletionModel => p::mistral::CompletionModel,
        MistralEmbeddingModel => p::mistral::EmbeddingModel,
        MistralTranscriptionModel => p::mistral::TranscriptionModel,
        OpenaiClient => p::openai::Client,
        OpenaiCompletionModel => p::openai::CompletionModel,
        OpenaiResponsesCompletionModel => p::openai::ResponsesCompletionModel,
        OpenaiCompletionsClient => p::openai::CompletionsClient,
        OpenaiCompletionsTranscriptionModel => p::openai::CompletionsTranscriptionModel,
        OpenaiEmbeddingModel => p::openai::EmbeddingModel,
        OpenaiTranscriptionModel => p::openai::TranscriptionModel,
        OpenrouterClient => p::openrouter::Client,
        OpenrouterCompletionModel => p::openrouter::CompletionModel,
        OpenrouterEmbeddingModel => p::openrouter::EmbeddingModel,
        OpenrouterTranscriptionModel => p::openrouter::TranscriptionModel,
        TogetherClient => p::together::Client,
        TogetherCompletionModel => p::together::CompletionModel,
        TogetherEmbeddingModel => p::together::EmbeddingModel,
        VeniceClient => p::venice::Client,
        VeniceCompletionModel => p::venice::CompletionModel,
        VeniceEmbeddingModel => p::venice::EmbeddingModel,
        VeniceTranscriptionModel => p::venice::TranscriptionModel,
        XaiClient => p::xai::Client,
        XaiCompletionModel => p::xai::CompletionModel,
    }
    // Only the capability features gate these; rig-core compiles every
    // provider unconditionally today.
    #[cfg(feature = "audio")]
    pin! {
        OpenaiAudioGenerationModel => p::openai::AudioGenerationModel,
        OpenaiCompletionsAudioGenerationModel => p::openai::CompletionsAudioGenerationModel,
        OpenrouterAudioGenerationModel => p::openrouter::AudioGenerationModel,
        VeniceAudioGenerationModel => p::venice::AudioGenerationModel,
        XaiAudioGenerationModel => p::xai::AudioGenerationModel,
    }
    // Only the capability features gate these; rig-core compiles every
    // provider unconditionally today.
    #[cfg(feature = "image")]
    pin! {
        GeminiImageGenerationModel => p::gemini::ImageGenerationModel,
        HuggingfaceImageGenerationModel => p::huggingface::ImageGenerationModel,
        OpenaiCompletionsImageGenerationModel => p::openai::CompletionsImageGenerationModel,
        OpenaiImageGenerationModel => p::openai::ImageGenerationModel,
        VeniceImageGenerationModel => p::venice::ImageGenerationModel,
        XaiImageGenerationModel => p::xai::ImageGenerationModel,
    }
    // Macro-generated types. `transport_generic_types` parses literal
    // declarations, so it cannot see these at all — the `impl_model_lister!`
    // listers or the `Client`/`AnthropicClient` pair the anthropic-compatible
    // macro emits. Dropping `= DefaultHttp` from either macro left both source
    // scans green while ~20 public types stopped resolving without an `H`, so
    // they are pinned by compilation instead.
    pin! {
        MacroDeepSeekModelLister => p::deepseek::DeepSeekModelLister,
        MacroGroqModelLister => p::groq::GroqModelLister,
        MacroLlamacppModelLister => p::llamacpp::LlamacppModelLister,
        MacroMiniMaxModelLister => p::minimax::MiniMaxModelLister,
        MacroMiraModelLister => p::mira::MiraModelLister,
        MacroMistralModelLister => p::mistral::MistralModelLister,
        MacroMoonshotModelLister => p::moonshot::MoonshotModelLister,
        MacroOpenAICompletionsModelLister => p::openai::OpenAICompletionsModelLister,
        MacroOpenAIModelLister => p::openai::OpenAIModelLister,
        MacroOpenRouterModelLister => p::openrouter::OpenRouterModelLister,
        MacroVeniceModelLister => p::venice::VeniceModelLister,
        MacroXiaomiMimoModelLister => p::xiaomimimo::XiaomiMimoModelLister,
        MacroMiniMaxClient => p::minimax::Client,
        MacroMiniMaxAnthropicClient => p::minimax::AnthropicClient,
        MacroMoonshotClient => p::moonshot::Client,
        MacroMoonshotAnthropicClient => p::moonshot::AnthropicClient,
        MacroXiaomiMimoClient => p::xiaomimimo::Client,
        MacroXiaomiMimoAnthropicClient => p::xiaomimimo::AnthropicClient,
        MacroZAiClient => p::zai::Client,
        MacroZAiAnthropicClient => p::zai::AnthropicClient,
    }
}

/// Every `impl_model_lister!` call site is pinned above. The listers are the
/// macro-generated types most likely to grow, and a new one that nobody pins
/// would inherit the same blind spot.
#[test]
fn macro_generated_listers_are_pinned() {
    let root = providers_dir();
    let pinned = include_str!("provider_transport_surface.rs");
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    let mut unpinned = Vec::new();
    for file in files {
        if file.ends_with("internal/model_listing.rs") {
            continue;
        }
        let rel = file.strip_prefix(&root).expect("under providers/");
        let module = rel
            .components()
            .next()
            .map(|c| {
                c.as_os_str()
                    .to_string_lossy()
                    .trim_end_matches(".rs")
                    .to_string()
            })
            .unwrap_or_default();
        let source = std::fs::read_to_string(&file).expect("readable source");
        let mut rest = source.as_str();
        while let Some(at) = rest.find("impl_model_lister!(") {
            rest = &rest[at + "impl_model_lister!(".len()..];
            // The first bare identifier after the doc-comment lines is the name.
            let name = rest
                .lines()
                .map(str::trim)
                .find(|line| {
                    !line.is_empty()
                        && !line.starts_with("///")
                        && !line.starts_with("//")
                        && !line.starts_with("#[")
                })
                .unwrap_or_default()
                .trim_end_matches(',')
                .to_string();
            if name.is_empty() {
                continue;
            }
            let path = format!("p::{module}::{name}");
            if !pinned.contains(&path) {
                unpinned.push(path);
            }
        }
    }
    unpinned.sort();
    unpinned.dedup();
    assert!(
        unpinned.is_empty(),
        "`impl_model_lister!` types with no compiled pin in `mod root_paths`: \
         {unpinned:?}"
    );
}

/// The compiled list above must cover every submodule-declared transport
/// type, or a newly hoisted type could regress unnoticed.
#[test]
fn root_hoists_are_exhaustive() {
    let root = providers_dir();
    let pinned = include_str!("provider_transport_surface.rs");
    let mut unpinned = Vec::new();
    for (module, name, file, _) in transport_generic_types() {
        let rel = file.strip_prefix(&root).expect("under providers/");
        // Skip only the PROVIDER root (`openai/mod.rs`), which declares at the
        // root by definition. A submodule root — `openai/responses_api/mod.rs`,
        // `gemini/interactions_api/mod.rs` — still owes a hoist, and skipping
        // every `mod.rs` is how `ResponsesCompletionModel` went unchecked.
        let is_provider_root =
            rel.components().count() == 2 && file.file_stem().is_some_and(|stem| stem == "mod");
        if rel.components().count() < 2 || is_provider_root {
            continue;
        }
        let path = format!("p::{module}::{name}");
        if !pinned.contains(&path) {
            unpinned.push(path);
        }
    }
    unpinned.sort();
    unpinned.dedup();
    assert!(
        unpinned.is_empty(),
        "transport-generic types declared in a submodule with no compiled \
         root-path pin in `mod root_paths`: {unpinned:?}\nAdd each to the \
         `pin!` list (and re-export it at the provider root if it does not \
         resolve)."
    );
}
