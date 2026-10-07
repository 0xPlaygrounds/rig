//! The provider keys are stable API from 0.44: every
//! `ProviderExtension::PROVIDER` value, which keys a request's provider
//! options and gates `CompletionResponse::extras`, and every vendor key the
//! model catalog files models under, with the models.dev keys it renames. A
//! caller writes them in code, in stored requests and in catalog override
//! files, so a rename must fail here first.

use std::collections::BTreeSet;
use std::path::Path;

use rig::catalog::Catalog;
use rig::completion::ProviderExtension;
use rig::providers;

/// `P`'s key.
fn key<P: ProviderExtension>() -> &'static str {
    P::PROVIDER
}

/// Every extension marker's key, as its crate's source directory, the
/// marker's key and the pinned key.
fn extension_keys() -> Vec<(&'static str, &'static str, &'static str)> {
    let core = "crates/rig-core/src";
    #[allow(unused_mut)]
    let mut keys = vec![
        (
            core,
            key::<providers::anthropic::extension::Anthropic>(),
            "anthropic",
        ),
        (
            core,
            key::<providers::azure::extension::Azure>(),
            "azure.openai",
        ),
        (
            core,
            key::<providers::chatgpt::extension::ChatGpt>(),
            "chatgpt",
        ),
        (
            core,
            key::<providers::cohere::extension::Cohere>(),
            "cohere",
        ),
        (
            core,
            key::<providers::copilot::extension::Copilot>(),
            "copilot",
        ),
        (
            core,
            key::<providers::deepseek::extension::DeepSeek>(),
            "deepseek",
        ),
        (
            core,
            key::<providers::gemini::extension::Gemini>(),
            "gcp.gemini",
        ),
        (core, key::<providers::groq::extension::Groq>(), "groq"),
        (
            core,
            key::<providers::llamacpp::extension::LlamaCpp>(),
            "llamacpp",
        ),
        (
            core,
            key::<providers::minimax::extension::MiniMax>(),
            "minimax",
        ),
        (
            core,
            key::<providers::mistral::extension::Mistral>(),
            "mistral",
        ),
        (
            core,
            key::<providers::moonshot::extension::Moonshot>(),
            "moonshot",
        ),
        (
            core,
            key::<providers::ollama::extension::Ollama>(),
            "ollama",
        ),
        (
            core,
            key::<providers::openai::extension::OpenAi>(),
            "openai",
        ),
        (
            core,
            key::<providers::openrouter::extension::OpenRouter>(),
            "openrouter",
        ),
        (
            core,
            key::<providers::perplexity::extension::Perplexity>(),
            "perplexity",
        ),
        (
            core,
            key::<providers::together::extension::Together>(),
            "together",
        ),
        (
            core,
            key::<providers::venice::extension::Venice>(),
            "venice",
        ),
        (core, key::<providers::xai::extension::Xai>(), "xai"),
        (
            core,
            key::<providers::xiaomimimo::extension::XiaomiMimo>(),
            "xiaomimimo",
        ),
        (core, key::<providers::zai::extension::Zai>(), "zai"),
    ];
    #[cfg(feature = "bedrock")]
    keys.push((
        "crates/rig-bedrock/src",
        key::<rig::bedrock::extension::Bedrock>(),
        "aws_bedrock",
    ));
    #[cfg(feature = "candle")]
    keys.push((
        "crates/rig-candle/src",
        key::<rig::candle::extension::Candle>(),
        "candle",
    ));
    #[cfg(feature = "gemini-grpc")]
    keys.push((
        "crates/rig-gemini-grpc/src",
        key::<rig::gemini_grpc::extension::GeminiGrpc>(),
        "gemini-grpc",
    ));
    #[cfg(feature = "vertexai")]
    keys.push((
        "crates/rig-vertexai/src",
        key::<rig::vertexai::extension::Vertex>(),
        "vertexai",
    ));
    keys
}

/// The vendor keys of the built-in catalog.
const CATALOG_VENDORS: [&str; 28] = [
    "anthropic",
    "aws_bedrock",
    "azure.openai",
    "chatgpt",
    "cohere",
    "copilot",
    "deepseek",
    "doubleword",
    "gcp.gemini",
    "gemini-grpc",
    "groq",
    "huggingface",
    "hyperbolic",
    "llamacpp",
    "minimax",
    "mistral",
    "moonshot",
    "ollama",
    "openai",
    "openrouter",
    "perplexity",
    "together",
    "venice",
    "vertexai",
    "voyageai",
    "xai",
    "xiaomimimo",
    "zai",
];

/// The models.dev provider keys the catalog reads under another vendor key.
const MODELS_DEV_RENAMES: [(&str, &str); 9] = [
    ("azure", "azure.openai"),
    ("google", "gcp.gemini"),
    ("google-vertex", "vertexai"),
    ("amazon-bedrock", "aws_bedrock"),
    ("togetherai", "together"),
    ("moonshotai", "moonshot"),
    ("xiaomi", "xiaomimimo"),
    ("ollama-cloud", "ollama"),
    ("github-copilot", "copilot"),
];

/// The number of `impl ProviderExtension for` items in the non-test
/// sources under `dir`.
fn markers_in(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .map(|path| {
            let name = path.file_name().and_then(|name| name.to_str());
            if path.is_dir() {
                return match name {
                    Some("tests") => 0,
                    _ => markers_in(&path),
                };
            }
            if name == Some("tests.rs")
                || path.extension().and_then(|ext| ext.to_str()) != Some("rs")
            {
                return 0;
            }
            std::fs::read_to_string(&path)
                .map(|text| text.matches("impl ProviderExtension for ").count())
                .unwrap_or(0)
        })
        .sum()
}

#[test]
fn every_provider_extension_key_is_pinned() {
    let keys = extension_keys();
    for (_, actual, pinned) in &keys {
        assert_eq!(actual, pinned, "a provider key is stable API");
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates: BTreeSet<&str> = keys.iter().map(|(dir, _, _)| *dir).collect();
    for dir in crates {
        let pinned = keys.iter().filter(|(owner, _, _)| *owner == dir).count();
        assert_eq!(
            markers_in(&root.join(dir)),
            pinned,
            "every ProviderExtension in {dir} has its key pinned here"
        );
    }
}

#[test]
fn every_catalog_vendor_key_is_pinned() {
    let vendors: BTreeSet<&str> = Catalog::builtin()
        .iter()
        .map(|spec| spec.provider.vendor())
        .collect();
    assert_eq!(vendors, BTreeSet::from(CATALOG_VENDORS));
}

#[test]
fn every_models_dev_rename_is_pinned() {
    for (models_dev, vendor) in MODELS_DEV_RENAMES {
        let json = format!(r#"{{"{models_dev}": {{"models": {{"m": {{}}}}}}}}"#);
        let catalog = Catalog::from_json(&json)
            .unwrap_or_else(|error| panic!("{models_dev} should read: {error}"));
        let read: Vec<&str> = catalog.iter().map(|spec| spec.provider.vendor()).collect();
        assert_eq!(read, [vendor], "models.dev `{models_dev}`");
    }
}
