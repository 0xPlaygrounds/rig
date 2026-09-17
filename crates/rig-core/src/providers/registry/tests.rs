//! The registry's invariant, and what a name resolves to.

use super::*;

/// The invariant every other test here rests on, and the one a new gateway
/// can break: a dialect name is unique across every format rig speaks. Two
/// dialects sharing a name would make `provider:model` ambiguous — z.ai,
/// MiniMax, Moonshot and Xiaomi MiMo each front an OpenAI-shaped endpoint
/// *and* an Anthropic-shaped one at different hosts, which is why the
/// Messages-format ones carry the `-anthropic` suffix.
#[test]
fn every_provider_name_is_unique() {
    let mut seen: Vec<&str> = Vec::new();
    for name in all() {
        assert!(
            !seen.contains(&name),
            "`{name}` names two providers; a name must resolve to one, or `ModelRef` cannot"
        );
        seen.push(name);
    }
    assert!(seen.len() > 20, "the registry looks empty: {seen:?}");
    // The four that used to collide, both halves present and distinct.
    for name in [
        "zai",
        "zai-anthropic",
        "minimax",
        "minimax-anthropic",
        "moonshot",
        "moonshot-anthropic",
        "xiaomimimo",
        "xiaomimimo-anthropic",
    ] {
        assert!(seen.contains(&name), "`{name}` is not in {seen:?}");
    }
}

/// Every name the registry lists is reachable *by* that name, resolves to a
/// configuration that reports it, and survives the string round trip. A
/// gateway cannot be added without being nameable.
#[test]
fn every_listed_provider_resolves_and_round_trips() {
    for name in all() {
        let config = by_name(name).unwrap_or_else(|| panic!("`{name}` is listed but unresolvable"));
        assert_eq!(config.provider(), name);
        assert!(
            config.credential().is_empty(),
            "`{name}` resolved with a credential in it"
        );
        // A host either has a default or names the variable that supplies
        // it: Azure's endpoint is per-resource, so its default is empty and
        // `AZURE_ENDPOINT` is how a caller learns to set one.
        assert!(
            !config.base_url().is_empty() || config.required_env().count() > 1,
            "`{name}` resolved with no host and names no variable that would give it one"
        );

        let reference: ModelRef = format!("{name}:some-model")
            .parse()
            .unwrap_or_else(|error| panic!("`{name}` does not parse: {error}"));
        assert_eq!(reference.provider(), name);
        assert_eq!(reference.model(), "some-model");
        assert_eq!(reference.to_string(), format!("{name}:some-model"));
        assert_eq!(reference.config(), config);
    }
}

/// The two halves of the old collision are two providers, at two hosts, in
/// two formats — which is the whole reason for the rename.
#[test]
fn the_two_zai_endpoints_are_two_providers() {
    let chat = by_name("zai").expect("zai");
    let messages = by_name("zai-anthropic").expect("zai-anthropic");
    assert!(matches!(chat, ProviderConfig::OpenAi(_)));
    assert!(matches!(messages, ProviderConfig::Anthropic(_)));
    assert_ne!(chat.base_url(), messages.base_url());
    // Same credential, different endpoints: the vendor issues one key.
    assert_eq!(
        chat.required_env().next(),
        messages.required_env().next(),
        "z.ai issues one key for both endpoints"
    );
}

/// A reference has to name a provider, and a wrong name is answered with
/// the list that would have worked.
#[test]
fn an_unnameable_reference_says_what_would_have_worked() {
    let unqualified = "gpt-5.2".parse::<ModelRef>().expect_err("no provider");
    assert!(
        matches!(unqualified, UnknownProvider::Unqualified { .. }),
        "{unqualified:?}"
    );
    assert!(
        unqualified.to_string().contains("provider:model"),
        "{unqualified}"
    );

    let unknown = "telepathy:tm-1".parse::<ModelRef>().expect_err("unknown");
    let message = unknown.to_string();
    assert!(message.contains("telepathy"), "{message}");
    for known in ["openai", "anthropic", "gemini", "zai-anthropic"] {
        assert!(
            message.contains(known),
            "the refusal must list what this build knows; `{known}` missing from: {message}"
        );
    }

    let modelless = "openai:".parse::<ModelRef>().expect_err("no model");
    assert!(
        matches!(modelless, UnknownProvider::NoModel { .. }),
        "{modelless:?}"
    );
}

/// A model id is not validated against a catalog: a provider ships models
/// without a rig release, so an unknown one is the provider's 404.
#[test]
fn a_model_id_is_taken_verbatim() {
    let reference: ModelRef = "openai:a-model-released-tomorrow"
        .parse()
        .expect("an unlisted model id is still a reference");
    assert_eq!(reference.model(), "a-model-released-tomorrow");
    // A colon inside the model id belongs to the model: only the first
    // splits provider from model.
    let versioned: ModelRef = "openai:ft:gpt-4.1:acme".parse().expect("a fine-tune id");
    assert_eq!(versioned.provider(), "openai");
    assert_eq!(versioned.model(), "ft:gpt-4.1:acme");
    assert_eq!(versioned.to_string(), "openai:ft:gpt-4.1:acme");
}

/// A reference serializes as the string it is, so a scene that overrides
/// nothing stores one line.
#[test]
fn a_reference_serializes_as_its_string() {
    let reference: ModelRef = "deepseek:deepseek-chat".parse().expect("deepseek");
    let json = serde_json::to_string(&reference).expect("a reference serializes");
    assert_eq!(json, "\"deepseek:deepseek-chat\"");
    assert_eq!(
        serde_json::from_str::<ModelRef>(&json).expect("a reference reloads"),
        reference
    );
    // And an unknown provider is refused by the reader, before anything is
    // built from it.
    let error = serde_json::from_str::<ModelRef>("\"telepathy:tm-1\"")
        .expect_err("an unknown provider is not a reference");
    assert!(error.to_string().contains("telepathy"), "{error}");
}

/// What a host needs in the environment, off the data alone.
#[test]
fn a_config_names_the_environment_it_reads() {
    let openai = by_name("openai").expect("openai");
    assert_eq!(
        openai.required_env().collect::<Vec<_>>(),
        vec!["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    );
    let anthropic = by_name("anthropic").expect("anthropic");
    assert_eq!(
        anthropic.required_env().collect::<Vec<_>>(),
        vec!["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]
    );
    // Gemini has one host and one variable.
    let gemini = by_name("gemini").expect("gemini");
    assert_eq!(
        gemini.required_env().collect::<Vec<_>>(),
        vec!["GEMINI_API_KEY"]
    );
    // Every provider names at least its credential, or a host cannot tell
    // what a scene needs.
    for name in all() {
        let config = by_name(name).expect("listed");
        assert!(
            config.required_env().next().is_some(),
            "`{name}` names no credential variable"
        );
    }
}

/// The persisted form carries the configuration and never the credential,
/// and the provider tag plus the dialect name are what identify it.
#[test]
fn a_config_persists_without_its_credential() {
    let config = by_name("deepseek")
        .expect("deepseek")
        .with_credential("sk-secret-value".into());
    let json = serde_json::to_string(&config).expect("a config serializes");
    assert!(json.contains("\"provider\":\"open_ai\""), "{json}");
    assert!(json.contains("\"dialect\":\"deepseek\""), "{json}");
    assert!(!json.contains("sk-secret-value"), "{json}");

    let reloaded: ProviderConfig = serde_json::from_str(&json).expect("a config reloads");
    assert_eq!(reloaded.provider(), "deepseek");
    assert!(reloaded.credential().is_empty());
}
