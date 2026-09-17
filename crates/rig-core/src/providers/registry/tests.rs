//! The registry's invariant, and what a name resolves to.

use super::*;

/// The host a configuration talks to. A test-only reading: no shipped
/// caller asks a `ProviderConfig` for its base URL — the wire it builds
/// uses it — so there is no accessor for it.
fn host_of(config: &ProviderConfig) -> &str {
    match config {
        ProviderConfig::OpenAi(config) => &config.base_url,
        ProviderConfig::Anthropic(config) => &config.base_url,
        ProviderConfig::Gemini(config) => &config.base_url,
    }
}

/// The invariant every other test here rests on: a provider is a *pair* —
/// vendor and format — and the pair is unique. A vendor may front several
/// doors (z.ai, MiniMax, Moonshot and Xiaomi MiMo each front an
/// OpenAI-shaped endpoint and an Anthropic-shaped one, at different
/// hosts), and that is representable rather than renamed away.
#[test]
fn a_provider_is_a_unique_vendor_and_format_pair() {
    let mut seen: Vec<(&str, Format)> = Vec::new();
    for id in all() {
        let pair = (id.vendor(), id.format());
        assert!(
            !seen.contains(&pair),
            "`{}/{}` is two providers; the pair must identify one",
            pair.0,
            pair.1
        );
        seen.push(pair);
    }
    assert!(seen.len() > 20, "the registry looks empty: {seen:?}");

    // The four two-door vendors: one vendor name, two formats, and nothing
    // in the tree pretends they are different companies.
    for vendor in ["zai", "minimax", "moonshot", "xiaomimimo"] {
        let doors: Vec<Format> = endpoints(vendor).map(|id| id.format()).collect();
        assert_eq!(
            doors,
            vec![Format::OpenAi, Format::Anthropic],
            "`{vendor}` fronts two doors"
        );
        assert_eq!(
            vendors().filter(|name| *name == vendor).count(),
            1,
            "`{vendor}` is one vendor"
        );
    }
    // And a one-door vendor is spelled by its name alone.
    assert_eq!(
        resolve("deepseek").expect("deepseek").spelling(),
        "deepseek"
    );
    assert_eq!(
        resolve("zai/anthropic").expect("zai/anthropic").spelling(),
        "zai/anthropic"
    );
}

/// Every name the registry lists is reachable *by* that name, resolves to a
/// configuration that reports it, and survives the string round trip. A
/// gateway cannot be added without being nameable.
#[test]
fn every_listed_provider_resolves_and_round_trips() {
    for id in all() {
        let name = id.spelling();
        let resolved = resolve(&name)
            .unwrap_or_else(|error| panic!("`{name}` is listed but unresolvable: {error}"));
        assert_eq!(resolved, id, "`{name}` resolves to a different provider");
        let config = id.config();
        assert_eq!(config.provider(), id.vendor());
        assert!(
            config.credential().is_empty(),
            "`{name}` resolved with a credential in it"
        );
        // A host either has a default or names the variable that supplies
        // it: Azure's endpoint is per-resource, so its default is empty and
        // `AZURE_ENDPOINT` is how a caller learns to set one.
        assert!(
            !host_of(&config).is_empty() || config.required_env().count() > 1,
            "`{name}` resolved with no host and names no variable that would give it one"
        );

        let reference: ProviderRef = format!("{name}:some-model")
            .parse()
            .unwrap_or_else(|error| panic!("`{name}` does not parse: {error}"));
        assert_eq!(reference.provider(), id.vendor());
        assert_eq!(reference.model(), "some-model");
        assert_eq!(reference.to_string(), format!("{name}:some-model"));
        assert_eq!(reference.config(), config);
    }
}

/// The two halves of the old collision are two providers, at two hosts, in
/// two formats — which is the whole reason for the rename.
#[test]
fn the_two_zai_endpoints_are_one_vendors_two_doors() {
    let chat = resolve("zai/openai").expect("zai/openai").config();
    let messages = resolve("zai/anthropic").expect("zai/anthropic").config();
    assert!(matches!(chat, ProviderConfig::OpenAi(_)));
    assert!(matches!(messages, ProviderConfig::Anthropic(_)));
    assert_ne!(host_of(&chat), host_of(&messages));
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
    let unqualified = "gpt-5.2".parse::<ProviderRef>().expect_err("no provider");
    assert!(
        matches!(unqualified, UnknownProvider::Unqualified { .. }),
        "{unqualified:?}"
    );
    assert!(
        unqualified.to_string().contains("provider:model"),
        "{unqualified}"
    );

    let unknown = "telepathy:tm-1"
        .parse::<ProviderRef>()
        .expect_err("unknown");
    let message = unknown.to_string();
    assert!(message.contains("telepathy"), "{message}");
    for known in ["openai", "anthropic", "gemini", "zai"] {
        assert!(
            message.contains(known),
            "the refusal must list what this build knows; `{known}` missing from: {message}"
        );
    }

    // A vendor with two doors and no qualifier is asked which, and told
    // what the doors are — never answered with a guess.
    let ambiguous = "zai:glm-4.6".parse::<ProviderRef>().expect_err("two doors");
    let message = ambiguous.to_string();
    assert!(
        matches!(ambiguous, UnknownProvider::Ambiguous { .. }),
        "{message}"
    );
    assert!(message.contains("zai/openai"), "{message}");
    assert!(message.contains("zai/anthropic"), "{message}");

    // …and a door the vendor does not have says so.
    let wrong_door = "deepseek/anthropic:x"
        .parse::<ProviderRef>()
        .expect_err("deepseek speaks one format");
    assert!(
        matches!(wrong_door, UnknownProvider::Format { .. }),
        "{wrong_door:?}"
    );
    assert!(wrong_door.to_string().contains("deepseek"), "{wrong_door}");

    let modelless = "openai:".parse::<ProviderRef>().expect_err("no model");
    assert!(
        matches!(modelless, UnknownProvider::Unqualified { .. }),
        "{modelless:?}"
    );
}

/// A model id is not validated against a catalog: a provider ships models
/// without a rig release, so an unknown one is the provider's 404.
#[test]
fn a_model_id_is_taken_verbatim() {
    let reference: ProviderRef = "openai:a-model-released-tomorrow"
        .parse()
        .expect("an unlisted model id is still a reference");
    assert_eq!(reference.model(), "a-model-released-tomorrow");
    // A colon inside the model id belongs to the model: only the first
    // splits provider from model.
    let versioned: ProviderRef = "openai:ft:gpt-4.1:acme".parse().expect("a fine-tune id");
    assert_eq!(versioned.provider(), "openai");
    assert_eq!(versioned.model(), "ft:gpt-4.1:acme");
    assert_eq!(versioned.to_string(), "openai:ft:gpt-4.1:acme");
}

/// A reference serializes as the string it is, so a scene that overrides
/// nothing stores one line.
#[test]
fn a_reference_serializes_as_its_string() {
    let reference: ProviderRef = "deepseek:deepseek-chat".parse().expect("deepseek");
    let json = serde_json::to_string(&reference).expect("a reference serializes");
    assert_eq!(json, "\"deepseek:deepseek-chat\"");
    assert_eq!(
        serde_json::from_str::<ProviderRef>(&json).expect("a reference reloads"),
        reference
    );
    // And an unknown provider is refused by the reader, before anything is
    // built from it.
    let error = serde_json::from_str::<ProviderRef>("\"telepathy:tm-1\"")
        .expect_err("an unknown provider is not a reference");
    assert!(error.to_string().contains("telepathy"), "{error}");
}

/// What a host needs in the environment, off the data alone.
#[test]
fn a_config_names_the_environment_it_reads() {
    let openai = resolve("openai").expect("openai").config();
    assert_eq!(
        openai.required_env().collect::<Vec<_>>(),
        vec!["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    );
    let anthropic = resolve("anthropic").expect("anthropic").config();
    assert_eq!(
        anthropic.required_env().collect::<Vec<_>>(),
        vec!["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]
    );
    // Gemini has one host and one variable.
    let gemini = resolve("gemini").expect("gemini").config();
    assert_eq!(
        gemini.required_env().collect::<Vec<_>>(),
        vec!["GEMINI_API_KEY"]
    );
    // Every provider names at least its credential, or a host cannot tell
    // what a scene needs.
    for id in all() {
        assert!(
            id.config().required_env().next().is_some(),
            "`{}` names no credential variable",
            id.spelling()
        );
    }
}

/// The persisted form carries the configuration and never the credential,
/// and the provider tag plus the dialect name are what identify it.
#[test]
fn a_config_persists_without_its_credential() {
    let config = resolve("deepseek")
        .expect("deepseek")
        .config()
        .with_credential("sk-secret-value".into());
    let json = serde_json::to_string(&config).expect("a config serializes");
    assert!(json.contains("\"provider\":\"open_ai\""), "{json}");
    assert!(json.contains("\"dialect\":\"deepseek\""), "{json}");
    assert!(!json.contains("sk-secret-value"), "{json}");

    let reloaded: ProviderConfig = serde_json::from_str(&json).expect("a config reloads");
    assert_eq!(reloaded.provider(), "deepseek");
    assert!(reloaded.credential().is_empty());
}

/// A self-hosted OpenAI-compatible endpoint needs no new mechanism: the
/// long form names the host, and it persists.
///
/// This is why there is no inline-dialect form. An inline gateway would
/// only add a *name*, and a name's whole job is to be resolvable by
/// [`by_name`] — which a value living in one scene is not. It would also
/// make "a name resolves to one provider" world-dependent instead of
/// build-dependent, which is the invariant `ProviderRef` rests on.
#[test]
fn a_self_hosted_endpoint_is_the_long_form() {
    let config = resolve("llamacpp")
        .expect("llamacpp")
        .config()
        .with_credential(Secret::default());
    let ProviderConfig::OpenAi(openai) = config else {
        panic!("llama.cpp speaks chat completions");
    };
    let config = ProviderConfig::OpenAi(openai.with_base_url("http://10.0.0.7:8080/v1"));
    assert_eq!(host_of(&config), "http://10.0.0.7:8080/v1");

    // And it round-trips, so a host stores exactly this.
    let json = serde_json::to_string(&config).expect("a config serializes");
    let reloaded: ProviderConfig = serde_json::from_str(&json).expect("a config reloads");
    assert_eq!(host_of(&reloaded), "http://10.0.0.7:8080/v1");
    assert_eq!(reloaded.provider(), "llamacpp");
}

/// The type is the proof: a `ProviderId` cannot be built for a name this
/// build lacks, so reading its configuration is not a lookup that might
/// fail — no `unwrap`, no `unreachable!`, and no snapshot of the defaults
/// taken when the name was read.
#[test]
fn an_id_is_proof_the_name_resolves() {
    assert!(resolve("telepathy").is_err());
    let id = resolve("venice").expect("venice");
    // Identity is the pair, so two lookups of one spelling are one
    // provider and two vendors never are.
    assert_eq!(id, resolve("venice").expect("venice"));
    assert_ne!(id, resolve("openai").expect("openai"));
    assert_eq!(id.vendor(), "venice");
    assert_eq!(id.config().provider(), "venice");
    // And a reference carries the id rather than a configuration, so
    // equality compares what the user wrote.
    let reference: ProviderRef = "venice:venice-uncensored".parse().expect("venice");
    assert_eq!(reference.provider(), id.vendor());
    assert_eq!(
        reference,
        "venice:venice-uncensored"
            .parse::<ProviderRef>()
            .expect("venice"),
        "a reference is its name, not a snapshot"
    );
}

/// The two forms are told apart by the shape of the input, and a bad
/// document is reported by what was wrong with it.
///
/// This is why `ProviderRef`'s reader dispatches on the input's shape
/// instead of trying variants: `#[serde(untagged)]` reads the same
/// documents but, once both variants have failed, can only say "data did
/// not match any variant" — the diagnostic quality the untyped
/// `extra_params` bag used to have.
#[test]
fn the_two_forms_are_read_by_shape_and_report_the_field() {
    // A string is a name.
    let named: ProviderRef =
        serde_json::from_str("\"deepseek:deepseek-chat\"").expect("the short form");
    assert!(matches!(named, ProviderRef::Named { .. }));
    assert_eq!(named.provider(), "deepseek");

    // A map is a configuration, and the model beside it. The document is
    // the one a host would have written: a config's own serialized form.
    let venice = match resolve("venice").expect("venice").config() {
        ProviderConfig::OpenAi(config) => config.with_base_url("http://local"),
        other => panic!("venice speaks chat completions, not {other:?}"),
    };
    let long_form = serde_json::json!({
        "config": ProviderConfig::OpenAi(venice),
        "model": "m",
    })
    .to_string();
    let configured: ProviderRef = serde_json::from_str(&long_form).expect("the long form");
    assert_eq!(configured.provider(), "venice");
    assert_eq!(host_of(&configured.config()), "http://local");
    assert_eq!(configured.model(), "m");

    // A wrong field in the long form is reported as that field…
    let error =
        serde_json::from_str::<ProviderRef>(&long_form.replace(r#""model":"m""#, r#""model":7"#))
            .expect_err("a numeric model is not a model id");
    let message = error.to_string();
    // The reader reports what was actually wrong with the value — here the
    // type it found where a model id belongs — instead of the union's
    // "data did not match any variant", which is all `#[serde(untagged)]`
    // can say once every variant has failed.
    assert!(message.contains("integer"), "{message}");
    assert!(
        !message.contains("did not match any variant"),
        "the union must not swallow the reason: {message}"
    );

    // …an unknown provider tag names the tag…
    let error =
        serde_json::from_str::<ProviderRef>(r#"{"config":{"provider":"telepathy"},"model":"m"}"#)
            .expect_err("an unknown provider is not a config");
    assert!(error.to_string().contains("telepathy"), "{error}");

    // …and a short form naming an unknown provider lists what would work.
    let error = serde_json::from_str::<ProviderRef>("\"telepathy:tm-1\"")
        .expect_err("an unknown provider is not a reference");
    let message = error.to_string();
    assert!(message.contains("telepathy"), "{message}");
    assert!(message.contains("openai"), "{message}");

    // Both forms round-trip through the writer they came from.
    for reference in [named, configured] {
        let json = serde_json::to_string(&reference).expect("a reference serializes");
        assert_eq!(
            serde_json::from_str::<ProviderRef>(&json).expect("a reference reloads"),
            reference
        );
    }
}

/// The four two-door vendors, pinned without credentials.
///
/// `zai`, `minimax`, `moonshot` and `xiaomimimo` each front an OpenAI-shaped
/// endpoint and an Anthropic-shaped one, and their cassette suites are
/// ignore-only wherever the keys are absent — so the fact that the two doors
/// are *different providers at different hosts* is asserted here, off the
/// dialect consts, where no key is needed. Without this, the pair model is
/// proved only where someone has credentials.
#[test]
fn each_two_door_vendor_has_two_endpoints_at_two_hosts() {
    for vendor in ["zai", "minimax", "moonshot", "xiaomimimo"] {
        let openai = resolve(&format!("{vendor}/openai"))
            .unwrap_or_else(|error| panic!("{vendor}/openai: {error}"));
        let anthropic = resolve(&format!("{vendor}/anthropic"))
            .unwrap_or_else(|error| panic!("{vendor}/anthropic: {error}"));

        assert_eq!(openai.vendor(), vendor);
        assert_eq!(anthropic.vendor(), vendor, "one vendor, not two");
        assert_eq!(openai.format(), Format::OpenAi);
        assert_eq!(anthropic.format(), Format::Anthropic);
        assert_ne!(openai, anthropic, "two doors are two providers");

        // Different hosts: the reason naming only the vendor has to refuse.
        let (chat, messages) = (openai.config(), anthropic.config());
        let hosts = (host_of(&chat).to_owned(), host_of(&messages).to_owned());
        assert_ne!(hosts.0, hosts.1, "{vendor}: {hosts:?}");
        assert!(!hosts.0.is_empty() && !hosts.1.is_empty(), "{hosts:?}");

        // …and the same vendor credential opens both.
        assert_eq!(
            chat.required_env().next(),
            messages.required_env().next(),
            "{vendor} issues one key"
        );

        // The unqualified name is refused, listing both doors.
        let error = format!("{vendor}:some-model")
            .parse::<ProviderRef>()
            .expect_err("two doors");
        let message = error.to_string();
        assert!(
            matches!(error, UnknownProvider::Ambiguous { .. }),
            "{message}"
        );
        assert!(message.contains(&format!("{vendor}/openai")), "{message}");
        assert!(
            message.contains(&format!("{vendor}/anthropic")),
            "{message}"
        );
    }

    // And a one-door vendor takes no qualifier: the short spelling is not a
    // special case, it is what all but four providers use.
    assert_eq!(
        all()
            .filter(|id| endpoints(id.vendor()).count() > 1)
            .count(),
        8,
        "four vendors, two doors each"
    );
}
