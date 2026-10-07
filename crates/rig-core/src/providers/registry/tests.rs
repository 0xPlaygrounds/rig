//! What the registry promises: qualified writes, shorthand only on input,
//! lossless configurations, and a configuration that reaches the wire.

use super::*;
use crate::providers::openai::Route;
use crate::providers::openai::responses_api::SystemInstructionsPlacement;
use crate::test_utils::{RecordingHttpClient, json_body};

/// A transport that sends nothing: every assertion here is about what is
/// built, never about a reply.
fn transport() -> crate::http_client::DynHttpClient {
    crate::http_client::DynHttpClient::new(RecordingHttpClient::new("{}"))
}

/// Every registered selection has a qualified spelling that parses back to
/// itself — the fixed point a persisted reference relies on.
#[test]
fn every_registered_identity_round_trips_through_its_qualified_spelling() {
    let mut unique = std::collections::HashSet::new();
    let mut seen = 0;
    for id in ProviderId::all() {
        assert!(unique.insert(id), "duplicate selection: {id}");
        assert_eq!(ProviderId::new(id.vendor(), id.format().unwrap()), Some(id));
        let qualified = id.to_string();
        assert_eq!(
            ProviderId::resolve(&qualified),
            Ok(id),
            "`{qualified}` must resolve to itself"
        );
        let reference = ProviderRef::registered(id, "m").unwrap();
        let json = serde_json::to_string(&reference).expect("a reference serializes");
        assert_eq!(json, format!("\"{qualified}:m\""));
        assert_eq!(
            serde_json::from_str::<ProviderRef>(&json).expect("and reads back"),
            reference
        );
        seen += 1;
    }
    assert!(seen > 20, "the registry is not empty: {seen}");
}

#[test]
fn configured_references_never_retain_a_credential() {
    for id in ProviderId::all() {
        let config = id.config("embedded-secret").unwrap();
        assert!(!config.is_unauthenticated());
        let reference = ProviderRef::configured(config.clone(), "model").unwrap();
        assert_eq!(
            reference,
            ProviderRef::configured(config.with_credential("another-secret"), "model").unwrap()
        );
        let mut json = serde_json::to_value(&reference).unwrap();
        let wire = json["config"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap();
        wire["api_key"] = serde_json::json!("deserialized-secret");
        assert_eq!(
            serde_json::from_value::<ProviderRef>(json).unwrap(),
            reference
        );
        let Provider::Configured(recipe) = reference.provider() else {
            panic!("an explicit recipe")
        };
        assert!(
            recipe.is_unauthenticated(),
            "{id}: credential must not enter persistent recipe data"
        );
        assert!(
            !reference.config("host-secret").is_unauthenticated(),
            "the host can still construct a credentialed model"
        );
    }
}

#[test]
fn configured_copilot_hosts_stay_explicit_when_credentials_change() {
    let token = "tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2";
    let id = ProviderId::resolve("copilot").unwrap();
    for (initial_key, resolved_key, expected_host) in [
        ("", token, "https://api.githubcopilot.com"),
        (
            token,
            "host-secret",
            "https://api.individual.githubcopilot.com",
        ),
    ] {
        let reference = ProviderRef::configured(id.config(initial_key).unwrap(), "gpt-4o").unwrap();
        let saved = serde_json::to_string(&reference).unwrap();
        let restored = serde_json::from_str::<ProviderRef>(&saved).unwrap();
        for reference in [reference, restored] {
            let ProviderConfig::OpenAi(config) = reference.config(resolved_key) else {
                panic!("Copilot is an OpenAI-family preset")
            };
            assert_eq!(config.base_url, expected_host);
            assert_eq!(config.api_key.expose(), resolved_key);
        }
    }
    let reference = ProviderRef::registered(id, "gpt-4o").unwrap();
    let ProviderConfig::OpenAi(config) = reference.config(token) else {
        panic!("Copilot is an OpenAI-family preset")
    };
    assert_eq!(config.base_url, "https://api.individual.githubcopilot.com");
}

#[test]
fn model_validation_applies_to_constructors_and_structured_input() {
    let id = ProviderId::resolve("deepseek").unwrap();
    assert!(ProviderRef::registered(id, "").is_err());
    assert!(ProviderRef::configured(id.config("").unwrap(), "").is_err());
    let json = serde_json::json!({"config": id.config("").unwrap(), "model": ""});
    assert!(serde_json::from_value::<ProviderRef>(json).is_err());
    for reference in [
        ProviderRef::registered(id, "namespace/model:tag").unwrap(),
        ProviderRef::configured(id.config("").unwrap(), "namespace/model:tag").unwrap(),
    ] {
        assert_eq!(reference.model(), "namespace/model:tag");
        assert_eq!(
            serde_json::from_value::<ProviderRef>(serde_json::to_value(&reference).unwrap())
                .unwrap(),
            reference
        );
    }
}

#[test]
fn configured_identity_never_mints_an_unregistered_or_custom_preset() {
    let custom =
        openai::wire::Dialect::gateway("private", "https://private.invalid", "PRIVATE_KEY");
    let config = ProviderConfig::OpenAi(openai::wire::OpenAIConfig::with_key(&custom, ""));
    assert_eq!(config.id(), None);
    let reference = ProviderRef::configured(config, "model").unwrap();
    assert_eq!(reference.id(), None);
    assert_eq!(reference.to_string(), "private/openai:model");
    assert!(
        serde_json::to_value(&reference).is_err(),
        "an unreloadable dialect must not enter persisted data"
    );

    let modified = openai::wire::Dialect {
        base_url: "https://modified.invalid",
        ..openai::wire::OPENAI
    };
    let config = ProviderConfig::OpenAi(openai::wire::OpenAIConfig::with_key(&modified, ""));
    assert!(
        serde_json::to_value(&config).is_err(),
        "custom dialect payload must not silently disappear"
    );
    let id = config.id().unwrap();
    assert_eq!(id, ProviderId::resolve("openai").unwrap());
    let ProviderConfig::OpenAi(preset) = id.config("").unwrap() else {
        panic!("OpenAI preset")
    };
    assert_eq!(preset.base_url, openai::wire::OPENAI.base_url);
}

#[test]
fn anthropic_custom_dialects_are_executable_but_not_lossily_persisted() {
    for dialect in [
        anthropic::wire::compatible("private", "https://private.invalid", "PRIVATE_KEY", None),
        anthropic::wire::Dialect {
            base_url: "https://modified.invalid",
            ..anthropic::wire::ANTHROPIC
        },
    ] {
        let config =
            ProviderConfig::Anthropic(anthropic::wire::AnthropicConfig::with_key(&dialect, ""));
        assert!(serde_json::to_value(&config).is_err());
        let _handler = config.completion_handler("custom", "model", transport());
    }
}

/// A vendor with two protocol families keeps both qualified references
/// unambiguous — the one upgrade guarantee a qualified write buys.
#[test]
fn a_second_format_for_a_vendor_leaves_qualified_references_unambiguous() {
    let dual: Vec<&str> = ["zai", "minimax", "moonshot", "xiaomimimo"].into();
    for vendor in dual {
        let selections: Vec<ProviderId> = ProviderId::vendor_selections(vendor).collect();
        assert_eq!(
            selections.len(),
            2,
            "`{vendor}` is registered for two families"
        );
        // Bare is refused, both qualified spellings still resolve to exactly
        // the selection they name.
        assert!(matches!(
            ProviderId::resolve(vendor),
            Err(SelectionError::Ambiguous { .. })
        ));
        for id in selections {
            assert_eq!(ProviderId::resolve(&id.to_string()), Ok(id));
        }
    }
    // The single-format vendors keep their shorthand.
    assert_eq!(
        ProviderId::resolve("deepseek"),
        ProviderId::new("deepseek", Format::OpenAi).ok_or(SelectionError::Unknown {
            vendor: "deepseek".to_owned()
        })
    );
}

/// An ambiguous shorthand refuses and hands back alternatives spelled the way
/// the resolver accepts them.
#[test]
fn ambiguous_shorthand_offers_resolvable_alternatives() {
    let error = ProviderId::resolve("zai").expect_err("`zai` is ambiguous");
    let SelectionError::Ambiguous { alternatives, .. } = &error else {
        panic!("expected an ambiguity: {error}");
    };
    assert_eq!(alternatives, &["zai/openai", "zai/anthropic"]);
    for alternative in alternatives {
        let id = ProviderId::resolve(alternative).expect("an alternative resolves");
        assert_eq!(&id.to_string(), alternative, "and is already canonical");
    }
    assert!(
        error.to_string().contains("zai/openai"),
        "the message names them: {error}"
    );
}

/// Malformed input is an error, never a silent fallback to some default.
#[test]
fn malformed_input_refuses() {
    use SelectionError::*;
    assert!(matches!(ProviderId::resolve(""), Err(Malformed { .. })));
    assert!(matches!(
        ProviderId::resolve("/openai"),
        Err(Malformed { .. })
    ));
    assert!(matches!(
        ProviderId::resolve("openai/openai/extra"),
        Err(Malformed { .. })
    ));
    assert!(matches!(
        ProviderId::resolve("openai/"),
        Err(UnknownFormat { .. })
    ));
    assert!(matches!(
        ProviderId::resolve("openai/messages"),
        Err(UnknownFormat { .. })
    ));
    assert!(matches!(
        ProviderId::resolve("nosuchvendor"),
        Err(Unknown { .. })
    ));
    let unregistered =
        ProviderId::resolve("openai/anthropic").expect_err("openai speaks no Messages endpoint");
    assert!(matches!(unregistered, Unregistered { .. }));
    assert!(
        unregistered.to_string().contains("openai/openai"),
        "and says what it does speak: {unregistered}"
    );

    use RefError::*;
    assert!(matches!(
        ProviderRef::parse("deepseek"),
        Err(NoModel { .. })
    ));
    assert!(matches!(
        ProviderRef::parse("deepseek:"),
        Err(NoModel { .. })
    ));
    assert!(matches!(ProviderRef::parse(":m"), Err(Selection(_))));
    assert!(matches!(ProviderRef::parse("zai:m"), Err(Selection(_))));
}

/// The parser splits at the first colon, so a model identifier may carry `:`
/// and `/`.
#[test]
fn the_model_identifier_keeps_its_separators() {
    let tagged =
        ProviderRef::parse("llamacpp/openai:qwen3:4b").expect("a tag is part of the model");
    assert_eq!(tagged.model(), "qwen3:4b");
    assert_eq!(tagged.to_string(), "llamacpp/openai:qwen3:4b");

    let pathed =
        ProviderRef::parse("together/openai:meta-llama/Llama-3-70b").expect("so is a namespace");
    assert_eq!(pathed.model(), "meta-llama/Llama-3-70b");
    assert_eq!(
        serde_json::from_str::<ProviderRef>(&serde_json::to_string(&pathed).unwrap()).unwrap(),
        pathed
    );
}

/// Shorthand is resolved on input and never written back: the stored form is
/// always qualified.
#[test]
fn shorthand_is_canonicalized_on_write() {
    let reference: ProviderRef =
        serde_json::from_str("\"deepseek:deepseek-chat\"").expect("shorthand reads");
    assert_eq!(
        serde_json::to_string(&reference).unwrap(),
        "\"deepseek/openai:deepseek-chat\""
    );
    // …and the qualified form is a fixed point from there on.
    let again: ProviderRef =
        serde_json::from_str(&serde_json::to_string(&reference).unwrap()).unwrap();
    assert_eq!(again, reference);
    assert_eq!(
        serde_json::to_string(&again).unwrap(),
        serde_json::to_string(&reference).unwrap()
    );
}

/// A configuration is written as an object, and two configurations that
/// differ in host, route or options stay different across a round trip.
#[test]
fn configurations_round_trip_and_stay_distinct() {
    let openai = ProviderId::resolve("openai/openai").unwrap();
    let plain = openai.config("sk-secret").unwrap();
    let elsewhere = match plain.clone() {
        ProviderConfig::OpenAi(provider) => {
            ProviderConfig::OpenAi(provider.with_base_url("https://gateway.invalid/v1"))
        }
        other => panic!("openai/openai is an OpenAI configuration: {other:?}"),
    };
    let on_chat = match plain.clone() {
        ProviderConfig::OpenAi(provider) => {
            ProviderConfig::OpenAi(provider.with_route(Route::Chat))
        }
        other => panic!("{other:?}"),
    };
    let as_messages = match plain.clone() {
        ProviderConfig::OpenAi(provider) => {
            ProviderConfig::OpenAi(provider.with_system_instructions_as_messages())
        }
        other => panic!("{other:?}"),
    };

    let mut written = Vec::new();
    for config in [&plain, &elsewhere, &on_chat, &as_messages] {
        let reference = ProviderRef::configured(config.clone(), "gpt-4.1-mini").unwrap();
        let json = serde_json::to_string(&reference).expect("a configuration serializes");
        assert!(
            json.starts_with(r#"{"config":{"openai":"#),
            "as an object, family first: {json}"
        );
        assert!(
            !json.contains("sk-secret"),
            "and never with the credential: {json}"
        );
        let read: ProviderRef = serde_json::from_str(&json).expect("and reads back");
        // Only the credential is lost, and it is lost by contract.
        assert_eq!(
            read,
            ProviderRef::configured(config.clone().with_credential(""), "gpt-4.1-mini").unwrap()
        );
        written.push(json);
    }
    for (index, json) in written.iter().enumerate() {
        for (other, sibling) in written.iter().enumerate() {
            assert_eq!(
                index == other,
                json == sibling,
                "configurations differing in host, route or placement must not collapse:\n{json}\n{sibling}"
            );
        }
    }
}

/// A configuration is an object, never the diagnostic string — and the
/// diagnostic string is still the canonical identity.
#[test]
fn a_configuration_is_never_written_as_shorthand() {
    let venice = ProviderId::resolve("venice/openai").unwrap();
    let config = match venice.config("vk-secret").unwrap() {
        ProviderConfig::OpenAi(provider) => {
            ProviderConfig::OpenAi(provider.with_base_url("https://private.invalid/api/v1"))
        }
        other => panic!("{other:?}"),
    };
    let reference = ProviderRef::configured(config, "venice-uncensored").unwrap();
    assert_eq!(
        reference.to_string(),
        "venice/openai:venice-uncensored",
        "the label is canonical identity"
    );
    let json = serde_json::to_string(&reference).unwrap();
    assert!(
        json.contains("https://private.invalid/api/v1"),
        "the object keeps the host: {json}"
    );
    assert_ne!(json, "\"venice/openai:venice-uncensored\"");
}

/// An invalid tag, dialect or option is an error that names what was wrong.
#[test]
fn bad_configuration_data_reports_what_was_wrong() {
    let bad_family = serde_json::from_str::<ProviderRef>(
        r#"{"config":{"messages":{"api_key":"x","base_url":"b","version":"v","betas":[],"dialect":"anthropic"}},"model":"m"}"#,
    )
    .expect_err("`messages` is not a family");
    assert!(bad_family.to_string().contains("messages"), "{bad_family}");

    let bad_dialect = serde_json::from_str::<ProviderRef>(
        r#"{"config":{"openai":{"api_key":"x","base_url":"b","dialect":"nosuch","auth":"Bearer"}},"model":"m"}"#,
    )
    .expect_err("`nosuch` is no dialect");
    assert!(bad_dialect.to_string().contains("nosuch"), "{bad_dialect}");

    let misspelled = serde_json::from_str::<ProviderRef>(
        r#"{"config":{"openai":{"api_key":"x","base_url":"b","dialect":"openai","auth":"Bearer","rout":"chat"}},"model":"m"}"#,
    )
    .expect_err("a misspelled option is not silently dropped");
    assert!(misspelled.to_string().contains("rout"), "{misspelled}");

    let bad_field = serde_json::from_str::<ProviderRef>(
        r#"{"configuration":{"gemini":{"api_key":"x","base_url":"b"}},"model":"m"}"#,
    )
    .expect_err("the object has two fields and no others");
    assert!(
        bad_field.to_string().contains("configuration"),
        "{bad_field}"
    );

    let wrong_shape = serde_json::from_str::<ProviderRef>("42")
        .expect_err("a reference is a string or an object");
    assert!(
        wrong_shape.to_string().contains("provider reference"),
        "{wrong_shape}"
    );
}

/// Credential guidance is a hint for an optional-auth selection and a
/// requirement for the rest.
#[test]
fn optional_auth_selections_are_not_described_as_requiring_a_credential() {
    let local = ProviderId::resolve("llamacpp/openai").unwrap();
    assert_eq!(local.api_key_env(), Some("LLAMACPP_API_KEY"));
    assert!(
        !local.requires_credential(),
        "a local llama-server authenticates optionally"
    );
    for qualified in ["openai/openai", "anthropic/anthropic", "gcp.gemini/gemini"] {
        let id = ProviderId::resolve(qualified).unwrap();
        assert!(id.requires_credential(), "{qualified} needs a credential");
        assert!(id.api_key_env().is_some_and(|name| !name.is_empty()));
    }
}

/// A gateway the old provider enum could not name is expressible as a
/// configuration and builds a handler.
#[test]
fn a_gateway_absent_from_the_old_vocabulary_is_materializable() {
    let json = r#"{"config":{"openai":{"api_key":"[redacted]","base_url":"https://api.venice.ai/api/v1","dialect":"venice","auth":"Bearer"}},"model":"venice-uncensored"}"#;
    let reference: ProviderRef = serde_json::from_str(json).expect("Venice reads from data");
    assert_eq!(reference.to_string(), "venice/openai:venice-uncensored");
    let config = reference.config("vk-test");
    assert!(!config.is_unauthenticated(), "the host rehydrated it");
    let handler = config.completion_handler("default", reference.model(), transport());
    let descriptor = handler.descriptor();
    assert!(
        format!("{descriptor:?}").contains("default"),
        "the label reaches the descriptor: {descriptor:?}"
    );
}

/// The registered and configured paths that mean the same thing build the
/// same handler descriptor.
#[test]
fn equivalent_reference_and_configuration_describe_themselves_alike() {
    let transport = || transport();
    let registered = ProviderRef::parse("deepseek:deepseek-chat").unwrap();
    let configured = ProviderRef::configured(
        ProviderId::resolve("deepseek/openai")
            .unwrap()
            .config("")
            .unwrap(),
        "deepseek-chat",
    )
    .unwrap();
    assert_eq!(registered.id(), configured.id());
    let from_reference = registered
        .config("k")
        .completion_handler("default", registered.model(), transport())
        .descriptor();
    let from_config = configured
        .config("k")
        .completion_handler("default", configured.model(), transport())
        .descriptor();
    assert_eq!(from_reference, from_config);
}

/// The instruction placement a configuration carries reaches the encoded
/// request body, in both directions.
#[test]
fn the_configured_instruction_placement_reaches_the_request_body() {
    use crate::completion::CompletionRequest;
    use crate::wire::{Mode, Wire};

    let request = || CompletionRequest::new("hello").preamble("be brief");
    let encode = |config: ProviderConfig| {
        let ProviderConfig::OpenAi(provider) = config else {
            panic!("an OpenAI configuration");
        };
        let wire = provider.responses("gpt-4.1-mini");
        let encoded = wire
            .encode(request(), Mode::Unary)
            .expect("the request encodes");
        json_body(&encoded.request)
    };

    let openai = ProviderId::resolve("openai/openai").unwrap();
    // Absent: the dialect's own placement — top-level `instructions`.
    let default = encode(openai.config("k").unwrap());
    assert_eq!(
        default.get("instructions").and_then(|value| value.as_str()),
        Some("be brief"),
        "{default}"
    );

    // Overridden in data: the preamble travels as a `system` message in
    // `input` and no top-level `instructions` is sent.
    let json = r#"{"config":{"openai":{"api_key":"[redacted]","base_url":"https://api.openai.com/v1","dialect":"openai","auth":"Bearer","system_instructions":"input_system_messages"}},"model":"gpt-4.1-mini"}"#;
    let reference: ProviderRef = serde_json::from_str(json).expect("the override reads from data");
    let rehydrated = reference.config("k");
    let ProviderConfig::OpenAi(config) = &rehydrated else {
        panic!("an OpenAI configuration");
    };
    assert_eq!(
        config.system_instructions,
        Some(SystemInstructionsPlacement::InputSystemMessages)
    );
    let overridden = encode(reference.config("k"));
    assert!(
        overridden.get("instructions").is_none(),
        "no top-level instructions: {overridden}"
    );
    let input = overridden
        .get("input")
        .and_then(|value| value.as_array())
        .expect("an input array");
    assert!(
        input.iter().any(|item| {
            item.get("role").and_then(|role| role.as_str()) == Some("system")
                && format!("{item}").contains("be brief")
        }),
        "the preamble is a system message instead: {overridden}"
    );
}

/// No serialized reference carries a credential, whichever door it came
/// through.
#[test]
fn credentials_never_enter_a_serialized_reference() {
    const SENTINEL: &str = "sk-do-not-leak";
    for id in ProviderId::all() {
        for reference in [
            ProviderRef::registered(id, "m").unwrap(),
            ProviderRef::configured(id.config(SENTINEL).unwrap(), "m").unwrap(),
        ] {
            let json = serde_json::to_string(&reference).expect("serializes");
            assert!(!json.contains(SENTINEL), "{json}");
            let read: ProviderRef = serde_json::from_str(&json).expect("reads back");
            assert!(
                match read.provider() {
                    Provider::Registered(_) => true,
                    Provider::Configured(config) => config.is_unauthenticated(),
                },
                "a reloaded configuration holds no credential: {json}"
            );
        }
    }
}

/// `Format`'s serialized name and the tag `ProviderConfig` writes are two
/// spellings of one thing; they must not drift.
#[test]
fn the_family_name_is_the_configuration_tag() {
    for id in ProviderId::all() {
        let config = id.config("").unwrap();
        let json = serde_json::to_value(&config).expect("a configuration serializes");
        let tag = json
            .as_object()
            .expect("externally tagged")
            .keys()
            .next()
            .expect("one tag")
            .clone();
        assert_eq!(
            tag,
            id.format().unwrap().as_str(),
            "{id}: the family name and the configuration tag agree"
        );
        assert_eq!(
            serde_json::to_string(&id.format().unwrap()).unwrap(),
            format!("\"{tag}\"")
        );
        assert_eq!(
            serde_json::from_str::<Format>(&format!("\"{tag}\"")).unwrap(),
            id.format().unwrap()
        );
    }
}

/// A configured URL survives serialization, for every family.
#[test]
fn a_configured_base_url_round_trips_through_serialization() {
    const HOST: &str = "https://gateway.invalid/v2";
    for selection in ["openai/openai", "anthropic/anthropic", "gcp.gemini/gemini"] {
        let config = ProviderId::resolve(selection)
            .unwrap()
            .config("sk-secret")
            .unwrap()
            .with_base_url(HOST);
        let json = serde_json::to_string(&config).expect("a configuration serializes");
        assert!(json.contains(HOST), "{selection}: {json}");
        let read: ProviderConfig = serde_json::from_str(&json).expect("and reads back");
        assert_eq!(read.base_url(), HOST, "{selection}");
        assert_eq!(read, config.with_credential(""), "{selection}");
    }
}

/// Both doors of a dual-format vendor resolve to the configuration and host
/// their dialect documents — no credential, no network.
#[test]
fn both_doors_of_a_dual_format_vendor_reach_their_own_endpoint() {
    let chat = ProviderId::resolve("zai/openai").unwrap();
    let messages = ProviderId::resolve("zai/anthropic").unwrap();
    assert_eq!(chat.vendor(), messages.vendor());
    assert_ne!(chat, messages);

    match chat.config("").unwrap() {
        ProviderConfig::OpenAi(provider) => {
            assert_eq!(provider.base_url, "https://api.z.ai/api/paas/v4");
            assert_eq!(provider.dialect.name, "zai");
        }
        other => panic!("zai/openai is an OpenAI configuration: {other:?}"),
    }
    match messages.config("").unwrap() {
        ProviderConfig::Anthropic(provider) => {
            assert_eq!(provider.base_url, "https://api.z.ai/api/anthropic");
            assert_eq!(provider.dialect.name, "zai");
        }
        other => panic!("zai/anthropic is a Messages configuration: {other:?}"),
    }
    assert_eq!(chat.api_key_env(), messages.api_key_env());
}

/// Ids of different vendors are different ids even when they share a
/// format. A hashed collection compares two such ids only when they share a
/// bucket, so this is the check that always runs.
#[test]
fn ids_of_different_vendors_differ_in_one_format() {
    let openai = ProviderId::resolve("openai/openai").unwrap();
    let venice = ProviderId::resolve("venice/openai").unwrap();
    assert_eq!(openai.format(), venice.format());
    assert_ne!(openai, venice);
}

/// `connect` takes a catalog entry or either reference grammar to the model
/// `ProviderRef` builds, and names the companion crate of a catalog-only
/// provider instead of building one.
#[test]
fn connect_selects_by_spec_or_reference() {
    let reference = |selector: ModelSelector<'_>| selector.provider_ref().map(|r| r.to_string());
    assert_eq!(
        reference("anthropic/claude-opus-5-5".into()),
        Ok("anthropic/anthropic:claude-opus-5-5".to_owned())
    );
    assert_eq!(
        reference("openrouter/anthropic/claude-sonnet-4.5".into()),
        Ok("openrouter/openai:anthropic/claude-sonnet-4.5".to_owned())
    );
    assert_eq!(
        reference("ollama/qwen3:4b".into()),
        Ok("ollama/openai:qwen3:4b".to_owned()),
        "a `:` inside the model id"
    );
    assert_eq!(
        reference("zai/anthropic:glm-5".into()),
        Ok("zai/anthropic:glm-5".to_owned()),
        "the selection grammar picks the format"
    );
    assert_eq!(
        reference("zai/glm-5".into()),
        Ok("zai/openai:glm-5".to_owned()),
        "`vendor/model` takes the vendor's first registered selection"
    );
    let spec = crate::catalog::Catalog::builtin()
        .resolve("openai/gpt-5.5")
        .expect("listed");
    assert_eq!(
        reference(spec.into()),
        Ok("openai/openai:gpt-5.5".to_owned())
    );

    let bedrock = crate::catalog::Catalog::builtin()
        .resolve("aws_bedrock/us.anthropic.claude-sonnet-5")
        .expect("listed");
    for selector in [
        ModelSelector::from(bedrock),
        "aws_bedrock/us.anthropic.claude-sonnet-5".into(),
    ] {
        let error = selector.provider_ref().expect_err("catalog-only");
        assert!(matches!(error, ConnectError::CatalogOnly { .. }), "{error}");
        assert!(error.to_string().contains("rig-bedrock"), "{error}");
    }
    assert!(matches!(
        reference("no-model".into()),
        Err(ConnectError::Malformed { .. })
    ));
    assert!(matches!(
        reference("nosuchvendor/model".into()),
        Err(ConnectError::Reference(RefError::Selection(
            SelectionError::Unknown { .. }
        )))
    ));
    assert!(
        ProviderRef::registered(bedrock.provider, "m").is_err(),
        "a reference never names a catalog-only provider"
    );

    let model = connect_with("openai/gpt-5.5", "sk-test", RecordingHttpClient::new("{}"))
        .expect("connects");
    assert_eq!(model.id(), Some("gpt-5.5"));
}

/// A catalog-only id has no credential variable of the registry's, says
/// whether its companion crate needs a credential, and reads back from its
/// serialized vendor name; a registered id serializes as its qualified
/// selection.
#[test]
fn a_catalog_only_id_describes_itself_and_round_trips() {
    for (vendor, credential) in [("aws_bedrock", true), ("candle", false)] {
        let id = ProviderId::catalog(vendor).expect("known");
        assert!(!id.is_registered(), "{vendor}");
        assert_eq!(id.format(), None, "{vendor}");
        assert_eq!(id.api_key_env(), None, "{vendor}");
        assert!(id.config("sk-test").is_none(), "{vendor}");
        assert_eq!(id.requires_credential(), credential, "{vendor}");
        let json = serde_json::to_string(&id).expect("serializes");
        assert_eq!(json, format!("\"{vendor}\""));
        assert_eq!(serde_json::from_str::<ProviderId>(&json).unwrap(), id);
    }
    let openai = ProviderId::catalog("openai").expect("registered");
    assert!(openai.is_registered());
    assert_eq!(serde_json::to_string(&openai).unwrap(), "\"openai/openai\"");
    assert!(ProviderId::catalog("nosuchvendor").is_none());
    assert!(serde_json::from_str::<ProviderId>("\"nosuchvendor\"").is_err());
}

/// References on two registered vendors differ, as their ids do.
#[test]
fn registered_references_of_different_vendors_differ() {
    let openai = ProviderRef::parse("openai/openai:gpt-5.5").unwrap();
    let venice = ProviderRef::parse("venice/openai:gpt-5.5").unwrap();
    assert_ne!(openai, venice);
    assert_eq!(openai, "openai/openai:gpt-5.5".parse().unwrap());
}

/// The object form names each field once.
#[test]
fn a_repeated_reference_field_is_refused() {
    let reference = ProviderRef::configured(
        ProviderId::resolve("openai/openai")
            .unwrap()
            .config("")
            .unwrap(),
        "gpt-5.5",
    )
    .unwrap();
    let json = serde_json::to_value(&reference).unwrap();
    let config = json["config"].to_string();
    for (repeated, text) in [
        (
            "config",
            format!(r#"{{"config":{config},"config":{config},"model":"gpt-5.5"}}"#),
        ),
        (
            "model",
            format!(r#"{{"config":{config},"model":"gpt-5.5","model":"gpt-5.5"}}"#),
        ),
    ] {
        let error = serde_json::from_str::<ProviderRef>(&text).expect_err("duplicate");
        assert!(
            error
                .to_string()
                .contains(&format!("duplicate field `{repeated}`")),
            "{error}"
        );
    }
}

/// `connect` and `connect_with` take every selector spelling: a catalog
/// entry, a `&str`, a `&String` and a `ModelRef`; a catalog-only provider is
/// refused before any model is built.
#[test]
fn connect_takes_every_selector_spelling() {
    let spec = crate::catalog::Catalog::builtin()
        .resolve("openai/gpt-5.5")
        .expect("listed");
    let owned = "openai/gpt-5.5".to_owned();
    let label = ModelRef::new("openai/gpt-5.5");
    for selector in [
        ModelSelector::from(spec),
        ModelSelector::from("openai/gpt-5.5"),
        ModelSelector::from(&owned),
        ModelSelector::from(&label),
    ] {
        let model = connect(selector, "sk-test").expect("connects");
        assert_eq!(model.id(), Some("gpt-5.5"));
        let model =
            connect_with(selector, "sk-test", RecordingHttpClient::new("{}")).expect("connects");
        assert_eq!(model.id(), Some("gpt-5.5"));
    }
    let error = connect("candle/qwen3", "").expect_err("catalog-only");
    assert_eq!(
        error,
        ConnectError::CatalogOnly {
            vendor: "candle".to_owned(),
            served_by: "the `rig-candle` crate",
        }
    );
    assert!(matches!(
        connect_with("no-model", "", RecordingHttpClient::new("{}")),
        Err(ConnectError::Malformed { .. })
    ));
}
