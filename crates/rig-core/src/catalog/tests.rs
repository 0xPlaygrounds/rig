//! What the catalog promises: the checked-in data loads whole, lookups and
//! references find the model they name, overrides win field by field, and
//! `validate` refuses what the model does not take.

use super::*;
use crate::completion::{CacheRetention, Effort, GenerationOptions, Reasoning};

fn options(reasoning: impl Into<Reasoning>) -> GenerationOptions {
    GenerationOptions::default().reasoning(reasoning)
}

/// A models.dev-style file read by the lenient reader.
fn models_dev(json: &str) -> Result<Catalog, CatalogError> {
    Catalog::from_models_dev(json).map(|(catalog, _)| catalog)
}

/// An override file read by the strict reader, every model in it new.
fn strict(json: &str) -> Result<Catalog, OverrideErrors> {
    Catalog::from_overrides(json, &Catalog::default())
}

fn spec<'a>(catalog: &'a Catalog, vendor: &str, model: &str) -> &'a ModelSpec {
    let provider = ProviderId::catalog(vendor).expect("a known vendor");
    catalog
        .get_exact(provider, model)
        .unwrap_or_else(|| panic!("{vendor}/{model} is listed"))
}

const SAMPLE: &str = r#"{
  "anthropic": {"name": "Anthropic", "models": {
    "claude-x": {
      "name": "Claude X",
      "reasoning": true,
      "reasoning_options": [
        {"type": "effort", "values": ["low", "medium", "high", "max"]},
        {"type": "budget_tokens", "min": 1024}
      ],
      "tool_call": true,
      "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
      "limit": {"context": 200000, "output": 64000},
      "cost": {"input": 3, "output": 15, "cache_read": 0.3},
      "rig": {"reasoning_default": "high", "cache": ["none", "short", "long"], "adaptive_thinking": true}
    }
  }},
  "google": {"models": {
    "gemini-x": {"reasoning": true, "reasoning_options": [{"type": "toggle"}, {"type": "budget_tokens", "min": 0, "max": 24576}], "status": "deprecated"}
  }},
  "groq": {"models": {
    "qwen-x": {"reasoning": true, "reasoning_options": [{"type": "effort", "values": ["none", "default", "low", "high"]}]}
  }},
  "amazon-bedrock": {"models": {"us.anthropic.claude-x-v1:0": {"name": "Claude X"}}},
  "a-provider-rig-does-not-serve": {"models": {"m": {"name": "M"}}}
}"#;

#[test]
fn the_checked_in_catalog_loads_whole() {
    let (catalog, skipped) = Catalog::from_models_dev(BUILTIN).expect("the checked-in data parses");
    assert_eq!(skipped, [], "every row reads");
    if let Err(errors) = Catalog::from_overrides(BUILTIN, &catalog) {
        panic!("the checked-in data names only keys rig reads: {errors}");
    }
    assert!(catalog.iter().count() > 1000, "{}", catalog.iter().count());
    assert_eq!(
        Catalog::builtin().iter().count(),
        catalog.iter().count(),
        "builtin() is the checked-in data"
    );
    let mut previous: Option<(&str, &str)> = None;
    for spec in catalog.iter() {
        let key = (spec.provider.vendor(), spec.id.as_str());
        assert!(previous < Some(key), "sorted and unique: {key:?}");
        previous = Some(key);
    }
}

#[test]
fn a_models_dev_row_becomes_a_spec() {
    let catalog = models_dev(SAMPLE).expect("parses");
    let claude = spec(&catalog, "anthropic", "claude-x");
    assert_eq!(claude.display_name, "Claude X");
    assert_eq!(claude.context_window, Some(200_000));
    assert_eq!(claude.max_output_tokens, Some(64_000));
    assert!(claude.input.text && claude.input.image && claude.input.pdf);
    assert!(!claude.input.audio && !claude.input.video);
    assert!(claude.tools && !claude.structured_output && !claude.deprecated);
    assert!(claude.reasoning.supported());
    assert_eq!(
        claude.reasoning.levels(),
        Some(&[Effort::Low, Effort::Medium, Effort::High, Effort::Max][..])
    );
    assert_eq!(
        claude.reasoning.budget().cloned(),
        Some(1024..=64_000),
        "max: the output limit"
    );
    assert_eq!(claude.reasoning.can_disable(), Some(false));
    assert_eq!(claude.reasoning.default_effort(), Some(Effort::High));
    assert_eq!(
        claude.caching.retention,
        [
            CacheRetention::None,
            CacheRetention::Short,
            CacheRetention::Long
        ]
    );
    assert!(claude.compat.adaptive_thinking && !claude.compat.binds_context);
    let pricing = claude.pricing.expect("priced");
    assert_eq!((pricing.input, pricing.output), (3.0, 15.0));
    assert_eq!((pricing.cache_read, pricing.cache_write), (Some(0.3), None));

    let gemini = spec(&catalog, "gcp.gemini", "gemini-x");
    assert_eq!(
        gemini.reasoning.can_disable(),
        Some(true),
        "a toggle turns it off"
    );
    assert_eq!(gemini.reasoning.budget().cloned(), Some(0..=24_576));
    assert!(gemini.deprecated);
    assert_eq!(gemini.pricing, None);

    let qwen = spec(&catalog, "groq", "qwen-x");
    assert_eq!(
        qwen.reasoning.levels(),
        Some(&[Effort::Low, Effort::High][..]),
        "`default` dropped"
    );
    assert_eq!(
        qwen.reasoning.can_disable(),
        Some(true),
        "`none` turns it off"
    );

    assert_eq!(
        catalog.iter().count(),
        4,
        "a provider rig does not serve is skipped"
    );
}

#[test]
fn an_override_wins_field_by_field() {
    let base = models_dev(SAMPLE).expect("parses");
    let overrides = Catalog::from_overrides(
        r#"{"anthropic": {"models": {
            "claude-x": {"limit": {"output": 128000}, "cost": {"input": 5}, "rig": {"binds_context": true}},
            "claude-y": {"name": "Claude Y", "reasoning": false, "tool_call": true, "modalities": {"input": ["text"]}}
        }}}"#,
        &base,
    )
    .expect("parses");
    let catalog = base.with_overrides(&overrides);
    let claude = spec(&catalog, "anthropic", "claude-x");
    assert_eq!(claude.max_output_tokens, Some(128_000));
    assert_eq!(claude.context_window, Some(200_000), "kept");
    assert_eq!(claude.reasoning.budget().cloned(), Some(1024..=128_000));
    let pricing = claude.pricing.expect("priced");
    assert_eq!((pricing.input, pricing.output), (5.0, 15.0));
    assert!(claude.compat.binds_context && claude.compat.adaptive_thinking);
    assert_eq!(
        claude.reasoning.default_effort(),
        Some(Effort::High),
        "kept"
    );
    assert_eq!(
        spec(&catalog, "anthropic", "claude-y").display_name,
        "Claude Y"
    );
    assert_eq!(catalog.iter().count(), 5);

    assert_eq!(base.iter().count(), 4, "the base is unchanged");
    assert_eq!(
        spec(&base, "anthropic", "claude-x").max_output_tokens,
        Some(64_000)
    );
    let shared = |catalog: &Catalog, vendor: &str, model: &str| {
        let position = catalog.position(vendor, model).expect("listed");
        Arc::clone(&catalog.entries.get(position).expect("in range").spec)
    };
    assert!(
        Arc::ptr_eq(
            &shared(&base, "gcp.gemini", "gemini-x"),
            &shared(&catalog, "gcp.gemini", "gemini-x")
        ),
        "a row the override leaves alone is shared, not copied"
    );
}

#[test]
fn cloning_a_catalog_shares_its_rows() {
    let builtin = Catalog::builtin();
    let copy = builtin.clone();
    assert!(Arc::ptr_eq(&builtin.entries, &copy.entries));
    let unchanged = builtin.with_overrides(&Catalog::default());
    assert!(Arc::ptr_eq(&builtin.entries, &unchanged.entries));
}

#[test]
fn references_name_a_vendor_and_a_model() {
    let catalog = models_dev(SAMPLE).expect("parses");
    let id = |reference: &str| {
        catalog
            .resolve(reference)
            .ok()
            .map(|resolved| resolved.spec.id.clone())
    };
    for reference in ["anthropic/claude-x", "anthropic/anthropic:claude-x"] {
        assert_eq!(id(reference).as_deref(), Some("claude-x"), "{reference}");
    }
    assert_eq!(
        catalog
            .resolve("aws_bedrock/us.anthropic.claude-x-v1:0")
            .ok()
            .map(|resolved| resolved.spec.provider.vendor()),
        Some("aws_bedrock"),
        "a `:` inside the model id leaves `vendor/model`"
    );
    for missing in [
        "claude-x",
        "anthropic:claude-x",
        "anthropic/",
        "/claude-x",
        "anthropic/claude-x-1",
        "",
    ] {
        assert_eq!(id(missing), None, "{missing}");
    }
    let openrouter = Catalog::builtin()
        .resolve("openrouter/anthropic/claude-sonnet-4.5")
        .expect("OpenRouter lists it");
    assert_eq!(openrouter.spec.id, "anthropic/claude-sonnet-4.5");
}

/// `get` and `resolve` follow one rule: the id as listed, else the longest
/// listed id the requested one extends with `-20` and a year. The result
/// says which step matched; `get_exact` takes the first step only.
#[test]
fn one_rule_finds_an_id_or_a_dated_snapshot_of_it() {
    let catalog = models_dev(SAMPLE).expect("parses");
    let anthropic = ProviderId::catalog("anthropic").expect("a known vendor");
    let matched = |model: &str| catalog.get(anthropic, model).map(|found| found.matched);
    assert_eq!(matched("claude-x"), Some(Matched::Exact));
    for model in [
        "claude-x-20251001",
        "claude-x-2025-10-01",
        "claude-x-20260601-v1:0",
    ] {
        assert_eq!(
            matched(model),
            Some(Matched::SnapshotOf("claude-x".to_owned())),
            "{model}"
        );
        assert_eq!(catalog.get_exact(anthropic, model), None, "{model}");
        assert_eq!(
            catalog
                .resolve(&format!("anthropic/{model}"))
                .map(|found| found.spec.id.clone()),
            Ok("claude-x".to_owned()),
            "{model}"
        );
    }
    for model in [
        "claude-x-1",
        "claude-x-0",
        "claude-x-25-10-01",
        "claude-x-20b",
        "claude-x-2",
        "claude-x.20251001",
    ] {
        assert_eq!(matched(model), None, "{model}");
    }

    let builtin = Catalog::builtin();
    let id = |model: &str| {
        builtin
            .get(anthropic, model)
            .map(|found| (found.spec.id.as_str(), found.matched))
    };
    assert_eq!(
        id("claude-opus-5-5-20260601-v1:0"),
        Some((
            "claude-opus-5-5",
            Matched::SnapshotOf("claude-opus-5-5".into())
        )),
        "the longest listed id wins over `claude-opus-5`"
    );
    assert_eq!(
        id("claude-sonnet-4-5-20250929"),
        Some(("claude-sonnet-4-5-20250929", Matched::Exact)),
        "a listed snapshot is its own row"
    );
    assert_eq!(
        id("claude-opus-5.5"),
        None,
        "another spelling is not listed"
    );
    assert_eq!(id("claude-opus-5-5-latest"), None);
}

/// Every listed id finds itself exactly, and its dated snapshot finds that
/// snapshot's own row when listed, else the id.
#[test]
fn get_reads_every_row_and_its_dated_snapshots() {
    let catalog = Catalog::builtin();
    let mut rows = 0;
    for spec in catalog.iter() {
        let found = |model: &str| {
            catalog
                .get(spec.provider, model)
                .map(|found| &found.spec.id)
        };
        assert_eq!(
            catalog
                .get(spec.provider, &spec.id)
                .map(|found| found.matched),
            Some(Matched::Exact),
            "{}",
            spec.id
        );
        for dated in [
            format!("{}-20260601", spec.id),
            format!("{}-2026-06-01", spec.id),
        ] {
            let listed = catalog
                .get_exact(spec.provider, &dated)
                .map(|listed| &listed.id);
            assert_eq!(found(&dated), listed.or(Some(&spec.id)), "{dated}");
        }
        rows += 1;
    }
    assert!(rows > 0, "the built-in catalog has rows");
}

/// A miss names close references to listed models that are not deprecated,
/// closest first, five at most, and says so in its message.
#[test]
fn a_miss_suggests_close_references() {
    let catalog = Catalog::builtin();
    for typo in [
        "anthropic/claude-opus-5.5",
        "antropic/claude-opus-5-5",
        "anthropic/claude-opus-5-55",
    ] {
        let missed = catalog.resolve(typo).expect_err("not listed");
        assert_eq!(missed.reference, typo);
        assert_eq!(
            missed.suggestions.first().map(String::as_str),
            Some("anthropic/claude-opus-5-5"),
            "{typo}: {:?}",
            missed.suggestions
        );
        assert!(missed.suggestions.len() <= 5, "{typo}");
        assert!(
            missed.to_string().contains("did you mean"),
            "{typo}: {missed}"
        );
    }
    let opus = catalog.resolve("anthropic/opus").expect_err("not listed");
    assert!(!opus.suggestions.is_empty(), "a part of an id finds models");
    for suggestion in &opus.suggestions {
        let found = catalog.resolve(suggestion).expect("a suggestion is listed");
        assert!(!found.spec.deprecated, "{suggestion}");
        assert!(suggestion.contains("opus"), "{suggestion}");
    }
    let nothing = catalog
        .resolve("nowhere/qqqqqqqqqqqqqqqqqqqq")
        .expect_err("not listed");
    assert_eq!(nothing.suggestions, Vec::<String>::new());
    assert_eq!(
        nothing.to_string(),
        "the catalog lists no model `nowhere/qqqqqqqqqqqqqqqqqqqq`"
    );
}

#[test]
fn edit_distance_counts_single_character_edits() {
    assert_eq!(lookup::edit_distance("", ""), 0);
    assert_eq!(lookup::edit_distance("abc", ""), 3);
    assert_eq!(lookup::edit_distance("", "abc"), 3);
    assert_eq!(lookup::edit_distance("antropic", "anthropic"), 1);
    assert_eq!(lookup::edit_distance("kitten", "sitting"), 3);
    assert_eq!(lookup::edit_distance("5.5", "5-5"), 1);
    // A swap of two adjacent characters is one edit.
    assert_eq!(lookup::edit_distance("inptu", "input"), 1);
    assert_eq!(lookup::edit_distance("ab", "ba"), 1);
    assert_eq!(lookup::edit_distance("ca", "abc"), 3);
    assert_eq!(lookup::edit_distance("output", "input"), 3);
}

#[test]
fn catalog_only_providers_have_entries_and_no_preset() {
    let catalog = models_dev(SAMPLE).expect("parses");
    let bedrock = spec(&catalog, "aws_bedrock", "us.anthropic.claude-x-v1:0");
    assert!(!bedrock.provider.is_registered());
    assert_eq!(bedrock.provider.format(), None);
    assert_eq!(bedrock.provider.to_string(), "aws_bedrock");
    assert!(bedrock.provider.config("key").is_none());
    let id: ProviderId = serde_json::from_str("\"aws_bedrock\"").expect("reads back");
    assert_eq!(id, bedrock.provider);
}

#[test]
fn validate_refuses_what_the_model_does_not_take() {
    let catalog = models_dev(SAMPLE).expect("parses");
    let claude = spec(&catalog, "anthropic", "claude-x");
    assert!(claude.validate(&options(Effort::High)).is_ok());
    assert!(
        claude
            .validate(&options(Reasoning::Budget { tokens: 2048 }))
            .is_ok()
    );
    let refused = claude
        .validate(&options(Effort::XHigh))
        .expect_err("no xhigh");
    assert_eq!(refused.option, "reasoning");
    assert_eq!(refused.provider, "anthropic");
    assert_eq!(refused.model, "claude-x");
    assert!(refused.reason.contains("`xhigh`"), "{}", refused.reason);
    assert!(claude.validate(&options(Reasoning::Off)).is_err());
    assert!(
        claude
            .validate(&options(Reasoning::Budget { tokens: 512 }))
            .is_err()
    );
    assert!(
        claude
            .validate(&GenerationOptions::default().cache(CacheRetention::Long))
            .is_ok()
    );

    let gemini = spec(&catalog, "gcp.gemini", "gemini-x");
    assert!(gemini.validate(&options(Reasoning::Off)).is_ok());
    let refused = gemini
        .validate(&options(Effort::Low))
        .expect_err("budget only");
    assert!(refused.reason.contains("budget"), "{}", refused.reason);
    let refused = gemini
        .validate(&options(Reasoning::Budget { tokens: 30_000 }))
        .expect_err("over the range");
    assert!(refused.reason.contains("0 to 24576"), "{}", refused.reason);
    assert!(
        gemini
            .validate(&GenerationOptions::default().cache(CacheRetention::Long))
            .is_ok(),
        "unknown caching refuses nothing"
    );

    let bedrock = spec(&catalog, "aws_bedrock", "us.anthropic.claude-x-v1:0");
    assert!(bedrock.validate(&options(Reasoning::Off)).is_ok());
    let refused = bedrock
        .validate(&options(Effort::Low))
        .expect_err("no reasoning");
    assert_eq!(refused.reason, "the model does not reason");
}

/// A reasoning row that lists no options is unknown and refuses nothing;
/// only the hand-entered `"reasoning_control": "none"` makes it refuse.
#[test]
fn a_reasoning_row_without_options_is_unknown_unless_it_says_none() {
    let catalog = models_dev(
        r#"{"openrouter": {"models": {
            "listed-nothing": {"reasoning": true, "reasoning_options": []},
            "absent": {"reasoning": true, "rig": {"reasoning_default": "medium"}},
            "unread-kind": {"reasoning": true, "reasoning_options": [{"type": "levels"}]},
            "rejects": {"reasoning": true, "reasoning_options": [], "rig": {"reasoning_control": "none"}}
        }}}"#,
    )
    .expect("parses");
    for model in ["listed-nothing", "absent", "unread-kind"] {
        let unknown = spec(&catalog, "openrouter", model);
        assert!(
            matches!(unknown.reasoning, ReasoningSupport::Unknown { .. }),
            "{model}: {:?}",
            unknown.reasoning
        );
        assert!(unknown.reasoning.supported() && unknown.reasoning.levels().is_none());
        assert_eq!(unknown.reasoning.can_disable(), None);
        for reasoning in [
            Reasoning::Off,
            Reasoning::Effort(Effort::XHigh),
            Reasoning::Budget { tokens: 1 },
        ] {
            assert!(unknown.validate(&options(reasoning)).is_ok(), "{model}");
        }
    }
    assert_eq!(
        spec(&catalog, "openrouter", "absent")
            .reasoning
            .default_effort(),
        Some(Effort::Medium)
    );

    let rejects = spec(&catalog, "openrouter", "rejects");
    assert_eq!(rejects.reasoning.levels(), Some(&[][..]));
    assert_eq!(rejects.reasoning.can_disable(), Some(false));
    for reasoning in [
        Reasoning::Off,
        Reasoning::Effort(Effort::High),
        Reasoning::Budget { tokens: 2048 },
    ] {
        assert!(rejects.validate(&options(reasoning)).is_err());
    }
}

/// The shipped rows keep the distinction: OpenAI's o1-mini rejects every
/// reasoning control, a gateway row with no options refuses none.
#[test]
fn the_builtin_catalog_tells_unknown_from_none() {
    let builtin = Catalog::builtin();
    let o1_mini = spec(builtin, "openai", "o1-mini");
    assert_eq!(o1_mini.reasoning.levels(), Some(&[][..]));
    assert!(o1_mini.validate(&options(Effort::Low)).is_err());
    let unknown = builtin
        .iter()
        .filter(|spec| matches!(spec.reasoning, ReasoningSupport::Unknown { .. }))
        .collect::<Vec<_>>();
    assert!(unknown.len() > 100, "{} unknown rows", unknown.len());
    assert!(
        unknown
            .iter()
            .all(|spec| spec.validate(&options(Effort::High)).is_ok())
    );
}

/// A spec's row keeps its reasoning state when laid over a row that says
/// otherwise.
#[test]
fn a_spec_row_carries_its_reasoning_state() {
    let openai = ProviderId::catalog("openai").expect("a known vendor");
    let states = [
        ReasoningSupport::Unknown {
            default: Some(Effort::Low),
        },
        ReasoningSupport::Listed {
            levels: Vec::new(),
            budget: None,
            can_disable: false,
            default: None,
        },
        ReasoningSupport::None,
    ];
    for state in states {
        let row = ModelSpec::new(openai, "o1-mini")
            .with_reasoning(state.clone())
            .to_row_json();
        let overrides =
            strict(&serde_json::json!({"openai": {"models": {"o1-mini": row}}}).to_string())
                .expect("parses");
        for base in [
            Catalog::builtin().clone(),
            models_dev(SAMPLE).expect("parses"),
        ] {
            let catalog = base.with_overrides(&overrides);
            assert_eq!(spec(&catalog, "openai", "o1-mini").reasoning, state);
        }
    }
}

#[test]
fn a_cache_retention_the_model_does_not_honour_is_refused() {
    let catalog = models_dev(r#"{"openai": {"models": {"m": {"rig": {"cache": ["short"]}}}}}"#)
        .expect("parses");
    let model = spec(&catalog, "openai", "m");
    let refused = model
        .validate(&GenerationOptions::default().cache(CacheRetention::Long))
        .expect_err("short only");
    assert_eq!(refused.option, "cache");
    assert!(refused.reason.contains("`long`"), "{}", refused.reason);
    assert!(
        model
            .validate(&GenerationOptions::default().cache(CacheRetention::Short))
            .is_ok()
    );
}

/// A spec built in code starts with nothing known, is added whole by
/// `insert`, and takes an override field by field like any other row.
#[test]
fn a_spec_built_in_code_joins_the_catalog() {
    let ollama = ProviderId::catalog("ollama").expect("a known vendor");
    let bare = ModelSpec::new(ollama, "qwen3:4b");
    assert_eq!(bare.display_name, "qwen3:4b");
    assert_eq!((bare.context_window, bare.max_output_tokens), (None, None));
    assert!(bare.input.text && !bare.input.image);
    assert!(!bare.reasoning.supported() && !bare.tools && !bare.deprecated);
    assert_eq!((bare.pricing, bare.sampling), (None, None));
    assert_eq!(bare.compat, Compat::default());

    let mut catalog = Catalog::builtin().clone();
    catalog.insert(
        bare.with_context_window(32_768)
            .with_tools(true)
            .with_pricing(Pricing::new(0.1, 0.4).with_cache_read(0.01))
            .with_compat(Compat::default().with_reasoning_field("reasoning_content")),
    );
    let qwen = spec(&catalog, "ollama", "qwen3:4b");
    assert_eq!(qwen.context_window, Some(32_768));
    assert!(qwen.tools);
    assert_eq!(
        qwen.compat.reasoning_field.as_deref(),
        Some("reasoning_content")
    );
    assert!(
        Catalog::builtin()
            .get_exact(ollama, "qwen3:4b")
            .is_none_or(|listed| listed != qwen),
        "the built-in catalog is unchanged"
    );

    let overrides = Catalog::from_overrides(
        r#"{"ollama": {"models": {"qwen3:4b": {"limit": {"output": 8192}}}}}"#,
        &catalog,
    )
    .expect("the model is listed in the catalog it is laid over");
    let overridden = catalog.with_overrides(&overrides);
    let qwen = spec(&overridden, "ollama", "qwen3:4b");
    assert_eq!(qwen.max_output_tokens, Some(8192));
    assert_eq!(qwen.context_window, Some(32_768), "kept");
    assert_eq!(
        qwen.pricing,
        Some(Pricing::new(0.1, 0.4).with_cache_read(0.01))
    );

    let mut sample = models_dev(SAMPLE).expect("parses");
    let anthropic = ProviderId::catalog("anthropic").expect("a known vendor");
    sample.insert(ModelSpec::new(anthropic, "claude-x"));
    let replaced = spec(&sample, "anthropic", "claude-x");
    assert_eq!(
        replaced,
        &ModelSpec::new(anthropic, "claude-x"),
        "replaced whole"
    );
    assert_eq!(sample.iter().count(), 4);
}

/// Every built-in spec reads back from its serde form and from its override
/// row, so a spec a program changed can be saved either way.
#[test]
fn every_spec_round_trips_through_serde_and_its_row() {
    for spec in Catalog::builtin().iter() {
        let json = serde_json::to_string(spec).expect("serializes");
        let read: ModelSpec = serde_json::from_str(&json).expect("reads back");
        assert_eq!(&read, spec, "{json}");

        let vendor = spec.provider.vendor();
        let file =
            serde_json::json!({ vendor: { "models": { spec.id.as_str(): spec.to_row_json() } } });
        let catalog = strict(&file.to_string()).expect("the row reads");
        assert_eq!(
            catalog.iter().collect::<Vec<_>>(),
            [spec],
            "{vendor}/{}",
            spec.id
        );
    }
}

/// A row from a spec replaces every fact the spec knows when laid over
/// another row, and keeps the facts it does not know.
#[test]
fn a_spec_row_overrides_what_the_spec_knows() {
    let anthropic = ProviderId::catalog("anthropic").expect("a known vendor");
    let row = ModelSpec::new(anthropic, "claude-x")
        .with_display_name("Claude X2")
        .to_row_json();
    let overrides =
        strict(&serde_json::json!({"anthropic": {"models": {"claude-x": row}}}).to_string())
            .expect("parses");
    let catalog = models_dev(SAMPLE)
        .expect("parses")
        .with_overrides(&overrides);
    let claude = spec(&catalog, "anthropic", "claude-x");
    assert_eq!(claude.display_name, "Claude X2");
    assert!(!claude.reasoning.supported() && !claude.tools && !claude.input.image);
    assert!(!claude.compat.adaptive_thinking);
    assert_eq!(
        claude.context_window,
        Some(200_000),
        "an unknown limit is kept"
    );
    assert!(claude.pricing.is_some(), "unknown prices are kept");
}

/// A gateway row of a Claude model carries the Anthropic model's wire facts
/// itself, so no reader rewrites its id into Anthropic's spelling.
#[test]
fn gateway_rows_of_a_claude_model_carry_its_facts() {
    let catalog = Catalog::builtin();
    let anthropic = spec(catalog, "anthropic", "claude-opus-5-5");
    assert!(anthropic.compat.binds_context);
    for (vendor, model) in [
        ("aws_bedrock", "us.anthropic.claude-opus-5-5"),
        ("openrouter", "anthropic/claude-opus-5.5"),
        ("vertexai", "claude-opus-5-5@default"),
    ] {
        let gateway = spec(catalog, vendor, model);
        assert_eq!(gateway.compat, anthropic.compat, "{vendor}/{model}");
        assert_eq!(gateway.sampling, anthropic.sampling, "{vendor}/{model}");
    }
    assert_eq!(
        spec(catalog, "aws_bedrock", "us.anthropic.claude-opus-5-5").reasoning,
        anthropic.reasoning,
        "Bedrock takes Anthropic's own reasoning fields"
    );
}

/// A row's `rig.format` names the family its model is reached by; without
/// it the vendor's primary applies. A family the vendor is not registered
/// for is an error, and a spec's row carries its family over another row.
#[test]
fn a_row_names_the_family_its_model_is_reached_by() {
    use crate::providers::registry::Format;

    let catalog = models_dev(
        r#"{"zai": {"models": {
            "glm-x": {"name": "GLM X"},
            "glm-y": {"name": "GLM Y", "rig": {"format": "anthropic"}}
        }}}"#,
    )
    .expect("parses");
    let zai = ProviderId::catalog("zai").expect("known");
    let format = |model: &str| catalog.get(zai, model).map(|found| found.spec.format());
    assert_eq!(format("glm-x"), Some(Some(Format::OpenAi)));
    assert_eq!(format("glm-y"), Some(Some(Format::Anthropic)));
    assert_eq!(
        Catalog::builtin()
            .resolve("minimax/MiniMax-M2.7")
            .expect("listed")
            .spec
            .format(),
        Some(Format::Anthropic),
        "MiniMax's primary family is Anthropic's"
    );
    assert_eq!(
        Catalog::builtin()
            .resolve("aws_bedrock/us.anthropic.claude-sonnet-5")
            .expect("listed")
            .spec
            .format(),
        None
    );

    let file = r#"{"openai": {"models": {"gpt-x": {"rig": {"format": "gemini"}}}}}"#;
    let error = Catalog::from_overrides(file, Catalog::builtin())
        .expect_err("OpenAI speaks no Gemini endpoint");
    assert_eq!(
        error.0.iter().map(|e| e.path.as_str()).collect::<Vec<_>>(),
        ["openai.models.gpt-x.rig.format", "openai.models.gpt-x"],
        "{error}"
    );
    let (lenient, skipped) = Catalog::from_models_dev(file).expect("reads");
    assert_eq!(lenient.iter().count(), 0);
    assert_eq!(skipped.len(), 1, "the row is skipped: {skipped:?}");

    // The override replaces the family the base row names.
    let messages = catalog.get_exact(zai, "glm-y").expect("listed").clone();
    let mut chat = messages.clone();
    chat.provider = zai;
    let overrides = strict(&format!(
        r#"{{"zai": {{"models": {{"glm-y": {}}}}}}}"#,
        chat.to_row_json()
    ))
    .expect("parses");
    let laid = catalog.with_overrides(&overrides);
    assert_eq!(
        laid.get_exact(zai, "glm-y").map(ModelSpec::format),
        Some(Some(Format::OpenAi))
    );
    assert_eq!(
        catalog
            .with_overrides(
                &strict(&format!(
                    r#"{{"zai": {{"models": {{"glm-x": {}}}}}}}"#,
                    messages.to_row_json()
                ))
                .expect("parses")
            )
            .get_exact(zai, "glm-x")
            .map(ModelSpec::format),
        Some(Some(Format::Anthropic))
    );
}

/// `refusals` applies the sampling rule beside reasoning and cache: a model
/// that takes no sampling parameters refuses `temperature` and `top_p`, and
/// one that takes them only with reasoning off refuses them while it
/// reasons, which with no `reasoning` set it does when it names a default
/// effort. `validate` checks what the options hold, `top_p` included, and a
/// request whose options are default is not checked.
#[test]
fn refusals_apply_the_sampling_rule_beside_reasoning_and_cache() {
    use crate::completion::CompletionRequest;
    let names = |spec: &ModelSpec, request: &CompletionRequest| -> Vec<String> {
        spec.refusals(request)
            .into_iter()
            .map(|refusal| refusal.option.into_owned())
            .collect()
    };
    let openai = ProviderId::catalog("openai").expect("a known vendor");
    let thinker = ModelSpec::new(openai, "thinker")
        .with_reasoning(ReasoningSupport::Listed {
            levels: vec![Effort::Low, Effort::High],
            budget: None,
            can_disable: true,
            default: Some(Effort::Low),
        })
        .with_caching(CacheSupport {
            retention: vec![CacheRetention::Short],
        })
        .with_sampling(Sampling::ReasoningOff);
    let sampled = CompletionRequest::new("hi").temperature(0.2).top_p(0.9);
    assert_eq!(names(&thinker, &sampled), ["temperature", "top_p"]);
    assert_eq!(
        names(&thinker, &sampled.clone().reasoning(Reasoning::Off)),
        Vec::<String>::new()
    );
    let everything = sampled.clone().reasoning(Effort::Medium).options(
        GenerationOptions::default()
            .reasoning(Effort::Medium)
            .cache(CacheRetention::Long)
            .top_p(0.9),
    );
    assert_eq!(
        names(&thinker, &everything),
        ["reasoning", "cache", "temperature", "top_p"]
    );
    let refusal = &thinker.refusals(&everything)[2];
    assert_eq!(
        (refusal.provider.as_str(), refusal.model.as_str()),
        ("openai", "thinker")
    );
    assert_eq!(
        thinker
            .validate(&GenerationOptions::default().top_p(0.9))
            .map_err(|refusal| refusal.option),
        Err("top_p".into())
    );
    assert!(
        thinker
            .refusals(&CompletionRequest::new("hi").temperature(0.2))
            .is_empty(),
        "default options are not checked"
    );

    let fixed = thinker.clone().with_sampling(Sampling::Never);
    assert_eq!(
        names(&fixed, &sampled.clone().reasoning(Reasoning::Off)),
        ["temperature", "top_p"]
    );
    let undecided = thinker.with_reasoning(ReasoningSupport::Listed {
        levels: vec![Effort::Low],
        budget: None,
        can_disable: true,
        default: None,
    });
    assert!(
        names(&undecided, &sampled).is_empty(),
        "a model that names no default effort and can turn reasoning off is not reasoning"
    );
}

/// The requests the consistency test checks every row with: each reasoning
/// value, each cache retention, the sampling parameters with and without
/// reasoning, and `stop`.
fn option_matrix() -> Vec<crate::completion::CompletionRequest> {
    use crate::completion::CompletionRequest;
    let base = || CompletionRequest::new("hi").max_tokens(4096);
    let mut requests: Vec<CompletionRequest> = [
        Reasoning::Off,
        Reasoning::Effort(Effort::Minimal),
        Reasoning::Effort(Effort::Low),
        Reasoning::Effort(Effort::Medium),
        Reasoning::Effort(Effort::High),
        Reasoning::Effort(Effort::XHigh),
        Reasoning::Effort(Effort::Max),
        Reasoning::Budget { tokens: 2048 },
        Reasoning::Budget { tokens: 30_000 },
    ]
    .into_iter()
    .map(|reasoning| base().reasoning(reasoning))
    .collect();
    for cache in [
        CacheRetention::None,
        CacheRetention::Short,
        CacheRetention::Long,
    ] {
        requests.push(base().options(GenerationOptions::default().cache(cache)));
    }
    requests.push(base().top_p(0.9));
    requests.push(base().temperature(0.5).top_p(0.9));
    requests.push(base().temperature(0.5).reasoning(Effort::High));
    requests.push(base().temperature(0.5).reasoning(Reasoning::Off));
    requests.push(base().stop(["END"]));
    requests
}

/// The options `check` refuses for `request`, or `None` when the request
/// cannot be built for another reason.
fn checked(
    model: &crate::DynModel<crate::operation::Completion>,
    request: &crate::completion::CompletionRequest,
) -> Option<Vec<String>> {
    use crate::completion::CheckError;
    match model.check(request) {
        Ok(()) => Some(Vec::new()),
        Err(CheckError::Unsupported(refused)) => Some(
            refused
                .into_iter()
                .map(|refusal| refusal.option.into_owned())
                .collect(),
        ),
        Err(_) => None,
    }
}

/// `request` without the options in `dropped`, as `check` runs it, under
/// `OnUnsupported::Ignore`: the request the catalog's rules see once the
/// wire's refusals of `dropped` are applied.
fn without(
    request: &crate::completion::CompletionRequest,
    dropped: &[&String],
) -> crate::completion::CompletionRequest {
    let mut request = request.clone();
    if !request.options.is_default() {
        request.options.on_unsupported = Some(crate::completion::OnUnsupported::Ignore);
    }
    let drops = |option: &str| dropped.iter().any(|dropped| *dropped == option);
    let options = &mut request.options;
    if drops("reasoning") {
        options.reasoning = None;
    }
    if drops("cache") {
        options.cache = None;
    }
    if drops("top_p") {
        options.top_p = None;
    }
    if drops("stop") {
        options.stop.clear();
    }
    if drops("temperature") {
        request.temperature = None;
    }
    request
}

fn option_names(refusals: Vec<crate::completion::UnsupportedOption>) -> Vec<String> {
    refusals
        .into_iter()
        .map(|refusal| refusal.option.into_owned())
        .collect()
}

/// `validate` and the wires share one rule set. For every row rig-core
/// connects and a matrix of requests, on the row's own wire:
///
/// - a request [`ModelSpec::refusals`] refuses, `DynModel::check` refuses,
///   and `validate`'s refusal is one of them;
/// - every option the catalog's rules refuse once the wire's own refusals
///   are applied, `check` refuses too;
/// - every other option `check` refuses, the wire refuses for that model id
///   when the catalog lists nothing, so it is the wire's own rule (its
///   route, its API or its naming rule) and reads no catalog fact.
///
/// The providers only the catalog knows are served, and checked, by their
/// companion crates.
#[test]
fn check_refuses_what_the_catalog_refuses_and_the_rest_is_the_wires() {
    use crate::providers::registry::ConnectOptions;
    let http = crate::test_utils::RecordingHttpClient::new("{}");
    let options = || {
        ConnectOptions::new()
            .api_key("sk-test")
            .base_url("http://localhost")
            .http(http.clone())
    };
    let requests = option_matrix();
    let empty = Catalog::default();
    let (mut rows, mut cells) = (0, 0);
    let mut disagreements = Vec::new();
    for spec in Catalog::builtin().iter() {
        // A provider only the catalog knows is served by its companion crate.
        let Some(format) = spec.format() else {
            continue;
        };
        let model = Catalog::builtin()
            .connect_with(spec, options())
            .unwrap_or_else(|error| panic!("{}: {error}", spec.id));
        rows += 1;
        let reference = format!("{}/{}", spec.provider.vendor(), spec.id);
        let mut bare = None;
        for request in &requests {
            let Some(refused) = checked(&model, request) else {
                continue;
            };
            cells += 1;
            let mut disagree = |what: String| {
                disagreements.push(format!("{reference}: {what} ({:?})", request.options));
            };
            let catalog = option_names(spec.refusals(request));
            if !catalog.is_empty() && refused.is_empty() {
                disagree(format!("the catalog refuses {catalog:?}, check nothing"));
            }
            if let Err(first) = spec.validate(&request.options)
                && !catalog.contains(&first.option.clone().into_owned())
            {
                disagree(format!("validate's {} is not a refusal", first.option));
            }
            // What `check` refuses beyond the catalog's rules: the wire's own
            // refusals, and the catalog's once those are applied.
            let others: Vec<&String> = refused
                .iter()
                .filter(|option| !catalog.contains(option))
                .collect();
            let after = option_names(spec.refusals(&without(request, &others)));
            for option in after.iter().filter(|option| !refused.contains(option)) {
                disagree(format!("{option} refused by the catalog, not by check"));
            }
            let caught_after = |kept: &String| {
                let dropped: Vec<&String> = others
                    .iter()
                    .copied()
                    .filter(|other| *other != kept)
                    .collect();
                option_names(spec.refusals(&without(request, &dropped))).contains(kept)
            };
            let own: Vec<&String> = others
                .iter()
                .copied()
                .filter(|option| !caught_after(option))
                .collect();
            if own.is_empty() {
                continue;
            }
            let bare = bare.get_or_insert_with(|| {
                empty
                    .connect_with(reference.as_str(), options().format(format))
                    .expect("an unlisted model of a known vendor connects")
            });
            let wire = checked(bare, request).unwrap_or_default();
            for option in own.into_iter().filter(|option| !wire.contains(option)) {
                disagree(format!(
                    "{option} refused by check by a catalog fact `refusals` does not apply"
                ));
            }
        }
    }
    assert!(rows > 1000 && cells > 15_000, "{rows} rows, {cells} cells");
    assert!(
        disagreements.is_empty(),
        "{} disagreements:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );
}

/// The sync's time reads as seconds since the epoch, across leap days and
/// centuries, and nothing else reads.
#[test]
fn the_generation_time_is_an_rfc_3339_utc_time() {
    for (time, seconds) in [
        ("1970-01-01T00:00:00Z", 0),
        ("2000-03-01T00:00:00Z", 951_868_800),
        ("2024-02-29T12:00:00Z", 1_709_208_000),
        ("2026-10-07T06:50:11Z", 1_791_355_811),
    ] {
        assert_eq!(unix_seconds(time), Some(seconds), "{time}");
    }
    for not_a_time in [
        "",
        "2026-10-07",
        "2026-10-07T06:50:11",
        "2026-10-07T06:50:11+00:00",
        "2026-13-07T06:50:11Z",
        "2026-10-07T24:00:00Z",
        "2026-1O-07T06:50:11Z",
    ] {
        assert_eq!(unix_seconds(not_a_time), None, "{not_a_time}");
    }
    let generated = Catalog::generated_at();
    assert!(
        generated > UNIX_EPOCH + Duration::from_secs(1_767_225_600),
        "after 2026"
    );
    assert!(generated <= SystemTime::now(), "not in the future");
}

/// A cache part with tokens and no listed rate is unknown, not priced at
/// the input rate: `total` sums the known parts and says it is a lower
/// bound. No tokens cost nothing whether or not the rate is listed.
#[test]
fn an_unlisted_cache_rate_leaves_its_part_unknown() {
    use crate::completion::Usage;
    let usage = Usage::new()
        .input_tokens(1_000_000)
        .output_tokens(1_000_000)
        .cached_input_tokens(500_000);

    let listed = Pricing::new(2.0, 8.0).with_cache_read(0.5);
    let cost = listed.cost(&usage).expect("input and output reported");
    assert_eq!(cost.cache_read, Some(0.25));
    assert_eq!(cost.cache_write, Some(0.0), "no cache writes");
    assert_eq!(cost.total, 1.0 + 8.0 + 0.25);
    assert!(cost.is_complete());

    let unlisted = Pricing::new(2.0, 8.0);
    let cost = unlisted.cost(&usage).expect("input and output reported");
    assert_eq!((cost.input, cost.output), (Some(1.0), Some(8.0)));
    assert_eq!(cost.cache_read, None, "not the input rate");
    assert_eq!(cost.total, 9.0, "the known parts");
    assert!(!cost.is_complete());

    let uncached = Usage::new().input_tokens(1_000).output_tokens(1_000);
    let cost = unlisted.cost(&uncached).expect("input and output reported");
    assert_eq!((cost.cache_read, cost.cache_write), (Some(0.0), Some(0.0)));
    assert!(cost.is_complete());
}
