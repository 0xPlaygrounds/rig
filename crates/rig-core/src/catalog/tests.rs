//! What the catalog promises: the checked-in data loads whole, lookups and
//! references find the model they name, overrides win field by field, and
//! `validate` refuses what the model does not take.

use super::*;
use crate::completion::{CacheRetention, Effort, GenerationOptions, Reasoning};

fn options(reasoning: impl Into<Reasoning>) -> GenerationOptions {
    GenerationOptions::default().reasoning(reasoning)
}

fn spec<'a>(catalog: &'a Catalog, vendor: &str, model: &str) -> &'a ModelSpec {
    let provider = ProviderId::catalog(vendor).expect("a known vendor");
    catalog
        .get(provider, model)
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
    let catalog = Catalog::from_json(BUILTIN).expect("the checked-in data parses");
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
    let catalog = Catalog::from_json(SAMPLE).expect("parses");
    let claude = spec(&catalog, "anthropic", "claude-x");
    assert_eq!(claude.display_name, "Claude X");
    assert_eq!(claude.context_window, Some(200_000));
    assert_eq!(claude.max_output_tokens, Some(64_000));
    assert!(claude.input.text && claude.input.image && claude.input.pdf);
    assert!(!claude.input.audio && !claude.input.video);
    assert!(claude.tools && !claude.structured_output && !claude.deprecated);
    assert!(claude.reasoning.supported);
    assert_eq!(
        claude.reasoning.levels,
        [Effort::Low, Effort::Medium, Effort::High, Effort::Max]
    );
    assert_eq!(
        claude.reasoning.budget,
        Some(1024..=64_000),
        "max: the output limit"
    );
    assert!(!claude.reasoning.can_disable);
    assert_eq!(claude.reasoning.default, Some(Effort::High));
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
    assert!(gemini.reasoning.can_disable, "a toggle turns it off");
    assert_eq!(gemini.reasoning.budget, Some(0..=24_576));
    assert!(gemini.deprecated);
    assert_eq!(gemini.pricing, None);

    let qwen = spec(&catalog, "groq", "qwen-x");
    assert_eq!(
        qwen.reasoning.levels,
        [Effort::Low, Effort::High],
        "`default` dropped"
    );
    assert!(qwen.reasoning.can_disable, "`none` turns it off");

    assert_eq!(
        catalog.iter().count(),
        4,
        "a provider rig does not serve is skipped"
    );
}

#[test]
fn an_unknown_rig_fact_is_an_error() {
    let json = r#"{"anthropic": {"models": {"m": {"rig": {"adaptive": true}}}}}"#;
    assert!(Catalog::from_json(json).is_err());
    assert!(Catalog::from_json("[]").is_err());
}

#[test]
fn an_override_wins_field_by_field() {
    let overrides = Catalog::from_json(
        r#"{"anthropic": {"models": {
            "claude-x": {"limit": {"output": 128000}, "cost": {"input": 5}, "rig": {"binds_context": true}},
            "claude-y": {"name": "Claude Y"}
        }}}"#,
    )
    .expect("parses");
    let base = Catalog::from_json(SAMPLE).expect("parses");
    let catalog = base.with_overrides(&overrides);
    let claude = spec(&catalog, "anthropic", "claude-x");
    assert_eq!(claude.max_output_tokens, Some(128_000));
    assert_eq!(claude.context_window, Some(200_000), "kept");
    assert_eq!(claude.reasoning.budget, Some(1024..=128_000));
    let pricing = claude.pricing.expect("priced");
    assert_eq!((pricing.input, pricing.output), (5.0, 15.0));
    assert!(claude.compat.binds_context && claude.compat.adaptive_thinking);
    assert_eq!(claude.reasoning.default, Some(Effort::High), "kept");
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
    let catalog = Catalog::from_json(SAMPLE).expect("parses");
    for reference in [
        "anthropic/claude-x",
        "anthropic:claude-x",
        "anthropic/anthropic:claude-x",
    ] {
        assert_eq!(
            catalog.resolve(reference).map(|spec| spec.id.as_str()),
            Some("claude-x"),
            "{reference}"
        );
    }
    assert_eq!(
        catalog
            .resolve("aws_bedrock/us.anthropic.claude-x-v1:0")
            .map(|spec| spec.provider.vendor()),
        Some("aws_bedrock"),
        "a `:` inside the model id leaves `vendor/model`"
    );
    for missing in [
        "claude-x",
        "anthropic/",
        "/claude-x",
        "anthropic/claude-x-1",
        "",
    ] {
        assert!(catalog.resolve(missing).is_none(), "{missing}");
    }
    let openrouter = Catalog::builtin()
        .resolve("openrouter/anthropic/claude-sonnet-4.5")
        .expect("OpenRouter lists it");
    assert_eq!(openrouter.id, "anthropic/claude-sonnet-4.5");
}

#[test]
fn the_encoders_find_a_dated_snapshot_by_its_model() {
    let catalog = Catalog::from_json(SAMPLE).expect("parses");
    for model in ["claude-x", "claude-x-20251001", "claude-x-2025-10-01"] {
        assert_eq!(
            catalog
                .find_vendor("anthropic", model)
                .map(|spec| spec.id.as_str()),
            Some("claude-x"),
            "{model}"
        );
    }
    for model in [
        "claude-x-1",
        "claude-x-2025101",
        "claude-x-0",
        "claude-x-25-10-01",
    ] {
        assert!(catalog.find_vendor("anthropic", model).is_none(), "{model}");
    }
}

/// `find` is `get` on every listed id, and finds every listed id from its
/// dated snapshot ids.
#[test]
fn find_reads_every_row_and_its_dated_snapshots() {
    let catalog = Catalog::builtin();
    let mut rows = 0;
    for spec in catalog.iter() {
        let found = |model: &str| catalog.find(spec.provider, model).map(|found| &found.id);
        assert_eq!(found(&spec.id), Some(&spec.id), "{}", spec.id);
        assert_eq!(
            catalog.find(spec.provider, &spec.id),
            catalog.get(spec.provider, &spec.id)
        );
        for dated in [
            format!("{}-20260601", spec.id),
            format!("{}-2026-06-01", spec.id),
        ] {
            let listed = catalog.get(spec.provider, &dated).map(|listed| &listed.id);
            assert_eq!(found(&dated), listed.or(Some(&spec.id)), "{dated}");
        }
        rows += 1;
    }
    assert!(rows > 0, "the built-in catalog has rows");
}

/// On every listed id and every eight-digit dated snapshot of one, the
/// snapshot lookup cost and Anthropic read agrees with `find`. It differs
/// only on a suffix past the date (`-20260601-v1:0`), which `find` does
/// not strip, so the two stay apart.
#[test]
fn the_snapshot_lookup_agrees_with_find_up_to_a_suffix_past_the_date() {
    let catalog = Catalog::builtin();
    for spec in catalog.iter() {
        let vendor = spec.provider.vendor();
        for model in [spec.id.clone(), format!("{}-20260601", spec.id)] {
            assert_eq!(
                lookup_snapshot(vendor, &model).map(|found| &found.id),
                catalog.find(spec.provider, &model).map(|found| &found.id),
                "{vendor}/{model}"
            );
        }
    }
    let past_the_date = "claude-opus-5-5-20260601-v1:0";
    assert!(lookup_snapshot("anthropic", past_the_date).is_some());
    let anthropic = ProviderId::catalog("anthropic").expect("a known vendor");
    assert!(catalog.find(anthropic, past_the_date).is_none());
}

/// A snapshot is any suffix from `-20` after a listed id, the longest such
/// id winning; another spelling is not listed.
#[test]
fn a_snapshot_lookup_reads_any_suffix_from_a_date() {
    let id = |model: &str| lookup_snapshot("anthropic", model).map(|spec| spec.id.as_str());
    assert_eq!(id("claude-opus-5-5"), Some("claude-opus-5-5"));
    assert_eq!(id("claude-opus-5-5-20260601"), Some("claude-opus-5-5"));
    assert_eq!(id("claude-opus-5-5-20260601-v1:0"), Some("claude-opus-5-5"));
    assert_eq!(id("claude-opus-5-20260601"), Some("claude-opus-5"));
    assert_eq!(id("claude-opus-5.5"), None);
    assert_eq!(id("claude-opus-5-5-latest"), None);
}

/// An id the catalog lists reads images as its entry says; any other id is
/// read by the rule the caller passes, never by a default.
#[test]
fn an_unlisted_model_reads_images_by_its_vendor_rule() {
    assert!(!reads_images_or("openai", "gpt-3.5-turbo", |_| true));
    assert!(reads_images_or("openai", "gpt-4o", |_| false));
    assert!(!reads_images_or("minimax", "minimax-m2.5", |_| false));
    assert!(reads_images_or("minimax", "minimax-m2.5", |_| true));
}

#[test]
fn catalog_only_providers_have_entries_and_no_preset() {
    let catalog = Catalog::from_json(SAMPLE).expect("parses");
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
    let catalog = Catalog::from_json(SAMPLE).expect("parses");
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

#[test]
fn a_cache_retention_the_model_does_not_honour_is_refused() {
    let catalog =
        Catalog::from_json(r#"{"openai": {"models": {"m": {"rig": {"cache": ["short"]}}}}}"#)
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
