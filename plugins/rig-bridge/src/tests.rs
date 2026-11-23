use std::ffi::OsStr;
use std::path::Path;

use rig_harness::rig_core::catalog::{Catalog, ModelSpec, ReasoningSupport};
use rig_harness::rig_core::completion::{Effort, Reasoning};
use rig_harness::rig_core::providers::registry::ProviderId;

use crate::harness::Vendor;
use crate::plugin::{enrich_limits, enrich_reasoning, expand_home_with};

#[test]
fn enrich_limits_populates_context_windows_across_bridge_vendors() {
    let catalog = Catalog::builtin();
    let check = |vendor: Vendor, id: &str| {
        let Ok(pid) = ProviderId::declare(vendor.as_str()) else {
            return None;
        };
        enrich_limits(catalog, vendor, ModelSpec::new(pid, id)).input_limit()
    };

    assert_eq!(check(Vendor::GrokBuild, "grok-4.7"), Some(500_000));
    assert_eq!(
        check(Vendor::GrokBuild, "grok-4.7-build-fast"),
        Some(500_000)
    );
    assert_eq!(
        check(Vendor::Antigravity, "gemini-4-neon-high"),
        Some(1_048_576)
    );
    assert_eq!(
        check(Vendor::Antigravity, "gemini-4-carbon-medium"),
        Some(1_048_576)
    );
    assert_eq!(
        check(Vendor::Antigravity, "gemini-pro-orchestrator-neon"),
        Some(1_048_576)
    );
    assert_eq!(
        check(Vendor::Antigravity, "claude-opus-5-5-medium"),
        Some(1_000_000)
    );
    assert_eq!(
        check(Vendor::Antigravity, "gpt-oss-120b-medium"),
        Some(131_072)
    );
    assert_eq!(check(Vendor::Muse, "muse-wonder-1.0"), Some(1_048_576));
    assert_eq!(check(Vendor::Muse, "muse-wonder-full-1.0"), Some(1_048_576));
    assert_eq!(check(Vendor::Copilot, "gpt-6.1-terra"), Some(922_000));
    assert_eq!(check(Vendor::Copilot, "claude-sonnet-5.5"), Some(1_000_000));
    assert_eq!(check(Vendor::Copilot, "mai-code-2"), Some(922_000));
    assert_eq!(check(Vendor::Copilot, "auto"), None);
}

#[test]
fn expand_home_resolves_tilde_prefix_only() {
    let home = Some(OsStr::new("/home/tester"));
    assert_eq!(expand_home_with(Path::new("agy"), home), Path::new("agy"));
    assert_eq!(
        expand_home_with(Path::new("/usr/bin/agy"), home),
        Path::new("/usr/bin/agy")
    );
    assert_eq!(
        expand_home_with(Path::new("~/.local/bin/agy"), home),
        Path::new("/home/tester/.local/bin/agy")
    );
    assert_eq!(expand_home_with(Path::new("~"), home), Path::new("~"));
    assert_eq!(
        expand_home_with(Path::new("~other/bin/agy"), home),
        Path::new("~other/bin/agy")
    );
    assert_eq!(
        expand_home_with(Path::new("~/.local/bin/agy"), Some(OsStr::new(""))),
        Path::new("~/.local/bin/agy")
    );
    assert_eq!(
        expand_home_with(Path::new("~/.local/bin/agy"), None),
        Path::new("~/.local/bin/agy")
    );
}

#[test]
fn enrich_reasoning_populates_expected_controls_across_bridge_vendors() {
    // Claude Code strips token budgets and `off` from catalog specs.
    let claude = enrich_reasoning(
        Vendor::ClaudeCode,
        "claude-sonnet-5-5",
        ReasoningSupport::Listed {
            levels: vec![Effort::Low, Effort::Medium, Effort::High, Effort::Max],
            budget: Some(1024..=65536),
            can_disable: true,
            default: Some(Effort::High),
        },
    );
    assert_eq!(
        claude,
        ReasoningSupport::Listed {
            levels: vec![Effort::Low, Effort::Medium, Effort::High, Effort::Max],
            budget: None,
            can_disable: false,
            default: Some(Effort::High),
        }
    );

    // Grok Build lists per-model levels and rejects `off`.
    let grok47 = enrich_reasoning(
        Vendor::GrokBuild,
        "grok-4.7",
        ReasoningSupport::Unknown { default: None },
    );
    assert_eq!(
        grok47.levels(),
        Some(&[Effort::Low, Effort::Medium, Effort::High, Effort::XHigh][..])
    );
    assert_eq!(grok47.default_effort(), Some(Effort::High));
    assert!(grok47.refusal(&Reasoning::Off).is_some());

    let grok45 = enrich_reasoning(
        Vendor::GrokBuild,
        "grok-4.5",
        ReasoningSupport::Unknown { default: None },
    );
    assert_eq!(
        grok45.levels(),
        Some(&[Effort::Low, Effort::Medium, Effort::High][..])
    );
    assert!(grok45.refusal(&Reasoning::Effort(Effort::XHigh)).is_some());

    // Muse lists minimal..max with High default.
    let muse = enrich_reasoning(
        Vendor::Muse,
        "muse-wonder-1.0",
        ReasoningSupport::Unknown { default: None },
    );
    assert_eq!(
        muse.levels(),
        Some(
            &[
                Effort::Minimal,
                Effort::Low,
                Effort::Medium,
                Effort::High,
                Effort::XHigh,
                Effort::Max
            ][..]
        )
    );
    assert_eq!(muse.default_effort(), Some(Effort::High));

    // Antigravity fixes effort in the model id and rejects separate `/effort` overrides.
    let agy = enrich_reasoning(
        Vendor::Antigravity,
        "gemini-4-neon-high",
        ReasoningSupport::Unknown { default: None },
    );
    assert_eq!(agy.levels(), Some(&[][..]));
    assert_eq!(agy.default_effort(), Some(Effort::High));
    assert!(agy.refusal(&Reasoning::Effort(Effort::Low)).is_some());
    assert!(agy.refusal(&Reasoning::Off).is_some());
}
