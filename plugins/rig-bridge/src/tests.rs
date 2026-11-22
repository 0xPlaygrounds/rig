use rig_harness::rig_core::catalog::{Catalog, ModelSpec};
use rig_harness::rig_core::providers::registry::ProviderId;

use crate::harness::Vendor;
use crate::plugin::enrich_limits;

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
    assert_eq!(check(Vendor::Copilot, "mai-code-2"), Some(200_000));
    assert_eq!(check(Vendor::Copilot, "auto"), None);
}
