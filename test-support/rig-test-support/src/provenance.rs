//! Declared provenance for this repository's provider cassette scenarios.
//!
//! `crates/rig-cassette/fixtures/scenarios.json` is the single declaration:
//! it says which committed recordings came from a real provider (`live`),
//! which were deliberately modified from one of those (`derived`), and which
//! response or transport behaviour is constructed in code (`scripted`). The
//! engine refuses to live-record anything but a `Live` scenario, so the
//! lookup here is what turns a scenario literal in a test into a recordable
//! one. A scenario nobody declared stays unrecordable.

use std::sync::LazyLock;

use rig_cassette::http::{CassetteSpec, Provenance};

mod manifest;

#[cfg(test)]
mod tests;

/// The declaration of one scripted fault family: the module that scripts
/// provider behaviour in code and the recordings it is allowed to borrow
/// bytes from.
#[derive(Debug)]
pub struct ScriptedFamily {
    provider: &'static str,
    declaration: &'static manifest::ScriptedFamilyDecl,
}

static DECLARATIONS: LazyLock<manifest::Manifest> = LazyLock::new(|| {
    manifest::Manifest::load(&crate::cassettes::workspace_root())
        .unwrap_or_else(|error| panic!("cassette scenario declarations: {error}"))
});

/// The declared provenance of one provider scenario. An undeclared scenario
/// is a failure, not a live one: the declaration is what authorizes a live
/// capture.
pub fn scenario_provenance(provider: &str, scenario: &str) -> Provenance {
    DECLARATIONS
        .provider(provider)
        .and_then(|provider| {
            if provider.live_scenarios().contains(&scenario) {
                Some(Provenance::Live)
            } else if provider
                .derived
                .iter()
                .any(|entry| entry.scenario == scenario)
            {
                Some(Provenance::Derived)
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            panic!(
                "{provider}/{scenario} has no entry in {}; declare it as live, derived or \
                 scripted before using it",
                manifest::MANIFEST_PATH
            )
        })
}

/// The spec a cassette session runs under, carrying the provenance declared
/// for its scenario.
pub fn declared_spec(provider: &str, spec: impl Into<CassetteSpec>) -> CassetteSpec {
    let spec = spec.into();
    spec.with_provenance(scenario_provenance(provider, spec.scenario()))
}

impl ScriptedFamily {
    /// The declared scripted family of one test module. Panics when the
    /// module scripts provider behaviour nobody declared.
    pub fn new(provider: &'static str, family: &'static str) -> Self {
        let declaration = DECLARATIONS
            .provider(provider)
            .and_then(|provider| provider.scripted.iter().find(|entry| entry.family == family))
            .unwrap_or_else(|| {
                panic!(
                    "scripted family {provider}/{family} is not declared in {}; declare the family, \
                     its sources and the behaviour it injects",
                    manifest::MANIFEST_PATH
                )
            });
        Self {
            provider,
            declaration,
        }
    }

    /// The provider whose wire this family scripts.
    pub fn provider(&self) -> &'static str {
        self.provider
    }

    /// The recorded SSE frames of a declared source recording, in wire order.
    pub fn recorded_sse_frames(&self, scenario: &str, interaction: usize) -> Vec<String> {
        self.assert_declared_source(scenario);
        crate::stream_faults::recorded_sse_frames(self.provider, scenario, interaction)
    }

    /// A declared source recording's body, replayed under `status`.
    pub fn status_reply(
        &self,
        scenario: &str,
        status: u16,
        retry_after: bool,
    ) -> rig_agent::test_utils::MockHttpResponse {
        self.assert_declared_source(scenario);
        crate::stream_faults::status_reply(self.provider, scenario, status, retry_after)
    }

    fn assert_declared_source(&self, scenario: &str) {
        let qualified = format!("{}/{scenario}", self.provider);
        let sources = &self.declaration.sources;
        assert!(
            sources.iter().any(|source| source == &qualified),
            "scripted family {}/{} borrows {qualified}, which is not one of its declared \
             sources {sources:?}; declare the source in {}",
            self.provider,
            self.declaration.family,
            manifest::MANIFEST_PATH
        );
    }
}
