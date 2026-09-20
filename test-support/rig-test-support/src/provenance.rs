//! Declared provenance for this repository's provider cassette scenarios.
//!
//! `crates/rig-cassette/fixtures/scenarios.yaml` authorizes source declarations:
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
    manifest::Manifest::parse(include_str!(concat!(env!("OUT_DIR"), "/scenarios.json")))
        .unwrap_or_else(|error| panic!("cassette scenario declarations: {error}"))
});

/// Provider directories whose fixtures must be scanned for secrets.
pub fn registered_providers() -> Vec<String> {
    DECLARATIONS
        .providers
        .iter()
        .map(|p| p.provider.clone())
        .collect()
}

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
            } else if provider
                .scripted
                .iter()
                .any(|s| s.fixture.as_deref() == Some(scenario))
            {
                Some(Provenance::Scripted)
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
    let provenance = scenario_provenance(provider, spec.scenario());
    if rig_cassette::http::CassetteMode::current() == rig_cassette::http::CassetteMode::Record
        && provenance == Provenance::Live
    {
        assert!(
            recording_scope_allows(
                std::env::var_os(manifest::RECORD_SCOPE_ENV).as_deref(),
                provider,
                spec.scenario()
            ),
            "{provider}/{} is outside the recording plan",
            spec.scenario()
        );
        assert!(
            DECLARATIONS.provider(provider).is_some_and(|p| p.recordable(&crate::cassettes::workspace_root(), spec.scenario())),
            "{provider}/{} has no capture; explicitly declare it unrecorded before recording",
            spec.scenario()
        );
    }
    spec.with_provenance(provenance)
}

fn recording_scope_allows(scope: Option<&std::ffi::OsStr>, provider: &str, scenario: &str) -> bool {
    scope.is_none_or(|scope| {
        scope.to_str().is_some_and(|ids| {
            ids.split(',')
                .any(|id| id == format!("{provider}/{scenario}"))
        })
    })
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

    /// Recorded unary replies this family is allowed to rewrite.
    pub fn recorded_statuses_and_bodies(&self, scenario: &str) -> Vec<(u16, String)> {
        self.assert_declared_source(scenario);
        crate::cassettes::recorded_statuses_and_bodies(self.provider, scenario)
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
