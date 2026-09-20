//! Repository recording authorization. The portable cassette engine remains policy-free.

use anyhow::Context;
use rig_cassette::http::{
    CassetteMode, CassetteSpec, ProviderCassette as EngineSession, Transport,
};
use std::path::{Path, PathBuf};

/// An authorized repository session. Its constructor is private; successful
/// finalization emits the receipt required by the recording command.
pub struct ProviderCassette {
    session: EngineSession,
    recording: Option<String>,
    fixture_root: PathBuf,
}

impl ProviderCassette {
    /// The endpoint the provider client should use for this session.
    pub fn base_url(&self) -> String {
        self.session.base_url()
    }

    /// Resolve credentials only after this session has been authorized.
    pub fn api_key(&self, variable: &str) -> String {
        self.session.api_key(variable)
    }

    /// The engine's intentionally invalid key for provider-error cells.
    pub fn bogus_api_key(&self) -> String {
        self.session.bogus_api_key()
    }

    /// Direct transports can append exchanges, but cannot choose a write destination.
    pub fn direct_recorder(&self) -> Option<rig_cassette::http::DirectRecorder> {
        self.session.direct_recorder()
    }

    /// Preserve attempt evidence outside the fixture corpus, always under this
    /// session's own authorized identity. Checkpoints never promote captures.
    pub async fn checkpoint_attempt(&self, directory: &Path) -> anyhow::Result<Option<PathBuf>> {
        let Some(id) = &self.recording else {
            return Ok(None);
        };
        let path = directory.join(format!("{id}.yaml"));
        let destination = resolve_destination(&path)?;
        let corpus = resolve_destination(&self.fixture_root)?;
        anyhow::ensure!(
            !destination.starts_with(&corpus),
            "checkpoint destination {} is inside the fixture corpus",
            path.display()
        );
        Ok(self
            .session
            .checkpoint_recording(&path)
            .await
            .then_some(path))
    }

    /// Finalize the engine before acknowledging a committed capture.
    pub async fn finish(self) {
        self.session.finish().await;
        receipt(self.recording);
    }

    /// Preserve the test's original panic and acknowledge only successful finalization.
    pub async fn finish_after_test(self, result: std::thread::Result<()>) {
        self.session.finish_after_test(result).await;
        receipt(self.recording);
    }

    /// Preserve fallible tests' errors without acknowledging an unfinished capture.
    pub async fn finish_after_test_result<E>(
        self,
        result: std::thread::Result<Result<(), E>>,
    ) -> Result<(), E> {
        self.session.finish_after_test_result(result).await?;
        receipt(self.recording);
        Ok(())
    }
}

// Resolve existing symlink prefixes without creating the missing suffix. Reject
// traversal rather than interpreting it differently from the eventual writer.
fn resolve_destination(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    anyhow::ensure!(
        !absolute
            .components()
            .any(|part| part == std::path::Component::ParentDir),
        "checkpoint paths must not contain '..'"
    );
    let mut ancestor = absolute.as_path();
    let mut suffix = Vec::new();
    loop {
        match std::fs::symlink_metadata(ancestor) {
            Ok(_) => {
                let mut resolved = ancestor.canonicalize()?;
                for part in suffix.iter().rev() {
                    resolved.push(part);
                }
                return Ok(resolved);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                suffix.push(
                    ancestor
                        .file_name()
                        .context("checkpoint path has no existing ancestor")?
                        .to_os_string(),
                );
                ancestor = ancestor.parent().context("checkpoint path has no parent")?;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn receipt(recording: Option<String>) {
    if let Some(id) = recording {
        println!("RIG_CASSETTE_RECORDED={id}");
    }
}
#[doc(hidden)]
pub use rig_cassette_inventory as declarations;
pub use rig_cassette_inventory::{Capture, Scenario, Test};

/// Declare the recorded inputs of a scripted family beside its implementation.
#[macro_export]
macro_rules! scripted_family {
    ($visibility:vis const $name:ident: $provider:literal, $family:literal, [$($source:expr),* $(,)?]) => {
        $visibility const $name: $crate::recording::ScriptedFamily =
            $crate::recording::ScriptedFamily::new($provider, &[$($source),*]);
        $crate::recording::declarations::inventory::submit! {
            $crate::recording::declarations::FamilyRegistration(|| $crate::recording::declarations::Family {
                name: concat!($provider, "/", $family).into(),
                sources: vec![$(format!("{}/{}", $provider, $source)),*],
            })
        }
    };
}

/// Check this binary's declarations against its committed fixtures.
pub fn check_inventory(root: &Path, provider: &str) -> anyhow::Result<()> {
    rig_cassette_inventory::validate(
        &rig_cassette_inventory::snapshot(),
        root,
        &std::collections::BTreeSet::from([provider.to_owned()]),
    )?;
    Ok(())
}

/// A scripted test family's explicit fixture dependencies.
#[derive(Clone, Copy)]
pub struct ScriptedFamily {
    provider: &'static str,
    sources: &'static [&'static str],
}

impl ScriptedFamily {
    /// Declare the only recordings this family may borrow.
    pub const fn new(provider: &'static str, sources: &'static [&'static str]) -> Self {
        Self { provider, sources }
    }

    /// The provider whose recorded wire this family borrows.
    pub fn provider(&self) -> &'static str {
        self.provider
    }

    fn authorize(&self, scenario: &str) {
        assert!(
            self.sources.contains(&scenario),
            "{}/{scenario}: undeclared scripted source",
            self.provider
        );
    }

    /// Borrow SSE frames after checking the family's declaration.
    pub fn recorded_sse_frames(&self, scenario: &str, interaction: usize) -> Vec<String> {
        self.authorize(scenario);
        crate::stream_faults::recorded_sse_frames(self.provider, scenario, interaction)
    }

    /// Borrow status and body pairs after checking the family's declaration.
    pub fn recorded_statuses_and_bodies(&self, scenario: &str) -> Vec<(u16, String)> {
        self.authorize(scenario);
        crate::cassettes::recorded_statuses_and_bodies(self.provider, scenario)
    }

    /// Replay a declared source body under a scripted status.
    pub fn status_reply(
        &self,
        scenario: &str,
        status: u16,
        retry_after: bool,
    ) -> rig_agent::test_utils::MockHttpResponse {
        self.authorize(scenario);
        crate::stream_faults::status_reply(self.provider, scenario, status, retry_after)
    }
}

/// Optional exact scenario scope supplied by the recording planner.
pub use rig_cassette_inventory::RECORDING_SCOPE_ENV as SCOPE_ENV;

/// Reject unauthorized capture before entering the HTTP engine.
fn authorize(root: &Path, provider: &str, scenario: &str) -> anyhow::Result<()> {
    let id = format!("{provider}/{scenario}");
    let tests = rig_cassette_inventory::tests();
    let scope = std::env::var_os(SCOPE_ENV)
        .map(|scope| {
            let scope = scope
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("{SCOPE_ENV} is not UTF-8"))?;
            Ok::<std::collections::BTreeSet<String>, anyhow::Error>(serde_json::from_str(scope)?)
        })
        .transpose()?;
    rig_cassette_inventory::authorize(&tests, root, &id, scope.as_ref())?;
    Ok(())
}

/// Repository equivalent of the engine's proxy entrypoint.
pub async fn start(
    provider: &'static str,
    spec: impl Into<CassetteSpec>,
    base_url: &str,
) -> ProviderCassette {
    start_via(Transport::Proxy, provider, spec, base_url).await
}

/// Repository entrypoint shared by direct and proxy recordings.
pub async fn start_via(
    transport: Transport,
    provider: &'static str,
    spec: impl Into<CassetteSpec>,
    base_url: &str,
) -> ProviderCassette {
    start_mode(
        transport,
        &crate::cassettes::cassette_root(),
        provider,
        spec.into(),
        base_url,
        CassetteMode::current(),
    )
    .await
    .unwrap_or_else(|error| panic!("{error:#}"))
}

async fn start_mode(
    transport: Transport,
    root: &Path,
    provider: &'static str,
    spec: CassetteSpec,
    base_url: &str,
    mode: CassetteMode,
) -> anyhow::Result<ProviderCassette> {
    let destination = rig_cassette::http::cassette_path(root, provider, spec.scenario());
    if mode == CassetteMode::Record {
        authorize(root, provider, spec.scenario())?;
        let expected =
            resolve_destination(root)?.join(format!("{provider}/{}.yaml", spec.scenario()));
        anyhow::ensure!(
            resolve_destination(&destination)? == expected,
            "{} aliases another recording destination",
            destination.display()
        );
    }
    let recording =
        (mode == CassetteMode::Record).then(|| format!("{provider}/{}", spec.scenario()));
    let session =
        EngineSession::start_at(transport, provider, spec, base_url, mode, destination).await;
    Ok(ProviderCassette {
        session,
        recording,
        fixture_root: root.to_owned(),
    })
}

#[cfg(test)]
mod tests;
