//! Why a provider cassette session could not start or finish.

use std::path::{Path, PathBuf};

/// A failure reported by [`ProviderCassette::try_start_at`] or
/// [`ProviderCassette::try_finish`]. Every variant carries the fixture path of
/// its session, also available through [`CassetteError::path`]. The display
/// text is the message [`ProviderCassette::start_at`] and
/// [`ProviderCassette::finish`] panic with.
///
/// [`ProviderCassette::try_start_at`]: super::ProviderCassette::try_start_at
/// [`ProviderCassette::try_finish`]: super::ProviderCassette::try_finish
/// [`ProviderCassette::start_at`]: super::ProviderCassette::start_at
/// [`ProviderCassette::finish`]: super::ProviderCassette::finish
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CassetteError {
    /// The provider base URL does not parse.
    #[error("invalid provider base URL {url:?}: {source}")]
    InvalidBaseUrl {
        /// The session's fixture path.
        path: PathBuf,
        /// The URL as given.
        url: String,
        /// Why it does not parse.
        source: url::ParseError,
    },
    /// Replay found no fixture to serve.
    #[error(
        "missing provider cassette {}; run with RIG_PROVIDER_TEST_MODE=record and the real API key to create it",
        .path.display()
    )]
    MissingFixture {
        /// The fixture path replay looked for.
        path: PathBuf,
    },
    /// The fixture exists but cannot be read.
    #[error("provider cassette {} should be readable: {source}", .path.display())]
    UnreadableFixture {
        /// The fixture path.
        path: PathBuf,
        /// The read failure.
        source: std::io::Error,
    },
    /// The fixture is not a stream of cassette interactions.
    #[error("provider cassette {} should deserialize: {source}", .path.display())]
    MalformedFixture {
        /// The fixture path.
        path: PathBuf,
        /// The YAML error.
        source: serde_yaml::Error,
    },
    /// A fixture interaction holds a body or response the replay server
    /// cannot serve.
    #[error(
        "provider cassette {} interaction {index} cannot be replayed: {reason}",
        .path.display()
    )]
    InvalidInteraction {
        /// The fixture path.
        path: PathBuf,
        /// The interaction's position in the fixture, from zero.
        index: usize,
        /// What is wrong with it.
        reason: String,
    },
    /// The recorded clock readings beside the fixture do not parse.
    #[error("clock readings {} should parse: {source}", .sidecar.display())]
    MalformedClock {
        /// The fixture path.
        path: PathBuf,
        /// The readings file.
        sidecar: PathBuf,
        /// The JSON error.
        source: serde_json::Error,
    },
    /// A local replay server or recording relay could not listen.
    #[error("cassette server for {} should bind: {source}", .path.display())]
    Bind {
        /// The fixture path.
        path: PathBuf,
        /// The socket error.
        source: std::io::Error,
    },
    /// Replay left recorded interactions unplayed or refused requests the
    /// fixture does not hold. At least one list is non-empty.
    #[error("{}", replay_mismatch_message(.path, .unused_interactions, .unexpected_requests))]
    ReplayMismatch {
        /// The fixture path.
        path: PathBuf,
        /// Each unplayed interaction as `[index] METHOD path`.
        unused_interactions: Vec<String>,
        /// A diagnostic for each refused request, in arrival order.
        unexpected_requests: Vec<String>,
    },
    /// Replay used fewer clock readings than the recording took.
    #[error(
        "replay read {used} of the {recorded} clock readings in {}: the code under test \
         reads time differently from its recording; re-record the fixture",
        .sidecar.display()
    )]
    UnusedClockReadings {
        /// The fixture path.
        path: PathBuf,
        /// The readings file.
        sidecar: PathBuf,
        /// How many readings replay handed out.
        used: usize,
        /// How many readings the recording holds.
        recorded: usize,
    },
    /// The recording captured no exchange, or its export failed.
    #[error(
        "provider cassette {} should contain at least one interaction",
        .path.display()
    )]
    EmptyRecording {
        /// The fixture path.
        path: PathBuf,
    },
    /// The recording holds an undeclared account failure or stored state it
    /// never deleted, so it did not become the fixture.
    #[error(
        "provider cassette {} was not written: {}\nthe recording was kept at {}",
        .path.display(),
        .refusals.join("; "),
        .kept_at.as_deref().map_or_else(
            || "<nowhere: write failed>".to_owned(),
            |kept| kept.display().to_string()
        )
    )]
    RecordingRefused {
        /// The fixture path, left untouched.
        path: PathBuf,
        /// Why the recording was refused.
        refusals: Vec<String>,
        /// Where the scrubbed recording was kept under the attempt root, or
        /// `None` when that write failed.
        kept_at: Option<PathBuf>,
    },
    /// Scrubbing left secrets or other unsafe material in the recording, so
    /// it was not written anywhere.
    #[error(
        "provider cassette {} still contains unsafe artifacts after scrubbing:\n{}",
        .path.display(),
        .failures.join("\n")
    )]
    UnsafeRecording {
        /// The fixture path, left untouched.
        path: PathBuf,
        /// One diagnostic per unsafe finding.
        failures: Vec<String>,
    },
    /// The scrubbed recording could not be written to the fixture path.
    #[error("provider cassette {} should be written: {source}", .path.display())]
    WriteFixture {
        /// The fixture path.
        path: PathBuf,
        /// The write failure.
        source: std::io::Error,
    },
    /// `RIG_CASSETTE_SNAPSHOTS` holds a value other than `off`, `check` or
    /// `write`.
    #[error(
        "RIG_CASSETTE_SNAPSHOTS must be off, check or write; got {value:?} (replaying {})",
        .path.display()
    )]
    InvalidSnapshotMode {
        /// The fixture path.
        path: PathBuf,
        /// The value as set.
        value: String,
    },
    /// The request snapshot beside the fixture cannot be read, does not
    /// parse, or does not apply to the fixture's recorded requests.
    #[error(
        "request snapshot {} cannot be used: {reason}; rewrite it with RIG_CASSETTE_SNAPSHOTS=write",
        .snapshot.display()
    )]
    InvalidSnapshot {
        /// The fixture path.
        path: PathBuf,
        /// The snapshot file.
        snapshot: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// Replay received requests that differ from the fixture's request
    /// snapshot.
    #[error(
        "requests replayed from {} differ from their snapshot {}:\n{}\nif the change is intended, \
         rewrite the snapshot with RIG_CASSETTE_SNAPSHOTS=write and review its diff",
        .path.display(),
        .snapshot.display(),
        .differences.join("\n")
    )]
    SnapshotMismatch {
        /// The fixture path.
        path: PathBuf,
        /// The snapshot file.
        snapshot: PathBuf,
        /// One readable difference per differing request, in arrival order.
        differences: Vec<String>,
    },
    /// The request snapshot could not be written or removed.
    #[error("request snapshot {} should be writable: {source}", .snapshot.display())]
    WriteSnapshot {
        /// The fixture path.
        path: PathBuf,
        /// The snapshot file.
        snapshot: PathBuf,
        /// The write failure.
        source: std::io::Error,
    },
    /// The recorded clock readings could not be written beside the fixture.
    #[error("clock readings {} should be writable: {source}", .sidecar.display())]
    ClockWrite {
        /// The fixture path.
        path: PathBuf,
        /// The readings file.
        sidecar: PathBuf,
        /// The write failure.
        source: std::io::Error,
    },
}

impl CassetteError {
    /// The fixture path of the session that failed.
    pub fn path(&self) -> &Path {
        match self {
            Self::InvalidBaseUrl { path, .. }
            | Self::MissingFixture { path }
            | Self::UnreadableFixture { path, .. }
            | Self::MalformedFixture { path, .. }
            | Self::InvalidInteraction { path, .. }
            | Self::MalformedClock { path, .. }
            | Self::Bind { path, .. }
            | Self::ReplayMismatch { path, .. }
            | Self::UnusedClockReadings { path, .. }
            | Self::EmptyRecording { path }
            | Self::RecordingRefused { path, .. }
            | Self::UnsafeRecording { path, .. }
            | Self::WriteFixture { path, .. }
            | Self::ClockWrite { path, .. }
            | Self::InvalidSnapshotMode { path, .. }
            | Self::InvalidSnapshot { path, .. }
            | Self::SnapshotMismatch { path, .. }
            | Self::WriteSnapshot { path, .. } => path,
        }
    }
}

fn replay_mismatch_message(
    path: &Path,
    unused_interactions: &[String],
    unexpected_requests: &[String],
) -> String {
    let mut failures = Vec::new();
    if !unused_interactions.is_empty() {
        failures.push(format!(
            "left unused interactions:\n{}",
            unused_interactions.join("\n")
        ));
    }
    if !unexpected_requests.is_empty() {
        let numbered = unexpected_requests
            .iter()
            .enumerate()
            .map(|(index, diagnostic)| format!("[{index}] {diagnostic}"))
            .collect::<Vec<_>>()
            .join("\n");
        failures.push(format!(
            "received unexpected replay request(s):\n{numbered}"
        ));
    }
    format!(
        "provider cassette replay failed for {}:\n{}",
        path.display(),
        failures.join("\n\n")
    )
}
