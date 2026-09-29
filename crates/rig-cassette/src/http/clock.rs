//! Wall time that replays: a recording session saves every reading its test
//! takes, and replay hands the same readings back in the same order.
//!
//! Code whose decisions depend on time (a cache that expires, a TTL chosen
//! from the gaps between calls) sends different requests when time differs,
//! and a replayed request must match its recording byte for byte. A test
//! gives such code a [`CassetteClock`] instead of the system clock.
//!
//! The readings live beside the fixture, in `<fixture>.clock.json`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::CassetteMode;

/// The longest pause [`super::ProviderCassette::pause`] accepts.
pub const MAX_PAUSE: std::time::Duration = std::time::Duration::from_secs(60);

#[derive(Default, Serialize, Deserialize)]
struct Readings {
    /// Unix seconds, in the order the test read them.
    readings: Vec<u64>,
}

struct ClockState {
    mode: CassetteMode,
    sidecar: PathBuf,
    readings: Mutex<Vec<u64>>,
    next: AtomicUsize,
}

/// A clock a cassette session records and replays. Clone it freely: every
/// clone reads the same sequence.
#[derive(Clone)]
pub struct CassetteClock {
    state: Arc<ClockState>,
}

impl std::fmt::Debug for CassetteClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CassetteClock")
            .field("sidecar", &self.state.sidecar)
            .field("mode", &self.state.mode)
            .finish_non_exhaustive()
    }
}

/// The readings file that belongs to the fixture at `cassette_path`.
pub fn clock_sidecar(cassette_path: &Path) -> PathBuf {
    cassette_path.with_extension("clock.json")
}

impl CassetteClock {
    /// The clock for the fixture at `cassette_path`. Replay loads the
    /// recorded readings; a fixture recorded without a clock replays none.
    pub(crate) fn start(mode: CassetteMode, cassette_path: &Path) -> Self {
        let sidecar = clock_sidecar(cassette_path);
        let readings = match mode {
            CassetteMode::Record => Vec::new(),
            CassetteMode::Replay => std::fs::read_to_string(&sidecar)
                .ok()
                .map(|text| {
                    serde_json::from_str::<Readings>(&text)
                        .unwrap_or_else(|error| {
                            panic!("clock readings {} should parse: {error}", sidecar.display())
                        })
                        .readings
                })
                .unwrap_or_default(),
        };
        Self {
            state: Arc::new(ClockState {
                mode,
                sidecar,
                readings: Mutex::new(readings),
                next: AtomicUsize::new(0),
            }),
        }
    }

    /// Unix seconds. Recording reads wall time and saves it; replay returns
    /// the next saved reading and panics when the test asks for more
    /// readings than were recorded.
    pub fn now(&self) -> u64 {
        let mut readings = self
            .state
            .readings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.state.mode {
            CassetteMode::Record => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_secs());
                readings.push(now);
                now
            }
            CassetteMode::Replay => {
                let index = self.state.next.fetch_add(1, Ordering::SeqCst);
                match readings.get(index) {
                    Some(reading) => *reading,
                    None => panic!(
                        "clock reading #{} requested, but {} recorded only {}: the code under \
                         test reads time differently from its recording; re-record the fixture",
                        index + 1,
                        self.state.sidecar.display(),
                        readings.len()
                    ),
                }
            }
        }
    }

    /// Wait `duration` while recording, so a live provider sees the pause;
    /// return at once on replay. Panics above [`MAX_PAUSE`] (60 s): no
    /// cassette test waits longer.
    pub async fn pause(&self, duration: std::time::Duration) {
        assert!(
            duration <= MAX_PAUSE,
            "a cassette pause of {duration:?} exceeds the {MAX_PAUSE:?} limit"
        );
        if self.state.mode == CassetteMode::Record {
            tokio::time::sleep(duration).await;
        }
    }

    /// Save the readings beside the fixture (recording) or check that
    /// replay used every one of them.
    pub(crate) async fn finish(&self) {
        let readings = self
            .state
            .readings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        match self.state.mode {
            CassetteMode::Record => {
                if readings.is_empty() {
                    let _ = tokio::fs::remove_file(&self.state.sidecar).await;
                    return;
                }
                let text = serde_json::to_string_pretty(&Readings { readings })
                    .unwrap_or_else(|error| panic!("clock readings should serialize: {error}"));
                tokio::fs::write(&self.state.sidecar, text + "\n")
                    .await
                    .unwrap_or_else(|error| {
                        panic!(
                            "clock readings {} should be writable: {error}",
                            self.state.sidecar.display()
                        )
                    });
            }
            CassetteMode::Replay => {
                let used = self.state.next.load(Ordering::SeqCst);
                assert!(
                    used >= readings.len(),
                    "replay read {used} of the {} clock readings in {}: the code under test \
                     reads time differently from its recording; re-record the fixture",
                    readings.len(),
                    self.state.sidecar.display()
                );
            }
        }
    }
}
