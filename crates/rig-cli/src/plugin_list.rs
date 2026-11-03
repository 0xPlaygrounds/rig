//! Which plugins a build holds and a session runs, as the settings say
//! ([`config`]): every one of rig's own and the home's extra crates are
//! built; the home's `[plugins]` and then the project's turn them on and
//! off when the agent starts, so that never needs a build.

use rig_harness_protocol::Home;
use rig_harness_protocol::config::{self, SettingsFile};

use super::Result;

pub use rig_harness_protocol::config::Plugins;

/// The plugins of the home's own settings, whose extra crates the build
/// holds.
pub fn of_home(home: &Home) -> Result<Plugins> {
    Ok(config::plugins(
        &SettingsFile::read(&home.settings())?,
        None,
    )?)
}

/// A short name for the list `on`, which keys what a build with those
/// plugins on says it adds.
pub fn key(on: &[String]) -> String {
    // FNV-1a: stable across runs and launchers, unlike std's hasher.
    let hash = (config::to_env(on).bytes()).fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests;
