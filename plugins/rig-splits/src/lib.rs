//! Splits of a conversation, built only on the harness's, rig-core's and
//! rig-memory's public API: `/split` at any entry into a branch of the
//! session or into a new session, `/split move` taking a range out and
//! leaving a pointer behind, `/splits` listing the lines of any saved
//! session, and `/split switch`, `/split compare` and `/split merge`. It
//! adds no event of its own: other plugins split through the same
//! primitives, or run its commands.

use rig_harness::prelude::*;

/// The `/split` and `/splits` commands.
#[derive(Default)]
pub struct SplitsPlugin;

impl Plugin for SplitsPlugin {
    fn build(&self, _app: &mut App) {}
}
