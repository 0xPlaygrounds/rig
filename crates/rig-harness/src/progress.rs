//! How far a running call got, for every front: [`ToolProgress`].

use bevy_ecs::prelude::*;

/// A line about a running call's progress, such as `12 of 40 files` or a
/// command's last line of output, that the call's owner keeps current on
/// the call's entity. The terminal view draws it under the call, and
/// `--print` writes each change to stderr.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolProgress(pub String);
