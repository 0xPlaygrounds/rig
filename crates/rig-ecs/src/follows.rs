//! Agent state that follows the line of its conversation: a component
//! reflected with [`ReflectFollows`] is kept in the agent's conversation, so
//! every split, switch, move, merge, fork and restore gives the agent the
//! value its line had at that point, and a sibling line never sees it.
//!
//! A value set on a line is kept as an aside of kind [`FOLLOWS`] by rig,
//! following the line's last entry, at most one per key and entry. While
//! the line has no entry of its own since it parted from a kept branch, the
//! value is one of the line's own values (rig-core's
//! `Conversation::set_line_value`), which travel with the line as its title
//! does, and is anchored after the line's first entry of its own.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::{CreateTypeData, Reflect, TypePath};

/// The kind of the asides holding followed values:
/// `{"key": <type path>, "v": n, "value": …}`, by rig.
pub const FOLLOWS: &str = "rig.follows";

/// Keeps an agent component with the line of the agent's conversation it was
/// set on: derive `Reflect` and add `Follows` (`#[reflect(Component,
/// Follows)]`, or opaque with serde as `Saved` allows). A split, a switch, a
/// move, a merge, a fork and a restore give the agent the value its line
/// had at that point; with none set on the line, the component is removed.
/// Versioned like a saved value. It takes the place of `Saved` for its type.
#[derive(Clone)]
pub struct ReflectFollows {
    pub(crate) watch: fn(&mut App),
}

impl<T: Component + Reflect + TypePath> CreateTypeData<T> for ReflectFollows {
    fn create_type_data(_: ()) -> Self {
        Self { watch: |_| {} }
    }
}
