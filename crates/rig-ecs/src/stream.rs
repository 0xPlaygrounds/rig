//! Per-effect cursors for append-only bus text. Raw events, errors, and usage
//! remain on the original [`Streamed`] component for applications needing them.

use crate::bus::Streamed;
use bevy_ecs::{entity::EntityHashMap, prelude::*};

/// The stored text offset is invalid after an application replaced a stream.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("text for effect {entity:?} was replaced; forget its cursor before reading it again")]
pub struct TextReplaced {
    /// The affected effect.
    pub entity: Entity,
}

/// Independent text positions for any number of interleaved effect entities.
///
/// This follows the bus's append-only text contract. Applications replacing
/// `Streamed.text` must call [`forget`](Self::forget), even when the replacement
/// has the same byte length. Call it also when an effect is removed, to bound
/// cursor storage. Entity generations keep reused indices separate.
#[derive(Debug, Default)]
pub struct StreamText {
    offsets: EntityHashMap<usize>,
}

impl StreamText {
    /// Borrow text appended since the previous read of this effect.
    /// Never slices through a UTF-8 code point. Invalid offsets are reported
    /// without advancing the cursor; prefix rewrites are not detected.
    pub fn read<'a>(
        &mut self,
        entity: Entity,
        stream: &'a Streamed,
    ) -> Result<&'a str, TextReplaced> {
        let offset = self.offsets.get(&entity).copied().unwrap_or(0);
        let text = stream.text.get(offset..).ok_or(TextReplaced { entity })?;
        self.offsets.insert(entity, stream.text.len());
        Ok(text)
    }

    /// Forget a removed effect or reset after explicitly replacing its text.
    pub fn forget(&mut self, entity: Entity) {
        self.offsets.remove(&entity);
    }
}
