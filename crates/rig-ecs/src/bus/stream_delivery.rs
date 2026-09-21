//! Live notifications for collected stream batches.
//!
//! ```
//! use bevy_ecs::prelude::*;
//! use rig_ecs::bus::StreamItemsDelivered;
//! fn observe(batch: On<StreamItemsDelivered>) {
//!     let item_count = batch.items.len();
//! }
//! ```

use bevy_ecs::prelude::*;
use rig_core::{effect::EffectId, error::ErrorReport, streaming::StreamEvent};

/// Newly collected items from one streaming effect, in their delivery order.
///
/// Observe with `On<StreamItemsDelivered>`. Each observer sees the same owned
/// batch after [`super::Streamed`] has been updated and before the collector
/// publishes [`super::EffectOutcome`]. Errors retain their positions among
/// events, including items accepted after the first terminal stream record.
/// Stream closure and run settlement are separate boundaries, not extra items.
/// Unary effects folded from a stream do not expose streamed consumer delivery.
///
/// Notifications are synchronous and have no retained subscriber queue. An
/// observer must return promptly; expensive work belongs outside its callback.
/// Independent observers do not drain each other's items. Late or re-enabled
/// observers receive only future batches: hydrate from existing `Streamed`
/// state before continuing the world. A checkpoint load never emits this event.
/// No observer ordering is promised. An observer may remove the owning graph;
/// the payload and id remain readable, but entity queries must handle absence.
/// Removing a sibling effect never suppresses that sibling's already accepted
/// batch: live collection and policy-visible replay both capture every batch
/// accepted in a pass, as an owned payload, before any observer runs.
///
/// Live batches follow collection limits and readiness. Policy-visible replay
/// preserves its recorded batches; ordinary cassette replay promises item order,
/// not identical network timing or collection grouping. Delivery does not depend
/// on installing a recorder or retaining event bytes in one.
///
/// ```
/// use bevy_ecs::prelude::*;
/// use rig_core::streaming::{Delta, StreamEvent};
/// use rig_ecs::bus::StreamItemsDelivered;
///
/// #[derive(Resource, Default)]
/// struct TextByEffect(std::collections::HashMap<Entity, String>);
/// let mut world = World::new();
/// world.init_resource::<TextByEffect>();
/// world.add_observer(
///     |delivery: On<StreamItemsDelivered>, mut text: ResMut<TextByEffect>| {
///         for item in &delivery.items {
///             if let Ok(StreamEvent::BlockDelta {
///                 delta: Delta::Text { text: piece }, ..
///             }) = item {
///                 text.0.entry(delivery.effect).or_default().push_str(piece);
///             }
///         }
///     },
/// );
/// // Late observers need prior text restored separately; notifications are not retained.
/// ```
#[derive(Event, Debug)]
pub struct StreamItemsDelivered {
    /// Effect entity that accepted these items; it may since have been removed.
    pub effect: Entity,
    /// Issued effect identity. Retries correlate through their distinct effects.
    pub id: EffectId,
    /// Zero-based item offset within this effect, counting events and errors.
    pub start: usize,
    /// Actual newly delivered items, preserving event/error interleaving.
    pub items: Vec<Result<StreamEvent, ErrorReport>>,
}
