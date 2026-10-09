//! Frames on a clock, for plugins that animate or poll: [`every`] is a run
//! condition that is true once per interval and keeps the loop awake for
//! it with [`Wake::after`], so no plugin needs a thread or a busy loop.
//!
//! ```
//! use std::time::Duration;
//! use rig_ecs::prelude::*;
//! use rig_ecs::timer::every;
//!
//! #[derive(Resource, Default)]
//! struct Spinner(u8);
//!
//! fn busy(agents: Query<(), With<ActiveTurn>>) -> bool {
//!     !agents.is_empty()
//! }
//!
//! fn spin(mut spinner: ResMut<Spinner>) {
//!     spinner.0 = spinner.0.wrapping_add(1);
//! }
//!
//! let mut app = App::new();
//! app.init_resource::<Spinner>()
//!     // `every` is checked only while an agent works, so an idle app sleeps.
//!     .add_systems(Update, spin.run_if(busy.and_then(every(Duration::from_millis(125)))));
//! ```

use std::time::Duration;

use bevy_ecs::prelude::*;
use web_time::Instant;

use crate::calls::Wake;

/// A run condition that is true on the first frame it is checked, then
/// once `interval` has passed since it was last true. While it is checked
/// it keeps a [`Wake::after`] pending for its next tick, so the loop runs
/// a frame then even when nothing else happens; once it is no longer
/// checked (behind a false condition in `a.and_then(every(..))`), at most one
/// more wake comes.
pub fn every(interval: Duration) -> impl FnMut(Res<Wake>) -> bool + Send + Sync + 'static {
    let mut next: Option<Instant> = None;
    let mut armed: Option<Instant> = None;
    move |wake: Res<Wake>| {
        let now = Instant::now();
        let due = next.is_none_or(|at| now >= at);
        let at = match next {
            Some(at) if !due => at,
            _ => now + interval,
        };
        next = Some(at);
        // One pending timer at a time: a new one only once the last fired.
        if armed.is_none_or(|armed| armed <= now) {
            wake.after(at.saturating_duration_since(now)).detach();
            armed = Some(at);
        }
        due
    }
}
