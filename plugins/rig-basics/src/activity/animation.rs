//! The working animation: while the shown agent works, six bars around a
//! gap like rig's mark stand before its status in the status line, and the
//! mark in the welcome pulses with them. They move only while the terminal
//! view is the run's front; `animate = false` in the `[activity]` table of
//! settings.toml keeps them still. Without this plugin the status has no
//! bars and the mark stays still.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins((
//!     rig_basics::activity::ActivityPlugin,
//!     rig_basics::activity::animation::WorkingAnimationPlugin,
//! ));
//! ```

#[cfg(feature = "tui")]
use super::{Activity, ActivitySystems, works};
use rig_harness::prelude::*;

/// The bars before the working agent's status and the welcome mark's
/// pulse; nothing without the `tui` feature, nor without the
/// [`ActivityPlugin`](super::ActivityPlugin) and the
/// [`TuiViewPlugin`](rig_tui::TuiViewPlugin), added before it or after.
#[derive(Default)]
pub struct WorkingAnimationPlugin;

impl Plugin for WorkingAnimationPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "tui")]
        app.add_setting::<AnimationSettings>("activity");
        #[cfg(not(feature = "tui"))]
        let _ = app;
    }

    // In `finish`, after every plugin's `build`, so the order they were
    // added in does not matter.
    #[cfg(feature = "tui")]
    fn finish(&self, app: &mut App) {
        if app.is_plugin_added::<super::ActivityPlugin>()
            && app.is_plugin_added::<rig_tui::TuiViewPlugin>()
            && app.runs_front(rig_tui::FRONT)
        {
            animate(app);
        }
    }
}

/// `[activity]` in `settings.toml`.
#[cfg(feature = "tui")]
#[derive(Resource, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AnimationSettings {
    /// Whether the bars and rig's mark move.
    animate: bool,
}

#[cfg(feature = "tui")]
impl Default for AnimationSettings {
    fn default() -> Self {
        Self { animate: true }
    }
}

/// How long each step of the working animation lasts.
#[cfg(feature = "tui")]
pub(crate) const STEP: Duration = Duration::from_millis(120);

/// The steps of one wave.
#[cfg(feature = "tui")]
pub const PERIOD: usize = 12;

/// How high a bar stands (0 to 7) at each step of its wave.
#[cfg(feature = "tui")]
const WAVE: [u8; PERIOD] = [0, 0, 1, 3, 5, 7, 7, 5, 3, 1, 0, 0];

/// How high each still bar stands, from the outer pair in: a little
/// higher toward the middle, as the mark's shades are stronger there.
#[cfg(feature = "tui")]
const STILL: [u8; 3] = [3, 4, 5];

/// The working animation, while [`WorkingAnimationPlugin`] runs and the
/// terminal view is the run's front: a view that draws its own working
/// sign reads it to move with rig's.
#[cfg(feature = "tui")]
#[derive(Resource, Default)]
pub struct Pulse(Option<usize>);

#[cfg(feature = "tui")]
impl Pulse {
    /// The animation's step, out of [`PERIOD`], while the shown agent works
    /// and `animate` is on; `None` while it stands still.
    pub fn step(&self) -> Option<usize> {
        self.0
    }
}

/// The entity that keeps the loop awake for the animation's steps.
#[cfg(feature = "tui")]
#[derive(Component)]
struct Pulsing;

/// The animation's [`Pulse`] and its steps: what the terminal view's
/// front adds.
#[cfg(feature = "tui")]
pub(crate) fn animate(app: &mut App) {
    app.init_resource::<Pulse>().add_systems(
        PostUpdate,
        (pace, step.run_if(on_real_timer(STEP)))
            .chain()
            .after(ActivitySystems)
            .before(rig_tui::StatusSystems::Show),
    );
}

/// Starts the animation, with one [`KeepAwake`] for its steps, when the
/// shown agent starts to work, and stops it when the agent stops.
#[cfg(feature = "tui")]
fn pace(
    shown: Res<rig_harness::main_agent::ShownAgent>,
    agents: Query<&Activity>,
    settings: Res<AnimationSettings>,
    awake: Query<Entity, With<Pulsing>>,
    mut pulse: ResMut<Pulse>,
    mut commands: Commands,
) {
    let activity = shown.0.and_then(|agent| agents.get(agent).ok());
    let busy = settings.animate && activity.is_some_and(|activity| works(&activity.status));
    match (busy, awake.iter().next()) {
        (true, None) => {
            commands.spawn((Name::new("activity animation"), Pulsing, KeepAwake(STEP)));
            pulse.0 = Some(0);
        }
        (false, Some(entity)) => {
            commands.entity(entity).despawn();
            pulse.0 = None;
        }
        _ => {}
    }
}

/// Moves the animation a step.
#[cfg(feature = "tui")]
fn step(mut pulse: ResMut<Pulse>) {
    if let Some(step) = pulse.0 {
        pulse.0 = Some((step + 1) % PERIOD);
    }
}

/// How high (0 to 7) the bar of pair `pair`, from the outer pair in,
/// stands at animation step `step`: a wave running out from the gap.
/// With no step it stands still.
#[cfg(feature = "tui")]
pub(crate) fn level(pair: usize, step: Option<usize>) -> u8 {
    let level = match step {
        Some(step) => WAVE.get((step + 2 * pair) % PERIOD),
        None => STILL.get(pair),
    };
    level.copied().unwrap_or_default()
}

/// Six bars around a gap like rig's mark, their heights at animation step
/// `step`. Height, not color, shows the step, so it moves under `NO_COLOR`.
#[cfg(feature = "tui")]
pub(crate) fn wave(step: Option<usize>) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let bar = |pair| {
        let level = usize::from(level(pair, step));
        BARS.get(level).copied().unwrap_or_default()
    };
    let left: String = (0..STILL.len()).map(bar).collect();
    let right: String = (0..STILL.len()).rev().map(bar).collect();
    format!("{left} {right}")
}

#[cfg(test)]
mod tests;
