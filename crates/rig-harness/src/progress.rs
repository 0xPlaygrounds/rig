//! How far a running call got, for every front: [`ToolProgress`], and the
//! output a running call's tool reports as it works ([`CallOutput`]).

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use crossbeam_channel::Receiver;
use rig_core::tool::{ToolErrorKind, ToolExecutionError, ToolResult};
use rig_ecs::stage::{AgentPass, IfStopped, Outcome, Stage};
use rig_ecs::tasks::Wake;
use rig_tools::{Live, OutputLimit};

/// A line about a running call's progress, such as `12 of 40 files` or a
/// command's last line of output, that the call's owner keeps current on
/// the call's entity. The terminal view draws it under the call, and
/// `--print` writes each change to stderr.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolProgress(pub String);

/// On a running call: the output its tool reports as it runs, through the
/// [`Live`] it was made with. The harness keeps the call's
/// [`ToolProgress`] (its last line) and its [`IfStopped`] (the output so
/// far, cut as tool output is cut, then `[stopped before it finished]`)
/// current from it, in [`Stage::Arrive`], so a call that is stopped answers
/// with what it printed. The `shell` tool reports this way; a tool of your
/// own does too:
///
/// ```
/// use rig_harness::prelude::*;
///
/// /// Gives each `build` call a `Live` its tool writes its output to.
/// fn follow_builds(
///     calls: Query<(Entity, &ToolRequest), (ToolCalls, Added<Admitted>)>,
///     wake: Res<Wake>,
///     mut commands: Commands,
/// ) {
///     for (call, request) in &calls {
///         if request.call.function.name.as_str() == "build" {
///             let (output, _live) = CallOutput::new(&wake);
///             commands.entity(call).insert(output);
///             // The call's handler, built with `_live` in `PrepareSystems::Edit`,
///             // calls `_live.line(..)` as it works.
///         }
///     }
/// }
/// ```
#[derive(Component, Debug)]
pub struct CallOutput {
    /// What the tool reported and was not read yet.
    lines: Receiver<String>,
    /// The output read so far, at most about twice what the cut keeps.
    kept: String,
    /// Lines left out before `kept`.
    cut_lines: usize,
}

impl CallOutput {
    /// A call's output, and the [`Live`] its tool reports into; each piece
    /// wakes the app.
    pub fn new(wake: &Wake) -> (Self, Live) {
        let (sender, lines) = crossbeam_channel::unbounded();
        let wake = wake.clone();
        let live = Live::new(move |text: &str| {
            if sender.send(text.to_owned()).is_ok() {
                wake.wake();
            }
        });
        let output = Self {
            lines,
            kept: String::new(),
            cut_lines: 0,
        };
        (output, live)
    }

    /// The output so far, cut as tool output is cut: its last lines, after
    /// a line saying how many earlier ones were left out.
    pub fn text(&self) -> String {
        let (kept, dropped) = OutputLimit::default().tail(&self.kept);
        match self.cut_lines + dropped {
            0 => kept.to_owned(),
            cut => format!("[{cut} earlier lines cut]\n{kept}"),
        }
    }

    /// Reads what came since the last read; whether anything did.
    fn read(&mut self) -> bool {
        let mut came = false;
        for piece in self.lines.try_iter() {
            self.kept.push_str(&piece);
            came = true;
        }
        let cut = OutputLimit::default();
        if self.kept.len() > 2 * cut.bytes {
            let (kept, dropped) = cut.tail(&self.kept);
            self.cut_lines += dropped;
            self.kept = kept.to_owned();
        }
        came
    }

    /// What the call answers if it is stopped now.
    fn if_stopped(&self) -> IfStopped {
        let mut text = self.text();
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("[stopped before it finished]");
        let stopped =
            ToolExecutionError::new(ToolErrorKind::Cancelled, "stopped before it finished")
                .with_model_feedback(text);
        IfStopped(Outcome::from(ToolResult::failed(stopped)))
    }
}

/// Follows each running call's [`CallOutput`].
pub(crate) struct ProgressPlugin;

impl Plugin for ProgressPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, follow_call_output.in_set(Stage::Arrive));
    }
}

/// Reads each [`CallOutput`] and, when output came, writes its call's
/// [`ToolProgress`] and [`IfStopped`].
fn follow_call_output(
    mut calls: Query<(Entity, &mut CallOutput), Without<Outcome>>,
    mut commands: Commands,
) {
    for (call, mut output) in &mut calls {
        if !output.read() {
            continue;
        }
        let last = output
            .kept
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty());
        let progress = ToolProgress(last.unwrap_or_default().trim_end().to_owned());
        let stopped = output.if_stopped();
        commands.entity(call).try_insert((progress, stopped));
    }
}

#[cfg(test)]
mod tests;
