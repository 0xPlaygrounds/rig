//! The `__name__` plugin of the rig agent, made by `rig plugin new`.
//!
//! The plugin guide, rig-harness's `PLUGINS.md` (its `plugin_guide` docs),
//! has a short copy-ready example of each extension point: a slash command,
//! a tool and how its calls look, a terminal panel, a timer, reading what
//! agents do and say, state kept with the session, and a window. Every name
//! they use comes from `rig_harness::prelude`.

use rig_harness::prelude::*;

/// Listed in plugins.toml; the agent adds it with `Default`.
#[derive(Default)]
pub struct ScaffoldPlugin;

impl Plugin for ScaffoldPlugin {
    fn build(&self, app: &mut App) {
        // `/__name__` runs `command` once per use, for the agent it was typed for.
        app.add_command(
            "__name__",
            "Count the words of the agent's last answer",
            command,
        );
        // The other extension points, each in the plugin guide:
        // - a tool:             app.add_tool(MyTool), a `PortableTool`
        // - after a turn:       app.add_observer(on_turn_ended), taking `On<TurnEnded>`
        // - on a timer:         app.add_systems(Update, tick.run_if(on_real_timer(Duration::from_secs(1))))
        //                       while an entity with KeepAwake(Duration::from_secs(1)) lives
        // - once, later:        commands.delayed().duration(Duration::from_secs(5)).trigger(..)
        // - saved state:        #[reflect(Component, Saved)] on a component deriving Reflect
        // - a terminal panel:   spawn TuiPanel::new(Placement::Right(Constraint::Length(30)))
    }
}

/// `/__name__`: a notice with the word count of the agent's last answer.
/// `args.agent` is the agent, `args.args` the text after the command name.
fn command(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    // `final_answer` is the text of a final answer, `None` for other messages.
    let answer = agents
        .get(args.agent)
        .ok()
        .and_then(|conversation| conversation.messages().iter().rev().find_map(final_answer));
    let text = match answer {
        Some(answer) => format!(
            "The last answer has {} words.",
            answer.split_whitespace().count()
        ),
        None => "No answer yet.".to_owned(),
    };
    // `Notice::error` reports a failure. To put a message in the agent's
    // conversation instead, trigger a `Deliver` (see the plugin guide).
    notices.write(Notice::info(args.agent, text));
}
