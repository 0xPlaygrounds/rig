//! The `__name__` plugin of the rig agent, made by `rig plugin new`.
//!
//! The plugin guide, rig-harness's `PLUGINS.md` (its `plugin_guide` docs),
//! has a short copy-ready example of each extension point: a tool and how
//! its calls look, a slash command, state kept with the session, turn hooks,
//! a timer, what agents do and say in a terminal panel, and a window. Every
//! name they use comes from `rig_harness::prelude`; `inspect` shows the rest.

use rig_harness::prelude::*;

/// Listed in plugins.toml; the agent adds it with `Default`.
#[derive(Default)]
pub struct ScaffoldPlugin;

/// How often `/__name__` ran for an agent. A component that derives
/// `Reflect` and says `#[reflect(Component, Saved)]` is kept with the
/// session, across `/reload` and `/resume`.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct Uses(u32);

impl Plugin for ScaffoldPlugin {
    fn build(&self, app: &mut App) {
        // Every agent has its own count. `/__name__` runs `command` once per
        // use, for the agent it was typed for.
        app.register_required_components::<Agent, Uses>();
        app.add_command(
            "__name__",
            "Count the words of the agent's last answer",
            command,
        );
        // The other extension points, each in the plugin guide:
        // - an event command:   app.add_command_event::<MyEvent>("name", "help"), its fields
        //                       parsed from the arguments
        // - a tool:             app.add_tool(MyTool), a `PortableTool`
        // - after a turn:       app.add_observer(on_turn_ended), taking `On<TurnEnded>`
        // - on a timer:         app.add_systems(Update, tick.run_if(on_real_timer(Duration::from_secs(1))))
        //                       while an entity with KeepAwake(Duration::from_secs(1)) lives
        // - once, later:        commands.delayed().duration(Duration::from_secs(5)).trigger(..)
        // - a terminal panel:   spawn TuiPanel::new(Placement::Right(Constraint::Length(30)))
        // - what is loaded:     Query<(&Name, &PluginSource)>, and ProvidedBy(plugin) on what each added
    }
}

/// `/__name__`: a notice with the word count of the agent's last answer.
/// `args.agent` is the agent, `args.args` the text after the command name.
fn command(
    In(args): In<CommandArgs>,
    mut agents: Query<(&Conversation, &mut Uses)>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((conversation, mut uses)) = agents.get_mut(args.agent) else {
        return;
    };
    uses.0 += 1;
    // `final_answer` is the text of a final answer, `None` for other messages.
    let answer = conversation.messages().iter().rev().find_map(final_answer);
    let text = match answer {
        Some(answer) => format!(
            "The last answer has {} words (use {}).",
            answer.split_whitespace().count(),
            uses.0
        ),
        None => format!("No answer yet (use {}).", uses.0),
    };
    // A `Notice::error` refuses the command: the line goes back in the
    // input with it. To put a message in the agent's conversation instead,
    // trigger a `Deliver` (see the plugin guide).
    notices.write(Notice::info(args.agent, text));
}
