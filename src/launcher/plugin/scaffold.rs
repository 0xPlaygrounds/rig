//! The `__name__` plugin of the rig agent, made by `rig plugin new`.
//!
//! The plugin guide, rig-harness's `PLUGINS.md` (its `plugin_guide` docs), has
//! the rules every plugin follows and a short example of each extension point.
//! Every name it uses comes from `rig_harness::prelude` or the crate an example
//! names; the optional `inspect` tool (crate `rig-inspect`) lists the rest.

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
        // Other extension points, each a section of the plugin guide:
        // a tool ("A tool"), gating its calls ("Gating and reviewing tool
        // calls"), asking the user ("Asking the user"), model requests, a
        // turn's end ("How a turn ends"), settings, a `rig <name>` command
        // ("Commands of your own"), replacing a built-in ("Turning off or
        // replacing"), the status line and a terminal panel (crate rig-tui),
        // time ("Time"), and what each plugin added (`ProvidedBy`).
    }
}

/// `/__name__`: a notice with the word count of the agent's last answer.
/// `args.agent` is the agent. `NoArgs` refuses text after the command name.
fn command(
    In(args): In<NoArgs>,
    mut agents: Query<(&Conversation, &mut Uses)>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((conversation, mut uses)) = agents.get_mut(args.agent) else {
        return;
    };
    uses.0 += 1;
    // `Message::final_answer` is the text of a final answer, `None` for other messages.
    let answer = conversation
        .messages()
        .iter()
        .rev()
        .find_map(|message| message.final_answer());
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
