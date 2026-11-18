Writing a plugin for the rig agent: what a plugin is, how to make one, the
few ideas every plugin uses, and where to find a working example of each.

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a crate
that depends on `rig-harness`. The agent itself is made of plugins of the same
kind. Its tools, commands, subagents, compaction, sessions and terminal view
are plugins built on the same public API, and `[plugins]` in
`RIG_HOME/settings.toml` turns each on or off. A plugin that needs another
adds it unless `app.is_plugin_added::<P>()`.

`use rig_harness::prelude::*;` brings in Bevy's preludes, the kernel's
components and stages, and the harness's services. The default plugins live
in `rig-basics`, the terminal view in `rig-tui`, and `rig_core`,
`rig_memory` and `rig_tools` are re-exported from this crate. Every item
named here has its own docs, which hold the details this guide leaves out.

# Making one

```sh
rig plugin new hello       # a crate in RIG_HOME/plugins/hello, added to [[plugins.extra]]
rig plugin list            # every plugin, on or off, with what it adds
rig plugin disable NAME    # turn one off in [plugins]; `enable` undoes it
rig plugin check --build   # build the agent as /reload would, and check it
```

`rig plugin new` writes a crate with a working, commented slash command and
adds it to `[[plugins.extra]]` in `RIG_HOME/settings.toml`. Its `Cargo.toml`
already depends on the crates the agent builds with, at the versions it
builds with; add `rig-basics` or `rig-tui` the same way when you use their
types. `/reload` in a running agent rebuilds it with the plugins that are on
and restarts in the same session. A failed build keeps the running agent.

# Where the examples are

Read a working plugin before writing one:

- The [plugin corpus] holds many plugins written against this public API,
  one per module, each with its tests in `<plugin>/tests.rs`. CI builds,
  lints and tests every one, so they stay current.
- [`rig-basics`] holds the default plugins: the coding tools, approval,
  compaction, subagents, sessions, models, `--print`.
- The optional `rig-inspect` plugin adds an `inspect` tool that lists the
  reflected types of the running agent and what each plugin added.

The sections below name the ideas most plugins use, each with an example to
read.

# Effects and stages

Every step of an agent's work is an entity called an effect: a model call
([`ModelRequest`]), a tool call ([`ToolRequest`]), a message to an agent
([`Incoming`]), a question for the user (`Ask`), or a plugin's own
([`EffectRequest`]). Each frame the kernel moves effects through the stages of
the [`AgentPass`] schedule, in the order of [`Stage`]:

- `Prepare` edits a request once, as it is admitted;
- `Gate` holds, refuses or answers an effect that is about to start;
- `Start` runs what nobody held or answered;
- `Review` reads or edits an outcome before it is settled;
- `Arrive`, `Commit` and `Admit` are the kernel's own bookkeeping.

A plugin steps in with an ordinary Bevy system in a stage. A gate decides
through the effect's [`Decision`], and the first decision wins. A hold is any
entity with [`HoldOf`] the effect. A reviewer edits the [`Outcome`] in place.
An observer suits reactions that need no order, such as counting.

```rust,no_run
use rig_harness::prelude::*;

#[derive(Default)]
pub struct SrcOnlyPlugin;

impl Plugin for SrcOnlyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, src_only.in_set(Stage::Gate));
    }
}

/// Refuses a `write` outside `src/` when it is about to run.
fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (request, mut decision) in &mut calls {
        let path = request.call.function.arguments["path"].as_str().unwrap_or_default();
        if request.call.function.name.as_str() == "write" && !path.starts_with("src/") {
            decision.decide(Decider::plugin("src-only"), Outcome::refused("write only under src/"));
        }
    }
}
```

`Prepare`, `Gate` and `Review` have phases and public system sets, such as
[`PrepareSystems`], [`ReviewSystems`] and [`GateSystems`]. A system that edits
data another plugin also edits goes in a public `SystemSet` of its own,
configured in its phase, so others can order against it.

Examples: [confirm_destructive] (a gate that asks first),
[loop_breaker] (a gate that counts), [output_truncate] (a reviewer),
[few_shot_examples] (a `Prepare` edit of model requests).

# A tool

A tool is any `rig_core::tool::Tool`. Register it in `build`:

- `app.add_tool(tool)` for a tool that runs on its own;
- `app.add_tool_per_call(tool, options)` for one built for each call's place,
  its workspace and working directory ([`PerCallTool`]);
- `app.add_open_tool(name, description, options, observer)` for a tool the
  plugin answers itself, later, from an observer of `ToolCalled`.

Its description tells the model what it does and when to pick it. Its
footprint says what each call reads and writes, so calls on different things
run side by side. Blocking work goes in `blocking`.

Examples: [git_history_tools] and [sql_query_tool] (per-call tools),
[ask_user] (an open tool), [web_fetch] (a tool with a gate of its own).

# Asking the user

A question is an `Ask` effect: a title, a body, choices and whether text
answers it. A plugin that waits on the answer spawns the `Ask` as a hold on
what it asks about. A tool call the user should approve gets
`NeedsApproval`, which `rig_basics::approval` turns into one question. A
choice from a list is a `Picker`.

Examples: [ask_user], [confirm_destructive], [approval_batching].

# A slash command

A command is a one-shot system, `app.add_command(name, help, system)`, taking
`In<CommandArgs>` (the agent it was typed for and the text after its name) or
`In<NoArgs>` when it takes no text.

```rust,no_run
use rig_harness::prelude::*;

fn build(app: &mut App) {
    app.add_command("remind", "Remind the agent of something after its turn", remind);
}

/// `/remind <text>`: a message the agent reads once its turn would end.
fn remind(In(args): In<CommandArgs>, mut commands: Commands, mut notices: MessageWriter<Notice>) {
    if args.args.is_empty() {
        notices.write(Notice::error(args.agent, "/remind needs a text"));
        return;
    }
    let reminder = Deliver::new(args.agent, format!("Reminder: {}", args.args), DeliveryMode::Queue);
    commands.trigger(reminder.with_author(Author::plugin("remind")));
}
```

Examples: the scaffold `rig plugin new` writes, [session_goal].

# Messages, model requests and turns

What the user types is a [`Deliver`] that spawns an `Incoming` effect for its
agent, and so is what a plugin sends. A model call's [`ModelRequest`] holds
the system prompt in parts and the messages sent; a `Prepare` system edits it
or routes it to another model. A turn ends with a [`TurnOutcome`] the kernel
proposes and a `Review` system may edit, hold or carry on.

Examples: [model_router], [model_fallback], [auto_continue],
[check_after_turn].

# Settings and saved state

`app.add_setting::<T>("section")` in `build` reads `[section]` of
`RIG_HOME/settings.toml` into the resource `T`. A component that derives
`Reflect` and says `#[reflect(Component, Saved)]` is kept with the session,
across a restart, `/reload` and `/resume`; a resource says
`#[reflect(Resource, Saved)]`.

Examples: [loop_breaker] (a setting and saved state), [read_before_edit].

# Agents and subagents

Every agent is made by the kernel's [`Spawner`] from an [`Origin`] and an
[`AgentSpec`]. The `rig_basics::subagents` plugin spawns the agents of the
model's `task` tool and of plugins' `SpawnSubagent` effects.

Examples: [handoff], [map_reduce], [review_agent].

# The terminal view

`rig-tui` draws the transcript, the status line, panels and the input. A
plugin adds to it with ordinary systems: a look for a tool's calls in the
`TranscriptRender` schedule, an item on the status line, a panel, a key
binding. A crate that also builds without the view puts `rig-tui` behind a
default `tui` feature.

Examples: [rich_tool_rendering] (looks), [git_status_item] (a status item),
[todo_tracker] (a panel and a key), [keybindings].

# Fronts and commands of your own

A front is how the user drives the agent: the terminal view, `--print`, a
window. A plugin offers one with `app.add_front(..)`, and
`Front::new(name).on_command(..)` makes it a `rig <name>` command.

Examples: [usage_report_cli], [export], [window_gui].

# Testing a plugin

A plugin's test builds a small app with the kernel, the plugin and the
services it needs, then runs it frame by frame with `app.update()`. An effect
spawned for an agent stands in for what its model would ask; a whole turn
runs on rig-core's scripted `MockCompletionModel`.
[`stage_conflicts`](crate::stage_conflicts) lists the systems of one stage
that write the same data in no order, and `rig plugin check --build` runs the
same check on the plugins that are on.

```rust
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{ToolFunction, ToolName};
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::MemoryStore;

# struct SrcOnlyPlugin;
# impl Plugin for SrcOnlyPlugin {
#     fn build(&self, app: &mut App) {
#         app.add_systems(AgentPass, src_only.in_set(Stage::Gate));
#     }
# }
# fn src_only(mut calls: Query<(&ToolRequest, &mut Decision), (ReadyToStart, ToolCalls)>) {
#     for (request, mut decision) in &mut calls {
#         let path = request.call.function.arguments["path"].as_str().unwrap_or_default();
#         if request.call.function.name.as_str() == "write" && !path.starts_with("src/") {
#             decision.decide(Decider::plugin("src-only"), Outcome::refused("write only under src/"));
#         }
#     }
# }
let mut app = App::new();
app.insert_resource(SessionStore::new(MemoryStore::default()))
    .add_plugins((AgentPlugin, SessionLogPlugin, SrcOnlyPlugin))
    .add_open_tool("write", "Write a file.", ToolOptions::default(), |_: On<ToolCalled<serde_json::Value>>| {});
app.finish();
assert!(rig_harness::stage_conflicts(app.world_mut()).is_empty());
// The first frame restores the session and spawns the first agent.
app.update();
let agent = app.world_mut().query_filtered::<Entity, With<Agent>>().single(app.world());
let (Ok(agent), Ok(name)) = (agent, ToolName::new("write")) else {
    panic!("the kernel spawns an agent");
};
// The call its model would ask for.
let call = ToolCall::from_wire("c1", ToolFunction::new(name, serde_json::json!({ "path": "README.md" })));
let call = app.world_mut().spawn((ToolRequest { call }, ForAgent(agent))).id();
app.update();
let refused = app.world().get::<Outcome>(call).and_then(|outcome| outcome.0.as_ref().err());
assert!(refused.is_some_and(|why| why.to_string().contains("only under src/")));
```

Every corpus plugin's `tests.rs` is a further example.

[plugin corpus]: https://github.com/0xPlaygrounds/rig/tree/main/corpus
[`rig-basics`]: https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-basics
[`ModelRequest`]: crate::prelude::ModelRequest
[`ToolRequest`]: crate::prelude::ToolRequest
[`Incoming`]: crate::prelude::Incoming
[`EffectRequest`]: crate::prelude::EffectRequest
[`AgentPass`]: crate::prelude::AgentPass
[`Stage`]: crate::prelude::Stage
[`Decision`]: crate::prelude::Decision
[`HoldOf`]: crate::prelude::HoldOf
[`Outcome`]: crate::prelude::Outcome
[`PrepareSystems`]: crate::prelude::PrepareSystems
[`ReviewSystems`]: crate::prelude::ReviewSystems
[`GateSystems`]: crate::prelude::GateSystems
[`PerCallTool`]: crate::prelude::PerCallTool
[`Deliver`]: crate::prelude::Deliver
[`TurnOutcome`]: crate::prelude::TurnOutcome
[`Spawner`]: crate::prelude::Spawner
[`Origin`]: crate::prelude::Origin
[`AgentSpec`]: crate::prelude::AgentSpec
[confirm_destructive]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/shared/src/confirm_destructive.rs
[loop_breaker]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/loop_breaker.rs
[output_truncate]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/output_truncate.rs
[few_shot_examples]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/shared/src/few_shot_examples.rs
[git_history_tools]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/ct/src/git_history_tools.rs
[sql_query_tool]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/oc/src/sql_query_tool.rs
[ask_user]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/ask_user.rs
[web_fetch]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/sn/src/web_fetch.rs
[approval_batching]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/oc/src/approval_batching.rs
[session_goal]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/session_goal.rs
[model_router]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/model_router.rs
[model_fallback]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/model_fallback.rs
[auto_continue]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/oc/src/auto_continue.rs
[check_after_turn]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/check_after_turn.rs
[read_before_edit]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/ho/src/read_before_edit.rs
[handoff]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/handoff.rs
[map_reduce]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/map_reduce.rs
[review_agent]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/review_agent.rs
[rich_tool_rendering]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/fh/src/rich_tool_rendering.rs
[git_status_item]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/git_status_item.rs
[todo_tracker]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/todo_tracker.rs
[keybindings]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/keybindings.rs
[usage_report_cli]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/sn/src/usage_report_cli.rs
[export]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/export.rs
[window_gui]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/window_gui.rs
