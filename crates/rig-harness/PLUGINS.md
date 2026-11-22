Writing a plugin for the rig agent: what a plugin is, how to make one, the
few ideas every plugin uses, and where to find a working example of each.

A plugin is a Bevy plugin: a type implementing `Plugin + Default` in a crate
that depends on `rig-harness`. The agent itself is made of plugins of the same
kind. Its tools, commands, subagents, compaction, sessions and terminal view
are plugins built on the same public API, and `[plugins]` in
`RIG_HOME/settings.toml` turns each on or off. A plugin that needs another
loads it in its `build` with `rig_harness::load::<P>`, which adds it unless
the app has it and records which plugin added it; a plugin added any other
way runs, but `/plugins` cannot see it. A plugin useless without
another checks `app.is_plugin_added::<Other>()` in its `finish`, once every
plugin is built, and adds nothing there without it; its settings are still
read in `build`, the only place a plugin may read them.

The kernel, `rig-ecs`, runs agents and their work; this crate, the harness,
adds sessions, settings, commands and fronts (how a user drives the agent).
`use rig_harness::prelude::*;` brings in Bevy's preludes, the kernel's
components and stages, and the harness's services. The default plugins live
in `rig-basics`, the terminal view in `rig-tui`, and `rig_core`,
`rig_memory` and `rig_tools` are re-exported from this crate. Every item
named here has its own docs, which hold the details this guide leaves out.

# Making one

```sh
rig plugin new hello       # a crate in RIG_HOME/plugins/hello, turned on
rig plugin list            # the tree of crates, areas and plugins, on or off
rig plugin disable NAME    # turn a plugin or branch off; `enable` undoes it
rig plugin check --build   # build the agent as /reload would, and check it
```

`rig plugin new` writes a crate with a working, commented slash command and
adds it to `[[plugins.extra]]` in `RIG_HOME/settings.toml`. Its `Cargo.toml`
already depends on `rig-harness`, Bevy's ECS and the serde and schemars crates
a tool needs, at the versions the agent builds with, and its comments say how
to add `rig-basics` or `rig-tui` when you use their types. `/reload` in a
running agent rebuilds it with the plugins that are on and restarts in the
same session; a failed build keeps the running agent.

# Where a plugin sits

Plugins form a tree of crates, areas and plugins. A plugin's area is the first
module under its crate in its type path:
`rig_basics::sessions::forks::ForksPlugin` is in `rig-basics/sessions`, and
`rig_basics::BasicCommandsPlugin`, directly under the crate root, is in none.
Nothing declares an area, so plugins that belong together go in one module.

`[plugins] off = ["rig-basics/sessions"]` turns a crate or an area off: the
agent loads none of its plugins, whatever their own choice in `enable` and
`disable`. A plugin that another one adds in its own `build` still runs. Each
keeps that choice, so turning the branch back on brings it back, and a plugin
added to an off branch later stays off. A name in `off` that is no crate or
area refuses the start, as an unknown plugin does. `/plugins` shows the same
tree, folded. Enter on a branch puts it into `off` or takes it out; on a
branch whose plugins are all off by their own choice, it turns each one on.

# Where the examples are

- The [plugin corpus] holds many plugins written against this public API,
  one per module, each with its tests in `<plugin>/tests.rs`. CI builds,
  lints and tests every one, so they stay current.
- [`rig-basics`] holds the default plugins: the coding tools, approval,
  compaction, subagents, sessions, models, `--print`.
- The optional `rig-inspect` plugin adds an `inspect` tool that lists the
  reflected types of the running agent and what each plugin added.

# Effects and stages

Every step of an agent's work is an entity called an effect: a model call
([`ModelRequest`]), a tool call ([`ToolRequest`]), a message to an agent
([`Incoming`]), a question for the user (`Ask`), or a plugin's own
([`EffectRequest`]). Each frame the kernel moves effects through the stages of
the [`AgentPass`] schedule, one [`Stage`] after another, until nothing moves:

- `Arrive`: finished work becomes an [`Outcome`];
- `Review`: plugins read or edit an outcome before it is settled;
- `Commit`: the kernel settles what is answered and records it;
- `Admit`: the kernel admits new effects;
- `Prepare`: plugins edit a request once, as it is admitted;
- `Gate`: plugins hold, refuse or answer an effect that is about to start;
- `Start`: the kernel runs what nobody held or answered.

A plugin steps in with an ordinary Bevy system in a stage. A gate decides
through the effect's [`Decision`], and the first decision wins. A hold is any
entity with [`HoldOf`] the effect. A reviewer edits the [`Outcome`] in place.
An observer suits reactions that need no order, such as counting. Observers
of one event run in no fixed order, so an observer never edits stage data
another plugin edits; it may decide an effect, since the first decision wins.

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
        let function = &request.call.function;
        let path = function.arguments.get("path").and_then(|path| path.as_str());
        if function.name.as_str() == "write" && !path.is_some_and(|path| path.starts_with("src/")) {
            decision.decide(Decider::plugin("src-only"), Outcome::refused("write only under src/"));
        }
    }
}
```

`Prepare`, `Gate` and `Review` have phases and public system sets, such as
[`PrepareSystems`], [`ReviewSystems`] and [`GateSystems`]. A system that edits
data another plugin also edits goes in a public `SystemSet` of its own,
configured in its phase, so others can order against it. Order against
another plugin's public set only where you must, never against a private
system. Two plugins of other authors never depend on each other only to order
themselves: where two meant to be used together edit the same data in one
phase, the later one depends on the earlier one's crate, orders against its
set, and tests both in one app.

Examples: [src_only_writes] (the plugin above, with its test),
[confirm_destructive] (a gate that asks first), [loop_breaker] (a gate that
counts), [output_truncate] (a reviewer), [few_shot_examples] (a `Prepare` edit).

# A tool

A tool is any `rig_core::tool::Tool`. Register it in `build`:

- `app.add_tool(tool)` for a tool that runs on its own;
- `app.add_tool_per_call(tool, options)` for one built for each call's place,
  its workspace and working directory ([`PerCallTool`]), as a tool that reads
  the project's files is;
- `app.add_open_tool(name, description, options, observer)` for a tool the
  plugin answers itself, later, from an observer of `ToolCalled`.

Its description tells the model what it does and when to pick it. Its
footprint says what each call reads and writes, so calls on different things
run side by side; a tool whose annotations say it only reads needs nothing
more. Blocking work goes in [`blocking`].

Examples: [word_count] (a small per-call tool, with a test that runs a scripted
model's call), [git_history_tools] and [sql_query_tool] (larger per-call tools),
[ask_user] (an open tool), [web_fetch] (a tool with a gate of its own).

# Asking the user

A question is an `Ask` effect: a title, a body, choices and whether text
answers it. A plugin that waits on the answer spawns the `Ask` as a hold on
what it asks about. A tool call the user should approve gets
`NeedsApproval`, which `rig_basics::approval` turns into one question. A
choice from a list is a `Picker`. An item's `PickItem::parent` makes the list
a tree, shown folded. In a picker of several answered with its ticks, an
unticked branch switches its items off, and none of their commands runs; an
item picked by itself runs its command.

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
or routes it to another model. Until Start, a request the kernel built has its
system prompt in `prompt`, a [`SystemPrompt`] of [`Section`]s, and its messages
in `history`: edit those there. When they are `None`, a plugin brought the
request whole, and `request_mut()` edits them; it edits every other field,
such as `max_tokens`, either way. A turn ends with a [`TurnOutcome`] the kernel
proposes and a `Review` system may edit, hold or carry on.

Examples: [model_router], [model_fallback], [auto_continue], [check_after_turn].

# Settings and saved state

`app.add_setting::<T>("section")` in `build` reads `[section]` of
`RIG_HOME/settings.toml` into the resource `T`. A component that derives
`Reflect` and says `#[reflect(Component, Saved)]` is kept with the session,
across a restart, `/reload` and `/resume`; a resource says
`#[reflect(Resource, Saved)]`.

Examples: [loop_breaker] (a setting and saved state), [read_before_edit].

# Agents and subagents

Every agent is made by the kernel from an [`Origin`] and an [`AgentSpec`], in
a system through a [`Spawner`]. The `rig_basics::subagents` plugin spawns the
agents of the model's `task` tool and of plugins' `SpawnSubagent` effects.

Examples: [handoff], [map_reduce], [review_agent].

# The terminal view

`rig-tui` draws the transcript, the status line, panels and the input. A
plugin adds to it with ordinary systems and entities:

- a look for some transcript items, a system in the `TranscriptRender`
  schedule that writes their lines;
- a status line item, shown in `StatusSystems::Show`;
- a panel, a `TuiPanel` entity drawn in `TuiSystems::Draw`;
- a key binding, `app.add_key_binding(binding, observer)`.

The mouse comes in with the keys, in `rig_tui::PendingInput`, whose
documentation says what the view does with it. A plugin that takes a drag
itself removes those events in an `InputSystems::Filter` system.

The line under the input is a plugin of its own,
`rig_tui::status::StatusLinePlugin`. Its `[status] hide` setting hides items
by name when the line is drawn; the items stay in `StatusItems` for other
readers. A plugin declares each part it draws, with a short label, in its
`build`, so `/plugins` can show and hide it before it is first drawn:
`app.add_status_part(&PINNED, "Pinned notes")` (`rig_tui::AppStatusPartsExt`),
`PINNED` being the part's `StatusItem`. A part drawn without one shows in
`/plugins` by its name once it is drawn. The key list over an empty input is
a plugin of its own too, `rig_tui::hints::KeyHintsPlugin`, and so is the
sticky prompt bar that pins the most recent off-screen user message over the
top row of a transcript pane, `rig_tui::sticky::StickyPromptPlugin`. Code
that adds `rig_tui::TuiPlugin` itself gets the view without them;
`rig_tui::TuiPlugins` adds the view with its optional parts, each unless the
app already has it.

The window title and the OSC 133 prompt marks around each turn come from
`rig_tui::title::TerminalTitlePlugin`, which works only with `TuiPlugin`;
`[title] mark_nonce_env` names the variable holding the shell's mark nonce.
An app that lists its own plugins adds `rig_tui::TuiPlugins` for the view
with both.

A crate that also builds without the view puts `rig-tui` behind a default
`tui` feature. The `rig_tui` crate docs have an example of each.

The bars before the working agent's status come from
`rig_basics::activity::animation::WorkingAnimationPlugin`. They move unless
`[activity] animate = false`, and with that plugin off there are none. A view
that draws its own working sign reads its `Pulse` resource to move with
rig's; the resource exists only while that plugin runs in the terminal view.

Examples: [rich_tool_rendering] (looks), [git_status_item] and [throughput_item]
(status items), [todo_tracker] (a panel and a key), [keybindings].

# Turning off or replacing what another plugin added

A whole plugin is replaced in `settings.toml`: name it in `[plugins] disable`
and add yours. Tools, slash commands, key bindings, options, fronts and prompt
sections are entities: Bevy's `Disabled` on one, inserted in
`Plugin::finish`, turns it off and frees its name for a replacement. Changing
how another plugin's tool runs is rarely a replacement: a `Prepare` or `Gate`
system on its calls does it, and a `Review` system changes its results.

Examples: [keep_last_compaction] (replaces the default compaction),
[submit_result] (a tool that starts off), [shell_rewrite] (edits another tool's
calls).

# Fronts and commands of your own

A front is how the user drives the agent: the terminal view, `--print`, a
window. A plugin offers one with `app.add_front(..)`, and
`Front::new(name).on_command(..)` makes it a `rig <name>` command.

Examples: [usage_report_cli], [export], [window_gui].

# Testing a plugin

A plugin's test builds a small app with the kernel (`AgentPlugin` and
`SessionLogPlugin`, on a `SessionStore` of a rig-memory `MemoryStore`), the
plugin and the services it needs, then runs it frame by frame with
`app.update()`. An effect spawned for an agent stands in for what its model
would ask; a whole turn runs on rig-core's scripted `MockCompletionModel`
(its `test-utils` feature).
[`stage_conflicts`](crate::stage_conflicts) lists the systems of one stage
that write the same data in no order, and `rig plugin check --build` runs the
same check on the plugins that are on.

Examples: [src_only_writes] tests a gate with no model, in its
[`tests.rs`][src_only_writes tests]; [word_count]'s test runs a scripted
model's tool call to its end. Corpus tests use the repository's unpublished
`rig-harness-test-support` crate for scripted models and for running until
something happens; copy what you need from it.

[plugin corpus]: https://github.com/0xPlaygrounds/rig/tree/main/corpus
[`rig-basics`]: https://github.com/0xPlaygrounds/rig/tree/main/plugins/rig-basics
[`ModelRequest`]: crate::prelude::ModelRequest
[`SystemPrompt`]: rig_core::completion::SystemPrompt
[`Section`]: rig_core::completion::Section
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
[`blocking`]: crate::prelude::blocking
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
[throughput_item]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/ho/src/throughput_item.rs
[keep_last_compaction]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/ct/src/keep_last_compaction.rs
[submit_result]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/ho/src/submit_result.rs
[shell_rewrite]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/shell_rewrite.rs
[src_only_writes]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/src_only_writes.rs
[src_only_writes tests]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/src_only_writes/tests.rs
[word_count]: https://github.com/0xPlaygrounds/rig/blob/main/corpus/base/src/word_count.rs
