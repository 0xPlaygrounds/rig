# One settings file, and plugins per project

A self-contained task for one agent run, which ends in one PR. You own it from start to finish: design, implementation, tests, a check in the real app, the PR, CI and review. You may change anything that makes the result better, including tests, internal APIs, protocols and docs. Read `AGENTS.md` and `DEVELOPING.md` first; they bind you. Don't build prototypes or spikes. Commit in the repo's conventional style, with `!` for a breaking change. No Co-Authored-By or any AI attribution in commits, the PR or comments. Never merge the PR.

## What this builds on

Two earlier PRs come first. Start from a `main` that has both, and read what they did before designing:
- **The launcher has its own crate.** Today the launcher (`src/launcher/`) is part of the published `rig` facade package. It reads `plugins.toml` with a hand-written TOML subset (`src/launcher/plugin_list.rs`), and the facade has no `toml` dependency. Any TOML library you need (`toml_edit` or the like) goes in the launcher's crate. Never add it to the `rig` facade package, because every library user would then build it.
- **One project root.** `ProjectRoot` (`crates/rig-harness/src/services.rs`) is the one folder that every reader of a project's files uses: `.rig/settings.toml`, `.rig/mcp.json`, `AGENTS.md` and skills. Use it to find a project's `.rig/settings.toml`; don't search for the project in a new way.

If either is missing from `main`, stop and say so in your report instead of doing that work here.

## Why

`RIG_HOME` (default `~/.rig`) holds two files that decide what rig runs:
- `plugins.toml` says which plugins are built in, as `[[plugin]]` tables;
- `settings.toml` says how they behave.

`plugins.toml` is a full copy of rig's default list (`src/launcher/plugins.toml`, written out by `PluginList::load` in `src/launcher/plugin_list.rs`). The copy is made once and never updated:
- a plugin rig adds to its defaults later stays off in every existing home, until the user notices it in `rig plugin list` and enables it by hand;
- a plugin rig renames or drops leaves a stale entry behind.

There are no per-project plugins. One home has one plugin list and one build.

A project's `.rig/settings.toml` doesn't layer every table over the home's. It merges only the tables a plugin registers with `add_project_setting` (`crates/rig-harness/src/settings.rs`, `apply_project_setting`). Today only plugins in `corpus/` call it; no shipped plugin does. So a project's settings file changes nothing for a user of rig's own plugins today.

## The outcome

### 1. One file the user edits: `settings.toml`

The plugin list moves into `settings.toml` and holds only the user's changes to rig's built-in default list:

```toml
[plugins]
enable  = ["rig_basics::budget::BudgetPlugin"]
disable = ["rig_basics::skills::SkillsPlugin"]

[[plugins.extra]]            # a plugin crate that is not one of rig's
path = "plugins/my-plugin"   # or git or version, as [[plugin]] allows today
crate = "my-plugin"
plugin = "my_plugin::MyPlugin"
```

The names and shape are yours to improve. These properties are fixed:
- a new default plugin reaches every home without the user doing anything;
- nothing the user didn't write is copied into their file;
- an unknown plugin name is refused, naming it.

**Delete `RIG_HOME/plugins.toml` outright,** with no reader for the old file and no migration code. Rig is pre-release. The migration note goes in the PR body under `## Migration`.
- The built-in default list stays one source of truth in rig, as `src/launcher/plugins.toml` is today. Whether it stays a TOML file compiled in or becomes code is your call.
- `test-support/rig-defaults-test-support/tests/removal.rs` already checks that its `DEFAULTS` and `OPTIONAL` lists match `src/launcher/plugins.toml`, in order, leaving out `rig_steel::SteelPlugin`, which those lists don't hold. Keep that check, pointed at the new single source. Better still, derive those lists from it so there is no copy to check.
- The comment beside each plugin there stays visible to the user through `rig plugin list`.

**`rig plugin enable/disable/list/new`, and `/plugins` through them, edit `settings.toml`** and keep the user's comments and layout.

**The rebuild is said.** The help text and docs say which changes to `[plugins]` rebuild rig and which don't. Changing any other key never does.

**These files stay as they are:**
- `defaults.json`: the last model and effort chosen, which rig writes once a turn on a newly chosen model is answered. It is state, not config.
- `auth/`: credentials, which keep their own file permissions.
- `mcp.json`: the common `mcpServers` format that other tools share, edited by `/mcp add` and `/mcp remove`.
- `/mcp approve` and `/mcp deny`, and the per-server approval in `plugins/rig-mcp-client/src/approval.rs`. They stay the guard on a project's MCP servers.
- `sessions/`, `history.jsonl` and the other data.

### 2. Plugins per project: on and off only

**A project's `.rig/settings.toml` takes `[plugins]` with `enable` and `disable` only,** on top of the home's list. They name rig's own plugins, the ones a `[[plugin]]` entry without `path`, `git` or `version` builds today (`src/launcher/plugin_list.rs`).

**A project can't add crates.** `extra` in a project's file is refused, in one line naming the file, and rig doesn't start. The reason: rig asks nothing before it reads a cloned repository's files, and building a crate compiles and runs its code (its `build.rs`). Only the home's `settings.toml` takes `extra`.

**The home's own folder is not a project.** Started in `~` with the default `RIG_HOME`, the project's `.rig/settings.toml` is the home's `~/.rig/settings.toml` (`ProjectRoot` is the working folder, and the project read appends `.rig/settings.toml`). Read that file once, as the home's, so its `extra` is accepted and its tables aren't applied twice. Compare the two paths after resolving them, not as strings.

**One build for every project.** Every one of rig's own plugins, on or off by default, is compiled into the home's build, along with the home's `extra` crates. The agent reads the on/off list when it starts: rig's defaults, then the home's `[plugins]`, then the project's. So a project's list never needs a build of its own, and starting in a new project is as fast as anywhere else.
- Some plugins exist only on some targets: `rig-claude-cloak` builds on Linux and macOS only (`plugins/rig-claude-cloak/src/lib.rs`). The build holds the plugins available on its target, and enabling one that isn't is refused, naming the plugin and the target.
- **Check first** whether the harness can leave out a plugin that is compiled in without side effects: no systems, resources, observers, settings, commands or tools of it remain, and nothing it registers at build time leaks. `crates/rig-harness/src/load.rs` records what each plugin added, and `plugins/rig-basics/src/reload/plugins.rs` is `/plugins`, which shows and toggles the list. Write down what you find in the PR.
- **If it can't,** make the harness able to, in the general way: deciding which plugins to add is the launcher's and agent's job, and a plugin needs no code of its own to be skippable. If that can't be done cleanly, stop, write down why, and report it as the first item under "Choices left to the owner" instead of building another design.
- Don't build one build per plugin set instead. It is the owner's alternative, and your report should give its cost if you stop: these paths are fixed per home and would each need keying by set: `Home::project()`; `Home::target()` with its `debug/rig-harness-agent`; `bin/good`, `bin/staged` and `bin/rejected`; `build.log`, `project/Cargo.lock.checked` and `bin/lock` (`crates/rig-harness-protocol/src/lib.rs`, `src/launcher/build.rs`). Helpers rebuild from `RIG_HOME`'s list (`compile_logged` in `src/launcher/build.rs`), so each would need its parent's set handed to it.

**Children follow their parent.** A `helper` child (`rig -p`) or a subagent runs with the plugin list of the session that started it.

**The project's list comes last.** If the home disables a plugin and the project enables it, the project wins: it is rig's own code either way. The docs say so.

## Tests

Each behaviour gets a test, following the repo's existing pattern. Each test fails on main where main has the behaviour. At least:
- a home with no `[plugins]` runs rig's default list;
- `enable` and `disable` change it, and an unknown name is refused, naming it;
- `rig plugin enable` keeps the user's comments;
- a project's `disable` leaves that plugin out of a session started there, and its `enable` puts an optional one in, with no rebuild;
- a project's `extra` is refused, naming the file, and nothing from it is built;
- started in the folder that holds `RIG_HOME`, the home's `extra` is accepted and its file is read once;
- a `helper` child of a session runs with that session's list;
- the defaults check in `removal.rs` reads the new single source.

## The real app

Use a disposable `RIG_HOME`, never your real one.
- **A fresh home:** no `plugins.toml` is written, and `rig plugin list` shows the defaults with their comments.
- **Enabling a plugin:** `rig plugin enable rig_basics::budget::BudgetPlugin` on a commented `settings.toml` keeps the comments, and on the next start `/budget` works.
- **A scratch project folder** whose `.rig/settings.toml` disables `rig_basics::agents_md::AgentsMdPlugin`, beside an `AGENTS.md` asking the model to end every answer with a marker word:
  - started there, rig doesn't rebuild, and the marker word doesn't appear;
  - started in a sibling folder with the same `AGENTS.md`, the marker word does appear.
- **The same folder with an `extra` table** in its `.rig/settings.toml`: rig refuses to start, naming the file, and builds nothing.

## The PR

- **What changed for a user,** with before and after for the file format (an old `plugins.toml` beside the new `[plugins]` table) and for `rig plugin …` output.
- **`## Changelog` and `## Migration`,** following `.github/PULL_REQUEST_TEMPLATE/other.md`.
- **A capability-changes table:** `plugins.toml` goes, and what replaces each of its uses.
- **What you found** about leaving out a compiled-in plugin, and what you changed for it.
- **Line deltas:** production, inline-test and test-file deltas separately, and the public item count before and after.
- **An interface sweep:** update every doc, help text, README and test harness that names `plugins.toml` (`git grep -n plugins.toml`), including `crates/rig-harness/PLUGINS.md`, `docs/agent.md` and `src/launcher/help-agents.txt`. Leave the generated `CHANGELOG.md` and `MIGRATING.md` files alone. Say what outside harnesses must change.
- **Choices left to the owner:** anything that removes a capability this file doesn't name, with your recommendation.
- **What you didn't verify.**
