# One settings file, plugins per project, and "do you trust this folder?"

A self-contained task for one agent run, which ends in one PR. You own it from start to finish: design, implementation, tests, a check in the real app, the PR, CI and review. You may change anything that makes the result better, including tests, internal APIs, protocols and docs. Read `AGENTS.md` and `DEVELOPING.md` first; they bind you. Don't build prototypes or spikes. Commit in the repo's conventional style, with `!` for a breaking change. No Co-Authored-By or any AI attribution in commits, the PR or comments. Never merge the PR.

## Why

`RIG_HOME` (default `~/.rig`) holds two files that decide what rig runs:
- `plugins.toml` says which plugins are built in;
- `settings.toml` says how they behave.

`plugins.toml` is a full copy of rig's default list (`src/launcher/plugins.toml`, written out by `PluginList::load` in `src/launcher/plugin_list.rs`). The copy is made once and never updated:
- a plugin rig adds to its defaults later stays off in every existing home, until the user notices it in `rig plugin list` and enables it by hand;
- a plugin rig renames or drops leaves a stale entry behind.

There are no per-project plugins. One home has one plugin list and one build, so a project's `.rig/settings.toml` can change settings but not which plugins run.

A folder's own files change what rig does as soon as rig starts there, with nothing asked:
- its `.rig/settings.toml`;
- its `AGENTS.md` files, from the folder up to the repository root, read by `plugins/rig-basics/src/agents_md.rs`;
- its `.rig/skills` and `.agents/skills`, read by `plugins/rig-basics/src/skills.rs`.

The `AGENTS.md` files and skills go into the model's prompt, so a cloned repository can inject instructions. Only the folder's MCP servers (`.rig/mcp.json`) are guarded today: each is approved one by one, in `plugins/rig-mcp-client/src/approval.rs`.

## The outcome

### 1. One file the user edits: `settings.toml`

The plugin list moves into `settings.toml` and holds only the user's changes to rig's built-in default list:

```toml
[plugins]
enable  = ["rig_basics::budget::BudgetPlugin"]
disable = ["rig_basics::skills::SkillsPlugin"]

[[plugins.extra]]            # a plugin crate that is not one of rig's
path = "plugins/my-plugin"   # or git/crate, as plugins.toml allows today
plugin = "my_plugin::MyPlugin"
```

The names and shape are yours to improve. These properties are fixed:
- a new default plugin reaches every home without the user doing anything;
- nothing the user didn't write is copied into their file;
- an unknown plugin name is refused, naming it.

**Delete `RIG_HOME/plugins.toml` outright,** with no reader for the old file and no migration code. Rig is pre-release. The migration note goes in the PR body under `## Migration`.
- The built-in default list stays one source of truth in rig, as `src/launcher/plugins.toml` is today. Whether it stays a TOML file compiled in or becomes code is your call.
- The comment beside each plugin there stays visible to the user through `rig plugin list`.

**`rig plugin enable/disable/list/new` edit `settings.toml`** and keep the user's comments and layout (with `toml_edit` or the like).

**The rebuild is said.** Changing `[plugins]` changes the build, and changing any other key doesn't. The help text and docs say so.

**These files stay as they are:**
- `defaults.json`: the last model and effort chosen, which rig writes on every `/model`. It is state, not config.
- `auth/`: credentials, which keep their own file permissions.
- `mcp.json`: the common `mcpServers` format that other tools share, edited by `/mcp add` and `/mcp remove`.
- `sessions/`, `history.jsonl` and the other data.

### 2. Plugins per project

- **A project's `.rig/settings.toml` takes the same `[plugins]` table** (`enable`, `disable`, `extra`) on top of the home's, the same way its other tables already layer: see the project read in `crates/rig-harness/src/settings.rs`.
- **One build per distinct plugin set,** named by a hash of that set. A project with the home's set uses the home's build.
  - Builds share what can safely be shared, so a second set compiles only the crates that differ.
  - Two rigs starting at once with different sets must not corrupt or wrongly reuse each other's output (see `src/launcher/locks.rs`).
  - Old builds that no set uses any more aren't cleaned up automatically unless that stays simple. Say what you did.
- **The first start of a new set says it is building,** as the home build does today.
- **Children follow their parent's set.** A `helper` child (`rig -p`) or a subagent runs with the plugin set of the session that started it, with no second build and no second trust question.

### 3. "Do you trust this folder?": yes or no, on by default

Before designing, read how two other agents do it, and don't copy either:
- pi: `packages/coding-agent/src/core/project-trust.ts` and `trust-manager.ts` in https://github.com/badlogic/pi-mono;
- codex: `TrustLevel` in `codex-rs/core/src/config/mod.rs` in https://github.com/openai/codex.

**What needs trust:** everything a folder brings that changes what rig runs, does, or tells the model:
- its `.rig/settings.toml`, every table;
- its plugins, `extra` crates above all, since building one compiles and runs its code (`build.rs`);
- its MCP servers (`.rig/mcp.json`). Folder trust replaces the per-server approval in `plugins/rig-mcp-client/src/approval.rs`, so there is one rule for a folder's files, not two. List the removed `/mcp approve` and `/mcp deny` in the PR's capability-changes table.
- the prompt text it brings: `AGENTS.md` files, skills, and any other project file rig puts into the prompt without being asked. The reason is prompt injection. Untrusted, none of it reaches the model.

**What doesn't:**
- The user's own files (in `RIG_HOME`, or the user-level skills directory) are always trusted.
- A file the user or the model reads on purpose during the session (`read`, `@file`) is not covered: that is asked work.

**When it asks:** only in a folder that brings something that needs trust and has no recorded yes. A folder with nothing to trust starts as today, with no question.
- Mind the case where the folder's `.rig` *is* `RIG_HOME`, for example when rig starts in `~` with `~/.rig` as its home.

**Very simple: yes or no.**
- **Yes:** the folder is trusted from then on, and rig starts with everything it brings.
- **No:** rig refuses to start and exits, saying the folder isn't trusted.
- There is no "run without the project's files" mode, no "not now" and no parent-folder option.
- A yes is kept in rig-written state beside `defaults.json`, not in the user's `settings.toml`. A no isn't remembered, so the next start asks again.
- Undoing a yes needs no new command if deleting a line from that state file does it. The docs say where the file is.

**Nothing before the yes:** no project plugin is built, no project setting takes effect, no project text reaches a model, and no project-named process starts.

**Without a screen** (`rig -p`, `--json`, ACP), rig can't ask. So in an untrusted folder that brings something, it refuses to start (exit 2), in one line naming the folder and the one way to trust it.
- That way is the least new surface that does the job: for example, a flag that records the yes, or running rig once in the TUI there.
- A `helper` child or subagent of a trusted session is trusted (part 2).

**It is a plugin, on by default.** It sits in rig's built-in default list, so a user turns it off with one `[plugins] disable` line in their home `settings.toml`, and every folder is then trusted.
- The question must be asked by code from the home's own build, before any project code is compiled. For example: start on the home's set, ask, then on a yes build the project's set and go on in it.
- If the question can't be a plugin and still keep "nothing before the yes", make it a home setting whose default is to ask, and say why in the PR.

## Tests

Each behaviour gets a test, following the repo's existing pattern. Each test fails on main where main has the behaviour. At least:
- a home with no `[plugins]` builds rig's default list;
- `enable` and `disable` change it, and an unknown name is refused, naming it;
- `rig plugin enable` keeps the user's comments;
- a project's `[plugins]` makes a second build, and a project with the home's set reuses the home's;
- "no" in a folder with an MCP server in `.rig/mcp.json` exits before any process starts or any project file takes effect;
- "yes": the folder's MCP server starts, its `AGENTS.md` and skills reach the model's request, and the next start asks nothing;
- a folder with nothing to trust is never asked;
- `-p` in an untrusted folder that brings something exits 2, naming the folder and how to trust it;
- a `helper` child of a trusted session isn't asked;
- with trust turned off, every folder is trusted.

## The real app

Use a disposable `RIG_HOME`, never your real one.
- **A fresh home:** no `plugins.toml` is written, and `rig plugin list` shows the defaults with their comments.
- **Enabling a plugin:** `rig plugin enable rig_basics::budget::BudgetPlugin` on a commented `settings.toml` keeps the comments. The next start rebuilds, and `/budget` works.
- **A scratch project folder** with three things in it:
  - an `AGENTS.md` asking the model to end every answer with a marker word;
  - a `.rig/settings.toml` that enables a plugin;
  - a `.rig/mcp.json` server.

  Check that:
  - the trust question appears;
  - "no": rig exits with the reason, and nothing from the folder ran;
  - "yes": the project's set builds once, and all three work (the marker word appears);
  - the next start asks nothing and doesn't rebuild.
- **`rig -p` in a second untrusted folder like it:** the one-line refusal and exit 2. Then the documented way to trust it, and `rig -p` runs.
- **Trust turned off in the home:** no question anywhere.

## The PR

- **What changed for a user,** with before and after for the file format (an old `plugins.toml` beside the new `[plugins]` table) and for `rig plugin …` output.
- **`## Changelog` and `## Migration`,** following `.github/PULL_REQUEST_TEMPLATE/other.md`.
- **A capability-changes table:** `plugins.toml` goes, project MCP approval goes, and what replaces each of their uses.
- **Line deltas:** production, inline-test and test-file deltas separately, and the public item count before and after.
- **An interface sweep:** update every doc, help text, README and test harness that names `plugins.toml` (`git grep -n plugins.toml`), including `crates/rig-harness/PLUGINS.md` and `src/launcher/help-agents.txt`. Say what outside harnesses must change.
- **Choices left to the owner:** anything that removes a capability this file doesn't name, with your recommendation.
- **What you didn't verify.**
