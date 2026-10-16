//! CT-14: leftover process reaper.
//!
//! Every process rig starts carries one more variable, [`TAG`] set to the
//! session's id, put in the app's [`Processes`] launcher as the plugin is
//! built: the `shell` tool, background jobs and any plugin's launcher pass
//! it on, and a process's children, grandchildren and daemons inherit it.
//! After each tool call and turn the plugin finds the live processes with
//! the tag in `/proc` and groups them into entries, one per process whose
//! parent is not tagged, with every tagged descendant; an entry belongs to
//! the agent and turn whose call it first appeared after. A process the
//! user started outside rig has no tag and is never touched.
//!
//! When a turn ends, its entries still running are listed to the user in
//! one line and to the model as a note. Entries of the agent's earlier
//! turns that nobody kept (`keep_process`, `/procs keep <id>`) are stopped
//! then; so is every entry on any exit but `/reload`, which holds the exit
//! until they are gone: `SIGTERM`, then `SIGKILL` after the grace period
//! (`[processes] grace_ms`). The entries are saved in the user's plugin
//! data with this process's pid and start time, so the next start finds
//! what a crashed run left and stops each process whose pid still names a
//! process started at its saved start time. Linux only, and only while the
//! workspace is this machine.

use std::collections::HashSet;
use std::fmt::Write as _;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod procfs;
#[cfg(test)]
mod tests;

/// The variable every process rig starts carries, set to the session's id.
pub const TAG: &str = "RIG_PROCESS_REAPER";

/// Where the entries are saved in [`PluginData::user`], one file per session.
const FILES: &str = "process-reaper/";

/// Who the plugin's decisions, notes and messages are from.
const NAME: &str = "process-reaper";

/// The `[processes]` table of the user's settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ReaperSettings {
    /// How long a process asked to stop has before it is killed, in ms.
    pub grace_ms: u64,
}

impl Default for ReaperSettings {
    fn default() -> Self {
        Self { grace_ms: 3000 }
    }
}

impl ReaperSettings {
    fn grace(&self) -> Duration {
        Duration::from_millis(self.grace_ms)
    }
}

/// A process: its pid and its start time in clock ticks since boot, which
/// tell it from a later process given the same pid.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Member {
    /// Its pid.
    pub pid: u32,
    /// When it started.
    pub start: u64,
}

/// A tracked process with its tagged descendants.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Entry {
    /// Its id, unique in the session.
    pub id: u32,
    /// The command line of its first process.
    pub command: String,
    /// The [`AgentId`] of the agent it belongs to, once known.
    pub agent: Option<String>,
    /// The index of the turn it belongs to, once known.
    pub turn: Option<u64>,
    /// Whether it survives turn ends.
    pub kept: bool,
    /// Its live processes, the first one first.
    pub members: Vec<Member>,
    /// Whether it is being stopped.
    #[serde(skip)]
    pub stopping: bool,
}

/// What is saved for a later run: who saved it and what it tracked.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Saved {
    /// The rig process that saved it.
    pub owner: Member,
    /// The entries.
    pub entries: Vec<Entry>,
    /// The next entry's id.
    pub next_id: u32,
}

/// What the plugin tracks in this run.
#[derive(Resource, Debug)]
pub struct Reaper {
    /// The value of [`TAG`] in this session.
    pub token: String,
    /// This rig process.
    pub owner: Member,
    /// The live entries.
    pub entries: Vec<Entry>,
    next_id: u32,
    /// Processes an earlier run left, being stopped: never new entries.
    ignored: HashSet<Member>,
    /// Whether this run's exit stopped, or is stopping, every entry.
    stopped_all: bool,
}

impl Reaper {
    fn file(&self) -> String {
        format!("{FILES}{}.json", self.token)
    }

    fn variable(&self) -> String {
        format!("{TAG}={}", self.token)
    }

    /// Finds the tagged processes again, drops the dead ones and the empty
    /// entries, and gives each new process its parent's entry, or a new
    /// one belonging to `agent`'s `turn`. Returns whether anything changed.
    fn refresh(&mut self, agent: Option<&str>, turn: Option<u64>) -> bool {
        let found = procfs::scan(&self.variable());
        let live: HashSet<Member> = found.iter().map(|proc| proc.member).collect();
        let before: usize = self.entries.iter().map(|entry| entry.members.len()).sum();
        for entry in &mut self.entries {
            entry.members.retain(|member| live.contains(member));
        }
        self.entries.retain(|entry| !entry.members.is_empty());
        let mut changed = before != self.entries.iter().map(|e| e.members.len()).sum::<usize>();
        for proc in found
            .iter()
            .filter(|proc| !self.ignored.contains(&proc.member))
        {
            if self.entry_of(|member| member == proc.member).is_some() {
                continue;
            }
            let parent = found
                .iter()
                .find(|other| other.member.pid == proc.ppid)
                .and_then(|parent| self.entry_of(|member| member == parent.member));
            changed = true;
            match parent.and_then(|at| self.entries.get_mut(at)) {
                Some(entry) => entry.members.push(proc.member),
                None => {
                    self.next_id += 1;
                    self.entries.push(Entry {
                        id: self.next_id,
                        command: proc.command.clone(),
                        agent: agent.map(str::to_owned),
                        turn,
                        kept: false,
                        members: vec![proc.member],
                        stopping: false,
                    });
                }
            }
        }
        changed
    }

    fn entry_of(&self, is: impl Fn(Member) -> bool) -> Option<usize> {
        let has = |entry: &Entry| entry.members.iter().any(|member| is(*member));
        self.entries.iter().position(has)
    }

    fn entry(&mut self, id: u32) -> Option<&mut Entry> {
        self.entries.iter_mut().find(|entry| entry.id == id)
    }

    /// Saves the entries for a later run.
    fn save(&self, data: Option<&PluginData>) {
        let saved = Saved {
            owner: self.owner,
            entries: self.entries.clone(),
            next_id: self.next_id,
        };
        if let (Some(data), Ok(json)) = (data, serde_json::to_vec(&saved)) {
            data.user.write_file(&self.file(), &json).ok();
        }
    }
}

/// Whether processes can be read here: the workspace is this machine.
fn local(workspace: Option<&ProjectWorkspace>) -> bool {
    workspace.is_none_or(|workspace| workspace.0.host().is_none())
}

/// Tracks, reports and stops the processes rig starts; the `processes`,
/// `stop_process` and `keep_process` tools and `/procs`.
#[derive(Default)]
pub struct ProcessReaperPlugin;

impl Plugin for ProcessReaperPlugin {
    fn build(&self, app: &mut App) {
        let owner = procfs::this_process();
        let session = app
            .world()
            .get_resource::<SessionPaths>()
            .and_then(|paths| paths.0.id());
        let token = session.map_or_else(
            || format!("run-{}-{}", owner.pid, owner.start),
            |id| id.as_str().to_owned(),
        );
        // In `build`, so a plugin that wraps the launcher in its `finish`,
        // such as a sandbox, keeps the variable.
        let processes = app.world_mut().get_resource_or_init::<Processes>().clone();
        processes.set(processes.launcher().with_env(TAG, &token));
        let tool = |footprint| ToolOptions {
            footprint: Some(footprint),
        };
        app.add_setting::<ReaperSettings>("processes")
            .insert_resource(Reaper {
                token,
                owner,
                entries: Vec::new(),
                next_id: 0,
                ignored: HashSet::new(),
                stopped_all: false,
            })
            .add_open_tool(
                "processes",
                "List the processes you started that still run: id, command, age, ports, CPU \
                 time and memory. Processes you start that keep running (servers, watchers) \
                 are stopped at the end of your next turn unless you `keep_process` them; \
                 `stop_process` stops one now.",
                tool(Touches::Reads(Scope::Anything)),
                on_list,
            )
            .add_open_tool(
                "stop_process",
                "Stop a process you started, with its child processes.",
                tool(Touches::Writes(Scope::Anything)),
                on_stop,
            )
            .add_open_tool(
                "keep_process",
                "Keep a process you started running across turns, until the session ends.",
                tool(Touches::Independent),
                on_keep,
            )
            .add_command(
                "procs",
                "The processes rig started: /procs [stop|keep <id>]",
                procs,
            )
            .add_systems(Startup, reap_leftovers)
            .add_systems(AgentPass, hold_exit.in_set(Stage::Gate))
            .add_systems(
                Last,
                stop_on_unheld_exit
                    .in_set(OnAppExitSystems)
                    .run_if(resource_exists::<ExitReason>),
            )
            .add_observer(after_call)
            .add_observer(after_turn)
            .add_observer(stopped);
    }
}

fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Stopping processes off the main thread; ends as a `Done<usize>`.
#[derive(Component)]
struct Stopping {
    /// The agent told when it is done, and what.
    tell: Option<(Entity, String)>,
    /// Whether it stops every entry for the exit it holds.
    exit: bool,
}

/// Stops `members` on a task of the entity it returns.
fn stop_later(members: Vec<Member>, grace: Duration, wake: &Wake) -> Running {
    let work = async move {
        blocking(move || Ok(procfs::stop(&members, grace)))
            .await
            .unwrap_or_default()
    };
    Running::spawn(pool(), wake, work)
}

/// Says what was stopped, releases an exit it held, and forgets the task.
fn stopped(
    done: On<Insert<Done<usize>>>,
    stops: Query<&Stopping>,
    reaper: Res<Reaper>,
    data: Option<Res<PluginData>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(stopping) = stops.get(done.entity) else {
        return;
    };
    if let Some((agent, text)) = &stopping.tell {
        notices.write(Notice::info(*agent, text.clone()));
    }
    if stopping.exit
        && let Some(data) = &data
    {
        data.user.remove_file(&reaper.file()).ok();
    }
    commands.entity(done.entity).despawn();
}

/// Finds what each tool call started, for its agent's turn.
fn after_call(
    settled: On<Add<Settled>>,
    calls: Query<(&ForAgent, Option<&CallOf>), With<ToolRequest>>,
    turns: Query<&Turn>,
    ids: Query<&AgentId>,
    mut reaper: ResMut<Reaper>,
    (data, workspace): (Option<Res<PluginData>>, Option<Res<ProjectWorkspace>>),
) {
    let Ok((&ForAgent(agent), call_of)) = calls.get(settled.entity) else {
        return;
    };
    if reaper.stopped_all || !local(workspace.as_deref()) {
        return;
    }
    let turn = call_of
        .and_then(|&CallOf(turn)| turns.get(turn).ok())
        .map(|turn| turn.index);
    let agent = ids.get(agent).ok().map(|id| id.0.as_str());
    if reaper.refresh(agent, turn) {
        reaper.save(data.as_deref());
    }
}

/// At the end of a turn: stops the agent's entries from earlier turns that
/// nobody kept, and tells the user and the model what still runs from
/// this one.
#[expect(
    clippy::too_many_arguments,
    reason = "an observer's queries and resources"
)]
fn after_turn(
    settled: On<Add<Settled>>,
    turns: Query<(&ForAgent, &Turn), With<TurnOutcome>>,
    ids: Query<&AgentId>,
    mut reaper: ResMut<Reaper>,
    (data, workspace): (Option<Res<PluginData>>, Option<Res<ProjectWorkspace>>),
    (settings, wake): (Res<ReaperSettings>, Res<Wake>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&ForAgent(agent), turn)) = turns.get(settled.entity) else {
        return;
    };
    if reaper.stopped_all || !local(workspace.as_deref()) {
        return;
    }
    let Some(id) = ids.get(agent).ok().map(|id| id.0.clone()) else {
        return;
    };
    reaper.refresh(Some(&id), Some(turn.index));
    let mut old = Vec::new();
    let mut new = Vec::new();
    for entry in &mut reaper.entries {
        if entry.agent.is_none() {
            entry.agent = Some(id.clone());
        }
        if entry.agent.as_deref() != Some(id.as_str()) || entry.stopping {
            continue;
        }
        if *entry.turn.get_or_insert(turn.index) == turn.index {
            new.push(entry.clone());
        } else if !entry.kept {
            entry.stopping = true;
            old.push(entry.clone());
        }
    }
    reaper.save(data.as_deref());
    if !old.is_empty() {
        let members = old.iter().flat_map(|entry| entry.members.clone()).collect();
        let text = format!(
            "Stopped {} from an earlier turn: {}.",
            count(old.len()),
            short_list(&old)
        );
        let tell = Some((agent, text));
        commands.spawn((
            Name::new("stopping processes"),
            Stopping { tell, exit: false },
            stop_later(members, settings.grace(), &wake),
        ));
    }
    if new.is_empty() {
        return;
    }
    let list = short_list(&new);
    let running = if new.len() == 1 { "is" } else { "are" };
    notices.write(Notice::info(
        agent,
        format!("{} still running: {list}", count(new.len())),
    ));
    let ids: Vec<String> = new.iter().map(|entry| entry.id.to_string()).collect();
    let note = format!(
        "{} you started {running} still running: {}. It is stopped at the end of your next \
         turn unless you call `keep_process` (ids {}); `stop_process` stops it now.",
        count(new.len()),
        list,
        ids.join(", ")
    );
    commands
        .trigger(Deliver::new(agent, note, DeliveryMode::Inform).with_author(Author::plugin(NAME)));
}

/// `1 process`, `2 processes`.
fn count(n: usize) -> String {
    if n == 1 {
        "1 process".to_owned()
    } else {
        format!("{n} processes")
    }
}

/// `` `npm run dev` (port 5173), `cargo watch` ``.
fn short_list(entries: &[Entry]) -> String {
    let short = |entry: &Entry| {
        let ports = procfs::usage(&entry.members).ports;
        let ports: Vec<String> = ports.iter().map(u16::to_string).collect();
        match ports.len() {
            0 => format!("`{}`", shorten(&entry.command)),
            1 => format!("`{}` (port {})", shorten(&entry.command), ports.join("")),
            _ => format!("`{}` (ports {})", shorten(&entry.command), ports.join(", ")),
        }
    };
    entries.iter().map(short).collect::<Vec<_>>().join(", ")
}

/// A command line cut to 60 characters.
fn shorten(command: &str) -> String {
    if command.chars().count() <= 60 {
        return command.to_owned();
    }
    format!("{}…", command.chars().take(59).collect::<String>())
}

/// One entry's line in a listing.
fn line(entry: &Entry) -> String {
    let usage = procfs::usage(&entry.members);
    let mut line = format!("{} `{}`", entry.id, shorten(&entry.command));
    if !usage.ports.is_empty() {
        let ports: Vec<String> = usage.ports.iter().map(u16::to_string).collect();
        write!(line, ", ports {}", ports.join(", ")).ok();
    }
    if let Some(children) = entry.members.len().checked_sub(1).filter(|n| *n > 0) {
        write!(line, ", {children} child process(es)").ok();
    }
    write!(
        line,
        ", up {}s, CPU {:.1}s, {} MiB",
        usage.age.as_secs(),
        usage.cpu.as_secs_f64(),
        usage.rss_kib / 1024
    )
    .ok();
    if entry.kept {
        line.push_str(", kept");
    }
    if entry.stopping {
        line.push_str(", stopping");
    }
    line
}

/// Every entry's line, after finding the processes again.
fn listing(reaper: &mut Reaper) -> String {
    reaper.refresh(None, None);
    if reaper.entries.is_empty() {
        return "No processes started by rig are running.".to_owned();
    }
    reaper
        .entries
        .iter()
        .map(line)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The arguments of `processes`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ListArgs {}

/// The arguments of `stop_process` and `keep_process`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct IdArgs {
    /// The process's id, as `processes` lists it.
    id: u32,
}

fn decide(
    calls: &mut Query<&mut Decision, ToolCalls>,
    call: Entity,
    result: Result<String, String>,
) {
    if let Ok(mut decision) = calls.get_mut(call) {
        let result = match result {
            Ok(text) => ToolResult::success(text.into()),
            Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
        };
        decision.decide(Decider::plugin(NAME), result);
    }
}

fn on_list(
    called: On<ToolCalled<ListArgs>>,
    mut reaper: ResMut<Reaper>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let text = listing(&mut reaper);
    decide(&mut calls, called.call, Ok(text));
}

/// Stops entry `id`, as `stop_process` and `/procs stop` do: the members
/// to stop and a line about them, or why not.
fn begin_stop(reaper: &mut Reaper, id: u32) -> Result<(Vec<Member>, String), String> {
    reaper.refresh(None, None);
    let entry = reaper
        .entry(id)
        .ok_or_else(|| format!("No process {id}; `processes` lists them."))?;
    entry.stopping = true;
    let children = match entry.members.len() {
        0 | 1 => String::new(),
        n => format!(" and its {} child process(es)", n - 1),
    };
    let text = format!(
        "Stopped process {id} `{}`{children}.",
        shorten(&entry.command)
    );
    Ok((entry.members.clone(), text))
}

fn on_stop(
    called: On<ToolCalled<IdArgs>>,
    mut reaper: ResMut<Reaper>,
    (settings, wake): (Res<ReaperSettings>, Res<Wake>),
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    match begin_stop(&mut reaper, called.args.id) {
        Ok((members, text)) => {
            let grace = settings.grace();
            let work = async move {
                blocking(move || Ok(procfs::stop(&members, grace)))
                    .await
                    .map(|_| text)
            };
            let answer = async move {
                match work.await {
                    Ok(text) => ToolResult::success(text.into()),
                    Err(error) => ToolResult::failed(error),
                }
            };
            commands
                .entity(called.call)
                .insert(Running::answering(pool(), &wake, answer));
        }
        Err(why) => decide(&mut calls, called.call, Err(why)),
    }
}

/// Keeps entry `id` across turn ends: a line about it, or why not.
fn keep(reaper: &mut Reaper, id: u32, data: Option<&PluginData>) -> Result<String, String> {
    reaper.refresh(None, None);
    let entry = reaper
        .entry(id)
        .ok_or_else(|| format!("No process {id}; `processes` lists them."))?;
    entry.kept = true;
    let text = format!(
        "Process {id} `{}` keeps running until the session ends.",
        shorten(&entry.command)
    );
    reaper.save(data);
    Ok(text)
}

fn on_keep(
    called: On<ToolCalled<IdArgs>>,
    mut reaper: ResMut<Reaper>,
    data: Option<Res<PluginData>>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let result = keep(&mut reaper, called.args.id, data.as_deref());
    decide(&mut calls, called.call, result);
}

/// `/procs`: the list, with a picker to stop or keep each; `/procs stop
/// <id>`, `/procs keep <id>`.
fn procs(
    In(args): In<CommandArgs>,
    mut reaper: ResMut<Reaper>,
    data: Option<Res<PluginData>>,
    (settings, wake): (Res<ReaperSettings>, Res<Wake>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let id = words.get(1).and_then(|id| id.parse::<u32>().ok());
    match (words.first().copied(), id) {
        (None, _) => {
            let text = listing(&mut reaper);
            notices.write(Notice::info(args.agent, text.clone()));
            if reaper.entries.is_empty() {
                return;
            }
            let mut picker = Picker::new("Processes").body(text).for_agent(args.agent);
            for entry in &reaper.entries {
                let command = shorten(&entry.command);
                picker = picker
                    .item(
                        format!("stop {} `{command}`", entry.id),
                        format!("procs stop {}", entry.id),
                    )
                    .item(
                        format!("keep {} `{command}`", entry.id),
                        format!("procs keep {}", entry.id),
                    );
            }
            commands.spawn(picker);
        }
        (Some("stop"), Some(id)) => match begin_stop(&mut reaper, id) {
            Ok((members, text)) => {
                commands.spawn((
                    Name::new("stopping a process"),
                    Stopping {
                        tell: Some((args.agent, text)),
                        exit: false,
                    },
                    stop_later(members, settings.grace(), &wake),
                ));
            }
            Err(why) => {
                notices.write(Notice::error(args.agent, why));
            }
        },
        (Some("keep"), Some(id)) => {
            let notice = match keep(&mut reaper, id, data.as_deref()) {
                Ok(text) => Notice::info(args.agent, text),
                Err(why) => Notice::error(args.agent, why),
            };
            notices.write(notice);
        }
        _ => {
            notices.write(Notice::error(args.agent, "Usage: /procs [stop|keep <id>]"));
        }
    }
}

/// At startup: stops what an earlier run left that is not running any
/// more, from its saved entries, each process only while its pid still
/// names the process started at its saved start time; after `/reload`
/// (or a rollback to the previous build) this session's entries are
/// taken over instead.
fn reap_leftovers(
    start: Option<Res<AppStart>>,
    mut reaper: ResMut<Reaper>,
    data: Option<Res<PluginData>>,
    workspace: Option<Res<ProjectWorkspace>>,
    (settings, wake): (Res<ReaperSettings>, Res<Wake>),
    mut commands: Commands,
) {
    let Some(data) = data else {
        return;
    };
    if !local(workspace.as_deref()) {
        return;
    }
    let restarted = start
        .is_some_and(|start| matches!(start.0, StartReason::Reloaded | StartReason::Restarted));
    let own = reaper.file();
    let files = data.user.files().unwrap_or_default();
    let mut left = Vec::new();
    for name in files
        .iter()
        .filter(|name| name.starts_with(FILES) && name.ends_with(".json"))
    {
        let saved = data.user.read_file(name).ok();
        let Some(saved) = saved.and_then(|bytes| serde_json::from_slice::<Saved>(&bytes).ok())
        else {
            continue;
        };
        let ours = *name == own;
        if ours && restarted {
            reaper.entries = saved.entries;
            reaper.next_id = saved.next_id;
            continue;
        }
        if !ours && procfs::alive(saved.owner) {
            // Another rig that runs now.
            continue;
        }
        left.extend(saved.entries.into_iter().flat_map(|entry| entry.members));
        if !ours {
            data.user.remove_file(name).ok();
        }
    }
    reaper.ignored.extend(left.iter().copied());
    reaper.save(Some(&data));
    if left.is_empty() {
        return;
    }
    commands.spawn((
        Name::new("stopping processes an earlier run left"),
        Stopping {
            tell: None,
            exit: false,
        },
        stop_later(left, settings.grace(), &wake),
    ));
}

/// Holds the first exit, unless it is a `/reload`, until every tracked
/// process is stopped.
fn hold_exit(
    exits: Query<(Entity, &ExitRequest), (ReadyToStart, PluginEffects)>,
    mut reaper: ResMut<Reaper>,
    (data, workspace): (Option<Res<PluginData>>, Option<Res<ProjectWorkspace>>),
    (settings, wake): (Res<ReaperSettings>, Res<Wake>),
    mut commands: Commands,
) {
    let Some((exit, ExitRequest(reason))) = exits.iter().next() else {
        return;
    };
    if reaper.stopped_all || *reason == ExitReason::Reload || !local(workspace.as_deref()) {
        return;
    }
    reaper.stopped_all = true;
    reaper.refresh(None, None);
    let members: Vec<Member> = reaper
        .entries
        .iter()
        .flat_map(|entry| entry.members.clone())
        .collect();
    if members.is_empty() {
        if let Some(data) = &data {
            data.user.remove_file(&reaper.file()).ok();
        }
        return;
    }
    commands.spawn((
        Name::new("stopping the processes rig started"),
        Stopping {
            tell: None,
            exit: true,
        },
        HoldOf(exit),
        stop_later(members, settings.grace(), &wake),
    ));
}

/// An exit nobody could hold, such as a window closed: stops every
/// tracked process now, unless it is a `/reload`.
fn stop_on_unheld_exit(
    reason: Res<ExitReason>,
    mut reaper: ResMut<Reaper>,
    (data, workspace): (Option<Res<PluginData>>, Option<Res<ProjectWorkspace>>),
    settings: Res<ReaperSettings>,
) {
    if reaper.stopped_all || *reason == ExitReason::Reload || !local(workspace.as_deref()) {
        return;
    }
    reaper.stopped_all = true;
    reaper.refresh(None, None);
    let members: Vec<Member> = reaper
        .entries
        .iter()
        .flat_map(|entry| entry.members.clone())
        .collect();
    procfs::stop(&members, settings.grace());
    if let Some(data) = &data {
        data.user.remove_file(&reaper.file()).ok();
    }
}
