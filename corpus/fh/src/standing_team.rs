//! A standing team of specialist agents. `team.toml`, in the project or
//! in `RIG_HOME`, names the members, each with a role, a model, the tools
//! it may use, an optional path scope and whether it may delegate itself:
//!
//! ```toml
//! [member.backend]
//! role = "Owns the server and the database schema."
//! model = "anthropic/claude-sonnet-4-5"
//! tools = ["read", "write", "edit", "shell"]
//! scope = "server/**"
//!
//! [member.frontend]
//! role = "Owns the web client."
//! ```
//!
//! The `delegate(member, task)` tool hands a member an assignment and
//! answers with the member's final reply. A member is a helper agent
//! ([`SpawnSubagent::helper`]) spawned on its first assignment and kept for
//! the session, marked [`TeamMember`] (saved), so each later assignment
//! continues its own conversation, across `/resume` too. Assignments to
//! different members run side by side; those to one member wait for each
//! other, in order. A scoped member's writes outside its scope are
//! refused, `/team reset <member>` clears a member's memory, and stopping
//! the delegating turn stops the members working for it.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;

use glob::Pattern;
use rig_basics::sessions::agents::{AgentColumn, ColumnOf};
use rig_basics::subagents::{MESSAGE, SpawnSubagent, SpawnedAgent, SubagentsPlugin, TASK, WAIT};
use schemars::JsonSchema;
use serde::Deserialize;

use rig_harness::prelude::*;

/// The tool that hands a member an assignment.
pub const DELEGATE: &str = "delegate";

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TeamSystems;

/// Reads `team.toml` and, when it names members, adds the `delegate` tool,
/// the members' path scopes, their line in `/agents` and `/team`.
#[derive(Default)]
pub struct StandingTeamPlugin;

impl Plugin for StandingTeamPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.register_type::<TeamMember>()
            .init_resource::<Desk>()
            .configure_sets(AgentPass, TeamSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, keep_in_scope.in_set(TeamSystems))
            .add_systems(PostUpdate, show_work.run_if(resource_changed::<Desk>))
            .add_observer(hired)
            .add_observer(join)
            .add_observer(started_on)
            .add_observer(finished)
            .add_observer(called_off)
            .add_command(
                "team",
                "The standing team: /team lists it, /team reset <member> clears one's memory",
                team_command,
            );
    }

    /// Reads the team once the project and `RIG_HOME` are known, and
    /// offers `delegate` with the members in its description.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        if !world.contains_resource::<Team>() {
            let root = world
                .get_resource::<ProjectRoot>()
                .map(|root| root.0.clone());
            let home = world
                .get_resource::<RigHome>()
                .map(|home| home.0.root().to_owned());
            let team = [root, home]
                .into_iter()
                .flatten()
                .map(|dir| Team::read(&dir.join("team.toml")))
                .find(|team| !team.members.is_empty() || !team.problems.is_empty())
                .unwrap_or_default();
            world.insert_resource(team);
        }
        let team = app.world().resource::<Team>().clone();
        if !team.problems.is_empty() {
            let problems = team.problems.clone();
            app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                for problem in &problems {
                    notices.write(Notice::warn(None, problem.clone()));
                }
            });
        }
        if team.members.is_empty() {
            return;
        }
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.add_open_tool(DELEGATE, &team.description(), options, delegated);
    }
}

/// The members `team.toml` names, by name, and what was wrong with it.
#[derive(Resource, Clone, Debug, Default)]
pub struct Team {
    /// Each member's settings, by name.
    pub members: BTreeMap<String, Member>,
    /// Why the file did not read as a team, to show the user.
    pub problems: Vec<String>,
}

impl Team {
    /// The team in the TOML file at `path`: none when it does not exist.
    pub fn read(path: &Path) -> Self {
        let file = Settings::read(path);
        let mut problems: Vec<String> = file.problems().to_vec();
        let members = file
            .get::<BTreeMap<String, Member>>("member")
            .unwrap_or_else(|problem| {
                problems.push(format!("{}: {problem}", path.display()));
                BTreeMap::new()
            });
        Self { members, problems }
    }

    /// What `delegate` tells the model: the members and their roles.
    fn description(&self) -> String {
        let mut text = "Hand an assignment to a member of your standing team and get its final \
                        reply. A member remembers its earlier assignments, so a follow-up can be \
                        short. Delegate to several members in one reply to have them work side \
                        by side; assignments to one member run one after the other. The members:"
            .to_owned();
        for (name, member) in &self.members {
            text.push_str(&format!("\n- `{name}`: {}", member.role.trim()));
            if let Some(scope) = &member.scope {
                text.push_str(&format!(" (writes only under `{scope}`)"));
            }
        }
        text
    }
}

/// A member's settings: a `[member.<name>]` table of `team.toml`.
#[derive(Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Member {
    /// What it does, shown to the agents that delegate and told to it.
    pub role: String,
    /// Its catalog model, `vendor/model`; the delegating agent's when not
    /// set.
    #[serde(default)]
    pub model: Option<String>,
    /// The tools it may use, from the delegating agent's; all of those but
    /// the delegating ones when not set.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Where it may write, a glob relative to the project, such as
    /// `server/**`.
    #[serde(default)]
    pub scope: Option<String>,
    /// Whether it may delegate itself: keep `delegate`, `task`, `message`
    /// and `wait`.
    #[serde(default)]
    pub delegate: bool,
}

/// On an agent: it is the team member of this name. Saved with the
/// session, so a restored member takes its next assignment.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct TeamMember(pub String);

/// The arguments of `delegate`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DelegateArgs {
    /// The member's name.
    pub member: String,
    /// The assignment: what to do, and what to answer with.
    pub task: String,
}

/// An assignment: the `delegate` call that answers with its reply.
#[derive(Clone, Debug)]
struct Job {
    call: Entity,
    task: String,
    /// Who asked, and the call's id, for the message's origin.
    asker: AgentId,
    request: RequestId,
    /// The message that hands it over, once sent.
    message: Option<Entity>,
    /// The member's turn it runs in, once delivered.
    turn: Option<Entity>,
}

/// The assignments: waiting, per member in order, and the one each member
/// works on.
#[derive(Resource, Default)]
struct Desk {
    waiting: HashMap<String, VecDeque<Job>>,
    working: HashMap<String, Job>,
    /// The members being spawned.
    hiring: Vec<String>,
}

/// On a `delegate` call that waits for a member.
#[derive(Component)]
struct Delegating;

/// On a member's [`SpawnSubagent`]: the member's name.
#[derive(Component)]
struct Hiring(String);

/// On the message that hands a member its assignment.
#[derive(Component)]
struct Assignment(String);

/// On a member's `/agents` column.
#[derive(Component)]
struct WorkColumn;

/// Takes a `delegate` call: queues it for its member, spawning the member
/// on its first assignment.
fn delegated(
    called: On<ToolCalled<DelegateArgs>>,
    team: Res<Team>,
    members: Query<&TeamMember>,
    callers: Query<&ToolAccess>,
    tools: Query<&ToolDef>,
    mut desk: ResMut<Desk>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let name = called.args.member.trim();
    let Some(member) = team.members.get(name) else {
        let names: Vec<&str> = team.members.keys().map(String::as_str).collect();
        let why = format!("No member `{name}`; the team: {}", names.join(", "));
        if let Ok(mut call) = calls.get_mut(called.call) {
            call.decide(Decider::plugin("fh-standing-team"), Outcome::refused(why));
        }
        return;
    };
    if members.get(called.agent).is_ok_and(|me| me.0 == name) {
        if let Ok(mut call) = calls.get_mut(called.call) {
            call.decide(
                Decider::plugin("fh-standing-team"),
                Outcome::refused(format!("You are `{name}`; do it yourself.")),
            );
        }
        return;
    }
    commands.entity(called.call).insert(Delegating);
    desk.waiting
        .entry(name.to_owned())
        .or_default()
        .push_back(Job {
            call: called.call,
            task: called.args.task.clone(),
            asker: called.agent_id.clone(),
            request: RequestId(called.tool_call.id.to_string()),
            message: None,
            turn: None,
        });
    let hired = members.iter().any(|member| member.0 == name);
    if hired || desk.hiring.iter().any(|hiring| hiring == name) {
        commands.run_system_cached(dispatch);
        return;
    }
    desk.hiring.push(name.to_owned());
    let allowed = match &member.tools {
        Some(tools) => tools.clone(),
        None => {
            let access = callers.get(called.agent).ok();
            let names = tools.iter().map(|def| def.0.name.to_string());
            names
                .filter(|name| access.is_none_or(|access| access.allows(name)))
                .collect()
        }
    };
    let delegating = [DELEGATE, TASK, MESSAGE, WAIT];
    let allowed = allowed
        .into_iter()
        .filter(|tool| member.delegate || !delegating.contains(&tool.as_str()))
        .collect();
    let mut request = SpawnSubagent::helper(name);
    request.model.clone_from(&member.model);
    request.tools = Some(allowed);
    commands.spawn((
        request,
        Hiring(name.to_owned()),
        ForAgent(called.agent),
        Effect::with_parent_id(Some(called.effect)),
    ));
}

/// Marks the member spawned, or refuses its assignments with why none was.
fn hired(
    settled: On<Add<Settled>>,
    hires: Query<(&Hiring, &Outcome, Option<&SpawnedAgent>)>,
    mut desk: ResMut<Desk>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok((Hiring(name), outcome, spawned)) = hires.get(settled.entity) else {
        return;
    };
    desk.hiring.retain(|hiring| hiring != name);
    match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(agent)), _) => {
            commands.entity(agent).insert(TeamMember(name.clone()));
            commands.run_system_cached(dispatch);
        }
        (None, result) => {
            let why = match result {
                Err(report) => report.message.clone(),
                Ok(_) => "it was not spawned".to_owned(),
            };
            for job in desk.waiting.remove(name).unwrap_or_default() {
                if let Ok(mut call) = calls.get_mut(job.call) {
                    call.decide(
                        Decider::plugin("fh-standing-team"),
                        Outcome::refused(format!("`{name}` could not start: {why}")),
                    );
                }
            }
        }
    }
}

/// Tells a member, new or restored, its role and scope, and gives it its
/// line in `/agents`.
fn join(
    added: On<Add<TeamMember>>,
    members: Query<&TeamMember>,
    team: Res<Team>,
    mut commands: Commands,
) {
    let agent = added.entity;
    let Ok(TeamMember(name)) = members.get(agent) else {
        return;
    };
    let Some(member) = team.members.get(name) else {
        return;
    };
    let mut role = format!(
        "You are `{name}`, a member of a standing team. Your role: {}\nAssignments come from \
         the agent that leads the team; your answer to each is your report. You keep your \
         memory between assignments.",
        member.role.trim()
    );
    if let Some(scope) = &member.scope {
        role.push_str(&format!("\nYou may only write files under `{scope}`."));
    }
    let section = PromptSection::new(PromptSection::ORDER_ROLE, "team", role);
    commands.spawn((section, SectionOf(agent)));
    commands.spawn((AgentColumn::new("team", ""), ColumnOf(agent), WorkColumn));
    commands.run_system_cached(dispatch);
}

/// Hands each idle member its next assignment.
fn dispatch(
    mut desk: ResMut<Desk>,
    members: Query<(Entity, &TeamMember, Has<ActiveTurn>)>,
    mut commands: Commands,
) {
    for (agent, TeamMember(name), busy) in &members {
        if busy || desk.working.contains_key(name) {
            continue;
        }
        let Some(mut job) = desk.waiting.get_mut(name).and_then(VecDeque::pop_front) else {
            continue;
        };
        let author = Author::agent(job.asker.clone(), Some(job.request.clone()));
        let author = author.titled(first_line(&job.task));
        let message = Incoming::new(job.task.clone(), DeliveryMode::Queue).with_author(author);
        let message = commands.spawn((message, ForAgent(agent), Assignment(name.clone())));
        job.message = Some(message.id());
        desk.working.insert(name.clone(), job);
    }
}

/// Notes the member's turn an assignment went into.
fn started_on(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&DeliveredIn, &Assignment)>,
    mut desk: ResMut<Desk>,
) {
    let Ok((&DeliveredIn(turn), Assignment(name))) = messages.get(delivered.entity) else {
        return;
    };
    if let Some(job) = desk.working.get_mut(name)
        && job.message == Some(delivered.entity)
    {
        job.turn = Some(turn);
    }
}

/// Answers the `delegate` call with the member's final reply when its
/// turn ends, and hands the member its next assignment.
fn finished(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent), With<Turn>>,
    members: Query<&TeamMember>,
    mut desk: ResMut<Desk>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(TeamMember(name)) = members.get(agent) else {
        return;
    };
    let done = desk
        .working
        .get(name)
        .is_some_and(|job| job.turn == Some(settled.entity));
    if done
        && let Some(job) = desk.working.remove(name)
        && let Ok(mut call) = calls.get_mut(job.call)
    {
        let answer: Outcome = match outcome {
            TurnOutcome::Answered(message) => {
                let text = message
                    .final_answer()
                    .unwrap_or_else(|| "(no answer)".to_owned());
                ToolResult::success(format!("`{name}` reports:\n{text}").into()).into()
            }
            TurnOutcome::Failed(why) => Outcome::refused(format!("`{name}` failed: {why}")),
            TurnOutcome::Stopped(why) => Outcome::refused(format!(
                "`{name}` was stopped{}",
                why.as_ref()
                    .map(|why| format!(": {why}"))
                    .unwrap_or_default()
            )),
        };
        call.decide(Decider::plugin("fh-standing-team"), answer);
    }
    commands.run_system_cached(dispatch);
}

/// A `delegate` call that settled without the member's reply, as when its
/// turn is stopped: its assignment is taken back, and stopped if the
/// member works on it, which keeps what it did so far in its conversation.
fn called_off(
    settled: On<Add<Settled>>,
    delegating: Query<(), With<Delegating>>,
    members: Query<(Entity, &TeamMember)>,
    mut desk: ResMut<Desk>,
    mut messages: Query<&mut Decision, IncomingMessages>,
    mut commands: Commands,
) {
    let call = settled.entity;
    if !delegating.contains(call) {
        return;
    }
    for waiting in desk.waiting.values_mut() {
        waiting.retain(|job| job.call != call);
    }
    let Some(name) = desk
        .working
        .iter()
        .find(|(_, job)| job.call == call)
        .map(|(name, _)| name.clone())
    else {
        return;
    };
    let Some(job) = desk.working.remove(&name) else {
        return;
    };
    let member = members.iter().find(|(_, member)| member.0 == name);
    match (job.turn, member) {
        (Some(_), Some((agent, _))) => {
            let reason = Some("the delegating agent was stopped".to_owned());
            commands.trigger(StopTurn {
                entity: agent,
                reason,
            });
        }
        _ => {
            if let Some(mut message) = job
                .message
                .and_then(|message| messages.get_mut(message).ok())
            {
                message.decide(
                    Decider::plugin("fh-standing-team"),
                    Outcome::custom("withdrawn"),
                );
            }
        }
    }
    commands.run_system_cached(dispatch);
}

/// Refuses a scoped member's write outside its scope, as its call is about
/// to run: what a call writes is its [`Footprint`], from the tool.
fn keep_in_scope(
    mut calls: Query<(&Footprint, &ForAgent, &mut Decision), (ReadyToStart, ToolCalls)>,
    members: Query<(&TeamMember, Option<&WorkDir>)>,
    team: Res<Team>,
    root: Res<ProjectRoot>,
) {
    for (footprint, &ForAgent(agent), mut decision) in &mut calls {
        let Touches::Writes(Scope::Paths(paths)) = &footprint.0 else {
            continue;
        };
        let Ok((TeamMember(name), work_dir)) = members.get(agent) else {
            continue;
        };
        let Some(scope) = team
            .members
            .get(name)
            .and_then(|member| member.scope.as_ref())
        else {
            continue;
        };
        let base = work_dir.map_or(&root.0, |dir| &dir.0);
        let pattern = Pattern::new(scope).ok();
        let inside = |path: &std::path::PathBuf| {
            let relative = path.strip_prefix(base).ok();
            relative
                .zip(pattern.as_ref())
                .is_some_and(|(relative, pattern)| pattern.matches_path(relative))
        };
        if let Some(outside) = paths.iter().find(|path| !inside(path)) {
            decision.decide(
                Decider::plugin("fh-standing-team"),
                Outcome::refused(format!(
                    "`{name}` may only write under `{scope}`, not {}",
                    outside.display()
                )),
            );
        }
    }
}

/// Keeps each member's `/agents` column on the assignment it works on.
fn show_work(
    desk: Res<Desk>,
    mut columns: Query<(&mut AgentColumn, &ColumnOf), With<WorkColumn>>,
    members: Query<&TeamMember>,
) {
    for (mut column, &ColumnOf(agent)) in &mut columns {
        let Ok(TeamMember(name)) = members.get(agent) else {
            continue;
        };
        let waiting = desk.waiting.get(name).map_or(0, VecDeque::len);
        let mut text = desk
            .working
            .get(name)
            .map(|job| format!("on \"{}\"", first_line(&job.task)))
            .unwrap_or_default();
        if waiting > 0 {
            text.push_str(&format!(" +{waiting} queued"));
        }
        column.set_if_neq(AgentColumn::new("team", text.trim()));
    }
}

/// `/team`: the members and what each does; `/team reset <member>` clears
/// one's conversation.
fn team_command(
    In(args): In<CommandArgs>,
    team: Res<Team>,
    desk: Res<Desk>,
    mut members: Query<(Entity, &TeamMember, &mut Conversation, Has<ActiveTurn>)>,
    mut notices: MessageWriter<Notice>,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    match words.as_slice() {
        [] => {
            let lines = team.members.keys().map(|name| {
                let state = match desk.working.get(name) {
                    Some(job) => format!("working on \"{}\"", first_line(&job.task)),
                    None if members.iter().any(|member| &member.1.0 == name) => "idle".to_owned(),
                    None => "not started".to_owned(),
                };
                format!("{name}: {state}")
            });
            let lines: Vec<String> = lines.collect();
            let text = match lines.is_empty() {
                true => "No team: add members to team.toml.".to_owned(),
                false => lines.join("\n"),
            };
            notices.write(Notice::info(args.agent, text));
        }
        ["reset", name] => {
            let member = members
                .iter_mut()
                .find(|(_, member, _, _)| member.0 == *name);
            match member {
                None => {
                    let why = format!("No member `{name}` has started.");
                    notices.write(Notice::error(args.agent, why));
                }
                Some((_, _, _, busy)) if busy || desk.working.contains_key(*name) => {
                    let why = format!("`{name}` is working; stop it first.");
                    notices.write(Notice::error(args.agent, why));
                }
                Some((_, _, mut conversation, _)) => {
                    conversation.splice(.., [], Author::plugin("fh-standing-team"));
                    notices.write(Notice::info(args.agent, format!("`{name}` starts afresh.")));
                }
            }
        }
        _ => {
            let why = "Use /team, or /team reset <member>.";
            notices.write(Notice::error(args.agent, why));
        }
    }
}

/// The first line of `text`, as a short title.
fn first_line(text: &str) -> String {
    text.trim().lines().next().unwrap_or_default().to_owned()
}

#[cfg(test)]
mod tests;
