//! CT-01: time-boxed permission grants. `/grant shell for 10m`, `/grant
//! edits:src/** until turn-end` or `/grant write for 2 turns` lets the
//! calls a grant covers run without the user's approval while it lives:
//! once every gate had its say, a covered call that only waits on
//! [`NeedsApproval`] is [`Approved`], so a refusal from another gate still
//! stands. A grant covers the agent it was made in and the subagents it
//! starts meanwhile, runs out by itself on the wall clock ([`WakeAt`]) or
//! by its agent's ended turns with a one-line notice, is listed by
//! `/grants` and ended early by `/revoke <n|all>`. Grants are saved with
//! the session ([`SavedGrants`]); a timed one whose time passed is gone on
//! resume. With the `tui` feature the status line counts the soonest one
//! down, `grant shell 4:12`.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(corpus_ct::timed_grants::TimedGrantsPlugin);
//! ```

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use glob::{MatchOptions, Pattern};
use rig_basics::subagents::SpawnedAgent;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{Scope, Touches};
use serde::{Deserialize, Serialize};

/// The plugin's name, in its notices.
pub const NAME: &str = "timed-grants";

/// How often a timed grant wakes the loop, for its countdown.
const TICK: Duration = Duration::from_secs(1);

/// `*` stays within a directory, `**` crosses them.
const GLOB: MatchOptions = MatchOptions {
    case_sensitive: true,
    require_literal_separator: true,
    require_literal_leading_dot: false,
};

/// `/grant`, `/grants` and `/revoke`, and the approvals grants give.
#[derive(Default)]
pub struct TimedGrantsPlugin;

impl Plugin for TimedGrantsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SavedGrants>()
            .register_required_components::<Agent, SavedGrants>()
            .keeps_saved::<SavedGrants>()
            .add_command(
                "grant",
                "Let calls run without asking: /grant <tool|edits|edits:glob> [for 10m | for N turns | until turn-end]",
                grant,
            )
            .add_command("grants", "List the live grants", list)
            .add_command("revoke", "End grants early: /revoke <n|all>", revoke)
            .add_systems(
                AgentPass,
                approve_granted
                    .after(GateSystems::After)
                    .before(ApprovalSystems)
                    .before(GateSystems::Fallback),
            )
            .add_systems(Update, save_grants.after(AgentLoop))
            .add_observer(run_out_in_time)
            .add_observer(count_turn)
            .add_observer(cover_subagent)
            .add_observer(restore_grants);
        #[cfg(feature = "tui")]
        app.add_systems(
            PostUpdate,
            show_soonest.in_set(rig_tui::StatusSystems::Show),
        );
    }
}

/// What a grant covers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cover {
    /// Every call of the tool of this name.
    Tool(String),
    /// Every call that writes named files (its footprint), whatever its
    /// tool.
    Edits,
    /// Every call whose written files all match this glob, relative to
    /// the agent's directory.
    EditsUnder(String),
}

impl Cover {
    /// `shell`, `edits` or `edits:<glob>`.
    fn parse(what: &str) -> Result<Self, String> {
        match what.split_once(':') {
            _ if what.is_empty() => Err("/grant needs what it grants".to_owned()),
            _ if what == "edits" => Ok(Self::Edits),
            Some(("edits", glob)) => Pattern::new(glob)
                .map(|_| Self::EditsUnder(glob.to_owned()))
                .map_err(|error| format!("Not a glob, {glob}: {error}")),
            _ => Ok(Self::Tool(what.to_owned())),
        }
    }

    /// As the user typed it.
    pub fn label(&self) -> String {
        match self {
            Self::Tool(name) => name.clone(),
            Self::Edits => "edits".to_owned(),
            Self::EditsUnder(glob) => format!("edits:{glob}"),
        }
    }

    /// Whether a call of `tool` that touches `touches`, made by an agent
    /// working in `base`, is covered.
    fn covers(&self, tool: &str, touches: Option<&Touches>, base: Option<&Path>) -> bool {
        match self {
            Self::Tool(name) => name == tool,
            Self::Edits => written(touches).is_some(),
            Self::EditsUnder(glob) => {
                let (Some(paths), Some(base), Ok(pattern)) =
                    (written(touches), base, Pattern::new(glob))
                else {
                    return false;
                };
                paths.iter().all(|path| {
                    path.strip_prefix(base)
                        .is_ok_and(|relative| pattern.matches_path_with(relative, GLOB))
                })
            }
        }
    }
}

/// The files a call writes, when it names them.
fn written(touches: Option<&Touches>) -> Option<&[PathBuf]> {
    match touches {
        Some(Touches::Writes(Scope::Paths(paths))) if !paths.is_empty() => Some(paths),
        _ => None,
    }
}

/// How a grant runs out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lasts {
    /// When its entity's [`WakeAt`] is due.
    Time,
    /// After this many more ended turns of its agent.
    Turns(u32),
    /// When its agent's running turn ends, or its next one if none runs.
    TurnEnd,
}

/// A live grant, on an entity of its own [`GrantOf`] the agent it was made
/// in, with a [`WakeAt`] when it is timed.
#[derive(Component, Clone, Debug)]
pub struct Grant {
    /// Its number in `/grants`, among its agent's.
    pub number: u32,
    /// What it covers.
    pub cover: Cover,
    /// How it runs out.
    pub lasts: Lasts,
    /// The agents whose calls it covers: its own and the subagents started
    /// while it lived.
    pub agents: Vec<AgentId>,
}

impl Grant {
    /// How much is left: `4:12 left`, `2 turns left`, `until the turn ends`.
    fn left(&self, wake: Option<&WakeAt>, now: SystemTime) -> String {
        match (self.lasts, wake) {
            (Lasts::Time, Some(wake)) => format!("{} left", clock(wake.left(now))),
            (Lasts::Time, None) => "running out".to_owned(),
            (Lasts::Turns(1), _) => "1 turn left".to_owned(),
            (Lasts::Turns(n), _) => format!("{n} turns left"),
            (Lasts::TurnEnd, _) => "until the turn ends".to_owned(),
        }
    }
}

/// The agent a grant was made in. Despawning the agent ends its grants.
#[derive(Component, Debug)]
#[relationship(relationship_target = Grants)]
pub struct GrantOf(pub Entity);

/// An agent's live grants, in the order they were made.
#[derive(Component, Debug)]
#[relationship_target(relationship = GrantOf, linked_spawn)]
pub struct Grants(Vec<Entity>);

/// An agent's grants as saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct SavedGrants(pub Vec<SavedGrant>);

/// One saved grant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedGrant {
    /// Its number in `/grants`.
    pub number: u32,
    /// What it covers.
    pub cover: Cover,
    /// How it runs out.
    pub lasts: Lasts,
    /// A timed grant's end, in milliseconds since the Unix epoch.
    pub until_ms: Option<u64>,
    /// The agents whose calls it covers.
    pub agents: Vec<AgentId>,
}

/// How long a grant asked for lasts.
enum Lasting {
    For(Duration),
    Turns(u32),
    TurnEnd,
}

/// `<what> [for <duration> | for <n> turns | until turn-end]`; a grant
/// with no end lasts until the turn ends.
fn parse(args: &str) -> Result<(Cover, Lasting), String> {
    let words: Vec<&str> = args.split_whitespace().collect();
    let (what, rest) = words
        .split_first()
        .map_or(("", &[][..]), |(what, rest)| (*what, rest));
    let cover = Cover::parse(what)?;
    let lasting = match rest {
        [] | ["until", "turn-end"] => Lasting::TurnEnd,
        ["for", n, "turn" | "turns"] => match n.parse() {
            Ok(n) if n > 0 => Lasting::Turns(n),
            _ => return Err(format!("Not a number of turns: {n}")),
        },
        ["for", wait] => Lasting::For(duration(wait).ok_or(format!("Not a duration: {wait}"))?),
        _ => {
            return Err(
                "/grant <tool|edits|edits:glob> [for 10m | for N turns | until turn-end]"
                    .to_owned(),
            );
        }
    };
    Ok((cover, lasting))
}

/// `90s`, `10m`, `1h30m`.
fn duration(text: &str) -> Option<Duration> {
    let (mut total, mut number) = (0u64, String::new());
    for c in text.chars() {
        if c.is_ascii_digit() {
            number.push(c);
            continue;
        }
        let n: u64 = number.parse().ok()?;
        number.clear();
        let unit = match c {
            's' => 1,
            'm' => 60,
            'h' => 3600,
            _ => return None,
        };
        total = total.checked_add(n.checked_mul(unit)?)?;
    }
    (number.is_empty() && total > 0).then(|| Duration::from_secs(total))
}

/// `4:12`, or `1:02:03` from an hour on.
fn clock(left: Duration) -> String {
    let secs = left.as_secs();
    let (hours, minutes, seconds) = (secs / 3600, secs / 60 % 60, secs % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

/// `/grant <what> [for …]`: a grant for the agent.
fn grant(
    In(args): In<CommandArgs>,
    agents: Query<(&AgentId, Option<&Grants>)>,
    grants: Query<&Grant>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((id, held)) = agents.get(args.agent) else {
        return;
    };
    let (cover, lasting) = match parse(&args.args) {
        Ok(asked) => asked,
        Err(why) => {
            notices.write(Notice::error(args.agent, why));
            return;
        }
    };
    let numbers = held
        .into_iter()
        .flat_map(|held| held.iter())
        .filter_map(|entity| grants.get(entity).ok());
    let number = numbers
        .map(|grant| grant.number)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let (lasts, wake, how) = match lasting {
        Lasting::For(wait) => (
            Lasts::Time,
            Some(WakeAt::after(wait)),
            format!("for {}", clock(wait)),
        ),
        Lasting::Turns(n) => (Lasts::Turns(n), None, format!("for {n} turn(s)")),
        Lasting::TurnEnd => (Lasts::TurnEnd, None, "until the turn ends".to_owned()),
    };
    let label = cover.label();
    let grant = Grant {
        number,
        cover,
        lasts,
        agents: vec![id.clone()],
    };
    let mut spawned = commands.spawn((
        grant,
        GrantOf(args.agent),
        Name::new(format!("grant {label}")),
    ));
    if let Some(wake) = wake {
        spawned.insert((wake, KeepAwake(TICK)));
    }
    notices.write(Notice::info(args.agent, format!("Granted {label} {how}.")));
}

/// `/grants`: the grants that cover the agent, with what is left of each.
fn list(
    In(args): In<CommandArgs>,
    agents: Query<&AgentId>,
    grants: Query<(&Grant, Option<&WakeAt>)>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(id) = agents.get(args.agent) else {
        return;
    };
    let now = SystemTime::now();
    let mut live: Vec<_> = grants
        .iter()
        .filter(|(grant, _)| grant.agents.contains(id))
        .collect();
    live.sort_by_key(|(grant, _)| grant.number);
    if live.is_empty() {
        notices.write(Notice::info(args.agent, "No live grants."));
    }
    for (grant, wake) in live {
        let line = format!(
            "{}. {}, {}",
            grant.number,
            grant.cover.label(),
            grant.left(wake, now)
        );
        notices.write(Notice::info(args.agent, line));
    }
}

/// `/revoke <n|all>`: ends the agent's grant `n`, or every one.
fn revoke(
    In(args): In<CommandArgs>,
    agents: Query<&Grants>,
    grants: Query<&Grant>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let all = args.args == "all";
    let number: Option<u32> = args.args.parse().ok();
    if !all && number.is_none() {
        notices.write(Notice::error(args.agent, "/revoke <n|all>"));
        return;
    }
    let held = agents
        .get(args.agent)
        .into_iter()
        .flat_map(|held| held.iter());
    let mut ended = Vec::new();
    for entity in held {
        if let Ok(grant) = grants.get(entity)
            && (all || number == Some(grant.number))
        {
            commands.entity(entity).despawn();
            ended.push(grant.cover.label());
        }
    }
    if ended.is_empty() {
        notices.write(Notice::error(
            args.agent,
            format!("No grant {}.", args.args),
        ));
    } else {
        notices.write(Notice::info(
            args.agent,
            format!("Revoked {}.", ended.join(", ")),
        ));
    }
}

/// Approves each covered call that waits on the user's approval and that
/// no gate decided, after every gate and before the harness asks.
fn approve_granted(
    calls: Query<
        (
            Entity,
            &ToolRequest,
            Option<&Footprint>,
            &ForAgent,
            &Decision,
        ),
        (
            ReadyToStart,
            ToolCalls,
            With<NeedsApproval>,
            Without<Approved>,
        ),
    >,
    agents: Query<(&AgentId, Option<&WorkDir>)>,
    grants: Query<&Grant>,
    (root, workspace): (Option<Res<ProjectRoot>>, Option<Res<ProjectWorkspace>>),
    mut commands: Commands,
) {
    if grants.is_empty() {
        return;
    }
    for (call, request, footprint, &ForAgent(agent), decision) in &calls {
        let Ok((id, work_dir)) = agents.get(agent) else {
            continue;
        };
        if decision.is_decided() {
            continue;
        }
        let base = work_dir
            .map(|dir| &dir.0)
            .or(root.as_ref().map(|root| &root.0));
        let base = base.map(|base| {
            workspace
                .as_ref()
                .map_or_else(|| base.clone(), |it| it.0.canonical(base))
        });
        let tool = request.call.function.name.as_str();
        let touches = footprint.map(|footprint| &footprint.0);
        let covered = grants.iter().any(|grant| {
            grant.agents.contains(id) && grant.cover.covers(tool, touches, base.as_deref())
        });
        if covered {
            commands.entity(call).insert(Approved);
        }
    }
}

/// Ends a grant with a one-line notice to its agent.
fn run_out(
    grant: Entity,
    label: &str,
    agent: Entity,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    commands.entity(grant).despawn();
    notices.write(Notice::info(
        agent,
        format!("Grant {label} ran out; its calls ask again."),
    ));
}

/// A timed grant runs out once its time passed.
fn run_out_in_time(
    due: On<Add<Due>>,
    grants: Query<(&Grant, &GrantOf)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if let Ok((grant, &GrantOf(agent))) = grants.get(due.entity) {
        run_out(
            due.entity,
            &grant.cover.label(),
            agent,
            &mut commands,
            &mut notices,
        );
    }
}

/// Each ended turn of an agent counts down its turn-based grants.
fn count_turn(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<Turn>>,
    agents: Query<&Grants>,
    mut grants: Query<&mut Grant>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(&ForAgent(agent)) = turns.get(settled.entity) else {
        return;
    };
    let Ok(held) = agents.get(agent) else {
        return;
    };
    for entity in held.iter() {
        let Ok(mut grant) = grants.get_mut(entity) else {
            continue;
        };
        let left = match grant.lasts {
            Lasts::Turns(n) => n.saturating_sub(1),
            Lasts::TurnEnd => 0,
            Lasts::Time => continue,
        };
        if left == 0 {
            run_out(
                entity,
                &grant.cover.label(),
                agent,
                &mut commands,
                &mut notices,
            );
        } else {
            grant.lasts = Lasts::Turns(left);
        }
    }
}

/// A subagent started by a covered agent is covered by the same grants.
fn cover_subagent(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &ForAgent)>,
    ids: Query<&AgentId>,
    mut grants: Query<&mut Grant>,
) {
    let Ok((&SpawnedAgent(child), &ForAgent(parent))) = requests.get(spawned.entity) else {
        return;
    };
    let (Ok(parent), Ok(child)) = (ids.get(parent), ids.get(child)) else {
        return;
    };
    for mut grant in &mut grants {
        if grant.agents.contains(parent) && !grant.agents.contains(child) {
            grant.agents.push(child.clone());
        }
    }
}

/// Keeps each agent's [`SavedGrants`] as its live grants are.
fn save_grants(
    mut agents: Query<(Option<&Grants>, &mut SavedGrants)>,
    grants: Query<(&Grant, Option<&WakeAt>)>,
) {
    for (held, mut saved) in &mut agents {
        let live = held
            .into_iter()
            .flat_map(|held| held.iter())
            .filter_map(|entity| grants.get(entity).ok());
        let live: Vec<SavedGrant> = live
            .map(|(grant, wake)| SavedGrant {
                number: grant.number,
                cover: grant.cover.clone(),
                lasts: grant.lasts,
                until_ms: wake.and_then(|wake| {
                    let since = wake.0.duration_since(UNIX_EPOCH).ok()?;
                    u64::try_from(since.as_millis()).ok()
                }),
                agents: grant.agents.clone(),
            })
            .collect();
        if saved.0 != live {
            saved.0 = live;
        }
    }
}

/// A restored agent gets its saved grants back, but a timed one whose time
/// passed.
fn restore_grants(
    inserted: On<Insert<SavedGrants>>,
    agents: Query<(&SavedGrants, Option<&Grants>)>,
    mut commands: Commands,
) {
    let agent = inserted.entity;
    let Ok((saved, held)) = agents.get(agent) else {
        return;
    };
    if held.is_some_and(|held| !held.is_empty()) {
        return;
    }
    let now = SystemTime::now();
    for grant in &saved.0 {
        let wake = match (grant.lasts, grant.until_ms) {
            (Lasts::Time, until) => {
                let until = until.and_then(|ms| UNIX_EPOCH.checked_add(Duration::from_millis(ms)));
                match until {
                    Some(until) if until > now => Some(WakeAt(until)),
                    _ => continue,
                }
            }
            _ => None,
        };
        let live = Grant {
            number: grant.number,
            cover: grant.cover.clone(),
            lasts: grant.lasts,
            agents: grant.agents.clone(),
        };
        let name = Name::new(format!("grant {}", grant.cover.label()));
        let mut spawned = commands.spawn((live, GrantOf(agent), name));
        if let Some(wake) = wake {
            spawned.insert((wake, KeepAwake(TICK)));
        }
    }
}

/// The status text of the soonest-running-out grant covering `id`: a
/// timed one's countdown, `grant shell 4:12`, else the fewest turns left.
pub fn soonest<'a>(
    id: &AgentId,
    grants: impl IntoIterator<Item = (&'a Grant, Option<&'a WakeAt>)>,
    now: SystemTime,
) -> Option<String> {
    let covering = grants
        .into_iter()
        .filter(|(grant, _)| grant.agents.contains(id));
    let rank = |(grant, wake): &(&Grant, Option<&WakeAt>)| match (grant.lasts, wake) {
        (Lasts::Time, Some(wake)) => (0, wake.left(now).as_secs()),
        (Lasts::Time, None) => (0, 0),
        (Lasts::TurnEnd, _) => (1, 1),
        (Lasts::Turns(n), _) => (1, u64::from(n)),
    };
    let (grant, wake) = covering.min_by_key(rank)?;
    let left = match (grant.lasts, wake) {
        (Lasts::Time, Some(wake)) => clock(wake.left(now)),
        (Lasts::Time, None) => clock(Duration::ZERO),
        (Lasts::Turns(1) | Lasts::TurnEnd, _) => "this turn".to_owned(),
        (Lasts::Turns(n), _) => format!("{n} turns"),
    };
    Some(format!("grant {} {left}", grant.cover.label()))
}

/// The status item: after the activity, gone early on a narrow line.
#[cfg(feature = "tui")]
const ITEM: rig_tui::StatusItem = rig_tui::StatusItem::new("grant", rig_tui::Side::Left, 75, 9);

/// Shows each agent the soonest grant covering it, each time its text
/// changes; a timed grant keeps the loop awake each second for it.
#[cfg(feature = "tui")]
fn show_soonest(
    mut agents: Query<(&AgentId, &mut rig_tui::StatusItems)>,
    grants: Query<(&Grant, Option<&WakeAt>)>,
) {
    let now = SystemTime::now();
    for (id, mut items) in &mut agents {
        let text = soonest(id, grants.iter(), now).unwrap_or_default();
        let shown = items
            .iter()
            .find(|item| item.name == ITEM.name)
            .map(|item| item.text.as_str());
        if shown.unwrap_or_default() != text {
            items.show(ITEM.says(text, rig_tui::Tone::Dim));
        }
    }
}

#[cfg(test)]
mod tests;
