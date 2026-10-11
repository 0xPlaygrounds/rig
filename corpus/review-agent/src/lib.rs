//! Plugin corpus: 35. Review agent (spec `35-review-agent.md`).
//!
//! A second agent, on a model of its own, reviews what an agent the user
//! talks to changed in a turn, and the agent addresses the findings.
//!
//! - Before a turn's first model request, a git snapshot of the working
//!   tree is taken (a tree written through a temporary index, so the
//!   user's index is untouched) as a call of the turn, which holds the
//!   request until it is done ([`PrepareRequest`]).
//! - When the turn ends answered, a second snapshot is diffed against the
//!   first. A diff that is not empty spawns a reviewer agent
//!   ([`Reviewer`], `SpawnedBy` the agent) on [`ReviewConfig::model`], with
//!   its prompt and read-only tools, briefed with the user's request and
//!   the diff.
//! - The reviewer's findings go to the agent as a queued message, headed
//!   "Reviewer (model) found:", which starts a turn; "LGTM" is a notice.
//! - At most [`ReviewConfig::max_reviews`] reviews follow one user message,
//!   so the fixes are not reviewed again by default.
//! - `/review` reviews the uncommitted diff now; `/review off` and
//!   `/review on` switch auto-review for the session ([`AutoReview`]).
//! - The reviewer is an agent: its model calls count in usage under it.
//! - While a review runs the agent has [`Reviewing`], shown as
//!   "reviewing…" in its status line; a user message cancels it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_harness::prelude::*;

/// The tree of an empty repository, what a repository without commits
/// diffs against.
const EMPTY_TREE: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

/// The reviewer's system prompt unless configured.
pub const DEFAULT_PROMPT: &str = "You review the changes another coding agent just made, \
    before the user looks at them. You get the user's request and the diff; read the \
    files with your read-only tools when the diff is not enough. Report only what is worth \
    fixing: bugs, what the request asked that the change misses or gets wrong, and changes \
    that break the code around them. Write each finding on its own line with its file and \
    what to change, and nothing else. When nothing is worth fixing, answer exactly LGTM.";

/// How reviews run. Insert it before adding the plugin to change it.
#[derive(Resource, Clone, Debug)]
pub struct ReviewConfig {
    /// The reviewer's catalog model, as `vendor/model`.
    pub model: String,
    /// The reviewer's system prompt.
    pub prompt: String,
    /// The tools the reviewer may use; every read-only tool when `None`.
    pub tools: Option<Vec<String>>,
    /// How many reviews may follow one user message: with 1, the fixes a
    /// review asked for are not reviewed again.
    pub max_reviews: u32,
    /// The git working tree reviewed.
    pub dir: PathBuf,
    /// The longest diff sent, in bytes; a longer one is cut.
    pub max_diff_bytes: usize,
}

impl Default for ReviewConfig {
    fn default() -> Self {
        Self {
            model: "openai/gpt-5.1".to_owned(),
            prompt: DEFAULT_PROMPT.to_owned(),
            tools: None,
            max_reviews: 1,
            dir: PathBuf::from("."),
            max_diff_bytes: 48 * 1024,
        }
    }
}

/// The reviewer's model, connected on the first review and kept for the
/// rest of the session; insert one to review on a model of your own.
#[derive(Resource, Clone, Default)]
pub struct ReviewerModel(pub Option<Connection>);

/// Whether turns are reviewed when they end, for the session.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Saved, Clone, Debug)]
pub struct AutoReview(pub bool);

/// On a reviewer agent: the model it reviews with, as it was named.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Reviewer(pub String);

/// On an agent while a review of its changes runs: the entity doing it,
/// the diff being taken or the reviewer.
#[derive(Component, Debug)]
pub struct Reviewing(pub Entity);

/// On an agent the user talks to: its reviews since the user's last
/// message, that message, and the snapshot its running turn started from.
#[derive(Component, Default, Debug)]
pub struct Reviews {
    /// Reviews started since the user's last message.
    pub done: u32,
    /// The user's last message.
    pub request: String,
    /// The turn the snapshot was taken for, and the snapshot's tree, if
    /// one could be taken.
    baseline: Option<(Entity, Option<String>)>,
}

/// A git tree of the working tree, or why there is none.
struct Snapshot(Result<String, String>);

/// A diff to review, or why there is none.
struct Diff(Result<String, String>);

/// On a call of a turn: it takes the turn's snapshot.
#[derive(Component)]
struct TakingSnapshot;

/// On an entity of its own: it takes the diff `agent` is reviewed on.
#[derive(Component)]
struct TakingDiff {
    agent: Entity,
    /// Asked with `/review`, which says when there is nothing to review.
    asked: bool,
}

/// Reviews by a second agent; see the [crate](self) docs.
#[derive(Default)]
pub struct ReviewAgentPlugin;

impl Plugin for ReviewAgentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReviewConfig>()
            .init_resource::<ReviewerModel>()
            .insert_resource(AutoReview(true))
            .register_required_components::<Agent, Reviews>()
            .add_command(
                "review",
                "Review the uncommitted changes now; `/review off` or `on` switches reviews after each turn",
                review_command,
            )
            .add_observer(on_user_message)
            .add_observer(take_snapshot)
            .add_observer(snapshot_taken)
            .add_observer(review_after_turn)
            .add_observer(start_reviewer)
            .add_observer(reviewer_answered)
            .add_observer(keep_reviewer_idle);
        #[cfg(feature = "tui")]
        app.add_systems(PostUpdate, show_reviewing.in_set(rig_tui::StatusSystems));
    }
}

/// The pool the snapshots wait on; the git commands run on threads of
/// their own.
fn pool() -> &'static TaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// A user's message: the request the next review names, a new count of
/// reviews, and the end of a review that runs.
fn on_user_message(
    sent: On<Deliver>,
    mut agents: Query<(&mut Reviews, Option<&Reviewing>), Without<SpawnedBy>>,
    busy: Query<(), With<ActiveTurn>>,
    takings: Query<(), With<TakingDiff>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if sent.origin.kind != OriginKind::User || sent.command().is_some() {
        return;
    }
    let agent = sent.entity;
    let Ok((mut reviews, reviewing)) = agents.get_mut(agent) else {
        return;
    };
    reviews.done = 0;
    reviews.request = sent.text.trim().to_owned();
    if let Some(&Reviewing(by)) = reviewing {
        commands.entity(agent).remove::<Reviewing>();
        if busy.contains(by) {
            commands.trigger(Interrupt { entity: by });
        } else if takings.contains(by) {
            commands.entity(by).despawn();
        }
        notices.write(Notice::info(agent, "Review cancelled by your message."));
    }
}

/// Before a turn's first request, while reviews are on, takes a snapshot
/// of the working tree as a call of the turn, which holds the request.
fn take_snapshot(
    prepare: On<PrepareRequest>,
    agents: Query<&Reviews, Without<SpawnedBy>>,
    calls: Query<&Calls>,
    taking: Query<(), With<TakingSnapshot>>,
    auto: Res<AutoReview>,
    config: Res<ReviewConfig>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (turn, agent) = (prepare.entity, prepare.agent);
    let Ok(reviews) = agents.get(agent) else {
        return;
    };
    let taken = reviews.baseline.as_ref().is_some_and(|(of, _)| *of == turn);
    let started = calls
        .get(turn)
        .is_ok_and(|calls| calls.iter().any(|call| taking.contains(call)));
    if !auto.0 || taken || started {
        return;
    }
    let dir = config.dir.clone();
    let work = async move { Snapshot(run_git(move || snapshot(&dir)).await) };
    commands.spawn((
        Name::new("snapshotting"),
        TakingSnapshot,
        CallOf(turn),
        Running::spawn(pool(), &wake, work),
    ));
}

/// Keeps the turn's snapshot on its agent and carries the turn on.
fn snapshot_taken(
    done: On<Add<Done<Snapshot>>>,
    calls: Query<(&CallOf, &Done<Snapshot>)>,
    turns: Query<&TurnOf>,
    mut agents: Query<&mut Reviews>,
    mut commands: Commands,
) {
    let Ok((&CallOf(turn), Done(Snapshot(tree)))) = calls.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    if let Ok(&TurnOf(agent)) = turns.get(turn)
        && let Ok(mut reviews) = agents.get_mut(agent)
    {
        reviews.baseline = Some((turn, tree.as_ref().ok().cloned()));
    }
    commands.trigger(CallModel { entity: turn });
}

/// After an answered turn of an agent the user talks to, takes the diff
/// since the turn's snapshot, unless reviews are off or done for the
/// user's message.
fn review_after_turn(
    end: On<TurnEnded>,
    mut agents: Query<(&mut Reviews, Has<Reviewing>), Without<SpawnedBy>>,
    auto: Res<AutoReview>,
    config: Res<ReviewConfig>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let agent = end.entity;
    let Ok((mut reviews, reviewing)) = agents.get_mut(agent) else {
        return;
    };
    let Some((_, Some(base))) = reviews.baseline.take() else {
        return;
    };
    let answered = matches!(end.outcome, TurnOutcome::Answered(_));
    if !answered || !auto.0 || reviewing || reviews.done >= config.max_reviews {
        return;
    }
    take_diff(&mut commands, &wake, &config, (agent, false), Some(base));
}

/// Takes the diff `agent` is reviewed on, from `base` (a tree), else from
/// `HEAD`, to the working tree now.
fn take_diff(
    commands: &mut Commands,
    wake: &Wake,
    config: &ReviewConfig,
    (agent, asked): (Entity, bool),
    base: Option<String>,
) {
    let (dir, cap) = (config.dir.clone(), config.max_diff_bytes);
    let work = async move { Diff(run_git(move || diff(&dir, base, cap)).await) };
    let taking = commands
        .spawn((
            Name::new("review diff"),
            TakingDiff { agent, asked },
            Running::spawn(pool(), wake, work),
        ))
        .id();
    commands.entity(agent).insert(Reviewing(taking));
}

/// Spawns the reviewer of a diff that is not empty, and briefs it.
fn start_reviewer(
    done: On<Add<Done<Diff>>>,
    takings: Query<(&TakingDiff, &Done<Diff>)>,
    mut agents: Query<(&mut Reviews, &Reviewing)>,
    tools: Query<(&ToolDef, &Footprint)>,
    config: Res<ReviewConfig>,
    models: Res<Models>,
    mut reviewer_model: ResMut<ReviewerModel>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let taking = done.entity;
    let Ok((&TakingDiff { agent, asked }, Done(Diff(diff)))) = takings.get(taking) else {
        return;
    };
    commands.entity(taking).despawn();
    let Ok((mut reviews, &Reviewing(by))) = agents.get_mut(agent) else {
        return;
    };
    if by != taking {
        return;
    }
    let diff = match diff {
        Ok(diff) if !diff.trim().is_empty() => diff,
        Ok(_) => {
            commands.entity(agent).remove::<Reviewing>();
            if asked {
                notices.write(Notice::info(agent, "No uncommitted changes to review."));
            }
            return;
        }
        Err(why) => {
            commands.entity(agent).remove::<Reviewing>();
            notices.write(Notice::error(agent, format!("No review: {why}")));
            return;
        }
    };
    let connection = match &reviewer_model.0 {
        Some(connection) => connection.clone(),
        None => match models.connect(&config.model) {
            Ok(connection) => reviewer_model.0.insert(connection).clone(),
            Err(why) => {
                commands.entity(agent).remove::<Reviewing>();
                notices.write(Notice::error(agent, format!("No review: {why}")));
                return;
            }
        },
    };
    let allowed: Vec<String> = match &config.tools {
        Some(named) => named.clone(),
        None => tools
            .iter()
            .filter(|(_, footprint)| **footprint == Footprint::ReadOnly)
            .map(|(def, _)| def.0.name.as_str().to_owned())
            .collect(),
    };
    let model = connection.spec.display_name.clone();
    let choice = ModelChoice(connection.spec.reference());
    let reviewer = commands
        .spawn((
            Agent,
            Name::new(format!("review ({model})")),
            Reviewer(model),
            SpawnedBy(agent),
            connection,
            choice,
            SystemPrompt(config.prompt.clone()),
            ToolAccess::Only(allowed),
        ))
        .id();
    let request = match reviews.request.as_str() {
        "" => "(none given)",
        request => request,
    };
    let brief = format!(
        "The user asked the agent:\n\n{request}\n\nThe agent's changes:\n\n```diff\n{diff}\n```"
    );
    let brief = Deliver::new(reviewer, brief, DeliveryMode::Queue);
    commands.trigger(brief.with_origin(Origin::plugin("review")));
    commands.entity(agent).insert(Reviewing(reviewer));
    reviews.done += 1;
}

/// Hands a reviewer's findings to the agent it reviewed, which starts a
/// turn on them, or says it found nothing.
fn reviewer_answered(
    end: On<TurnEnded>,
    reviewers: Query<(&Reviewer, &SpawnedBy, &AgentId)>,
    reviewing: Query<&Reviewing>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let reviewer = end.entity;
    let Ok((Reviewer(model), &SpawnedBy(agent), id)) = reviewers.get(reviewer) else {
        return;
    };
    if !reviewing.get(agent).is_ok_and(|by| by.0 == reviewer) {
        return;
    }
    commands.entity(agent).remove::<Reviewing>();
    let answer = match &end.outcome {
        TurnOutcome::Answered(message) => final_answer(message),
        TurnOutcome::Failed(why) => {
            notices.write(Notice::error(agent, format!("The review failed: {why}")));
            return;
        }
        TurnOutcome::Stopped => return,
    };
    let Some(answer) = answer.filter(|answer| !answer.trim().is_empty()) else {
        notices.write(Notice::error(agent, "The reviewer gave no answer."));
        return;
    };
    if is_lgtm(&answer) {
        notices.write(Notice::info(agent, format!("Reviewer ({model}): LGTM.")));
        return;
    }
    let text = format!(
        "Reviewer ({model}) found:\n\n{}\n\nAddress these findings, or say why one does not apply.",
        answer.trim()
    );
    let origin = Origin::agent(id.clone(), None).titled("review");
    commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue).with_origin(origin));
}

/// Whether a reviewer's answer says there is nothing to fix.
fn is_lgtm(answer: &str) -> bool {
    let first = answer
        .split(|c: char| !c.is_alphanumeric())
        .find(|word| !word.is_empty());
    first.is_some_and(|word| word.eq_ignore_ascii_case("lgtm"))
}

/// A restored reviewer stays idle: the review it was on is gone.
fn keep_reviewer_idle(mut restored: On<Restored>, reviewers: Query<(), With<Reviewer>>) {
    if reviewers.contains(restored.entity) {
        restored.event_mut().resume = false;
    }
}

/// `/review`: reviews the uncommitted changes now; `/review off` and
/// `/review on` switch reviews after each turn.
fn review_command(
    In(args): In<CommandArgs>,
    agents: Query<(Has<ActiveTurn>, Has<Reviewing>), Without<SpawnedBy>>,
    mut auto: ResMut<AutoReview>,
    config: Res<ReviewConfig>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    match args.args.trim() {
        "off" => {
            auto.0 = false;
            notices.write(Notice::info(
                agent,
                "Turns are not reviewed for the rest of the session.",
            ));
        }
        "on" => {
            auto.0 = true;
            notices.write(Notice::info(agent, "Turns that change files are reviewed."));
        }
        "" => match agents.get(agent) {
            Err(_) => {
                notices.write(Notice::error(
                    agent,
                    "Only an agent you talk to is reviewed.",
                ));
            }
            Ok((true, _)) => {
                notices.write(Notice::turn_running(agent));
            }
            Ok((_, true)) => {
                notices.write(Notice::error(agent, "A review is running already."));
            }
            Ok(_) => take_diff(&mut commands, &wake, &config, (agent, true), None),
        },
        other => {
            notices.write(Notice::error(
                agent,
                format!("/review takes `on`, `off` or nothing, not `{other}`."),
            ));
        }
    }
}

/// "reviewing…" in the status line of an agent while it is reviewed.
#[cfg(feature = "tui")]
fn show_reviewing(
    mut items: Query<&mut rig_tui::StatusItems>,
    added: Query<Entity, Added<Reviewing>>,
    mut removed: RemovedComponents<Reviewing>,
    reviewing: Query<(), With<Reviewing>>,
) {
    use rig_tui::{Side, StatusItem, Tone};
    /// After the agent's status; gone before it.
    const REVIEWING: StatusItem = StatusItem::at(Side::Left, 55, 14);
    for agent in &added {
        if let Ok(mut items) = items.get_mut(agent) {
            items.show(REVIEWING.says("reviewing…", Tone::Cyan));
        }
    }
    for agent in removed.read() {
        if !reviewing.contains(agent)
            && let Ok(mut items) = items.get_mut(agent)
        {
            items.show(REVIEWING);
        }
    }
}

/// `work` on a thread of its own.
async fn run_git<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    let work = move || work().map_err(ToolExecutionError::other);
    blocking(work)
        .await
        .map_err(|error| error.message().to_owned())
}

/// What `git` says to `args` in `dir`, on the index `index` when given.
fn git(dir: &Path, index: Option<&Path>, args: &[&str]) -> Result<String, String> {
    let mut command = Command::new("git");
    command.arg("-C").arg(dir).args(args);
    if let Some(index) = index {
        command.env("GIT_INDEX_FILE", index);
    }
    let output = command
        .output()
        .map_err(|error| format!("git could not run: {error}"))?;
    if !output.status.success() {
        let said = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git {}: {}", args.join(" "), said.trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The working tree of `dir`, untracked files that git does not ignore
/// included, written as a git tree through a temporary copy of the index,
/// so the user's index is left as it is.
fn snapshot(dir: &Path) -> Result<String, String> {
    static TAKEN: AtomicU64 = AtomicU64::new(0);
    let git_dir = git(dir, None, &["rev-parse", "--absolute-git-dir"])?;
    let git_dir = PathBuf::from(git_dir.trim());
    let n = TAKEN.fetch_add(1, Ordering::Relaxed);
    let index = git_dir.join(format!("rig-review-index-{}-{n}", std::process::id()));
    // The copy keeps what git knows of unchanged files; without one, every
    // file is read again.
    std::fs::copy(git_dir.join("index"), &index).ok();
    let tree = git(dir, Some(&index), &["add", "--all"])
        .and_then(|_| git(dir, Some(&index), &["write-tree"]));
    std::fs::remove_file(&index).ok();
    Ok(tree?.trim().to_owned())
}

/// The diff from the tree `base`, else from `HEAD`, to the working tree,
/// cut to `cap` bytes.
fn diff(dir: &Path, base: Option<String>, cap: usize) -> Result<String, String> {
    let base = match base {
        Some(base) => base,
        None => git(
            dir,
            None,
            &["rev-parse", "--verify", "--quiet", "HEAD^{tree}"],
        )
        .map(|head| head.trim().to_owned())
        .unwrap_or_else(|_| EMPTY_TREE.to_owned()),
    };
    let now = snapshot(dir)?;
    let mut diff = git(
        dir,
        None,
        &["diff", "--no-color", "--no-ext-diff", &base, &now],
    )?;
    if diff.len() > cap {
        let mut end = cap;
        while !diff.is_char_boundary(end) {
            end -= 1;
        }
        diff.truncate(end);
        diff.push_str("\n[The diff is cut here; read the changed files for the rest.]");
    }
    Ok(diff)
}

#[cfg(test)]
mod tests;
