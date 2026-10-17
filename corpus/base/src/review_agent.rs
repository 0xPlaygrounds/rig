//! Plugin corpus: 35. Review agent (spec `35-review-agent.md`).
//!
//! A second agent, on a model of its own, reviews what an agent the user
//! talks to changed, and the agent addresses the findings.
//!
//! - Before a turn's first call that may change files (an
//!   [`Touches::Writes`] tool call), the call is held while a git
//!   snapshot of the working tree is taken ([`Baseline`], on the turn).
//! - When such a turn proposes to end with an answer, a review effect
//!   ([`Review`], named "reviewing", which the agent's status shows) holds
//!   that end. It diffs the working tree against the snapshot and, when
//!   the diff is not empty, spawns a reviewer: a helper agent
//!   ([`SpawnSubagent`], [`Reviewer`]) on [`ReviewSettings::model`] with
//!   its prompt and the read-only tools, briefed with the user's request
//!   and the diff.
//! - Findings go to the agent as a queued message headed "Reviewer (model)
//!   found:", which carries its turn on to address them; "LGTM" is a
//!   notice, and the turn ends.
//! - At most [`ReviewSettings::max_reviews`] reviews follow one user
//!   message, so the fixes are not reviewed again by default.
//! - `/review` reviews the uncommitted diff now; `/review off` and
//!   `/review on` switch auto-review for the session ([`AutoReview`]).
//! - The reviewer is an agent that does one job: its model calls count in
//!   usage under it, and it retires once its review is done.
//! - A user message, Esc, or anything else that carries the turn on
//!   cancels the review, and stops the reviewer.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectKind;
use rig_tools::process::Launcher;
use serde::Deserialize;
use serde_json::json;

/// The kind of a review effect in the record.
pub const REVIEW: &str = "review";

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

/// The `[review]` table of the settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ReviewSettings {
    /// The reviewer's catalog model, as `vendor/model`.
    pub model: String,
    /// The reviewer's system prompt.
    pub prompt: String,
    /// The tools the reviewer may use; every read-only tool when not set.
    pub tools: Option<Vec<String>>,
    /// How many reviews may follow one user message: with 1, the fixes a
    /// review asked for are not reviewed again.
    pub max_reviews: u32,
    /// The longest diff sent, in bytes; a longer one is cut.
    pub max_diff_bytes: usize,
}

impl Default for ReviewSettings {
    fn default() -> Self {
        Self {
            model: "openai/gpt-5.1".to_owned(),
            prompt: DEFAULT_PROMPT.to_owned(),
            tools: None,
            max_reviews: 1,
            max_diff_bytes: 48 * 1024,
        }
    }
}

/// Whether turns are reviewed when they end, for the session.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Saved, Clone, Debug)]
pub struct AutoReview(pub bool);

/// On a reviewer agent: the model it reviews with, as configured.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Reviewer(pub String);

/// On an agent the user talks to: the reviews since the user's last
/// message, and that message.
#[derive(Component, Default, Debug)]
pub struct Reviews {
    /// Reviews started since the user's last message.
    pub done: u32,
    /// The user's last message.
    pub request: String,
}

/// On a turn that ran a call that may change files: the git tree of the
/// working tree before that call, `None` when none could be taken.
#[derive(Component, Debug)]
pub struct Baseline(pub Option<String>);

/// A review: an open effect for the agent it reviews, answered once the
/// review is done. An automatic one is a call of the turn it reviews and
/// holds that turn's proposed end.
#[derive(Component, Debug)]
pub struct Review {
    /// The tree the diff starts from; `HEAD` when `None`.
    pub base: Option<String>,
    /// Asked with `/review`, which says when there is nothing to review.
    pub asked: bool,
}

/// On a review: the agent reviewing.
#[derive(Component, Debug)]
#[relationship(relationship_target = Reviewing)]
pub struct ReviewedBy(pub Entity);

/// On a reviewer: the review it works on.
#[derive(Component, Debug)]
#[relationship_target(relationship = ReviewedBy)]
pub struct Reviewing(Entity);

/// On a hold on a call: it takes the snapshot of this turn.
#[derive(Component)]
struct Snapshotting(Entity);

/// A snapshot's tree, if one could be taken.
struct Tree(Option<String>);

/// A review's diff, or why there is none.
struct Diff(Result<String, String>);

/// The plugin's stage systems, to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewAgentSystems {
    /// In [`Stage::Gate`]: holds a turn's first call that may change files
    /// while the working tree is snapshot, and stops a restored reviewer.
    Gate,
    /// In [`Stage::Review`]: holds a reviewed turn's answered end with a
    /// review.
    Review,
}

/// Reviews by a second agent; see the [crate](self) docs.
#[derive(Default)]
pub struct ReviewAgentPlugin;

impl Plugin for ReviewAgentPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.add_setting::<ReviewSettings>("review")
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .insert_resource(AutoReview(true))
            .register_type::<AutoReview>()
            .register_type::<Reviewer>()
            .register_required_components::<Agent, Reviews>()
            .add_command(
                "review",
                "Review the uncommitted changes now; `/review off` or `on` switches reviews after each turn",
                review_command,
            )
            .configure_sets(AgentPass, ReviewAgentSystems::Gate.in_set(Stage::Gate))
            .configure_sets(AgentPass, ReviewAgentSystems::Review.in_set(Stage::Review))
            .add_systems(
                AgentPass,
                (
                    (snapshot_before_changes, stop_restored_reviewers)
                        .in_set(ReviewAgentSystems::Gate),
                    review_proposed_end.in_set(ReviewAgentSystems::Review),
                ),
            )
            .add_observer(on_user_message)
            .add_observer(snapshot_taken)
            .add_observer(take_diff)
            .add_observer(diff_taken)
            .add_observer(reviewer_spawned)
            .add_observer(reviewer_answered)
            .add_observer(stop_reviewer);
    }
}

/// The pool git is waited on; it runs on threads of its own.
fn pool() -> &'static TaskPool {
    AsyncComputeTaskPool::get_or_init(TaskPool::default)
}

/// A review of `agent`'s changes since `base`.
fn review(agent: Entity, base: Option<String>, asked: bool, model: &str) -> impl Bundle {
    let kind = EffectKind::Custom {
        kind: REVIEW.into(),
        payload: json!({ "model": model, "asked": asked }),
    };
    (
        Name::new("reviewing"),
        Review { base, asked },
        ForAgent(agent),
        EffectRequest {
            kind,
            handler: None,
        },
    )
}

/// The decisions of the reviews, which their observers answer.
type Decisions<'w, 's> = Query<'w, 's, &'static mut Decision, With<Review>>;

/// Answers `review` with `how`, and lets go of the turn end it held.
fn finish(
    decisions: &mut Decisions,
    commands: &mut Commands,
    review: Entity,
    how: serde_json::Value,
) {
    if let Ok(mut decision) = decisions.get_mut(review) {
        decision.decide(Decider::plugin("review-agent"), Outcome::custom(how));
    }
    if let Ok(mut review) = commands.get_entity(review) {
        review.remove::<HoldOf>();
    }
}

/// A user's message: the request the next review names, a new count of
/// reviews, and the end of the agent's open review.
fn on_user_message(
    started: On<Add<Started>>,
    messages: Query<(&Incoming, &ForAgent)>,
    mut agents: Query<&mut Reviews, Without<SpawnedBy>>,
    open: Query<(Entity, &ForAgent), (With<Review>, Without<Outcome>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((message, &ForAgent(agent))) = messages.get(started.entity) else {
        return;
    };
    if message.author.kind != AuthorKind::User || message.command().is_some() {
        return;
    }
    let Ok(mut reviews) = agents.get_mut(agent) else {
        return;
    };
    reviews.done = 0;
    message.text().trim().clone_into(&mut reviews.request);
    for (review, _) in open.iter().filter(|(_, of)| of.0 == agent) {
        commands.entity(review).try_despawn();
        notices.write(Notice::info(agent, "Review cancelled by your message."));
    }
}

/// Holds a turn's first call that may change files while the working tree
/// is snapshot, when auto-review is on.
fn snapshot_before_changes(
    calls: Query<(Entity, &Footprint, &Decision, &CallOf, &ForAgent), (ReadyToStart, ToolCalls)>,
    turns: Query<(), (With<TurnOf>, Without<Baseline>)>,
    agents: Query<(), (With<Reviews>, Without<SpawnedBy>)>,
    (auto, root, processes, wake): (Res<AutoReview>, Res<ProjectRoot>, Res<Processes>, Res<Wake>),
    mut commands: Commands,
) {
    if !auto.0 {
        return;
    }
    for (call, footprint, decision, &CallOf(turn), &ForAgent(agent)) in &calls {
        let first = turns.contains(turn) && agents.contains(agent);
        if !matches!(footprint.0, Touches::Writes(_)) || decision.is_decided() || !first {
            continue;
        }
        commands.entity(turn).insert(Baseline(None));
        let git = Git::new(processes.launcher(), &root.0);
        let work = async move { Tree(blocking_git(move || git.snapshot()).await.ok()) };
        commands.spawn((
            Name::new("review snapshot"),
            Snapshotting(turn),
            HoldOf(call),
            Running::spawn(pool(), &wake, work),
        ));
    }
}

/// Keeps the snapshot on its turn, and lets the call go.
fn snapshot_taken(
    done: On<Add<Done<Tree>>>,
    holds: Query<(&Snapshotting, &Done<Tree>)>,
    mut commands: Commands,
) {
    let Ok((&Snapshotting(turn), Done(Tree(tree)))) = holds.get(done.entity) else {
        return;
    };
    commands.entity(turn).try_insert(Baseline(tree.clone()));
    commands.entity(done.entity).despawn();
}

/// Holds the answered end of a turn that changed files with a review,
/// unless reviews are off or done for the user's message.
fn review_proposed_end(
    proposed: Query<(Entity, &TurnOf, &TurnOutcome, &Baseline), Added<TurnOutcome>>,
    mut agents: Query<&mut Reviews, Without<SpawnedBy>>,
    (auto, settings): (Res<AutoReview>, Res<ReviewSettings>),
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent), outcome, Baseline(base)) in &proposed {
        let (TurnOutcome::Answered(_), Some(base)) = (outcome, base) else {
            continue;
        };
        let Ok(mut reviews) = agents.get_mut(agent) else {
            continue;
        };
        if !auto.0 || reviews.done >= settings.max_reviews {
            continue;
        }
        reviews.done += 1;
        let review = review(agent, Some(base.clone()), false, &settings.model);
        commands.spawn((review, CallOf(turn), HoldOf(turn)));
    }
}

/// Takes the diff of a review that started, off the main thread; it ends
/// as a [`Done<Diff>`] on the review.
fn take_diff(
    started: On<Add<Started>>,
    reviews: Query<&Review>,
    (settings, root, processes, wake): (
        Res<ReviewSettings>,
        Res<ProjectRoot>,
        Res<Processes>,
        Res<Wake>,
    ),
    mut commands: Commands,
) {
    let Ok(review) = reviews.get(started.entity) else {
        return;
    };
    let (base, cap) = (review.base.clone(), settings.max_diff_bytes);
    let git = Git::new(processes.launcher(), &root.0);
    let work = async move { Diff(blocking_git(move || git.diff(base, cap)).await) };
    commands
        .entity(started.entity)
        .insert(Running::spawn(pool(), &wake, work));
}

/// Spawns the reviewer of a diff that is not empty, briefed with it.
fn diff_taken(
    done: On<Add<Done<Diff>>>,
    reviews: Query<(&Review, &ForAgent, &Done<Diff>)>,
    agents: Query<(&Reviews, &ToolAccess)>,
    tools: Query<(&ToolDef, &Footprint)>,
    settings: Res<ReviewSettings>,
    mut decisions: Decisions,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let review = done.entity;
    let Ok((&Review { asked, .. }, &ForAgent(agent), Done(Diff(diff)))) = reviews.get(review)
    else {
        return;
    };
    let Ok((reviews, access)) = agents.get(agent) else {
        return;
    };
    let diff = match diff {
        Ok(diff) if !diff.trim().is_empty() => diff,
        Ok(_) => {
            if asked {
                notices.write(Notice::info(agent, "No uncommitted changes to review."));
            }
            finish(
                &mut decisions,
                &mut commands,
                review,
                json!({ "skipped": "no changes" }),
            );
            return;
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("No review: {why}")));
            finish(
                &mut decisions,
                &mut commands,
                review,
                json!({ "skipped": why }),
            );
            return;
        }
    };
    let allowed = settings.tools.clone().unwrap_or_else(|| {
        let reads = tools.iter().filter(|(def, footprint)| {
            footprint.0.reads_only() && access.allows(def.0.name.as_str())
        });
        reads.map(|(def, _)| def.0.name.to_string()).collect()
    });
    let request = match reviews.request.as_str() {
        "" => "(none given)",
        request => request,
    };
    let brief = format!(
        "The user asked the agent:\n\n{request}\n\nThe agent's changes:\n\n```diff\n{diff}\n```"
    );
    let title = format!("review ({})", short_model(&settings.model));
    let reviewer = SpawnSubagent {
        model: Some(settings.model.clone()),
        tools: Some(allowed),
        ..SpawnSubagent::helper(title)
            .brief(brief)
            .system_prompt(settings.prompt.clone())
            .once()
    };
    commands.spawn((reviewer, ForAgent(agent), ChildOf(review)));
}

/// Marks the agent a review's request spawned as its reviewer, or ends the
/// review with why none was spawned.
fn reviewer_spawned(
    settled: On<Add<Settled>>,
    requests: Query<(&ChildOf, &Outcome, Option<&SpawnedAgent>), With<SpawnSubagent>>,
    reviews: Query<&ForAgent, With<Review>>,
    settings: Res<ReviewSettings>,
    mut decisions: Decisions,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((of, outcome, spawned)) = requests.get(settled.entity) else {
        return;
    };
    let review = of.parent();
    let Ok(&ForAgent(agent)) = reviews.get(review) else {
        return;
    };
    match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(reviewer)), _) => {
            commands
                .entity(reviewer)
                .insert(Reviewer(settings.model.clone()));
            commands.entity(review).try_insert(ReviewedBy(reviewer));
        }
        (None, result) => {
            let why = match result {
                Err(report) => report.message.clone(),
                Ok(_) => "no reviewer was spawned".to_owned(),
            };
            notices.write(Notice::error(agent, format!("No review: {why}")));
            finish(
                &mut decisions,
                &mut commands,
                review,
                json!({ "skipped": why }),
            );
        }
    }
}

/// Hands a reviewer's findings to the agent it reviewed, which carries on
/// with them, or says it found nothing.
fn reviewer_answered(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    reviewers: Query<(&Reviewer, &AgentId, &Reviewing)>,
    reviews: Query<&ForAgent, With<Review>>,
    mut decisions: Decisions,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, &ForAgent(reviewer))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((Reviewer(model), id, &Reviewing(review))) = reviewers.get(reviewer) else {
        return;
    };
    let Ok(&ForAgent(agent)) = reviews.get(review) else {
        return;
    };
    let model = short_model(model);
    let answer = match outcome {
        TurnOutcome::Answered(message) => message.final_answer().unwrap_or_default(),
        TurnOutcome::Failed(why) => {
            notices.write(Notice::error(agent, format!("The review failed: {why}")));
            finish(
                &mut decisions,
                &mut commands,
                review,
                json!({ "failed": why }),
            );
            return;
        }
        TurnOutcome::Stopped(_) => {
            finish(
                &mut decisions,
                &mut commands,
                review,
                json!({ "stopped": true }),
            );
            return;
        }
    };
    let answer = answer.trim();
    if answer.is_empty() || is_lgtm(answer) {
        notices.write(Notice::info(agent, format!("Reviewer ({model}): LGTM.")));
        finish(
            &mut decisions,
            &mut commands,
            review,
            json!({ "lgtm": true }),
        );
        return;
    }
    finish(
        &mut decisions,
        &mut commands,
        review,
        json!({ "findings": answer }),
    );
    let text = format!(
        "Reviewer ({model}) found:\n\n{answer}\n\nAddress these findings, or say why one does not apply."
    );
    let author = Author::agent(id.clone(), None).titled(REVIEW);
    commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue).with_author(author));
}

/// Stops the reviewer of a review that went away while it works, as when a
/// user message carried the reviewed turn on.
fn stop_reviewer(
    removed: On<Remove<ReviewedBy>>,
    reviews: Query<&ReviewedBy>,
    busy: Query<(), With<ActiveTurn>>,
    mut commands: Commands,
) {
    if let Ok(&ReviewedBy(reviewer)) = reviews.get(removed.entity)
        && busy.contains(reviewer)
    {
        commands.trigger(StopTurn {
            entity: reviewer,
            reason: Some("the review was cancelled".to_owned()),
        });
    }
}

/// A restored reviewer's turn is stopped, so it retires: the review it was
/// on is gone.
fn stop_restored_reviewers(
    turns: Query<(Entity, &TurnOf), (With<Restored>, Without<TurnOutcome>)>,
    reviewers: Query<(), With<Reviewer>>,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent)) in &turns {
        if reviewers.contains(agent) {
            let stopped = TurnOutcome::Stopped(Some("restart".to_owned()));
            commands.entity(turn).insert(stopped);
        }
    }
}

/// `/review`: reviews the uncommitted changes now; `/review off` and
/// `/review on` switch reviews after each turn.
fn review_command(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Reviews, Has<ActiveTurn>), Without<SpawnedBy>>,
    open: Query<&ForAgent, (With<Review>, Without<Outcome>)>,
    (mut auto, settings): (ResMut<AutoReview>, Res<ReviewSettings>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    match args.args.as_str() {
        "off" => {
            auto.0 = false;
            let off = "Turns are not reviewed for the rest of the session.";
            notices.write(Notice::info(agent, off));
        }
        "on" => {
            auto.0 = true;
            notices.write(Notice::info(agent, "Turns that change files are reviewed."));
        }
        "" => match agents.get_mut(agent) {
            Err(_) => {
                let why = "Only an agent you talk to is reviewed.";
                notices.write(Notice::error(agent, why));
            }
            Ok((_, true)) => {
                notices.write(Notice::turn_running(agent));
            }
            Ok(_) if open.iter().any(|of| of.0 == agent) => {
                notices.write(Notice::error(agent, "A review is running already."));
            }
            Ok((mut reviews, false)) => {
                reviews.done += 1;
                commands.spawn(review(agent, None, true, &settings.model));
            }
        },
        other => {
            let why = format!("/review takes `on`, `off` or nothing, not `{other}`.");
            notices.write(Notice::error(agent, why));
        }
    }
}

/// Whether a reviewer's answer says there is nothing to fix.
fn is_lgtm(answer: &str) -> bool {
    let mut words = answer
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty());
    words
        .next()
        .is_some_and(|word| word.eq_ignore_ascii_case("lgtm"))
}

/// `vendor/model` without its vendor.
fn short_model(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

/// `work`, which runs git, on a thread of its own.
async fn blocking_git<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    blocking(move || work().map_err(ToolExecutionError::other))
        .await
        .map_err(|error| error.message().to_owned())
}

/// Git in the project, started as every child process is.
struct Git {
    launcher: Launcher,
    dir: PathBuf,
}

impl Git {
    fn new(launcher: Launcher, dir: &Path) -> Self {
        Self {
            launcher,
            dir: dir.to_owned(),
        }
    }

    /// What `git` says to `args`, on the index `index` when given.
    fn run(&self, index: Option<&Path>, args: &[&str]) -> Result<String, String> {
        let mut command = self.launcher.command("git");
        command.arg("-C").arg(&self.dir).args(args);
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

    /// The working tree, untracked files that git does not ignore
    /// included, written as a tree through a temporary copy of the index,
    /// so the user's index is left as it is.
    fn snapshot(&self) -> Result<String, String> {
        static TAKEN: AtomicU64 = AtomicU64::new(0);
        let git_dir = self.run(None, &["rev-parse", "--absolute-git-dir"])?;
        let git_dir = PathBuf::from(git_dir.trim());
        let n = TAKEN.fetch_add(1, Ordering::Relaxed);
        let index = git_dir.join(format!("rig-review-index-{}-{n}", std::process::id()));
        // The copy keeps what git knows of unchanged files; without one,
        // every file is read again.
        std::fs::copy(git_dir.join("index"), &index).ok();
        let tree = self
            .run(Some(&index), &["add", "--all"])
            .and_then(|_| self.run(Some(&index), &["write-tree"]));
        std::fs::remove_file(&index).ok();
        Ok(tree?.trim().to_owned())
    }

    /// The diff from the tree `base`, else from `HEAD`, to the working
    /// tree, cut to `cap` bytes.
    fn diff(&self, base: Option<String>, cap: usize) -> Result<String, String> {
        let base = match base {
            Some(base) => base,
            None => self
                .run(None, &["rev-parse", "--verify", "--quiet", "HEAD^{tree}"])
                .map_or_else(|_| EMPTY_TREE.to_owned(), |head| head.trim().to_owned()),
        };
        let now = self.snapshot()?;
        let args = ["diff", "--no-color", "--no-ext-diff", &base, &now];
        let mut diff = self.run(None, &args)?;
        if diff.len() > cap {
            let end = (0..=cap).rev().find(|&end| diff.is_char_boundary(end));
            diff.truncate(end.unwrap_or_default());
            diff.push_str("\n[The diff is cut here; read the changed files for the rest.]");
        }
        Ok(diff)
    }
}

#[cfg(test)]
mod tests;
