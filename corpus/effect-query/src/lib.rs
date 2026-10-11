//! Plugin corpus: 55. Effect log query tool (spec `55-effect-query.md`).
//!
//! The `effects` and `effects_detail` tools, and `/effects`, read the
//! session's effect log (`effects.jsonl`, written by rig-telemetry's
//! `EffectLogPlugin`) back from disk, so they see every model and tool
//! call of the session, plugins' own and those from before a `/reload`.
//! They only read it.

mod query;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use bevy_tasks::AsyncComputeTaskPool;
use rig_cassette::effect_log::{EffectLog, jsonl};
use rig_cassette::journal::EFFECT_LOG;
use rig_harness::prelude::*;

pub use query::{DetailArgs, Filter, Status};

/// The tool that lists effects.
pub const EFFECTS_TOOL: &str = "effects";
/// The tool that shows one effect.
pub const DETAIL_TOOL: &str = "effects_detail";

const EFFECTS_DESCRIPTION: &str = "List the model and tool calls of this session, every agent's, \
    from its effect log: id, agent, turn, kind, name, status, request and result sizes, and the \
    model and cost of model calls. Filters narrow it; the latest `limit` rows are shown.";
const DETAIL_DESCRIPTION: &str = "Show one effect of `effects` in full: its request and its \
    result or error, cut to 20000 characters.";

#[derive(Default)]
pub struct EffectQueryPlugin;

impl Plugin for EffectQueryPlugin {
    fn build(&self, app: &mut App) {
        let log = app
            .world()
            .get_resource::<SessionPaths>()
            .map(|paths| paths.path().join(EFFECT_LOG));
        let read_only = || ToolOptions {
            rules: &[],
            footprint: Footprint::ReadOnly,
        };
        app.insert_resource(LogFile(log))
            .add_open_tool(EFFECTS_TOOL, EFFECTS_DESCRIPTION, read_only(), on_effects)
            .add_open_tool(DETAIL_TOOL, DETAIL_DESCRIPTION, read_only(), on_detail)
            .add_command(
                "effects",
                "List the session's model and tool calls: /effects [status|kind|name|key=value…]",
                effects_command,
            )
            .add_observer(show_listing);
    }
}

/// Where the effect log is, when the session has a directory.
#[derive(Resource, Clone)]
struct LogFile(Option<PathBuf>);

impl LogFile {
    /// The log as it is on disk; an empty one before its first record.
    fn read(&self) -> Result<EffectLog, String> {
        let Some(path) = &self.0 else {
            return Err("This session has no directory, so no effect log.".to_owned());
        };
        match jsonl::read(path) {
            Ok(log) => Ok(log),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(EffectLog::default()),
            Err(error) => Err(format!("Reading the effect log failed: {error}")),
        }
    }
}

/// Each agent's name by its id, for the table.
fn agent_names(agents: &Query<(&AgentId, &Name)>) -> Arc<HashMap<String, String>> {
    Arc::new(
        agents
            .iter()
            .map(|(id, name)| (id.0.clone(), name.as_str().to_owned()))
            .collect(),
    )
}

/// Runs `answer` on the effect log off the main thread and ends `call`
/// with what it says.
fn answer_call(
    commands: &mut Commands,
    wake: &Wake,
    call: Entity,
    file: LogFile,
    answer: impl FnOnce(EffectLog) -> Result<String, String> + Send + 'static,
) {
    let work = blocking(move || {
        file.read()
            .and_then(answer)
            .map_err(ToolExecutionError::other)
    });
    let work = async move {
        match work.await {
            Ok(text) => ToolResult::success(text.into()),
            Err(error) => ToolResult::failed(error),
        }
    };
    let pool = AsyncComputeTaskPool::get_or_init(Default::default);
    commands
        .entity(call)
        .insert(Running::answering(pool, wake, work));
}

fn on_effects(
    mut called: On<ToolCalled<Filter>>,
    agents: Query<(&AgentId, &Name)>,
    file: Res<LogFile>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let names = agent_names(&agents);
    let filter = std::mem::take(&mut called.args);
    answer_call(
        &mut commands,
        &wake,
        called.call,
        file.clone(),
        move |log| Ok(query::table(&log, &names, &filter)),
    );
}

fn on_detail(
    called: On<ToolCalled<DetailArgs>>,
    file: Res<LogFile>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let id = called.args.id;
    answer_call(
        &mut commands,
        &wake,
        called.call,
        file.clone(),
        move |log| query::detail(&log, id).ok_or_else(|| format!("No effect {id} is in the log.")),
    );
}

/// A `/effects` listing being read, for the agent it was typed for.
struct Listing {
    agent: Entity,
    text: Result<String, String>,
}

/// `/effects [filter]`: the same table as the tool, as a notice.
fn effects_command(
    In(args): In<CommandArgs>,
    agents: Query<(&AgentId, &Name)>,
    file: Res<LogFile>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    if args.args.trim() == "slow" {
        notices.write(Notice::error(
            agent,
            "/effects slow needs each effect's duration, which the effect log does not record.",
        ));
        return;
    }
    let filter = match Filter::parse(&args.args) {
        Ok(filter) => filter,
        Err(why) => {
            notices.write(Notice::error(agent, format!("/effects: {why}")));
            return;
        }
    };
    let (names, file) = (agent_names(&agents), file.clone());
    let read = blocking(move || Ok(file.read().map(|log| query::table(&log, &names, &filter))));
    let work = async move {
        let text = read.await.map_err(|error| error.to_string()).flatten();
        Listing { agent, text }
    };
    let pool = AsyncComputeTaskPool::get_or_init(Default::default);
    commands.spawn(Running::spawn(pool, &wake, work));
}

fn show_listing(
    done: On<Add<Done<Listing>>>,
    listings: Query<&Done<Listing>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if let Ok(Done(listing)) = listings.get(done.entity) {
        notices.write(match &listing.text {
            Ok(text) => Notice::info(listing.agent, text.clone()),
            Err(why) => Notice::error(listing.agent, why.clone()),
        });
    }
    commands.entity(done.entity).despawn();
}

#[cfg(test)]
mod tests;
