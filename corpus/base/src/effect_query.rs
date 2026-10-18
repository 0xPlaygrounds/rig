//! Plugin corpus: 55. Effect log query tool (spec `55-effect-query.md`).
//!
//! The `effects` and `effects_detail` tools, and `/effects`, read the
//! session's effect log back through `rig_basics::telemetry`'s [`EffectLog`], so
//! they see every model and tool call of the session, plugins' own and
//! those from before a `/reload`, with the turn each was made in. They
//! only read it.

mod query;

use std::collections::HashMap;
use std::sync::Arc;

use bevy_tasks::AsyncComputeTaskPool;
use rig_basics::telemetry::effect_log::EffectLog;
use rig_core::effect::EffectRecord;
use rig_harness::prelude::*;

pub use query::{DetailArgs, Filter, Status};

/// The tool that lists effects.
pub const EFFECTS_TOOL: &str = "effects";
/// The tool that shows one effect.
pub const DETAIL_TOOL: &str = "effects_detail";

const EFFECTS_DESCRIPTION: &str = "List the model and tool calls of this session, every agent's, \
    from its effect log: id, agent, turn, kind, name, status, start time, duration, request and \
    result sizes, and the model and cost of model calls. Filters narrow it; the latest `limit` \
    rows are shown, or with `slowest` the slowest.";
const DETAIL_DESCRIPTION: &str = "Show one effect of `effects` in full: its request and its \
    result or error, cut to 20000 characters.";

#[derive(Default)]
pub struct EffectQueryPlugin;

impl Plugin for EffectQueryPlugin {
    fn build(&self, app: &mut App) {
        let read_only = || ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.add_open_tool(EFFECTS_TOOL, EFFECTS_DESCRIPTION, read_only(), on_effects)
            .add_open_tool(DETAIL_TOOL, DETAIL_DESCRIPTION, read_only(), on_detail)
            .add_command(
                "effects",
                "List the session's model and tool calls: /effects [slow|status|kind|name|key=value…]",
                effects_command,
            )
            .add_observer(show_listing);
    }
}

/// The live agents' names by their ids.
type Names = Arc<HashMap<String, String>>;

fn names(agents: &Query<(&AgentId, &Name)>) -> Names {
    let names = agents
        .iter()
        .map(|(id, name)| (id.0.clone(), name.as_str().to_owned()));
    Arc::new(names.collect())
}

/// Reads the whole log off the main thread, then `answer` makes the text
/// of what it holds.
async fn read_log(
    log: Option<EffectLog>,
    answer: impl FnOnce(Vec<EffectRecord>) -> Result<String, String> + Send + 'static,
) -> Result<String, ToolExecutionError> {
    let log = log.ok_or_else(|| ToolExecutionError::other("This session keeps no effect log."))?;
    blocking(move || {
        let records = log.read().map_err(|error| {
            ToolExecutionError::other(format!("Reading the effect log failed: {error}"))
        })?;
        answer(records).map_err(ToolExecutionError::other)
    })
    .await
}

/// Answers `call` off the main thread with what `answer` says of the log.
fn answer_call(
    commands: &mut Commands,
    wake: &Wake,
    call: Entity,
    log: Option<&EffectLog>,
    answer: impl FnOnce(Vec<EffectRecord>) -> Result<String, String> + Send + 'static,
) {
    let read = read_log(log.cloned(), answer);
    let work = async move {
        match read.await {
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
    log: Option<Res<EffectLog>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (names, filter) = (names(&agents), std::mem::take(&mut called.args));
    let answer = move |records: Vec<_>| Ok(query::table(&records, &names, &filter));
    answer_call(&mut commands, &wake, called.call, log.as_deref(), answer);
}

fn on_detail(
    called: On<ToolCalled<DetailArgs>>,
    log: Option<Res<EffectLog>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let id = called.args.id;
    let answer = move |records: Vec<_>| {
        query::detail(&records, id).ok_or_else(|| format!("No effect {id} is in the log."))
    };
    answer_call(&mut commands, &wake, called.call, log.as_deref(), answer);
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
    log: Option<Res<EffectLog>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = args.agent;
    let filter = match Filter::parse(&args.args) {
        Ok(filter) => filter,
        Err(why) => {
            notices.write(Notice::error(agent, format!("/effects: {why}")));
            return;
        }
    };
    let names = names(&agents);
    let read = read_log(log.as_deref().cloned(), move |records| {
        Ok(query::table(&records, &names, &filter))
    });
    let work = async move {
        let text = read.await.map_err(|error| error.to_string());
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
