//! OC-04: issue and pull request references in prompts. A user's message
//! that names `#142`, `!87` (or `PR#87`) or a tracker URL, outside code
//! spans and fenced blocks, is held while each reference loads from the
//! tracker in parallel (5 s each), then goes with a block per reference
//! after its text: title, state, labels, author, body, the newest
//! comments, and for a pull request its changed files and unresolved
//! review comments. A reference that fails goes as one line, `could not
//! load #142: <why>`.
//!
//! The tracker is the `origin` remote's (GitHub or GitLab), its API and
//! token from the `[tracker]` table of `settings.toml`
//! ([`TrackerSettings`]) or the usual environment variables. Loaded
//! blocks are kept for the session ([`ReferenceCache`]); `/refresh #142`
//! loads one again, and the `tracker_get` tool gives the model the same
//! block for a reference it finds itself. Typing `#` in the terminal
//! view's input offers the open issues, listed with their titles
//! ([`OpenIssues`]).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use reqwest::Client;
use rig_basics::coding_tools::AttachSystems;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness::services::ServicesPlugin;
use schemars::JsonSchema;
use serde::Deserialize;

mod complete;
mod refs;
mod tracker;

pub use complete::OpenIssues;

pub use refs::{RefKind, Reference, references};
pub use tracker::{
    BODY_CHARS, COMMENTS, Place, TIMEOUT, Tracker, TrackerKind, TrackerSettings, failure_note,
    remote,
};

/// Who decides the `tracker_get` calls.
const NAME: &str = "issue-references";

/// The tool that loads a reference.
pub const TRACKER_GET: &str = "tracker_get";

/// The blocks loaded this session, by reference (`#142`, `!87`), saved
/// with the session so `/reload` and `/resume` do not load them again.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved)]
pub struct ReferenceCache {
    /// Each reference's block, as the model read it.
    pub blocks: HashMap<String, String>,
}

/// Where the plugin attaches the references a message names, in
/// [`PrepareSystems::Compose`] after the `@path` files; a reference that
/// is not cached is attached later, once loaded, under a hold on the
/// message.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceSystems;

/// Issue and pull request references in the user's messages, `/refresh`
/// and the `tracker_get` tool.
#[derive(Default)]
pub struct IssueReferencesPlugin;

impl Plugin for IssueReferencesPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ServicesPlugin>() {
            app.add_plugins(ServicesPlugin);
        }
        app.add_setting::<TrackerSettings>("tracker");
        let settings = app.world().resource::<TrackerSettings>().clone();
        let runtime = app
            .world()
            .get_resource::<TokioRuntime>()
            .map(|it| it.0.clone());
        let client = Client::builder()
            .timeout(TIMEOUT)
            .build()
            .map_err(|failure| failure.to_string());
        let shared = Shared {
            client,
            runtime,
            settings,
            tracker: OnceLock::new(),
        };
        let options = ToolOptions {
            footprint: Some(Touches::Reads(Scope::Paths(Vec::new()))),
        };
        app.insert_resource(State(Arc::new(shared)))
            .init_resource::<ReferenceCache>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .configure_sets(
                AgentPass,
                ReferenceSystems
                    .in_set(PrepareSystems::Compose)
                    .after(AttachSystems),
            )
            .add_systems(AgentPass, attach_references.in_set(ReferenceSystems))
            .add_observer(fetched)
            .add_open_tool(
                TRACKER_GET,
                "Read an issue or pull request of the tracker that you find in code, commits \
                 or messages: `ref` is `#142` (an issue), `!87` (a pull request) or a tracker \
                 URL.",
                options,
                tracker_get,
            )
            .add_command(
                "refresh",
                "Load an issue or pull request again: /refresh #142",
                refresh,
            );
        complete::add(app);
    }
}

/// What loading needs, shared with the work off the main thread.
struct Shared {
    client: Result<Client, String>,
    runtime: Option<tokio::runtime::Handle>,
    settings: TrackerSettings,
    tracker: OnceLock<Result<Tracker, String>>,
}

#[derive(Resource)]
struct State(Arc<Shared>);

/// Where the project is, for the tracker's `origin` remote.
#[derive(SystemParam)]
struct Locate<'w> {
    processes: Res<'w, Processes>,
    workspace: Res<'w, ProjectWorkspace>,
    root: Res<'w, ProjectRoot>,
}

impl Locate<'_> {
    fn place(&self) -> Place {
        Place {
            launcher: self.processes.launcher(),
            workspace: self.workspace.0.clone(),
            root: self.root.0.clone(),
        }
    }
}

/// Who waits for references being loaded.
#[derive(Clone, Copy)]
enum Waiting {
    /// A message, which the loading entity holds.
    Message(Entity),
    /// A `tracker_get` call.
    Call(Entity),
    /// `/refresh` for an agent.
    Refresh(Entity),
}

/// On an entity loading references: who waits, and every reference it
/// waits for, cached ones included, in order.
#[derive(Component)]
struct Loading {
    waiting: Waiting,
    references: Vec<Reference>,
}

/// What each reference loading loaded as, or why not.
#[derive(Component)]
struct Loaded(Vec<(Reference, Result<String, String>)>);

impl From<Vec<(Reference, Result<String, String>)>> for Loaded {
    fn from(loaded: Vec<(Reference, Result<String, String>)>) -> Self {
        Self(loaded)
    }
}

/// The arguments of `tracker_get`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct GetArgs {
    /// The reference: `#142`, `!87`, `PR#87` or the issue's URL.
    #[serde(rename = "ref")]
    reference: String,
}

/// The tracker, resolved the first time, with the client and runtime
/// to reach it on.
async fn connect(
    shared: &Arc<Shared>,
    place: Place,
) -> Result<(Tracker, Client, tokio::runtime::Handle), String> {
    let tracker = match shared.tracker.get() {
        Some(tracker) => tracker.clone(),
        None => {
            let shared = shared.clone();
            let resolve = move || Ok(Tracker::once(&shared.tracker, &shared.settings, &place));
            blocking(resolve)
                .await
                .unwrap_or_else(|failure| Err(failure.to_string()))
        }
    }?;
    match (&shared.client, &shared.runtime) {
        (Ok(client), Some(runtime)) => Ok((tracker, client.clone(), runtime.clone())),
        (Err(why), _) => Err(why.clone()),
        (_, None) => Err("no tokio runtime".to_owned()),
    }
}

/// Loads `references` in parallel, each under [`TIMEOUT`], on the agent's
/// tokio runtime.
async fn load_all(
    shared: Arc<Shared>,
    place: Place,
    references: Vec<Reference>,
) -> Vec<(Reference, Result<String, String>)> {
    let (tracker, client, runtime) = match connect(&shared, place).await {
        Ok(connected) => connected,
        Err(why) => {
            return references
                .into_iter()
                .map(|it| (it, Err(why.clone())))
                .collect();
        }
    };
    let tasks = references.iter().map(|reference| {
        let (client, tracker, reference) = (client.clone(), tracker.clone(), reference.clone());
        runtime.spawn(async move {
            let load = tracker::load(&client, &tracker, &reference);
            tokio::time::timeout(TIMEOUT, load)
                .await
                .unwrap_or_else(|_| Err(format!("timed out after {} s", TIMEOUT.as_secs())))
        })
    });
    let loaded = futures::future::join_all(tasks).await;
    let loaded = loaded
        .into_iter()
        .map(|it| it.unwrap_or_else(|failure| Err(failure.to_string())));
    references.into_iter().zip(loaded).collect()
}

/// The tracker's open issues, for `#` completion.
async fn load_open_issues(shared: Arc<Shared>, place: Place) -> Result<Vec<(u64, String)>, String> {
    let (tracker, client, runtime) = connect(&shared, place).await?;
    let task = runtime.spawn(async move {
        let load = tracker::open_issues(&client, &tracker);
        tokio::time::timeout(TIMEOUT, load)
            .await
            .unwrap_or_else(|_| Err("timed out".to_owned()))
    });
    task.await.map_err(|failure| failure.to_string())?
}

/// Spawns the loading of each of `references` not in `cache` for
/// `waiting`, holding the message it waits for, if any; `None` when all
/// are cached.
fn start_loading(
    waiting: Waiting,
    references: Vec<Reference>,
    cache: &ReferenceCache,
    (state, locate, wake): (&State, &Locate, &Wake),
    commands: &mut Commands,
) -> Option<Entity> {
    let missing: Vec<Reference> = references
        .iter()
        .filter(|it| !cache.blocks.contains_key(&it.to_string()))
        .cloned()
        .collect();
    if missing.is_empty() {
        return None;
    }
    let names: Vec<String> = missing.iter().map(ToString::to_string).collect();
    let work = load_all(state.0.clone(), locate.place(), missing);
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let mut loading = commands.spawn((
        Name::new(format!("loading {}", names.join(", "))),
        Loading {
            waiting,
            references,
        },
        Running::spawn_into::<Loaded, _>(pool, wake, work),
    ));
    if let Waiting::Message(message) = waiting {
        loading.insert(HoldOf(message));
    }
    Some(loading.id())
}

/// What the model reads for `reference`: its block, else why not.
fn reference_text(
    reference: &Reference,
    loaded: &[(Reference, Result<String, String>)],
    cache: &ReferenceCache,
) -> Result<String, String> {
    let fresh = loaded
        .iter()
        .find(|(it, _)| it == reference)
        .map(|(_, it)| it.clone());
    fresh
        .or_else(|| cache.blocks.get(&reference.to_string()).cloned().map(Ok))
        .unwrap_or_else(|| Err("not loaded".to_owned()))
        .map_err(|why| failure_note(reference, &why))
}

/// Puts a block (or a failure's line) per reference after the message's
/// text.
fn attach(
    typed: &mut Incoming,
    references: &[Reference],
    loaded: &[(Reference, Result<String, String>)],
    cache: &ReferenceCache,
) {
    for reference in references {
        let text = reference_text(reference, loaded, cache).unwrap_or_else(|note| note);
        let content = UserContent::text(text);
        typed.parts.push(MessagePart::Attachment(Attachment {
            label: reference.to_string(),
            content,
        }));
    }
}

/// Attaches the references each user's message names as it comes in: at
/// once when all are cached, else once loaded, holding the message. A
/// message put back in the inbox on restore ([`Started`]) has them already.
fn attach_references(
    mut typed: Query<
        (Entity, &mut Incoming),
        (Added<Admitted>, IncomingMessages, Without<Started>),
    >,
    cache: Res<ReferenceCache>,
    services: (Res<State>, Locate, Res<Wake>),
    mut commands: Commands,
) {
    let services = (&*services.0, &services.1, &*services.2);
    for (message, mut typed) in &mut typed {
        if typed.author.kind != AuthorKind::User || typed.command().is_some() {
            continue;
        }
        let found = references(typed.text());
        if found.is_empty() {
            continue;
        }
        let waiting = Waiting::Message(message);
        if start_loading(waiting, found.clone(), &cache, services, &mut commands).is_none() {
            attach(&mut typed, &found, &[], &cache);
        }
    }
}

/// Keeps what was loaded, then gives it to whoever waits: the message's
/// attachments (and releases it), the call's result, or `/refresh`'s
/// notice.
fn fetched(
    loaded: On<Insert<Loaded>>,
    loadings: Query<(&Loading, &Loaded)>,
    mut cache: ResMut<ReferenceCache>,
    mut messages: Query<&mut Incoming>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((loading, Loaded(results))) = loadings.get(loaded.entity) else {
        return;
    };
    for (reference, result) in results {
        if let Ok(block) = result {
            cache.blocks.insert(reference.to_string(), block.clone());
        }
    }
    match loading.waiting {
        Waiting::Message(message) => {
            if let Ok(mut typed) = messages.get_mut(message) {
                attach(&mut typed, &loading.references, results, &cache);
            }
        }
        Waiting::Call(call) => {
            if let (Ok(mut decision), Some(reference)) =
                (calls.get_mut(call), loading.references.first())
            {
                decision.decide(
                    Decider::plugin(NAME),
                    answer(reference_text(reference, results, &cache)),
                );
            }
        }
        Waiting::Refresh(agent) => {
            for (reference, result) in results {
                notices.write(match result {
                    Ok(_) => Notice::info(agent, format!("Loaded {reference} again.")),
                    Err(why) => Notice::warn(agent, failure_note(reference, why)),
                });
            }
        }
    }
    commands.entity(loaded.entity).despawn();
}

/// A `tracker_get` call's result: the block, else a failure with its line.
fn answer(text: Result<String, String>) -> ToolResult {
    match text {
        Ok(block) => ToolResult::success(ToolOutput::text(block)),
        Err(note) => ToolResult::failed(ToolExecutionError::other(note)),
    }
}

/// `tracker_get`: the reference's block, from the cache or loaded.
fn tracker_get(
    called: On<ToolCalled<GetArgs>>,
    cache: Res<ReferenceCache>,
    services: (Res<State>, Locate, Res<Wake>),
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let services = (&*services.0, &services.1, &*services.2);
    let Some(reference) = references(&called.args.reference).into_iter().next() else {
        if let Ok(mut decision) = calls.get_mut(called.call) {
            let why = format!(
                "not a reference: {:?}; try #142 or !87",
                called.args.reference
            );
            decision.decide(Decider::plugin(NAME), answer(Err(why)));
        }
        return;
    };
    let waiting = Waiting::Call(called.call);
    let started = start_loading(
        waiting,
        vec![reference.clone()],
        &cache,
        services,
        &mut commands,
    );
    if started.is_none()
        && let Ok(mut decision) = calls.get_mut(called.call)
    {
        decision.decide(
            Decider::plugin(NAME),
            answer(reference_text(&reference, &[], &cache)),
        );
    }
}

/// `/refresh #142 !87`: forgets the references and loads them again.
fn refresh(
    In(args): In<CommandArgs>,
    mut cache: ResMut<ReferenceCache>,
    services: (Res<State>, Locate, Res<Wake>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let services = (&*services.0, &services.1, &*services.2);
    let found = references(&args.args);
    if found.is_empty() {
        notices.write(Notice::error(
            args.agent,
            "/refresh needs a reference, such as #142 or !87",
        ));
        return;
    }
    for reference in &found {
        cache.blocks.remove(&reference.to_string());
    }
    start_loading(
        Waiting::Refresh(args.agent),
        found,
        &cache,
        services,
        &mut commands,
    );
}

#[cfg(test)]
mod tests;
