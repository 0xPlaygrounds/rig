//! SN-06: a `web_fetch` tool with a domain policy. The tool reads a web
//! page as clean text: HTML as Markdown without scripts, navigation or
//! footers, JSON pretty-printed, text as it is, binary types refused. Long
//! pages come in parts, and a page is reused for 15 minutes.
//!
//! Which hosts it reaches is `RIG_HOME/fetch.toml`'s ([`FetchConfig`]): a
//! gate refuses a denied host, and asks the user about a host on neither
//! list, once per domain: allow it for the session, allow this call, or
//! deny it. A run nobody answers refuses it. Loopback, link-local and
//! private addresses are refused unless the host is on the allow list by
//! its exact name, checked again on every redirect and after every DNS
//! lookup.

use std::sync::{Arc, PoisonError};

use rig_harness::prelude::*;
use rig_harness::services::ServicesPlugin;

mod fetch;
mod markdown;
mod policy;

pub use fetch::{CACHE_FOR, DEFAULT_MAX_CHARS, MAX_REDIRECTS, TIMEOUT, WEB_FETCH};
pub use policy::{CONFIG, FetchConfig, Verdict};

use fetch::{Shared, WebFetch, host_of, parse};

/// Who decides the calls the policy refuses.
const DECIDER: &str = "sn-web-fetch";

/// The answer that allows a domain for the rest of the session.
pub const ALLOW_SESSION: &str = "session";
/// The answer that allows the one call.
pub const ALLOW_ONCE: &str = "once";
/// The answer that denies the domain for the rest of the session.
pub const DENY: &str = "deny";

/// The question's title about `host`.
pub fn title(host: &str) -> String {
    format!("Let the agent fetch from {host}?")
}

/// The domains the user allowed or denied in this session, by host. Saved
/// with the session, so `/reload` and `/resume` do not ask again.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved)]
pub struct FetchDomains {
    /// Hosts fetched without asking for the rest of the session.
    pub allowed: Vec<String>,
    /// Hosts refused for the rest of the session.
    pub denied: Vec<String>,
}

/// The `web_fetch` tool and the gate that applies `RIG_HOME/fetch.toml`.
#[derive(Default)]
pub struct WebFetchPlugin;

impl Plugin for WebFetchPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ServicesPlugin>() {
            app.add_plugins(ServicesPlugin);
        }
        let path = RigHome::of(app.world()).root().join(CONFIG);
        let (config, problem) = FetchConfig::read(&path);
        if let Some(problem) = problem {
            app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                notices.write(Notice::warn(None, problem.clone()));
            });
        }
        let runtime = app
            .world()
            .get_resource::<TokioRuntime>()
            .map(|runtime| runtime.0.clone());
        let shared = Arc::new(Shared::new(runtime, config));
        app.add_tool_with(
            WebFetch(shared.clone()),
            ToolOptions {
                // It reads nothing of the workspace and changes nothing.
                footprint: Some(Touches::Reads(Scope::Paths(Vec::new()))),
            },
        )
        .insert_resource(FetchState(shared))
        .init_resource::<FetchDomains>()
        .add_systems(AgentPass, (answered, gate).chain().in_set(Stage::Gate))
        .add_systems(Update, mirror.run_if(resource_changed::<FetchDomains>));
    }
}

/// What the tool and the gate share.
#[derive(Resource)]
struct FetchState(Arc<Shared>);

/// On the question about a domain: the host it asks about.
#[derive(Component)]
struct DomainAsk(String);

/// On a hold of a call whose domain is being asked about by another call's
/// question.
#[derive(Component)]
struct DomainWait(String);

/// On a call the user allowed once.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct AllowedOnce;

/// Gives the tool this session's answers, for the redirects it follows,
/// as they change and once restored.
fn mirror(domains: Res<FetchDomains>, state: Res<FetchState>) {
    let mut session = state
        .0
        .session
        .write()
        .unwrap_or_else(PoisonError::into_inner);
    *session = domains.clone();
}

/// Refuses each `web_fetch` the policy denies, asks about a host on
/// neither list, or refuses it when nobody can answer.
fn gate(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &mut Decision),
        (ReadyToStart, ToolCalls, Without<AllowedOnce>),
    >,
    open: Query<&DomainAsk, Without<Outcome>>,
    state: Res<FetchState>,
    domains: Res<FetchDomains>,
    answerers: Answerers,
    mut commands: Commands,
) {
    let mut asking: Vec<String> = open.iter().map(|ask| ask.0.clone()).collect();
    for (call, request, &ForAgent(agent), mut decision) in &mut calls {
        if request.call.function.name.as_str() != WEB_FETCH || decision.is_decided() {
            continue;
        }
        let url = request
            .call
            .function
            .arguments
            .get("url")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let host = match parse(url) {
            Ok(url) => host_of(&url),
            Err(failure) => {
                let why = failure.model_feedback().unwrap_or("not a web address");
                decision.decide(Decider::plugin(DECIDER), Outcome::refused(why));
                continue;
            }
        };
        let why = match state.0.config.verdict(&host, &domains) {
            Verdict::Allowed => continue,
            Verdict::Denied(why) => {
                decision.decide(Decider::plugin(DECIDER), Outcome::refused(why));
                continue;
            }
            Verdict::Ask(why) => why,
        };
        if !answerers.can_answer(Some(agent)) {
            let why = format!("{why}, and nobody can allow it in this run");
            decision.decide(Decider::plugin(DECIDER), Outcome::refused(why));
        } else if asking.contains(&host) {
            let name = Name::new(format!("waiting for the answer about {host}"));
            commands.spawn((name, DomainWait(host), HoldOf(call)));
        } else {
            let ask = Ask::new(title(&host))
                .body(url.to_owned())
                .choice(ALLOW_SESSION, "Allow domain for this session")
                .choice(ALLOW_ONCE, "Allow once")
                .choice(DENY, "Deny");
            commands.spawn((ask, DomainAsk(host.clone()), ForAgent(agent), HoldOf(call)));
            asking.push(host);
        }
    }
}

/// Applies each answer: remembers a domain allowed or denied for the
/// session, lets an allowed-once call run, and refuses a denied one. The
/// calls waiting on the same domain are gated again.
fn answered(
    asks: Query<(Entity, &Outcome, &DomainAsk, Option<&HoldOf>), (With<Ask>, Added<Outcome>)>,
    waits: Query<(Entity, &DomainWait)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut domains: ResMut<FetchDomains>,
    mut commands: Commands,
) {
    for (ask, outcome, DomainAsk(host), held) in &asks {
        let answer = Answer::of(outcome);
        let is = |id: &str| answer.as_ref().is_some_and(|answer| answer.is(id));
        let call = held.map(|&HoldOf(call)| call);
        if is(ALLOW_SESSION) {
            domains.allowed.push(host.clone());
        } else if is(ALLOW_ONCE) {
            if let Some(call) = call {
                commands.entity(call).try_insert(AllowedOnce);
            }
        } else {
            if is(DENY) {
                domains.denied.push(host.clone());
            }
            if let Some(mut decision) = call.and_then(|call| calls.get_mut(call).ok()) {
                let why = format!("the user did not allow fetching from {host}");
                decision.decide(Decider::Person, Outcome::refused(why));
            }
        }
        commands.entity(ask).remove::<HoldOf>();
        for (wait, DomainWait(waiting)) in &waits {
            if waiting == host {
                commands.entity(wait).despawn();
            }
        }
    }
}

#[cfg(test)]
mod tests;
