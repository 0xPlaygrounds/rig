//! Retains caller intent that the generic Anthropic encoder otherwise resolves.

use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::{
    completion::{CacheRetention, CompletionRequest, Reasoning},
    effect::{Answered, EffectId, EffectKind},
    error::ErrorReport,
    message::{Message, UserContent},
    serve::{AfterServe, BeforeServe, Intercept},
};
use serde_json::{Map, Value};

#[derive(Default)]
pub(crate) struct RequestPolicy {
    pub(crate) adaptive_only: bool,
    pub(crate) sampling_allowed: bool,
    pub(crate) forced_tools_allowed: bool,
    pub(crate) max_output_tokens: Option<u32>,
}

fn annotate(request: &mut CompletionRequest) -> bool {
    let explicit = request.tool_choice.is_some()
        || request
            .additional_params
            .as_ref()
            .and_then(|p| p.get("tool_choice"))
            .is_some();
    let retention = match request.options.cache {
        Some(CacheRetention::None) => "none",
        Some(CacheRetention::Short) => "short",
        Some(CacheRetention::Long) | None => "long",
        Some(_) => return false,
    };
    let budget = match request.options.reasoning {
        Some(Reasoning::Budget { tokens }) => Some(tokens),
        _ => None,
    };
    let thinking_requested = request
        .additional_params
        .as_ref()
        .and_then(|params| params.pointer("/thinking/type"))
        .and_then(Value::as_str)
        .map(|kind| kind != "disabled")
        .or(match request.options.reasoning {
            Some(Reasoning::Budget { .. } | Reasoning::Effort(_)) => Some(true),
            Some(Reasoning::Off) => Some(false),
            _ => None,
        });
    let params = request
        .additional_params
        .get_or_insert_with(|| Value::Object(Map::new()));
    let Some(params) = params.as_object_mut() else {
        return false;
    };
    let explicit_cache_control = params.contains_key("cache_control");
    let control = params
        .entry("_omp")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(control) = control.as_object_mut() else {
        return false;
    };
    control.insert("tool_choice_explicit".into(), explicit.into());
    control.insert("cache_retention".into(), retention.into());
    control.insert(
        "explicit_cache_control".into(),
        explicit_cache_control.into(),
    );
    if let Some(thinking) = thinking_requested {
        control.insert("thinking_requested".into(), thinking.into());
    }
    if let Some(budget) = budget {
        control.insert("thinking_budget".into(), budget.into());
    }
    true
}

impl Intercept for RequestPolicy {
    fn name(&self) -> String {
        "anthropic-oauth-request-policy".into()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> BeforeServe {
        let EffectKind::Completion { request, stream } = kind else {
            return BeforeServe::Proceed;
        };
        let mut request = request.as_ref().clone();
        if !annotate(&mut request) {
            return BeforeServe::deny(
                "Anthropic OAuth requires object request parameters and supported cache retention",
            );
        }
        if let Some(control) = request
            .additional_params
            .as_mut()
            .and_then(|p| p.get_mut("_omp"))
            .and_then(Value::as_object_mut)
        {
            control.insert("adaptive_only".into(), self.adaptive_only.into());
            control.insert("sampling_allowed".into(), self.sampling_allowed.into());
            control.insert(
                "forced_tools_allowed".into(),
                self.forced_tools_allowed.into(),
            );
            if let Some(limit) = self.max_output_tokens {
                control.insert("max_output_tokens".into(), limit.into());
            }
        }
        BeforeServe::Patch(EffectKind::Completion {
            request: Arc::new(request),
            stream: *stream,
        })
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

/// Each agent retains its own provider session identity across reloads.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Saved, Default, Clone)]
pub(crate) struct OAuthSession(String);

impl Default for OAuthSession {
    fn default() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

/// The human suffix and source message survive compaction, resume and forks.
#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Saved, Default, Clone)]
pub(crate) struct HumanFingerprint(Option<String>, Option<u64>);

/// Retains the three selected UTF-16 units needed to refresh a profile suffix.
#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component, Saved, Default, Clone)]
pub(crate) struct FingerprintSeed {
    units: Option<[u16; 3]>,
    version: String,
}

#[derive(Component)]
pub(crate) struct OmittedCache(bool);

pub(crate) fn default_cache(
    mut calls: Query<
        (&mut ModelRequest, &ForAgent, Option<&Supersedes>),
        (Added<Admitted>, Without<Started>),
    >,
    previous: Query<(&ModelRequest, &OmittedCache), With<Started>>,
    agents: Query<&Connection>,
) {
    use bevy_ecs::relationship::RelationshipTarget;
    for (mut call, ForAgent(agent), supersedes) in &mut calls {
        // A linked retry sharing the started request inherited its resolved default.
        // Editing via request_mut breaks that sharing and keeps the caller's choice.
        if supersedes.is_some_and(|supersedes| {
            supersedes.iter().any(|old| {
                previous.get(old).is_ok_and(|(request, omitted)| {
                    omitted.0 && Arc::ptr_eq(&call.request, &request.request)
                })
            })
        }) {
            call.request_mut().options.cache = None;
        }
        let Some(connection) = call.connection.as_ref().or_else(|| agents.get(*agent).ok()) else {
            continue;
        };
        if connection.spec.provider.vendor() != crate::VENDOR {
            continue;
        }
        // Kernel-built calls already contain catalog defaults. Input runs before
        // request editors, whose explicit retention must remain authoritative.
        if call.prompt.is_some() && call.history.is_some() {
            call.request_mut().options.cache = None;
        }
    }
}

pub(crate) fn identify_requests(
    mut calls: Query<(Entity, &ModelRequest, &ForAgent, &mut Adapters), Added<Admitted>>,
    mut commands: Commands,
    mut agents: Query<(
        &OAuthSession,
        &Conversation,
        &mut HumanFingerprint,
        &mut FingerprintSeed,
    )>,
) {
    for (entity, call, ForAgent(agent), mut adapters) in &mut calls {
        commands
            .entity(entity)
            .insert(OmittedCache(call.request.options.cache.is_none()));
        let Ok((session, conversation, mut fingerprint, mut seed)) = agents.get_mut(*agent) else {
            continue;
        };
        refresh_fingerprint(&conversation.0, &mut fingerprint, &mut seed);
        let identity = AgentIdentity {
            session: session.0.clone(),
            suffix: fingerprint.0.clone(),
            default_cache: call.request.options.cache.is_none(),
        };
        adapters.0.push(Adapter::new(
            "claude-cloak-identity",
            move |mut connection| {
                if connection.spec.provider.vendor() == crate::VENDOR {
                    connection.handler.0 = connection.handler.0.layered(identity.clone());
                }
                connection
            },
        ));
    }
}

fn refresh_fingerprint(
    conversation: &rig_harness::rig_core::conversation::Conversation,
    fingerprint: &mut HumanFingerprint,
    seed: &mut FingerprintSeed,
) {
    if seed.units.is_none() {
        let human = conversation.entries().find(|entry| entry.is_users_own());
        if human.is_some() || fingerprint.0.is_some() {
            // Older saves have no seed. Their first retained human starts the new profile.
            seed.units = Some(crate::profile::seed(&first_human_text(conversation)));
            fingerprint.1 = human.map(|entry| entry.id.get());
        }
    }
    if let Some(units) = seed.units
        && (fingerprint.0.is_none() || seed.version != crate::profile::VERSION)
    {
        fingerprint.0 = Some(crate::profile::suffix_from_seed(&units));
        seed.version = crate::profile::VERSION.into();
    }
}

#[derive(Clone)]
struct AgentIdentity {
    session: String,
    suffix: Option<String>,
    default_cache: bool,
}

impl Intercept for AgentIdentity {
    fn name(&self) -> String {
        "claude-cloak-agent-identity".into()
    }

    async fn before(&self, _id: EffectId, kind: &EffectKind) -> BeforeServe {
        let EffectKind::Completion { request, stream } = kind else {
            return BeforeServe::Proceed;
        };
        let mut request = request.as_ref().clone();
        if self.default_cache {
            request.options.cache = Some(crate::profile::CACHE);
        }
        let params = request
            .additional_params
            .get_or_insert_with(|| Value::Object(Map::new()));
        let Some(params) = params.as_object_mut() else {
            return BeforeServe::deny("Anthropic OAuth requires object request parameters");
        };
        let control = params
            .entry("_omp")
            .or_insert_with(|| Value::Object(Map::new()));
        let Some(control) = control.as_object_mut() else {
            return BeforeServe::deny("Anthropic OAuth requires object private parameters");
        };
        control.insert("session_id".into(), self.session.clone().into());
        control.insert("agent_request".into(), true.into());
        if let Some(suffix) = &self.suffix {
            control.insert("human_suffix".into(), suffix.clone().into());
        }
        BeforeServe::Patch(EffectKind::Completion {
            request: Arc::new(request),
            stream: *stream,
        })
    }

    async fn after(
        &self,
        _id: EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Answered, ErrorReport>,
    ) -> AfterServe {
        AfterServe::Keep
    }
}

pub(crate) fn fork_fingerprint(
    fork: On<AgentForked>,
    mut agents: Query<(
        &Conversation,
        &mut HumanFingerprint,
        &mut OAuthSession,
        &mut FingerprintSeed,
    )>,
) {
    let Ok((conversation, mut fingerprint, mut session, mut seed)) = agents.get_mut(fork.fork)
    else {
        return;
    };
    *session = OAuthSession::default();
    let Some(through) = fork.through else {
        return;
    };
    if fingerprint.1.is_some_and(|id| through.get() <= id)
        || (fingerprint.1.is_none() && !conversation.entries().any(|entry| entry.is_users_own()))
    {
        *fingerprint = HumanFingerprint::default();
        *seed = FingerprintSeed::default();
    }
}

fn first_human_text(conversation: &rig_harness::rig_core::conversation::Conversation) -> String {
    conversation
        .entries()
        .find(|entry| entry.is_users_own())
        .and_then(|entry| match entry.message {
            Message::User { content } => content.iter().find_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            }),
            _ => None,
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
