//! Retains caller intent that the generic Anthropic encoder otherwise resolves.

use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::{
    completion::{CacheRetention, CompletionRequest, Reasoning},
    effect::{Answered, EffectId, EffectKind},
    error::ErrorReport,
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
    let control = params
        .entry("_omp")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(control) = control.as_object_mut() else {
        return false;
    };
    control.insert("tool_choice_explicit".into(), explicit.into());
    control.insert("cache_retention".into(), retention.into());
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

pub(crate) fn identify_requests(
    mut calls: Query<(&mut ModelRequest, &ForAgent), Added<Admitted>>,
    agents: Query<(&OAuthSession, Option<&Connection>)>,
) {
    for (mut call, ForAgent(agent)) in &mut calls {
        let Ok((session, connection)) = agents.get(*agent) else {
            continue;
        };
        let Some(connection) = call.connection.as_ref().or(connection) else {
            continue;
        };
        if connection.spec.provider.vendor() != "anthropic-oauth" {
            continue;
        }
        // Set the OAuth default before Start fills Rig's short-cache default.
        call.request_mut()
            .options
            .cache
            .get_or_insert(CacheRetention::Long);
        let params = call
            .request_mut()
            .additional_params
            .get_or_insert_with(|| Value::Object(Map::new()));
        let Some(params) = params.as_object_mut() else {
            continue;
        };
        let control = params
            .entry("_omp")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(control) = control.as_object_mut() {
            control.insert("session_id".into(), session.0.clone().into());
        }
    }
}

#[cfg(test)]
mod tests;
