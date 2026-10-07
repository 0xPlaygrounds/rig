//! The one owner of a dispatch id. A [`DispatchScope`] mints the id, fires
//! `on_dispatch`, and closes the id with `on_outcome`; nothing else in the
//! engine mints an id or builds either event. A [`CompletionScope`] holds a
//! completion's scope across its model turn and closes it at most once.

use super::super::hook::{DispatchAction, DispatchEvent, OutcomeEvent};
use super::*;
use crate::run::response::CompletionCall;
use crate::tool::ToolContext;
use rig_core::{completion::CompletionRequest, streaming::CompletionStream};
use rig_core::{completion::CompletionResponse, effect::EffectId, message::CallId};

/// One dispatched effect: its id and the effect after any hook's patch.
pub(crate) struct DispatchScope {
    id: EffectId,
    kind: EffectKind,
}

impl DispatchScope {
    /// Mint an id and fire `on_dispatch`. The stack has already turned a patch
    /// that changes the family or the tool target into a denial. A denied
    /// effect never reaches the bus and has no outcome unless the caller
    /// closes the returned scope, which carries the patch an earlier hook made
    /// before the denial.
    pub(crate) async fn open(
        runner: &AgentRunner,
        ctx: &HookContext,
        kind: EffectKind,
        call: Option<(&CallId, &ToolContext)>,
    ) -> Result<Self, (Self, ErrorReport)> {
        let id = runner.config.bus.dispatcher().mint_id();
        let (call_id, context) = call.unzip();
        let event = DispatchEvent {
            id,
            kind: &kind,
            turn: ctx.turn(),
            call_id,
            context,
        };
        match runner.config.hooks.on_dispatch(ctx, event).await {
            DispatchAction::Proceed => Ok(Self { id, kind }),
            DispatchAction::Patch(kind) => Ok(Self { id, kind }),
            DispatchAction::Deny(report) => {
                let kind = ctx.take_salvaged_patch(id).unwrap_or(kind);
                Err((Self { id, kind }, report))
            }
        }
    }

    pub(crate) fn kind(&self) -> &EffectKind {
        &self.kind
    }

    /// Bus options that dispatch under this scope's id.
    pub(crate) fn options(&self) -> crate::bus::DispatchOptions {
        crate::bus::DispatchOptions::default().with_id(self.id)
    }

    /// Fire `on_outcome` for `outcome`, closing the id.
    pub(crate) async fn close(
        &self,
        runner: &AgentRunner,
        ctx: &HookContext,
        outcome: &Result<Outcome, ErrorReport>,
        call: Option<(&CallId, &ToolContext)>,
    ) -> OutcomeAction {
        let ((call_id, context), turn) = (call.unzip(), ctx.turn());
        let (id, kind) = (self.id, &self.kind);
        let event = OutcomeEvent {
            id,
            kind,
            outcome,
            turn,
            call_id,
            context,
        };
        runner.config.hooks.on_outcome(ctx, event).await
    }
}

/// A completion between its dispatch and its outcome. The model turn records
/// the bus's answer here, and the outcome fires once: from [`Self::close`]
/// for a settled turn, or from [`Self::close_unsettled`] on any other exit.
pub(crate) struct CompletionScope {
    scope: DispatchScope,
    answer: Option<Result<CompletionResponse, ErrorReport>>,
    state: State,
}

enum State {
    Open,
    Settled,
}

/// Open a completion dispatch whose effect, patched or not, keeps the
/// medium's `stream` flag. A `Cancelled` denial cancels the run and any other
/// denial fails it; a denied completion has no outcome.
pub(crate) async fn open_completion(
    runner: &AgentRunner,
    ctx: &HookContext,
    run: &AgentRun,
    request: CompletionRequest,
    stream: bool,
) -> Result<CompletionScope, PromptError> {
    let kind = EffectKind::Completion { request, stream };
    let mut scope = match DispatchScope::open(runner, ctx, kind, None).await {
        Ok(scope) => scope,
        Err((_, report)) if report.kind == ErrorKind::Cancelled => {
            return Err(run.cancel_error(report.message));
        }
        Err((_, report)) => return Err(PromptError::Report(report)),
    };
    if let EffectKind::Completion { stream: s, .. } = &mut scope.kind {
        *s = stream;
    }
    let (answer, state) = (None, State::Open);
    Ok(CompletionScope {
        scope,
        answer,
        state,
    })
}

impl CompletionScope {
    /// Dispatch a unary completion and record the bus's answer.
    pub(crate) async fn dispatch_response(
        &mut self,
        runner: &AgentRunner,
        model: &ModelHandle,
    ) -> Result<&CompletionResponse, PromptError> {
        let (kind, options) = (self.scope.kind.clone(), self.scope.options());
        let bus = runner.config.bus.dispatcher();
        let answer = match bus.dispatch_with(model.key(), kind, options).await {
            Ok(Outcome::Completion(response)) => Ok(response),
            Ok(other) => Err(wrong_outcome("a completion", &other)),
            Err(report) => Err(report),
        };
        let answer = self.answer.insert(answer).as_ref();
        answer.map_err(|report| PromptError::Report(report.clone()))
    }

    /// Dispatch a streaming completion; [`Self::finish_stream`] records its end.
    pub(crate) fn dispatch_stream(
        &self,
        runner: &AgentRunner,
        model: &ModelHandle,
    ) -> CompletionStream {
        let (kind, options) = (self.scope.kind.clone(), self.scope.options());
        let bus = runner.config.bus.dispatcher();
        let events = bus.dispatch_stream_with(model.key(), kind, options);
        crate::bus::wrap_stream(model.label().to_string(), events)
    }

    /// End `stream` and record its response or error here and, with the
    /// reply's usage, in `run`. The recorded response is returned so the turn
    /// can amend the choice the hooks see.
    pub(crate) async fn finish_stream(
        &mut self,
        stream: CompletionStream,
        run: &mut AgentRun,
        span: &tracing::Span,
    ) -> Result<(&mut CompletionResponse, CompletionCall), PromptError> {
        let response = match stream.finish().await {
            Ok(response) => self.answer.insert(Ok(response)),
            Err(error) => {
                let error = PromptError::from(error);
                self.answer = Some(Err(report_of(&error)));
                return Err(error);
            }
        };
        let Ok(response) = response else {
            return Err(PromptError::Report(no_answer()));
        };
        span.record_token_usage(&response.usage);
        let (usage, identity, raw) = (response.usage, response.identity(), response.raw.clone());
        let call =
            run.record_streamed_completion_call(usage, identity, response.finish_reason(), raw)?;
        Ok((response, call))
    }

    /// The choice of the recorded response, empty when there is none.
    pub(crate) fn choice(&self) -> &[AssistantContent] {
        let response = self.answer.as_ref().and_then(|answer| answer.as_ref().ok());
        response.map_or(&[], |response| &response.choice)
    }

    /// Settle an accepted turn: fire `on_outcome` with the recorded response
    /// and return the hook's action with that response.
    pub(crate) async fn close(
        &mut self,
        runner: &AgentRunner,
        ctx: &HookContext,
    ) -> Result<(OutcomeAction, &CompletionResponse), PromptError> {
        let action = self.fire(runner, ctx, None).await;
        match &self.answer {
            Some(Ok(response)) => Ok((action, response)),
            _ => Err(PromptError::Report(no_answer())),
        }
    }

    /// Close an attempt that ended unsettled: a rejected, recovered or
    /// abandoned one with its answer, a failed one with `error`. The outcome
    /// is observe-only, so a replacement is ignored. No-op once settled.
    pub(crate) async fn close_unsettled(
        &mut self,
        runner: &AgentRunner,
        ctx: &HookContext,
        error: Option<&PromptError>,
    ) {
        self.fire(runner, ctx, error).await;
    }

    async fn fire(
        &mut self,
        runner: &AgentRunner,
        ctx: &HookContext,
        error: Option<&PromptError>,
    ) -> OutcomeAction {
        if let State::Settled = std::mem::replace(&mut self.state, State::Settled) {
            return OutcomeAction::Proceed;
        }
        let outcome = match &self.answer {
            // The hook sees the turn's normalized fields, not the abort marker.
            Some(Ok(response)) => {
                let mut folded = response
                    .clone()
                    .with_optional_finish_reason(response.finish_reason());
                folded.aborted = None;
                Ok(Outcome::Completion(folded))
            }
            Some(Err(report)) => Err(report.clone()),
            None => Err(error.map_or_else(no_answer, report_of)),
        };
        self.scope.close(runner, ctx, &outcome, None).await
    }
}

fn no_answer() -> ErrorReport {
    ErrorReport::new(ErrorKind::Internal, "completion ended without an answer")
}

/// The report a failed completion attempt closes with.
fn report_of(error: &PromptError) -> ErrorReport {
    match error {
        PromptError::Report(report) => report.clone(),
        PromptError::Provider(error) => error.report(),
        PromptError::Cancelled { reason, .. } => ErrorReport::new(ErrorKind::Cancelled, reason),
        PromptError::Memory(error) => error.report(),
        other => ErrorReport::new(ErrorKind::Internal, other.to_string()),
    }
}
