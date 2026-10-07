//! The one owner of a dispatch id. A [`DispatchScope`] mints the id, fires
//! `on_dispatch`, and closes the id with `on_outcome`; nothing else in the
//! engine mints an id or builds either event. [`model_step`] holds a
//! completion's [`CompletionScope`] across its model turn and closes it once.

use super::super::hook::{DispatchAction, DispatchEvent, OutcomeEvent};
use super::*;
use crate::run::response::CompletionCall;
use crate::tool::ToolContext;
use rig_core::{completion::CompletionRequest, streaming::CompletionStream};
use rig_core::{completion::CompletionResponse, effect::EffectId, message::CallId};
use std::marker::PhantomData;

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

/// A completion between its dispatch and its outcome, for the turn source
/// `S`: only `S`'s medium has a dispatch method, and the effect's `stream`
/// flag is [`TurnSource::STREAMS`]. Only [`model_step`] opens one, and it
/// closes it once the model turn ends by any exit: from [`Self::close`] for
/// a settled turn, or observe-only otherwise. The turn dispatches once and
/// records the bus's answer here.
pub(crate) struct CompletionScope<S> {
    scope: DispatchScope,
    answer: Option<Result<CompletionResponse, ErrorReport>>,
    state: State,
    source: PhantomData<fn() -> S>,
}

/// Opened, then sent once (a second send is refused), then closed.
enum State {
    Open,
    Dispatched,
    Settled,
}

/// One model step: open the completion dispatch (its effect, patched or not,
/// keeps the medium's `stream` flag), run `source`'s model turn through it,
/// and close it, observe-only, if the turn did not settle it. An error from
/// the turn is yielded after the close. A `Cancelled` denial cancels the run
/// and any other denial fails it; a denied completion has no outcome.
pub(crate) fn model_step<'a, S: TurnSource>(
    source: &'a mut S,
    runner: &'a AgentRunner,
    hook_ctx: &'a HookContext,
    run: &'a mut AgentRun,
    request: CompletionRequest,
    prepared: PreparedCompletionRequest,
    chat_span: tracing::Span,
    agent_span: &'a tracing::Span,
) -> DriveStream<'a> {
    Box::pin(async_stream::stream! {
        let kind = EffectKind::Completion { request, stream: S::STREAMS };
        let opened = DispatchScope::open(runner, hook_ctx, kind, None)
            .instrument(chat_span.clone())
            .await;
        let mut scope = match opened {
            Ok(scope) => scope,
            Err((_, report)) => {
                yield Err(match report.kind {
                    ErrorKind::Cancelled => run.cancel_error(report.message),
                    _ => PromptError::Report(report),
                });
                return;
            }
        };
        if let EffectKind::Completion { stream, .. } = &mut scope.kind {
            *stream = S::STREAMS;
        }
        let (answer, state, source_marker) = (None, State::Open, PhantomData);
        let mut scope = CompletionScope::<S> { scope, answer, state, source: source_marker };
        let mut turn =
            source.run_model_turn(runner, hook_ctx, run, prepared, chat_span, agent_span, &mut scope);
        let error = loop {
            match turn.next().await {
                Some(Ok(item)) => yield Ok(item),
                Some(Err(err)) => break Some(err),
                None => break None,
            }
        };
        drop(turn);
        scope.fire(runner, hook_ctx, error.as_ref()).await;
        if let Some(err) = error {
            yield Err(err);
        }
    })
}

impl<M> CompletionScope<M> {
    /// Mark the completion sent, or refuse a second send: one scope is one
    /// dispatch id, sent once.
    fn send(&mut self) -> Result<(EffectKind, crate::bus::DispatchOptions), PromptError> {
        if !matches!(self.state, State::Open) {
            let report = ErrorReport::new(ErrorKind::Internal, "a completion was dispatched twice");
            self.answer = Some(Err(report.clone()));
            return Err(PromptError::Report(report));
        }
        self.state = State::Dispatched;
        Ok((self.scope.kind.clone(), self.scope.options()))
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

    /// Fire the outcome once: the recorded answer, else `error`. No-op once
    /// settled. Outside [`Self::close`] it is observe-only, so a replacement
    /// is ignored.
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

impl CompletionScope<UnaryTurnSource> {
    /// Dispatch the unary completion and record the bus's answer.
    pub(crate) async fn dispatch_response(
        &mut self,
        runner: &AgentRunner,
        model: &ModelHandle,
    ) -> Result<&CompletionResponse, PromptError> {
        let (kind, options) = self.send()?;
        let bus = runner.config.bus.dispatcher();
        let answer = match bus.dispatch_with(model.key(), kind, options).await {
            Ok(Outcome::Completion(response)) => Ok(response),
            Ok(other) => Err(wrong_outcome("a completion", &other)),
            Err(report) => Err(report),
        };
        let answer = self.answer.insert(answer).as_ref();
        answer.map_err(|report| PromptError::Report(report.clone()))
    }
}

impl CompletionScope<StreamingTurnSource> {
    /// Dispatch the streaming completion; [`Self::finish_stream`] records its
    /// end.
    pub(crate) fn dispatch_stream(
        &mut self,
        runner: &AgentRunner,
        model: &ModelHandle,
    ) -> Result<CompletionStream, PromptError> {
        let (kind, options) = self.send()?;
        let bus = runner.config.bus.dispatcher();
        let events = bus.dispatch_stream_with(model.key(), kind, options);
        Ok(crate::bus::wrap_stream(model.label().to_string(), events))
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
