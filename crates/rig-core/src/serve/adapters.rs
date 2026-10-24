//! Effect handlers adapting model, tool, and retrieval traits.
//! Each adapter translates its effect family into trait calls and returns
//! outcomes or streaming events.
//!
//! ```
//! use rig_core::message::ToolName;
//! use rig_core::serve::Serve;
//! use rig_core::tool::{DynamicTool, ToolContext, ToolOutput};
//! use rig_core::wasm_compat::WasmBoxedFuture;
//!
//! let handler = DynamicTool::new_with_context(
//!     ToolName::new("echo")?,
//!     "Echo the arguments",
//!     serde_json::json!({ "type": "object" }),
//!     |_context: &mut ToolContext, args: serde_json::Value| {
//!         Box::pin(async move { Ok(ToolOutput::text(args.to_string())) })
//!             as WasmBoxedFuture<'_, _>
//!     },
//! );
//! assert_eq!(handler.descriptor().key, rig_core::effect::tool_key("echo"));
//! # Ok::<(), rig_core::message::EmptyToolName>(())
//! ```

use std::sync::{Arc, Mutex};

use crate::{
    completion::{CompletionRequest, ModelRef, UnsupportedOption},
    driver::{DynModel, Sending, without_messages},
    effect::{
        Answered, EffectFamily, EffectId, EffectKind, EmbedInputs, EmbedModality, EmbedOutputs,
        FamilyDescriptor, HandlerDescriptor, HandlerKey, RetrieveQuery, RetrievedDocuments,
        ToolEmbeddingDescriptor,
    },
    error::{ErrorKind, ErrorReport},
    operation::{Completion, Embedding, Rerank},
    providers::registry::ProviderId,
    tool::{Tool, ToolEmbedding, ToolExecutionError},
    vector_store::{VectorStoreError, VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
    wire::Operation,
};

use super::{Dispatch, Reply, Serve};
use crate::effect::family;

fn wrong_family(handler: EffectFamily, kind: &EffectKind) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!(
            "a {handler} handler cannot serve a `{}` effect",
            kind.name()
        ),
    )
}

/// A model as a handler under `label`. Completion, text-embedding and
/// rerank models serve their effect families. The model is erased to its
/// operation, so one adapter type serves every wire and transport.
pub struct ModelAdapter<Op: Operation> {
    label: ModelRef,
    model: DynModel<Op>,
    /// The last completion request [`Serve::check`] answered, as it was
    /// checked (without its messages), and what the model refused of it.
    checked: Mutex<Option<(CompletionRequest, Vec<UnsupportedOption>)>>,
}

impl<Op: Operation> ModelAdapter<Op> {
    /// Wrap `model` under `label`.
    pub fn new(label: impl Into<ModelRef>, model: impl Into<DynModel<Op>>) -> Self {
        Self {
            label: label.into(),
            model: model.into(),
            checked: Mutex::new(None),
        }
    }

    /// How the model sends for `dispatch` under `handler`: its HTTP request
    /// carries the [`Serving`], observed under `observation`.
    fn sending(
        &self,
        handler: HandlerKey,
        dispatch: &Dispatch,
        observation: Option<crate::observe::AdapterContext>,
    ) -> Sending {
        let mut extensions = http::Extensions::new();
        extensions.insert(Serving {
            effect: dispatch.id(),
            handler,
            provider: self.model.spec().map(|spec| spec.provider),
        });
        Sending {
            observation,
            extensions,
        }
    }
}

/// What an outgoing HTTP request to a model serves, in its
/// [extensions](http::Request::extensions): the effect it answers and the
/// handler it was dispatched to. A model adapter sets it on every request
/// it sends, so a [transport wrapper](crate::catalog::Connector::wrap_http)
/// can tell one model call's requests from another's and from those of no
/// call, such as a model listing. The request's headers and body are the
/// wrapper's to read, such as rig-cassette's traffic log, which records them.
///
/// ```
/// use rig_core::http_client::Request;
/// use rig_core::serve::Serving;
///
/// fn served(request: &Request<Vec<u8>>) -> Option<u64> {
///     request
///         .extensions()
///         .get::<Serving>()
///         .map(|serving| serving.effect.as_u64())
/// }
/// assert_eq!(served(&Request::new(Vec::new())), None);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Serving {
    /// The effect the request answers, such as an ECS effect's id.
    pub effect: EffectId,
    /// The handler it was dispatched to, such as `model:openai/gpt-5`.
    pub handler: HandlerKey,
    /// The provider the model is reached through, when the model was
    /// connected from a catalog spec.
    pub provider: Option<ProviderId>,
}

impl Serving {
    /// What a request answering `effect` through `handler` serves, reached
    /// through no provider a catalog names.
    pub fn new(effect: EffectId, handler: HandlerKey) -> Self {
        Self {
            effect,
            handler,
            provider: None,
        }
    }
}

/// Unary and streaming completions both route here; the descriptor carries
/// the model's label and capability snapshot.
impl Serve for ModelAdapter<Completion> {
    type Family = family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::model_key(self.label.as_str()),
            family: FamilyDescriptor::Completion {
                model: self.label.clone(),
                capabilities: self.model.capabilities().completion,
            },
            layers: Vec::new(),
        }
    }

    /// What the model's wire or the model leaves out of a completion's
    /// request ([`DynModel::check`]); nothing for a request that cannot be
    /// built at all, which sending reports, or for another family. The
    /// check reads no messages but the system prompt, so the answer for
    /// the last request is kept for the next that differs only in its
    /// messages. A request with provider options is always checked: two
    /// equal entries can still refuse differently.
    fn check(&self, kind: &EffectKind) -> Vec<UnsupportedOption> {
        let EffectKind::Completion { request, .. } = kind else {
            return Vec::new();
        };
        let memo = request.provider_options.is_empty();
        let checked = without_messages(request);
        if memo
            && let Ok(last) = self.checked.lock()
            && let Some((last, refused)) = last.as_ref()
            && *last == checked
        {
            return refused.clone();
        }
        let refused = match self.model.check(request) {
            Err(crate::completion::CheckError::Unsupported(refused)) => refused,
            Ok(()) | Err(crate::completion::CheckError::Invalid(_)) => Vec::new(),
        };
        if memo && let Ok(mut last) = self.checked.lock() {
            *last = Some((checked, refused.clone()));
        }
        refused
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let model = &self.model;
        let sending = self.sending(
            crate::effect::model_key(self.label.as_str()),
            &dispatch,
            dispatch.adapter_context(),
        );
        match kind {
            EffectKind::Completion {
                request,
                stream: false,
            } => {
                let result = model.finished(Arc::unwrap_or_clone(request), sending).await;
                let result = result.map(Arc::new);
                Reply::Outcome(result.map(Answered::Completion).map_err(ErrorReport::from))
            }
            EffectKind::Completion {
                request,
                stream: true,
            } => match model.opened(Arc::unwrap_or_clone(request), sending) {
                Ok(stream) => Reply::Stream(stream.into_relay()),
                Err(error) => Reply::Outcome(Err(ErrorReport::from(error))),
            },
            other @ (EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Completion, &other)))
            }
        }
    }
}

/// A text embedding model embeds texts and refuses images.
impl Serve for ModelAdapter<Embedding> {
    type Family = family::Embed;

    fn descriptor(&self) -> HandlerDescriptor {
        let capabilities = self.model.capabilities();
        HandlerDescriptor {
            key: crate::effect::embed_key(self.label.as_str()),
            family: FamilyDescriptor::Embed {
                model: self.label.to_string(),
                dims: Some(capabilities.ndims),
                max_documents: capabilities.max_documents,
                modality: EmbedModality::Text,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let model = &self.model;
        let sending = self.sending(
            crate::effect::embed_key(self.label.as_str()),
            &dispatch,
            None,
        );
        match kind {
            EffectKind::Embed {
                inputs: EmbedInputs::Texts(texts),
            } => Reply::Outcome(
                model
                    .finished(texts, sending)
                    .await
                    .map(|response| Answered::Embeddings(EmbedOutputs::Texts(response)))
                    .map_err(ErrorReport::from),
            ),
            EffectKind::Embed {
                inputs: EmbedInputs::Images(_),
            } => Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                "a text embedding handler cannot embed images",
            ))),
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Embed, &other)))
            }
        }
    }
}

/// A reranking model orders documents for a query.
impl Serve for ModelAdapter<Rerank> {
    type Family = family::Rerank;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::rerank_key(self.label.as_str()),
            family: FamilyDescriptor::Rerank {
                model: self.label.to_string(),
                max_documents: self.model.capabilities().max_documents,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let model = &self.model;
        let sending = self.sending(
            crate::effect::rerank_key(self.label.as_str()),
            &dispatch,
            None,
        );
        match kind {
            EffectKind::Rerank { request } => Reply::Outcome(
                model
                    .finished(request, sending)
                    .await
                    .map(Answered::Reranked)
                    .map_err(ErrorReport::from),
            ),
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Rerank, &other)))
            }
        }
    }
}

/// Serve one tool call: `args` parsed and handed to `call`, with the
/// driver's inbound values from the dispatch's scope (`ToolContext`, as
/// `for_dispatch`), else an empty context, and every scope of the dispatch
/// attached, so the tool reaches its runtime by type for the length of the
/// call. What the tool published goes back beside the dispatch when the
/// driver asked for it ([`PublishedContext`](crate::tool::PublishedContext)
/// in the dispatch's scope); the result carries data only. A call to
/// another tool than `bound`, when given, is refused, so key routing never
/// runs a different tool than the bound target.
pub(crate) async fn call_tool<A, O>(
    kind: EffectKind,
    dispatch: Dispatch,
    bound: Option<&str>,
    call: impl AsyncFnOnce(&mut crate::tool::ToolContext, A) -> Result<O, ToolExecutionError>,
) -> Reply
where
    A: serde::de::DeserializeOwned,
    O: crate::tool::IntoToolOutput,
{
    let args = match (kind, bound) {
        (EffectKind::ToolCall { name, .. }, Some(bound)) if bound != name => {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Internal,
                format!("tool handler `{bound}` asked to run `{name}`"),
            )));
        }
        (EffectKind::ToolCall { args, .. }, _) => args,
        (
            other @ (EffectKind::Completion { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Retrieve { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }),
            _,
        ) => return Reply::Outcome(Err(wrong_family(EffectFamily::Tool, &other))),
    };
    let result = run_tool(&args, &dispatch, call).await;
    Reply::Outcome(Ok(Answered::ToolResult { result }))
}

/// The body of [`call_tool`] once the call is known to be the tool's: `args`
/// parsed and handed to `call` in the dispatch's context, and what the tool
/// published handed back.
pub(crate) async fn run_tool<A, O>(
    args: &str,
    dispatch: &Dispatch,
    call: impl AsyncFnOnce(&mut crate::tool::ToolContext, A) -> Result<O, ToolExecutionError>,
) -> crate::tool::ToolResult
where
    A: serde::de::DeserializeOwned,
    O: crate::tool::IntoToolOutput,
{
    let mut context = dispatch
        .scope::<crate::tool::ToolContext>()
        .map(|inbound| inbound.for_dispatch())
        .unwrap_or_default()
        .with_scopes(dispatch.scopes());
    let result = run_in(args, &mut context, call).await;
    if let Some(published) = dispatch.scope::<crate::tool::PublishedContext>() {
        published.publish(context);
    }
    result
}

/// A direct call of a tool, outside any dispatch: [`run_in`] over the
/// caller's inbound values (as `for_dispatch`), with what the tool published
/// handed back into `context` (as `accept_dispatch_result`).
pub(crate) async fn run_direct<A, O>(
    context: &mut crate::tool::ToolContext,
    args: serde_json::Value,
    call: impl AsyncFnOnce(&mut crate::tool::ToolContext, A) -> Result<O, ToolExecutionError>,
) -> Result<crate::tool::ToolOutput, ToolExecutionError>
where
    A: serde::de::DeserializeOwned,
    O: crate::tool::IntoToolOutput,
{
    let mut dispatched = context.for_dispatch();
    let result = run_in(&args.to_string(), &mut dispatched, call).await;
    context.accept_dispatch_result(dispatched);
    result.into_result()
}

/// Every tool call's body: `args` parsed and handed to `call` in `context`,
/// and its output converted.
async fn run_in<A, O>(
    args: &str,
    context: &mut crate::tool::ToolContext,
    call: impl AsyncFnOnce(&mut crate::tool::ToolContext, A) -> Result<O, ToolExecutionError>,
) -> crate::tool::ToolResult
where
    A: serde::de::DeserializeOwned,
    O: crate::tool::IntoToolOutput,
{
    match crate::tool::contextual::parse_tool_args::<A>(args) {
        Ok(args) => crate::tool::contextual::tool_result_from(call(context, args).await),
        Err(error) => crate::tool::ToolResult::failed(error),
    }
}

/// A [`Tool`] as a handler, keyed by its name.
pub struct ToolAdapter<T> {
    tool: T,
    embedding: Option<ToolEmbeddingDescriptor>,
}

impl<T: Tool> ToolAdapter<T> {
    /// Wrap a static tool.
    pub fn new(tool: T) -> Self {
        Self {
            tool,
            embedding: None,
        }
    }

    /// Wraps a tool with embedding descriptions and serialized reconstruction
    /// context. Returns an error if context serialization fails.
    pub fn retrievable(tool: T) -> Result<Self, serde_json::Error>
    where
        T: ToolEmbedding,
    {
        let embedding = ToolEmbeddingDescriptor {
            context: serde_json::to_value(tool.context())?,
            embedding_docs: tool.embedding_docs(),
        };
        Ok(Self {
            tool,
            embedding: Some(embedding),
        })
    }

    /// The wrapped tool.
    pub fn tool(&self) -> &T {
        &self.tool
    }

    /// The retrieval descriptor of a [`Self::retrievable`] adapter.
    pub fn embedding(&self) -> Option<&ToolEmbeddingDescriptor> {
        self.embedding.as_ref()
    }

    /// Runs the tool on JSON `args` with an empty [`ToolContext`], as
    /// [`Self::run_with`] does.
    ///
    /// [`ToolContext`]: crate::tool::ToolContext
    pub async fn run(
        &self,
        args: serde_json::Value,
    ) -> Result<crate::tool::ToolOutput, ToolExecutionError> {
        self.run_with(&mut crate::tool::ToolContext::new(), args)
            .await
    }

    /// Runs the tool on JSON `args` as a dispatch of its call runs it:
    /// arguments parsed, errors normalized by [`Tool::map_error`], output
    /// converted. The tool sees `context`'s inbound values, and what it
    /// publishes with `insert_result` lands in `context`. No registry or
    /// agent is involved.
    ///
    /// ```
    /// use rig_core::{serve::adapters::ToolAdapter, tool::{ContextValue, Tool, ToolContext}};
    /// use serde_json::json;
    ///
    /// #[derive(serde::Serialize, serde::Deserialize)]
    /// struct Calls(u32);
    /// impl ContextValue for Calls { const KEY: &'static str = "calls"; }
    ///
    /// struct Double;
    ///
    /// impl Tool for Double {
    ///     const NAME: &'static str = "double";
    ///     type Args = i64;
    ///     type Output = i64;
    ///     type Error = rig_core::tool::ToolContextError;
    ///
    ///     fn description(&self) -> String {
    ///         "Doubles a number.".to_string()
    ///     }
    ///
    ///     fn parameters(&self) -> serde_json::Value {
    ///         json!({ "type": "integer" })
    ///     }
    ///
    ///     async fn call(&self, context: &mut ToolContext, n: i64) -> Result<i64, Self::Error> {
    ///         context.insert_result(Calls(1))?;
    ///         Ok(n * 2)
    ///     }
    /// }
    ///
    /// let double = ToolAdapter::new(Double);
    /// let mut context = ToolContext::new();
    /// let output = futures::executor::block_on(double.run_with(&mut context, json!(21)))?;
    /// assert_eq!(output.as_json(), Some(&json!(42)));
    /// assert_eq!(context.result::<Calls>()?.map(|calls| calls.0), Some(1));
    /// assert!(futures::executor::block_on(double.run(json!("21"))).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub async fn run_with(
        &self,
        context: &mut crate::tool::ToolContext,
        args: serde_json::Value,
    ) -> Result<crate::tool::ToolOutput, ToolExecutionError> {
        run_direct(context, args, async |context, args| {
            self.call(context, args).await
        })
        .await
    }

    /// The tool's call with its error normalized, as every run makes it.
    async fn call(
        &self,
        context: &mut crate::tool::ToolContext,
        args: T::Args,
    ) -> Result<T::Output, ToolExecutionError> {
        self.tool
            .call(context, args)
            .await
            .map_err(|error| self.tool.map_error(error))
    }
}

impl<T> Serve for ToolAdapter<T>
where
    T: Tool + 'static,
{
    type Family = family::Tool;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: crate::effect::tool_key(T::NAME),
            family: FamilyDescriptor::Tool {
                name: T::NAME.to_owned(),
                description: self.tool.description(),
                parameters: self.tool.parameters(),
                embedding: self.embedding.clone(),
            },
            layers: Vec::new(),
        }
    }

    fn key(&self) -> HandlerKey {
        crate::effect::tool_key(T::NAME)
    }

    fn footprint(&self, kind: &EffectKind) -> Option<crate::tool::Touches> {
        match kind {
            EffectKind::ToolCall { name, args } if name == T::NAME => {
                let args = crate::tool::contextual::parse_tool_args::<T::Args>(args).ok()?;
                Some(self.tool.footprint(&args))
            }
            _ => None,
        }
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        call_tool(kind, dispatch, Some(T::NAME), async |context, args| {
            self.call(context, args).await
        })
        .await
    }
}

/// Contextual tool callback accepting JSON arguments and returning canonical
/// output or a tool execution error.
pub trait ToolCallback:
    for<'a> Fn(
        &'a mut crate::tool::ToolContext,
        serde_json::Value,
    ) -> WasmBoxedFuture<
        'a,
        Result<crate::tool::ToolOutput, crate::tool::ToolExecutionError>,
    > + WasmCompatSend
    + WasmCompatSync
{
}

impl<F> ToolCallback for F where
    F: for<'a> Fn(
            &'a mut crate::tool::ToolContext,
            serde_json::Value,
        ) -> WasmBoxedFuture<
            'a,
            Result<crate::tool::ToolOutput, crate::tool::ToolExecutionError>,
        > + WasmCompatSend
        + WasmCompatSync
{
}

/// A [`VectorStoreIndex`] as a handler. The index's filter type is rebuilt
/// from the dynamic filter on the wire; documents come back as JSON, and the
/// typed view deserialises on the client side.
pub struct RetrieveAdapter<I> {
    index: I,
    label: Option<String>,
}

impl<I> RetrieveAdapter<I> {
    /// Wrap `index` as an agent's own index, under the bare `retrieve` key.
    pub fn new(index: I) -> Self {
        Self { index, label: None }
    }

    /// Wrap `index` under `retrieve:<label>`, so a host can serve several
    /// indexes in one table and dispatch to each by
    /// [`retrieve_key`](crate::effect::retrieve_key).
    pub fn labelled(label: impl Into<String>, index: I) -> Self {
        Self {
            index,
            label: Some(label.into()),
        }
    }

    /// The wrapped index.
    pub fn index(&self) -> &I {
        &self.index
    }
}

impl<I, F> Serve for RetrieveAdapter<I>
where
    I: VectorStoreIndex<Filter = F> + 'static,
    F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
{
    type Family = family::Retrieve;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: self
                .label
                .as_deref()
                .map_or_else(|| HandlerKey::from("retrieve"), crate::effect::retrieve_key),
            family: FamilyDescriptor::Retrieve {},
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, _dispatch: Dispatch) -> Reply {
        match kind {
            EffectKind::Retrieve { query } => {
                let outcome = match query {
                    RetrieveQuery::TopN { req } => {
                        match req.try_map_filter(F::from_dynamic_filter) {
                            Ok(req) => self
                                .index
                                .top_n::<serde_json::Value>(req)
                                .await
                                .map(|results| {
                                    Answered::Documents(RetrievedDocuments::Scored(
                                        results
                                            .into_iter()
                                            .map(|result| {
                                                (
                                                    result.score,
                                                    result.id,
                                                    F::normalize_dynamic_document(result.document),
                                                )
                                            })
                                            .collect(),
                                    ))
                                })
                                .map_err(ErrorReport::from),
                            Err(error) => Err(ErrorReport::from(VectorStoreError::from(error))),
                        }
                    }
                    RetrieveQuery::TopNIds { req } => {
                        match req.try_map_filter(F::from_dynamic_filter) {
                            Ok(req) => self
                                .index
                                .top_n_ids(req)
                                .await
                                .map(|results| {
                                    Answered::Documents(RetrievedDocuments::Ids(
                                        results
                                            .into_iter()
                                            .map(|result| (result.score, result.id))
                                            .collect(),
                                    ))
                                })
                                .map_err(ErrorReport::from),
                            Err(error) => Err(ErrorReport::from(VectorStoreError::from(error))),
                        }
                    }
                };
                Reply::Outcome(outcome)
            }
            other @ (EffectKind::Completion { .. }
            | EffectKind::ToolCall { .. }
            | EffectKind::Embed { .. }
            | EffectKind::Rerank { .. }
            | EffectKind::Custom { .. }) => {
                Reply::Outcome(Err(wrong_family(EffectFamily::Retrieve, &other)))
            }
        }
    }
}
