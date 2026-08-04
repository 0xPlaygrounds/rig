//! Out-of-tree completion-provider authoring and runtime registration.
//!
//! [`ExternalProviderConfig`] is honest serde data: an exact-version driver
//! identity, model, and provider-owned JSON settings. Executable behavior and
//! capability data live in an [`ExternalCompletionProviderEntry`] registered
//! with the process-local [`ExternalProviderRegistry`].
//!
//! Typed providers implement [`ExternalCompletionProvider`].
//! [`ExternalCompletionProviderEntry::from_provider`] erases that authoring
//! type into private callbacks, just as `PortableDynamicTool::from_portable`
//! erases `PortableTool`. Runtime-facing agent types remain non-generic.

use std::{collections::HashMap, fmt, sync::Arc};

use rig_core::{
    completion::{CompletionError, CompletionRequest, CompletionResponse},
    http_runtime::HttpRuntime,
    providers::descriptor::ProviderDescriptor,
    streaming::CompletionStream,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Stable, namespaced identity for an external provider implementation and
/// its configuration schema.
///
/// IDs use `<namespace>/<driver>@<version>`, for example
/// `com.acme.weather/chat@1`. Registry lookup always uses the complete value;
/// there is no name or version fallback.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ExternalProviderId(String);

impl ExternalProviderId {
    /// Validate and construct an exact-version external provider ID.
    pub fn new(value: impl Into<String>) -> Result<Self, ExternalProviderIdError> {
        let value = value.into();
        validate_external_provider_id(&value)?;
        Ok(Self(value))
    }

    /// The stable serialized identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ExternalProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ExternalProviderId")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for ExternalProviderId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<String> for ExternalProviderId {
    type Error = ExternalProviderIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ExternalProviderId> for String {
    fn from(value: ExternalProviderId) -> Self {
        value.0
    }
}

fn validate_external_provider_id(value: &str) -> Result<(), ExternalProviderIdError> {
    let Some((qualified_name, version)) = value.rsplit_once('@') else {
        return Err(ExternalProviderIdError::MissingVersion {
            value: value.to_owned(),
        });
    };
    let Some((namespace, driver)) = qualified_name.rsplit_once('/') else {
        return Err(ExternalProviderIdError::MissingNamespace {
            value: value.to_owned(),
        });
    };
    if namespace.is_empty() || driver.is_empty() || version.is_empty() {
        return Err(ExternalProviderIdError::EmptyComponent {
            value: value.to_owned(),
        });
    }
    if value.chars().any(char::is_control) {
        return Err(ExternalProviderIdError::ContainsControl {
            value: value.to_owned(),
        });
    }
    if value.chars().any(char::is_whitespace) {
        return Err(ExternalProviderIdError::ContainsWhitespace {
            value: value.to_owned(),
        });
    }
    Ok(())
}

/// Validation failure for an [`ExternalProviderId`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ExternalProviderIdError {
    /// The `@version` component is absent.
    #[error("external provider ID {value:?} must end with `@<version>`")]
    MissingVersion { value: String },
    /// The namespaced `namespace/driver` component is absent.
    #[error("external provider ID {value:?} must contain `<namespace>/<driver>`")]
    MissingNamespace { value: String },
    /// Namespace, driver, or version is empty.
    #[error("external provider ID {value:?} contains an empty component")]
    EmptyComponent { value: String },
    /// Control characters are unsafe in stable IDs and log output.
    #[error("external provider ID {value:?} must not contain control characters")]
    ContainsControl { value: String },
    /// Whitespace would make the stable identifier ambiguous in logs/config.
    #[error("external provider ID {value:?} must not contain whitespace")]
    ContainsWhitespace { value: String },
}

/// Serializable selection of an out-of-tree completion provider.
///
/// This type deliberately contains no descriptor or executable behavior.
#[derive(Clone, Serialize, Deserialize)]
pub struct ExternalProviderConfig {
    /// Exact implementation and config-schema identity.
    pub driver: ExternalProviderId,
    /// Model selected for this agent.
    pub model: String,
    /// Provider-owned JSON settings.
    pub settings: serde_json::Value,
}

impl ExternalProviderConfig {
    /// Serialize typed provider settings into an external provider config.
    pub fn new<C>(
        driver: ExternalProviderId,
        model: impl Into<String>,
        settings: C,
    ) -> Result<Self, ExternalProviderConfigError>
    where
        C: Serialize,
    {
        let settings = serde_json::to_value(settings)
            .map_err(|_| ExternalProviderConfigError::SerializeSettings)?;
        Self::from_value(driver, model, settings)
    }

    /// Construct a config from already-serialized provider settings.
    pub fn from_value(
        driver: ExternalProviderId,
        model: impl Into<String>,
        settings: serde_json::Value,
    ) -> Result<Self, ExternalProviderConfigError> {
        let model = model.into();
        if model.is_empty() {
            return Err(ExternalProviderConfigError::EmptyModel);
        }
        Ok(Self {
            driver,
            model,
            settings,
        })
    }
}

impl fmt::Debug for ExternalProviderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalProviderConfig")
            .field("driver", &self.driver)
            .field("model", &self.model)
            .field("settings", &"<redacted>")
            .finish()
    }
}

/// Invalid serialized or typed settings for an external provider.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExternalProviderConfigError {
    /// The selected model is empty.
    #[error("external provider model must not be empty")]
    EmptyModel,
    /// Typed settings could not be represented as JSON.
    #[error("external provider settings could not be serialized")]
    SerializeSettings,
    /// Stored JSON did not match the provider's associated config type.
    #[error("external provider settings do not match the registered config schema")]
    InvalidSettings,
    /// The typed provider rejected an otherwise well-formed config.
    #[error("external provider settings were rejected: {message}")]
    Rejected {
        /// Content-free validation reason supplied by the provider.
        message: String,
    },
}

impl ExternalProviderConfigError {
    /// Reject typed settings with a provider-owned reason.
    pub fn rejected(message: impl Into<String>) -> Self {
        Self::Rejected {
            message: message.into(),
        }
    }

    fn invalid_settings(_source: serde_json::Error) -> Self {
        Self::InvalidSettings
    }
}

/// Owned capability sheet for a registered external completion provider.
///
/// It mirrors [`ProviderDescriptor`] without requiring external identity or
/// verification paths to have a forged `'static` lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OwnedProviderDescriptor {
    /// Canonical provider name used in normalized response metadata.
    pub name: String,
    /// Whether the provider supports tool calling.
    pub supports_tools: bool,
    /// Whether the provider supports native structured output.
    pub supports_response_format: bool,
    /// Whether streaming requests ask for usage on the terminal chunk.
    pub stream_include_usage: bool,
    /// Whether one chunk can contain a complete tool call.
    pub emits_complete_single_chunk_tool_calls: bool,
    /// Whether native structured output composes with tools.
    pub composes_native_output_with_tools: bool,
    /// Maximum embedding batch size, retained for descriptor parity.
    pub max_embedding_documents: Option<usize>,
    /// Optional credential-verification path, retained for descriptor parity.
    pub verify_path: Option<String>,
}

impl OwnedProviderDescriptor {
    /// A descriptor with every capability disabled.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            supports_tools: false,
            supports_response_format: false,
            stream_include_usage: false,
            emits_complete_single_chunk_tool_calls: false,
            composes_native_output_with_tools: false,
            max_embedding_documents: None,
            verify_path: None,
        }
    }

    /// Enable or disable tool calling.
    pub fn with_tools(mut self, value: bool) -> Self {
        self.supports_tools = value;
        self
    }

    /// Enable or disable native structured output.
    pub fn with_response_format(mut self, value: bool) -> Self {
        self.supports_response_format = value;
        self
    }

    /// Configure whether streamed usage is requested on the terminal chunk.
    pub fn with_stream_include_usage(mut self, value: bool) -> Self {
        self.stream_include_usage = value;
        self
    }

    /// Configure whole-tool-call streaming behavior.
    pub fn with_single_chunk_tool_calls(mut self, value: bool) -> Self {
        self.emits_complete_single_chunk_tool_calls = value;
        self
    }

    /// Configure native structured-output/tool composition.
    pub fn with_composes_native_output_with_tools(mut self, value: bool) -> Self {
        self.composes_native_output_with_tools = value;
        self
    }

    /// Borrow this owned descriptor through the common runtime view.
    pub fn as_view(&self) -> ProviderDescriptorView<'_> {
        ProviderDescriptorView {
            name: &self.name,
            supports_tools: self.supports_tools,
            supports_response_format: self.supports_response_format,
            stream_include_usage: self.stream_include_usage,
            emits_complete_single_chunk_tool_calls: self.emits_complete_single_chunk_tool_calls,
            composes_native_output_with_tools: self.composes_native_output_with_tools,
            max_embedding_documents: self.max_embedding_documents,
            verify_path: self.verify_path.as_deref(),
        }
    }
}

impl From<&ProviderDescriptor> for OwnedProviderDescriptor {
    fn from(descriptor: &ProviderDescriptor) -> Self {
        Self {
            name: descriptor.name.to_owned(),
            supports_tools: descriptor.supports_tools,
            supports_response_format: descriptor.supports_response_format,
            stream_include_usage: descriptor.stream_include_usage,
            emits_complete_single_chunk_tool_calls: descriptor
                .emits_complete_single_chunk_tool_calls,
            composes_native_output_with_tools: descriptor.composes_native_output_with_tools,
            max_embedding_documents: descriptor.max_embedding_documents,
            verify_path: descriptor.verify_path.map(str::to_owned),
        }
    }
}

/// Borrowed capability view returned for both bundled and external providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProviderDescriptorView<'a> {
    /// Canonical provider name.
    pub name: &'a str,
    /// Whether the provider supports tool calling.
    pub supports_tools: bool,
    /// Whether the provider supports native structured output.
    pub supports_response_format: bool,
    /// Whether streaming requests ask for usage on the terminal chunk.
    pub stream_include_usage: bool,
    /// Whether one chunk can contain a complete tool call.
    pub emits_complete_single_chunk_tool_calls: bool,
    /// Whether native structured output composes with tools.
    pub composes_native_output_with_tools: bool,
    /// Maximum embedding batch size.
    pub max_embedding_documents: Option<usize>,
    /// Optional credential-verification path.
    pub verify_path: Option<&'a str>,
}

impl<'a> From<&'a ProviderDescriptor> for ProviderDescriptorView<'a> {
    fn from(descriptor: &'a ProviderDescriptor) -> Self {
        Self {
            name: descriptor.name,
            supports_tools: descriptor.supports_tools,
            supports_response_format: descriptor.supports_response_format,
            stream_include_usage: descriptor.stream_include_usage,
            emits_complete_single_chunk_tool_calls: descriptor
                .emits_complete_single_chunk_tool_calls,
            composes_native_output_with_tools: descriptor.composes_native_output_with_tools,
            max_embedding_documents: descriptor.max_embedding_documents,
            verify_path: descriptor.verify_path,
        }
    }
}

/// Typed authoring contract for an out-of-tree completion provider.
///
/// The trait is deliberately not used as a trait object. Implementors return
/// ordinary futures; [`ExternalCompletionProviderEntry::from_provider`] owns
/// the runtime boxing boundary.
pub trait ExternalCompletionProvider: Sized + Send + Sync {
    /// Provider-owned typed representation of [`ExternalProviderConfig::settings`].
    type Config: DeserializeOwned + WasmCompatSend + 'static;

    /// Exact implementation and config-schema identity.
    fn id(&self) -> ExternalProviderId;

    /// Canonical runtime capability sheet.
    fn descriptor(&self) -> OwnedProviderDescriptor;

    /// Validate typed settings before invoking provider I/O.
    fn validate_config(&self, _config: &Self::Config) -> Result<(), ExternalProviderConfigError> {
        Ok(())
    }

    /// Fulfill one normalized completion request.
    fn complete(
        &self,
        config: Self::Config,
        model: String,
        runtime: HttpRuntime,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, CompletionError>> + WasmCompatSend;

    /// Open one normalized completion stream.
    fn open_stream(
        &self,
        config: Self::Config,
        model: String,
        runtime: HttpRuntime,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionStream, CompletionError>> + WasmCompatSend;
}

type ConfigValidator = dyn Fn(&ExternalProviderConfig) -> Result<(), ExternalProviderConfigError>
    + Send
    + Sync
    + 'static;
type CompleteCallback = dyn Fn(
        ExternalProviderConfig,
        HttpRuntime,
        CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<CompletionResponse, CompletionError>>
    + Send
    + Sync
    + 'static;
type StreamCallback = dyn Fn(
        ExternalProviderConfig,
        HttpRuntime,
        CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<CompletionStream, CompletionError>>
    + Send
    + Sync
    + 'static;

fn validate_entry_config(
    validator: &ConfigValidator,
    config: &ExternalProviderConfig,
) -> Result<(), ExternalProviderConfigError> {
    if config.model.is_empty() {
        return Err(ExternalProviderConfigError::EmptyModel);
    }
    validator(config)
}

#[derive(Clone, Copy)]
enum FulfillmentValidation {
    BeforeCallback,
    InCallback,
}

/// Concrete runtime record for one external completion provider.
///
/// Identity and capabilities remain ordinary owned data. Only heterogeneous
/// callback/future types are erased in private fields.
#[derive(Clone)]
pub struct ExternalCompletionProviderEntry {
    id: ExternalProviderId,
    descriptor: OwnedProviderDescriptor,
    validate_config: Arc<ConfigValidator>,
    fulfillment_validation: FulfillmentValidation,
    complete: Arc<CompleteCallback>,
    open_stream: Arc<StreamCallback>,
}

impl fmt::Debug for ExternalCompletionProviderEntry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalCompletionProviderEntry")
            .field("id", &self.id)
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl ExternalCompletionProviderEntry {
    /// Create an erased record from ordinary async callbacks over raw external
    /// config data.
    pub fn new<C, CFut, S, SFut>(
        id: ExternalProviderId,
        descriptor: OwnedProviderDescriptor,
        complete: C,
        open_stream: S,
    ) -> Self
    where
        C: Fn(ExternalProviderConfig, HttpRuntime, CompletionRequest) -> CFut
            + Send
            + Sync
            + 'static,
        CFut:
            Future<Output = Result<CompletionResponse, CompletionError>> + WasmCompatSend + 'static,
        S: Fn(ExternalProviderConfig, HttpRuntime, CompletionRequest) -> SFut
            + Send
            + Sync
            + 'static,
        SFut: Future<Output = Result<CompletionStream, CompletionError>> + WasmCompatSend + 'static,
    {
        Self {
            id,
            descriptor,
            validate_config: Arc::new(|_| Ok(())),
            fulfillment_validation: FulfillmentValidation::BeforeCallback,
            complete: Arc::new(move |config, runtime, request| {
                Box::pin(complete(config, runtime, request))
            }),
            open_stream: Arc::new(move |config, runtime, request| {
                Box::pin(open_stream(config, runtime, request))
            }),
        }
    }

    /// Create an erased callback record retaining shared host state.
    pub fn with_state<State, C, CFut, S, SFut>(
        id: ExternalProviderId,
        descriptor: OwnedProviderDescriptor,
        state: State,
        complete: C,
        open_stream: S,
    ) -> Self
    where
        State: Send + Sync + 'static,
        C: Fn(Arc<State>, ExternalProviderConfig, HttpRuntime, CompletionRequest) -> CFut
            + Send
            + Sync
            + 'static,
        CFut:
            Future<Output = Result<CompletionResponse, CompletionError>> + WasmCompatSend + 'static,
        S: Fn(Arc<State>, ExternalProviderConfig, HttpRuntime, CompletionRequest) -> SFut
            + Send
            + Sync
            + 'static,
        SFut: Future<Output = Result<CompletionStream, CompletionError>> + WasmCompatSend + 'static,
    {
        let state = Arc::new(state);
        let complete_state = Arc::clone(&state);
        Self::new(
            id,
            descriptor,
            move |config, runtime, request| {
                complete(Arc::clone(&complete_state), config, runtime, request)
            },
            move |config, runtime, request| {
                open_stream(Arc::clone(&state), config, runtime, request)
            },
        )
    }

    /// Erase a typed provider implementation into a concrete runtime record.
    pub fn from_provider<P>(provider: P) -> Self
    where
        P: ExternalCompletionProvider + 'static,
    {
        let id = provider.id();
        let descriptor = provider.descriptor();
        let provider = Arc::new(provider);
        let complete_provider = Arc::clone(&provider);
        let stream_provider = Arc::clone(&provider);
        let validate_provider = Arc::clone(&provider);

        Self::new(
            id,
            descriptor,
            move |config, runtime, request| {
                let provider = Arc::clone(&complete_provider);
                async move {
                    let settings = serde_json::from_value::<P::Config>(config.settings)
                        .map_err(ExternalProviderConfigError::invalid_settings)
                        .map_err(|source| {
                            CompletionError::from(ExternalProviderRuntimeError::InvalidConfig {
                                driver: config.driver.clone(),
                                source,
                            })
                        })?;
                    provider.validate_config(&settings).map_err(|source| {
                        CompletionError::from(ExternalProviderRuntimeError::InvalidConfig {
                            driver: config.driver.clone(),
                            source,
                        })
                    })?;
                    provider
                        .complete(settings, config.model, runtime, request)
                        .await
                }
            },
            move |config, runtime, request| {
                let provider = Arc::clone(&stream_provider);
                async move {
                    let settings = serde_json::from_value::<P::Config>(config.settings)
                        .map_err(ExternalProviderConfigError::invalid_settings)
                        .map_err(|source| {
                            CompletionError::from(ExternalProviderRuntimeError::InvalidConfig {
                                driver: config.driver.clone(),
                                source,
                            })
                        })?;
                    provider.validate_config(&settings).map_err(|source| {
                        CompletionError::from(ExternalProviderRuntimeError::InvalidConfig {
                            driver: config.driver.clone(),
                            source,
                        })
                    })?;
                    provider
                        .open_stream(settings, config.model, runtime, request)
                        .await
                }
            },
        )
        .with_typed_config_validator(move |config| {
            let settings = serde_json::from_value::<P::Config>(config.settings.clone())
                .map_err(ExternalProviderConfigError::invalid_settings)?;
            validate_provider.validate_config(&settings)
        })
    }

    /// Replace the default no-op raw-config validator.
    pub fn with_config_validator<V>(mut self, validator: V) -> Self
    where
        V: Fn(&ExternalProviderConfig) -> Result<(), ExternalProviderConfigError>
            + Send
            + Sync
            + 'static,
    {
        self.validate_config = Arc::new(validator);
        self.fulfillment_validation = FulfillmentValidation::BeforeCallback;
        self
    }

    fn with_typed_config_validator<V>(mut self, validator: V) -> Self
    where
        V: Fn(&ExternalProviderConfig) -> Result<(), ExternalProviderConfigError>
            + Send
            + Sync
            + 'static,
    {
        self.validate_config = Arc::new(validator);
        self.fulfillment_validation = FulfillmentValidation::InCallback;
        self
    }

    /// Exact registry key.
    pub fn id(&self) -> &ExternalProviderId {
        &self.id
    }

    /// Canonical handler-owned descriptor.
    pub fn descriptor(&self) -> &OwnedProviderDescriptor {
        &self.descriptor
    }

    pub(crate) fn validate(
        &self,
        config: &ExternalProviderConfig,
    ) -> Result<(), ExternalProviderConfigError> {
        validate_entry_config(self.validate_config.as_ref(), config)
    }

    fn validate_for_fulfillment(
        &self,
        config: &ExternalProviderConfig,
    ) -> Result<(), ExternalProviderConfigError> {
        match self.fulfillment_validation {
            FulfillmentValidation::BeforeCallback => {
                validate_entry_config(self.validate_config.as_ref(), config)
            }
            FulfillmentValidation::InCallback if config.model.is_empty() => {
                Err(ExternalProviderConfigError::EmptyModel)
            }
            FulfillmentValidation::InCallback => Ok(()),
        }
    }

    pub(crate) fn complete(
        &self,
        mut config: ExternalProviderConfig,
        runtime: HttpRuntime,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<CompletionResponse, CompletionError>> {
        if let Some(model) = request.model.as_ref() {
            config.model.clone_from(model);
        }
        if let Err(source) = self.validate_for_fulfillment(&config) {
            let error = CompletionError::from(ExternalProviderRuntimeError::InvalidConfig {
                driver: config.driver.clone(),
                source,
            });
            return Box::pin(std::future::ready(Err(error)));
        }
        (self.complete)(config, runtime, request)
    }

    pub(crate) fn open_stream(
        &self,
        mut config: ExternalProviderConfig,
        runtime: HttpRuntime,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<CompletionStream, CompletionError>> {
        if let Some(model) = request.model.as_ref() {
            config.model.clone_from(model);
        }
        if let Err(source) = self.validate_for_fulfillment(&config) {
            let error = CompletionError::from(ExternalProviderRuntimeError::InvalidConfig {
                driver: config.driver.clone(),
                source,
            });
            return Box::pin(std::future::ready(Err(error)));
        }
        (self.open_stream)(config, runtime, request)
    }
}

/// Immutable registry of exact-version external completion handlers.
///
/// [`Self::register`] consumes the registry and uses copy-on-write storage, so
/// existing runtime clones never observe replacement or mutation.
#[derive(Clone, Default)]
pub struct ExternalProviderRegistry {
    providers: Arc<HashMap<ExternalProviderId, ExternalCompletionProviderEntry>>,
}

impl fmt::Debug for ExternalProviderRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExternalProviderRegistry")
            .field("provider_ids", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ExternalProviderRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a new registry containing `provider`.
    pub fn register(
        mut self,
        provider: ExternalCompletionProviderEntry,
    ) -> Result<Self, ExternalProviderRegistryError> {
        if self.providers.contains_key(provider.id()) {
            return Err(ExternalProviderRegistryError::Duplicate {
                driver: provider.id().clone(),
            });
        }
        Arc::make_mut(&mut self.providers).insert(provider.id().clone(), provider);
        Ok(self)
    }

    /// Look up an exact-version provider record.
    pub fn get(&self, id: &ExternalProviderId) -> Option<&ExternalCompletionProviderEntry> {
        self.providers.get(id)
    }

    /// Number of registered external completion providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Whether the registry contains no providers.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

/// Failure while constructing an external-provider registry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ExternalProviderRegistryError {
    /// An exact-version ID is already present.
    #[error("external provider `{driver}` is already registered")]
    Duplicate { driver: ExternalProviderId },
}

/// Failure resolving or binding serialized external-provider configuration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExternalProviderRuntimeError {
    /// The runtime has no handler for this exact-version ID.
    #[error("external provider `{driver}` is not registered in this runtime")]
    NotRegistered { driver: ExternalProviderId },
    /// The handler rejected or could not deserialize the stored settings.
    #[error("external provider `{driver}` has invalid configuration: {source}")]
    InvalidConfig {
        /// Exact provider identity.
        driver: ExternalProviderId,
        /// Typed configuration failure.
        #[source]
        source: ExternalProviderConfigError,
    },
}

impl From<ExternalProviderRuntimeError> for CompletionError {
    fn from(error: ExternalProviderRuntimeError) -> Self {
        Self::RequestError(Box::new(error))
    }
}

#[allow(dead_code)]
fn _assert_external_records_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExternalCompletionProviderEntry>();
    assert_send_sync::<ExternalProviderRegistry>();
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[allow(dead_code)]
fn _assert_external_entry_accepts_worker_local_futures() {
    use std::rc::Rc;

    let Ok(id) = ExternalProviderId::new("local.browser/provider@1") else {
        return;
    };
    let _entry = ExternalCompletionProviderEntry::new(
        id,
        OwnedProviderDescriptor::named("browser-local"),
        |_config, _runtime, _request| async move {
            let local = Rc::new(());
            std::future::pending::<()>().await;
            drop(local);
            Ok(CompletionResponse::new(
                rig_core::OneOrMany::one(rig_core::message::AssistantContent::text("done")),
                rig_core::completion::Usage::new(),
                "browser-local",
            ))
        },
        |_config, _runtime, _request| async move {
            let local = Rc::new(());
            std::future::pending::<()>().await;
            drop(local);
            Ok(CompletionStream::from_stream(futures::stream::empty()))
        },
    );
}
