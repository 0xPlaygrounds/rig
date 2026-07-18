//! ECS data and systems for grants, output policy, retry, and suppression.

use std::{collections::BTreeMap, fmt, sync::Arc};

use bevy_ecs::{component::Component, resource::Resource};
use rig_core::{
    completion::ToolDefinition,
    tool::{DynamicTool, IntoToolOutput, Tool, ToolExecutionError, ToolOutput, ToolResult},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

use crate::{
    components::{AdvertisedTool, CapabilitySnapshot},
    topology::{AgentId, CapabilityId, TenantId},
};

/// Persistable ECS policy attached to an agent entity.
#[derive(Component, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentPolicy {
    /// Stable owning agent.
    pub agent: AgentId,
    /// Total model-call limit inherited by new runs.
    pub max_calls: usize,
    /// Maximum concurrently executing tool bodies.
    pub max_tool_concurrency: usize,
    /// Invalid-tool disposition.
    pub invalid_tool: InvalidToolPolicy,
    /// Structured-output selection policy.
    pub output_mode: OutputMode,
    /// Structured-output recovery policy.
    pub response_retry: ResponseRetryPolicy,
    /// Optional portable JSON Schema value.
    pub output_schema: Option<serde_json::Value>,
    /// Optional runtime-only provider-specific request parameters.
    ///
    /// Snapshots omit this value; restoration requires explicit host rebinding.
    pub additional_params: Option<serde_json::Value>,
    /// Optional sampling temperature.
    pub temperature: Option<f64>,
    /// Optional output-token limit.
    pub max_tokens: Option<u64>,
}

impl fmt::Debug for AgentPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentPolicy")
            .field("agent", &self.agent)
            .field("max_calls", &self.max_calls)
            .field("max_tool_concurrency", &self.max_tool_concurrency)
            .field("invalid_tool", &invalid_tool_label(&self.invalid_tool))
            .field("output_mode", &self.output_mode)
            .field(
                "response_retry",
                &(
                    self.response_retry.max_retries,
                    self.response_retry.retries,
                    self.response_retry.best_effort,
                ),
            )
            .field(
                "output_schema",
                &self.output_schema.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "additional_params",
                &self.additional_params.as_ref().map(|_| "<redacted>"),
            )
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

fn invalid_tool_label(policy: &InvalidToolPolicy) -> &'static str {
    match policy {
        InvalidToolPolicy::Fail => "fail",
        InvalidToolPolicy::Retry { .. } => "retry",
        InvalidToolPolicy::Repair { .. } => "repair",
        InvalidToolPolicy::Skip { .. } => "skip",
        InvalidToolPolicy::Stop { .. } => "stop",
    }
}

/// ECS-native invalid-tool disposition.
#[derive(Component, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InvalidToolPolicy {
    /// Fail the run.
    Fail,
    /// Retry the model response with corrective feedback.
    Retry { feedback: String },
    /// Replace the name/arguments and revalidate against the same snapshot.
    Repair { name: String, arguments: String },
    /// Suppress execution and commit a skipped result.
    Skip { reason: String },
    /// Stop the run without dispatching the call.
    Stop { reason: String },
}

impl fmt::Debug for InvalidToolPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(invalid_tool_label(self))
    }
}

/// Structured-output strategy selected before each model operation.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OutputMode {
    /// Use provider-native schema constraints.
    Native,
    /// Use a collision-safe synthetic terminal tool.
    Tool,
    /// Add schema instructions to the prompt and decode best effort.
    Prompted,
    /// Select Native when provider composition permits it, otherwise Tool.
    #[default]
    Auto,
}

/// Bounded response retry policy stored as data, not callback hooks.
#[derive(Component, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseRetryPolicy {
    /// Maximum corrective retries.
    pub max_retries: usize,
    /// Retries already consumed.
    pub retries: usize,
    /// Corrective feedback inserted into the next preparation.
    pub feedback: String,
    /// Best-effort mode accepts the final undecodable response on exhaustion.
    pub best_effort: bool,
}

impl fmt::Debug for ResponseRetryPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResponseRetryPolicy")
            .field("max_retries", &self.max_retries)
            .field("retries", &self.retries)
            .field("feedback", &"<redacted>")
            .field("best_effort", &self.best_effort)
            .finish()
    }
}

impl ResponseRetryPolicy {
    /// Consume one retry if available.
    pub fn consume(&mut self) -> bool {
        if self.retries >= self.max_retries {
            return false;
        }
        self.retries += 1;
        true
    }
}

/// Explicit tool grant binding one tenant to one name/revision.
#[derive(Component, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolGrant {
    /// Tenant that may use the capability.
    pub tenant: TenantId,
    /// Provider-facing tool name.
    pub name: String,
    /// Exact allowed revision.
    pub revision: u64,
    /// Whether the grant is currently active.
    pub active: bool,
}

trait ErasedPortableTool: WasmCompatSend + WasmCompatSync {
    fn definition(&self, name: &str) -> ToolDefinition;
    fn execute(&self, arguments: String) -> WasmBoxedFuture<'_, ToolResult>;
}

struct TypedToolAdapter<T>(T);

impl<T> ErasedPortableTool for TypedToolAdapter<T>
where
    T: Tool,
{
    fn definition(&self, name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: Tool::description(&self.0),
            parameters: Tool::parameters(&self.0),
        }
    }

    fn execute(&self, arguments: String) -> WasmBoxedFuture<'_, ToolResult> {
        Box::pin(async move {
            let args = match serde_json::from_str::<T::Args>(&arguments) {
                Ok(args) => args,
                Err(original) if arguments.trim() == "null" => {
                    match serde_json::from_str::<T::Args>("{}") {
                        Ok(args) => args,
                        Err(_) => {
                            return ToolResult::failed(
                                ToolExecutionError::invalid_args(format!(
                                    "failed to parse tool arguments: {original}"
                                ))
                                .with_source(original),
                            );
                        }
                    }
                }
                Err(error) => {
                    return ToolResult::failed(
                        ToolExecutionError::invalid_args(format!(
                            "failed to parse tool arguments: {error}"
                        ))
                        .with_source(error),
                    );
                }
            };
            match Tool::call(&self.0, args).await {
                Ok(output) => match output.into_tool_output() {
                    Ok(output) => ToolResult::success(output),
                    Err(error) => ToolResult::failed(error),
                },
                Err(error) => ToolResult::failed(Tool::map_error(&self.0, error)),
            }
        })
    }
}

struct DynamicToolAdapter(DynamicTool);

impl ErasedPortableTool for DynamicToolAdapter {
    fn definition(&self, name: &str) -> ToolDefinition {
        let mut definition = self.0.definition();
        definition.name = name.to_string();
        definition
    }

    fn execute(&self, arguments: String) -> WasmBoxedFuture<'_, ToolResult> {
        Box::pin(async move {
            let args = match serde_json::from_str(&arguments) {
                Ok(args) => args,
                Err(error) => {
                    return ToolResult::failed(
                        ToolExecutionError::invalid_args(format!(
                            "failed to parse tool arguments: {error}"
                        ))
                        .with_source(error),
                    );
                }
            };
            match self.0.call(args).await {
                Ok(output) => ToolResult::success(output),
                Err(error) => ToolResult::failed(error),
            }
        })
    }
}

#[derive(Clone)]
struct ToolRevision {
    tenant: TenantId,
    revision: u64,
    implementation: Arc<dyn ErasedPortableTool>,
    retired: bool,
}

/// Runtime implementation registry. Domain snapshots persist names/revisions,
/// never these trait-object pointers.
#[derive(Resource, Default, Clone)]
pub struct ToolCatalog {
    tools: BTreeMap<(String, u64), ToolRevision>,
    current: BTreeMap<(TenantId, String), u64>,
    next_revision: BTreeMap<String, u64>,
}

impl ToolCatalog {
    /// Register a portable typed tool and return its monotonically increasing revision.
    pub fn register<T>(&mut self, tenant: TenantId, tool: T) -> u64
    where
        T: Tool + 'static,
    {
        self.register_named(
            tenant,
            T::NAME.to_string(),
            Arc::new(TypedToolAdapter(tool)),
        )
    }

    /// Register a portable dynamic tool and return its revision.
    pub fn register_dynamic(&mut self, tenant: TenantId, tool: DynamicTool) -> u64 {
        let name = tool.name().to_string();
        self.register_named(tenant, name, Arc::new(DynamicToolAdapter(tool)))
    }

    pub(crate) fn bind<T>(&mut self, tenant: TenantId, revision: u64, tool: T) -> bool
    where
        T: Tool + 'static,
    {
        self.bind_named(
            tenant,
            T::NAME.to_string(),
            revision,
            Arc::new(TypedToolAdapter(tool)),
        )
    }

    pub(crate) fn bind_dynamic(
        &mut self,
        tenant: TenantId,
        revision: u64,
        tool: DynamicTool,
    ) -> bool {
        let name = tool.name().to_string();
        self.bind_named(tenant, name, revision, Arc::new(DynamicToolAdapter(tool)))
    }

    fn bind_named(
        &mut self,
        tenant: TenantId,
        name: String,
        revision: u64,
        implementation: Arc<dyn ErasedPortableTool>,
    ) -> bool {
        if revision == 0 || self.tools.contains_key(&(name.clone(), revision)) {
            return false;
        }
        self.tools.insert(
            (name.clone(), revision),
            ToolRevision {
                tenant,
                revision,
                implementation,
                retired: false,
            },
        );
        let next = self.next_revision.entry(name.clone()).or_default();
        *next = (*next).max(revision);
        let current = self.current.entry((tenant, name)).or_default();
        *current = (*current).max(revision);
        true
    }

    pub(crate) fn contains(&self, tenant: TenantId, name: &str, revision: u64) -> bool {
        self.tools
            .get(&(name.to_string(), revision))
            .is_some_and(|tool| tool.tenant == tenant)
    }

    fn register_named(
        &mut self,
        tenant: TenantId,
        name: String,
        implementation: Arc<dyn ErasedPortableTool>,
    ) -> u64 {
        let revision = self
            .next_revision
            .get(&name)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        self.tools.insert(
            (name.clone(), revision),
            ToolRevision {
                tenant,
                revision,
                implementation,
                retired: false,
            },
        );
        self.next_revision.insert(name.clone(), revision);
        self.current.insert((tenant, name), revision);
        revision
    }

    /// Retire future advertisement while preserving pinned in-flight revisions.
    pub fn retire(&mut self, name: &str, revision: u64) -> bool {
        let Some(tool) = self.tools.get_mut(&(name.to_string(), revision)) else {
            return false;
        };
        tool.retired = true;
        if self.current.get(&(tool.tenant, name.to_string())) == Some(&revision) {
            self.current.remove(&(tool.tenant, name.to_string()));
        }
        true
    }

    /// Build an immutable, sorted advertised snapshot after tenant/grant checks.
    pub fn snapshot(
        &self,
        effect: crate::topology::EffectIdentity,
        grants: &[ToolGrant],
    ) -> CapabilitySnapshot {
        let mut tools = self
            .current
            .iter()
            .filter_map(|((tenant, name), revision)| {
                if *tenant != effect.tenant {
                    return None;
                }
                let tool = self.tools.get(&(name.clone(), *revision))?;
                let granted = grants.iter().any(|grant| {
                    grant.active
                        && grant.tenant == effect.tenant
                        && grant.name == *name
                        && grant.revision == *revision
                });
                (tool.tenant == effect.tenant && !tool.retired && granted).then(|| AdvertisedTool {
                    name: name.clone(),
                    revision: tool.revision,
                    definition: tool.implementation.definition(name),
                })
            })
            .collect::<Vec<_>>();
        tools.sort_by(|left, right| left.name.cmp(&right.name));
        CapabilitySnapshot {
            id: CapabilityId::allocate(),
            effect,
            run: effect.run,
            tenant: effect.tenant,
            tools,
        }
    }

    /// Execute the exact snapshotted revision after validating tenant identity.
    pub async fn execute(
        &self,
        tenant: TenantId,
        name: &str,
        revision: u64,
        arguments: String,
    ) -> ToolResult {
        let Some(tool) = self.tools.get(&(name.to_string(), revision)) else {
            return ToolResult::failed(ToolExecutionError::not_found(format!(
                "tool `{name}` revision {revision} is not bound"
            )));
        };
        if tool.tenant != tenant {
            return ToolResult::failed(ToolExecutionError::permission_denied(
                "tool capability belongs to another tenant",
            ));
        }
        tool.implementation.execute(arguments).await
    }
}

/// Collision-safe name for a synthetic structured-output tool.
pub fn synthetic_output_tool_name(existing: impl IntoIterator<Item = String>) -> String {
    let existing = existing
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let base = "__rig_submit";
    if !existing.contains(base) {
        return base.to_string();
    }
    let mut suffix = 1_u64;
    loop {
        let candidate = format!("{base}_{suffix}");
        if !existing.contains(&candidate) {
            return candidate;
        }
        suffix = suffix.saturating_add(1);
    }
}

/// Convert a canonical result to the model-visible portable output.
pub fn result_output(result: &ToolResult) -> &ToolOutput {
    result.output()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::RunId;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Args {
        value: i32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("test tool failed")]
    struct TestError;

    struct AddOne;

    impl Tool for AddOne {
        const NAME: &'static str = "add_one";
        type Args = Args;
        type Output = i32;
        type Error = TestError;

        fn description(&self) -> String {
            "Add one".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args.value + 1)
        }
    }

    #[test]
    fn snapshots_are_tenant_scoped_and_revision_pinned() {
        let mut catalog = ToolCatalog::default();
        let first = catalog.register(TenantId(1), AddOne);
        let run = RunId::allocate();
        let effect = crate::topology::EffectIdentity {
            world: crate::topology::WorldId::allocate(),
            tenant: TenantId(1),
            run,
            operation: crate::topology::OperationId::allocate(),
            generation: crate::topology::Generation(0),
            correlation: 1,
        };
        let snapshot = catalog.snapshot(
            effect,
            &[ToolGrant {
                tenant: TenantId(1),
                name: "add_one".into(),
                revision: first,
                active: true,
            }],
        );
        assert_eq!(snapshot.tools.len(), 1);
        assert_eq!(snapshot.tools[0].revision, first);

        let second = catalog.register(TenantId(1), AddOne);
        assert_ne!(first, second);
        assert_eq!(snapshot.tools[0].revision, first);
    }

    #[test]
    fn same_name_registrations_remain_current_for_each_tenant() {
        let mut catalog = ToolCatalog::default();
        let first = catalog.register(TenantId(1), AddOne);
        let second = catalog.register(TenantId(2), AddOne);
        let snapshot_for = |catalog: &ToolCatalog, tenant, revision| {
            let effect = crate::topology::EffectIdentity {
                world: crate::topology::WorldId::allocate(),
                tenant,
                run: RunId::allocate(),
                operation: crate::topology::OperationId::allocate(),
                generation: crate::topology::Generation(0),
                correlation: 1,
            };
            catalog.snapshot(
                effect,
                &[ToolGrant {
                    tenant,
                    name: "add_one".into(),
                    revision,
                    active: true,
                }],
            )
        };

        assert_eq!(snapshot_for(&catalog, TenantId(1), first).tools.len(), 1);
        assert_eq!(snapshot_for(&catalog, TenantId(2), second).tools.len(), 1);
    }

    #[test]
    fn synthetic_tool_names_do_not_collide() {
        assert_eq!(
            synthetic_output_tool_name(["__rig_submit".into(), "__rig_submit_1".into()]),
            "__rig_submit_2"
        );
    }

    #[test]
    fn retry_policy_is_bounded() {
        let mut retry = ResponseRetryPolicy {
            max_retries: 1,
            retries: 0,
            feedback: "valid JSON required".into(),
            best_effort: false,
        };
        assert!(retry.consume());
        assert!(!retry.consume());
    }

    #[test]
    fn provider_request_parameters_are_redacted_from_debug_output() {
        let secret = "never-log-this";
        let policy = AgentPolicy {
            agent: AgentId::allocate(),
            max_calls: 1,
            max_tool_concurrency: 1,
            invalid_tool: InvalidToolPolicy::Repair {
                name: secret.into(),
                arguments: secret.into(),
            },
            output_mode: OutputMode::Auto,
            response_retry: ResponseRetryPolicy {
                max_retries: 0,
                retries: 0,
                feedback: secret.into(),
                best_effort: true,
            },
            output_schema: None,
            additional_params: Some(serde_json::json!({"secret": secret})),
            temperature: Some(0.5),
            max_tokens: Some(128),
        };

        let debug = format!("{policy:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains(secret));
        assert!(!format!("{:?}", policy.invalid_tool).contains(secret));
        assert!(!format!("{:?}", policy.response_retry).contains(secret));
    }
}
