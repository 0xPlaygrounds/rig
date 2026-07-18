//! Tenant-scoped tool capabilities, grants, immutable snapshots, and owned effects.

use crate::{AgentId, CapabilityId, CorrelationId, GrantId, OperationId, RunId, TenantId, WorldId};
use bevy_ecs::prelude::Component;
use rig_core::{
    completion::ToolDefinition,
    tool::{IntoToolOutput, Tool, ToolExecutionError, ToolResult, tool_definition},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};
use std::{fmt, sync::Arc};

/// Object-safe adapter for a portable tool implementation.
pub trait ToolImplementation: WasmCompatSend + WasmCompatSync + 'static {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, arguments: String) -> WasmBoxedFuture<'_, ToolResult>;
}

/// Adapter that makes any portable [`Tool`] executable by the ECS runtime.
#[derive(Clone, Debug)]
pub struct PortableTool<T>(pub T);

impl<T> PortableTool<T> {
    pub fn new(tool: T) -> Self {
        Self(tool)
    }
}

impl<T> ToolImplementation for PortableTool<T>
where
    T: Tool + 'static,
{
    fn definition(&self) -> ToolDefinition {
        tool_definition(&self.0)
    }

    fn execute(&self, arguments: String) -> WasmBoxedFuture<'_, ToolResult> {
        Box::pin(async move {
            let args = match serde_json::from_str(&arguments) {
                Ok(args) => args,
                Err(error) => {
                    return ToolResult::failed(ToolExecutionError::invalid_args(error.to_string()));
                }
            };
            match self.0.call(args).await {
                Ok(output) => match output.into_tool_output() {
                    Ok(output) => ToolResult::success(output),
                    Err(error) => ToolResult::failed(error),
                },
                Err(error) => ToolResult::failed(self.0.map_error(error)),
            }
        })
    }
}

/// Authoritative capability metadata. Implementations live in the runtime's
/// non-persisted rebinding table, never in ECS or snapshots.
#[derive(Component, Clone, Debug)]
pub struct ToolCapability {
    pub capability_id: CapabilityId,
    pub tenant_id: TenantId,
    pub revision: u64,
    pub definition: ToolDefinition,
    pub retired: bool,
}

/// Explicit permission from an agent to one tenant-owned capability.
#[derive(Component, Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilityGrant {
    pub grant_id: GrantId,
    pub tenant_id: TenantId,
    pub agent_id: AgentId,
    pub capability_id: CapabilityId,
}

/// Exact immutable capability revision advertised for one turn.
#[derive(Clone, Debug)]
pub struct ToolSnapshotEntry {
    pub capability_id: CapabilityId,
    pub revision: u64,
    pub definition: ToolDefinition,
}

/// Immutable per-turn advertised tool snapshot.
#[derive(Clone, Debug)]
pub struct ToolSnapshot {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub generation: u64,
    pub entries: Vec<ToolSnapshotEntry>,
}

/// Owned tool effect. No ECS borrow or lock guard enters execution.
#[derive(Clone)]
pub struct ToolEffectRequest {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub call_index: usize,
    pub capability_id: CapabilityId,
    pub revision: u64,
    pub name: String,
    pub arguments: String,
    pub(crate) implementation: Arc<dyn ToolImplementation>,
}

impl fmt::Debug for ToolEffectRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolEffectRequest")
            .field("world_id", &self.world_id)
            .field("tenant_id", &self.tenant_id)
            .field("run_id", &self.run_id)
            .field("operation_id", &self.operation_id)
            .field("correlation_id", &self.correlation_id)
            .field("generation", &self.generation)
            .field("call_index", &self.call_index)
            .field("capability_id", &self.capability_id)
            .field("revision", &self.revision)
            .field("name", &self.name)
            .field("arguments", &"<redacted>")
            .finish()
    }
}

impl ToolEffectRequest {
    /// Execute the snapshotted implementation with owned arguments.
    pub async fn execute(self) -> ToolIngress {
        let result = self.implementation.execute(self.arguments).await;
        ToolIngress {
            world_id: self.world_id,
            tenant_id: self.tenant_id,
            run_id: self.run_id,
            operation_id: self.operation_id,
            correlation_id: self.correlation_id,
            generation: self.generation,
            call_index: self.call_index,
            capability_id: self.capability_id,
            revision: self.revision,
            name: self.name,
            result,
        }
    }
}

/// Owned tool completion committed only by ingress validation.
#[derive(Clone, Debug)]
pub struct ToolIngress {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub call_index: usize,
    pub capability_id: CapabilityId,
    pub revision: u64,
    pub name: String,
    pub result: ToolResult,
}

#[derive(Clone)]
pub(crate) struct BoundTool {
    pub entity: bevy_ecs::entity::Entity,
    pub revision: u64,
    pub implementation: Arc<dyn ToolImplementation>,
}

/// Explicit executable rebinding supplied during snapshot restoration.
#[derive(Clone)]
pub struct ToolRebinding {
    pub capability_id: CapabilityId,
    pub revision: u64,
    pub implementation: Arc<dyn ToolImplementation>,
}
