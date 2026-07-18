//! Correlated owned memory effects.

use crate::{CorrelationId, OperationId, RunId, StoreId, TenantId, WorldId};
use bevy_ecs::prelude::Component;
use rig_core::{
    completion::Message,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};
use std::{fmt, sync::Arc};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryEffectError {
    #[error("memory backend failed: {message}")]
    Backend { message: String },
}

pub trait MemoryImplementation: WasmCompatSend + WasmCompatSync + 'static {
    fn load(
        &self,
        conversation_id: String,
    ) -> WasmBoxedFuture<'_, Result<Vec<Message>, MemoryEffectError>>;
    fn append(
        &self,
        conversation_id: String,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'_, Result<(), MemoryEffectError>>;
}

/// Explicit executable rebinding used when restoring persisted memory state.
#[derive(Clone)]
pub struct MemoryRebinding {
    pub store_id: StoreId,
    pub implementation: Arc<dyn MemoryImplementation>,
}

#[derive(Component, Clone, Debug)]
pub(crate) struct MemoryState {
    pub store_id: StoreId,
    pub conversation_id: String,
    pub loaded: bool,
    pub load_correlation: Option<(OperationId, CorrelationId, u64)>,
    pub appended_generation: Option<u64>,
    pub append_correlation: Option<(OperationId, CorrelationId, u64)>,
    pub persist_from: usize,
}

#[derive(Clone)]
pub struct MemoryLoadEffect {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub store_id: StoreId,
    pub conversation_id: String,
    pub(crate) implementation: Arc<dyn MemoryImplementation>,
}

impl fmt::Debug for MemoryLoadEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryLoadEffect")
            .field("run_id", &self.run_id)
            .field("operation_id", &self.operation_id)
            .field("correlation_id", &self.correlation_id)
            .field("generation", &self.generation)
            .field("store_id", &self.store_id)
            .field("conversation_id", &"<redacted>")
            .finish()
    }
}

impl MemoryLoadEffect {
    pub async fn execute(self) -> MemoryLoadIngress {
        let result = self.implementation.load(self.conversation_id).await;
        MemoryLoadIngress {
            world_id: self.world_id,
            tenant_id: self.tenant_id,
            run_id: self.run_id,
            operation_id: self.operation_id,
            correlation_id: self.correlation_id,
            generation: self.generation,
            store_id: self.store_id,
            result,
        }
    }
}

#[derive(Debug)]
pub struct MemoryLoadIngress {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub store_id: StoreId,
    pub result: Result<Vec<Message>, MemoryEffectError>,
}

#[derive(Clone)]
pub struct MemoryAppendEffect {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub store_id: StoreId,
    pub conversation_id: String,
    pub messages: Vec<Message>,
    pub(crate) implementation: Arc<dyn MemoryImplementation>,
}

impl fmt::Debug for MemoryAppendEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryAppendEffect")
            .field("run_id", &self.run_id)
            .field("generation", &self.generation)
            .field("store_id", &self.store_id)
            .field("conversation_id", &"<redacted>")
            .field("message_count", &self.messages.len())
            .finish()
    }
}

impl MemoryAppendEffect {
    pub async fn execute(self) -> MemoryAppendIngress {
        let result = self
            .implementation
            .append(self.conversation_id, self.messages)
            .await;
        MemoryAppendIngress {
            world_id: self.world_id,
            tenant_id: self.tenant_id,
            run_id: self.run_id,
            operation_id: self.operation_id,
            correlation_id: self.correlation_id,
            generation: self.generation,
            store_id: self.store_id,
            result,
        }
    }
}

/// Correlated acknowledgement for an externally executed memory append.
#[derive(Debug)]
pub struct MemoryAppendIngress {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub store_id: StoreId,
    pub result: Result<(), MemoryEffectError>,
}
