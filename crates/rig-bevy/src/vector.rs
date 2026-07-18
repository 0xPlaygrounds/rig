//! Owned vector-store effects for hosted or local ECS executors.

use crate::{CorrelationId, OperationId, RunId, StoreId, TenantId, WorldId};
use rig_core::vector_store::{
    VectorStoreError, VectorStoreIndexDyn,
    request::{Filter, VectorSearchRequest},
};
use std::{fmt, sync::Arc};

/// Owned vector query. No ECS borrow or runtime guard enters the future.
#[derive(Clone)]
pub struct VectorSearchEffect {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub store_id: StoreId,
    pub request: VectorSearchRequest<Filter<serde_json::Value>>,
    pub(crate) implementation: Arc<dyn VectorStoreIndexDyn>,
}

impl fmt::Debug for VectorSearchEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorSearchEffect")
            .field("run_id", &self.run_id)
            .field("operation_id", &self.operation_id)
            .field("correlation_id", &self.correlation_id)
            .field("generation", &self.generation)
            .field("store_id", &self.store_id)
            .field("request", &"<redacted>")
            .finish()
    }
}

impl VectorSearchEffect {
    pub async fn execute(self) -> VectorSearchIngress {
        let result = self.implementation.top_n(self.request).await;
        VectorSearchIngress {
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

/// Correlated vector result validated by runtime ingress before use.
#[derive(Debug)]
pub struct VectorSearchIngress {
    pub world_id: WorldId,
    pub tenant_id: TenantId,
    pub run_id: RunId,
    pub operation_id: OperationId,
    pub correlation_id: CorrelationId,
    pub generation: u64,
    pub store_id: StoreId,
    pub result: Result<Vec<(f64, String, serde_json::Value)>, VectorStoreError>,
}
