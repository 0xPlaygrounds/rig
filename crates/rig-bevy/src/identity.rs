use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

macro_rules! domain_id {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
        )]
        pub struct $name(pub u64);

        impl $name {
            /// Allocate a process-unique domain identifier.
            pub fn new() -> Self {
                Self::fresh()
            }

            pub(crate) fn fresh() -> Self {
                Self(next_id())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

domain_id!(WorldId);
domain_id!(TenantId);
domain_id!(AgentId);
domain_id!(RunId);
domain_id!(OperationId);
domain_id!(CorrelationId);
domain_id!(CapabilityId);
domain_id!(GrantId);
domain_id!(StoreId);
