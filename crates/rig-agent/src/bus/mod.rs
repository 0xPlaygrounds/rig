#![forbid(unsafe_code)]

//! Effect dispatch, handler registration, and serving through a host-driven bus.
//!
//! [`Dispatcher`] submits work, [`Registrar`] changes handlers, and [`BusDriver`]
//! owns and polls them. Hosts must drive or spawn the driver; dropping it closes
//! the bus. Handler-scoped dispatch preserves cancellation ancestry.
//!
//! ```
//! let (dispatcher, registrar, driver) = rig_agent::bus::Bus::channel();
//! assert_eq!(driver.in_flight(), 0);
//! ```

mod dispatcher;
mod driver;
mod handle;
mod registrar;

pub use dispatcher::{BusId, DispatchOptions, Dispatcher, EffectStream, Pending};
pub use driver::BusDriver;
pub(crate) use driver::Recording;
pub(crate) use handle::wrap_stream;
pub use handle::{
    Completion, DispatchScope, EmbedHandle, Handle, IndexHandle, MemoryHandle, ModelHandle,
    RerankHandle, Retrieval, ToolAnswer, ToolCall, ToolHandle, Typed,
};
pub use registrar::Registrar;
use rig_core::serve::ServingPolicy;

use std::sync::Arc;

// Shared dispatch values must remain thread-safe even when WASM handlers are not.
const _: () = {
    const fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    const fn assert_send_static<T: Send + 'static>() {}
    assert_send_sync_static::<ModelHandle>();
    assert_send_sync_static::<Dispatcher>();
    assert_send_static::<Pending>();
    assert_send_static::<EffectStream>();
};

/// Constructors for a bus.
#[derive(Debug, Clone, Copy)]
pub struct Bus;

impl Bus {
    /// A bus with the default [`ServingPolicy`]: the dispatcher, the registrar
    /// and the driver. Register handlers on the driver, then drive it or
    /// spawn it; register through the registrar once it is spawned.
    pub fn channel() -> (Dispatcher, Registrar, BusDriver) {
        Self::channel_with(ServingPolicy::default())
    }

    /// A bus with an explicit config.
    pub fn channel_with(config: ServingPolicy) -> (Dispatcher, Registrar, BusDriver) {
        let shared = Arc::new(dispatcher::Shared::new(config));
        let mailbox = Arc::new(registrar::Mailbox::new());
        let dispatcher = Dispatcher::open(shared.clone(), config.stream_capacity.max(1));
        let driver = BusDriver::new(shared, mailbox, config);
        let registrar = driver.registrar();
        (dispatcher, registrar, driver)
    }

    /// A bus whose driver is handed to `spawn` after `register` has filled
    /// its handler table. `spawn` is the host's executor entry point
    /// (`tokio::spawn`, a task pool, `spawn_local`); rig-agent supplies none.
    pub fn new_with(
        config: ServingPolicy,
        register: impl FnOnce(&mut BusDriver),
        spawn: impl FnOnce(BusDriver),
    ) -> (Dispatcher, Registrar) {
        let (dispatcher, registrar, mut driver) = Self::channel_with(config);
        register(&mut driver);
        spawn(driver);
        (dispatcher, registrar)
    }
}

#[cfg(all(test, rig_loom))]
mod loom_models;
#[cfg(all(test, not(rig_loom)))]
mod tests;
