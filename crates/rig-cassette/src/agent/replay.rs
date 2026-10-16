//! Register a recorded log's replayers in a classic agent's handler table.
//! The shared replayer requires only rig-core; this adapter supplies the
//! independently enabled rig-agent integration.
//!
//! ```
//! use rig_cassette::{agent::replay::register_all, effect_log::EffectLog};
//!
//! fn register(log: &EffectLog, handlers: &rig_agent::handlers::Handlers)
//!     -> Result<(), rig_core::error::ErrorReport>
//! {
//!     register_all(log, handlers)
//! }
//! ```

use crate::effect_log::{EffectLog, EffectLogReplayer, RequestCheck};
use rig_core::error::ErrorReport;

use rig_agent::handlers::Handlers;

/// Register a payload-checking replayer for every key in `log` in `handlers`.
/// Returns an error for invalid log metadata, incompatible families, or failed
/// handler registration.
pub fn register_all(log: &EffectLog, handlers: &Handlers) -> Result<(), ErrorReport> {
    register_all_checking(log, handlers, RequestCheck::Payload)
}

/// Register replayers using `check` for request comparisons.
/// Returns the same validation and registration errors as [`register_all`].
pub fn register_all_checking(
    log: &EffectLog,
    handlers: &Handlers,
    check: RequestCheck,
) -> Result<(), ErrorReport> {
    EffectLogReplayer::check_header(log)?;
    for replayer in EffectLogReplayer::for_log(log)? {
        let key = replayer.key().clone();
        handlers.register_erased(
            key,
            rig_core::serve::ErasedHandler::new(replayer.checking(check)),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
