//! Register a recorded log's replayers on a classic-agent bus driver.
//! The shared replayer requires only rig-core; this adapter supplies the
//! independently enabled rig-agent integration.
//!
//! ```
//! use rig_cassette::{agent::replay::register_all, effect_log::EffectLog};
//!
//! fn register(log: &EffectLog, driver: &mut rig_agent::bus::BusDriver)
//!     -> Result<(), rig_core::error::ErrorReport>
//! {
//!     register_all(log, driver)
//! }
//! ```

use crate::effect_log::{EffectLog, EffectLogReplayer, RequestCheck};
use rig_core::error::ErrorReport;

use rig_agent::bus::BusDriver;

/// Register a payload-checking replayer for every key in `log` on `driver`.
/// Returns an error for invalid log metadata, incompatible families, or failed
/// handler registration.
pub fn register_all(log: &EffectLog, driver: &mut BusDriver) -> Result<(), ErrorReport> {
    register_all_checking(log, driver, RequestCheck::Payload)
}

/// Register replayers using `check` for request comparisons.
/// Returns the same validation and registration errors as [`register_all`].
pub fn register_all_checking(
    log: &EffectLog,
    driver: &mut BusDriver,
    check: RequestCheck,
) -> Result<(), ErrorReport> {
    EffectLogReplayer::check_header(log)?;
    for replayer in EffectLogReplayer::for_log(log)? {
        let key = replayer.key().clone();
        driver.register_erased(
            key,
            rig_core::serve::ErasedHandler::new(replayer.checking(check)),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
