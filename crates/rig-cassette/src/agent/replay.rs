//! Register a recorded log's replayers on a classic-agent bus driver.
//! The shared replayer requires only rig-core; this adapter supplies the
//! independently enabled rig-agent integration.

use crate::effect_log::{EffectLog, EffectLogReplayer, RequestCheck};
use rig_core::error::ErrorReport;

use rig_agent::bus::BusDriver;

/// Register a replayer for every key in `log` on `driver`. Refuses a log of
/// another format, and a log whose signature names a family its records do
/// not answer — before the first dispatch, not at the record where it would
/// have diverged.
pub fn register_all(log: &EffectLog, driver: &mut BusDriver) -> Result<(), ErrorReport> {
    register_all_checking(log, driver, RequestCheck::Payload)
}

/// [`register_all`] with every replayer comparing requests as `check` says.
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
