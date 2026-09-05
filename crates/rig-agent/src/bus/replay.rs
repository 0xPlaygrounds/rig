//! Registering a log's replayers on this bus's driver. The log, the
//! recorder and the replayer itself are rig-effect-log's and need rig-core
//! alone; putting a replayer behind every key of a log on a [`BusDriver`]
//! is this runtime's, so it lives here (a second runtime registers the same
//! replayer through its own registry).

use rig_core::error::ErrorReport;
use rig_effect_log::{EffectLog, EffectLogReplayer, RequestCheck};

use super::BusDriver;

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

#[cfg(all(test, not(rig_loom)))]
mod tests;
