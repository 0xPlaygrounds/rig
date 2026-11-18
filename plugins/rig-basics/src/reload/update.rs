//! `/update`: reinstalls rig through the launcher's `rig update`, which
//! says the command it runs and builds the agent with the new launcher,
//! then restarts as `/reload` does; the launcher, whose own file was
//! replaced, starts the new one on the same session. `/update cancel`
//! drops a queued update or stops a running one.

use rig_harness::prelude::*;

use crate::reload::Rebuild;

/// `/update`, on the rebuild it shares with `/reload`.
#[derive(Default)]
pub struct UpdatePlugin;

impl Plugin for UpdatePlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "update",
            "Reinstall rig with cargo (from crates.io, or the checkout it was installed from), \
             build the agent with it and restart once no turn runs; `/update cancel` cancels",
            update,
        );
        super::rebuilds(app);
    }
}

fn update(
    In(args): In<CommandArgs>,
    turns: Query<(), With<TurnOf>>,
    status: ResMut<super::ReloadStatus>,
    commands: Commands,
    notices: MessageWriter<Notice>,
) {
    super::queue(args, Rebuild::Update, turns, status, commands, notices);
}

#[cfg(test)]
mod tests;
