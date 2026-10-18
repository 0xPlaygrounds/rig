//! The command-line verbs of the rig agent, each a client of the rig agent
//! server through `rig-client`. The generated agent binary calls [`run`] in
//! its `cli` role; the Bevy app is never built for it.

use rig_protocol::Exit;

/// Runs the verb in `args` (without the program name) and returns the exit
/// code: one of [`Exit`]'s.
pub fn run(args: Vec<String>) -> u8 {
    let _ = args;
    eprintln!("rig: the command-line verbs are not built yet");
    Exit::Usage.code()
}
