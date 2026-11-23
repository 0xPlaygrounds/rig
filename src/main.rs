//! The `rig` command, which `cargo install rig` installs: the launcher of the
//! rig coding agent ([`rig_cli::main`]).

// The launcher is native-only, so on the web the binary has no entry point.
#![cfg_attr(target_family = "wasm", no_main)]

#[cfg(not(target_family = "wasm"))]
fn main() -> std::process::ExitCode {
    rig_cli::main()
}
