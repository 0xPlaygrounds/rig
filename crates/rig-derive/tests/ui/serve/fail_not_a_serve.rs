//! A type that does not implement `Serve` is told to implement `Serve` —
//! never the handler table's boxed trait or the sealed family trait.

use rig_agent::handlers::Handlers;

struct NotAHandler;

fn main() {
    let handlers = Handlers::default();
    let _ = handlers.register("key", NotAHandler);
}
