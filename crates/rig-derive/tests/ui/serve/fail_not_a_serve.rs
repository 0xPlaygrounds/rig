//! A type that does not implement `Serve` is told to implement `Serve` —
//! never the bus's own boxed trait or the sealed family trait.

use rig_core::bus::Bus;

struct NotAHandler;

fn main() {
    let (_dispatcher, registrar, _driver) = Bus::channel();
    let _ = registrar.register("key", NotAHandler);
}
