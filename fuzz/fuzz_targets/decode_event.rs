//! Any JSON as an event record: typed decoding never panics, and a decoded
//! built-in event re-encodes to data that decodes to the same event.
#![no_main]

use libfuzzer_sys::fuzz_target;
use rig_protocol::{Event, EventRecord};

fuzz_target!(|data: &[u8]| {
    let Ok(record) = serde_json::from_slice::<EventRecord>(data) else {
        return;
    };
    if let Some(event) = Event::decode(&record) {
        let again = EventRecord {
            data: event.encode(),
            ..record
        };
        assert_eq!(Event::decode(&again), Some(event));
    }
});
