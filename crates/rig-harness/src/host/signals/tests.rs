use super::Signals;

#[test]
fn signals_a_millisecond_apart_ask_for_one_exit_and_force_none() {
    let signals = Signals::new();
    assert!(!signals.take());
    assert!(!signals.signal());
    std::thread::sleep(std::time::Duration::from_millis(1));
    assert!(!signals.signal());
    assert!(signals.take());
    assert!(!signals.take());
}
