//! The example's input bridge yields on absent/partial input and owns no thread.
#![cfg(unix)]
#![allow(clippy::unwrap_used, reason = "test assertions")]
#[path = "../examples/human_in_the_loop/input.rs"]
mod input;
use input::{Input, InputEvent};
use std::{io::Write, os::unix::net::UnixStream};

#[test]
fn absent_and_partial_input_yield_until_a_complete_line_or_eof() {
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let mut input = Input::new(reader);
    assert!(input.poll().unwrap().is_none());
    writer.write_all(b"appro").unwrap();
    assert!(input.poll().unwrap().is_none());
    writer.write_all(b"ve\n").unwrap();
    assert!(matches!(input.poll().unwrap(), Some(InputEvent::Line(line)) if line == "approve"));
    drop(writer);
    assert!(matches!(input.poll().unwrap(), Some(InputEvent::Closed)));
}

#[test]
fn dropping_an_idle_reader_releases_its_source_without_a_join_or_input() {
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let mut input = Input::new(reader);
    assert!(input.poll().unwrap().is_none());
    drop(input);
    assert!(writer.write_all(b"late").is_err());
}

#[test]
fn eof_preserves_the_last_unterminated_line() {
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let mut input = Input::new(reader);
    writer.write_all(b"deny").unwrap();
    drop(writer);
    assert!(matches!(input.poll().unwrap(), Some(InputEvent::Line(line)) if line == "deny"));
    assert!(matches!(input.poll().unwrap(), Some(InputEvent::Closed)));
}
