//! A bounded, polled line reader owned by the CLI loop. No worker thread is
//! started, so dropping the reader never waits for a blocked stdin read.

use nix::{
    sys::{
        select::{FdSet, select},
        time::TimeVal,
    },
    unistd::read,
};
use std::{io, os::fd::AsFd};

pub enum InputEvent {
    Line(String),
    Closed,
}

pub struct Input<F> {
    source: F,
    line: Vec<u8>,
}

impl<F: AsFd> Input<F> {
    pub fn new(source: F) -> Self {
        Self {
            source,
            line: Vec::new(),
        }
    }

    pub fn poll(&mut self) -> io::Result<Option<InputEvent>> {
        // The CLI owns this input source: no other reader may race the readiness
        // check. Raw reads avoid stdio buffering hiding bytes from select.
        for _ in 0..256 {
            let mut ready = FdSet::new();
            ready.insert(self.source.as_fd());
            let mut timeout = TimeVal::new(0, 0);
            if select(None, &mut ready, None, None, &mut timeout)? == 0 {
                return Ok(None);
            }
            let mut byte = [0];
            if read(&self.source, &mut byte)? == 0 {
                return if self.line.is_empty() {
                    Ok(Some(InputEvent::Closed))
                } else {
                    self.take_line().map(Some)
                };
            }
            if byte[0] == b'\n' {
                return self.take_line().map(Some);
            }
            self.line.push(byte[0]);
            if self.line.len() > 4096 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "approval input exceeds 4096 bytes",
                ));
            }
        }
        Ok(None)
    }

    fn take_line(&mut self) -> io::Result<InputEvent> {
        String::from_utf8(std::mem::take(&mut self.line))
            .map(InputEvent::Line)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}
