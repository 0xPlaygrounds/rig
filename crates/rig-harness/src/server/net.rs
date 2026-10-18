//! The I/O edge: one thread accepts, and each connection has a reader and
//! a writer thread joined to the world by bounded channels, waking the loop
//! when a frame arrives (D25: blocking std threads, never Bevy's pools).

use crossbeam_channel::{Receiver, Sender, TrySendError};
use rig_ecs::tasks::Wake;
use rig_protocol::{Frame, FrameError, Listener, Peer, RequestId, Response, read_frame, write_frame};

/// Frames a connection may have waiting to be read by the world; past it
/// the connection's reader blocks, so only the flooding client waits.
pub const INBOX: usize = 256;
/// Frames a connection may have waiting to be written.
pub const OUTBOX: usize = 1024;
/// Out-queue room kept for responses: fan-out stops short of it.
pub const RESERVE: usize = 64;

/// What a connection's reader hands the world.
pub enum Inbound {
    Frame(Frame),
    /// The connection closed or failed.
    Closed,
}

/// A connection just accepted.
pub struct Accepted {
    pub out: Sender<Frame>,
    pub inbox: Receiver<Inbound>,
    pub peer: Peer,
}

/// Accepts connections until the listener fails, each handed to the world
/// through `accepted`.
pub fn accept(listener: Listener, accepted: Sender<Accepted>, wake: Wake) {
    std::thread::Builder::new()
        .name("rigd accept".into())
        .spawn(move || {
            loop {
                let (conn, peer) = match listener.accept() {
                    Ok(conn) => conn,
                    Err(e) => {
                        bevy_log::warn!("rigd stopped accepting: {e}");
                        return;
                    }
                };
                let Ok((reader, writer)) = conn.split() else {
                    continue;
                };
                let (out, outgoing) = crossbeam_channel::bounded(OUTBOX);
                let (inbound, inbox) = crossbeam_channel::bounded(INBOX);
                spawn_writer(writer, outgoing);
                spawn_reader(reader, inbound, out.clone(), wake.clone());
                if accepted.send(Accepted { out, inbox, peer }).is_err() {
                    return;
                }
                wake.wake();
            }
        })
        .ok();
}

fn spawn_writer(mut writer: impl std::io::Write + Send + 'static, outgoing: Receiver<Frame>) {
    std::thread::Builder::new()
        .name("rigd write".into())
        .spawn(move || {
            for frame in outgoing {
                if write_frame(&mut writer, &frame).is_err() {
                    return;
                }
            }
        })
        .ok();
}

fn spawn_reader(
    mut reader: impl std::io::BufRead + Send + 'static,
    inbound: Sender<Inbound>,
    out: Sender<Frame>,
    wake: Wake,
) {
    std::thread::Builder::new()
        .name("rigd read".into())
        .spawn(move || {
            let mut line = Vec::new();
            loop {
                let next = match read_frame(&mut reader, &mut line) {
                    Ok(Some(frame)) => Inbound::Frame(frame),
                    Ok(None) | Err(FrameError::Io(_)) => Inbound::Closed,
                    Err(e) => {
                        // A bad line is answered here and the connection goes on.
                        let id = match &e {
                            FrameError::Version { id: Some(id), .. } => id.clone(),
                            _ => RequestId::Num(0),
                        };
                        if let Some(error) = e.rpc() {
                            out.send(Frame::Response(Response::new(id, Err(error)))).ok();
                        }
                        continue;
                    }
                };
                let closed = matches!(next, Inbound::Closed);
                let sent = inbound.send(next).is_ok();
                wake.wake();
                if closed || !sent {
                    return;
                }
            }
        })
        .ok();
}

/// Queues `frame` without blocking; `false` when the queue is full or the
/// connection is gone.
pub fn offer(out: &Sender<Frame>, frame: Frame) -> bool {
    match out.try_send(frame) {
        Ok(()) => true,
        Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => false,
    }
}

/// Whether fan-out may queue another event on `out`.
pub fn has_room(out: &Sender<Frame>) -> bool {
    out.len() + RESERVE < OUTBOX
}
