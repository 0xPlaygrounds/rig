//! The ACP lines of this process's stdin and stdout, bounded both ways.

use std::io::Read;

use futures::StreamExt;
use rig_harness::host::pipes::{self, Stream, Writer};

/// ACP lines read from `read`, this process's stdin, at most `limit` a line
/// ([`crate::LINE_LIMIT`]), and written by `writer`, its stdout, which never
/// waits. Lines are read whatever the writer's backlog, so a cancel or the
/// end of the input always comes through; a peer that asks and never reads
/// is gone once the backlog passes its limit ([`Writer::gone`]). A line past
/// the limit, or one that is not UTF-8, is said on stderr, by `who`, and
/// ends the input as a closed stdin does, so the requests still open are
/// answered.
pub(crate) fn stdio(
    read: impl Read + Send + 'static,
    writer: Writer,
    limit: usize,
    who: &'static str,
) -> agent_client_protocol::Lines<
    impl futures::Sink<String, Error = std::io::Error> + Send + 'static,
    impl futures::Stream<Item = std::io::Result<String>> + Send + 'static,
> {
    let incoming = incoming(read, writer.clone(), limit, who);
    agent_client_protocol::Lines::new(outgoing(writer), incoming.boxed())
}

/// The lines [`stdio`] writes.
fn outgoing(
    writer: Writer,
) -> std::pin::Pin<Box<impl futures::Sink<String, Error = std::io::Error> + Send + 'static>> {
    Box::pin(futures::sink::unfold(
        writer,
        async |writer, line: String| {
            writer.write(Stream::Out, format!("{line}\n").into_bytes());
            Ok::<_, std::io::Error>(writer)
        },
    ))
}

/// The lines [`stdio`] reads.
fn incoming(
    read: impl Read + Send + 'static,
    writer: Writer,
    limit: usize,
    who: &'static str,
) -> impl futures::Stream<Item = std::io::Result<String>> + Send + 'static {
    let read = pipes::read_lines(read, limit);
    futures::stream::unfold((read, writer), move |(read, writer)| async move {
        let line = read.recv().await.ok()?;
        match accepted(line) {
            Ok(line) => Some((Ok(line), (read, writer))),
            Err(why) => {
                let said = format!("{who}: {why} came on stdin; reading ends\n");
                writer.write(Stream::Err, said.into_bytes());
                None
            }
        }
    })
}

/// Why a line is refused.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Refused {
    #[error("a line of {0}")]
    TooLarge(pipes::TooLarge),
    #[error("a line that is not UTF-8")]
    NotUtf8,
}

/// A line as ACP text, or why it is refused: past the limit, or not UTF-8.
pub(crate) fn accepted(line: Result<Vec<u8>, pipes::TooLarge>) -> Result<String, Refused> {
    let line = line.map_err(Refused::TooLarge)?;
    String::from_utf8(line).map_err(|_| Refused::NotUtf8)
}

#[cfg(test)]
mod tests;
