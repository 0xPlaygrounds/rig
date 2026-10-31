//! The ACP lines of this process's stdin and stdout, bounded both ways.

use futures::StreamExt;
use rig_harness::host::pipes::{self, Stream, Writer};

/// ACP lines on this process's stdin, read at most `limit` a line
/// ([`crate::LINE_LIMIT`]), and its stdout, written by `writer`, which
/// never waits. The next line is read only while the writer keeps up, so
/// a peer that does not read what it asked for cannot grow what waits. A
/// line past the limit is said on stderr, by `who`, and ends the input as
/// a closed stdin does, so the requests still open are answered.
pub(crate) fn stdio(
    writer: Writer,
    limit: usize,
    who: &'static str,
) -> agent_client_protocol::Lines<
    impl futures::Sink<String, Error = std::io::Error> + Send + 'static,
    impl futures::Stream<Item = std::io::Result<String>> + Send + 'static,
> {
    let sink = futures::sink::unfold(writer.clone(), async |writer, line: String| {
        writer.write(Stream::Out, format!("{line}\n").into_bytes());
        Ok::<_, std::io::Error>(writer)
    });
    let read = pipes::read_lines(std::io::stdin(), limit);
    let lines = futures::stream::unfold((read, writer), move |(read, writer)| async move {
        writer.room().await;
        match read.recv().await.ok()? {
            Ok(line) => {
                let line = String::from_utf8_lossy(&line).into_owned();
                Some((Ok(line), (read, writer)))
            }
            Err(refused) => {
                let said = format!("{who}: a line of {refused} came on stdin; reading ends\n");
                writer.write(Stream::Err, said.into_bytes());
                None
            }
        }
    });
    agent_client_protocol::Lines::new(Box::pin(sink), lines.boxed())
}
