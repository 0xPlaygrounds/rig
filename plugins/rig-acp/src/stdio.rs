//! The ACP lines of this process's stdin and stdout, bounded both ways.

use futures::StreamExt;
use rig_harness::host::pipes::{self, Stream, Writer};

use crate::LINE_LIMIT;

/// ACP lines on this process's stdin, read at most [`LINE_LIMIT`] a
/// line, and its stdout, written by `writer`, which never waits. A line
/// past the limit is said on stderr, by `who`, and ends the input as a
/// closed stdin does, so the requests still open are answered.
pub(crate) fn stdio(
    writer: Writer,
    who: &'static str,
) -> agent_client_protocol::Lines<
    impl futures::Sink<String, Error = std::io::Error> + Send + 'static,
    impl futures::Stream<Item = std::io::Result<String>> + Send + 'static,
> {
    let sink = futures::sink::unfold(writer, async |writer, line: String| {
        writer.write(Stream::Out, format!("{line}\n").into_bytes());
        Ok::<_, std::io::Error>(writer)
    });
    let read = pipes::read_lines(std::io::stdin(), LINE_LIMIT);
    let lines = read.map(move |line| match line {
        Ok(line) => Some(Ok(String::from_utf8_lossy(&line).into_owned())),
        Err(refused) => {
            eprintln!("{who}: a line of {refused} came on stdin; reading ends");
            None
        }
    });
    let lines = lines.take_while(|line| std::future::ready(line.is_some()));
    agent_client_protocol::Lines::new(Box::pin(sink), lines.filter_map(std::future::ready).boxed())
}
