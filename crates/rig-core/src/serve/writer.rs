//! A stream writer over an [`OutcomeSink`]: a handler says what it means —
//! text, reasoning, a tool call, the terminal record — and never names a
//! [`BlockId`](crate::streaming::BlockId). Block identity is minted per
//! stream by the same rules the provider adapters use (`AdapterOutput`), so
//! a bus handler's stream is well formed by construction.

use crate::{
    error::ErrorReport,
    providers::internal::adapter::AdapterOutput,
    streaming::{StreamEvent, StreamFinal, SyntheticIds, ToolCallEnd},
};

use super::{OutcomeSink, SinkClosed};

/// A streaming answer under construction: what a handler writes into its
/// [`OutcomeSink`] for a streaming dispatch. Obtained with
/// [`OutcomeSink::writer`]; ended with [`finish`](Self::finish).
///
/// ```ignore
/// async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
///     let mut out = sink.writer();
///     let _ = out.text("tick ").await;
///     let _ = out.finish(StreamFinal::new("mock", Usage::new())).await;
/// }
/// ```
pub struct StreamWriter {
    sink: OutcomeSink,
    output: AdapterOutput,
    tool_ids: SyntheticIds,
}

impl OutcomeSink {
    /// Answer this dispatch as a stream through a writer that mints block
    /// ids itself. Consumes the sink: a stream is answered through the
    /// writer or not at all.
    pub fn writer(self) -> StreamWriter {
        StreamWriter {
            sink: self,
            output: AdapterOutput::self_closing(),
            tool_ids: SyntheticIds::tool(),
        }
    }
}

impl StreamWriter {
    /// A text fragment: extends the open text block, or opens one.
    pub async fn text(&mut self, text: impl Into<String>) -> Result<(), SinkClosed> {
        self.output.text(text);
        self.flush().await
    }

    /// A reasoning fragment: extends the open reasoning block, or opens one
    /// (closing an open text block — reasoning and text never interleave in
    /// one block).
    pub async fn reasoning(&mut self, text: impl Into<String>) -> Result<(), SinkClosed> {
        self.output.reasoning(text);
        self.flush().await
    }

    /// A whole tool call, under a minted id.
    pub async fn tool_call(
        &mut self,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Result<(), SinkClosed> {
        let id = self.tool_ids.mint();
        self.output
            .tool_call(id, ToolCallEnd::whole(name, arguments));
        self.flush().await
    }

    /// An event the writer did not build (a provider's own record, a
    /// message id): passed through with the writer's block bookkeeping, so
    /// later bare text opens a fresh block after it.
    pub async fn event(&mut self, event: StreamEvent) -> Result<(), SinkClosed> {
        self.output.push(Ok(event));
        self.flush().await
    }

    /// An in-band error: the consumer's next item.
    pub async fn error(&mut self, report: ErrorReport) -> Result<(), SinkClosed> {
        self.flush().await?;
        self.sink.send(Err(report)).await
    }

    /// The terminal record: closes the blocks bare fragments opened, sends
    /// `record`, and ends the stream.
    pub async fn finish(mut self, record: StreamFinal) -> Result<(), SinkClosed> {
        self.output.close_active_blocks();
        self.output.final_record(record);
        self.flush().await
    }

    /// Whether the consumer is still listening.
    pub fn is_closed(&self) -> bool {
        self.sink.is_closed()
    }

    async fn flush(&mut self) -> Result<(), SinkClosed> {
        let items: Vec<_> = self.output.drain().collect();
        for item in items {
            self.sink
                .send(item.map_err(|error| ErrorReport::from(&error)))
                .await?;
        }
        Ok(())
    }
}
