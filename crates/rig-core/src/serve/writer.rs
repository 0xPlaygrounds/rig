//! A co-polled stream writer with block identity and self-closing output.

use futures::{SinkExt, StreamExt, channel::mpsc};

use crate::{
    error::ErrorReport,
    providers::internal::adapter::AdapterOutput,
    streaming::{StreamEvent, StreamFinal, SyntheticIds, ToolCallEnd},
};

use super::{Reply, SinkClosed};
use crate::wasm_compat::WasmCompatSend;

/// A streaming answer under construction. Obtained by [`Reply::written`];
/// [`finish`](Self::finish) emits the terminal, while ordinary drop without
/// a terminal leaves a truncated stream.
pub struct StreamWriter {
    events: mpsc::Sender<Result<StreamEvent, ErrorReport>>,
    output: AdapterOutput,
    tool_ids: SyntheticIds,
}

impl Reply {
    /// Return a stream that owns and polls the writing future alongside its
    /// private receiver. No task is spawned. The bridge has zero shared
    /// capacity and one sender-reserved slot; it is not a rendezvous channel.
    /// Dropping the returned stream drops the writing future and receiver.
    pub fn written<F, Fut>(write: F) -> Self
    where
        F: FnOnce(StreamWriter) -> Fut,
        Fut: Future<Output = ()> + WasmCompatSend + 'static,
    {
        let (events, mut receiver) = mpsc::channel(0);
        let writer = StreamWriter {
            events,
            output: AdapterOutput::self_closing(),
            tool_ids: SyntheticIds::tool(),
        };
        let mut writing = Some(Box::pin(write(writer)));
        Self::Stream(Box::pin(futures::stream::poll_fn(move |cx| {
            if let Some(future) = &mut writing
                && future.as_mut().poll(cx).is_ready()
            {
                writing = None;
            }
            match receiver.poll_next_unpin(cx) {
                std::task::Poll::Ready(None) if writing.is_some() => std::task::Poll::Pending,
                next => next,
            }
        })))
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
        self.flush().await.map_err(|_| SinkClosed)?;
        self.events.send(Err(report)).await.map_err(|_| SinkClosed)
    }

    /// The terminal record: closes the blocks bare fragments opened, sends
    /// `record`. The returned stream ends when the writing future also finishes.
    pub async fn finish(mut self, record: StreamFinal) -> Result<(), SinkClosed> {
        self.output.close_active_blocks();
        self.output.final_record(record);
        self.flush().await
    }

    /// Whether the consumer is still listening.
    pub fn is_closed(&self) -> bool {
        self.events.is_closed()
    }

    async fn flush(&mut self) -> Result<(), SinkClosed> {
        for item in self.output.drain() {
            self.events
                .send(item.map_err(|error| ErrorReport::from(&error)))
                .await
                .map_err(|_| SinkClosed)?;
        }
        Ok(())
    }
}
