//! Co-polled streaming replies written part by part, closed when the reply
//! ends.
//!
//! ```
//! use rig_core::serve::Reply;
//!
//! let reply = Reply::written(|mut writer| async move {
//!     let _ = writer.text("hello").await;
//! });
//! # let _ = reply;
//! ```

use futures::{SinkExt, StreamExt, channel::mpsc};

use crate::{
    error::{ErrorReport, ProviderError},
    message::{CallId, LocalCallId, ToolCall, ToolFunction, ToolName},
    operation::{Finish, Turn},
    streaming::{Item, Relayed, StreamEvent},
    wire::{Fold, Reply as WireReply},
};

use super::{Reply, SinkClosed};
use crate::wasm_compat::WasmCompatSend;

/// A streaming answer under construction, the bus's completion writer.
/// Obtained by [`Reply::written`]; [`finish`](Self::finish) ends the reply
/// with the response it folds into, while a writer dropped without it
/// leaves a truncated stream.
pub struct StreamWriter {
    events: mpsc::Sender<Result<Relayed, ErrorReport>>,
    turn: Turn,
    items: std::collections::VecDeque<Result<Item<StreamEvent>, ProviderError>>,
    /// The open text and reasoning parts bare fragments extend.
    text: Option<usize>,
    reasoning: Option<usize>,
    raw: serde_json::Value,
    request_id: Option<String>,
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
            turn: Turn::new(""),
            items: std::collections::VecDeque::new(),
            text: None,
            reasoning: None,
            raw: serde_json::Value::Null,
            request_id: None,
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
    /// A text fragment: extends the open text part, or opens one after
    /// closing an open reasoning part.
    pub async fn text(&mut self, text: impl Into<String>) -> Result<(), SinkClosed> {
        self.close_reasoning();
        let slot = match self.text {
            Some(slot) => slot,
            None => {
                let slot = self.turn.open_text();
                self.text = Some(slot);
                slot
            }
        };
        self.turn.write_text(&mut self.items, slot, &text.into());
        self.flush().await
    }

    /// A reasoning fragment: extends the open reasoning part, or opens one
    /// after closing an open text part.
    pub async fn reasoning(&mut self, text: impl Into<String>) -> Result<(), SinkClosed> {
        self.close_text();
        let slot = match self.reasoning {
            Some(slot) => slot,
            None => {
                let slot = self.turn.open_reasoning();
                self.reasoning = Some(slot);
                slot
            }
        };
        self.turn
            .write_reasoning(&mut self.items, slot, &text.into());
        self.flush().await
    }

    /// A whole tool call, under an id rig issues.
    pub async fn tool_call(
        &mut self,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Result<(), SinkClosed> {
        self.close_text();
        self.close_reasoning();
        let Ok(name) = ToolName::new(name) else {
            return self
                .error(ErrorReport::from(&ProviderError::Response(
                    "a tool call needs a name".to_owned(),
                )))
                .await;
        };
        let call = ToolCall {
            id: CallId::Local(LocalCallId::new()),
            function: ToolFunction { name, arguments },
            signature: None,
            additional_params: None,
            native: None,
        };
        if let Err(error) = self.turn.write_call(&mut self.items, call) {
            return self.error(ErrorReport::from(&error)).await;
        }
        self.flush().await
    }

    /// The reply's provider document, the response's `raw`.
    pub fn raw(&mut self, raw: serde_json::Value) {
        self.raw = raw;
    }

    /// The provider's transport request id, the response's
    /// `provider_request_id`: what a writer relaying a provider's reply
    /// reports in place of a transport. An empty id is no id.
    pub fn request_id(&mut self, request_id: impl Into<String>) {
        self.request_id = Some(request_id.into());
    }

    /// An in-band error: the consumer's last item.
    pub async fn error(&mut self, report: ErrorReport) -> Result<(), SinkClosed> {
        self.flush().await.map_err(|_| SinkClosed)?;
        self.events.send(Err(report)).await.map_err(|_| SinkClosed)
    }

    /// End the reply as `provider` ended it: closes the parts still open,
    /// then sends the response the reply folds into. The returned stream
    /// ends when the writing future also finishes.
    pub async fn finish(
        mut self,
        provider: impl Into<String>,
        finish: Finish,
    ) -> Result<(), SinkClosed> {
        self.close_text();
        self.close_reasoning();
        self.turn.close_open(&mut self.items);
        self.flush().await?;
        let reply = WireReply {
            provider: provider.into(),
            raw: std::mem::take(&mut self.raw),
            provider_request_id: self.request_id.take(),
        };
        let turn = std::mem::replace(&mut self.turn, Turn::new(""));
        let item = match turn.finish(finish, reply) {
            Ok(response) => Ok(Relayed::Done(Box::new(response))),
            Err(error) => Err(ErrorReport::from(&error)),
        };
        self.events.send(item).await.map_err(|_| SinkClosed)
    }

    /// Whether the consumer has closed the receiving side.
    pub fn is_closed(&self) -> bool {
        self.events.is_closed()
    }

    fn close_text(&mut self) {
        if let Some(slot) = self.text.take() {
            self.turn.end_text(&mut self.items, slot);
        }
    }

    fn close_reasoning(&mut self) {
        if let Some(slot) = self.reasoning.take() {
            self.turn.end_reasoning(&mut self.items, slot);
        }
    }

    /// Send what the writer emitted, folding each event as it leaves.
    async fn flush(&mut self) -> Result<(), SinkClosed> {
        while let Some(item) = self.items.pop_front() {
            let item = match item {
                Ok(item) => {
                    if let Item::Event(event) = &item {
                        let _ = self.turn.absorb(event);
                    }
                    Ok(Relayed::Item(item))
                }
                Err(error) => Err(ErrorReport::from(&error)),
            };
            self.events.send(item).await.map_err(|_| SinkClosed)?;
        }
        Ok(())
    }
}
