//! The [`Batching`] implementations of the OpenAI, Anthropic and Gemini
//! completion wires: each step one call through the driver, over the
//! provider's batch wire, which encodes and decodes it.

use crate::completion::batch::{
    BatchOutcome, BatchPrice, BatchState, Batching, EncodedItem, ItemReply,
};
use crate::driver::{Model, Transport};
use crate::error::ProviderError;
use crate::providers::anthropic::Messages;
use crate::providers::anthropic::batch as anthropic;
use crate::providers::gemini::batch as gemini;
use crate::providers::gemini::completion::GenerateContent;
use crate::providers::openai::batch as openai;
use crate::providers::openai::responses_api::wire::Responses;
use crate::providers::openai::wire::{Chat, OpenAIConfig, OpenAiWire};
use crate::wasm_compat::WasmCompatSend;

impl Batching for Messages {
    type Batches = anthropic::MessageBatches;

    /// Batch pricing stacks with prompt-caching pricing.
    const PRICE: BatchPrice = BatchPrice::HALF;

    fn batches(&self) -> Result<anthropic::MessageBatches, ProviderError> {
        if self.provider.dialect.name != crate::providers::anthropic::PROVIDER_NAME {
            return Err(ProviderError::request(format!(
                "{} has no Message Batches API",
                self.provider.dialect.name
            )));
        }
        Ok(anthropic::MessageBatches {
            provider: self.provider.clone(),
        })
    }

    fn submit<T: Transport<anthropic::MessageBatches>>(
        batches: &Model<anthropic::MessageBatches, T>,
        items: Vec<EncodedItem>,
    ) -> impl Future<Output = Result<String, ProviderError>> + WasmCompatSend {
        let request = anthropic::create_request(items);
        async move {
            let state = batches.call(request?).await?.batch()?;
            Ok(state.id)
        }
    }

    fn status<T: Transport<anthropic::MessageBatches>>(
        batches: &Model<anthropic::MessageBatches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
        let reply = batches.call(anthropic::MessageBatchRequest::Get(id.to_owned()));
        async move { reply.await?.batch() }
    }

    fn results<T: Transport<anthropic::MessageBatches>>(
        batches: &Model<anthropic::MessageBatches, T>,
        id: &str,
    ) -> impl Future<Output = Result<Vec<(String, BatchOutcome<ItemReply>)>, ProviderError>>
    + WasmCompatSend {
        let reply = batches.call(anthropic::MessageBatchRequest::Results(id.to_owned()));
        async move {
            match reply.await? {
                anthropic::MessageBatchReply::Results(lines) => Ok(lines
                    .into_iter()
                    .map(|line| (line.custom_id, line.result.into()))
                    .collect()),
                anthropic::MessageBatchReply::Batch(..) => Err(ProviderError::Response(
                    "Anthropic answered with a message batch, not its results".to_owned(),
                )),
            }
        }
    }

    fn cancel<T: Transport<anthropic::MessageBatches>>(
        batches: &Model<anthropic::MessageBatches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
        let reply = batches.call(anthropic::MessageBatchRequest::Cancel(id.to_owned()));
        async move { reply.await?.batch() }
    }
}

async fn openai_submit<T: Transport<openai::Batches>>(
    batches: &Model<openai::Batches, T>,
    items: Vec<EncodedItem>,
) -> Result<String, ProviderError> {
    let (lines, endpoint) = openai::input_file(items)?;
    let input_file_id = match batches.call(openai::BatchRequest::Upload(lines)).await? {
        openai::BatchReply::Object(openai::Object::File { id }, _) => id,
        _ => return Err(openai::mismatch("a file")),
    };
    let request = openai::BatchRequest::Create {
        input_file_id,
        endpoint,
    };
    let (batch, _) = batches.call(request).await?.batch()?;
    Ok(batch.id)
}

async fn openai_status<T: Transport<openai::Batches>>(
    batches: &Model<openai::Batches, T>,
    id: &str,
) -> Result<BatchState, ProviderError> {
    let (batch, raw) = batches
        .call(openai::BatchRequest::Get(id.to_owned()))
        .await?
        .batch()?;
    Ok(batch.state(raw))
}

/// The lines of an OpenAI batch's output and error files.
async fn openai_results<T: Transport<openai::Batches>>(
    batches: &Model<openai::Batches, T>,
    id: &str,
) -> Result<Vec<(String, BatchOutcome<ItemReply>)>, ProviderError> {
    let (batch, raw) = batches
        .call(openai::BatchRequest::Get(id.to_owned()))
        .await?
        .batch()?;
    let state = batch.state(raw);
    if !state.status.is_finished() {
        return Err(ProviderError::Response(format!(
            "OpenAI batch `{id}` is {:?}: its results are not ready",
            batch.status
        )));
    }
    let mut outcomes = Vec::new();
    for file in [batch.output_file_id, batch.error_file_id]
        .into_iter()
        .flatten()
    {
        match batches
            .call(openai::BatchRequest::FileContent(file))
            .await?
        {
            openai::BatchReply::Lines(lines) => outcomes.extend(
                lines
                    .into_iter()
                    .map(|line| (line.custom_id.clone(), line.into())),
            ),
            openai::BatchReply::Object(..) => return Err(openai::mismatch("a results file")),
        }
    }
    Ok(outcomes)
}

async fn openai_cancel<T: Transport<openai::Batches>>(
    batches: &Model<openai::Batches, T>,
    id: &str,
) -> Result<BatchState, ProviderError> {
    let (batch, raw) = batches
        .call(openai::BatchRequest::Cancel(id.to_owned()))
        .await?
        .batch()?;
    Ok(batch.state(raw))
}

/// The configuration an OpenAI completion wire sends with.
trait Configured {
    fn config(&self) -> &OpenAIConfig;
}

impl Configured for Chat {
    fn config(&self) -> &OpenAIConfig {
        &self.provider
    }
}

impl Configured for Responses {
    fn config(&self) -> &OpenAIConfig {
        &self.provider
    }
}

impl Configured for OpenAiWire {
    fn config(&self) -> &OpenAIConfig {
        match self {
            Self::Chat(wire) => &wire.provider,
            Self::Responses(wire) => &wire.provider,
        }
    }
}

/// Implement [`Batching`] for an OpenAI completion wire.
macro_rules! openai_batching {
    ($wire:ty) => {
        impl Batching for $wire {
            type Batches = openai::Batches;

            const PRICE: BatchPrice = BatchPrice::HALF;

            fn batches(&self) -> Result<openai::Batches, ProviderError> {
                openai::batches(self.config())
            }

            fn submit<T: Transport<openai::Batches>>(
                batches: &Model<openai::Batches, T>,
                items: Vec<EncodedItem>,
            ) -> impl Future<Output = Result<String, ProviderError>> + WasmCompatSend {
                openai_submit(batches, items)
            }

            fn status<T: Transport<openai::Batches>>(
                batches: &Model<openai::Batches, T>,
                id: &str,
            ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
                openai_status(batches, id)
            }

            fn results<T: Transport<openai::Batches>>(
                batches: &Model<openai::Batches, T>,
                id: &str,
            ) -> impl Future<Output = Result<Vec<(String, BatchOutcome<ItemReply>)>, ProviderError>>
            + WasmCompatSend {
                openai_results(batches, id)
            }

            fn cancel<T: Transport<openai::Batches>>(
                batches: &Model<openai::Batches, T>,
                id: &str,
            ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
                openai_cancel(batches, id)
            }
        }
    };
}

openai_batching!(Chat);
openai_batching!(Responses);
openai_batching!(OpenAiWire);

impl Batching for GenerateContent {
    type Batches = gemini::Batches;

    /// Cached tokens keep their standard price.
    const PRICE: BatchPrice = BatchPrice {
        input: 0.5,
        output: 0.5,
        cache_read: 1.0,
        cache_write: 1.0,
    };

    fn batches(&self) -> Result<gemini::Batches, ProviderError> {
        Ok(gemini::Batches {
            provider: self.provider.clone(),
        })
    }

    fn submit<T: Transport<gemini::Batches>>(
        batches: &Model<gemini::Batches, T>,
        items: Vec<EncodedItem>,
    ) -> impl Future<Output = Result<String, ProviderError>> + WasmCompatSend {
        let request = gemini::create_request(items);
        async move {
            let (operation, _) = batches.call(request?).await?.operation()?;
            Ok(operation.name)
        }
    }

    fn status<T: Transport<gemini::Batches>>(
        batches: &Model<gemini::Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
        let reply = batches.call(gemini::BatchRequest::Get(id.to_owned()));
        async move {
            let (operation, raw) = reply.await?.operation()?;
            operation.state(raw)
        }
    }

    fn results<T: Transport<gemini::Batches>>(
        batches: &Model<gemini::Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<Vec<(String, BatchOutcome<ItemReply>)>, ProviderError>>
    + WasmCompatSend {
        let reply = batches.call(gemini::BatchRequest::Get(id.to_owned()));
        let id = id.to_owned();
        async move {
            let (operation, raw) = reply.await?.operation()?;
            let state = operation.state(raw)?;
            if !state.status.is_finished() {
                return Err(ProviderError::Response(format!(
                    "Gemini batch `{id}` is {:?}: its results are not ready",
                    state.status
                )));
            }
            operation.inlined().iter().map(gemini::outcome).collect()
        }
    }

    /// Gemini answers a cancellation with nothing, so the batch is read
    /// back after it.
    fn cancel<T: Transport<gemini::Batches>>(
        batches: &Model<gemini::Batches, T>,
        id: &str,
    ) -> impl Future<Output = Result<BatchState, ProviderError>> + WasmCompatSend {
        let cancel = batches.call(gemini::BatchRequest::Cancel(id.to_owned()));
        let get = batches.call(gemini::BatchRequest::Get(id.to_owned()));
        async move {
            cancel.await?;
            let (operation, raw) = get.await?.operation()?;
            operation.state(raw)
        }
    }
}
