//! Experimental typed Jev evaluations alongside Rig agents.
//!
//! Bind [`Jev`] to any Rig HTTP transport, then use [`Evaluate::evaluate`]
//! with serializable state and a named [`Query`]. Its associated output type
//! fixes the answer structure at compile time. Decisions
//! retain distributions; routing and threshold policy remain in application code.
//! Reported probabilities are preserved, including bounded hundredth-rounding
//! error observed in live responses; they are not silently normalized.

mod decode;
mod dynamic;
mod questions;
pub mod types;
mod validation;
mod wire;

pub use dynamic::DynamicQuery;
pub use questions::{
    Choice, ChoiceAnswer, DynamicScore, DynamicScoreAnswer, JoinedQuery, MappedQuery, NamedQuery,
    Noul, NoulAnswer, Query, Score, ScoreAnswer,
};
use rig_core::error::ProviderError;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use rig_core::{
    driver::{Bound, call},
    http_client::HttpClientExt,
};
use serde::Serialize;
pub use wire::{Evaluation, Jev};

/// Typed answers together with provider metadata.
#[derive(Debug, Clone)]
pub struct EvaluationResult<A> {
    /// Answers with the structure determined by the question set.
    pub answers: A,
    /// The model identifier returned by the provider.
    pub model: String,
    /// Token accounting when reported by the provider.
    pub usage: Option<types::Usage>,
    /// Transport request identifier for diagnostics.
    pub provider_request_id: Option<String>,
}

/// Typed evaluation on Jev bound to a Rig transport.
pub trait Evaluate {
    /// Evaluate a shared state in one call and validate the typed answers.
    /// The input question type determines the successful response type:
    ///
    /// ```compile_fail
    /// use rig_typesafeai::{Evaluate, EvaluationResult, DynamicScoreAnswer, Noul, Query};
    /// async fn wrong(client: impl Evaluate) -> Result<(), rig_core::error::ProviderError> {
    ///     let question = Noul::new("Ready?")?.named("ready")?;
    ///     let result: EvaluationResult<DynamicScoreAnswer> = client.evaluate(&"state", question).await?;
    ///     Ok(())
    /// }
    /// ```
    fn evaluate<S, Q>(
        &self,
        state: &S,
        questions: Q,
    ) -> impl std::future::Future<Output = Result<EvaluationResult<Q::Output>, ProviderError>>
    + WasmCompatSend
    where
        S: Serialize + WasmCompatSync,
        Q: Query;
}
impl<H: HttpClientExt> Evaluate for Bound<Jev, H> {
    async fn evaluate<S, Q>(
        &self,
        state: &S,
        questions: Q,
    ) -> Result<EvaluationResult<Q::Output>, ProviderError>
    where
        S: Serialize + WasmCompatSync,
        Q: Query,
    {
        let encoded = serde_json::value::to_raw_value(&questions)?;
        let ids = validation::request_ids(&encoded)?;
        let response = call(
            &self.wire,
            &self.http,
            types::Request {
                state: questions::state(state)?,
                questions: encoded,
            },
            None,
        )
        .await?;
        if ids != validation::response_ids(&response.answers)? {
            return Err(ProviderError::Response(
                "response question IDs differ from request".into(),
            ));
        }
        Ok(EvaluationResult {
            answers: questions.decode(serde_json::from_str(response.answers.get())?)?,
            model: response.model,
            usage: response.usage,
            provider_request_id: response.provider_request_id,
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod query_tests;
