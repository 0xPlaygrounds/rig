//! Evals.
//! From OpenAI's evals repo:
//! > Evals provide a framework for evaluating large language models (LLMs) or systems built using LLMs. We offer an existing registry of evals to test different dimensions of OpenAI models and the ability to write your own custom evals for use cases you care about. You can also use your data to build private evals which represent the common LLMs patterns in your workflow without exposing any of that data publicly.

use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    completion::CompletionModel,
    embeddings::{EmbeddingError, EmbeddingModel},
    extractor::Extractor,
};

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "outcome", content = "data")]
pub enum EvalOutcome<Output> {
    /// Evaluation passed
    Pass(Output),
    /// Evaluation failed
    Fail(Output),
    /// Evaluation was invalidated (reason in field)
    Invalid(String),
}

pub trait Eval<Output>
where
    Output: for<'a> Deserialize<'a> + Serialize + Clone + Send + Sync,
    Self: Sized + Send + Sync + 'static,
{
    fn eval(&self, input: String) -> impl Future<Output = EvalOutcome<Output>> + Send;

    /// Send a bunch of inputs to be evaluated all in one call.
    /// You can set the concurrency limit to help alleviate issues
    /// with model provider API limits, as sending requests too quickly may
    /// result in throttling or temporary request refusal.
    fn eval_batch(
        &self,
        input: Vec<String>,
        concurrency_limit: usize,
    ) -> impl Future<Output = Vec<EvalOutcome<Output>>> + Send {
        async move {
            let thing: Vec<EvalOutcome<Output>> = futures::stream::iter(input)
                .map(|x| Self::eval(self, x))
                .buffered(concurrency_limit)
                .collect()
                .await;

            thing
        }
    }
}

/// A semantic similarity metric. Uses cosine similarity.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SemanticSimilarityMetric<E> {
    embedding_model: E,
    threshold: f64,
    reference_answer: String,
    reference_answer_embedding: Vec<f64>,
}

impl<E> SemanticSimilarityMetric<E>
where
    E: EmbeddingModel,
{
    pub fn reference_answer(&self) -> String {
        self.reference_answer.clone()
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[non_exhaustive]
pub struct SemanticSimilarityMetricScore {
    pub score: f64,
}

impl SemanticSimilarityMetricScore {
    pub fn new(score: f64) -> Self {
        Self { score }
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

impl<E> SemanticSimilarityMetric<E>
where
    E: EmbeddingModel,
{
    pub async fn new(
        embedding_model: E,
        reference_answer: String,
        threshold: f64,
    ) -> Result<Self, EmbeddingError> {
        let reference_answer_embedding = embedding_model.embed_text(&reference_answer).await?.vec;

        let res = Self {
            embedding_model,
            threshold,
            reference_answer,
            reference_answer_embedding,
        };

        Ok(res)
    }
}

impl<E> Eval<SemanticSimilarityMetricScore> for SemanticSimilarityMetric<E>
where
    E: EmbeddingModel + 'static,
{
    async fn eval(&self, input: String) -> EvalOutcome<SemanticSimilarityMetricScore> {
        let input = match self.embedding_model.embed_text(&input).await {
            Ok(res) => res.vec,
            Err(e) => return EvalOutcome::Invalid(e.to_string()),
        };
        let ref_answer = &self.reference_answer_embedding;

        let dot: f64 = input.iter().zip(ref_answer).map(|(x, y)| x * y).sum();
        let norm_a = input.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b = ref_answer.iter().map(|x| x * x).sum::<f64>().sqrt();

        let cosine_sim = dot / (norm_a * norm_b);

        if cosine_sim >= self.threshold {
            EvalOutcome::Pass(SemanticSimilarityMetricScore { score: cosine_sim })
        } else {
            EvalOutcome::Fail(SemanticSimilarityMetricScore { score: cosine_sim })
        }
    }
}

pub struct LlmJudgeMetric<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a>,
{
    ext: Extractor<M, T>,
    criteria: String,
}

impl<M, T> From<Extractor<M, T>> for LlmJudgeMetric<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a>,
{
    fn from(ext: Extractor<M, T>) -> Self {
        Self {
            ext,
            criteria: String::new(),
        }
    }
}

impl<M, T> LlmJudgeMetric<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a>,
{
    pub fn new(ext: Extractor<M, T>, criteria: &str) -> Self {
        Self {
            ext,
            criteria: criteria.to_string(),
        }
    }
}
