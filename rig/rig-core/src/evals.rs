//! Evals.
//! From OpenAI's evals repo:
//! > Evals provide a framework for evaluating large language models (LLMs) or systems built using LLMs. We offer an existing registry of evals to test different dimensions of OpenAI models and the ability to write your own custom evals for use cases you care about. You can also use your data to build private evals which represent the common LLMs patterns in your workflow without exposing any of that data publicly.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    completion::CompletionModel,
    embeddings::EmbeddingModel,
    extractor::{Extractor, ExtractorBuilder},
};

/// Evaluation errors.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// A mandatory field was null when attempting to initialise a struct
    #[error("Field must not be null: {0}")]
    FieldCannotBeNull(String),
    /// Generic eval module error
    #[error("Eval error: {0}")]
    Custom(String),
}

/// The outcome of an evaluation (ie, sending an input to an LLM which then gets tested against a set of criteria).
/// Invalid results due to things like functions returning errors should be encoded as invalid evaluation outcomes.
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

impl<Output> EvalOutcome<Output> {
    /// Check whether or not an evaluation has passed.
    pub fn is_pass(&self) -> bool {
        matches!(self, EvalOutcome::Pass(_))
    }

    /// Gets the score from an eval (assuming it isn't invalid).
    pub fn score(&self) -> Option<&Output> {
        match self {
            EvalOutcome::Pass(o) | EvalOutcome::Fail(o) => Some(o),
            EvalOutcome::Invalid(_) => None,
        }
    }
}

/// A trait to encode evaluators - types that can be used to test LLM outputs against criteria.
/// Evaluators come in all shapes and sizes, and additionally may themselves use LLMs (although there are many heuristics you can use that don't).
/// There are three possible states that an LLM can result in:
/// - Pass (the output passed all criteria)
/// - Fail (the output failed one or all criteria)
/// - Invalid (the output was unable to be retrieved due to an external failure like an API call fail)
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
        use futures::StreamExt;
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
/// In broad terms, cosine similarity can be used to measure how similar two documents are.
/// This can be useful for things like quickly testing semantic similarity between two documents.
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
    pub fn builder(embedding_model: E) -> SemanticSimilarityMetricBuilder<E> {
        SemanticSimilarityMetricBuilder::new(embedding_model)
    }

    pub fn reference_answer(&self) -> &str {
        &self.reference_answer
    }
}

/// A builder struct for [`SemanticSimilarityMetric`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SemanticSimilarityMetricBuilder<E> {
    embedding_model: E,
    threshold: Option<f64>,
    reference_answer: Option<String>,
}

impl<E> SemanticSimilarityMetricBuilder<E>
where
    E: EmbeddingModel,
{
    pub fn new(embedding_model: E) -> Self {
        Self {
            embedding_model,
            threshold: None,
            reference_answer: None,
        }
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn reference_answer(mut self, reference_answer: &str) -> Self {
        self.reference_answer = Some(reference_answer.to_string());
        self
    }

    pub async fn build(self) -> Result<SemanticSimilarityMetric<E>, EvalError> {
        let threshold = self
            .threshold
            .ok_or(EvalError::FieldCannotBeNull("threshold".into()))?;
        let reference_answer = self
            .reference_answer
            .ok_or(EvalError::FieldCannotBeNull("reference_answer".into()))?;
        let reference_answer_embedding = self
            .embedding_model
            .embed_text(&reference_answer)
            .await
            .map_err(|x| EvalError::Custom(x.to_string()))?
            .vec;

        let res = SemanticSimilarityMetric {
            embedding_model: self.embedding_model,
            threshold,
            reference_answer,
            reference_answer_embedding,
        };

        Ok(res)
    }
}

/// The scoring metric used for [`SemanticSimilarityMetric`].
#[derive(Deserialize, Serialize, Clone, Debug)]
#[non_exhaustive]
pub struct SemanticSimilarityMetricScore {
    pub score: f64,
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

/// An LLM as a judge that judges an output by a given schema (and outputs the schema).
/// The schema type uses the `Judgment` trait, which simply enforces a single function that checks whether it passes or not.
pub struct LlmJudgeMetric<M, T>
where
    M: CompletionModel,
    T: Judgment + Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a>,
{
    ext: Extractor<M, T>,
}

/// An LLM as a judge that judges an output by a given schema (and outputs the schema).
/// Unlike `LlmJudgeMetric`, this type uses a function pointer that takes the type and returns a `bool` instead.
pub struct LlmJudgeMetricWithFn<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a>,
{
    ext: Extractor<M, T>,
    evaluator: Box<dyn Fn(&T) -> bool + Send + Sync>,
}

pub struct LlmJudgeBuilder<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a> + 'static,
{
    ext: ExtractorBuilder<M, T>,
}

pub struct LlmJudgeBuilderWithFn<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a> + 'static,
{
    ext: ExtractorBuilder<M, T>,
    evaluator: Box<dyn Fn(&T) -> bool + Send + Sync>,
}

impl<M, T> LlmJudgeBuilder<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a>,
{
    pub fn new(ext: ExtractorBuilder<M, T>) -> Self {
        Self { ext }
    }

    pub fn with_fn<F>(self, f: F) -> LlmJudgeBuilderWithFn<M, T>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        LlmJudgeBuilderWithFn {
            ext: self.ext,
            evaluator: Box::new(f),
        }
    }

    pub fn build(self) -> LlmJudgeMetric<M, T>
    where
        T: Judgment + 'static,
    {
        let ext = self
            .ext
            .preamble(
                "Judge the prompt input by the schema given and return it as a JSON tool result",
            )
            .build();
        LlmJudgeMetric { ext }
    }
}

impl<M, T> LlmJudgeBuilderWithFn<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a> + 'static,
{
    pub fn with_fn<F2>(mut self, f: F2) -> Self
    where
        F2: Fn(&T) -> bool + Send + Sync + 'static,
    {
        self.evaluator = Box::new(f);
        self
    }

    pub fn build(self) -> LlmJudgeMetricWithFn<M, T> {
        let ext = self
            .ext
            .preamble(
                "Judge the prompt input by the schema given and return it as a JSON tool result",
            )
            .build();
        LlmJudgeMetricWithFn {
            ext,
            evaluator: self.evaluator,
        }
    }
}

/// A helper trait for `LlmJudgeMetric`.
/// Types that implement `Judgment` generally have a very standard way of either passing or failing.
/// As such, this can be enforced as a trait.
pub trait Judgment {
    fn passes(&self) -> bool;
}

impl<M, T> Eval<T> for LlmJudgeMetric<M, T>
where
    M: CompletionModel + 'static,
    T: Judgment + Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a> + Clone + 'static,
{
    async fn eval(&self, input: String) -> EvalOutcome<T> {
        match self.ext.extract(input).await {
            Ok(judgment) => {
                if judgment.passes() {
                    EvalOutcome::Pass(judgment)
                } else {
                    EvalOutcome::Fail(judgment)
                }
            }
            Err(e) => EvalOutcome::Invalid(e.to_string()),
        }
    }
}

impl<M, T> Eval<T> for LlmJudgeMetricWithFn<M, T>
where
    M: CompletionModel + 'static,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a> + Clone + 'static,
{
    async fn eval(&self, input: String) -> EvalOutcome<T> {
        match self.ext.extract(input).await {
            Ok(judgment) => {
                if (self.evaluator)(&judgment) {
                    EvalOutcome::Pass(judgment)
                } else {
                    EvalOutcome::Fail(judgment)
                }
            }
            Err(e) => EvalOutcome::Invalid(e.to_string()),
        }
    }
}

impl<M, T> From<ExtractorBuilder<M, T>> for LlmJudgeBuilder<M, T>
where
    M: CompletionModel,
    T: Send + Sync + JsonSchema + Serialize + for<'a> Deserialize<'a>,
{
    fn from(ext: ExtractorBuilder<M, T>) -> Self {
        Self::new(ext)
    }
}

/// An eval that scores an output based on some given criteria.
#[non_exhaustive]
pub struct LlmScoreMetric<M>
where
    M: CompletionModel,
{
    agent: Extractor<M, LlmScoreMetricScore>,
    threshold: f64,
}

/// The scoring output returned by `LlmScoreMetric`.
/// Must also be used as the Extractor return type when passed into `LlmScoreMetric`.
#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
pub struct LlmScoreMetricScore {
    /// A score between 0.0 and 1.0 inclusive.
    pub score: f64,
    /// Feedback on a given input in relation to the required criteria to be met.
    pub feedback: String,
}

impl<M> Eval<LlmScoreMetricScore> for LlmScoreMetric<M>
where
    M: CompletionModel + 'static,
{
    async fn eval(&self, input: String) -> EvalOutcome<LlmScoreMetricScore> {
        let res = match self.agent.extract(input).await {
            Ok(res) => res,
            Err(e) => return EvalOutcome::Invalid(e.to_string()),
        };

        if !(0.0..=1.0).contains(&res.score) {
            return EvalOutcome::Invalid(format!(
                "Score {} outside valid range [0.0, 1.0]",
                res.score
            ));
        }

        if res.score >= self.threshold {
            EvalOutcome::Pass(res)
        } else {
            EvalOutcome::Fail(res)
        }
    }
}

#[non_exhaustive]
pub struct LlmScoreMetricBuilder<M>
where
    M: CompletionModel,
{
    agent: ExtractorBuilder<M, LlmScoreMetricScore>,
    criteria: Vec<String>,
    threshold: Option<f64>,
}

impl<M> LlmScoreMetricBuilder<M>
where
    M: CompletionModel,
{
    pub fn new(agent: ExtractorBuilder<M, LlmScoreMetricScore>) -> Self {
        Self {
            agent,
            criteria: Vec::new(),
            threshold: None,
        }
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn criteria(mut self, criteria: &str) -> Self {
        self.criteria.push(criteria.to_string());
        self
    }

    pub fn build(self) -> Result<LlmScoreMetric<M>, EvalError> {
        let threshold = self
            .threshold
            .ok_or(EvalError::FieldCannotBeNull("threshold".into()))?;
        let preamble = format!(
            "You are an evaluation model. Score the input based on these criteria:\n{}\n\n\
            Provide a score between 0.0 and 1.0 (where 1.0 is best) and explain your reasoning.",
            self.criteria.join("\n")
        );

        let agent = self.agent.preamble(&preamble).build();

        Ok(LlmScoreMetric { agent, threshold })
    }
}
