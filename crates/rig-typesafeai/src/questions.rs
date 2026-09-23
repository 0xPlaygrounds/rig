//! Typed definitions bind each question to its answer type.
use crate::types::{Answer, Question};
use rig_core::error::ProviderError;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::{collections::BTreeMap, marker::PhantomData};

/// A selected application value and the full distribution over alternatives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceAnswer<T: Ord> {
    /// The selected application value.
    pub choice: T,
    /// The full distribution over the supplied alternatives or levels.
    pub probabilities: BTreeMap<T, f64>,
    /// Provider-reported concentration, not a probability of correctness.
    pub confidence: f64,
}
/// Expected zero-based rubric position and the distribution that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicScoreAnswer {
    /// Probability-weighted mean of zero-based rubric indices.
    pub score: f64,
    /// The full distribution over the supplied alternatives or levels.
    pub probabilities: BTreeMap<usize, f64>,
    /// Level indices mapped to their original descriptions.
    pub legend: BTreeMap<usize, Value>,
    /// Provider-reported distribution concentration.
    pub confidence: f64,
}
/// Probability of yes. Thresholds belong to the application's policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulAnswer {
    /// Probability that the answer is yes.
    pub noul: f64,
}

/// A choice question whose successful answer contains `T`.
#[derive(Debug)]
pub struct Choice<T> {
    instructions: Value,
    criteria: BTreeMap<String, Option<Value>>,
    marker: PhantomData<fn() -> T>,
}

/// A score question with a statically known ordered rubric.
#[derive(Debug)]
pub struct Score<L> {
    instructions: Value,
    criteria: Vec<Value>,
    levels: Vec<L>,
}

/// A score question whose rubric is supplied at runtime.
#[derive(Debug, Clone)]
pub struct DynamicScore {
    instructions: Value,
    criteria: Vec<Value>,
}

/// A yes/no question whose successful answer is `NoulAnswer`.
#[derive(Debug, Clone)]
pub struct Noul {
    instructions: Value,
    criteria: Option<BTreeMap<String, Value>>,
}

impl<T> Clone for Choice<T> {
    fn clone(&self) -> Self {
        Self {
            instructions: self.instructions.clone(),
            criteria: self.criteria.clone(),
            marker: PhantomData,
        }
    }
}
impl<L: Clone> Clone for Score<L> {
    fn clone(&self) -> Self {
        Self {
            instructions: self.instructions.clone(),
            criteria: self.criteria.clone(),
            levels: self.levels.clone(),
        }
    }
}

/// Expected zero-based position and a distribution keyed by rubric variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreAnswer<L: Ord> {
    /// Probability-weighted mean of zero-based positions in the supplied rubric.
    pub score: f64,
    /// Probability for each rubric variant.
    pub probabilities: BTreeMap<L, f64>,
    /// Provider-reported distribution concentration.
    pub confidence: f64,
}

impl<L: Ord> Score<L> {
    /// Define an ordered, typed rubric. Array order determines zero-based wire
    /// positions; the enum's discriminants and ordering do not.
    ///
    /// ```compile_fail,E0080
    /// use rig_typesafeai::Score;
    /// let _ = Score::new("Rate risk", [(0u8, "Only one level")]);
    /// ```
    pub fn new<const N: usize, D: Serialize>(
        instructions: impl Serialize,
        levels: [(L, D); N],
    ) -> Result<Self, ProviderError> {
        const { assert!(N >= 2 && N <= 10, "score requires 2 to 10 levels") };
        Self::try_from_iter(instructions, levels)
    }

    /// Define a typed rubric from a runtime collection, validating its size and
    /// rejecting repeated variants before evaluation.
    pub fn try_from_iter<I, D>(
        instructions: impl Serialize,
        levels: I,
    ) -> Result<Self, ProviderError>
    where
        I: IntoIterator<Item = (L, D)>,
        D: Serialize,
    {
        let mut variants = Vec::new();
        let mut criteria = Vec::new();
        for (level, description) in levels {
            if variants.contains(&level) {
                return Err(ProviderError::Request("score levels must be unique".into()));
            }
            variants.push(level);
            criteria.push(content(description)?);
        }
        if !(2..=10).contains(&criteria.len()) {
            return Err(ProviderError::Request(
                "score requires 2 to 10 levels".into(),
            ));
        }
        Ok(Self {
            instructions: content(instructions)?,
            criteria,
            levels: variants,
        })
    }
}

impl<L: Ord + Clone + WasmCompatSend + WasmCompatSync> Query for Score<L> {
    type Response = Answer;
    type Output = ScoreAnswer<L>;
    fn decode(&self, response: Answer) -> Result<Self::Output, ProviderError> {
        let answer: DynamicScoreAnswer = decode_question(&self.definition(), &response)?;
        let probabilities = answer
            .probabilities
            .into_iter()
            .map(|(index, probability)| {
                let level = self.levels.get(index).ok_or_else(|| {
                    ProviderError::Response(format!("unknown score level: {index}"))
                })?;
                Ok((level.clone(), probability))
            })
            .collect::<Result<_, ProviderError>>()?;
        Ok(ScoreAnswer {
            score: answer.score,
            confidence: answer.confidence,
            probabilities,
        })
    }
}
impl<L> Score<L> {
    fn definition(&self) -> Question {
        Question::Score {
            instructions: self.instructions.clone(),
            criteria: self.criteria.clone(),
        }
    }
}

fn content(value: impl Serialize) -> Result<Value, ProviderError> {
    let value = serde_json::to_value(value)?;
    if !matches!(
        value,
        Value::String(_) | Value::Object(_) | Value::Array(_) | Value::Null
    ) {
        return Err(ProviderError::Request(
            "question content must be a string, object, array, or null".into(),
        ));
    }
    Ok(value)
}
fn question_id(id: impl Into<String>) -> Result<String, ProviderError> {
    let id = id.into();
    if id.is_empty() {
        return Err(ProviderError::Request("question ID cannot be empty".into()));
    }
    Ok(id)
}
impl<T: Serialize + DeserializeOwned + Ord> Choice<T> {
    /// Define a fixed set of alternatives, with its count checked at compile time.
    /// Serde string names become labels; descriptions accept strings, objects,
    /// arrays, or `None` when the label needs no explanation.
    ///
    /// ```compile_fail,E0080
    /// use rig_typesafeai::Choice;
    /// let _ = Choice::new("Choose a route", [("only".to_owned(), "One option")]);
    /// ```
    pub fn new<const N: usize, D: Serialize>(
        instructions: impl Serialize,
        alternatives: [(T, D); N],
    ) -> Result<Self, ProviderError> {
        const { assert!(N >= 2 && N <= 255, "choice requires 2 to 255 alternatives") };
        Self::try_from_iter(instructions, alternatives)
    }

    /// Define runtime-supplied alternatives, validating their count at construction.
    pub fn try_from_iter<I, D>(
        instructions: impl Serialize,
        alternatives: I,
    ) -> Result<Self, ProviderError>
    where
        I: IntoIterator<Item = (T, D)>,
        D: Serialize,
    {
        let mut criteria = BTreeMap::new();
        for (value, description) in alternatives {
            let Value::String(label) = serde_json::to_value(&value)? else {
                return Err(ProviderError::Request(
                    "choice values must serialize to strings".into(),
                ));
            };
            let decoded: T =
                serde_json::from_value(Value::String(label.clone())).map_err(|_| {
                    ProviderError::Request(
                        "choice labels must deserialize to their original values".into(),
                    )
                })?;
            if decoded != value {
                return Err(ProviderError::Request(
                    "choice labels must round-trip without changing values".into(),
                ));
            }
            let description = serde_json::to_value(description)?;
            let description = match description {
                Value::Null => None,
                description => Some(content(description)?),
            };
            if label.is_empty() || criteria.insert(label, description).is_some() {
                return Err(ProviderError::Request(
                    "choice labels must be nonempty and unique".into(),
                ));
            }
        }
        if !(2..=255).contains(&criteria.len()) {
            return Err(ProviderError::Request(
                "choice requires 2 to 255 alternatives".into(),
            ));
        }
        Ok(Self {
            instructions: content(instructions)?,
            criteria,
            marker: PhantomData,
        })
    }
}
impl DynamicScore {
    /// Define a fixed rubric with two to ten levels, checked at compile time.
    /// Levels run from low to high; descriptions may be structured.
    ///
    /// ```compile_fail,E0080
    /// use rig_typesafeai::DynamicScore;
    /// let _ = DynamicScore::new("Rate severity", ["Only one level"]);
    /// ```
    pub fn new<const N: usize, D: Serialize>(
        instructions: impl Serialize,
        levels: [D; N],
    ) -> Result<Self, ProviderError> {
        const { assert!(N >= 2 && N <= 10, "score requires 2 to 10 levels") };
        Self::try_from_iter(instructions, levels)
    }

    /// Define a runtime-supplied rubric, validating its count at construction.
    pub fn try_from_iter<I, D>(
        instructions: impl Serialize,
        levels: I,
    ) -> Result<Self, ProviderError>
    where
        I: IntoIterator<Item = D>,
        D: Serialize,
    {
        let criteria = levels
            .into_iter()
            .map(content)
            .collect::<Result<Vec<_>, _>>()?;
        if !(2..=10).contains(&criteria.len()) {
            return Err(ProviderError::Request(
                "score requires 2 to 10 levels".into(),
            ));
        }
        Ok(Self {
            instructions: content(instructions)?,
            criteria,
        })
    }
}
impl Noul {
    /// Define a statement or question for which a high probability means yes.
    pub fn new(instructions: impl Serialize) -> Result<Self, ProviderError> {
        Ok(Self {
            instructions: content(instructions)?,
            criteria: None,
        })
    }
    /// Clarify outcomes using strings, objects, arrays, or null descriptions.
    pub fn criteria(
        mut self,
        yes: impl Serialize,
        no: impl Serialize,
    ) -> Result<Self, ProviderError> {
        self.criteria = Some(BTreeMap::from([
            ("true".into(), content(yes)?),
            ("false".into(), content(no)?),
        ]));
        Ok(self)
    }
}

/// A serializable query with a statically determined response and output.
/// Struct field names (including Serde renames) supply the wire question IDs.
/// Serde handles encoding and response structure; `decode` validates each field
/// against its original question and converts it to its application answer.
///
/// ```
/// use rig_core::error::ProviderError;
/// use rig_typesafeai::{Noul, NoulAnswer, Query};
/// use serde::{Serialize, Deserialize};
///
/// // The fields are declared once, for both questions and answers.
/// #[derive(Serialize, Deserialize)]
/// struct Assessment<R, V> {
///     ready: R,
///     needs_review: V,
/// }
///
/// impl<R: Query, V: Query> Query for Assessment<R, V> {
///     type Response = Assessment<R::Response, V::Response>;
///     type Output = Assessment<R::Output, V::Output>;
///
///     fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
///         Ok(Assessment {
///             ready: self.ready.decode(response.ready)?,
///             needs_review: self.needs_review.decode(response.needs_review)?,
///         })
///     }
/// }
///
/// let query = Assessment {
///     ready: Noul::new("Is this ready to ship?")?,
///     needs_review: Noul::new("Does this need human review?")?,
/// };
/// # Ok::<(), ProviderError>(())
/// ```
pub trait Query: Serialize + WasmCompatSend + WasmCompatSync {
    /// The intermediate response shape deserialized by Serde.
    type Response: DeserializeOwned;
    /// The validated application answer shape.
    type Output;
    /// Validate and convert a deserialized response. Composite queries delegate
    /// to their fields, keeping validation inside the evaluation error boundary.
    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError>;

    /// Give a standalone question a runtime ID. Named struct fields do not need this.
    fn named(self, id: impl Into<String>) -> Result<NamedQuery<Self>, ProviderError>
    where
        Self: Sized,
    {
        Ok(NamedQuery {
            id: question_id(id)?,
            question: self,
        })
    }

    /// Flatten two object-shaped queries into one evaluation request.
    /// Duplicate IDs are rejected before sending the request. Child queries must
    /// serialize with stable field names: decoding uses those names to partition
    /// the response before delegating to each child.
    fn join<Q: Query>(self, other: Q) -> JoinedQuery<Self, Q>
    where
        Self: Sized,
    {
        JoinedQuery {
            first: self,
            second: other,
        }
    }

    /// Project validated answers into an application value.
    fn map<A, F>(self, map: F) -> MappedQuery<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Output) -> A + WasmCompatSend + WasmCompatSync,
    {
        MappedQuery {
            questions: self,
            map,
        }
    }
}

/// A standalone question serialized under a runtime ID.
#[derive(Debug, Clone)]
pub struct NamedQuery<Q> {
    id: String,
    question: Q,
}
impl<Q: Serialize> Serialize for NamedQuery<Q> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry(&self.id, &self.question)?;
        map.end()
    }
}
impl<Q: Query> Query for NamedQuery<Q> {
    type Response = BTreeMap<String, Q::Response>;
    type Output = Q::Output;
    fn decode(&self, mut response: Self::Response) -> Result<Self::Output, ProviderError> {
        if response.len() != 1 {
            return Err(ProviderError::Response(
                "response question IDs differ from request".into(),
            ));
        }
        let answer = response
            .remove(&self.id)
            .ok_or_else(|| ProviderError::Response(format!("missing answer: {}", self.id)))?;
        self.question.decode(answer)
    }
}

/// Two object-shaped queries serialized into a single flat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinedQuery<A, B> {
    #[serde(flatten)]
    first: A,
    #[serde(flatten)]
    second: B,
}
impl<A: Query, B: Query> Query for JoinedQuery<A, B> {
    type Response = Value;
    type Output = (A::Output, B::Output);
    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
        Ok((
            self.first
                .decode(select_response(&self.first, &response)?)?,
            self.second
                .decode(select_response(&self.second, &response)?)?,
        ))
    }
}

fn select_response<Q: Query>(query: &Q, response: &Value) -> Result<Q::Response, ProviderError> {
    let definitions = serde_json::to_value(query)?;
    let definitions = definitions
        .as_object()
        .ok_or_else(|| ProviderError::Request("joined queries must serialize as objects".into()))?;
    let response = response
        .as_object()
        .ok_or_else(|| ProviderError::Response("answers must be an object".into()))?;
    let selected = definitions
        .keys()
        .map(|id| {
            response
                .get(id)
                .cloned()
                .map(|value| (id.clone(), value))
                .ok_or_else(|| ProviderError::Response(format!("missing answer: {id}")))
        })
        .collect::<Result<serde_json::Map<_, _>, _>>()?;
    Ok(serde_json::from_value(Value::Object(selected))?)
}

/// A serializable query with a typed projection of its validated answers.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct MappedQuery<Q, F> {
    questions: Q,
    #[serde(skip)]
    map: F,
}
impl<Q: Query, F, A> Query for MappedQuery<Q, F>
where
    F: Fn(Q::Output) -> A + WasmCompatSend + WasmCompatSync,
{
    type Response = Q::Response;
    type Output = A;
    fn decode(&self, response: Self::Response) -> Result<A, ProviderError> {
        Ok((self.map)(self.questions.decode(response)?))
    }
}
impl<Q: Query + ?Sized> Query for &Q {
    type Response = Q::Response;
    type Output = Q::Output;
    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
        (**self).decode(response)
    }
}
impl<Q: Query> Query for BTreeMap<String, Q> {
    type Response = BTreeMap<String, Q::Response>;
    type Output = BTreeMap<String, Q::Output>;
    fn decode(&self, response: Self::Response) -> Result<Self::Output, ProviderError> {
        if !self.keys().eq(response.keys()) {
            return Err(ProviderError::Response(
                "response question IDs differ from request".into(),
            ));
        }
        self.iter()
            .zip(response)
            .map(|((_, question), (id, answer))| Ok((id, question.decode(answer)?)))
            .collect()
    }
}

fn decode_question<A: crate::decode::DecodeAnswer>(
    question: &Question,
    answer: &Answer,
) -> Result<A, ProviderError> {
    validate(question, answer)?;
    A::decode(answer)
}
macro_rules! primitive {
    ([$($generics:tt)*] $question:ty => $answer:ty, $kind:ident) => {
        impl $($generics)* $question {
            fn definition(&self) -> Question {
                Question::$kind { instructions: self.instructions.clone(), criteria: self.criteria.clone() }
            }
        }
        impl $($generics)* Query for $question {
            type Response = Answer;
            type Output = $answer;
            fn decode(&self, response: Answer) -> Result<Self::Output, ProviderError> {
                decode_question(&self.definition(), &response)
            }
        }
    };
}
primitive!([<T: DeserializeOwned + Ord>] Choice<T> => ChoiceAnswer<T>, Choice);
primitive!([] DynamicScore => DynamicScoreAnswer, Score);
primitive!([] Noul => NoulAnswer, Noul);

impl<T> Serialize for Choice<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Question::Choice {
            instructions: self.instructions.clone(),
            criteria: self.criteria.clone(),
        }
        .serialize(serializer)
    }
}
impl<L> Serialize for Score<L> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.definition().serialize(serializer)
    }
}
impl Serialize for DynamicScore {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.definition().serialize(serializer)
    }
}
impl Serialize for Noul {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.definition().serialize(serializer)
    }
}

pub(crate) fn validate_definition(id: &str, question: &Question) -> Result<(), ProviderError> {
    question_id(id)?;
    match question {
        Question::Choice {
            instructions,
            criteria,
        } => {
            content(instructions)?;
            if !(2..=255).contains(&criteria.len()) || criteria.keys().any(String::is_empty) {
                return Err(ProviderError::Request(
                    "choice requires 2 to 255 nonempty labels".into(),
                ));
            }
            for description in criteria.values().flatten() {
                content(description)?;
            }
        }
        Question::Score {
            instructions,
            criteria,
        } => {
            content(instructions)?;
            if !(2..=10).contains(&criteria.len()) {
                return Err(ProviderError::Request(
                    "score requires 2 to 10 levels".into(),
                ));
            }
            for description in criteria {
                content(description)?;
            }
        }
        Question::Noul {
            instructions,
            criteria,
        } => {
            content(instructions)?;
            if let Some(criteria) = criteria {
                if criteria.len() != 2
                    || !criteria.contains_key("true")
                    || !criteria.contains_key("false")
                {
                    return Err(ProviderError::Request(
                        "Noul criteria require true and false descriptions".into(),
                    ));
                }
                for description in criteria.values() {
                    content(description)?;
                }
            }
        }
    }
    Ok(())
}

fn probability(value: f64) -> Result<(), ProviderError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ProviderError::Response(
            "probability or confidence is outside [0, 1]".into(),
        ));
    }
    Ok(())
}
fn rounded_to_hundredths(probabilities: &BTreeMap<String, f64>) -> bool {
    probabilities.values().all(|value| {
        let cents = value * 100.0;
        (cents - cents.round()).abs() <= 1e-9
    })
}
pub(crate) fn distribution(probabilities: &BTreeMap<String, f64>) -> Result<(), ProviderError> {
    for &weight in probabilities.values() {
        probability(weight)?;
    }
    // Recorded Jev responses round probabilities to hundredths independently.
    // Allow their accumulated rounding error, but cap the relaxation so a large
    // choice set cannot make a substantially unnormalized distribution valid.
    let rounded_to_cents = rounded_to_hundredths(probabilities);
    let tolerance = if rounded_to_cents {
        (probabilities.len() as f64 * 0.005).min(0.02) + 1e-12
    } else {
        1e-3
    };
    if (probabilities.values().sum::<f64>() - 1.0).abs() > tolerance {
        return Err(ProviderError::Response(
            "probabilities do not sum to one".into(),
        ));
    }
    Ok(())
}
pub(crate) fn validate(question: &Question, answer: &Answer) -> Result<(), ProviderError> {
    match (question, answer) {
        (
            Question::Choice { criteria, .. },
            Answer::Choice {
                choice,
                probabilities,
                confidence,
            },
        ) => {
            if !criteria.contains_key(choice) || !criteria.keys().eq(probabilities.keys()) {
                return Err(ProviderError::Response(
                    "choice labels differ from the requested alternatives".into(),
                ));
            }
            distribution(probabilities)?;
            probability(*confidence)?;
            let selected = probabilities.get(choice).copied().ok_or_else(|| {
                ProviderError::Response("selected choice has no probability".into())
            })?;
            if probabilities.values().any(|p| *p > selected + 1e-6) {
                return Err(ProviderError::Response(
                    "selected choice is not a maximum".into(),
                ));
            }
        }
        (
            Question::Score { criteria, .. },
            Answer::Score {
                score,
                probabilities,
                legend,
                confidence,
            },
        ) => {
            let expected = criteria
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect::<BTreeMap<_, _>>();
            if &expected != legend || !expected.keys().eq(probabilities.keys()) {
                return Err(ProviderError::Response(
                    "score levels differ from the requested rubric".into(),
                ));
            }
            distribution(probabilities)?;
            probability(*confidence)?;
            if !score.is_finite() || *score < 0.0 || *score > (criteria.len() - 1) as f64 {
                return Err(ProviderError::Response(
                    "score is outside the rubric".into(),
                ));
            }
            let total = probabilities.values().sum::<f64>();
            // Keys were checked against the zero-based rubric above (at most ten
            // levels), so BTreeMap order is also numeric order here.
            let mean = probabilities
                .values()
                .enumerate()
                .map(|(index, weight)| index as f64 * weight)
                .sum::<f64>()
                / total;
            // Independently rounded weights can move the normalized mean by
            // sum(|index - mean| * 0.005), since the unrounded weights sum
            // to one. The score itself is also
            // rounded to hundredths in recorded responses.
            let weight_error = if rounded_to_hundredths(probabilities) {
                (0..criteria.len())
                    .map(|index| (index as f64 - mean).abs() * 0.005)
                    .sum::<f64>()
            } else {
                0.0
            };
            if (*score - mean).abs() > weight_error + 0.005 + 1e-12 {
                return Err(ProviderError::Response(
                    "score differs from the probability-weighted mean".into(),
                ));
            }
        }
        (Question::Noul { .. }, Answer::Noul { noul }) => probability(*noul)?,
        _ => {
            return Err(ProviderError::Response(
                "answer kind differs from question kind".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn state(value: impl Serialize) -> Result<Value, ProviderError> {
    let value = content(value)?;
    if value.is_null() {
        return Err(ProviderError::Request("state cannot be null".into()));
    }
    Ok(value)
}
