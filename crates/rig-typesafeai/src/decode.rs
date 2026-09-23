//! Typed conversion after protocol validation.
use rig_core::error::ProviderError;
use serde::de::DeserializeOwned;

/// Direct conversion from a validated protocol answer into its application type.
pub(crate) trait DecodeAnswer: Sized {
    fn decode(answer: &crate::types::Answer) -> Result<Self, ProviderError>;
}

fn wrong_kind() -> ProviderError {
    ProviderError::Response("answer kind differs from question kind".into())
}

impl<T: DeserializeOwned + Ord> DecodeAnswer for crate::ChoiceAnswer<T> {
    fn decode(answer: &crate::types::Answer) -> Result<Self, ProviderError> {
        let crate::types::Answer::Choice {
            choice,
            probabilities,
            confidence,
        } = answer
        else {
            return Err(wrong_kind());
        };
        let label = |label: &str| -> Result<T, ProviderError> {
            Ok(T::deserialize(serde::de::value::StrDeserializer::<
                serde_json::Error,
            >::new(label))?)
        };
        Ok(Self {
            choice: label(choice)?,
            probabilities: probabilities
                .iter()
                .map(|(key, value)| Ok((label(key)?, *value)))
                .collect::<Result<_, ProviderError>>()?,
            confidence: *confidence,
        })
    }
}

fn index(key: &str) -> Result<usize, ProviderError> {
    key.parse()
        .map_err(|_| ProviderError::Response(format!("invalid score index: {key}")))
}

impl DecodeAnswer for crate::DynamicScoreAnswer {
    fn decode(answer: &crate::types::Answer) -> Result<Self, ProviderError> {
        let crate::types::Answer::Score {
            score,
            probabilities,
            legend,
            confidence,
        } = answer
        else {
            return Err(wrong_kind());
        };
        Ok(Self {
            score: *score,
            probabilities: probabilities
                .iter()
                .map(|(key, value)| Ok((index(key)?, *value)))
                .collect::<Result<_, ProviderError>>()?,
            legend: legend
                .iter()
                .map(|(key, value)| Ok((index(key)?, value.clone())))
                .collect::<Result<_, ProviderError>>()?,
            confidence: *confidence,
        })
    }
}

impl DecodeAnswer for crate::NoulAnswer {
    fn decode(answer: &crate::types::Answer) -> Result<Self, ProviderError> {
        let crate::types::Answer::Noul { noul } = answer else {
            return Err(wrong_kind());
        };
        Ok(Self { noul: *noul })
    }
}
