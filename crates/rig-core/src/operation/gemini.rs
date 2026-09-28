//! Gemini's resource operations beside completion: counting a request's
//! tokens, the Files API, and batch jobs. Each reply is one whole document.
//!
//! ```
//! use rig_core::operation::{BatchJobs, FileStore, TokenCount};
//! use rig_core::wire::Operation;
//!
//! fn whole<Op: Operation>() {}
//! whole::<TokenCount>();
//! whole::<FileStore>();
//! whole::<BatchJobs>();
//! ```

use super::Whole;
use crate::completion::CompletionRequest;
use crate::providers::gemini::api::CountTokensResponse;
use crate::providers::gemini::batches::{BatchReply, BatchRequest};
use crate::providers::gemini::files::{FileReply, FileRequest};
use crate::wire::{Call, Free, Operation};

/// Counts the tokens a completion request would send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCount;

impl Operation for TokenCount {
    type Request = CompletionRequest;
    type Event = std::convert::Infallible;
    type End = CountTokensResponse;
    type Response = CountTokensResponse;
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Whole::new()
    }
}

/// Uploads, reads, lists and deletes files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStore;

impl Operation for FileStore {
    type Request = FileRequest;
    type Event = std::convert::Infallible;
    type End = FileReply;
    type Response = FileReply;
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Whole::new()
    }
}

/// Creates, reads, lists, cancels and deletes batch jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchJobs;

impl Operation for BatchJobs {
    type Request = BatchRequest;
    type Event = std::convert::Infallible;
    type End = BatchReply;
    type Response = BatchReply;
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Whole::new()
    }
}
