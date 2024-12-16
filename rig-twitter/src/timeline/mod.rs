pub mod home;
pub mod search;
pub mod v1;
pub mod v2;
pub mod timeline_async;
pub mod tweet_utils;
#[derive(Debug, Clone, Default)]
pub struct TimelineParams {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    pub include_replies: bool,
}
