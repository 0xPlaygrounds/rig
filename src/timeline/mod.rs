pub mod home;
pub mod search;
pub mod tweet_utils;
pub mod v1;
pub mod v2;
#[derive(Debug, Clone, Default)]
pub struct TimelineParams {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    pub include_replies: bool,
}
