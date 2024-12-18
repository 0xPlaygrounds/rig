use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub username: String,
    pub name: String,
    pub description: Option<String>,
    pub location: Option<String>,
    pub url: Option<String>,
    pub protected: bool,
    pub verified: bool,
    pub followers_count: i32,
    pub following_count: i32,
    pub tweets_count: i32,
    pub listed_count: i32,
    pub created_at: DateTime<Utc>,
    pub profile_image_url: Option<String>,
    pub profile_banner_url: Option<String>,
    pub pinned_tweet_id: Option<String>,
    pub is_blue_verified: Option<bool>,
}