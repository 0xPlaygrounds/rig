use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tweet {
    pub ext_views: Option<i32>,
    pub created_at: Option<String>,
    pub bookmark_count: Option<i32>,
    pub conversation_id: Option<String>,
    pub hashtags: Vec<String>,
    pub html: Option<String>,
    pub id: Option<String>,
    pub in_reply_to_status: Option<Box<Tweet>>,
    pub in_reply_to_status_id: Option<String>,
    pub is_quoted: Option<bool>,
    pub is_pin: Option<bool>,
    pub is_reply: Option<bool>,
    pub is_retweet: Option<bool>,
    pub is_self_thread: Option<bool>,
    pub likes: Option<i32>,
    pub name: Option<String>,
    pub mentions: Vec<Mention>,
    pub permanent_url: Option<String>,
    pub photos: Vec<Photo>,
    pub place: Option<PlaceRaw>,
    pub quoted_status: Option<Box<Tweet>>,
    pub quoted_status_id: Option<String>,
    pub replies: Option<i32>,
    pub retweets: Option<i32>,
    pub retweeted_status: Option<Box<Tweet>>,
    pub retweeted_status_id: Option<String>,
    pub text: Option<String>,
    pub thread: Vec<Tweet>,
    pub time_parsed: Option<DateTime<Utc>>,
    pub timestamp: Option<i64>,
    pub urls: Vec<String>,
    pub user_id: Option<String>,
    pub username: Option<String>,
    pub videos: Vec<Video>,
    pub views: Option<i32>,
    pub sensitive_content: Option<bool>,
    pub poll: Option<PollV2>,
    pub quote_count: Option<i32>,
    pub reply_count: Option<i32>,
    pub retweet_count: Option<i32>,
    pub screen_name: Option<String>,
    pub thread_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mention {
    pub id: String,
    pub username: Option<String>,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Photo {
    pub id: String,
    pub url: String,
    pub alt_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    pub id: String,
    pub preview: String,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaceRaw {
    pub id: Option<String>,
    pub place_type: Option<String>,
    pub name: Option<String>,
    pub full_name: Option<String>,
    pub country_code: Option<String>,
    pub country: Option<String>,
    pub bounding_box: Option<BoundingBox>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub coordinates: Option<Vec<Vec<Vec<f64>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollV2 {
    pub id: Option<String>,
    pub end_datetime: Option<String>,
    pub voting_status: Option<String>,
    pub options: Vec<PollOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollOption {
    pub position: Option<i32>,
    pub label: String,
    pub votes: Option<i32>,
}
