use crate::models::tweets::Mention;
use crate::models::tweets::PlaceRaw;
use crate::models::{Profile, Tweet};
use crate::profile::LegacyUserRaw;
use crate::timeline::tweet_utils::{parse_media_groups, reconstruct_tweet_html};
use chrono::DateTime;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize, Serialize)]
pub struct Hashtag {
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineUserMentionBasicRaw {
    pub id_str: Option<String>,
    pub name: Option<String>,
    pub screen_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineMediaBasicRaw {
    pub media_url_https: Option<String>,
    pub r#type: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineUrlBasicRaw {
    pub expanded_url: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ExtSensitiveMediaWarningRaw {
    pub adult_content: Option<bool>,
    pub graphic_violence: Option<bool>,
    pub other: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VideoVariant {
    pub bitrate: Option<i32>,
    pub url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VideoInfo {
    pub variants: Option<Vec<VideoVariant>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineMediaExtendedRaw {
    pub id_str: Option<String>,
    pub media_url_https: Option<String>,
    pub ext_sensitive_media_warning: Option<ExtSensitiveMediaWarningRaw>,
    pub r#type: Option<String>,
    pub url: Option<String>,
    pub video_info: Option<VideoInfo>,
    pub ext_alt_text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SearchResultRaw {
    pub rest_id: Option<String>,
    pub __typename: Option<String>,
    pub core: Option<UserResultsCore>,
    pub views: Option<Views>,
    pub note_tweet: Option<NoteTweet>,
    pub quoted_status_result: Option<QuotedStatusResult>,
    pub legacy: Option<LegacyTweetRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UserResultsCore {
    pub user_results: Option<UserResults>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UserResults {
    pub result: Option<UserResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UserResult {
    pub is_blue_verified: Option<bool>,
    pub legacy: Option<LegacyUserRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Views {
    pub count: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NoteTweet {
    pub note_tweet_results: Option<NoteTweetResults>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NoteTweetResults {
    pub result: Option<NoteTweetResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct NoteTweetResult {
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct QuotedStatusResult {
    pub result: Option<Box<SearchResultRaw>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineResultRaw {
    pub result: Option<Box<TimelineResultRaw>>,
    pub rest_id: Option<String>,
    pub __typename: Option<String>,
    pub core: Option<TimelineCore>,
    pub views: Option<TimelineViews>,
    pub note_tweet: Option<TimelineNoteTweet>,
    pub quoted_status_result: Option<Box<TimelineQuotedStatus>>,
    pub legacy: Option<Box<LegacyTweetRaw>>,
    pub tweet: Option<Box<TimelineResultRaw>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineCore {
    pub user_results: Option<TimelineUserResults>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineUserResults {
    pub result: Option<TimelineUserResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineUserResult {
    pub is_blue_verified: Option<bool>,
    pub legacy: Option<LegacyUserRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineViews {
    pub count: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineNoteTweet {
    pub note_tweet_results: Option<TimelineNoteTweetResults>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineNoteTweetResults {
    pub result: Option<TimelineNoteTweetResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineNoteTweetResult {
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineQuotedStatus {
    pub result: Option<Box<TimelineResultRaw>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LegacyTweetRaw {
    pub bookmark_count: Option<i32>,
    pub conversation_id_str: Option<String>,
    pub created_at: Option<String>,
    pub favorite_count: Option<i32>,
    pub full_text: Option<String>,
    pub entities: Option<TweetEntities>,
    pub extended_entities: Option<TweetExtendedEntities>,
    pub id_str: Option<String>,
    pub in_reply_to_status_id_str: Option<String>,
    pub place: Option<PlaceRaw>,
    pub reply_count: Option<i32>,
    pub retweet_count: Option<i32>,
    pub retweeted_status_id_str: Option<String>,
    pub retweeted_status_result: Option<TimelineRetweetedStatus>,
    pub quoted_status_id_str: Option<String>,
    pub time: Option<String>,
    pub user_id_str: Option<String>,
    pub ext_views: Option<TweetExtViews>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TweetEntities {
    pub hashtags: Option<Vec<Hashtag>>,
    pub media: Option<Vec<TimelineMediaBasicRaw>>,
    pub urls: Option<Vec<TimelineUrlBasicRaw>>,
    pub user_mentions: Option<Vec<TimelineUserMentionBasicRaw>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TweetExtendedEntities {
    pub media: Option<Vec<TimelineMediaExtendedRaw>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineRetweetedStatus {
    pub result: Option<TimelineResultRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TweetExtViews {
    pub state: Option<String>,
    pub count: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineGlobalObjectsRaw {
    pub tweets: Option<HashMap<String, Option<LegacyTweetRaw>>>,
    pub users: Option<HashMap<String, Option<LegacyUserRaw>>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRawCursor {
    pub value: Option<String>,
    pub cursor_type: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRawEntity {
    pub id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRawModuleItem {
    pub client_event_info: Option<ClientEventInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ClientEventInfo {
    pub details: Option<ClientEventDetails>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ClientEventDetails {
    pub guide_details: Option<GuideDetails>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GuideDetails {
    pub transparent_guide_details: Option<TransparentGuideDetails>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TransparentGuideDetails {
    pub trend_metadata: Option<TrendMetadata>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TrendMetadata {
    pub trend_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRawAddEntry {
    pub content: Option<TimelineEntryContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRawPinEntry {
    pub content: Option<TimelinePinContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelinePinContent {
    pub item: Option<TimelineItem>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRawReplaceEntry {
    pub content: Option<TimelineReplaceContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineReplaceContent {
    pub operation: Option<TimelineOperation>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRawInstruction {
    pub add_entries: Option<TimelineAddEntries>,
    pub pin_entry: Option<TimelineDataRawPinEntry>,
    pub replace_entry: Option<TimelineDataRawReplaceEntry>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineAddEntries {
    pub entries: Option<Vec<TimelineDataRawAddEntry>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineDataRaw {
    pub instructions: Option<Vec<TimelineDataRawInstruction>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineV1 {
    pub global_objects: Option<TimelineGlobalObjectsRaw>,
    pub timeline: Option<TimelineDataRaw>,
}

#[derive(Debug)]
pub enum ParseTweetResult {
    Success { tweet: Tweet },
    Error { err: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryTweetsResponse {
    pub tweets: Vec<Tweet>,
    pub next: Option<String>,
    pub previous: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct QueryProfilesResponse {
    pub profiles: Vec<Profile>,
    pub next: Option<String>,
    pub previous: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineEntryContent {
    pub item: Option<TimelineItem>,
    pub operation: Option<TimelineOperation>,
    pub timeline_module: Option<TimelineModule>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineItem {
    pub content: Option<TimelineContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineContent {
    pub tweet: Option<TimelineDataRawEntity>,
    pub user: Option<TimelineDataRawEntity>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineOperation {
    pub cursor: Option<TimelineDataRawCursor>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineModule {
    pub items: Option<Vec<TimelineModuleItemWrapper>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineModuleItemWrapper {
    pub item: Option<TimelineDataRawModuleItem>,
}

#[derive(Debug)]
pub struct UserMention {
    pub id: String,
    pub username: String,
    pub name: String,
}

pub fn parse_timeline_tweet(timeline: &TimelineV1, id: &str) -> ParseTweetResult {
    let empty_tweets = HashMap::new();
    let tweets = match &timeline.global_objects {
        Some(go) => go.tweets.as_ref().unwrap_or(&empty_tweets),
        None => {
            return ParseTweetResult::Error {
                err: "No global objects found".to_string(),
            }
        }
    };

    let tweet = match tweets.get(id) {
        Some(Some(t)) => t,
        _ => {
            return ParseTweetResult::Error {
                err: format!("Tweet \"{}\" was not found in the timeline object.", id),
            }
        }
    };

    let user_id = match &tweet.user_id_str {
        Some(id) => id,
        None => {
            return ParseTweetResult::Error {
                err: "Tweet has no user ID".to_string(),
            }
        }
    };

    let empty_users = HashMap::new();
    let users = match &timeline.global_objects {
        Some(go) => go.users.as_ref().unwrap_or(&empty_users),
        None => {
            return ParseTweetResult::Error {
                err: "No users found".to_string(),
            }
        }
    };

    let user = match users.get(user_id) {
        Some(Some(u)) => u,
        _ => {
            return ParseTweetResult::Error {
                err: format!("User \"{}\" has no username data.", user_id),
            }
        }
    };

    let hashtags = tweet
        .entities
        .as_ref()
        .and_then(|e| e.hashtags.as_ref())
        .map(|h| h.iter().filter_map(|tag| tag.text.clone()).collect())
        .unwrap_or_default();

    let mentions = tweet
        .entities
        .as_ref()
        .and_then(|e| e.user_mentions.as_ref())
        .map(|m| {
            m.iter()
                .filter_map(|mention| {
                    if let (Some(id), Some(screen_name), Some(name)) =
                        (&mention.id_str, &mention.screen_name, &mention.name)
                    {
                        Some(Mention {
                            id: id.clone(),
                            username: Some(screen_name.clone()),
                            name: Some(name.clone()),
                        })
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let empty_media = Vec::new();
    let media = tweet
        .extended_entities
        .as_ref()
        .and_then(|e| e.media.as_ref())
        .unwrap_or(&empty_media);

    let urls = tweet
        .entities
        .as_ref()
        .and_then(|e| e.urls.as_ref())
        .map(|u| {
            u.iter()
                .filter_map(|url| url.expanded_url.clone())
                .collect()
        })
        .unwrap_or_default();

    let (photos, videos, sensitive_content) = parse_media_groups(media);

    let mut tweet_obj = Tweet {
        conversation_id: tweet.conversation_id_str.clone(),
        id: Some(id.to_string()),
        hashtags,
        likes: tweet.favorite_count,
        mentions,
        name: user.name.clone(),
        permanent_url: Some(format!(
            "https://twitter.com/{}/status/{}",
            user.screen_name.as_ref().unwrap_or(&String::new()),
            id
        )),
        photos,
        replies: tweet.reply_count,
        retweets: tweet.retweet_count,
        text: tweet.full_text.clone(),
        thread: Vec::new(),
        urls,
        user_id: tweet.user_id_str.clone(),
        username: user.screen_name.clone(),
        videos,
        time_parsed: None,
        timestamp: None,
        place: None,
        is_quoted: Some(false),
        quoted_status_id: None,
        quoted_status: None,
        is_reply: Some(false),
        in_reply_to_status_id: None,
        in_reply_to_status: None,
        is_retweet: Some(false),
        retweeted_status_id: None,
        retweeted_status: None,
        views: None,
        is_pin: Some(false),
        sensitive_content: Some(sensitive_content),
        html: None,
        bookmark_count: None,
        is_self_thread: None,
        poll: None,
        created_at: None,
        ext_views: None,
        quote_count: None,
        reply_count: None,
        retweet_count: None,
        screen_name: None,
        thread_id: None,
    };

    if let Some(created_at) = &tweet.created_at {
        if let Ok(parsed_time) = DateTime::parse_from_str(created_at, "%a %b %d %H:%M:%S %z %Y") {
            tweet_obj.time_parsed = Some(parsed_time.with_timezone(&Utc));
            tweet_obj.timestamp = Some(parsed_time.timestamp());
        }
    }

    if let Some(place) = &tweet.place {
        tweet_obj.place = Some(place.clone());
    }

    if let Some(quoted_id) = &tweet.quoted_status_id_str {
        tweet_obj.is_quoted = Some(true);
        tweet_obj.quoted_status_id = Some(quoted_id.clone());

        if let ParseTweetResult::Success {
            tweet: quoted_tweet,
        } = parse_timeline_tweet(timeline, quoted_id)
        {
            tweet_obj.quoted_status = Some(Box::new(quoted_tweet));
        }
    }

    if let Some(ext_views) = &tweet.ext_views {
        if let Some(count) = &ext_views.count {
            if let Ok(views) = count.parse::<i32>() {
                tweet_obj.views = Some(views);
            }
        }
    }

    tweet_obj.html = reconstruct_tweet_html(tweet, &tweet_obj.photos, &tweet_obj.videos);

    ParseTweetResult::Success { tweet: tweet_obj }
}
