use crate::error::Result;
use crate::error::TwitterError;
use crate::models::tweets::Mention;
use crate::models::Tweet;
use crate::profile::LegacyUserRaw;
use crate::timeline::tweet_utils::parse_media_groups;
use crate::timeline::v1::{LegacyTweetRaw, TimelineResultRaw};
use chrono::Utc;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
lazy_static! {
    static ref EMPTY_INSTRUCTIONS: Vec<TimelineInstruction> = Vec::new();
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Timeline {
    pub timeline: Option<TimelineItems>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineContent {
    pub instructions: Option<Vec<TimelineInstruction>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineData {
    pub user: Option<TimelineUser>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineEntities {
    pub hashtags: Option<Vec<Hashtag>>,
    pub user_mentions: Option<Vec<UserMention>>,
    pub urls: Option<Vec<UrlEntity>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineEntry {
    #[serde(rename = "entryId")]
    pub entry_id: Option<String>,
    pub content: Option<EntryContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineEntryItemContent {
    pub item_type: Option<String>,
    pub tweet_display_type: Option<String>,
    pub tweet_result: Option<TweetResult>,
    pub tweet_results: Option<TweetResult>,
    pub user_display_type: Option<String>,
    pub user_results: Option<TimelineUserResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineEntryItemContentRaw {
    #[serde(rename = "itemType")]
    pub item_type: Option<String>,
    #[serde(rename = "tweetDisplayType")]
    pub tweet_display_type: Option<String>,
    #[serde(rename = "tweetResult")]
    pub tweet_result: Option<TweetResultRaw>,
    pub tweet_results: Option<TweetResultRaw>,
    #[serde(rename = "userDisplayType")]
    pub user_display_type: Option<String>,
    pub user_results: Option<TimelineUserResultRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineItems {
    pub instructions: Option<Vec<TimelineInstruction>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineUser {
    pub result: Option<TimelineUserResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineUserResult {
    pub rest_id: Option<String>,
    pub legacy: Option<LegacyUserRaw>,
    pub is_blue_verified: Option<bool>,
    pub timeline_v2: Option<Box<TimelineV2>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineUserResultRaw {
    pub result: Option<TimelineUserResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineV2 {
    pub data: Option<TimelineData>,
    pub timeline: Option<TimelineItems>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ThreadedConversation {
    pub data: Option<ThreadedConversationData>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ThreadedConversationData {
    pub threaded_conversation_with_injections_v2: Option<TimelineContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TweetResult {
    pub result: Option<TimelineResultRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TweetResultRaw {
    pub result: Option<TimelineResultRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EntryContent {
    #[serde(rename = "cursorType")]
    pub cursor_type: Option<String>,
    pub value: Option<String>,
    pub items: Option<Vec<EntryItem>>,
    #[serde(rename = "itemContent")]
    pub item_content: Option<TimelineEntryItemContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EntryItem {
    #[serde(rename = "entryId")]
    pub entry_id: Option<String>,
    pub item: Option<ItemContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ItemContent {
    pub content: Option<TimelineEntryItemContent>,
    #[serde(rename = "itemContent")]
    pub item_content: Option<TimelineEntryItemContent>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Hashtag {
    pub text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UrlEntity {
    pub expanded_url: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UserMention {
    pub id_str: Option<String>,
    pub name: Option<String>,
    pub screen_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimelineInstruction {
    pub entries: Option<Vec<TimelineEntry>>,
    pub entry: Option<TimelineEntry>,
    #[serde(rename = "type")]
    pub type_: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SearchEntryRaw {
    #[serde(rename = "entryId")]
    pub entry_id: String,
    #[serde(rename = "sortIndex")]
    pub sort_index: String,
    pub content: Option<SearchEntryContentRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SearchEntryContentRaw {
    #[serde(rename = "cursorType")]
    pub cursor_type: Option<String>,
    #[serde(rename = "entryType")]
    pub entry_type: Option<String>,
    #[serde(rename = "__typename")]
    pub typename: Option<String>,
    pub value: Option<String>,
    pub items: Option<Vec<SearchEntryItemRaw>>,
    #[serde(rename = "itemContent")]
    pub item_content: Option<TimelineEntryItemContentRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SearchEntryItemRaw {
    pub item: Option<SearchEntryItemInnerRaw>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SearchEntryItemInnerRaw {
    pub content: Option<TimelineEntryItemContentRaw>,
}

pub fn parse_legacy_tweet(
    user: Option<&LegacyUserRaw>,
    tweet: Option<&LegacyTweetRaw>,
) -> Result<Tweet> {
    let tweet = tweet.ok_or(TwitterError::Api(
        "Tweet was not found in the timeline object".into(),
    ))?;
    let user = user.ok_or(TwitterError::Api(
        "User was not found in the timeline object".into(),
    ))?;

    let id_str = tweet
        .id_str
        .as_ref()
        .or(tweet.conversation_id_str.as_ref())
        .ok_or(TwitterError::Api("Tweet ID was not found in object".into()))?;

    let hashtags = tweet
        .entities
        .as_ref()
        .and_then(|e| e.hashtags.as_ref())
        .map(|h| h.iter().filter_map(|h| h.text.clone()).collect())
        .unwrap_or_default();

    let mentions = tweet
        .entities
        .as_ref()
        .and_then(|e| e.user_mentions.as_ref())
        .map(|mentions| {
            mentions
                .iter()
                .filter_map(|m| {
                    Some(Mention {
                        id: m.id_str.clone().unwrap_or_default(),
                        name: m.name.clone(),
                        username: m.screen_name.clone(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let (photos, videos, _) =
        if let Some(extended_entities) = &tweet.extended_entities {
            if let Some(media) = &extended_entities.media {
                parse_media_groups(media)
            } else {
                (Vec::new(), Vec::new(), false)
            }
        } else {
            (Vec::new(), Vec::new(), false)
        };

    let mut tweet = Tweet {
        bookmark_count: tweet.bookmark_count,
        conversation_id: tweet.conversation_id_str.clone(),
        id: Some(id_str.clone()),
        hashtags,
        likes: tweet.favorite_count,
        mentions,
        name: user.name.clone(),
        permanent_url: Some(format!(
            "https://twitter.com/{}/status/{}",
            user.screen_name.as_ref().unwrap_or(&String::new()),
            id_str
        )),
        photos,
        replies: tweet.reply_count,
        retweets: tweet.retweet_count,
        text: tweet.full_text.clone(),
        thread: Vec::new(),
        urls: tweet
            .entities
            .as_ref()
            .and_then(|e| e.urls.as_ref())
            .map(|urls| urls.iter().filter_map(|u| u.expanded_url.clone()).collect())
            .unwrap_or_default(),
        user_id: tweet.user_id_str.clone(),
        username: user.screen_name.clone(),
        videos,
        is_quoted: Some(false),
        is_reply: Some(false),
        is_retweet: Some(false),
        is_pin: Some(false),
        sensitive_content: Some(false),
        quoted_status: None,
        quoted_status_id: tweet.quoted_status_id_str.clone(),
        in_reply_to_status_id: tweet.in_reply_to_status_id_str.clone(),
        retweeted_status: None,
        retweeted_status_id: None,
        views: None,
        html: None,
        time_parsed: None,
        timestamp: None,
        place: tweet.place.clone(),
        in_reply_to_status: None,
        is_self_thread: None,
        poll: None,
        created_at: tweet.created_at.clone(),
        ext_views: None,
        quote_count: None,
        reply_count: None,
        retweet_count: None,
        screen_name: None,
        thread_id: None,
    };

    if let Some(created_at) = &tweet.created_at {
        if let Ok(time) = chrono::DateTime::parse_from_str(created_at, "%a %b %d %H:%M:%S %z %Y") {
            tweet.time_parsed = Some(time.with_timezone(&Utc));
            tweet.timestamp = Some(time.timestamp());
        }
    }

    if let Some(views) = &tweet.ext_views {
        tweet.views = Some(*views);
    }

    // Set HTML
    // tweet.html = reconstruct_tweet_html(tweet, &photos, &videos);

    Ok(tweet)
}

pub fn parse_timeline_entry_item_content_raw(
    content: &TimelineEntryItemContent,
    _entry_id: &str,
    is_conversation: bool,
) -> Option<Tweet> {
    let result = content
        .tweet_results
        .as_ref()
        .or(content.tweet_result.as_ref())
        .and_then(|r| r.result.as_ref())?;

    let tweet_result = parse_result(result);
    if tweet_result.success {
        let mut tweet = tweet_result.tweet?;

        if is_conversation && content.tweet_display_type.as_deref() == Some("SelfThread") {
            tweet.is_self_thread = Some(true);
        }

        return Some(tweet);
    }

    None
}

pub fn parse_and_push(
    tweets: &mut Vec<Tweet>,
    content: &TimelineEntryItemContent,
    entry_id: String,
    is_conversation: bool,
) {
    if let Some(tweet) = parse_timeline_entry_item_content_raw(content, &entry_id, is_conversation)
    {
        tweets.push(tweet);
    }
}

pub fn parse_result(result: &TimelineResultRaw) -> ParseTweetResult {
    let tweet_result = parse_legacy_tweet(
        result
            .core
            .as_ref()
            .and_then(|c| c.user_results.as_ref())
            .and_then(|u| u.result.as_ref())
            .and_then(|r| r.legacy.as_ref()),
        result.legacy.as_deref(),
    );

    let mut tweet = match tweet_result {
        Ok(tweet) => tweet,
        Err(e) => {
            return ParseTweetResult {
                success: false,
                tweet: None,
                err: Some(e),
            }
        }
    };

    if tweet.views.is_none() {
        if let Some(count) = result
            .views
            .as_ref()
            .and_then(|v| v.count.as_ref())
            .and_then(|c| c.parse().ok())
        {
            tweet.views = Some(count);
        }
    }

    if let Some(quoted) = result.quoted_status_result.as_ref() {
        if let Some(quoted_result) = quoted.result.as_ref() {
            let quoted_tweet_result = parse_result(quoted_result);
            if quoted_tweet_result.success {
                tweet.quoted_status = quoted_tweet_result.tweet.map(Box::new);
            }
        }
    }

    ParseTweetResult {
        success: true,
        tweet: Some(tweet),
        err: None,
    }
}

pub struct ParseTweetResult {
    pub success: bool,
    pub tweet: Option<Tweet>,
    pub err: Option<TwitterError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryTweetsResponse {
    pub tweets: Vec<Tweet>,
    pub next: Option<String>,
    pub previous: Option<String>,
}

pub fn parse_timeline_tweets_v2(timeline: &TimelineV2) -> QueryTweetsResponse {
    let mut tweets = Vec::new();
    let mut bottom_cursor = None;
    let mut top_cursor = None;

    let instructions = timeline
        .data
        .as_ref()
        .and_then(|data| data.user.as_ref())
        .and_then(|user| user.result.as_ref())
        .and_then(|result| result.timeline_v2.as_ref())
        .and_then(|timeline| timeline.timeline.as_ref())
        .and_then(|timeline| timeline.instructions.as_ref())
        .unwrap_or(&EMPTY_INSTRUCTIONS);

    let expected_entry_types = ["tweet-", "profile-conversation-"];

    for instruction in instructions {
        let entries = instruction
            .entries.as_deref()
            .unwrap_or_else(|| {
                instruction
                    .entry
                    .as_ref()
                    .map(std::slice::from_ref)
                    .unwrap_or_default()
            });

        for entry in entries {
            let content = match &entry.content {
                Some(content) => content,
                None => continue,
            };

            if let Some(cursor_type) = &content.cursor_type {
                match cursor_type.as_str() {
                    "Bottom" => {
                        bottom_cursor = content.value.clone();
                        continue;
                    }
                    "Top" => {
                        top_cursor = content.value.clone();
                        continue;
                    }
                    _ => {}
                }
            }

            let entry_id = match &entry.entry_id {
                Some(id) => id,
                None => continue,
            };
            if !expected_entry_types
                .iter()
                .any(|prefix| entry_id.starts_with(prefix))
            {
                continue;
            }

            if let Some(ref item_content) = content.item_content {
                parse_and_push(&mut tweets, item_content, entry_id.clone(), false);
            }

            if let Some(items) = &content.items {
                for item in items {
                    if let Some(item) = &item.item {
                        if let Some(item_content) = &item.item_content {
                            parse_and_push(&mut tweets, item_content, entry_id.clone(), false);
                        }
                    }
                }
            }
        }
    }

    QueryTweetsResponse {
        tweets,
        next: bottom_cursor,
        previous: top_cursor,
    }
}

pub fn parse_threaded_conversation(conversation: &ThreadedConversation) -> Option<Tweet> {
    let mut main_tweet: Option<Tweet> = None;
    let mut replies: Vec<Tweet> = Vec::new();

    let instructions = conversation
        .data
        .as_ref()
        .and_then(|data| data.threaded_conversation_with_injections_v2.as_ref())
        .and_then(|conv| conv.instructions.as_ref())
        .unwrap_or(&EMPTY_INSTRUCTIONS);

    for instruction in instructions {
        let entries = instruction
            .entries.as_deref()
            .unwrap_or_default();

        for entry in entries {
            if let Some(content) = &entry.content {
                if let Some(item_content) = &content.item_content {
                    if let Some(tweet) = parse_timeline_entry_item_content_raw(
                        item_content,
                        entry.entry_id.as_deref().unwrap_or_default(),
                        true,
                    ) {
                        if main_tweet.is_none() {
                            main_tweet = Some(tweet);
                        } else {
                            replies.push(tweet);
                        }
                    }
                }

                if let Some(items) = &content.items {
                    for item in items {
                        if let Some(item) = &item.item {
                            if let Some(item_content) = &item.item_content {
                                if let Some(tweet) = parse_timeline_entry_item_content_raw(
                                    item_content,
                                    entry.entry_id.as_deref().unwrap_or_default(),
                                    true,
                                ) {
                                    replies.push(tweet);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(mut main_tweet) = main_tweet {
        for reply in &replies {
            if let Some(reply_id) = &reply.in_reply_to_status_id {
                if let Some(main_id) = &main_tweet.id {
                    if reply_id == main_id {
                        main_tweet.replies = Some(replies.len() as i32);
                        break;
                    }
                }
            }
        }

        if main_tweet.is_self_thread == Some(true) {
            let thread = replies
                .iter()
                .filter(|t| t.is_self_thread == Some(true))
                .cloned()
                .collect::<Vec<_>>();

            if thread.is_empty() {
                main_tweet.is_self_thread = Some(false);
            } else {
                main_tweet.thread = thread;
            }
        }

        // main_tweet.html = reconstruct_tweet_html(&main_tweet);

        Some(main_tweet)
    } else {
        None
    }
}
