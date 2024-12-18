use crate::api::requests::request_api;
use crate::auth::user_auth::TwitterAuth;
use crate::error::Result;
use serde::Deserialize;
use serde_json::Value;
use urlencoding;
use reqwest::Client;
use reqwest::header::HeaderMap;
use reqwest::Method;

#[derive(Debug, Deserialize)]
pub struct HomeTimelineResponse {
    pub data: Option<HomeData>,
}

#[derive(Debug, Deserialize)]
pub struct HomeData {
    pub home: Home,
}

#[derive(Debug, Deserialize)]
pub struct Home {
    #[serde(rename = "home_timeline_urt")]
    pub home_timeline: HomeTimeline,
}

#[derive(Debug, Deserialize)]
pub struct HomeTimeline {
    pub instructions: Vec<TimelineInstruction>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum TimelineInstruction {
    #[serde(rename = "TimelineAddEntries")]
    AddEntries { entries: Vec<TimelineEntry> },
    // Add other variants as needed
}

#[derive(Debug, Deserialize)]
pub struct TimelineEntry {
    pub content: EntryContent,
}

#[derive(Debug, Deserialize)]
pub struct EntryContent {
    #[serde(rename = "itemContent")]
    pub item_content: Option<ItemContent>,
}

#[derive(Debug, Deserialize)]
pub struct ItemContent {
    pub tweet_results: Option<TweetResults>,
}

#[derive(Debug, Deserialize)]
pub struct TweetResults {
    pub result: Option<Value>,
}

pub async fn fetch_home_timeline(
    client: &Client,
    auth: &dyn TwitterAuth,
    count: i32,
    seen_tweet_ids: Vec<String>,
) -> Result<Vec<Value>> {
    let variables = serde_json::json!({
        "count": count,
        "includePromotedContent": true,
        "latestControlAvailable": true,
        "requestContext": "launch",
        "withCommunity": true,
        "seenTweetIds": seen_tweet_ids,
    });

    let features = serde_json::json!({
        "rweb_tipjar_consumption_enabled": true,
        "responsive_web_graphql_exclude_directive_enabled": true,
        "verified_phone_label_enabled": false,
        "creator_subscriptions_tweet_preview_api_enabled": true,
        "responsive_web_graphql_timeline_navigation_enabled": true,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": false,
        "communities_web_enable_tweet_community_results_fetch": true,
        "c9s_tweet_anatomy_moderator_badge_enabled": true,
        "articles_preview_enabled": true,
        "responsive_web_edit_tweet_api_enabled": true,
        "graphql_is_translatable_rweb_tweet_is_translatable_enabled": true,
        "view_counts_everywhere_api_enabled": true,
        "longform_notetweets_consumption_enabled": true,
        "responsive_web_twitter_article_tweet_consumption_enabled": true,
        "tweet_awards_web_tipping_enabled": false,
        "creator_subscriptions_quote_tweet_preview_enabled": false,
        "freedom_of_speech_not_reach_fetch_enabled": true,
        "standardized_nudges_misinfo": true,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": true,
        "rweb_video_timestamps_enabled": true,
        "longform_notetweets_rich_text_read_enabled": true,
        "longform_notetweets_inline_media_enabled": true,
        "responsive_web_enhance_cards_enabled": false,
    });

    let url = format!(
        "https://x.com/i/api/graphql/HJFjzBgCs16TqxewQOeLNg/HomeTimeline?variables={}&features={}",
        urlencoding::encode(&variables.to_string()),
        urlencoding::encode(&features.to_string())
    );

    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let (response, _) = request_api::<HomeTimelineResponse>(
        client,
        &url,
        headers,
        Method::GET,
        None,
    ).await?;

    let home = response
        .data.map(|data| data.home.home_timeline.instructions);

    let mut entries = Vec::new();

    if let Some(instructions) = home {
        for instruction in instructions {
            match instruction {
                TimelineInstruction::AddEntries {
                    entries: new_entries,
                } => {
                    for entry in new_entries {
                        if let Some(item_content) = entry.content.item_content {
                            if let Some(tweet_results) = item_content.tweet_results {
                                if let Some(result) = tweet_results.result {
                                    entries.push(result);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(entries)
}
