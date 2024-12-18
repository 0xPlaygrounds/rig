use crate::api::requests::request_api;
use crate::api::requests::request_form_api;
use crate::auth::user_auth::TwitterAuth;
use crate::error::{Result, TwitterError};
use crate::models::Profile;
use crate::timeline::v1::QueryProfilesResponse;
use chrono::{DateTime, Utc};
use reqwest::Method;
use serde::Deserialize;
use serde_json::{json, Value};
use reqwest::Client;
#[derive(Debug, Deserialize)]
pub struct RelationshipResponse {
    pub data: Option<RelationshipData>,
    #[serde(skip)]
    pub errors: Option<Vec<TwitterError>>,
}

#[derive(Debug, Deserialize)]
pub struct RelationshipData {
    pub user: UserRelationships,
}

#[derive(Debug, Deserialize)]
pub struct UserRelationships {
    pub result: UserResult,
}

#[derive(Debug, Deserialize)]
pub struct UserResult {
    pub timeline: Timeline,
    pub rest_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Timeline {
    pub timeline: TimelineData,
}

#[derive(Debug, Deserialize)]
pub struct TimelineData {
    pub instructions: Vec<TimelineInstruction>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum TimelineInstruction {
    #[serde(rename = "TimelineAddEntries")]
    AddEntries { entries: Vec<TimelineEntry> },
    #[serde(rename = "TimelineReplaceEntry")]
    ReplaceEntry { entry: TimelineEntry },
}

#[derive(Debug, Deserialize)]
pub struct TimelineEntry {
    pub content: EntryContent,
    pub entry_id: String,
    pub sort_index: String,
}

#[derive(Debug, Deserialize)]
pub struct EntryContent {
    #[serde(rename = "itemContent")]
    pub item_content: Option<ItemContent>,
    pub cursor: Option<CursorContent>,
}

#[derive(Debug, Deserialize)]
pub struct ItemContent {
    #[serde(rename = "user_results")]
    pub user_results: Option<UserResults>,
    #[serde(rename = "userDisplayType")]
    pub user_display_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UserResults {
    pub result: UserResultData,
}

#[derive(Debug, Deserialize)]
pub struct UserResultData {
    #[serde(rename = "typename")]
    pub type_name: Option<String>,
    #[serde(rename = "mediaColor")]
    pub media_color: Option<MediaColor>,
    pub id: Option<String>,
    pub rest_id: Option<String>,
    pub affiliates_highlighted_label: Option<Value>,
    pub has_graduated_access: Option<bool>,
    pub is_blue_verified: Option<bool>,
    pub profile_image_shape: Option<String>,
    pub legacy: Option<UserLegacy>,
    pub professional: Option<Professional>,
}

#[derive(Debug, Deserialize)]
pub struct MediaColor {
    pub r: Option<ColorPalette>,
}

#[derive(Debug, Deserialize)]
pub struct ColorPalette {
    pub ok: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct UserLegacy {
    pub following: Option<bool>,
    pub followed_by: Option<bool>,
    pub screen_name: Option<String>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub location: Option<String>,
    pub url: Option<String>,
    pub protected: Option<bool>,
    pub verified: Option<bool>,
    pub followers_count: Option<i32>,
    pub friends_count: Option<i32>,
    pub statuses_count: Option<i32>,
    pub listed_count: Option<i32>,
    pub created_at: Option<String>,
    pub profile_image_url_https: Option<String>,
    pub profile_banner_url: Option<String>,
    pub pinned_tweet_ids_str: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Professional {
    pub rest_id: Option<String>,
    pub professional_type: Option<String>,
    pub category: Option<Vec<ProfessionalCategory>>,
}

#[derive(Debug, Deserialize)]
pub struct ProfessionalCategory {
    pub id: i64,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct CursorContent {
    pub value: String,
    pub cursor_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RelationshipTimeline {
    pub data: Option<RelationshipTimelineData>,
    pub errors: Option<Vec<TwitterError>>,
}

#[derive(Debug, Deserialize)]
pub struct RelationshipTimelineData {
    pub user: UserData,
}

#[derive(Debug, Deserialize)]
pub struct UserData {
    pub result: RelationshipUserResult,
}

#[derive(Debug, Deserialize)]
pub struct RelationshipUserResult {
    pub timeline: Timeline,
}

#[derive(Debug, Deserialize)]
pub struct InnerTimeline {
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Instruction {
    #[serde(rename = "TimelineAddEntries")]
    AddEntries {
        entries: Vec<RelationshipTimelineEntry>,
    },
    #[serde(rename = "TimelineReplaceEntry")]
    ReplaceEntry { entry: RelationshipTimelineEntry },
}

#[derive(Debug, Deserialize)]
pub struct RelationshipTimelineEntry {
    pub content: EntryContent,
    pub entry_id: String,
    pub sort_index: String,
}

#[derive(Debug, Deserialize)]
pub struct RelationshipTimelineContainer {
    pub timeline: InnerTimeline,
}

#[derive(Debug, Deserialize)]
pub struct RelationshipTimelineWrapper {
    pub timeline: InnerTimeline,
}
pub async fn get_following(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    count: i32,
    cursor: Option<String>,
) -> Result<(Vec<Profile>, Option<String>)> {
    let response = fetch_profile_following(client, auth, user_id, count, cursor).await?;
    Ok((response.profiles, response.next))
}
pub async fn get_followers(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    count: i32,
    cursor: Option<String>,
) -> Result<(Vec<Profile>, Option<String>)> {
    let response = fetch_profile_following(client, auth, user_id, count, cursor).await?;
    Ok((response.profiles, response.next))
}

pub async fn fetch_profile_following(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    max_profiles: i32,
    cursor: Option<String>,
) -> Result<QueryProfilesResponse> {
    let timeline = get_following_timeline(client, auth, user_id, max_profiles, cursor).await?;

    Ok(parse_relationship_timeline(&timeline))
}

async fn get_following_timeline(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    max_items: i32,
    cursor: Option<String>,
) -> Result<RelationshipTimeline> {

    let count = if max_items > 50 { 50 } else { max_items };

    let mut variables = json!({
        "userId": user_id,
        "count": count,
        "includePromotedContent": false,
    });

    if let Some(cursor_val) = cursor {
        if !cursor_val.is_empty() {
            variables["cursor"] = json!(cursor_val);
        }
    }

    let features = json!({
        "responsive_web_twitter_article_tweet_consumption_enabled": false,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": true,
        "longform_notetweets_inline_media_enabled": true,
        "responsive_web_media_download_video_enabled": false,
    });

    let url = format!(
        "https://twitter.com/i/api/graphql/iSicc7LrzWGBgDPL0tM_TQ/Following?variables={}&features={}",
        urlencoding::encode(&variables.to_string()),
        urlencoding::encode(&features.to_string())
    );

    let mut headers = reqwest::header::HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let (_data, _) = request_api::<RelationshipTimeline>(client, &url, headers, Method::GET, None).await?;

    Ok(_data)
}

fn parse_relationship_timeline(timeline: &RelationshipTimeline) -> QueryProfilesResponse {
    let mut profiles = Vec::new();
    let mut next_cursor = None;
    let mut previous_cursor = None;

    if let Some(data) = &timeline.data {
        for instruction in &data.user.result.timeline.timeline.instructions {
            match instruction {
                TimelineInstruction::AddEntries { entries } => {
                    for entry in entries {
                        if let Some(item_content) = &entry.content.item_content {
                            if let Some(user_results) = &item_content.user_results {
                                if let Some(legacy) = &user_results.result.legacy {
                                    let profile = Profile {
                                        username: legacy.screen_name.clone().unwrap_or_default(),
                                        name: legacy.name.clone().unwrap_or_default(),
                                        id: user_results
                                            .result
                                            .rest_id
                                            .as_ref()
                                            .map(String::from)
                                            .unwrap_or_default(),
                                        description: legacy.description.clone(),
                                        location: legacy.location.clone(),
                                        url: legacy.url.clone(),
                                        protected: legacy.protected.unwrap_or_default(),
                                        verified: legacy.verified.unwrap_or_default(),
                                        followers_count: legacy.followers_count.unwrap_or_default(),
                                        following_count: legacy.friends_count.unwrap_or_default(),
                                        tweets_count: legacy.statuses_count.unwrap_or_default(),
                                        listed_count: legacy.listed_count.unwrap_or_default(),
                                        created_at: legacy
                                            .created_at
                                            .as_ref()
                                            .and_then(|date| {
                                                DateTime::parse_from_str(
                                                    date,
                                                    "%a %b %d %H:%M:%S %z %Y",
                                                )
                                                .ok()
                                                .map(|dt| dt.with_timezone(&Utc))
                                            })
                                            .unwrap_or_default(),
                                        profile_image_url: legacy.profile_image_url_https.clone(),
                                        profile_banner_url: legacy.profile_banner_url.clone(),
                                        pinned_tweet_id: legacy.pinned_tweet_ids_str.clone(),
                                        is_blue_verified: Some(
                                            user_results.result.is_blue_verified.unwrap_or(false),
                                        ),
                                    };

                                    profiles.push(profile);
                                }
                            }
                        } else if let Some(cursor_content) = &entry.content.cursor {
                            match cursor_content.cursor_type.as_deref() {
                                Some("Bottom") => next_cursor = Some(cursor_content.value.clone()),
                                Some("Top") => previous_cursor = Some(cursor_content.value.clone()),
                                _ => {}
                            }
                        }
                    }
                }
                TimelineInstruction::ReplaceEntry { entry } => {
                    if let Some(cursor_content) = &entry.content.cursor {
                        match cursor_content.cursor_type.as_deref() {
                            Some("Bottom") => next_cursor = Some(cursor_content.value.clone()),
                            Some("Top") => previous_cursor = Some(cursor_content.value.clone()),
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    QueryProfilesResponse {
        profiles,
        next: next_cursor,
        previous: previous_cursor,
    }
}

pub async fn follow_user(client: &Client, auth: &dyn TwitterAuth, username: &str) -> Result<()> {
    let user_id = crate::profile::get_user_id_by_screen_name(client, auth, username).await?;

    let url = "https://api.twitter.com/1.1/friendships/create.json";

    let form = vec![
        (
            "include_profile_interstitial_type".to_string(),
            "1".to_string(),
        ),
        ("skip_status".to_string(), "true".to_string()),
        ("user_id".to_string(), user_id),
    ];

    let mut headers = reqwest::header::HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    headers.insert(
        "Content-Type",
        "application/x-www-form-urlencoded".parse().unwrap(),
    );
    headers.insert(
        "Referer",
        format!("https://twitter.com/{}", username).parse().unwrap(),
    );
    headers.insert("X-Twitter-Active-User", "yes".parse().unwrap());
    headers.insert("X-Twitter-Auth-Type", "OAuth2Session".parse().unwrap());
    headers.insert("X-Twitter-Client-Language", "en".parse().unwrap());

    let (_, _) = request_form_api::<Value>(client, url, headers, form).await?;

    Ok(())
}

pub async fn unfollow_user(client: &Client, auth: &dyn TwitterAuth, username: &str) -> Result<()> {

    let user_id = crate::profile::get_user_id_by_screen_name(client, auth, username).await?;

    let url = "https://api.twitter.com/1.1/friendships/destroy.json";

    let form = vec![
        (
            "include_profile_interstitial_type".to_string(),
            "1".to_string(),
        ),
        ("skip_status".to_string(), "true".to_string()),
        ("user_id".to_string(), user_id),
    ];

    let mut headers = reqwest::header::HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    headers.insert(
        "Content-Type",
        "application/x-www-form-urlencoded".parse().unwrap(),
    );
    headers.insert(
        "Referer",
        format!("https://twitter.com/{}", username).parse().unwrap(),
    );
    headers.insert("X-Twitter-Active-User", "yes".parse().unwrap());
    headers.insert("X-Twitter-Auth-Type", "OAuth2Session".parse().unwrap());
    headers.insert("X-Twitter-Client-Language", "en".parse().unwrap());

    let (_, _) = request_form_api::<Value>(client, url, headers, form).await?;

    Ok(())
}
