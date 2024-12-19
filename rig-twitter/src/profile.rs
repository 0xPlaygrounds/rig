use crate::api::requests::request_api;
use crate::auth::user_auth::TwitterAuth;
use crate::error::{Result, TwitterError};
use crate::models::Profile;
use chrono::{DateTime, Utc};
use lazy_static::lazy_static;
use reqwest::header::HeaderMap;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Mutex;
use reqwest::Client;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub id: String,
    pub id_str: String,
    pub name: String,
    pub screen_name: String,
    pub location: Option<String>,
    pub description: Option<String>,
    pub url: Option<String>,
    pub protected: bool,
    pub followers_count: i32,
    pub friends_count: i32,
    pub listed_count: i32,
    pub created_at: String,
    pub favourites_count: i32,
    pub verified: bool,
    pub statuses_count: i32,
    pub profile_image_url_https: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyUserRaw {
    pub created_at: Option<String>,
    pub description: Option<String>,
    pub entities: Option<UserEntitiesRaw>,
    pub favourites_count: Option<i32>,
    pub followers_count: Option<i32>,
    pub friends_count: Option<i32>,
    pub media_count: Option<i32>,
    pub statuses_count: Option<i32>,
    pub id_str: Option<String>,
    pub listed_count: Option<i32>,
    pub name: Option<String>,
    pub location: String,
    pub geo_enabled: Option<bool>,
    pub pinned_tweet_ids_str: Option<Vec<String>>,
    pub profile_background_color: Option<String>,
    pub profile_banner_url: Option<String>,
    pub profile_image_url_https: Option<String>,
    pub protected: Option<bool>,
    pub screen_name: Option<String>,
    pub verified: Option<bool>,
    pub has_custom_timelines: Option<bool>,
    pub has_extended_profile: Option<bool>,
    pub url: Option<String>,
    pub can_dm: Option<bool>,
    #[serde(rename = "userId")]
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserEntitiesRaw {
    pub url: Option<UserUrlEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserUrlEntity {
    pub urls: Option<Vec<ExpandedUrl>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedUrl {
    pub expanded_url: Option<String>,
}

lazy_static! {
    static ref ID_CACHE: Mutex<HashMap<String, String>> = Mutex::new(HashMap::new());
}

pub fn parse_profile(user: &LegacyUserRaw, is_blue_verified: Option<bool>) -> Profile {
    let mut profile = Profile {
        id: user.user_id.clone().unwrap_or_default(),
        username: user.screen_name.clone().unwrap_or_default(),
        name: user.name.clone().unwrap_or_default(),
        description: user.description.clone(),
        location: Some(user.location.clone()),
        url: user.url.clone(),
        protected: user.protected.unwrap_or(false),
        verified: user.verified.unwrap_or(false),
        followers_count: user.followers_count.unwrap_or(0),
        following_count: user.friends_count.unwrap_or(0),
        tweets_count: user.statuses_count.unwrap_or(0),
        listed_count: user.listed_count.unwrap_or(0),
        is_blue_verified: Some(is_blue_verified.unwrap_or(false)),
        created_at: user
            .created_at
            .as_ref()
            .and_then(|date_str| {
                DateTime::parse_from_str(date_str, "%a %b %d %H:%M:%S %z %Y")
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            })
            .unwrap_or_else(Utc::now),
        profile_image_url: user
            .profile_image_url_https
            .as_ref()
            .map(|url| url.replace("_normal", "")),
        profile_banner_url: user.profile_banner_url.clone(),
        pinned_tweet_id: user
            .pinned_tweet_ids_str
            .as_ref()
            .and_then(|ids| ids.first().cloned()),
    };

    // Set website URL from entities using functional chaining
    user.entities
        .as_ref()
        .and_then(|entities| entities.url.as_ref())
        .and_then(|url_entity| url_entity.urls.as_ref())
        .and_then(|urls| urls.first())
        .and_then(|first_url| first_url.expanded_url.as_ref())
        .map(|expanded_url| profile.url = Some(expanded_url.clone()));

    profile
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserResults {
    pub result: UserResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "__typename")]
pub enum UserResult {
    User(UserData),
    UserUnavailable(UserUnavailable),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserData {
    pub id: String,
    pub rest_id: String,
    pub affiliates_highlighted_label: Option<serde_json::Value>,
    pub has_graduated_access: bool,
    pub is_blue_verified: bool,
    pub profile_image_shape: String,
    pub legacy: LegacyUserRaw,
    pub smart_blocked_by: bool,
    pub smart_blocking: bool,
    pub legacy_extended_profile: Option<serde_json::Value>,
    pub is_profile_translatable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserUnavailable {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRaw {
    pub data: UserRawData,
    pub errors: Option<Vec<TwitterApiErrorRaw>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRawData {
    pub user: UserRawUser,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRawUser {
    pub result: UserRawResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRawResult {
    pub rest_id: Option<String>,
    pub is_blue_verified: Option<bool>,
    pub legacy: LegacyUserRaw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwitterApiErrorRaw {
    pub message: String,
    pub code: i32,
}

pub async fn get_profile(client: &Client, auth: &dyn TwitterAuth,screen_name: &str) -> Result<Profile> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let variables = json!({
        "screen_name": screen_name,
        "withSafetyModeUserFields": true
    });

    let features = json!({
        "hidden_profile_likes_enabled": false,
        "hidden_profile_subscriptions_enabled": false,
        "responsive_web_graphql_exclude_directive_enabled": true,
        "verified_phone_label_enabled": false,
        "subscriptions_verification_info_is_identity_verified_enabled": false,
        "subscriptions_verification_info_verified_since_enabled": true,
        "highlights_tweets_tab_ui_enabled": true,
        "creator_subscriptions_tweet_preview_api_enabled": true,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": false,
        "responsive_web_graphql_timeline_navigation_enabled": true
    });

    let field_toggles = json!({
        "withAuxiliaryUserLabels": false
    });

    let (response, _) = request_api::<UserRaw>(
        client,
        "https://twitter.com/i/api/graphql/G3KGOASz96M-Qu0nwmGXNg/UserByScreenName",
        headers,
        Method::GET,
        Some(json!({
            "variables": variables,
            "features": features,
            "fieldToggles": field_toggles
        })),
    )
    .await?;

    if let Some(errors) = response.errors {
        if !errors.is_empty() {
            return Err(TwitterError::Api(errors[0].message.clone()));
        }
    }
    let user_raw_result = &response.data.user.result;
    let mut legacy = user_raw_result.legacy.clone();
    let rest_id = user_raw_result.rest_id.clone();
    let is_blue_verified = user_raw_result.is_blue_verified;
    legacy.user_id = rest_id;
    if legacy.screen_name.is_none() || legacy.screen_name.as_ref().unwrap().is_empty() {
        return Err(TwitterError::Api(format!(
            "Either {} does not exist or is private.",
            screen_name
        )));
    }
    Ok(parse_profile(&legacy, is_blue_verified))
}

pub async fn get_screen_name_by_user_id(client: &Client, auth: &dyn TwitterAuth,user_id: &str) -> Result<String> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let variables = json!({
        "userId": user_id,
        "withSafetyModeUserFields": true
    });

    let features = json!({
        "hidden_profile_subscriptions_enabled": true,
        "rweb_tipjar_consumption_enabled": true,
        "responsive_web_graphql_exclude_directive_enabled": true,
        "verified_phone_label_enabled": false,
        "highlights_tweets_tab_ui_enabled": true,
        "responsive_web_twitter_article_notes_tab_enabled": true,
        "subscriptions_feature_can_gift_premium": false,
        "creator_subscriptions_tweet_preview_api_enabled": true,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": false,
        "responsive_web_graphql_timeline_navigation_enabled": true
    });

    let (response, _) = request_api::<UserRaw>(
        client,
        "https://twitter.com/i/api/graphql/xf3jd90KKBCUxdlI_tNHZw/UserByRestId",
        headers,
        Method::GET,
        Some(json!({
            "variables": variables,
            "features": features
        })),
    )
    .await?;

    if let Some(errors) = response.errors {
        if !errors.is_empty() {
            return Err(TwitterError::Api(errors[0].message.clone()));
        }
    }

    if let Some(user) = response.data.user.result.legacy.screen_name {
        Ok(user)
    } else {
        Err(TwitterError::Api(format!(
            "Either user with ID {} does not exist or is private.",
            user_id
        )))
    }
}

pub async fn get_user_id_by_screen_name(
    client: &Client,
    auth: &dyn TwitterAuth,
    screen_name: &str,
) -> Result<String> {
    if let Some(cached_id) = ID_CACHE.lock().unwrap().get(screen_name) {
        return Ok(cached_id.clone());
    }

    let profile = get_profile(client, auth, screen_name).await?;
    if let Some(user_id) = Some(profile.id) {
        ID_CACHE
            .lock()
            .unwrap()
            .insert(screen_name.to_string(), user_id.clone());
        Ok(user_id)
    } else {
        Err(TwitterError::Api("User ID is undefined".into()))
    }
}
