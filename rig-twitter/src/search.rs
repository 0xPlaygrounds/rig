use crate::api::requests::request_api;
use crate::auth::user_auth::TwitterAuth;
use crate::error::Result;
use crate::timeline::search::{
    parse_search_timeline_tweets, parse_search_timeline_users, SearchTimeline,
};
use crate::timeline::v1::{QueryProfilesResponse, QueryTweetsResponse};
use reqwest::Method;
use serde_json::json;
use reqwest::Client;
#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Top,
    Latest,
    Photos,
    Videos,
    Users,
}

pub async fn fetch_search_tweets(
    client: &Client,
    auth: &dyn TwitterAuth,
    query: &str,
    max_tweets: i32,
    search_mode: SearchMode,
    cursor: Option<String>,
) -> Result<QueryTweetsResponse> {
    let timeline = get_search_timeline(client, query, max_tweets, search_mode, auth, cursor).await?;

    Ok(parse_search_timeline_tweets(&timeline))
}

pub async fn search_profiles(
    client: &Client,
    auth: &dyn TwitterAuth,
    query: &str,
    max_profiles: i32,
    cursor: Option<String>,
) -> Result<QueryProfilesResponse> {
    let timeline =
        get_search_timeline(client, query, max_profiles, SearchMode::Users, auth, cursor).await?;

    Ok(parse_search_timeline_users(&timeline))
}

async fn get_search_timeline(
    client: &Client, 
    query: &str,
    max_items: i32,
    search_mode: SearchMode,
    auth: &dyn TwitterAuth,
    _cursor: Option<String>,
) -> Result<SearchTimeline> {

    let max_items = if max_items > 50 { 50 } else { max_items };

    let mut variables = json!({
        "rawQuery": query,
        "count": max_items,
        "querySource": "typed_query",
        "product": "Top"
    });

    // Set product based on search mode
    match search_mode {
        SearchMode::Latest => {
            variables["product"] = json!("Latest");
        }
        SearchMode::Photos => {
            variables["product"] = json!("Photos");
        }
        SearchMode::Videos => {
            variables["product"] = json!("Videos");
        }
        SearchMode::Users => {
            variables["product"] = json!("People");
        }
        _ => {}
    }

    let features = json!({
        "longform_notetweets_inline_media_enabled": true,
        "responsive_web_enhance_cards_enabled": false,
        "responsive_web_media_download_video_enabled": false,
        "responsive_web_twitter_article_tweet_consumption_enabled": false,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": true,
        "interactive_text_enabled": false,
        "responsive_web_text_conversations_enabled": false,
        "vibe_api_enabled": false,
        "rweb_lists_timeline_redesign_enabled": true,
        "responsive_web_graphql_exclude_directive_enabled": true,
        "verified_phone_label_enabled": false,
        "creator_subscriptions_tweet_preview_api_enabled": true,
        "responsive_web_graphql_timeline_navigation_enabled": true,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": false,
        "tweetypie_unmention_optimization_enabled": true,
        "responsive_web_edit_tweet_api_enabled": true,
        "graphql_is_translatable_rweb_tweet_is_translatable_enabled": true,
        "view_counts_everywhere_api_enabled": true,
        "longform_notetweets_consumption_enabled": true,
        "tweet_awards_web_tipping_enabled": false,
        "freedom_of_speech_not_reach_fetch_enabled": true,
        "standardized_nudges_misinfo": true,
        "longform_notetweets_rich_text_read_enabled": true,
        "responsive_web_enhance_cards_enabled": false,
        "subscriptions_verification_info_enabled": true,
        "subscriptions_verification_info_reason_enabled": true,
        "subscriptions_verification_info_verified_since_enabled": true,
        "super_follow_badge_privacy_enabled": false,
        "super_follow_exclusive_tweet_notifications_enabled": false,
        "super_follow_tweet_api_enabled": false,
        "super_follow_user_api_enabled": false,
        "android_graphql_skip_api_media_color_palette": false,
        "creator_subscriptions_subscription_count_enabled": false,
        "blue_business_profile_image_shape_enabled": false,
        "unified_cards_ad_metadata_container_dynamic_card_content_query_enabled": false
    });

    let field_toggles = json!({
        "withArticleRichContentState": false
    });

    let params = [("variables", serde_json::to_string(&variables)?),
        ("features", serde_json::to_string(&features)?),
        ("fieldToggles", serde_json::to_string(&field_toggles)?)];

    let query_string = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoding::encode(v)))
        .collect::<Vec<_>>()
        .join("&");

    let mut headers = reqwest::header::HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let url = format!(
        "https://api.twitter.com/graphql/gkjsKepM6gl_HmFWoWKfgg/SearchTimeline?{}",
        query_string
    );

    let (response, _) = request_api::<SearchTimeline>(client, &url, headers, Method::GET, None).await?;

    Ok(response)
}
