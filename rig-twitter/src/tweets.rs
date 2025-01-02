use crate::api::endpoints::Endpoints;
use crate::api::requests::{request_api, request_multipart_api};
use crate::auth::user_auth::TwitterAuth;
use crate::error::{Result, TwitterError};
use crate::models::tweets::Tweet;
use crate::profile::get_user_id_by_screen_name;
use crate::timeline::v2::parse_threaded_conversation;
use crate::timeline::v2::parse_timeline_tweets_v2;
use crate::timeline::v2::QueryTweetsResponse;
use crate::timeline::v2::ThreadedConversation;
use reqwest::header::HeaderMap;
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use reqwest::Client;

pub const DEFAULT_EXPANSIONS: &[&str] = &[
    "attachments.poll_ids",
    "attachments.media_keys",
    "author_id",
    "referenced_tweets.id",
    "in_reply_to_user_id",
    "edit_history_tweet_ids",
    "geo.place_id",
    "entities.mentions.username",
    "referenced_tweets.id.author_id",
];

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

pub async fn fetch_tweets(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    max_tweets: i32,
    cursor: Option<&str>,
) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let mut variables = json!({
        "userId": user_id,
        "count": max_tweets.min(200),
        "includePromotedContent": false
    });

    if let Some(cursor_val) = cursor {
        variables["cursor"] = json!(cursor_val);
    }

    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/YNXM2DGuE2Sff6a2JD3Ztw/UserTweets",
        headers,
        Method::GET,
        Some(json!({
            "variables": variables,
            "features": get_default_features()
        })),
    )
    .await?;

    Ok(value)
}

pub async fn fetch_tweets_and_replies(  
    client: &Client,
    auth: &dyn TwitterAuth,
    username: &str,
    max_tweets: i32,
    cursor: Option<&str>,
) -> Result<QueryTweetsResponse> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let user_id = get_user_id_by_screen_name(client, auth, username).await?;

    let endpoint = Endpoints::user_tweets_and_replies(&user_id, max_tweets.min(40), cursor);

    let (value, _headers) =
        request_api(client, &endpoint.to_request_url(), headers, Method::GET, None).await?;

    let parsed_response = parse_timeline_tweets_v2(&value);
    Ok(parsed_response)
}

pub async fn fetch_tweets_and_replies_by_user_id(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    max_tweets: i32,
    cursor: Option<&str>,
) -> Result<QueryTweetsResponse> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let endpoint = Endpoints::user_tweets_and_replies(user_id, max_tweets.min(40), cursor);

    let (value, _headers) =
        request_api(client, &endpoint.to_request_url(), headers, Method::GET, None).await?;

    let parsed_response = parse_timeline_tweets_v2(&value);
    Ok(parsed_response)
}

pub async fn fetch_list_tweets(
    client: &Client,
    auth: &dyn TwitterAuth,
    list_id: &str,
    max_tweets: i32,
    cursor: Option<&str>,
) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let mut variables = json!({
        "listId": list_id,
        "count": max_tweets.min(200)
    });

    if let Some(cursor_val) = cursor {
        variables["cursor"] = json!(cursor_val);
    }

    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/LFKj1wqHNTsEJ4Oq7TzaNA/ListLatestTweetsTimeline",
        headers,
        Method::GET,
        Some(json!({
            "variables": variables,
            "features": get_default_features()
        })),
    )
    .await?;

    Ok(value)
}

pub async fn create_quote_tweet(
    client: &Client,
    auth: &dyn TwitterAuth,
    text: &str,
    quoted_tweet_id: &str,
    media_data: Option<Vec<(Vec<u8>, String)>>,
) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let mut variables = json!({
        "tweet_text": text,
        "dark_request": false,
        "attachment_url": format!("https://twitter.com/twitter/status/{}", quoted_tweet_id),
        "media": {
            "media_entities": [],
            "possibly_sensitive": false
        },
        "semantic_annotation_ids": []
    });

    if let Some(media_files) = media_data {
        let mut media_entities = Vec::new();

        for (file_data, media_type) in media_files {
            let media_id = upload_media(client, auth, file_data, &media_type).await?;
            media_entities.push(json!({
                "media_id": media_id,
                "tagged_users": []
            }));
        }

        variables["media"]["media_entities"] = json!(media_entities);
    }

    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/a1p9RWpkYKBjWv_I3WzS-A/CreateTweet",
        headers,
        Method::POST,
        Some(json!({
            "variables": variables,
            "features": create_quote_tweet_features()
        })),
    )
    .await?;

    Ok(value)
}

pub async fn like_tweet(client: &Client, auth: &dyn TwitterAuth, tweet_id: &str) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/lI07N6Otwv1PhnEgXILM7A/FavoriteTweet",
        headers,
        Method::POST,
        Some(json!({
            "variables": {
                "tweet_id": tweet_id
            }
        })),
    )
    .await?;

    Ok(value)
}

pub async fn retweet(client: &Client, auth: &dyn TwitterAuth, tweet_id: &str) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/ojPdsZsimiJrUGLR1sjUtA/CreateRetweet",
        headers,
        Method::POST,
        Some(json!({
            "variables": {
                "tweet_id": tweet_id,
                "dark_request": false
            }
        })),
    )
    .await?;

    Ok(value)
}

pub async fn create_long_tweet(
    client: &Client,
    auth: &dyn TwitterAuth,
    text: &str,
    reply_to: Option<&str>,
    media_ids: Option<Vec<String>>,
) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let mut variables = json!({
        "tweet_text": text,
        "dark_request": false,
        "media": {
            "media_entities": [],
            "possibly_sensitive": false
        },
        "semantic_annotation_ids": []
    });

    if let Some(reply_id) = reply_to {
        variables["reply"] = json!({
            "in_reply_to_tweet_id": reply_id
        });
    }

    if let Some(media) = media_ids {
        variables["media"]["media_entities"] = json!(media
            .iter()
            .map(|id| json!({
                "media_id": id,
                "tagged_users": []
            }))
            .collect::<Vec<_>>());
    }

    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/YNXM2DGuE2Sff6a2JD3Ztw/CreateNoteTweet",
        headers,
        Method::POST,
        Some(json!({
            "variables": variables,
            "features": get_long_tweet_features()
        })),
    )
    .await?;

    Ok(value)
}

pub async fn fetch_liked_tweets(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    max_tweets: i32,
    cursor: Option<&str>,
) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let mut variables = json!({
        "userId": user_id,
        "count": max_tweets.min(200),
        "includePromotedContent": false
    });

    if let Some(cursor_val) = cursor {
        variables["cursor"] = json!(cursor_val);
    }

    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/YlkSUg4Czo2Zx7yRqpwDow/Likes",
        headers,
        Method::GET,
        Some(json!({
            "variables": variables,
            "features": get_default_features()
        })),
    )
    .await?;

    Ok(value)
}
pub async fn upload_media(
    client: &Client,
    auth: &dyn TwitterAuth,
    file_data: Vec<u8>,
    media_type: &str,
) -> Result<String> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let upload_url = "https://upload.twitter.com/1.1/media/upload.json";

    // Check if media is video
    let is_video = media_type.starts_with("video/");

    if is_video {
        // Handle video upload using chunked upload
        upload_video_in_chunks(client, file_data, media_type, headers).await
    } else {
        // Handle image upload directly
        let form = reqwest::multipart::Form::new()
            .part("media", reqwest::multipart::Part::bytes(file_data));

        let (response, _) = request_multipart_api::<Value>(client, upload_url, headers, form).await?;

        response["media_id_string"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| TwitterError::Api("Failed to get media_id".into()))
    }
}

async fn upload_video_in_chunks(
    client: &Client,
    file_data: Vec<u8>,
    media_type: &str,
    headers: HeaderMap,
) -> Result<String> {
    let upload_url = "https://upload.twitter.com/1.1/media/upload.json";

    // INIT command
    let (init_response, _) = request_api::<Value>(
        client,
        upload_url,
        headers.clone(),
        Method::POST,
        Some(json!({
            "command": "INIT",
            "total_bytes": file_data.len(),
            "media_type": media_type
        })),
    )
    .await?;

    let media_id = init_response["media_id_string"]
        .as_str()
        .ok_or_else(|| TwitterError::Api("Failed to get media_id".into()))?
        .to_string();

    // APPEND command - upload in chunks
    let chunk_size = 5 * 1024 * 1024; // 5MB chunks
    let mut segment_index = 0;

    for chunk in file_data.chunks(chunk_size) {
        let form = reqwest::multipart::Form::new()
            .text("command", "APPEND")
            .text("media_id", media_id.clone())
            .text("segment_index", segment_index.to_string())
            .part("media", reqwest::multipart::Part::bytes(chunk.to_vec()));

        let (_, _) = request_multipart_api::<Value>(client, upload_url, headers.clone(), form).await?;

        segment_index += 1;
    }

    // FINALIZE command
    let (finalize_response, _) = request_api::<Value>(
        client,
        &format!("{}?command=FINALIZE&media_id={}", upload_url, media_id),
        headers.clone(),
        Method::POST,
        None,
    )
    .await?;

    // Check processing status for videos
    if finalize_response.get("processing_info").is_some() {
        check_upload_status(client, &media_id, &headers).await?;
    }

    Ok(media_id)
}

async fn check_upload_status(client: &Client, media_id: &str, headers: &HeaderMap) -> Result<()> {
    let upload_url = "https://upload.twitter.com/1.1/media/upload.json";

    for _ in 0..20 {
        // Maximum 20 attempts
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await; // Wait 5 seconds

        let (status_response, _) = request_api::<Value>(
            client,
            &format!("{}?command=STATUS&media_id={}", upload_url, media_id),
            headers.clone(),
            Method::GET,
            None,
        )
        .await?;

        if let Some(processing_info) = status_response.get("processing_info") {
            match processing_info["state"].as_str() {
                Some("succeeded") => return Ok(()),
                Some("failed") => return Err(TwitterError::Api("Video processing failed".into())),
                _ => continue,
            }
        }
    }

    Err(TwitterError::Api("Video processing timeout".into()))
}

pub async fn get_tweet(client: &Client, auth: &dyn TwitterAuth, id: &str) -> Result<Tweet> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;
    let tweet_detail_request = Endpoints::tweet_detail(id);
    let url = tweet_detail_request.to_request_url();

    let (response, _) = request_api::<Value>(client, &url, headers, Method::GET, None).await?;
    let data = response.clone();
    let conversation: ThreadedConversation = serde_json::from_value(data)?;
    let tweets = parse_threaded_conversation(&conversation);
    tweets.into_iter().next().ok_or_else(|| TwitterError::Api("No tweets found".into()))
}

fn create_tweet_features() -> Value {
    json!({
        "interactive_text_enabled": true,
        "longform_notetweets_inline_media_enabled": false,
        "responsive_web_text_conversations_enabled": false,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": false,
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
        "unified_cards_ad_metadata_container_dynamic_card_content_query_enabled": false,
        "rweb_video_timestamps_enabled": false,
        "c9s_tweet_anatomy_moderator_badge_enabled": false,
        "responsive_web_twitter_article_tweet_consumption_enabled": false
    })
}
fn get_default_features() -> Value {
    json!({
        "interactive_text_enabled": true,
        "longform_notetweets_inline_media_enabled": false,
        "responsive_web_text_conversations_enabled": false,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": false,
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
        "unified_cards_ad_metadata_container_dynamic_card_content_query_enabled": false,
        "rweb_video_timestamps_enabled": true,
        "c9s_tweet_anatomy_moderator_badge_enabled": true,
        "responsive_web_twitter_article_tweet_consumption_enabled": false,
        "creator_subscriptions_quote_tweet_preview_enabled": false,
        "profile_label_improvements_pcf_label_in_post_enabled": false,
        "rweb_tipjar_consumption_enabled": true,
        "articles_preview_enabled": true
    })
}

// Helper function for long tweet features
fn get_long_tweet_features() -> Value {
    json!({
        "premium_content_api_read_enabled": false,
        "communities_web_enable_tweet_community_results_fetch": true,
        "c9s_tweet_anatomy_moderator_badge_enabled": true,
        "responsive_web_grok_analyze_button_fetch_trends_enabled": true,
        "responsive_web_edit_tweet_api_enabled": true,
        "graphql_is_translatable_rweb_tweet_is_translatable_enabled": true,
        "view_counts_everywhere_api_enabled": true,
        "longform_notetweets_consumption_enabled": true,
        "responsive_web_twitter_article_tweet_consumption_enabled": true,
        "tweet_awards_web_tipping_enabled": false,
        "longform_notetweets_rich_text_read_enabled": true,
        "longform_notetweets_inline_media_enabled": true,
        "responsive_web_graphql_exclude_directive_enabled": true,
        "verified_phone_label_enabled": false,
        "freedom_of_speech_not_reach_fetch_enabled": true,
        "standardized_nudges_misinfo": true,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": true,
        "responsive_web_graphql_timeline_navigation_enabled": true,
        "responsive_web_enhance_cards_enabled": false
    })
}

pub async fn create_tweet_request(
    client: &Client,
    auth: &dyn TwitterAuth,
    text: &str,
    reply_to: Option<&str>,
    media_data: Option<Vec<(Vec<u8>, String)>>,
) -> Result<Value> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    // Prepare variables
    let mut variables = json!({
        "tweet_text": text,
        "dark_request": false,
        "media": {
            "media_entities": [],
            "possibly_sensitive": false
        },
        "semantic_annotation_ids": []
    });

    // Add reply information if provided
    if let Some(reply_id) = reply_to {
        variables["reply"] = json!({
            "in_reply_to_tweet_id": reply_id
        });
    }

    // Handle media uploads if provided
    if let Some(media_files) = media_data {
        let mut media_entities = Vec::new();

        // Upload each media file and collect media IDs
        for (file_data, media_type) in media_files {
            let media_id = upload_media(client, auth, file_data, &media_type).await?;
            media_entities.push(json!({
                "media_id": media_id,
                "tagged_users": []
            }));
        }

        variables["media"]["media_entities"] = json!(media_entities);
    }
    let features = create_tweet_features();
    // Make the create tweet request
    let (value, _headers) = request_api(
        client,
        "https://twitter.com/i/api/graphql/a1p9RWpkYKBjWv_I3WzS-A/CreateTweet",
        headers,
        Method::POST,
        Some(json!({
            "variables": variables,
            "features": features,
            "fieldToggles": {}
        })),
    )
    .await?;

    Ok(value)
}

fn create_quote_tweet_features() -> Value {
    json!({
        "interactive_text_enabled": true,
        "longform_notetweets_inline_media_enabled": false,
        "responsive_web_text_conversations_enabled": false,
        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": false,
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
        "unified_cards_ad_metadata_container_dynamic_card_content_query_enabled": false,
        "rweb_video_timestamps_enabled": true,
        "c9s_tweet_anatomy_moderator_badge_enabled": true,
        "responsive_web_twitter_article_tweet_consumption_enabled": false
    })
}

pub async fn fetch_user_tweets(
    client: &Client,
    auth: &dyn TwitterAuth,
    user_id: &str,
    max_tweets: i32,
    cursor: Option<&str>,
    
) -> Result<QueryTweetsResponse> {
    let mut headers = HeaderMap::new();
    auth.install_headers(&mut headers).await?;

    let endpoint = Endpoints::user_tweets(user_id, max_tweets.min(200), cursor);

    let (value, _headers) =
        request_api(client, &endpoint.to_request_url(), headers, Method::GET, None).await?;

    let parsed_response = parse_timeline_tweets_v2(&value);
    Ok(parsed_response)
}
