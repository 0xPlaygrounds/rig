use std::collections::HashMap;
use urlencoding;

// Constants for default options matching TypeScript
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

pub const DEFAULT_TWEET_FIELDS: &[&str] = &[
    "attachments",
    "author_id",
    "context_annotations",
    "conversation_id",
    "created_at",
    "entities",
    "geo",
    "id",
    "in_reply_to_user_id",
    "lang",
    "public_metrics",
    "edit_controls",
    "possibly_sensitive",
    "referenced_tweets",
    "reply_settings",
    "source",
    "text",
    "withheld",
    "note_tweet",
];

#[derive(Debug, Clone)]
pub struct ApiEndpoint {
    pub url: String,
    pub variables: Option<HashMap<String, serde_json::Value>>,
    pub features: Option<HashMap<String, bool>>,
    pub field_toggles: Option<HashMap<String, bool>>,
}

impl ApiEndpoint {
    pub fn to_request_url(&self) -> String {
        let mut params = Vec::new();

        if let Some(variables) = &self.variables {
            params.push(format!(
                "variables={}",
                urlencoding::encode(&serde_json::to_string(&variables).unwrap())
            ));
        }

        if let Some(features) = &self.features {
            params.push(format!(
                "features={}",
                urlencoding::encode(&serde_json::to_string(&features).unwrap())
            ));
        }

        if let Some(toggles) = &self.field_toggles {
            params.push(format!(
                "fieldToggles={}",
                urlencoding::encode(&serde_json::to_string(&toggles).unwrap())
            ));
        }

        if params.is_empty() {
            self.url.clone()
        } else {
            format!("{}?{}", self.url, params.join("&"))
        }
    }
}

pub struct Endpoints;

impl Endpoints {
    pub fn tweet_detail(tweet_id: &str) -> ApiEndpoint {
        ApiEndpoint {
            url: "https://twitter.com/i/api/graphql/xOhkmRac04YFZmOzU9PJHg/TweetDetail".to_string(),
            variables: Some(HashMap::from([
                ("focalTweetId".to_string(), tweet_id.into()),
                ("with_rux_injections".to_string(), false.into()),
                ("includePromotedContent".to_string(), true.into()),
                ("withCommunity".to_string(), true.into()),
                (
                    "withQuickPromoteEligibilityTweetFields".to_string(),
                    true.into(),
                ),
                ("withBirdwatchNotes".to_string(), true.into()),
                ("withVoice".to_string(), true.into()),
                ("withV2Timeline".to_string(), true.into()),
            ])),
            features: Some(HashMap::from([
                (
                    "responsive_web_graphql_exclude_directive_enabled".to_string(),
                    true,
                ),
                ("verified_phone_label_enabled".to_string(), false),
                (
                    "creator_subscriptions_tweet_preview_api_enabled".to_string(),
                    true,
                ),
                (
                    "responsive_web_graphql_timeline_navigation_enabled".to_string(),
                    true,
                ),
                (
                    "responsive_web_graphql_skip_user_profile_image_extensions_enabled".to_string(),
                    false,
                ),
                ("tweetypie_unmention_optimization_enabled".to_string(), true),
                ("responsive_web_edit_tweet_api_enabled".to_string(), true),
                (
                    "graphql_is_translatable_rweb_tweet_is_translatable_enabled".to_string(),
                    true,
                ),
                ("view_counts_everywhere_api_enabled".to_string(), true),
                ("longform_notetweets_consumption_enabled".to_string(), true),
                ("tweet_awards_web_tipping_enabled".to_string(), false),
                (
                    "freedom_of_speech_not_reach_fetch_enabled".to_string(),
                    true,
                ),
                ("standardized_nudges_misinfo".to_string(), true),
                (
                    "responsive_web_twitter_article_tweet_consumption_enabled".to_string(),
                    false,
                ),
                (
                    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled"
                        .to_string(),
                    true,
                ),
                (
                    "longform_notetweets_rich_text_read_enabled".to_string(),
                    true,
                ),
                ("longform_notetweets_inline_media_enabled".to_string(), true),
                (
                    "responsive_web_media_download_video_enabled".to_string(),
                    false,
                ),
                ("responsive_web_enhance_cards_enabled".to_string(), false),
            ])),
            field_toggles: Some(HashMap::from([(
                "withArticleRichContentState".to_string(),
                false,
            )])),
        }
    }

    pub fn tweet_by_rest_id(tweet_id: &str) -> ApiEndpoint {
        ApiEndpoint {
            url: "https://twitter.com/i/api/graphql/DJS3BdhUhcaEpZ7B7irJDg/TweetResultByRestId"
                .to_string(),
            variables: Some(HashMap::from([
                ("tweetId".to_string(), tweet_id.into()),
                ("withCommunity".to_string(), false.into()),
                ("includePromotedContent".to_string(), false.into()),
                ("withVoice".to_string(), false.into()),
            ])),
            features: Some(HashMap::from([
                (
                    "creator_subscriptions_tweet_preview_api_enabled".to_string(),
                    true,
                ),
                ("tweetypie_unmention_optimization_enabled".to_string(), true),
                ("responsive_web_edit_tweet_api_enabled".to_string(), true),
                (
                    "graphql_is_translatable_rweb_tweet_is_translatable_enabled".to_string(),
                    true,
                ),
                ("view_counts_everywhere_api_enabled".to_string(), true),
                ("longform_notetweets_consumption_enabled".to_string(), true),
                (
                    "responsive_web_twitter_article_tweet_consumption_enabled".to_string(),
                    false,
                ),
                ("tweet_awards_web_tipping_enabled".to_string(), false),
                (
                    "freedom_of_speech_not_reach_fetch_enabled".to_string(),
                    true,
                ),
                ("standardized_nudges_misinfo".to_string(), true),
            ])),
            field_toggles: None,
        }
    }

    pub fn user_tweets(user_id: &str, count: i32, cursor: Option<&str>) -> ApiEndpoint {
        let mut variables = HashMap::from([
            ("userId".to_string(), user_id.into()),
            ("count".to_string(), count.into()),
            ("includePromotedContent".to_string(), true.into()),
            (
                "withQuickPromoteEligibilityTweetFields".to_string(),
                true.into(),
            ),
            ("withVoice".to_string(), true.into()),
            ("withV2Timeline".to_string(), true.into()),
        ]);

        if let Some(cursor_value) = cursor {
            variables.insert("cursor".to_string(), cursor_value.into());
        }

        ApiEndpoint {
            url: "https://twitter.com/i/api/graphql/V7H0Ap3_Hh2FyS75OCDO3Q/UserTweets".to_string(),
            variables: Some(variables),
            features: Some(HashMap::from([
                ("rweb_tipjar_consumption_enabled".to_string(), true),
                (
                    "responsive_web_graphql_exclude_directive_enabled".to_string(),
                    true,
                ),
                ("verified_phone_label_enabled".to_string(), false),
                (
                    "creator_subscriptions_tweet_preview_api_enabled".to_string(),
                    true,
                ),
                (
                    "responsive_web_graphql_timeline_navigation_enabled".to_string(),
                    true,
                ),
                (
                    "responsive_web_graphql_skip_user_profile_image_extensions_enabled".to_string(),
                    false,
                ),
                (
                    "communities_web_enable_tweet_community_results_fetch".to_string(),
                    true,
                ),
                (
                    "c9s_tweet_anatomy_moderator_badge_enabled".to_string(),
                    true,
                ),
                ("articles_preview_enabled".to_string(), true),
                ("tweetypie_unmention_optimization_enabled".to_string(), true),
                ("responsive_web_edit_tweet_api_enabled".to_string(), true),
                (
                    "graphql_is_translatable_rweb_tweet_is_translatable_enabled".to_string(),
                    true,
                ),
                ("view_counts_everywhere_api_enabled".to_string(), true),
                ("longform_notetweets_consumption_enabled".to_string(), true),
                (
                    "responsive_web_twitter_article_tweet_consumption_enabled".to_string(),
                    true,
                ),
                ("tweet_awards_web_tipping_enabled".to_string(), false),
                (
                    "creator_subscriptions_quote_tweet_preview_enabled".to_string(),
                    false,
                ),
                (
                    "freedom_of_speech_not_reach_fetch_enabled".to_string(),
                    true,
                ),
                ("standardized_nudges_misinfo".to_string(), true),
                (
                    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled"
                        .to_string(),
                    true,
                ),
                ("rweb_video_timestamps_enabled".to_string(), true),
                (
                    "longform_notetweets_rich_text_read_enabled".to_string(),
                    true,
                ),
                ("longform_notetweets_inline_media_enabled".to_string(), true),
                ("responsive_web_enhance_cards_enabled".to_string(), false),
            ])),
            field_toggles: Some(HashMap::from([("withArticlePlainText".to_string(), false)])),
        }
    }

    pub fn user_tweets_and_replies(user_id: &str, count: i32, cursor: Option<&str>) -> ApiEndpoint {
        let mut variables = HashMap::from([
            ("userId".to_string(), user_id.into()),
            ("count".to_string(), count.into()),
            ("includePromotedContent".to_string(), true.into()),
            ("withCommunity".to_string(), true.into()),
            ("withVoice".to_string(), true.into()),
            ("withV2Timeline".to_string(), true.into()),
        ]);

        if let Some(cursor_value) = cursor {
            variables.insert("cursor".to_string(), cursor_value.into());
        }

        ApiEndpoint {
            url: "https://twitter.com/i/api/graphql/E4wA5vo2sjVyvpliUffSCw/UserTweetsAndReplies"
                .to_string(),
            variables: Some(variables),
            features: Some(HashMap::from([
                ("rweb_tipjar_consumption_enabled".to_string(), true),
                (
                    "responsive_web_graphql_exclude_directive_enabled".to_string(),
                    true,
                ),
                ("verified_phone_label_enabled".to_string(), false),
                (
                    "creator_subscriptions_tweet_preview_api_enabled".to_string(),
                    true,
                ),
                (
                    "responsive_web_graphql_timeline_navigation_enabled".to_string(),
                    true,
                ),
                (
                    "responsive_web_graphql_skip_user_profile_image_extensions_enabled".to_string(),
                    false,
                ),
                (
                    "communities_web_enable_tweet_community_results_fetch".to_string(),
                    true,
                ),
                (
                    "c9s_tweet_anatomy_moderator_badge_enabled".to_string(),
                    true,
                ),
                ("articles_preview_enabled".to_string(), true),
                ("tweetypie_unmention_optimization_enabled".to_string(), true),
                ("responsive_web_edit_tweet_api_enabled".to_string(), true),
                (
                    "graphql_is_translatable_rweb_tweet_is_translatable_enabled".to_string(),
                    true,
                ),
                ("view_counts_everywhere_api_enabled".to_string(), true),
                ("longform_notetweets_consumption_enabled".to_string(), true),
                (
                    "responsive_web_twitter_article_tweet_consumption_enabled".to_string(),
                    true,
                ),
                ("tweet_awards_web_tipping_enabled".to_string(), false),
                (
                    "creator_subscriptions_quote_tweet_preview_enabled".to_string(),
                    false,
                ),
                (
                    "freedom_of_speech_not_reach_fetch_enabled".to_string(),
                    true,
                ),
                ("standardized_nudges_misinfo".to_string(), true),
                (
                    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled"
                        .to_string(),
                    true,
                ),
                ("rweb_video_timestamps_enabled".to_string(), true),
                (
                    "longform_notetweets_rich_text_read_enabled".to_string(),
                    true,
                ),
                ("longform_notetweets_inline_media_enabled".to_string(), true),
                ("responsive_web_enhance_cards_enabled".to_string(), false),
            ])),
            field_toggles: Some(HashMap::from([("withArticlePlainText".to_string(), false)])),
        }
    }
}
