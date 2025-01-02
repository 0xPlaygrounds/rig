use crate::profile::parse_profile;
use crate::timeline::v1::{QueryProfilesResponse, QueryTweetsResponse};
use crate::timeline::v2::{parse_legacy_tweet, SearchEntryRaw};
use lazy_static::lazy_static;
use serde::Deserialize;

lazy_static! {
    static ref EMPTY_INSTRUCTIONS: Vec<SearchInstruction> = Vec::new();
    static ref EMPTY_ENTRIES: Vec<SearchEntryRaw> = Vec::new();
}

#[derive(Debug, Deserialize)]
pub struct SearchTimeline {
    pub data: Option<SearchData>,
}

#[derive(Debug, Deserialize)]
pub struct SearchData {
    pub search_by_raw_query: Option<SearchByRawQuery>,
}

#[derive(Debug, Deserialize)]
pub struct SearchByRawQuery {
    pub search_timeline: Option<SearchTimelineData>,
}

#[derive(Debug, Deserialize)]
pub struct SearchTimelineData {
    pub timeline: Option<TimelineData>,
}

#[derive(Debug, Deserialize)]
pub struct TimelineData {
    pub instructions: Option<Vec<SearchInstruction>>,
}

#[derive(Debug, Deserialize)]
pub struct SearchInstruction {
    pub entries: Option<Vec<SearchEntryRaw>>,
    pub entry: Option<SearchEntryRaw>,
    #[serde(rename = "type")]
    pub instruction_type: Option<String>,
}

pub fn parse_search_timeline_tweets(timeline: &SearchTimeline) -> QueryTweetsResponse {
    let mut bottom_cursor = None;
    let mut top_cursor = None;
    let mut tweets = Vec::new();

    let instructions = timeline
        .data
        .as_ref()
        .and_then(|data| data.search_by_raw_query.as_ref())
        .and_then(|search| search.search_timeline.as_ref())
        .and_then(|timeline| timeline.timeline.as_ref())
        .and_then(|timeline| timeline.instructions.as_ref())
        .unwrap_or(&EMPTY_INSTRUCTIONS);

    for instruction in instructions {
        if let Some(instruction_type) = &instruction.instruction_type {
            if instruction_type == "TimelineAddEntries"
                || instruction_type == "TimelineReplaceEntry"
            {
                if let Some(entry) = &instruction.entry {
                    if let Some(content) = &entry.content {
                        match content.cursor_type.as_deref() {
                            Some("Bottom") => {
                                bottom_cursor = content.value.clone();
                                continue;
                            }
                            Some("Top") => {
                                top_cursor = content.value.clone();
                                continue;
                            }
                            _ => {}
                        }
                    }
                }

                // Process entries
                let entries = instruction.entries.as_ref().unwrap_or(&EMPTY_ENTRIES);
                for entry in entries {
                    if let Some(content) = &entry.content {
                        if let Some(item_content) = &content.item_content {
                            if item_content.tweet_display_type.as_deref() == Some("Tweet") {
                                if let Some(tweet_results) = &item_content.tweet_results {
                                    if let Some(result) = &tweet_results.result {
                                        let user_legacy = result
                                            .core
                                            .as_ref()
                                            .and_then(|core| core.user_results.as_ref())
                                            .and_then(|user_results| user_results.result.as_ref())
                                            .and_then(|result| result.legacy.as_ref());

                                        if let Ok(tweet_result) = parse_legacy_tweet(
                                            user_legacy,
                                            result.legacy.as_deref(),
                                        )
                                        {
                                            if tweet_result.views.is_none() {
                                                if let Some(views) = &result.views {
                                                    if let Some(count) = &views.count {
                                                        if let Ok(view_count) = count.parse::<i32>()
                                                        {
                                                            let mut tweet = tweet_result;
                                                            tweet.views = Some(view_count);
                                                            tweets.push(tweet);
                                                        }
                                                    }
                                                }
                                            } else {
                                                tweets.push(tweet_result);
                                            }
                                        }
                                    }
                                }
                            }
                        } else if let Some(cursor_type) = &content.cursor_type {
                            match cursor_type.as_str() {
                                "Bottom" => bottom_cursor = content.value.clone(),
                                "Top" => top_cursor = content.value.clone(),
                                _ => {}
                            }
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

pub fn parse_search_timeline_users(timeline: &SearchTimeline) -> QueryProfilesResponse {
    let mut bottom_cursor = None;
    let mut top_cursor = None;
    let mut profiles = Vec::new();

    let instructions = timeline
        .data
        .as_ref()
        .and_then(|data| data.search_by_raw_query.as_ref())
        .and_then(|search| search.search_timeline.as_ref())
        .and_then(|timeline| timeline.timeline.as_ref())
        .and_then(|timeline| timeline.instructions.as_ref())
        .unwrap_or(&EMPTY_INSTRUCTIONS);

    for instruction in instructions {
        if let Some(instruction_type) = &instruction.instruction_type {
            if instruction_type == "TimelineAddEntries"
                || instruction_type == "TimelineReplaceEntry"
            {
                if let Some(entry) = &instruction.entry {
                    if let Some(content) = &entry.content {
                        match content.cursor_type.as_deref() {
                            Some("Bottom") => {
                                bottom_cursor = content.value.clone();
                                continue;
                            }
                            Some("Top") => {
                                top_cursor = content.value.clone();
                                continue;
                            }
                            _ => {}
                        }
                    }
                }

                // Process entries
                let entries = instruction.entries.as_ref().unwrap_or(&EMPTY_ENTRIES);
                for entry in entries {
                    if let Some(content) = &entry.content {
                        if let Some(item_content) = &content.item_content {
                            if item_content.user_display_type.as_deref() == Some("User") {
                                if let Some(user_results) = &item_content.user_results {
                                    if let Some(result) = &user_results.result {
                                        if let Some(legacy) = &result.legacy {
                                            let mut profile =
                                                parse_profile(legacy, result.is_blue_verified);

                                            if profile.id.is_empty() {
                                                profile.id =
                                                    result.rest_id.clone().unwrap_or_default();
                                            }

                                            profiles.push(profile);
                                        }
                                    }
                                }
                            }
                        } else if let Some(cursor_type) = &content.cursor_type {
                            match cursor_type.as_str() {
                                "Bottom" => bottom_cursor = content.value.clone(),
                                "Top" => top_cursor = content.value.clone(),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    QueryProfilesResponse {
        profiles,
        next: bottom_cursor,
        previous: top_cursor,
    }
}
