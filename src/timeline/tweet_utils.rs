use crate::models::{Photo, Video};
use crate::timeline::v1::{LegacyTweetRaw, TimelineMediaExtendedRaw};
use lazy_static::lazy_static;
use regex::Regex;
lazy_static! {
    static ref RE_HASHTAG: Regex = Regex::new(r"\B(\#\S+\b)").unwrap();
    static ref RE_CASHTAG: Regex = Regex::new(r"\B(\$\S+\b)").unwrap();
    static ref RE_TWITTER_URL: Regex =
        Regex::new(r"https:(\/\/t\.co\/([A-Za-z0-9]|[A-Za-z]){10})").unwrap();
    static ref RE_USERNAME: Regex = Regex::new(r"\B(\@\S{1,15}\b)").unwrap();
}

pub type NonNullableMediaFields = TimelineMediaExtendedRaw;

pub fn parse_media_groups(media: &[TimelineMediaExtendedRaw]) -> (Vec<Photo>, Vec<Video>, bool) {
    let mut photos = Vec::new();
    let mut videos = Vec::new();
    let mut sensitive_content = false;

    for m in media
        .iter()
        .filter(|m| m.id_str.is_some() && m.media_url_https.is_some())
    {
        match m.r#type.as_deref() {
            Some("photo") => {
                photos.push(Photo {
                    id: m.id_str.clone().unwrap(),
                    url: m.media_url_https.clone().unwrap(),
                    alt_text: m.ext_alt_text.clone(),
                });
            }
            Some("video") => {
                videos.push(parse_video(m));
            }
            _ => {}
        }

        if let Some(warning) = &m.ext_sensitive_media_warning {
            sensitive_content = warning.adult_content.unwrap_or(false)
                || warning.graphic_violence.unwrap_or(false)
                || warning.other.unwrap_or(false);
        }
    }

    (photos, videos, sensitive_content)
}

fn parse_video(m: &NonNullableMediaFields) -> Video {
    let mut video = Video {
        id: m.id_str.clone().unwrap(),
        preview: m.media_url_https.clone().unwrap(),
        url: None,
    };

    let mut max_bitrate = 0;
    if let Some(video_info) = &m.video_info {
        if let Some(variants) = &video_info.variants {
            for variant in variants {
                if let (Some(bitrate), Some(url)) = (&variant.bitrate, &variant.url) {
                    if *bitrate > max_bitrate {
                        let mut variant_url = url.clone();
                        if let Some(idx) = variant_url.find("?tag=10") {
                            variant_url = variant_url[..idx + 1].to_string();
                        }
                        video.url = Some(variant_url);
                        max_bitrate = *bitrate;
                    }
                }
            }
        }
    }

    video
}

pub fn reconstruct_tweet_html(
    tweet: &LegacyTweetRaw,
    photos: &[Photo],
    videos: &[Video],
) -> Option<String> {
    let mut html = tweet.full_text.clone().unwrap_or_default();
    let mut media = Vec::new();

    // Replace entities with HTML
    html = RE_HASHTAG
        .replace_all(&html, |caps: &regex::Captures| link_hashtag_html(&caps[0]))
        .to_string();
    html = RE_CASHTAG
        .replace_all(&html, |caps: &regex::Captures| link_cashtag_html(&caps[0]))
        .to_string();
    html = RE_USERNAME
        .replace_all(&html, |caps: &regex::Captures| link_username_html(&caps[0]))
        .to_string();
    html = RE_TWITTER_URL
        .replace_all(&html, |caps: &regex::Captures| {
            unwrap_tco_url_html(tweet, &mut media, &caps[0])
        })
        .to_string();

    // Add media
    for photo in photos {
        if !media.contains(&photo.url) {
            html.push_str(&format!("<br><img src=\"{}\"/>", photo.url));
        }
    }

    for video in videos {
        if !media.contains(&video.preview) {
            html.push_str(&format!("<br><img src=\"{}\"/>", video.preview));
        }
    }

    // Replace newlines with <br>
    html = html.replace('\n', "<br>");

    Some(html)
}

fn link_hashtag_html(hashtag: &str) -> String {
    format!(
        "<a href=\"https://twitter.com/hashtag/{}\">{}</a>",
        &hashtag[1..],
        hashtag
    )
}

fn link_cashtag_html(cashtag: &str) -> String {
    format!(
        "<a href=\"https://twitter.com/search?q=%24{}\">{}</a>",
        &cashtag[1..],
        cashtag
    )
}

fn link_username_html(username: &str) -> String {
    format!(
        "<a href=\"https://twitter.com/{}\">{}</a>",
        &username[1..],
        username
    )
}

fn unwrap_tco_url_html(tweet: &LegacyTweetRaw, found_media: &mut Vec<String>, tco: &str) -> String {
    if let Some(entities) = &tweet.entities {
        // Check URLs
        if let Some(urls) = &entities.urls {
            for entity in urls {
                if let (Some(url), Some(expanded)) = (&entity.url, &entity.expanded_url) {
                    if url == tco {
                        return format!("<a href=\"{}\">{}</a>", expanded, tco);
                    }
                }
            }
        }

        // Check media
        if let Some(media) = &entities.media {
            for entity in media {
                if let (Some(url), Some(media_url)) = (&entity.url, &entity.media_url_https) {
                    if url == tco {
                        found_media.push(media_url.clone());
                        return format!("<br><a href=\"{}\"><img src=\"{}\"/></a>", tco, media_url);
                    }
                }
            }
        }
    }

    tco.to_string()
}
