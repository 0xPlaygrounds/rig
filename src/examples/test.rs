use rig_twitter::{error::Result, scraper::Scraper};
use std::fs::File;
use std::io::Write;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Create a new scraper instance
    let mut scraper = Scraper::new().await?;
    // let username = "".to_string();
    // let password = "".to_string();
    // let email = Some("@gmail.com".to_string());
    // let two_factor_secret = Some("".to_string());
    // scraper.login(username, password, email, two_factor_secret).await?;
    // Perform login
    scraper.set_from_cookie_string("").await?;
    //let is_logged_in = scraper.is_logged_in().await?;
    let tweets = scraper
        .fetch_tweets_and_replies("0xleductam", 20, None)
        .await?;
    let mut file = File::create("result.json")?;
    write!(file, "{}", serde_json::to_string_pretty(&tweets).unwrap())?;
    // println!("Tweet: {:?}", tweets);
    // let mut file = File::create("result.json")?;
    // write!(file, "{}", serde_json::to_string_pretty(&tweets).unwrap())?;
    // // Load the image file
    // let mut file = File::open("image.jpg")?;
    // let mut image_data = Vec::new();
    // file.read_to_end(&mut image_data)?;

    // // Create media data tuple with the image data and MIME type
    // let media_data = vec![(image_data, "image/jpeg".to_string())];

    // // Send the tweet with the image
    // let text = "Your tweet text here"; // Replace with your desired tweet text
    // let result = scraper.send_tweet(text, Some(("1868291574762963179")), Some(media_data)).await?;
    // println!("Tweet sent successfully: {:?}", result);

    // Test the connection by fetching the home timeline
    // let tweets: Vec<Value> = scraper.get_home_timeline(20, vec![]).await?;

    // println!("Successfully retrieved {} tweets from home timeline", tweets.len());

    // // Print tweet details
    // for tweet in tweets {
    //     if let Some(tweet_data) = tweet.get("tweet") {
    //         if let Some(text) = tweet_data.get("full_text") {
    //             if let Some(text_str) = text.as_str() {
    //                 println!("Tweet: {}", text_str);
    //             }
    //         }
    //     }
    // }

    Ok(())
}
