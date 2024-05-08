import streamlit as st
import feedparser
from textblob import TextBlob
import pandas as pd

# Function to parse RSS feed and perform sentiment analysis
def fetch_rss_feed(url):
    feed = feedparser.parse(url)
    news_items = []
    for entry in feed.entries:
        sentiment = TextBlob(entry.title).sentiment
        news_items.append([entry.title, sentiment.polarity, sentiment.subjectivity])

    df = pd.DataFrame(news_items, columns=['Title', 'Sentiment Polarity', 'Sentiment Subjectivity'])
    return df

def main():
    st.title("RSS Feed Sentiment Analysis")
    
    # Predefined list of RSS feeds
    rss_feeds = {
        "The New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "The Guardian UK": "https://www.theguardian.com/uk/rss",
        "Al Jazeera English": "https://www.aljazeera.com/xml/rss/all.xml"
    }

    # Dropdown to select RSS feed
    feed_selection = st.selectbox("Choose an RSS Feed", list(rss_feeds.keys()))
    
    # Button to perform sentiment analysis
    if st.button("Analyze Sentiment"):
        with st.spinner('Fetching and analyzing the RSS feed...'):
            selected_feed_url = rss_feeds[feed_selection]
            df = fetch_rss_feed(selected_feed_url)
            st.write("Sentiment Analysis Results:")
            st.dataframe(df)

if __name__ == '__main__':
    main()
