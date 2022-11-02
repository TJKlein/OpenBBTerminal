"""Twitter Model"""
__docformat__ = "numpy"

from datetime import datetime, timedelta
import logging
from typing import Optional, List, Tuple

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from openbb_terminal import config_terminal as cfg
from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import clean_tweet, get_data
from openbb_terminal.rich_config import console

logger = logging.getLogger(__name__)

analyzer = SentimentIntensityAnalyzer()


@log_start_end(log=logger)
def load_tweets(
    symbol: str,
    limit: int = 100,
    start_date: Optional[str] = "",
    end_date: Optional[str] = "",
) -> Tuple[List[str], List[str]]:
    """Load tweets from twitter API and analyzes using VADER

    Parameters
    ----------
    symbol: str
        Ticker symbol to search twitter for
    limit: int
        Number of tweets to analyze
    start_date: Optional[str]
        If given, the start time to get tweets from
    end_date: Optional[str]
        If given, the end time to get tweets from

    Returns
    -------
    df_tweet: pd.DataFrame
        Dataframe of tweets and sentiment
    """
    params = {
        "query": rf"(\${symbol}) (lang:en)",
        "max_results": str(limit),
        "tweet.fields": "created_at,lang",
    }

    if start_date:
        # Assign from and to datetime parameters for the API
        params["start_time"] = start_date
    if end_date:
        params["end_time"] = end_date

    # Request Twitter API
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/recent",
        params=params,  # type: ignore
        headers={"authorization": "Bearer " + cfg.API_TWITTER_BEARER_TOKEN},
    )

    # Create dataframe
    df_tweets = pd.DataFrame()

    # Check that the API response was successful
    if response.status_code == 200:
        tweets = []
        for tweet in response.json()["data"]:
            row = get_data(tweet)
            tweets.append(row)
        df_tweets = pd.DataFrame(tweets)
    elif response.status_code == 401:
        console.print("Twitter API Key provided is incorrect\n")
        return pd.DataFrame()
    elif response.status_code == 400:
        console.print(
            """
            Status Code 400.
            This means you are requesting data from beyond the API's 7 day limit"""
        )
        return pd.DataFrame()
    elif response.status_code == 403:
        console.print(
            f"""
            Status code 403.
            It seems you're twitter credentials are invalid - {response.text}
        """
        )
        return pd.DataFrame()
    else:
        console.print(
            f"""
            Status code {response.status_code}.
            Something went wrong - {response.text}
        """
        )
        return pd.DataFrame()

    clean_texts = []
    for s_tweet in df_tweets["text"].to_list():
        tweet = clean_tweet(s_tweet, symbol)
        clean_texts.append(tweet)
    return clean_texts, df_tweets["created_at"]


@log_start_end(log=logger)
def load_analyze_tweets(
    symbol: str,
    limit: int = 100,
    start_date: Optional[str] = "",
    end_date: Optional[str] = "",
) -> pd.DataFrame:
    """Load tweets from twitter API and analyzes using VADER

    Parameters
    ----------
    symbol: str
        Ticker symbol to search twitter for
    limit: int
        Number of tweets to analyze
    start_date: Optional[str]
        If given, the start time to get tweets from
    end_date: Optional[str]
        If given, the end time to get tweets from

    Returns
    -------
    df_tweet: pd.DataFrame
        Dataframe of tweets and sentiment
    """
    params = {
        "query": rf"(\${symbol}) (lang:en)",
        "max_results": str(limit),
        "tweet.fields": "created_at,lang",
    }

    if start_date:
        # Assign from and to datetime parameters for the API
        params["start_time"] = start_date
    if end_date:
        params["end_time"] = end_date

    # Request Twitter API
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/recent",
        params=params,  # type: ignore
        headers={"authorization": "Bearer " + cfg.API_TWITTER_BEARER_TOKEN},
    )

    # Create dataframe
    df_tweets = pd.DataFrame()

    # Check that the API response was successful
    if response.status_code == 200:
        tweets = []
        for tweet in response.json()["data"]:
            row = get_data(tweet)
            tweets.append(row)
        df_tweets = pd.DataFrame(tweets)
    elif response.status_code == 401:
        console.print("Twitter API Key provided is incorrect\n")
        return pd.DataFrame()
    elif response.status_code == 400:
        console.print(
            """
            Status Code 400.
            This means you are requesting data from beyond the API's 7 day limit"""
        )
        return pd.DataFrame()
    elif response.status_code == 403:
        console.print(
            f"""
            Status code 403.
            It seems you're twitter credentials are invalid - {response.text}
        """
        )
        return pd.DataFrame()
    else:
        console.print(
            f"""
            Status code {response.status_code}.
            Something went wrong - {response.text}
        """
        )
        return pd.DataFrame()

    sentiments = []
    pos = []
    neg = []
    neu = []

    for s_tweet in df_tweets["text"].to_list():
        tweet = clean_tweet(s_tweet, symbol)
        sentiments.append(analyzer.polarity_scores(tweet)["compound"])
        pos.append(analyzer.polarity_scores(tweet)["pos"])
        neg.append(analyzer.polarity_scores(tweet)["neg"])
        neu.append(analyzer.polarity_scores(tweet)["neu"])
    # Add sentiments to tweets dataframe
    df_tweets["sentiment"] = sentiments
    df_tweets["positive"] = pos
    df_tweets["negative"] = neg
    df_tweets["neutral"] = neu

    return df_tweets


@log_start_end(log=logger)
def get_tweet_data(
    symbol: str,
    n_tweets: int = 15,
    n_days_past: int = 2,
):
    """Get sentiments from symbol

    Parameters
    ----------
    symbol: str
        Stock ticker symbol to get sentiment for
    n_tweets: int
        Number of tweets to get per hour
    n_days_past: int
        Number of days to extract tweets for
    """
    # Date format string required by twitter
    dt_format = "%Y-%m-%dT%H:%M:%SZ"

    # Algorithm to extract
    dt_recent = datetime.utcnow() - timedelta(seconds=20)
    dt_old = dt_recent - timedelta(days=n_days_past)
    console.print(
        f"From {dt_recent.date()} retrieving {n_tweets*24} tweets ({n_tweets} tweets/hour)"
    )

    tweets = []
    created_at_list = []
    while True:
        # Iterate until we haven't passed the old number of days
        if dt_recent < dt_old:
            break
        # Update past datetime
        dt_past = dt_recent - timedelta(minutes=60)

        temp, temp_created_at = load_tweets(
            symbol,
            n_tweets,
            start_date=dt_past.strftime(dt_format),
            end_date=dt_recent.strftime(dt_format),
        )
        created_at_list.append(temp_created_at)
        tweets.append(temp)

        if dt_past.day < dt_recent.day:
            console.print(
                f"From {dt_past.date()} retrieving {n_tweets*24} tweets ({n_tweets} tweets/hour)"
            )

        # Update recent datetime
        dt_recent = dt_past

    # turn list of lists into flat list
    return [item for sublist in tweets for item in sublist], [
        item for sublist in created_at_list for item in sublist
    ]


@log_start_end(log=logger)
def get_sentiment(
    symbol: str,
    n_tweets: int = 15,
    n_days_past: int = 2,
):
    """Get sentiments from symbol

    Parameters
    ----------
    symbol: str
        Stock ticker symbol to get sentiment for
    n_tweets: int
        Number of tweets to get per hour
    n_days_past: int
        Number of days to extract tweets for
    """
    # Date format string required by twitter
    dt_format = "%Y-%m-%dT%H:%M:%SZ"

    # Algorithm to extract
    dt_recent = datetime.utcnow() - timedelta(seconds=20)
    dt_old = dt_recent - timedelta(days=n_days_past)
    console.print(
        f"From {dt_recent.date()} retrieving {n_tweets*24} tweets ({n_tweets} tweets/hour)"
    )

    df_tweets = pd.DataFrame(
        columns=[
            "created_at",
            "text",
            "sentiment",
            "positive",
            "negative",
            "neutral",
        ]
    )
    while True:
        # Iterate until we haven't passed the old number of days
        if dt_recent < dt_old:
            break
        # Update past datetime
        dt_past = dt_recent - timedelta(minutes=60)

        temp = load_analyze_tweets(
            symbol,
            n_tweets,
            start_date=dt_past.strftime(dt_format),
            end_date=dt_recent.strftime(dt_format),
        )

        if temp.empty:
            return pd.DataFrame()

        df_tweets = pd.concat([df_tweets, temp])

        if dt_past.day < dt_recent.day:
            console.print(
                f"From {dt_past.date()} retrieving {n_tweets*24} tweets ({n_tweets} tweets/hour)"
            )

        # Update recent datetime
        dt_recent = dt_past

    # Sort tweets per date
    df_tweets.sort_index(ascending=False, inplace=True)
    df_tweets["cumulative_compound"] = df_tweets["sentiment"].cumsum()
    df_tweets["prob_sen"] = 1

    # df_tweets.to_csv(r'notebooks/tweets.csv', index=False)
    df_tweets.reset_index(inplace=True)
    df_tweets["Month"] = pd.to_datetime(df_tweets["created_at"]).apply(
        lambda x: x.month
    )
    df_tweets["Day"] = pd.to_datetime(df_tweets["created_at"]).apply(lambda x: x.day)
    df_tweets["date"] = pd.to_datetime(df_tweets["created_at"])
    df_tweets = df_tweets.sort_values(by="date")
    df_tweets["cumulative_compound"] = df_tweets["sentiment"].cumsum()

    return df_tweets
