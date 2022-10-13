""" Sentiment Controller """
__docformat__ = "numpy"

import argparse
import logging
from typing import List

import numpy as np
import pandas as pd
from openbb_terminal.custom_prompt_toolkit import NestedCompleter
from openbb_terminal.menu import session
from openbb_terminal import feature_flags as obbff
from openbb_terminal.common.behavioural_analysis import (
    twitter_view,
    reddit_model,
    twitter_model,
)

from openbb_terminal.stocks.sentiment import sentiment_view, sentiment_model

from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import (
    EXPORT_BOTH_RAW_DATA_AND_FIGURES,
    check_positive,
    check_int_range,
)

from openbb_terminal.parent_classes import BaseController
from openbb_terminal.rich_config import console, MenuText, get_ordered_list_sources

logger = logging.getLogger(__name__)


class SentimentController(BaseController):
    """Sentiment Controller class"""

    CHOICES_COMMANDS = ["load", "source", "vader", "transformer"]

    PATH = "/stocks/sentiment/"

    def __init__(
        self,
        ticker: str,
        data_source: str = "",
        queue: List[str] = None,
    ):
        """Constructor"""
        super().__init__(queue)

        # stock["Returns"] = stock["Adj Close"].pct_change()
        # stock["LogRet"] = np.log(stock["Adj Close"]) - np.log(
        #     stock["Adj Close"].shift(1)
        # )
        # stock = stock.rename(columns={"Adj Close": "AdjClose"})
        # stock = stock.dropna()
        # print(stock)

        # self.stock = stock
        self.ticker = ticker
        self.data_source = data_source
        self.selected_data_source = None
        self.data_source_list = ["Twitter", "Reddit"]

        if session and obbff.USE_PROMPT_TOOLKIT:
            choices: dict = {c: {} for c in self.controller_choices}

            choices["load"] = {
                "--ticker": None,
                "-t": "--ticker",
                "--source": {
                    c: {} for c in get_ordered_list_sources(f"{self.PATH}load")
                },
            }
            
            choices["model"] = {"--model": None, "-t": "--model"}

            choices["exp"] = {str(c): {} for c in range(len(self.data_source_list))}

        self.choices = choices
        self.completer = NestedCompleter.from_nested_dict(choices)

    def print_help(self):
        """Print help"""
        mt = MenuText("stocks/sentiment/")
        mt.add_cmd("load")
        mt.add_cmd("source")
        mt.add_raw("\n")
        mt.add_param("_ticker", self.ticker)
        # mt.add_param("_target", self.target)
        mt.add_param("_source", self.selected_data_source or "")
        mt.add_raw("\n")
        mt.add_info("_models_")
        mt.add_cmd("vader")
        mt.add_cmd("transformer")
        # mt.add_cmd("transformer")
        console.print(text=mt.menu_text, menu="Stocks - Sentiment Techniques")

    def custom_reset(self):
        """Class specific component of reset command"""
        if self.ticker:
            return ["stocks", f"load {self.ticker}", "sentiment"]
        return []

    def __add_reddit_args(self, parser) -> argparse.ArgumentParser:

        parser.add_argument(
            "-s",
            "--sort",
            action="store",
            dest="sort",
            choices=["relevance", "hot", "top", "new", "comments"],
            default="relevance",
            help="search sorting type",
        )
        parser.add_argument(
            "-c",
            "--company",
            action="store",
            dest="company",
            default=None,
            help="explicit name of company to search for, will override ticker symbol",
        )
        parser.add_argument(
            "--subreddits",
            action="store",
            dest="subreddits",
            default="all",
            help="comma-separated list of subreddits to search",
        )
        parser.add_argument(
            "-l",
            "--limit",
            action="store",
            dest="limit",
            default=10,
            type=check_positive,
            help="how many posts to gather from each subreddit",
        )
        parser.add_argument(
            "-t",
            "--time",
            action="store",
            dest="time",
            default="week",
            choices=["hour", "day", "week", "month", "year", "all"],
            help="time period to get posts from -- all, year, month, week, or day; defaults to week",
        )
        parser.add_argument(
            "--full",
            action="store_true",
            dest="full_search",
            default=False,
            help="enable comprehensive search",
        )
        parser.add_argument(
            "-g",
            "--graphic",
            action="store_true",
            dest="graphic",
            default=True,
            help="display graphic",
        )
        parser.add_argument(
            "-d",
            "--display",
            action="store_true",
            dest="display",
            default=False,
            help="Print table of sentiment values",
        )

        return parser

    def __add_twitter_args(self, parser) -> argparse.ArgumentParser:
        # in reality this argument could be 100, but after testing it takes too long
        # to compute which may not be acceptable
        # TODO: use https://github.com/twintproject/twint instead of twitter API
        parser.add_argument(
            "-l",
            "--limit",
            action="store",
            dest="limit",
            type=check_int_range(10, 62),
            default=15,
            help="limit of tweets to extract per hour.",
        )
        parser.add_argument(
            "-d",
            "--days",
            action="store",
            dest="n_days_past",
            type=check_int_range(1, 6),
            default=6,
            help="number of days in the past to extract tweets.",
        )
        parser.add_argument(
            "-c",
            "--compare",
            action="store_true",
            dest="compare",
            help="show corresponding change in stock price",
        )
        return parser

    @log_start_end(log=logger)
    def call_transformer(self, other_args: List[str]):

        """Process transformer command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="vader",
            description="""
                Change target variable
            """,
        )
        
        parser.add_argument(
            "-m",
            "--model",
            action="store",
            type=str,
            default="finiteautomata/bertweet-base-sentiment-analysis",
            help="model name or path",
        )

        if self.selected_data_source == "Twitter":

            parser = self.__add_twitter_args(parser)
            if other_args and "-" not in other_args[0][0]:
                other_args.insert(0, "-l")
            ns_parser = self.parse_known_args_and_warn(
                parser, other_args, EXPORT_BOTH_RAW_DATA_AND_FIGURES
            )

            if ns_parser:
                if self.ticker:
                    tweet_list, created_at_list = twitter_model.get_tweet_data(
                        symbol=self.ticker,
                        n_tweets=ns_parser.limit,
                        n_days_past=ns_parser.n_days_past,
                    )
                    print(f"Number of tweets: {len(tweet_list)}")
                    print(f"Model: {ns_parser.model}")

                    # https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
                    df_tweets = sentiment_model.get_transformer_sentiment(
                        tweet_list,
                        model_name_or_path=ns_parser.model,
                    )
                    df_tweets["created_at"] = np.array(created_at_list)

                    print(df_tweets.head(5))  # will print out the first 5 rows)

                    # Sort tweets per date
                    df_tweets.sort_index(ascending=False, inplace=True)

                    df_tweets.reset_index(inplace=True)
                    df_tweets["Month"] = pd.to_datetime(df_tweets["created_at"]).apply(
                        lambda x: x.month
                    )
                    df_tweets["Day"] = pd.to_datetime(df_tweets["created_at"]).apply(
                        lambda x: x.day
                    )
                    df_tweets["date"] = pd.to_datetime(df_tweets["created_at"])
                    df_tweets = df_tweets.sort_values(by="date")
                    df_tweets["cumulative_compound"] = 1
                    df_tweets["sentiment"] = 1

                    twitter_view.display_sentiment_from_df(
                        symbol=self.ticker,
                        df_tweets=df_tweets,
                        compare=ns_parser.compare,
                        export=ns_parser.export,
                        n_days_past=ns_parser.n_days_past,
                    )
            else:
                console.print("No ticker loaded. Please load using 'load <ticker>'\n")

        elif self.selected_data_source == "Reddit":
            parser = self.__add_reddit_args(parser)
            if other_args and "-" not in other_args[0][0]:
                other_args.insert(0, "-l")
            ns_parser = self.parse_known_args_and_warn(
                parser, other_args, EXPORT_BOTH_RAW_DATA_AND_FIGURES
            )

            if ns_parser:
                ticker = ns_parser.company if ns_parser.company else self.ticker
            if self.ticker:
                post_list, _ = reddit_model.get_posts_data(
                    symbol=ticker,
                    sortby=ns_parser.sort,
                    limit=ns_parser.limit,
                    # graphic=ns_parser.graphic,
                    time_frame=ns_parser.time,
                    full_search=ns_parser.full_search,
                    subreddits=ns_parser.subreddits,
                )
                print(f"Number of Reddit posts: {len(post_list)}")

    @log_start_end(log=logger)
    def call_vader(self, other_args: List[str]):

        """Process pick command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="vader",
            description="""
                Change target variable
            """,
        )

        if self.selected_data_source == "Twitter":

            parser = self.__add_twitter_args(parser)
            if other_args and "-" not in other_args[0][0]:
                other_args.insert(0, "-l")
            ns_parser = self.parse_known_args_and_warn(
                parser, other_args, EXPORT_BOTH_RAW_DATA_AND_FIGURES
            )

            if ns_parser:
                if self.ticker:
                    tweet_list, created_at_list = twitter_model.get_tweet_data(
                        symbol=self.ticker,
                        n_tweets=ns_parser.limit,
                        n_days_past=ns_parser.n_days_past,
                    )
                    print(f"Number of tweets: {len(tweet_list)}")

                    df_tweets = sentiment_model.get_vader_sentiment(tweet_list)
                    print(f"Len: {len(created_at_list)}")
                    df_tweets["created_at"] = np.array(created_at_list)

                    # Sort tweets per date
                    df_tweets.sort_index(ascending=False, inplace=True)
                    df_tweets[
                        "cumulative_compound"
                    ] = 1  # df_tweets["sentiment"].cumsum()
                    df_tweets["prob_sen"] = 1

                    # df_tweets.to_csv(r'notebooks/tweets.csv', index=False)
                    df_tweets.reset_index(inplace=True)
                    df_tweets["Month"] = pd.to_datetime(df_tweets["created_at"]).apply(
                        lambda x: x.month
                    )
                    df_tweets["Day"] = pd.to_datetime(df_tweets["created_at"]).apply(
                        lambda x: x.day
                    )
                    df_tweets["date"] = pd.to_datetime(df_tweets["created_at"])
                    df_tweets = df_tweets.sort_values(by="date")
                    df_tweets[
                        "cumulative_compound"
                    ] = 1  # df_tweets["sentiment"].cumsum()

                    twitter_view.display_sentiment_from_df(
                        symbol=self.ticker,
                        df_tweets=df_tweets,
                        compare=ns_parser.compare,
                        export=ns_parser.export,
                    )
            else:
                console.print("No ticker loaded. Please load using 'load <ticker>'\n")

        elif self.selected_data_source == "Reddit":
            parser = self.__add_reddit_args(parser)
            if other_args and "-" not in other_args[0][0]:
                other_args.insert(0, "-l")
            ns_parser = self.parse_known_args_and_warn(
                parser, other_args, EXPORT_BOTH_RAW_DATA_AND_FIGURES
            )

            if ns_parser:
                ticker = ns_parser.company if ns_parser.company else self.ticker
            if self.ticker:
                post_list, _ = reddit_model.get_posts_data(
                    symbol=ticker,
                    sortby=ns_parser.sort,
                    limit=ns_parser.limit,
                    time_frame=ns_parser.time,
                    full_search=ns_parser.full_search,
                    subreddits=ns_parser.subreddits,
                )
                print(f"Number of Reddit posts: {len(post_list)}")

    @log_start_end(log=logger)
    def call_source(self, other_args: List[str]):
        """Process source command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="source",
            description="""
                Change target variable
            """,
        )
        parser.add_argument(
            "-i",
            "--index",
            dest="index",
            action="store",
            type=int,
            default=-1,
            choices=range(len(self.data_source_list)),
            help="Select index for expiry date.",
        )

        if other_args and "-" not in other_args[0][0]:
            other_args.insert(0, "-i")
        ns_parser = self.parse_known_args_and_warn(parser, other_args)
        # if ns_parser:
        #     self.target = ns_parser.target
        #     console.print("")
        if ns_parser.index == -1:
            sentiment_view.display_data_sources(self.data_source_list)
            console.print("")
        else:
            data_source = self.data_source_list[ns_parser.index]
            console.print(f"Data source set to {data_source} \n")
            self.selected_data_source = data_source
