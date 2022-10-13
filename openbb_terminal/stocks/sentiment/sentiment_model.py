"""Sentiment stock model"""
__docformat__ = "numpy"

import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
import numpy as np
import pandas as pd

from openbb_terminal.decorators import log_start_end
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@log_start_end(log=logger)
def get_transformer_sentiment(
    data_list: List[str], model_name_or_path: str, batch_size: int = 150, max_length=255
) -> pd.DataFrame:

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()

    df_result = pd.DataFrame()

    for index in range(0, len(data_list), batch_size):

        token_batch = tokenizer.batch_encode_plus(
            data_list[index : np.min([index + batch_size, len(data_list)])],
            return_tensors="pt",
            padding=True,
            max_length=max_length,
        )
        logits = model(**token_batch).logits
        temp = pd.DataFrame(torch.softmax(logits, dim=1).detach().numpy())

        df_result = pd.concat([df_result, temp])

    df_result.rename(columns={0: "negative", 1: "neutral", 2: "positive"}, inplace=True)
    return df_result


@log_start_end(log=logger)
def get_vader_sentiment(data_list: List[str]) -> pd.DataFrame:

    # Vader sentinment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Create dataframe
    df_tweets = pd.DataFrame()

    sentiments = []
    pos = []
    neg = []
    neu = []

    for data in data_list:
        sentiments.append(analyzer.polarity_scores(data)["compound"])
        pos.append(analyzer.polarity_scores(data)["pos"])
        neg.append(analyzer.polarity_scores(data)["neg"])
        neu.append(analyzer.polarity_scores(data)["neu"])
    # Add sentiments to tweets dataframe
    df_tweets["sentiment"] = sentiments
    df_tweets["positive"] = pos
    df_tweets["negative"] = neg
    df_tweets["neutral"] = neu

    return df_tweets
