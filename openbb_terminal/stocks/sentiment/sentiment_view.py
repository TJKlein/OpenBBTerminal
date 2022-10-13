"""Sentiment view"""
__docformat__ = "numpy"

import logging
import warnings
import pandas as pd


from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import (
    print_rich_table,
)

logger = logging.getLogger(__name__)

column_map = {"mid_iv": "iv", "open_interest": "oi", "volume": "vol"}
warnings.filterwarnings("ignore")


@log_start_end(log=logger)
def display_data_sources(source_list: list):
    """Display available data sources

    Parameters
    ----------
    source_list: list
        The list of data sources for sentiment analysis.
    """
    source_list_df = pd.DataFrame(source_list, columns=["Sources"])

    print_rich_table(
        source_list_df,
        headers=list(source_list_df.columns),
        title="Available data sources",
        show_index=True,
        index_name="Identifier",
    )
