# Copyright 2020 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from moonshot import Moonshot
from moonshot.commission import PerShareCommission
from quantrocket.fundamental import get_sharadar_fundamentals_reindexed_like

class USStockCommission(PerShareCommission):
    BROKER_COMMISSION_PER_SHARE = 0.005

class QuantitativeValue(Moonshot):
    """
    Value strategy modeled on Alpha Architect's QVAL ETF.

    Intended to be run with Sharadar fundamentals and prices.

    Strategy rules:

    1. Universe selection
      a. Starting universe: all NYSE stocks
      b. Exclude financials, ADRs, REITs
      c. Liquidity screen: select top N percent of stocks by dollar
         volume (N=60)
    2. Apply value screen: select cheapest N percent of stocks by
       enterprise multiple (EV/EBIT) (N=10)
    3. Rank by quality: of the value stocks, select the N percent
       with the highest quality, as ranked by Piotroski F-Score (N=50)
    4. Apply equal weights
    5. Rebalance portfolio quarterly
    """

    CODE = "qval"
    DB = "sharadar-us-stk-1d"
    DB_FIELDS = ["Close", "Volume"]
    DOLLAR_VOLUME_TOP_N_PCT = 60
    DOLLAR_VOLUME_WINDOW = 90
    UNIVERSES = "nyse-stk"
    EXCLUDE_UNIVERSES = ["nyse-financials", "nyse-adrs", "nyse-reits"]
    VALUE_TOP_N_PCT = 10
    QUALITY_TOP_N_PCT = 50
    REBALANCE_INTERVAL = "Q"
    COMMISSION_CLASS = USStockCommission

    def prices_to_signals(self, prices: pd.DataFrame):

        # Step 1.c: get a mask of stocks with adequate dollar volume
        closes = prices.loc["Close"]
        volumes = prices.loc["Volume"]
        avg_dollar_volumes = (closes * volumes).rolling(self.DOLLAR_VOLUME_WINDOW).mean()
        dollar_volume_ranks = avg_dollar_volumes.rank(axis=1, ascending=False, pct=True)
        have_adequate_dollar_volumes = dollar_volume_ranks <= (self.DOLLAR_VOLUME_TOP_N_PCT/100)

        # Step 2. Apply value screen: select cheapest N percent of stocks by
        # enterprise multiple (EV/EBITDA) (N=10)
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
            fields=["EVEBIT", "EBIT"],
            dimension="ART")
        enterprise_multiples = fundamentals.loc["EVEBIT"]
        ebits = fundamentals.loc["EBIT"]
        # Ignore negative earnings
        enterprise_multiples = enterprise_multiples.where(ebits > 0)
        # Only apply rankings to stocks with adequate dollar volume
        value_ranks = enterprise_multiples.where(have_adequate_dollar_volumes).rank(axis=1, ascending=True, pct=True)
        are_value_stocks = value_ranks <= (self.VALUE_TOP_N_PCT/100)

        # Step 3: Rank by quality: of the value stocks, select the N percent
        # with the highest quality, as ranked by Piotroski F-Score (N=50)
        f_scores = self.get_f_scores(closes)
        # Rank the value stocks by F-Score
        quality_ranks = f_scores.where(are_value_stocks).rank(axis=1, ascending=False, pct=True)
        long_signals = quality_ranks <= (self.QUALITY_TOP_N_PCT/100)

        return long_signals.astype(int)

    def get_f_scores(self, closes: pd.DataFrame):

        # Step 1: query relevant indicators
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
           dimension="ART", # As-reported trailing twelve month reports
           fields=[
               "ROA", # Return on assets
               "ASSETS", # Total Assets
               "NCFO", # Net Cash Flow from Operations
               "DE", # Debt to Equity Ratio
               "CURRENTRATIO", # Current ratio
               "SHARESWA", # Outstanding shares
               "GROSSMARGIN", # Gross margin
               "ASSETTURNOVER", # Asset turnover
           ])
        return_on_assets = fundamentals.loc["ROA"]
        total_assets = fundamentals.loc["ASSETS"]
        operating_cash_flows = fundamentals.loc["NCFO"]
        leverages = fundamentals.loc["DE"]
        current_ratios = fundamentals.loc["CURRENTRATIO"]
        shares_out = fundamentals.loc["SHARESWA"]
        gross_margins = fundamentals.loc["GROSSMARGIN"]
        asset_turnovers = fundamentals.loc["ASSETTURNOVER"]

        # Step 2: many Piotroski F-score components compare current to previous
        # values, so get DataFrames of previous values

        # Step 2.a: get a boolean mask of the first day of each newly reported fiscal
        # period
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
            dimension="ART", # As-reported trailing twelve month reports
            fields=["REPORTPERIOD"])
        fiscal_periods = fundamentals.loc["REPORTPERIOD"]
        are_new_fiscal_periods = fiscal_periods != fiscal_periods.shift()

        # Step 2.b: shift the ROAs forward one fiscal period by (1) shifting the ratios one day,
        # (2) keeping only the ones that fall on the first day of the newly reported
        # fiscal period, and (3) forward-filling
        previous_return_on_assets = return_on_assets.shift().where(are_new_fiscal_periods).fillna(method="ffill")

        # Step 2.c: Repeat for other indicators
        previous_leverages = leverages.shift().where(are_new_fiscal_periods).fillna(method="ffill")
        previous_current_ratios = current_ratios.shift().where(are_new_fiscal_periods).fillna(method="ffill")
        previous_shares_out = shares_out.shift().where(are_new_fiscal_periods).fillna(method="ffill")
        previous_gross_margins = gross_margins.shift().where(are_new_fiscal_periods).fillna(method="ffill")
        previous_asset_turnovers = asset_turnovers.shift().where(are_new_fiscal_periods).fillna(method="ffill")

        # Step 3: calculate F-Score components; each resulting component is a DataFrame
        # of booleans
        have_positive_return_on_assets = return_on_assets > 0
        have_positive_operating_cash_flows = operating_cash_flows > 0
        have_increasing_return_on_assets = return_on_assets > previous_return_on_assets
        have_more_cash_flow_than_incomes = operating_cash_flows / total_assets > return_on_assets
        have_decreasing_leverages = leverages < previous_leverages
        have_increasing_current_ratios = current_ratios > previous_current_ratios
        have_no_new_shares = shares_out <= previous_shares_out
        have_increasing_gross_margins = gross_margins > previous_gross_margins
        have_increasing_asset_turnovers = asset_turnovers > previous_asset_turnovers

        # Step 4: convert the booleans to integers and sum to get F-Score (0-9)
        f_scores = (
            have_positive_return_on_assets.astype(int)
            + have_positive_operating_cash_flows.astype(int)
            + have_increasing_return_on_assets.astype(int)
            + have_more_cash_flow_than_incomes.astype(int)
            + have_decreasing_leverages.astype(int)
            + have_increasing_current_ratios.astype(int)
            + have_no_new_shares.astype(int)
            + have_increasing_gross_margins.astype(int)
            + have_increasing_asset_turnovers.astype(int)
        )

        self.save_to_results("FScore", f_scores)
        return f_scores

    def signals_to_target_weights(self, signals: pd.DataFrame, prices: pd.DataFrame):
        # Step 4: equal weights
        daily_signal_counts = signals.abs().sum(axis=1)
        weights = signals.div(daily_signal_counts, axis=0).fillna(0)

        # Step 5: Rebalance quarterly
        # Resample daily to quarterly, taking the last day's signal
        # For pandas offset aliases, see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        weights = weights.resample(self.REBALANCE_INTERVAL).last()
        # Reindex back to daily and fill forward
        weights = weights.reindex(prices.loc["Close"].index, method="ffill")

        return weights

    def target_weights_to_positions(self, weights: pd.DataFrame, prices: pd.DataFrame):
        # Enter the position the day after the signal
        return weights.shift()

    def positions_to_gross_returns(self, positions: pd.DataFrame, prices: pd.DataFrame):

        closes = prices.loc["Close"]
        position_ends = positions.shift()

        # The return is the security's percent change over the period,
        # multiplied by the position.
        gross_returns = closes.pct_change() * position_ends

        return gross_returns
