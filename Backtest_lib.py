import pandas as pd
import numpy as np
from itertools import product
import plotly.express as px

class MA_Backtester():
    ''' Class for the vectorized backtesting of simple Long-Short trading strategies.
    
    Attributes
    ============
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade
    
    
    Methods
    =======
    get_data:
        imports the data.
        
    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).
        
    prepare_data:
        prepares the data for backtesting.
    
    run_backtest:
        runs the strategy backtest.
        
    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold.
        
    parametric_analysis:
        backtests strategy for different parameter values.
        
    '''    
    
    def __init__(self, filepath, symbol, start, end, tc):
        
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        
    def __repr__(self):
        return "MA_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)
        
    def get_data(self):
        ''' Imports the data.
        '''
        raw = pd.read_excel(self.filepath, parse_dates = ["Date"], index_col = "Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw
        
    def test_strategy(self, smas, print_result=True):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).
         
        Parameters
        ============
        smas: tuple (SMA_S, SMA_M, SMA_L)
            Simple Moving Averages to be considered for the strategy.
            
        '''
        
        self.SMA_S = smas[0]
        self.SMA_M = smas[1]
        self.SMA_L = smas[2]
        
        
        self.prepare_data(smas = smas)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        
        strategy_multiple = round(self.calculate_multiple(data.strategy), 6)
        bh_multiple =       round(self.calculate_multiple(data.returns), 6)
        outperf =           round(strategy_multiple - bh_multiple, 6)
        cagr =              round(self.calculate_cagr(data.strategy), 6)
        ann_mean =          round(self.calculate_annualized_mean(data.strategy), 6)
        ann_std =           round(self.calculate_annualized_std(data.strategy), 6)
        sharpe =            round(self.calculate_sharpe(data.strategy), 6)
        total_trades =      self.calculate_total_trades()
        max_drawdown =      round(self.calculate_mdd(), 2)
        winning_percentage= round(self.calculate_winning_percentage()*100, 2)
        
        metrics = {'Strategy Multiple':[strategy_multiple], 
                    'Outperformance':[outperf], 
                    'CAGR':[cagr],
                    'Annualized Mean':[ann_mean],
                    'Annualized Std':[ann_std],
                    'Sharpe Ratio':[sharpe],
                    'Total Trades':[total_trades],
                    'Max Drawdown':[max_drawdown],
                    'Winning Percentage':[winning_percentage]}
        
        metrics = pd.DataFrame(metrics)
        
        self.portfolio_metrics = metrics
        
        if print_result == True:
            display(self.portfolio_metrics)
    
    def prepare_data(self, smas):
        ''' Prepares the Data for Backtesting, returning the position for each timestep.
        '''
        ########################## Strategy-Specific #############################
        
        data = self.data[["Close", "returns"]].copy()
        data["SMA_S"] = data.Close.rolling(window = smas[0]).mean()
        data["SMA_M"] = data.Close.rolling(window = smas[1]).mean()
        data["SMA_L"] = data.Close.rolling(window = smas[2]).mean()
        
        data.dropna(inplace = True)
                
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        ##########################################################################
        
        self.results = data
    
    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
        
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
        
        self.results = data
    
    def plot_results(self, plotly=True):
        '''  Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            self.results.rename(columns={'creturns': 'Buy and Hold', 'cstrategy':'Strategy'}, inplace=True)
            if plotly:
                fig = px.line(self.results,
                       x=self.results.index,
                       y=['Buy and Hold', 'Strategy'])
                fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Returns")
                fig.show()
            else:
                title = "{} | TC = {}".format(self.symbol, self.tc)
                self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            
    def parametric_analysis(self, SMA_S_range, SMA_M_range, SMA_L_range):
        '''
        Backtests strategy for different parameter values
         
        Parameters
        ============
        SMA_S_range: tuple
            tuples of the form (start, end, step size).
        
        SMA_M_range: tuple
            tuples of the form (start, end, step size).
            
        SMA_L_range: tuple
            tuples of the form (start, end, step size).
        '''
        
        #self.metric = metric
        
        #if metric == "Multiple":
        #    performance_function = self.calculate_multiple
        #elif metric == "Sharpe":
        #    performance_function = self.calculate_sharpe
        
        SMA_S_range = range(*SMA_S_range)
        SMA_M_range = range(*SMA_M_range)
        SMA_L_range = range(*SMA_L_range)
        
        combinations = list(product(SMA_S_range, SMA_M_range, SMA_L_range))
        
        multiple = []
        volatility = []
        sharpe = []
        mdd = []
        total_trades_list = []
        w_percentage = []
        
        for comb in combinations:

            self.test_strategy(smas = comb, print_result=False)
            multiple.append(self.portfolio_metrics.loc[0, 'Strategy Multiple'])
            sharpe.append(self.portfolio_metrics.loc[0, 'Sharpe Ratio'])
            mdd.append(self.portfolio_metrics.loc[0, 'Max Drawdown'])
            w_percentage.append(self.portfolio_metrics.loc[0, 'Winning Percentage'])
            total_trades_list.append(self.portfolio_metrics.loc[0, 'Total Trades'])
    
        self.results_overview =  pd.DataFrame(data = np.array(combinations), columns = ["SMA_S", "SMA_M", "SMA_L"])
        self.results_overview["Strategy Multiple"] = multiple
        self.results_overview["Sharpe"] = sharpe
        self.results_overview["Max Drawdown"] = mdd
        self.results_overview["Winning Percentage"] = w_percentage
        self.results_overview["Total Trades"] = total_trades_list
        
        
            
    ############################## Performance ######################################
    
        
    def calculate_multiple(self, series):
        return np.exp(series.sum())
    
    def calculate_cagr(self, series):
        return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
    
    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)
    
    def calculate_total_trades(self):
        return self.results['trades'].sum().astype(int)
    
    def calculate_mdd(self):
        self.results['Max Return'] = self.results['cstrategy'].cummax()
        self.results['Drawdown'] = (self.results['cstrategy'] / self.results['Max Return']) - 1
        max_drawdown = 100*self.results['Drawdown'].min()
        return max_drawdown
    
    def calculate_winning_percentage(self):
        # Plot the percentage of winning months
        winningmonths_df = self.results[['cstrategy']].copy()
        winningmonths_df.columns = ['Open']
        winningmonths_df['Close'] = winningmonths_df['Open'].copy()
        winningmonths_df['Month'] = winningmonths_df.index.to_period('M').astype(str) + '-01'
        winningmonths_df = winningmonths_df.groupby(['Month']).agg({'Open':'first','Close':'last'}).reset_index()
        winningmonths_df['Month Counter'] = np.arange(len(winningmonths_df)) + 1
        winningmonths_df['Win'] = np.where(winningmonths_df['Close'] > winningmonths_df['Open'], 1, 0)
        winning_percentage = round(winningmonths_df['Win'].mean(), 4)
#         winningmonths_df['Winning Percentage'] = winningmonths_df['Win'].cumsum() / winningmonths_df['Month Counter']
        return winning_percentage
        