import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

# pandas suppress warnings
pd.options.mode.chained_assignment = None

# from tkinter import font as tkfont

# Choose the two CSV files to test for, named A and B, and choose the duration of the lag and the output file name. Display these with a single GUI window and make to import the required libraries.

def main():
    def select_file(var):
        filename = filedialog.askopenfilename()
        var.set(filename)

    def validate_lag(P):
        if P.isdigit() or P == "":
            return True
        else:
            return False
        
    def calculate_lagged_correlation(spy_path, vix_path, output_file, low_thres, high_thres, lookback, lookahead, time_diff):

        if not os.path.exists(spy_path) or not os.path.exists(vix_path) or time_diff < 0:
            return
        
        asset_a_df = pd.read_csv(spy_path)
        asset_b_df = pd.read_csv(vix_path)

        # convert Date column to datetime
        asset_a_df['Time'] = pd.to_datetime(asset_a_df['Time'])

        # sort by date
        asset_a_df.sort_values(by='Time', inplace=True)
        asset_a_df.reset_index(drop=True, inplace=True)
        asset_a_change = asset_a_df[['Time', 'Open']]

        asset_b_df['Time'] = pd.to_datetime(asset_b_df['Time'])
        asset_b_df.sort_values(by='Time', inplace=True)
        asset_b_df.reset_index(drop=True, inplace=True)
        asset_b_change = asset_b_df[['Time', 'Open']]

        times = {d: i for i, d in enumerate(asset_a_change['Time'])}

        # make a new row of the Time column with the Time subtracted by lookback minutes
        lookback_mask = asset_a_change['Time'] - pd.Timedelta(minutes=lookback)

        # make a mask if the time is in Times. If it is, give the difference in rows, if not, give NaN
        lookback_mask = lookback_mask.map(times.get).dropna()

        lookback_pct_change = pd.Series(index=asset_a_change.index, dtype='float64')
        lookback_change = pd.Series(index=asset_a_change.index, dtype='float64')

        for i in lookback_mask.index:
            lookback_pct_change[i] = (asset_a_change.loc[i, 'Open'] / asset_a_change.loc[lookback_mask[i], 'Open'] - 1) * 100
            lookback_change[i] = asset_a_change.loc[i, 'Open'] - asset_a_change.loc[lookback_mask[i], 'Open']

        asset_a_change['Chg%'] = lookback_pct_change
        asset_a_change['Chg'] = lookback_change

        # split datetime into date and time
        asset_a_change['Date'] = asset_a_change['Time'].dt.date
        asset_a_change['Time'] = asset_a_change['Time'].dt.time

        # convert both to datetime objects
        asset_a_change['Date'] = pd.to_datetime(asset_a_change['Date'])
        asset_a_change['Time'] = pd.to_datetime(asset_a_change['Time'], format='%H:%M:%S')

        asset_a_change_pivot = asset_a_change.pivot(index='Date', columns='Time', values='Chg%')

        # drop columns that are all NaN
        asset_a_change_pivot = asset_a_change_pivot.dropna(axis=1, how='all')

        # drop rows that are all NaN
        asset_a_change_pivot = asset_a_change_pivot.dropna(axis=0, how='all')
        asset_a_change_pivot = asset_a_change_pivot.applymap(lambda x: 1 if high_thres[1] >= x >= high_thres[0] else -1 if low_thres[0] <= x <= low_thres[1] else pd.NaT)

       
        times_b = {d: i for i, d in enumerate(asset_b_change['Time'])}

        # make a new row of the Time column with the Time subtracted by lookback minutes
        lookahead_mask = asset_b_change['Time'] + pd.Timedelta(minutes=lookahead)

        # make a mask if the time is in Times. If it is, give the difference in rows, if not, give NaN
        lookahead_mask = lookahead_mask.map(times_b.get).dropna()

        # for the rows in lookback_mask, calculate the difference between the Open price of a time and the Open price of the time in lookback_mask
        lookahead_pct_change = pd.Series(index=asset_b_change.index, dtype='float64')
        lookahead_change = pd.Series(index=asset_b_change.index, dtype='float64')

        for i in lookahead_mask.index:
            lookahead_pct_change[i] = (asset_b_change.loc[lookahead_mask[i], 'Open'] / asset_b_change.loc[i, 'Open'] - 1) * 100
            lookahead_change[i] = asset_b_change.loc[i, 'Open'] - asset_b_change.loc[lookahead_mask[i], 'Open']

        asset_b_change['Chg%'] = lookahead_pct_change
        asset_b_change['Chg'] = lookahead_change


        asset_b_change['Date'] = asset_b_change['Time'].dt.date
        asset_b_change['Time'] = asset_b_change['Time'].dt.time

        asset_b_change['Date'] = pd.to_datetime(asset_b_change['Date'])
        asset_b_change['Time'] = pd.to_datetime(asset_b_change['Time'], format='%H:%M:%S')

        asset_b_change_pivot = asset_b_change.pivot(index='Date', columns='Time', values='Chg%')
        asset_b_vals_pivot = asset_b_change.pivot(index='Date', columns='Time', values='Chg')

        asset_b_change_pivot = asset_b_change_pivot.dropna(axis=1, how='all')
        asset_b_change_pivot = asset_b_change_pivot.dropna(axis=0, how='all')

        asset_b_change_orig = asset_b_change_pivot.copy()

        asset_b_change_pivot = asset_b_change_pivot.applymap(lambda x: 1 if x > 0 else -1 if x <= 0 else pd.NaT)

        # For the two pivot tables, find the Date rows and Time columns that are in both tables
        common_dates = asset_a_change_pivot.index.intersection(asset_b_change_pivot.index)
        common_times = asset_a_change_pivot.columns.intersection(asset_b_change_pivot.columns)

        # slice the pivot tables to only include the common dates and times
        asset_a_change_pivot = asset_a_change_pivot.loc[common_dates, common_times]
        asset_b_change_pivot = asset_b_change_pivot.loc[common_dates, common_times]

        # set the rows and columns of asset_b_vals_pivot to match the rows and columns of asset_a_change_pivot
        # asset_b_vals_pivot = asset_b_vals_pivot.loc[common_dates, common_times]
        # asset_b_change_orig = asset_b_change_orig.loc[common_dates, common_times]

        probs_df = pd.DataFrame(columns = ['a_down_b_down',  'a_down_b_up', 'a_up_b_down', 'a_up_b_up'])
        dist_df = pd.DataFrame(columns = ['a_down_b_down',  'a_down_b_up', 'a_up_b_down', 'a_up_b_up'])
        trade_df = pd.DataFrame(columns = ['a_down_b_down',  'a_down_b_up', 'a_up_b_down', 'a_up_b_up'])

        # for each column in asset_a_change_pivot
        for col in asset_a_change_pivot.columns:
            asset_a_col = asset_a_change_pivot[col]

            # shift datetime.time by time_diff
            b_col = col + pd.Timedelta(minutes=time_diff)

            if b_col not in asset_b_change_pivot.columns:
                continue

            asset_b_col = asset_b_change_pivot[b_col]

            # get the combinations in tuple pairs, and any pairs with NaN are dropped
            combo_col = pd.DataFrame(list(zip(asset_a_col, asset_b_col))).dropna()

            asset_a_prob = combo_col[0].value_counts(normalize=True)
            conditional_probs = combo_col.groupby(0)[1].value_counts(normalize=True)

            conditional = conditional_probs.unstack().fillna(0).to_dict()

            full_conditional = {}

            full_conditional_stats = {}

            trade_stats = {}

            rename = {-1: "down", 1: "up"}

            for i in [-1, 1]:
                for j in [-1, 1]:
                    if i not in conditional or j not in conditional[i]:
                        full_conditional[f"a_{rename[j]}_b_{rename[i]}"] = 0
                        full_conditional_stats[f"a_{rename[j]}_b_{rename[i]}"] = [0,0,0]
                        trade_stats[f"a_{rename[j]}_b_{rename[i]}"] = []

                    else:
                        full_conditional[f"a_{rename[j]}_b_{rename[i]}"] = conditional[i][j]

                        # get the rows for each combination of a and b
                        a_down_b_down = combo_col[(combo_col[0] == j) & (combo_col[1] == i)]
                        
                        # get the count, mean, and std from asset_b_vals_pivot for the rows in a_down_b_down for the hour
                        a_down_b_down_vals = asset_b_change_orig.iloc[a_down_b_down.index][b_col].astype(float).describe().fillna(0).to_dict()
                        full_conditional_stats[f"a_{rename[j]}_b_{rename[i]}"] = [int(a_down_b_down_vals['count']), a_down_b_down_vals['mean'], a_down_b_down_vals['std']]

                        trade_history = asset_b_change_orig.iloc[a_down_b_down.index][b_col].astype(float).to_dict()
                        
                        # convert the dictionary to a list of tuples, converting Timestamp to string of only the date
                        trade_history = [(str(k.date()), v) for k, v in trade_history.items()]

                        trade_stats[f"a_{rename[j]}_b_{rename[i]}"] = trade_history

            name = col.strftime('%H:%M:%S')
            
            # # add the percentages to the dataframe using the name as the index
            probs_df.loc[name] = full_conditional
            dist_df.loc[name] = full_conditional_stats
            trade_df.loc[name] = trade_stats

        # split the dates by year and generate the same dataframes as above for each year
        years = asset_a_change_pivot.index.year.unique()

        probs_df_yearly = {}

        for year in years:
            asset_a_change_pivot_year = asset_a_change_pivot[asset_a_change_pivot.index.year == year]
            asset_b_change_pivot_year = asset_b_change_pivot[asset_b_change_pivot.index.year == year]
            asset_b_vals_pivot_year = asset_b_vals_pivot[asset_b_vals_pivot.index.year == year]

            probs_df_yearly[year] = pd.DataFrame(columns = ['a_down_b_down',  'a_down_b_up', 'a_up_b_down', 'a_up_b_up'])

            time_diff = 0

            # for each column in asset_a_change_pivot
            for col in asset_a_change_pivot_year.columns:
                asset_a_col = asset_a_change_pivot_year[col]

                # shift datetime.time by time_diff
                b_col = col + pd.Timedelta(minutes=time_diff)

                if b_col not in asset_b_change_pivot_year.columns:
                    continue

                asset_b_col = asset_b_change_pivot_year[b_col]

                # get the combinations in tuple pairs, and any pairs with NaN are dropped
                combo_col = pd.DataFrame(list(zip(asset_a_col, asset_b_col))).dropna()

                asset_a_prob = combo_col[0].value_counts(normalize=True)
                conditional_probs = combo_col.groupby(0)[1].value_counts(normalize=True)

                conditional = conditional_probs.unstack().fillna(0).to_dict()

                full_conditional = {}

                full_conditional_stats = {}

                trade_stats = {}

                rename = {-1: "down", 1: "up"}

                for i in [-1, 1]:
                    for j in [-1, 1]:
                        if i not in conditional or j not in conditional[i]:
                            full_conditional[f"a_{rename[j]}_b_{rename[i]}"] = 0
                            full_conditional_stats[f"a_{rename[j]}_b_{rename[i]}"] = [0,0,0]
                            trade_stats[f"a_{rename[j]}_b_{rename[i]}"] = []

                        else:
                            full_conditional[f"a_{rename[j]}_b_{rename[i]}"] = conditional[i][j]

                            # get the rows for each combination of a and b
                            a_down_b_down = combo_col[(combo_col[0] == j) & (combo_col[1] == i)]
                            
                            # get the count, mean, and std from asset_b_vals_pivot for the rows in a_down_b_down for the hour
                            a_down_b_down_vals = asset_b_change_orig.iloc[a_down_b_down.index][b_col].astype(float).describe().fillna(0).to_dict()

                            full_conditional_stats[f"a_{rename[j]}_b_{rename[i]}"] = [int(a_down_b_down_vals['count']), a_down_b_down_vals['mean'], a_down_b_down_vals['std']]

                            trade_history = asset_b_vals_pivot_year.iloc[a_down_b_down.index][b_col].astype(float).to_dict()
                            
                            # convert the dictionary to a list of tuples, converting Timestamp to string of only the date
                            trade_history = [(str(k.date()), v) for k, v in trade_history.items()]

                            trade_stats[f"a_{rename[j]}_b_{rename[i]}"] = trade_history

                name = col.strftime('%H:%M:%S')

                # add the percentages to the dataframe using the name as the index
                probs_df_yearly[year].loc[name] = full_conditional

        # get base file name without extension
        output_file_base = os.path.splitext(output_file)[0]

        probs_df.to_csv(output_file)
        dist_df.to_csv(f"{output_file_base}_stats.csv")
        trade_df.to_csv(f"{output_file_base}_trades.csv")

        for year in years:
            probs_df_yearly[year].to_csv(f"{output_file_base}_{year}.csv")

        # print full output path
        print("\n\n")
        print("---------------------------------------")
        print(f"Output file: {os.path.abspath(output_file)}")
        print("\n---------------------------------------\n\n")

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    file_a = filedialog.askopenfilename(title="Select CSV A") 
    print(f"File A: {file_a}")
    file_b = filedialog.askopenfilename(title="Select CSV B")
    print(f"File B: {file_b}")
    output_file = input("Enter the output file name: ")

    # low_thresh = abs(float(input("Enter range for negative movements (separate with comma, no spaces): ")))
    # high_thresh = abs(float(input("Enter range for for positive movements (separate with comma, no spaces): ")))
    low_thres = input("Enter range for negative movements (Enter in format X,Y with no spaces): ")
    high_thres = input("Enter range for for positive movements (Enter in format X,Y with no spaces): ")

    low_thres = low_thres.split(",")
    high_thres = high_thres.split(",")

    low_thres = [float(x) for x in low_thres]
    high_thres = [float(x) for x in high_thres]

    lookback = int(input("Enter the lookback period in minutes: "))
    lookahead = int(input("Enter the lookahead period in minutes: "))

    time_diff = int(input("Enter the gap between the lookback and lookahead windows (You probably want this at 0): "))

    calculate_lagged_correlation(file_a, file_b, output_file, low_thres, high_thres, lookback, lookahead, time_diff)

    root.destroy()

if __name__ == "__main__":
    main()