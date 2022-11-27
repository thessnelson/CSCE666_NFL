import os
import pandas as pd
import numpy as np

selectToKeep = ['year','week_num','day_of_week','kickoff_time','temperature','wind_speed']

selectToAvg = ['first_down_off','rush_attempts','rush_yds_off','fumbles_lost',
                'completed_passes','pass_attempts','pass_yds_off','interceptions_thrown',
                'sack_yds_off','penalty_yds_against','plays_run_off','avg_ko_return',
                'avg_punt_return','avg_punt_dist','first_down_def','rushes_defended',
                'rush_yds_def','fumbles_forced','completions_allowed','passes_defended',
                'pass_yds_def','interceptions_caught','sack_yds_def','penalty_yds_for',
                'plays_defended','avg_ko_return_against','avg_punt_return_against',
                'avg_punt_dist_against']
        
for root, dirs, files in os.walk("data"):
    for filename in files:
        writefile = 'avgData/'+filename
        filepath = ("%s/%s" % (root, filename))
        for i in range(1970,2022):
            df = pd.read_csv(filepath)
            thisYear = df.loc[df['year'] == i]
            if thisYear.empty: continue
            thisYear = thisYear.reset_index(drop=True)

            meta = thisYear[selectToKeep]
            toAvg = thisYear[selectToAvg]
            indexes = toAvg.index.values.tolist()
            results = thisYear['result']
            weekNums = thisYear['week_num'].tolist()
            # print(weekNums)
            # input()
            
            totals = np.zeros(len(selectToAvg))
            averages = pd.DataFrame(columns=selectToAvg)
            

            for j in range(len(indexes)):
                totals += toAvg.loc[indexes[j]]
                averages.loc[len(averages)] = totals/(j+1)
            # print(indexes)
            # averages.set_index(indexes)
            
            meta = meta.iloc[1: , :]
            averages = averages.iloc[1: , :]
            # weekNums = weekNums[1:]
            # averages['week_num'] = weekNums
            results = results.iloc[1:]
            # results['week_num'] = weekNums
            # meta = meta.reset_index(inplace=True)
            # averages = averages.reset_index(inplace=True)
            # results = results.reset_index(inplace=True)

            # print(i)
            # print(meta)
            # print(averages)
            # print(results)
            # input()


            merged = pd.merge(meta, averages, left_index=True, right_index=True)
            # print(merged)
            # print(results)
            # input()
            merged = pd.merge(merged, results, left_index=True, right_index=True)
            # print(merged)
            # input()
            merged.to_csv(writefile, mode='a', header=not os.path.exists(writefile), index=False)
        