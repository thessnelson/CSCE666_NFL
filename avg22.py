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

for root, dirs, files in os.walk("testData"):
    for filename in files:
        filepath = ("%s/%s" % (root, filename))
        df = pd.read_csv(filepath)
        meta = df[selectToKeep]
        toAvg = df[selectToAvg]
        results = df['result']
        totals = np.zeros(len(selectToAvg))
        averages = pd.DataFrame(columns=selectToAvg)
        averages.loc[len(averages)] = totals
        for i in range(len(toAvg)-1):
            totals += toAvg.iloc[i]
            # if i==0: continue
            averages.loc[len(averages)] = totals/(i+1)

        meta = meta.drop(index=0)
        averages = averages.drop(index=0)
        results = results.drop(index=0)
        merged = pd.merge(meta, averages, left_index=True, right_index=True)
        merged = pd.merge(merged, results, left_index=True, right_index=True)
        
        writefile = 'avgData22/'+filename
        merged.to_csv(writefile, index=False)

