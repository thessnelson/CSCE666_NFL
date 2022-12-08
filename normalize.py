import pandas as pd
import os

# get all teams
teams = [#'atl', 
        'buf', 'car', 'chi', 'cin', 'cle', 'clt', 'crd', 'dal', 'den', 'det', 'gnb', 'htx', 'jax', 'kan', 'mia', 'min', 'nor', 'nwe', 'nyg', 'nyj', 'oti', 'phi', 'pit', 'rai', 'ram', 'rav', 'sdg', 'sea', 'sfo', 'tam', 'was']

# for splitting dataframe by columns
selectToKeep = ['year','week_num','day_of_week','kickoff_time','temperature','wind_speed']
selectToAvg = ['first_down_off','rush_attempts','rush_yds_off','fumbles_lost',
                'completed_passes','pass_attempts','pass_yds_off','interceptions_thrown',
                'sack_yds_off','penalty_yds_against','plays_run_off','avg_ko_return',
                'avg_punt_return','avg_punt_dist',
                'first_down_def','rushes_defended', 'rush_yds_def','fumbles_forced',
                'completions_allowed','passes_defended','pass_yds_def','interceptions_caught',
                'sack_yds_def','penalty_yds_for','plays_defended','avg_ko_return_against',
                'avg_punt_return_against','avg_punt_dist_against']

# compile all data into one dataframe
allStats = pd.read_csv('data/atl.csv')
for team in teams:
    df = pd.read_csv('data/'+team+'.csv')
    allStats = pd.concat([allStats, df])
    df = pd.read_csv('testData/'+team+'.csv')
    allStats = pd.concat([allStats, df])

# get average stats for each year
avgData = pd.DataFrame(columns=selectToAvg)
for year in range(1970,2023):
    print(year)
    yearStats = allStats.loc[allStats['year'] == year]
    thisYear = []
    for stat in selectToAvg:
        mean = yearStats[stat].mean()
        thisYear.append(mean)
    avgData.loc[len(avgData)] = thisYear
avgData.insert(0, 'year', range(1970,2023))
print(avgData)

# normalize each team's stats by year
for root, dirs, files in os.walk("testData"):
    for filename in files:
        print(filename)
        save = pd.DataFrame(columns=selectToKeep+selectToAvg+['result'])
        writefile = 'normData22/'+filename
        filepath = ("%s/%s" % (root, filename))
        for year in range(2022,2023):
            df = pd.read_csv(filepath)
            thisYear = df.loc[df['year'] == year]
            thisYear.reset_index(drop=True, inplace=True)
            if thisYear.empty: continue
            avg = avgData.loc[avgData['year'] == year]

            for game in range(len(thisYear)):
                for stat in selectToAvg:
                    if(stat=='year'): continue
                    thisYear.loc[game, stat] = thisYear.loc[game, stat] / avg[stat].values[0]

            save = pd.concat([save, thisYear])
        save.to_csv(writefile, index=False)
