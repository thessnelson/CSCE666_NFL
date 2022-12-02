from bs4 import BeautifulSoup
import requests
import pandas as pd
from advancedStats import process, closeBrowser
import time
import random

# team abbreviation tags and range of years played
teams = [
        ("atl", 1970),
        ("crd", 1970),

        
        ("buf", 1970),
        ("car", 1995),
        ("chi", 1970),
        ("cin", 1970),
        ("cle", 1999),
        ("dal", 1970),
        ("den", 1970),
        ("det", 1970),
        ("gnb", 1970),
        ("htx", 2002),
        ("clt", 1970),
        ("jax", 1995),
        ("kan", 1970),

        ("sdg", 1970),
        
        ("mia", 1970),
        ("min", 1970),
        ("nwe", 1970),
        ("nor", 2015),
        ("nyg", 1970),
        ("nyj", 1970),
        ("phi", 1970),
        ("pit", 1970),
        ("rai", 1970),
        ("ram", 1970),
        ("rav", 1996),
        ("sfo", 1970),
        ("sea", 1976),
        ("tam", 1976),
        ("oti", 1970),
        ("was", 1970)
    ]

stat_labels = ["year", "week_num", "day_of_week", "kickoff_time", 
                "temperature", "wind_speed", "first_down_off", "rush_attempts",
                "rush_yds_off", "fumbles_lost", "completed_passes", "pass_attempts", 
                "pass_yds_off", "interceptions_thrown", "sack_yds_off", "penalty_yds_against",
                "plays_run_off", "avg_ko_return", "avg_punt_return", "avg_punt_dist", 
                "first_down_def", "rushes_defended", "rush_yds_def", "fumbles_forced",
                "completions_allowed", "passes_defended", "pass_yds_def", "interceptions_caught",
                "sack_yds_def", "penalty_yds_for", "plays_defended", "avg_ko_return_against",
                "avg_punt_return_against", "avg_punt_dist_against", "result"]

url = 'https://www.pro-football-reference.com'
for team in teams:
    print(team[0])

    data = pd.DataFrame(columns=stat_labels)

    abbr, start = team[0], team[1]
    save_file = ("data/%s.csv" % abbr)

    for year in range(start, 2022):
        print(year)
        path = ("%s/teams/%s/%s.htm" % (url,abbr,year))

        r = requests.get(path)
        soup = BeautifulSoup(r.content, 'html.parser')

        tables = soup.findChildren('table')
        if len(tables) < 2: continue
        game_table = tables[1]

        games = game_table.findChildren('tr')
        games = games[2:]

        for game in games:
            row = [year]
            stats = game.findChildren('td')
            if(stats[8].text == "Bye Week"):
                continue
            if(stats[1].text == "Playoffs"):
                break

            weekNum = game.findChild('th').text
            row.append(int(weekNum))
            row.append(stats[0].text)
            row.append(stats[2].text)
            home = stats[7].text != '@'

            box_wrapper = stats[3].find('a', href=True)
            boxscore_path = box_wrapper['href']
            boxscore_url = ("%s%s" %(url,boxscore_path))
            
            full_row = process(data=row, home=home, url=boxscore_url)
            full_row.append(stats[4].text)

            print(full_row)
            data.loc[len(data)] = full_row
            data.to_csv(save_file, index=False)
            time.sleep(random.randrange(1,3))

closeBrowser()