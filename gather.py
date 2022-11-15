from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import random

# team abbreviation tags and range of years played
teams = [
        ("crd", 1940, 2022),
        ("atl", 1966, 2022),
        ("rav", 1996, 2022),
        ("buf", 1960, 2022),
        ("car", 1995, 2022),
        ("chi", 1940, 2022),
        ("cin", 1968, 2022),
        ("cle", 1999, 2022),
        ("dal", 1960, 2022),
        ("den", 1960, 2022),
        ("det", 1940, 2022),
        ("gnb", 1940, 2022),
        ("htx", 2002, 2022),
        ("clt", 1953, 2022),
        ("jax", 1995, 2022),
        ("kan", 1960, 2022),
        ("rai", 1960, 2022),
        ("sdg", 1960, 2022),
        ("ram", 1944, 2022),
        ("mia", 1966, 2022),
        ("min", 1961, 2022),
        ("nwe", 1960, 2022),
        ("nor", 1967, 2022),
        ("nyg", 1940, 2022),
        ("nyj", 1960, 2022),
        ("phi", 1940, 2022),
        ("pit", 1945, 2022),
        ("sfo", 1946, 2022),
        ("sea", 1976, 2022),
        ("tam", 1976, 2022),
        ("oti", 1960, 2022),
        ("was", 1940, 2022)
    ]

# make look like user
headers = requests.utils.default_headers()
headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})

# labels and corresponding index in html table
stat_labels = ["year", "week_num", "game_day_of_week", "game_time", "opp", "first_down_off", "pass_yds_off", "rush_yds_off", "to_off", "first_down_def", "pass_yds_def", "rush_yds_def", "to_def", "result"]
indexes = [0, 2, 8, 11, 13, 14, 15, 16, 18, 19, 20, 4]

# iterate through team pages
url = 'https://www.pro-football-reference.com/teams/'
for team in teams:
    print(team[0])
    if team[0] == "nwe": continue

    data = pd.DataFrame(columns=stat_labels)

    abbr, start, end = team[0], team[1], team[2]
    filename = abbr + ".csv"

    for year in range(start,end):
        print(year)
        tag = abbr + "/" + str(year)+".htm"
        path = url + tag

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
            row.append(weekNum)
            
            for i in range(len(indexes)):
                if(stats[indexes[i]].text != ""):
                    row.append(stats[indexes[i]].text)
                else:
                    row.append("0")
            data.loc[len(data)] = row

        data.to_csv(filename, index=False)
        time.sleep(random.randint(1,7))