from selenium import webdriver  
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import NoSuchElementException  
from selenium.webdriver.common.keys import Keys  
from bs4 import BeautifulSoup

options = Options()
options.binary_location = r"C:/Program Files/Mozilla Firefox/firefox.exe"
global browser
browser = webdriver.Firefox(options=options, executable_path="D:\School\Fall 22\Pattern Analysis\geckodriver.exe")

def getTables(url):
    browser.get(url=url)
    html_source = browser.page_source

    soup = BeautifulSoup(html_source, 'html.parser')
    tables = soup.findAll('table',{'id':True})

    game_info_table, team_stat_table, returns_table, kicking_table = 0,0,0,0

    for i in range(len(tables)):
        if tables[i]['id'] == 'game_info':
            game_info_table = tables[i]
        elif tables[i]['id'] == 'team_stats':
            team_stat_table = tables[i]
        elif tables[i]['id'] == 'returns':
            returns_table = tables[i]
        elif tables[i]['id'] == 'kicking':
            kicking_table = tables[i]
    # print(game_info_table)
    # print("\n\n")
    # print(team_stat_table)
    return (game_info_table, team_stat_table, returns_table, kicking_table)

def getWeather(table, year):
    game_info_rows = table.findChildren('tr')
    temperature, wind, weather_stat = -1, -1, -1
    if year < 1999:
        for row in game_info_rows:
            a_tag = row.findChild('a')
            if a_tag is not None and a_tag.text == 'Weather*':
                weather_stat = row.findChild('td').text
                break
    else:
        for row in game_info_rows:
            if row.findChild('th') is not None and row.findChild('th').text == 'Weather':
                weather_stat = row.findChild('td').text
    
    if weather_stat != -1:
        spl = weather_stat.split(' ')
        for i in range(len(spl)):
            if spl[i] == 'degrees' or spl[i] == 'degrees,':
                temperature = int(spl[i-1])
            if spl[i] == 'mph' or spl[i] == 'mph,':
                wind = int(spl[i-1])
    if temperature == -1:
        temperature = 70
    if wind == -1:
        wind = 0
    return(temperature, wind)

def getStats(table):
    team_stat_rows = table.findChildren('tr')
    away_abbr = table.findChildren('th')[1].text

    away, home = [], []
    awaySnaps, homeSnaps = 0, 0

    # First downs
    firstDowns = team_stat_rows[1].findChildren('td')
    away.append(int(firstDowns[0].text))
    home.append(int(firstDowns[1].text))

    # Rush attempts, yards
    rushStats = team_stat_rows[2].findChildren('td')
    awayRush = rushStats[0].text.split('-')
    homeRush = rushStats[1].text.split('-')
    if len(awayRush)==3:
        away.append(int(awayRush[0]))
        away.append(int(awayRush[1]))
    elif len(awayRush)==4:
        away.append(int(awayRush[0]))
        awayRush[2] = '-'+awayRush[2]
        away.append(int(awayRush[2]))
    if len(homeRush)==3:
        home.append(int(homeRush[0]))
        home.append(int(homeRush[1]))
    elif len(homeRush)==4:
        home.append(int(homeRush[0]))
        homeRush[2] = '-'+homeRush[2]
        home.append(int(homeRush[2]))
    awaySnaps += int(awayRush[0])
    homeSnaps += int(homeRush[0])

    # Fumbles lost
    fumbles = team_stat_rows[7].findChildren('td')
    awayFumble = fumbles[0].text.split('-')
    away.append(int(awayFumble[1]))
    homeFumble = fumbles[1].text.split('-')
    home.append(int(homeFumble[1]))

    # Completed passes, attempted passes, total passing yards, interceptions thrown
    passStats = team_stat_rows[3].findChildren('td')
    awayPass = passStats[0].text.split('-')
    homePass = passStats[1].text.split('-')
    away.append(int(awayPass[0]))
    away.append(int(awayPass[1]))
    away.append(int(awayPass[2]))
    away.append(int(awayPass[4]))
    home.append(int(homePass[0]))
    home.append(int(homePass[1]))
    home.append(int(homePass[2]))
    home.append(int(homePass[4]))
    awaySnaps += int(awayPass[1])
    homeSnaps += int(homePass[1])

    # Sack yards
    sacks = team_stat_rows[4].findChildren('td')
    awaySacks = sacks[0].text.split('-')
    homeSacks = sacks[1].text.split('-')
    away.append(int(awaySacks[1]))
    home.append(int(homeSacks[1]))
    awaySnaps += int(awaySacks[0])
    homeSnaps += int(homeSacks[0])

    # Penalty yards
    pens = team_stat_rows[9].findChildren('td')
    awayPens = pens[0].text.split('-')
    homePens = pens[1].text.split('-')
    away.append(int(awayPens[1]))
    home.append(int(homePens[1]))

    # total offensive plays
    away.append(awaySnaps)
    home.append(homeSnaps)

    # print(away)
    # print(home)
    return (away, home, away_abbr)

def getReturns(table, away_abbr):
    if table == 0:
        return([0,0],[0,0])
    returns_rows = table.findChildren('tr')
    away, home = [], []
    away_kick_ret, away_kick_ret_yds, away_punt_ret, away_punt_ret_yds = 0, 0, 0, 0
    home_kick_ret, home_kick_ret_yds, home_punt_ret, home_punt_ret_yds = 0, 0, 0, 0
    i=2
    for i in range(2, len(returns_rows)):
        row = returns_rows[i]
        if len(row.findChildren('th')) > 1:
            continue
        stats = row.findChildren('td')
        if stats[0].text == away_abbr:
            if(stats[1].text) != '': away_kick_ret += int(stats[1].text)
            if(stats[2].text) != '': away_kick_ret_yds += int(stats[2].text)
            if(stats[6].text) != '': away_punt_ret += int(stats[6].text)
            if(stats[7].text) != '': away_punt_ret_yds += int(stats[7].text)
        else:
            if(stats[1].text) != '': home_kick_ret += int(stats[1].text)
            if(stats[2].text) != '': home_kick_ret_yds += int(stats[2].text)
            if(stats[6].text) != '': home_punt_ret += int(stats[6].text)
            if(stats[7].text) != '': home_punt_ret_yds += int(stats[7].text)
        # if i == (len(returns_rows)-1):
        #     break
        i+=1

    if away_kick_ret == 0:
        away.append(0)
    else:
        away.append(away_kick_ret_yds/away_kick_ret)
    if away_punt_ret == 0:
        away.append(0)
    else:
        away.append(away_punt_ret_yds/away_punt_ret)
    if home_kick_ret == 0:
        home.append(0)
    else:
        home.append(home_kick_ret_yds/home_kick_ret)
    if home_punt_ret == 0:
        home.append(0)
    else:
        home.append(home_punt_ret_yds/home_punt_ret)
    return(away, home)

def getPunts(table):
    punt_rows = table.findChildren('tr')
    away, home = [], []
    punts, punt_yds = 0, 0
    i=2
    while True:
        row = punt_rows[i]
        if len(row.findChildren('th')) > 1:
            if punts == 0:
                away.append(0)
            else:
                away.append(punt_yds/punts)
            i+=2
            punts, punt_yds = 0, 0
            continue
        stats = row.findChildren('td')
        if(stats[5].text) != '': punts += int(stats[5].text)
        if(stats[6].text) != '': punt_yds += int(stats[6].text)
        if i == (len(punt_rows)-1):
            break
        i+=1

    if punts == 0:
        home.append(0)
    else:
        home.append(punt_yds/punts)
    # print(away)
    # print(home)
    return(away, home)

def closeBrowser():
    browser.close()


def process(data, home, url):
    year = data[0]
    game, team, ret, kick = getTables(url=url)
    temp, wind = getWeather(game, year)
    data.append(temp)
    data.append(wind)
    away_stats, home_stats, away_abbr = getStats(team)
    away_returns, home_returns = getReturns(ret, away_abbr)
    away_punts, home_punts = getPunts(kick)
    if home:
        for stat in home_stats:
            data.append(stat)
        for ret in home_returns:
            data.append(ret)
        for punt in home_punts:
            data.append(punt)
        for stat in away_stats:
            data.append(stat)
        for ret in away_returns:
            data.append(ret)
        for punt in away_punts:
            data.append(punt)
    else:
        for stat in away_stats:
            data.append(stat)
        for ret in away_returns:
            data.append(ret)
        for punt in away_punts:
            data.append(punt)
        for stat in home_stats:
            data.append(stat)
        for ret in home_returns:
            data.append(ret)
        for punt in home_punts:
            data.append(punt)
    return data

# process([1992, 5, 'Sun', '1:00'], True, 'https://www.pro-football-reference.com/boxscores/199210040crd.htm')