#File:          animePlanetData.py
#Name:          Jacob Strickland
#Date Created:  Feb 6, 2020
#Date Modified: Feb 9, 2020
#Description:   The program takes anime data from anime-planet.com and stores it
#               into a .csv file in the format: {Anime title, Anime rank, Media type,
#               Year, Studio, Rating out of 10, Number of users that have rated it,
#               [Similar anime the site recommends], [Tags]}

#import libraries
import requests
import csv
import os
import time
from bs4 import BeautifulSoup

start_time = time.time() #to track how long the program takes to run

#opening up the csv file
csv_path = os.path.join(r'D:\Python','anime-planet-ranking1.csv')
with open(csv_path, 'w', newline='', encoding="utf-8") as csv_file:
    data_writer = csv.writer(csv_file)
    f = open(csv_path)
    headers = ["Name", "Rank", "Media Type", "Year", "Studio", "Rating of 10", "Number of Ratings", "Tags"]
    data_writer.writerow(headers)

    pages = [] #list used to store the webpages on the top list

    for i in range(1,337): #range is the number of pages for the top list, there are 400 pages but after 326 they don't have ratings
        url = 'https://www.anime-planet.com/anime/top-anime?page=' + str(i)
        pages.append(url)

    count = 1
    for item in pages:
        pageTop = requests.get(item)
        soup = BeautifulSoup(pageTop.text, 'html.parser')

        anime_name_list = soup.find(class_='pure-1 md-4-5')
        anime_name_list_items = anime_name_list.find_all('a')                   #used to find the title of the anime
        anime_rank_list_items = anime_name_list.find_all(class_='tableRank')    #used to find the rank of the anime
        anime_type_list_items = anime_name_list.find_all(class_='tableType')    #used to find the media type of the anime
        anime_year_list_items = anime_name_list.find_all(class_='tableYear')    #used to find the year of the anime
        
        tableIndex = 1
        for anime_name in anime_name_list_items:
            names = anime_name.contents[0]                                      #put the anime name into a variable and get rid of the tags
            rank = anime_rank_list_items[tableIndex].contents[0]                #put the anime rank into a variable and get rid of the tags
            typeMedia = anime_type_list_items[tableIndex].contents[0]           #put the anime media type into a variable and get rid of the tags

            if str(typeMedia) != 'OVA' and str(typeMedia) != 'Music Video' and str(typeMedia) != 'Web':             #don't want to include these media types
                mainPageURL = 'https://www.anime-planet.com' + anime_name.get('href')                               #get the url of the individual anime page
                pageMain = requests.get(mainPageURL)
                soup2 = BeautifulSoup(pageMain.text, 'html.parser')

                #get the rating stats for the anime and store it into a list
                anime_stats_list = []

                if(soup2.find(itemprop='ratingValue') != None):                                                     #do a check to see if there are ratings
                    anime_stats_list.append(soup2.find(itemprop='ratingValue').get("content"))                      #get the average user rating of the anime out of 10
                    anime_stats_list.append(soup2.find(itemprop='ratingCount').get("content"))                      #get the number of users that have rated the anime
                else:
                    anime_stats_list.append('N/A')
                    anime_stats_list.append('N/A')

                anime_year_check = soup2.find(class_='iconYear')

                if(anime_year_check.contents[0][1:-1] != 'TBA' and anime_year_check.contents[0][1:5] != '2020'):    #do a check to see if the anime has been released yet
                    #find the tags for the anime and store it into a list
                    tags = []

                    anime_tag_list = soup2.find(class_='tags')
                    if(anime_tag_list == None):
                        anime_tag_list = soup2.find(class_='tags ')                     #a couple of the tags have spaces after in HTML
                        if(anime_tag_list == None):
                            tags = []
                            tags.append('N/A')
                        else:
                            anime_tag_list_items = anime_tag_list.find_all('a')
                            for anime_tag in anime_tag_list_items:
                                tags.append(anime_tag.contents[0].strip())
                    else:
                        anime_tag_list_items = anime_tag_list.find_all('a')
                        for anime_tag in anime_tag_list_items:
                            tags.append(anime_tag.contents[0].strip())

                    #find the studio that created the anime, N/A in rare cases where it isn't listed
                    anime_studio_list = soup2.find(class_='pure-g entryBar')
                    anime_studio_list_items = anime_studio_list.find_all('a')
                    if(anime_studio_list_items != []):                                  #if there is a studio listed
                        if not anime_studio_list_items[0].contents[0][-4:].isdigit():   #fixes issue where if there is no studio but there is a season and year it will put the season and year as the studio
                            studio = anime_studio_list_items[0].contents[0]
                    else:
                        studio = "N/A"

                    #find the recommended anime and store it into a list
                    #similar = []

                    #anime_similar_list = soup2.find(class_='cardDeck cardGrid cardGrid7')
                    #if(anime_similar_list != None):                                     #if there are similar recommended anime
                        #anime_similar_list_items = anime_similar_list.find_all(class_='cardName')
                        #for anime_similar in anime_similar_list_items:
                            #similar.append(anime_similar.contents[0])
                    #else:
                        #similar = []

                    #if(similar == []):
                        #similar.append('N/A')

                    year = anime_year_list_items[tableIndex].contents[0]                #put the anime year into a variable and get rid of the tags, moved away from other setters in case year is invalid

                    #write the data to the csv file
                    if(anime_stats_list[0] != 'N/A' and anime_stats_list[1] != 'N/A' and tags[0] != 'N/A'):  #check to make sure the anime has ratings and tags
                        animeinfo = [names, rank, typeMedia, year, studio, anime_stats_list[0], anime_stats_list[1]]
                        for tag in tags:
                            animeinfo.append(tag)
                        data_writer.writerow(animeinfo)

            tableIndex+=1
            if(count%10 == 0):
                print(count, "anime done",(time.time()-start_time), "seconds elapsed")
            count+=1   

print((time.time()-start_time), "seconds") #output how long the program took to run
