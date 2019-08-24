# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:21:04 2019

@author: 593787
"""
import pandas as pd
import numpy as np
import re

#Episode list
# |EpisodeNumber   = 150
# |EpisodeNumber2  = 1
# |RTitle          = <center> Aftermath of [[Brexit]]
# |Viewers         = 1.03<ref>{{cite web|url=http://www.showbuzzdaily.com/articles/showbuzzdailys-top-150-sunday-cable-originals-network-finals-2-17-2019.html|title=Updated: ShowBuzzDaily's Top 150 Sunday Cable Originals & Network Finals: 2.17.2019|accessdate=February 24, 2019|date=February 19, 2019|website=showbuzzdaily.com|last=Metcalf|first=Mitch}}</ref>
# |OriginalAirDate = {{Start date|2019|2|17}}
# |ShortSummary    =  ''Other segments'': [[National Emergency Concerning the Southern Border of the United States]], [[New Zealand]]'s exclusion from world maps<br /> ''Guests'': [[Stephen Fry]] (voice-over)
#''Note'': When broadcast in the United Kingdom, a clip from Parliament is replaced with a clip from “Muscle Motion”, a 1983 exercise video featuring the [[Chippendales]].
# |LineColor       = 006550
 
with open('Episode_info.txt', 'r', encoding='utf-8') as file:
    text = file.readlines()
    
searchText = 'Episode list'    
#print(text)
for i, line in enumerate(text):
    if searchText in line:
        EpisodeNumber = re.sub(r'.*= (\d+).*$', r'\1', text[i+2]).strip()
        EpisodeNumber2 = re.sub(r'.*= (\d+).*$', r'\1', text[i+1]).strip()
        #EpisodeNumber = 
        
        raw_title = re.sub(r'({|}|\[|])', "", re.sub(r'.*<center> (.*)$', r'\1', re.sub(r'\[\[.*?\|(.*?)]]', r'\1', text[i+3])))
        wordList = re.sub(r'\W', ' ', raw_title).split()
        Title = "_".join([x[0].upper()+x[1:] for x in wordList])
        OriginalAirDate = re.sub(r'.*Start date\|(\d+)\|(\d+)\|(\d+).*?$', r'\1_\2_\3', text[i+5]).strip()
        #print(text[i+5])
        #print(Title)
        #print(OriginalAirDate.strip())
        if int(EpisodeNumber) < 25:
            season = '1'
        elif int(EpisodeNumber) < 60:
            season = '2'
        elif int(EpisodeNumber) < 90:
            season = '3'
        elif int(EpisodeNumber) < 120:
            season = '4'
        filename = f'{EpisodeNumber}_S{season}_E{EpisodeNumber2}_{OriginalAirDate}_{Title}'
        filename = filename.replace('_RTitle_Center', '')
        print(filename)
        #print(re.sub(r'.*Start date\|(\d+)\|(\d+)\|(\d+).*', r'\1_\2_\3', text[i+5]))
            
#%%
import os        
files = os.listdir()
for file in files:
    if (file[0:3]!='Epi'):
        if (int(file[0:3])>=115) and (int(file[0:3]) <= 119):
            new_file = file.replace('_S6_', '_S4_')
            os.rename(file, new_file)

#%%
files = os.listdir()
files

episode_list= np.array(files[:-3])

episodes = pd.DataFrame(data = episode_list, columns = ['name'])

e_ex = episodes.name.str.split('_', expand = True)
index = episodes.name.str.split('_', expand = True)[0].astype(int)
Season = episodes.name.str.split('_', expand = True)[1].str[1].astype(int)
Season_episode = episodes.name.str.split('_', expand = True)[2].str[1:].astype(int)
date = pd.to_datetime(e_ex[3].astype(str)+'-'+e_ex[4].astype(str)+'-'+e_ex[5].astype(str))
episodes['episodeNumber'] = index
episodes['Season'] = Season
episodes['Season_episode'] = Season_episode
episodes['date'] = date
episodes['duration_seconds'] = 60
episodes['description'] = 'description'


for i, filename in enumerate(episode_list):
    description =re.sub(r'\d+?_.+?_.+?_\d+?_\d+?_\d+?_(.*).txt$', r'\1', filename).replace('_', ' ')
    episodes.iloc[i, 6] = description

df = episodes.set_index('episodeNumber', drop = True).sort_index()
df.to_csv('episodes.csv')

            




#%%

giant_ass_string = ""
files = os.listdir()
for filename in files:
    if (filename[0:3]!='Epi'):
        with open(filename, 'r', encoding='utf-8') as file:
            giant_ass_string += "".join(file.readlines())



#%%
len(giant_ass_string)
combined = giant_ass_string.lower()
applause = re.findall('\[Applause\]', giant_ass_string)
music = re.findall('\[Music\]', giant_ass_string)
everything = re.findall('(\[.*?\]|\(.*?\))', giant_ass_string)
laughs = re.findall('(\[.*?laugh.*?\]|\(.*?laugh.*?\))', combined)

#%%
music[0:3]
giant_ass_string[0:10].strip()
everything_df = pd.DataFrame(data = everything, columns = ['sound'])
laughs_df = pd.DataFrame(data = laughs, columns = ['laughs'])
#everything_df.sound.value_counts()
laughs_df.laughs.value_counts()

loaded_episodes = pd.read_csv('episodes.csv')

loaded_episodes
    
#%%
import seaborn as sns
loaded_episodes['words'] = 0
loaded_episodes['laughs'] = 0
loaded_episodes['applause']= 0 
loaded_episodes['word_per_sec']=0

for episode_name in episode_list:
    
    with open(episode_name, 'r', encoding='utf-8') as file:
        file_contents = "".join(file.readlines())
    cleaned_file = file_contents.replace('\n', ' ').replace('\ufeff', '').replace('-', ' ').lower()
    laughs = len(re.findall('(\[.*?laugh.*?\]|\(.*?laugh.*?\))', cleaned_file))
    words = len(cleaned_file.split())
    applause = len(re.findall('(\[.*?applau.*?\]|\(.*?applau.*?\))', cleaned_file))
    #print(laughs, words, applause)
    loaded_episodes.loc[loaded_episodes.name == episode_name ,'laughs'] = laughs
    loaded_episodes.loc[loaded_episodes.name == episode_name ,'words'] = words
    loaded_episodes.loc[loaded_episodes.name == episode_name ,'applause'] = applause
    
loaded_episodes['word_per_sec'] = loaded_episodes['words']/loaded_episodes['duration_seconds']
loaded_episodes.word_per_sec.plot()
plt.show()
#%%
    sns.regplot(x = 'episodeNumber', y = 'word_per_sec', data = loaded_episodes, ci = 99)

#%%
loaded_episodes.to_csv('episodes.csv')