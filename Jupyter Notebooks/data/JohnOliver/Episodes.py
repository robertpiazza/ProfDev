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
 
with open('Episode_info.txt', 'rb') as file:
    text = file.readlines()
#%%
searchText = b'Episode list'    
#print(text)
for j in range(9):
    for i, line in enumerate(text):
        if searchText in line:
            print(text[i+j])

