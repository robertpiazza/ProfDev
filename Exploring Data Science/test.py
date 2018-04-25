# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:16:24 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

pill_assignment = ['a', 'b']
pill_assignment.append('c')
print(pill_assignment)

qualifying =['d', 'hd', 'ht']

def makequalprev(input):
    for condition in qualifying:        
        if condition in input.lower():
            return 1
        else condition not in input.lower():
            return 0
print(makequalprev('hd'))

def makequal(input):
    print(qualifying)
    if not type(input) == str:
        return 0
    elif input.lower() in qualifying:
        return 1
    else:
        return 0

print(makequal(['d', 'hp', 'ht']))