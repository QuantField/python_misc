# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 23:58:51 2017

@author: ks_work
"""

import random



class FairRoulette():
    def __init__(self):
        self.pockets    = [ r for r in range(1,37)]
        self.ball       = None
        self.pocketOdds = len(self.pockets) - 1
    
    def spin(self):
        self.ball = random.choice(self.pockets)
    
    def betPocket(self, pocket, amt):
        if str(pocket) == str(self.ball):
            return amt*self.pocketOdds
        else: return -amt
    
    def __str__(self):
        return 'Fair Roulette'