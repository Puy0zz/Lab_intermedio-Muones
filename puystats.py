# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:38:58 2024

@author: Juan Puyo
"""

import numpy as np

class linregress_w:
    def __init__(self, x, y, dy):
        self.x= x
        self.y = y
        self.dy = dy
        self.w = 1/dy**2
        self.D= np.sum(self.w*np.sum(self.w*(self.x**2))) - (np.sum(self.w*self.x))**2
    def slope(self):
        return (np.sum(self.w*np.sum(self.w*self.x*self.y))  -  np.sum(self.w*self.x*np.sum(self.w*self.y)))/self.D
    def err_slope(self):
        return np.sqrt(np.sum(self.w)/self.D)
    def intercept(self):
        return (np.sum(self.w*(self.x)**2*np.sum(self.w*self.y)) - np.sum(self.w*self.x*np.sum(self.w*self.x*self.y)))/self.D
    def err_intercept(self):
        return np.sqrt(np.sum(self.w*self.x**2)/self.D)

        
        

        