# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 01:00:20 2021

@author: Nikita
"""

import matplotlib.pyplot as plt
import numpy as np
import math


def generate_multibrot(per_axis : int, power : int, start : complex = None):
    x_coords = []
    y_coords = []
    z_coords = []
    if (power < 0):
        print("Cannot generate correct plots for negative powers using this "
              "function.")
        return
    
    # Multibrot typically within these bounds
    limits = [-2.0,2.0]
    x = np.arange(limits[0],limits[1],(limits[1]- limits[0])/per_axis)
    y = np.arange(limits[0],limits[1],(limits[1]- limits[0])/per_axis)
    
    # Check if coordinates belong to Multibrot set -> iter ~ 100 : No, iter ~ 0 Yes
    for coord_x in x:
        for coord_y in y:
            candidate = coord_x + coord_y*1j
            recurrence = 0 + 0j
            iterations = 100
            if (start != None):
                recurrence = candidate
                candidate = start
            magnitude = abs(recurrence)
                
            while (iterations > 0) and (magnitude < 4):
                last_rec = recurrence
                recurrence = pow(recurrence,power) + candidate
                magnitude = abs(recurrence)
                iterations -= 1
                if abs(recurrence - last_rec) < abs(last_rec*0.01):          
                    iterations = 0
            
            x_coords.append(coord_x)
            y_coords.append(coord_y)
            z_coords.append(iterations)
            
    
    plt.figure(figsize=(10,10))
    plt.scatter(x_coords ,y_coords, c = z_coords, cmap = "PiYG")
    plt.box(False)
    plt.axis('off')
    if (start != None):
        plt.title("Multibrot set for z_(n+1) = z_n^{} + {}".format(power,start))
    else:
        plt.title("Multibrot set for z_(n+1) = z_n^{} + c".format(power))
        
def generate_negatve_multibrot(per_axis : int, power : int,iters : int,
                               start : complex = None, debug : bool = False):
    x_coords = []
    y_coords = []
    z_coords = []
    if (power > 0):
        print("Cannot generate correct plots for positive powers using this "
              "function.")
        return
    
    # Negative Multibrot typically within these bounds
    limits = [-2.0,2.0]
    x = np.arange(limits[0],limits[1],(limits[1]- limits[0])/per_axis)
    y = np.arange(limits[0],limits[1],(limits[1]- limits[0])/per_axis)
    arr_iters  = []
    # Check if coordinates belong to negative Multibrot set/ 
    for coord_x in x:
        for coord_y in y:
            seen = {}
            candidate = coord_x + coord_y*1j
            recurrence = candidate
            iterations = iters
            if (start != None):
                candidate = start
                
            while (iterations > 0) and not (recurrence in seen):
                seen[recurrence] = 1
                recurrence = pow(recurrence,power) + candidate
                # If either real or complex becomes NaN, both end up becoming 
                # Nan
                if math.isnan(recurrence):
                    break
                iterations -= 1
                
            
            if (debug):
                arr_iters.append(iters - iterations)
            
            x_coords.append(coord_x)
            y_coords.append(coord_y)
            z_coords.append(iterations)
    
    plt.figure(figsize=(10,10))
    plt.scatter(x_coords ,y_coords, c = z_coords, cmap = "PiYG")
    plt.box(False)
    plt.axis('off')
    
    if (start != None):
        plt.title("Negative Multibrot set for z_(n+1) = z_n^{} + {}".format(power,start))
    else:
        plt.title("Negative Multibrot set for z_(n+1) = z_n^{} + c".format(power))
    
    if debug:
        return arr_iters
    else:
        return None
    
def plot_iterations_distribution(i : int, size : int):
    def gaussian(x : float, dev : float, mean : float):
        constant = 1/(dev*(2*np.pi)**(0.5))
        exponent = -0.5*((x-mean)/dev)**2
        return constant*np.exp(exponent)
    
    def trick (x : float):
        nonlocal AAA_average, AAA_std
        return gaussian(x,AAA_std,AAA_average)

    iters = generate_negatve_multibrot(size,-2,i,0j, True)
    AAA_smallest = min(iters)
    AAA_biggest = max(iters)
    AAA_average = sum(iters)/len(iters)
    AAA_std_arr = [(elem - AAA_average)**2 for elem in iters]
    AAA_std = (sum(AAA_std_arr)/len(AAA_std_arr))**(0.5)
    print ("Minimum at {} | Maximum at {} | Average : {} | Dev : {}".format(
        AAA_smallest,AAA_biggest, AAA_average, AAA_std))
    

    plt.figure()
    x_axis = np.arange(-5000,5000, 0.01) 
    y_axis = list(map(trick,x_axis))
    plt.plot(x_axis,y_axis)


generate_negatve_multibrot(1000,-2,500,-1, False)
#
#polar_theta = []
# polar_r =[]
# for x,y in zip(x_coords,y_coords):
#     polar_theta.append(np.arctan2(y,x))
#     polar_r.append(np.sqrt(x**2 + y**2))
#ax = fig.add_subplot(111, polar=True)
#plt.axis('off')
#ax.plot(projection='polar')
#plt.savefig('juliafractal_{}_{}x{}_{}iter_.png'.format(z0,perAxis,perAxis,100), dpi=1000)
#plt.show()