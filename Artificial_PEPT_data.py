# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:40:24 2020

@author: Zixun
"""

# generate 2D artificial LoR data

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# main function
if __name__ == "__main__":
    
    # set dimensions of the mill
    mill_radius = 0.225
    
    # get random coordinate of the tracer within the field of view of the mill
    theta_tracer = np.random.random_sample() * 2 * np.pi - np.pi
    r_tracer = np.random.random() * mill_radius
    x0 = r_tracer * np.cos(theta_tracer)
    y0 = r_tracer * np.sin(theta_tracer)
    
    # create random points inside the circle around the tracer
    sample_num = 400
    t = np.random.random(size = sample_num) * 2 * np.pi - np.pi
    x = np.zeros(sample_num)
    y = np.zeros(sample_num)
    xi = np.cos(t)
    yi = np.sin(t)
    scatter_r = 0.01
    for i in range(sample_num):
        length = scatter_r * np.random.random()
        x[i] = xi[i] * length + x0
        y[i] = yi[i] * length + y0
        while x[i]**2 + y[i]**2 > mill_radius**2:  # adjust points outside the mill
            length = scatter_r * np.random.random()
            x[i] = xi[i] * length + x0
            y[i] = yi[i] * length + y0      
           
    # distribute LoR to scatter points (both true and spurious)
    spu_perc = 0.4 # percentage of spurious LoRs
    true_num = int(sample_num * (1 - spu_perc))
    spu_num = sample_num - true_num
    true_angle = np.random.random(size = true_num) * 2 * np.pi - np.pi
    true_slope = np.tan(true_angle)
    mill_geom = sp.geometry.Circle(sp.Point(0,0), mill_radius)
    # initailize coordinates of the true events' intersections
    true_intersection = np.zeros((true_num,4)) 
    # calculating intersections of true events by using sympy.intersection
    for i in range(true_num):
        LoR = sp.Line(sp.Point(x[i],y[i]), slope = true_slope[i])
        start_x = float(sp.intersection(mill_geom, LoR)[0].coordinates[0])
        start_y = float(sp.intersection(mill_geom, LoR)[0].coordinates[1])
        end_x = float(sp.intersection(mill_geom, LoR)[1].coordinates[0])
        end_y = float(sp.intersection(mill_geom, LoR)[1].coordinates[1])
        true_intersection[i] = [start_x, start_y, end_x, end_y]
        
    #initialize coordinates of the spurious events'intersections
    spu_intersection = np.zeros((spu_num,4))
    for i in range(spu_num):
        j = i + true_num
        # get first random radian for one of the rays of spurious events
        radian1 = np.random.random() * 2 * np.pi - np.pi  
        # get second radian to set the angle between two rays 170 to 190 degree
        radian2 = radian1 + (np.pi * 30/180) + np.random.random() * 300 / 180
        ray1 = sp.Ray(sp.Point(x[j],y[j]), angle = radian1)
        ray2 = sp.Ray(sp.Point(x[j],y[j]), angle = radian2)
        #print(type(sp.intersection(mill_geom, ray1)),sp.intersection(mill_geom, ray1))
        endpoint1_x = float(sp.intersection(mill_geom, ray1)[0].coordinates[0])            
        endpoint1_y = float(sp.intersection(mill_geom, ray1)[0].coordinates[1])
        endpoint2_x = float(sp.intersection(mill_geom, ray2)[0].coordinates[0])
        endpoint2_y = float(sp.intersection(mill_geom, ray2)[0].coordinates[1])
        spu_intersection[i] = [endpoint1_x, endpoint1_y, endpoint2_x, endpoint2_y]
    
    # integrate true and spurious data into one numpy array
    LoR_data = np.vstack((true_intersection, spu_intersection))
    LoR_x = np.vstack((LoR_data[:,0], LoR_data[:,2]))
    LoR_y = np.vstack((LoR_data[:,1], LoR_data[:,3]))

    #save PEPT data as csv file
    tracer_data = np.array([x0, y0, 0, 0])
    LoR_n_tracer_data = np.insert(LoR_data, 0, tracer_data, axis = 0 )
    np.savetxt('test_data.csv', LoR_n_tracer_data, delimiter = ",")  
    
    # visualization
    plt.figure(figsize=(10,10),dpi=125)
    plt.plot(x,y,'ro')
    _t = np.arange(0, 7, 0.1)
    _x = np.cos(_t) * mill_radius
    _y = np.sin(_t) * mill_radius
    plt.plot(LoR_x, LoR_y, "b")
    plt.plot(_x,_y,'g-')
    plt.xlim(-0.3,0.3)
    plt.ylim(-0.3,0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Random Scatter Points Around the Tracer')
    plt.grid(True)
    plt.show()    
