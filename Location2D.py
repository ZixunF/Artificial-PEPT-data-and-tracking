# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:19:42 2020

@author: Zixun
"""
import sympy
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def get_distance_square(LoR):
    
    x, y = sympy.symbols("x y")
    x1 = LoR[0]
    y1 = LoR[1]
    x2 = LoR[2]
    y2 = LoR[3]
    x12 = x1 - x2
    y12 = y1 - y2
    r_square = x12 * x12 + y12 * y12  #length of LoR (sqaure)
    delta_square = (y * x12 - (x - x2) * y12 - y2 * x12)**2 / r_square
    #perpendicular distance from (x,y) to LoR (sqaure)
    
    return delta_square
                
def get_extremum(data):
    
    x1_coordinate = data[:,0]
    x2_coordinate = data[:,2]
    y1_coordinate = data[:,1]
    y2_coordinate = data[:,3]
    x_coordinate = np.hstack((x1_coordinate, x2_coordinate))
    y_coordinate = np.hstack((y1_coordinate, y2_coordinate))
    x_min = np.min(x_coordinate)
    x_max = np.max(x_coordinate)
    y_min = np.min(y_coordinate)
    y_max = np.max(y_coordinate)

    return [x_min, x_max, y_min, y_max]
    #extremum of the coordinates of start & end points
 

def MDP(Ds):
    
    #returning minimum distant point
    xy = (x, y)
    func = sympy.lambdify(xy, Ds)

    # Build Jacobian:
    jac_f = [Ds.diff(x) for x in xy]
    jac_fn = [sympy.lambdify(xy, jf) for jf in jac_f]
    
    bnds = ((extremum_coordinate[0],extremum_coordinate[1]),
            (extremum_coordinate[2],extremum_coordinate[3]))  # bounds
    
    def fun_v(zz):
        # Helper for receiving vector parameters 
        return func(zz[0], zz[1])


    def jac_v(zz):
        # Jacobian Helper for receiving vector parameters 
        return np.array([jfn(zz[0], zz[1]) for jfn in jac_fn])

    start_point = np.array([extremum_coordinate[0],extremum_coordinate[2]])
    res = opt.minimize(fun_v, start_point, method="SLSQP",jac=jac_v, bounds=bnds)
    
    return res
    
# Main function
if __name__ == "__main__":
    # Points
    x, y = sympy.symbols("x y")
    n_LoR = 50 # number of lines of response
    f = 0.9 # discard fraction
    
    #load LoR data from file
    test_data = np.loadtxt(open("test_data.csv", "rb"), delimiter = ",", skiprows = 0)  
    x0 = test_data[0][0]
    y0 = test_data[0][1]
    test_data = np.delete(test_data, 0 , axis = 0)
    
    extremum_coordinate = get_extremum(test_data)    #data format:[x_min, x_max, y_min, y_max]
    
    sum_dist = 0
    
    MDP_cdnt = []
    #for i in range(n_LoR):
    n = n_LoR   # number of LoRs remaining 
    distance_ratio = 1.2
    while n >= n_LoR * f:
        d_data = []    # store perpendicular distance from MDP to each LoR
        row_del = [] # save the number of rows that need to be deleted

        print("MDP coordinate: ", MDP_cdnt)   
        for j in range(n):
            LoR = np.zeros(4)
            for k in range(4):
                LoR[k] = test_data[j][k]    
            
            p_dist = get_distance_square(LoR)    # Calculate squared perpendicular distance from (x,y) to each LoRs 
            d_data.append(sympy.sqrt(p_dist))
            sum_dist += p_dist   # the sum of the square of perpendicular distance to each LoRs      
        
        RMS_dist = sympy.sqrt(sum_dist / n)    # RMS distance from this point to the set of LORs
        MDP_cdnt = MDP(RMS_dist).x
        n_dist = n 
        for m in range(n_dist):
            distance = d_data[m].evalf(subs = {x:MDP_cdnt[0], y:MDP_cdnt[1]})
            RMS_distance = RMS_dist.evalf(subs = {x:MDP_cdnt[0], y:MDP_cdnt[1]})
            if distance > RMS_distance * distance_ratio:
                n -= 1
                row_del.append(m) 
        print(n_dist - n, "LoRs discarded")
        if n < n_LoR * f:
            n = n_dist
            break
        test_data = np.delete(test_data, row_del, axis = 0) #delete the m'th row of the frame data  
        if len(row_del) == 0:
            distance_ratio -= 0.1  

    # plot clean LOR data after iterative discarding
    mill_radius = 0.225
    clean_LoR_x = np.vstack((test_data[:,0], test_data[:,2]))
    clean_LoR_y = np.vstack((test_data[:,1], test_data[:,3]))
    plt.figure(figsize=(10,10),dpi=125)
    _t = np.arange(0, 7, 0.1)
    _x = np.cos(_t) * mill_radius
    _y = np.sin(_t) * mill_radius
    plt.plot(_x,_y,'g-')
    plt.plot(clean_LoR_x, clean_LoR_y, "b")
    plt.xlim(-0.3,0.3)
    plt.ylim(-0.3,0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Artificial LoR data after cleaning')
    plt.grid(True)
    plt.show()
    
    # calculate accuracy

    print("number of LoRs remaining:", n)
    print("real tracer location:", x0, y0)
    accuracy = np.sqrt((MDP_cdnt[0] - x0)**2 + (MDP_cdnt[1]-y0)**2)
    print("Accuracy (distance from real tracer to MDP) =", accuracy)    
    print("CPU time:", time.process_time() )

