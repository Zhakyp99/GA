import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import path
import math
from random import randint
from pyevolve import G2DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Initializators, Mutators, Consts
import pyevolve
from pyevolve import G2DBinaryString
from pyevolve import Crossovers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]
    m = np.zeros((100, 1))
    m[:] = 0.4
    z[:,:,-1] = m
#    print(np.zeros((100, 1)).shape, np.linspace(0, alpha, 100)[:,None].shape)

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, extent=[xmin, xmax, ymin, ymax],
                   origin='upper', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im

def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)

def line_coef(x1, y1, x2, y2):
    k = (y1 - y2 + 0.0) / (x1 - x2 + 0.0)
    b = y1 - k * x1
    return k, b

# Jensen Wake Model 

#Initial Params 
D = 50  #diameter 
k = 0.075 # Land Case 
Uinf = 10 # Freestream 
Ct = 0.43 # thrust 
#x = [200, 400, 250]
#y = [140, 200, 550]

x = []
y = []

dis = 50
ind = 0

number_of_turbines = 2

while len(x) != number_of_turbines:
    new_x = randint(0, 2000)
    new_y = randint(0, 2000)
    if ind == 0:
        x.append(new_x)
        y.append(new_y)
    else:
        norm_dis = True
        for j in range(len(x)):
            if math.sqrt((x[j] - new_x)**2 + (y[j] - new_y)**2) < dis:
                norm_dis = False
                break
        if norm_dis:
            x.append(new_x)
            y.append(new_y)
            
    ind = ind + 1
    
print(x)
print(y)

# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes
def eval_func(population):
    
    x = []
    y = []
    k = 0.075 # Land Case 
    
    for i in range(population.getSize()[0]):
        x.append(population.getItem(i, 0))
        y.append(population.getItem(i, 1))
    
#    for i in xrange(chromosome.getHeight()):
#        print(chromosome[i])
#        for j in range(population.getSize()[1]):
#            print(population.getItem(i, j))
#            if population.getItem(i, j) == 1:
#                x.append(200 * i + 50)
#                y.append(200 * j + 50)
                
    print(len(x), len(y))
    
    _s = []
    _Dw = []
    s = []
    Dw = []
    for i in range(len(x)):
        for _x in range(0, 15 * D + 1):
            s.append((_x + 0.0) / (D + 0.0))
            Dw.append(D * (1 + 2 * k * ((_x + 0.0) / (D + 0.0))))
        _s.append(s)
        _Dw.append(Dw)
        
    # Bounding Box 
    # Use polygon to bound area affected by wake to apply slowdown factor 
    # Polygon Bounding 
    # Where xv,yv are the verticies of the polygons denoted as a bounding box 
    # and xq,yq are the scattered data points from the cfd experimentation
    
    _Dwy = []
    _B1y = []
    _B1x = []
    _B2y = []
    _B2x = []
    _B3y = []
    _B3x = []
    _B4y = []
    _B4x = []
    
    for i in range(len(_Dw)):
        _Dwy.append(_Dw[i][len(_Dw[i]) - 1])
    #print(len(_Dwy))
    #Dwy = Dw[len(Dw) - 1]
    
    for i in range(len(y)):
        _B1y.append(y[i] + (D / 2))
    #print(len(_B1y))
    #B1y = y + (D / 2)
    
    for i in range(len(x)):
        _B1x.append(x[i])
    #print(len(_B1x))
    #B1x = x
    
    for i in range(len(y)):
        _B2y.append(y[i] + (_Dwy[i] / 2))
    #print(len(_B2y))
    #B2y = y + (Dwy / 2)
    
    for i in range(len(x)):
        _B2x.append(x[i] + 15 * D)
    #print(len(_B2x)) 
    #B2x = x + 15 * D
    
    for i in range(len(x)):
        _B3x.append(x[i] + 15 * D)
    #print(len(_B3x))  
    #B3x = x + 15 * D
    
    for i in range(len(y)):
        _B3y.append(y[i] - (_Dwy[i] / 2))
    #print(len(_B3y))  
    #B3y = y - (Dwy / 2)
    
    for i in range(len(y)):
        _B4y.append(y[i] - (_Dwy[i] / 2))
    #print(len(_B4y))  
    #B4y = y - (D/2)
    
    for i in range(len(x)):
        _B4x.append(x[i])
    #print(len(_B4x))  
    #B4x = x
    
    _xv = []
    _yv = []
    
    # x verticies of polygon 
    for i in range(len(_B1x)):
        _xv.append(np.array([_B1x[i], _B2x[i], _B3x[i], _B4x[i]]))
        _yv.append(np.array([_B1y[i], _B2y[i], _B3y[i], _B4y[i]]))
    
    #print(len(_xv),len(_yv))
    
    _xq = []
    _yq = []
    _uq = []
    
    for i in range(len(x)):
        xq = []
        yq = []
        uq = []
        
        for i in range(5000):
            xq.append(random.uniform(10, 2000))
            yq.append(random.uniform(10, 2000))
            uq.append(random.uniform(10, 10.1))
        
        _xq.append(np.array(xq))
        _yq.append(np.array(yq))
        _uq.append(uq)
    
    _xq = np.array(_xq)
    _yq = np.array(_yq)
    _xv = np.array(_xv)
    _yv = np.array(_yv)
    
    #  Boolean for data within set parameters to apply function 
    __in = []
    for i in range(len(x)):
        _in = inpolygon(_xq[i], _yq[i], _xv[i], _yv[i])
        __in.append(_in)
    
#        plt.plot(_xv[i], _yv[i], linewidth=2.0)
#        plt.axis('equal')
        
    #    _xq[i][_in] = [int(j) for j in _xq[i][_in]]
    #    _yq[i][_in] = [int(j) for j in _yq[i][_in]]
#        plt.plot(_xq[i][_in], _yq[i][_in], 'ro')
        
    inter_x = []
    inter_y = []
    inter_bool = []
    
    for i in range(len(x)):
        change = False
        for j in range(len(x)):
            if i != j:
                _in_1 = inpolygon(_xq[i], _yq[i], _xv[i], _yv[i])
                _in_2 = inpolygon(_xq[j], _yq[j], _xv[j], _yv[j])
    #            print(len(_xq[i][_in_1]),len(_yq[i][_in_1]), len(_xq[j][_in_2]), len(_yq[j][_in_2]))
                
                k1, b1 = line_coef(_B1x[i], _B1y[i], _B2x[i], _B2y[i])
                k2, b2 = line_coef(_B3x[i], _B3y[i], _B4x[i], _B4y[i])
                
                _x = _xq[j][_in_2]
                _y = _yq[j][_in_2]
                
                for k in range(len(_x)):
                    if _y[k] <= k1 * _x[k] + b1 and _y[k] >= k2 * _x[k] + b2 and _x[k] <= _B2x[i] and _x[k] >= _B1x[i]:
                        inter_x.append(_x[k])
                        inter_y.append(_y[k])
                if len(inter_x) != 0:
                    if _B1x[i] <= _B1x[j]:
                        change = True
        inter_bool.append(change)
                        
#    for i in range(len(inter_x)):
#        plt.plot(inter_x[i], inter_y[i], 'bs')
    #plt.plot(_x[~_in], _y[~_in], 'bs')
    
#    plt.show()
    
    for j in range(len(x)):
        # Application of U velocity dependent on local position of point to rotor
        for i in range(_xq[j].shape[0]):
            s = _xq[j][__in[j]] / D
            if __in[j][i] == 1:
                _uq[j][i] = Uinf * (1.0 - ((1.0 - math.sqrt(1.0 - Ct)) / (1.0 + 2.0 * k * s) ** 2))
                _uq[j][i] = sum(_uq[j][i]) / float(len(_uq[j][i]))
    #            print(j,i,_uq[j][i])
            if __in[j][i] == 0:
                _uq[j][i] = Uinf
                
    inter_ind = []
    for i in range(len(_xq)):
        print(i, len(_xq))
        for j in range(len(_xq[i])):
            for k in range(len(inter_x)):
                if _xq[i][j] == inter_x[k] and _yq[i][j] == inter_y[k]:
                    inter_ind.append(np.array([i, j]))
                    break
                
    for k in range(len(inter_ind)):
        i = inter_ind[k][0]
        j = inter_ind[k][1]
        _uq[i][j] = Uinf * (1.0 - (1.0 - (_uq[i][j] / Uinf)) * math.sqrt(2))
     
    wake_x = []
    wake_y = []
    wake_u = []
    for i in range(len(_B1x)):
        wake_x.append(_B1x[i])
        wake_y.append((_B1y[i] + _B4y[i]) / 2)
        if not inter_bool[i]:
            for j in range(_xq[i].shape[0]):
                if __in[i][j] == 1:
                    wake_u.append(_uq[i][j])
                    break
        else:
            wake_u.append(Uinf)
            
    
    _n = len(wake_x)
#    print("N", _n)
    p = 0
    for i in range(len(wake_x)):
        print(i, wake_x[i],wake_y[i], wake_u[i])
        p = p + wake_u[i]**3
    p = 0.3 * p
#    print("P", p)
    cost = _n * ((2.0 / 3.0) + (1.0 / 3.0) * (2.71828 ** (_n * _n * (-1 * 0.00174))))
#    print("cost", cost)
    obj = cost / p
    print("objective func", obj)
    print("---------------------------------------------------")
    
    _chromosome = []
    for i in range(10):
        temp = []
        for j in range(10):
            temp.append(0)
        _chromosome.append(np.array(temp))
    _chromosome = np.array(_chromosome)
    for i in range(len(wake_x)):
        chromosome = np.concatenate((wake_x[i], wake_y[i]), axis=None)
        chromosome = chromosome.reshape(1,2)
        _x = int(chromosome[0][0] / 200)
        _y = int(chromosome[0][1] / 200)
        _chromosome[_x][_y] = 1
        
    for i in range(len(_chromosome)):
        print(_chromosome[i])
    
    return obj


# Construct Bounding Box for Point Selection use find function to capture
# all points in the required area. 
count = 1
innercount = 1
_s = []
_Dw = []
s = []
Dw = []
for i in range(len(x)):
    for _x in range(0, 15 * D + 1):
        s.append((_x + 0.0) / (D + 0.0))
        Dw.append(D * (1 + 2 * k * ((_x + 0.0) / (D + 0.0))))
    _s.append(s)
    _Dw.append(Dw)
    
# Bounding Box 
# Use polygon to bound area affected by wake to apply slowdown factor 
# Polygon Bounding 
# Where xv,yv are the verticies of the polygons denoted as a bounding box 
# and xq,yq are the scattered data points from the cfd experimentation

_Dwy = []
_B1y = []
_B1x = []
_B2y = []
_B2x = []
_B3y = []
_B3x = []
_B4y = []
_B4x = []

for i in range(len(_Dw)):
    _Dwy.append(_Dw[i][len(_Dw[i]) - 1])
#print(len(_Dwy))
#Dwy = Dw[len(Dw) - 1]

for i in range(len(y)):
    _B1y.append(y[i] + (D / 2))
#print(len(_B1y))
#B1y = y + (D / 2)

for i in range(len(x)):
    _B1x.append(x[i])
#print(len(_B1x))
#B1x = x

for i in range(len(y)):
    _B2y.append(y[i] + (_Dwy[i] / 2))
#print(len(_B2y))
#B2y = y + (Dwy / 2)

for i in range(len(x)):
    _B2x.append(x[i] + 15 * D)
#print(len(_B2x)) 
#B2x = x + 15 * D

for i in range(len(x)):
    _B3x.append(x[i] + 15 * D)
#print(len(_B3x))  
#B3x = x + 15 * D

for i in range(len(y)):
    _B3y.append(y[i] - (_Dwy[i] / 2))
#print(len(_B3y))  
#B3y = y - (Dwy / 2)

for i in range(len(y)):
    _B4y.append(y[i] - (_Dwy[i] / 2))
#print(len(_B4y))  
#B4y = y - (D/2)

for i in range(len(x)):
    _B4x.append(x[i])
#print(len(_B4x))  
#B4x = x

_xv = []
_yv = []

# x verticies of polygon 
for i in range(len(_B1x)):
    _xv.append(np.array([_B1x[i], _B2x[i], _B3x[i], _B4x[i]]))
    _yv.append(np.array([_B1y[i], _B2y[i], _B3y[i], _B4y[i]]))

#print(len(_xv),len(_yv))

_xq = []
_yq = []
_uq = []

for i in range(len(x)):
    xq = []
    yq = []
    uq = []
    
    for i in range(5000):
        xq.append(random.uniform(10, 2500))
        yq.append(random.uniform(10, 2500))
        uq.append(random.uniform(10, 10.1))
    
    _xq.append(np.array(xq))
    _yq.append(np.array(yq))
    _uq.append(uq)

_xq = np.array(_xq)
_yq = np.array(_yq)
_xv = np.array(_xv)
_yv = np.array(_yv)

#  Boolean for data within set parameters to apply function 
__in = []
for i in range(len(x)):
    _in = inpolygon(_xq[i], _yq[i], _xv[i], _yv[i])
    __in.append(_in)

    plt.plot(_xv[i], _yv[i], linewidth=2.0)
    print("x", _xv[i])
    print("y", _yv[i])
    plt.axis('equal')
    temp_x = _xv[i]
    temp_y = _yv[i]
    gradient_fill(temp_x, temp_y)
    
#    _xq[i][_in] = [int(j) for j in _xq[i][_in]]
#    _yq[i][_in] = [int(j) for j in _yq[i][_in]]
    plt.plot(_xq[i][_in], _yq[i][_in], 'ro')
    
inter_x = []
inter_y = []
inter_bool = []

for i in range(len(x)):
    change = False
    for j in range(len(x)):
        if i != j:
            _in_1 = inpolygon(_xq[i], _yq[i], _xv[i], _yv[i])
            _in_2 = inpolygon(_xq[j], _yq[j], _xv[j], _yv[j])
#            print(len(_xq[i][_in_1]),len(_yq[i][_in_1]), len(_xq[j][_in_2]), len(_yq[j][_in_2]))
            
            k1, b1 = line_coef(_B1x[i], _B1y[i], _B2x[i], _B2y[i])
            k2, b2 = line_coef(_B3x[i], _B3y[i], _B4x[i], _B4y[i])
            
            _x = _xq[j][_in_2]
            _y = _yq[j][_in_2]
            
            for k in range(len(_x)):
                if _y[k] <= k1 * _x[k] + b1 and _y[k] >= k2 * _x[k] + b2 and _x[k] <= _B2x[i] and _x[k] >= _B1x[i]:
                    inter_x.append(_x[k])
                    inter_y.append(_y[k])
            if len(inter_x) != 0:
                if _B1x[i] <= _B1x[j]:
                    change = True
    inter_bool.append(change)
                    
for i in range(len(inter_x)):
    plt.plot(inter_x[i], inter_y[i], 'bs')
    gradient_fill(inter_x[i], inter_y[i])
#plt.plot(_x[~_in], _y[~_in], 'bs')

plt.show()





          
inter_ind = []
for i in range(len(_xq)):
    for j in range(len(_xq[i])):
        for k in range(len(inter_x)):
            if _xq[i][j] == inter_x[k] and _yq[i][j] == inter_y[k]:
                inter_ind.append(np.array([i, j]))
                break
            
for k in range(len(inter_ind)):
    i = inter_ind[k][0]
    j = inter_ind[k][1]
    _uq[i][j] = Uinf * (1.0 - (1.0 - (_uq[i][j] / Uinf)) * math.sqrt(2))
 
   
wake_x = []
wake_y = []
wake_u = []
for i in range(len(_B1x)):
    wake_x.append(_B1x[i])
    wake_y.append((_B1y[i] + _B4y[i]) / 2)
    if not inter_bool[i]:
        for j in range(_xq[i].shape[0]):
            if __in[i][j] == 1:
                wake_u.append(_uq[i][j])
                break
    else:
        wake_u.append(Uinf)







chromosomes = []
_chromosome = []
for i in range(10):
    temp = []
    for j in range(10):
        temp.append(0)
    _chromosome.append(np.array(temp))
_chromosome = np.array(_chromosome)
for i in range(len(wake_x)):
    chromosome = np.concatenate((wake_x[i], wake_y[i]), axis=None)
    chromosome = chromosome.reshape(1,2)
    _x = int(chromosome[0][0] / 200)
    _y = int(chromosome[0][1] / 200)
    chromosome[0][0] = int(chromosome[0][0] / 200) * 200 + 50
    chromosome[0][1] = int(chromosome[0][1] / 200) * 200 + 50
    _chromosome[_x][_y] = 1
    print(chromosome, _x, _y)
    chromosomes.append(chromosome)

for i in range(len(_chromosome)):
    print(_chromosome[i])




population = chromosomes[0]
for i in range(1, len(wake_x)):
    population = np.concatenate((population, chromosomes[i]), axis=0)

print(population)
print(population.shape, _chromosome.shape)

# Enable the pyevolve logging system
pyevolve.logEnable()

# Genome instance, 1D List of 2 elements
#genome = G2DBinaryString.G2DBinaryString(10, 10)
genome = G2DList.G2DList(len(wake_x), 2)

#for i in range(_chromosome.shape[0]):
#    for j in range(_chromosome.shape[1]):
#        genome.setItem(i, j, 0)
#
#for i in range(_chromosome.shape[0]):
#    for j in range(_chromosome.shape[1]):
#        if _chromosome[i][j] == 1:
#            genome.setItem(i, j, 1)



#for i in range(_chromosome.shape[0]):
#    for j in range(_chromosome.shape[1]):
#        genome.setItem(i, j, population[i][j])

       
for i in range(population.shape[0]):
    temp = []
    for j in range(population.shape[1]):
        temp.append(genome.getItem(i, j))
    print(temp)

# The evaluator function (evaluation function)
genome.evaluator.set(eval_func)
#genome.crossover.set(Crossovers.G2DBinaryStringXSingleHPoint)
#genome.mutator.set(Mutators.G2DBinaryStringMutatorSwap)
# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)

# Set the Roulette Wheel selector method, the number of generations and
# the termination criteria
ga.selector.set(Selectors.GRouletteWheel)
ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
ga.setMinimax(Consts.minimaxType["minimize"])
ga.setGenerations(1)
ga.setCrossoverRate(0.8)
ga.setMutationRate(0.06)

# Do the evolution, with stats dump
# frequency of 20 generations
ga.evolve(freq_stats=1)

# Best individual
print ga.bestIndividual()
