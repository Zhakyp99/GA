from pyevolve import G2DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Crossovers
from pyevolve import Mutators
from pyevolve import DBAdapters
import Jensen as js
import numpy as np
from pyevolve import Consts
import scipy.integrate




#import scipy.integrate as integrate
#import scipy.special as special
#result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)

# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes
def eval_func(chromosome):
   #score = 0.0
   U = 2.5
   binary = np.array(chromosome[:], dtype=np.float32)
   _n = np.sum(binary)
#   print('n_tur =', _n)
   X = binary*arr_x
   Y = binary*arr_y
   
   #x = np.array([item for item in X.flatten()])
   #y = np.array([item for item in Y.flatten()])
   
   x = np.array(X.flatten())
   y = np.array(Y.flatten())
   z = np.ones(x.shape)*h_hub
   
   P_total = js.Jensen_Wake_Model(x,y,z, U)
   #cost = _n * ((2.0 / 3.0) + (1.0 / 3.0) * (2.71828 ** (_n * _n * (-1 * 0.00174))))
#   print(cost)
#   print(P_total)
#   #print('P_total =', P_total)
#   
   #obj = cost/P_total
   #print(obj)
   AEP = 20*P_total*121*0.001*0.340036422
   I = AEP*0.6*0.9901
   CPC = 1640*6.2*_n
   OM = 13.5*AEP*0.001
   NPV = 20*(I - OM - CPC)

   print('n =',_n)
   print('P =', P_total )
   print('NPV=',NPV)
   return NPV

def run_GA(D, width, length, z_hub):
    
    # note: Wind turbines need to be positioned so that the distances between them are between 3-10 rotor diameters
    # divide into number of rows and columns (3D minimum distance apart)
    delta = 5*D
    nrow = int(np.ceil(length*1000/delta))
    ncol = int(np.ceil(width*1000/delta))
    global arr_x, arr_y, h_hub
    h_hub = z_hub
    '''
    coord_x = []
    coord_y = []
    for i in range(nrow):
        for j in range(ncol):
            coord_x.append(i*delta + delta/20.)
            coord_y.append(j*delta + delta/20.)
    arr_x = np.array(coord_x, dtype=np.float32).reshape(nrow,ncol)
    arr_y = np.array(coord_y, dtype=np.float32).reshape(nrow,ncol) 
    '''
    
    arr_y,arr_x = np.mgrid[0:length*1000:nrow*1j, 0:width*1000:ncol*1j]
    arr_x = np.array(arr_x)
    arr_y = np.array(arr_y)
    
#    print nrow
#    print ncol
#    print arr_y
#    print arr_x
     
    # Genome instance
    genome = G2DBinaryString.G2DBinaryString(nrow, ncol)

    # The evaluator function (objective function)
    genome.evaluator.set(eval_func)
    genome.crossover.set(Crossovers.G2DBinaryStringXSingleHPoint)
    genome.mutator.set(Mutators.G2DBinaryStringMutatorSwap)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setMinimax(Consts.minimaxType["maximize"])
    ga.setElitism(True)
    ga.setGenerations(500)
    ga.setCrossoverRate(0.6)
    ga.setMutationRate(0.01)
    ga.setPopulationSize(60)
    # Do the evolution, with stats dump
    # frequency of 10 generations
    ga.evolve(freq_stats=10)
    
    
    sqlite_adapter = DBAdapters.DBSQLite(identify="plot")
    ga.setDBAdapter(sqlite_adapter)
    # Best individual
    print( ga.bestIndividual() )
    
    return None

def main():
    
    # diameters of wind turbines (m)
    D = 12.9
    #D = 40
    # define geographical space width x length (km x km)
    width = 0.645 
    length = 0.645
    
    # define turbine height
    z_hub = 18  #hub height of each turbine
    
    
    # run GA 
    run_GA(D, width, length, z_hub)
   
    return None
    
    
if __name__ == "__main__":
   main()












































