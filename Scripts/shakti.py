import sys
sys.path.insert(0, '../icenumerics')
import numpy as np
import icenumerics as ice
ureg = ice.ureg
import pandas as pd

from matplotlib import patches


def unit_cell_shakti(a):
    
    """ We choose the plaquette size as the Cairo one. Interparticle distances seems to not be the optimal. (DO NOT USE IT). """
    
    torad = np.pi/180

    # <Parameters from the Cairo lattice>
    a = 19.5458
    l = 1.37*a
    trap_sep = 10

    plaquette_cte = 2*l*np.sin(60*torad)
    
    centers = [[0,0,0], # <First plaquette>
           [-(plaquette_cte-2*trap_sep*2)/4-trap_sep, plaquette_cte/2,0],
           [+(plaquette_cte-2*trap_sep*2)/4+trap_sep, plaquette_cte/2,0],
           [-(plaquette_cte-2*trap_sep*2)/4-trap_sep, -plaquette_cte/2,0],
           [+(plaquette_cte-2*trap_sep*2)/4+trap_sep, -plaquette_cte/2,0],
           [plaquette_cte, 0, 0], # <Second plaquette>
           [plaquette_cte/2,+(plaquette_cte-2*trap_sep*2)/4+trap_sep, 0],
           [plaquette_cte/2,-(plaquette_cte-2*trap_sep*2)/4-trap_sep, 0],
           [(3/2)*plaquette_cte,+(plaquette_cte-2*trap_sep*2)/4+trap_sep, 0],
           [(3/2)*plaquette_cte,-(plaquette_cte-2*trap_sep*2)/4-trap_sep, 0],
           [0,-plaquette_cte, 0], # <Third plaquette>
           [-plaquette_cte/2, -3/4*plaquette_cte,0],
           [-plaquette_cte/2, -5/4*plaquette_cte,0],
           [+plaquette_cte/2, -3/4*plaquette_cte,0],
           [+plaquette_cte/2, -5/4*plaquette_cte,0],
           [plaquette_cte, -plaquette_cte,0], # <Fourth plaquette>
           [3/4*plaquette_cte, -plaquette_cte/2,0],
           [5/4*plaquette_cte, -plaquette_cte/2,0],
           [3/4*plaquette_cte, -3/2*plaquette_cte,0],
           [5/4*plaquette_cte, -3/2*plaquette_cte,0],
          ]*ureg.um

    directions = [[0,3*a,0], # <First plaquette>
                  [1*a,0,0],
                  [1*a,0,0],
                  [1*a,0,0],
                  [1*a,0,0],
                  [3*a,0,0], # <Second plaquette>
                  [0,1*a,0],
                  [0,1*a,0],
                  [0,1*a,0],
                  [0,1*a,0],
                  [3*a,0,0], #  <Third plaquette>
                  [0,1*a,0],
                  [0,1*a,0],
                  [0,1*a,0],
                  [0,1*a,0],
                  [0,3*a,0], # <Fourth plaquette>
                  [1*a,0,0],
                  [1*a,0,0],
                  [1*a,0,0],
                  [1*a,0,0],
                 ]*ureg.um
    
    return centers, directions

def unit_cell_shakti_def(trap_sep, interparticle_dist):
    
    """ Here, I draw again the lattice but this time the interparticle separation is the same than in the cairo lattice, 3 um.
    This way the vertices will have the same interaction at the same magnetic field value as in the cairo lattice. The trap_sep value will be 10 um. """
    
    trap_sep = 10
    part_d = 13*np.sqrt(2)
    plaquette_cte = part_d*2+2*trap_sep

    centers = [
               [0,0,0], # <First plaquette>
               [-plaquette_cte/4, plaquette_cte/2,0],
               [plaquette_cte/4, plaquette_cte/2,0],
               [-plaquette_cte/4, -plaquette_cte/2,0],
               [plaquette_cte/4, -plaquette_cte/2,0],

               [plaquette_cte, 0, 0], # <Second plaquette>
               [plaquette_cte/2,plaquette_cte/4, 0],
               [plaquette_cte/2,-plaquette_cte/4, 0],
               [(3/2)*plaquette_cte,plaquette_cte/4, 0],
               [(3/2)*plaquette_cte,-plaquette_cte/4, 0],

               [0,-plaquette_cte, 0], # <Third plaquette>
               [-plaquette_cte/2, -3/4*plaquette_cte,0],
               [-plaquette_cte/2, -5/4*plaquette_cte,0],
               [+plaquette_cte/2, -3/4*plaquette_cte,0],
               [+plaquette_cte/2, -5/4*plaquette_cte,0],

               [plaquette_cte, -plaquette_cte,0], # <Fourth plaquette>
               [3/4*plaquette_cte, -plaquette_cte/2,0],
               [5/4*plaquette_cte, -plaquette_cte/2,0],
               [3/4*plaquette_cte, -3/2*plaquette_cte,0],
               [5/4*plaquette_cte, -3/2*plaquette_cte,0],
              ]*ureg.um


    directions = [
                  [0,trap_sep,0], # <First plaquette>
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                  [trap_sep,0,0], # <Second plaquette>
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [trap_sep,0,0], #  <Third plaquette>
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,trap_sep,0], # <Fourth plaquette>
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                 ]*ureg.um
    
    return centers, directions

def shakti_spin_ice_geometry(Sx,Sy,lattice,border):
    
    trap_sep = 10
    part_d = 13*np.sqrt(2)

    plaquette_cte = part_d*2+2*trap_sep
    
    """In this function we can build the shakti lattice. Sx and Sy is the n of times that we want to repeat the unit_cell,
    the lattice parameter comes from the Cairo lattice, and border is a str indicating the type of frontier that we want."""
    
    x = np.arange(0,Sx)
    y = np.arange(0,Sy)
    
    if border == "periodic":
        
        centers, directions = unit_cell_shakti_def(trap_sep, part_d)
        
    else: 
        raise(ValueError(border+" is not a supported border type. Supported borsers are: closed spin, fixed conf, GS? and periodic"))
        
    # < From here we will compute the centers of the lattice >
        
    # < tx and ty are the translations in the x and y direccion respectively that are needed to extend the lattice >
    
    tx = 2*plaquette_cte
    ty = 2*plaquette_cte
    
    centersX = []
    centersY = []
    
    for i in np.nditer(x):
        for j in range(len(centers)):

            centersX.append(centers[j,0].magnitude+(i*tx))

    new_centers = np.zeros((len(centersX),3))
    new_centers[:,0] = centersX
    
    centersY = np.tile(centers[:,1],(Sx))
    new_centers[:,1] = centersY
    
    if Sy != 1:
        
        centersX = []
        centersY = []

        for i in np.nditer(y):
            for j in range(len(new_centers)):

                centersY.append(new_centers[j,1]+(i*ty))

        new_centers_def = np.zeros((len(centersY),3))
        new_centers_def[:,1] = centersY

        centersX = np.tile(new_centers[:,0],(Sy))
        new_centers_def[:,0] = centersX

    # < Now I will define the directions of the new lattice >
   
    new_directions = np.tile(directions,(Sy*Sx,1))
    
    return new_centers_def, new_directions


class spin():
    """ 
    A spin is defined by two vectors in R3 space.
    The vector center gives the position of the center of the spin
    The vector direction gives the dipole moment of the spin.
    """
    
    def __init__(self,center,direction):
        
        self.center = np.array(center.magnitude,dtype="float")*center.units
        self.direction = np.array(direction.magnitude,dtype="float")*center.units
        
    def __str__(self):
        return("Spin with Center at [%d %d %d] and Direction [%d %d %d]\n" %\
               (tuple(self.center)+tuple(self.direction)))

    def display(self,ax1):
        
        X=self.center[0].magnitude
        Y=self.center[1].magnitude
        DX=self.direction[0].magnitude*0.3
        DY=self.direction[1].magnitude*0.3
        W = np.sqrt(DX**2+DY**2)
        self.width = W
        ax1.plot([X],[Y],'b')
        #ax1.plot([X-DX,X+DX],[Y-DY,Y+DY],'-+')
        ax1.add_patch(patches.Arrow(X-DX,Y-DY,2*DX,2*DY,width=W,fc='b'))

class spins(list): 
    """ `spins` is a very general class that contains a list of spin objects. The only feature of this list is that it is created from the centers and directions of the spins, and also that it contains a ´display´ method. """ 
    
    def __init__(self, centers = [], directions = None, lattice_constant=1):
        """To initialize, we can give the centers and directions of the spins contained. However, we can also initialize an empty list, and then populate it using the `extend` method """
        self.lattice = lattice_constant
        
        if len(centers)>0:
            self = self.extend([spin(c,d) for (c,d) in zip(centers,directions)])
        
    def display(self,ax = None, ix = False):
        """ This displays the spins in a pyplot axis. The ix parameter allows us to obtain the spins index, which is useful to access specific indices."""
        if not ax:
            ax = plt.gca() 

        for s in self:
            s.display(ax)
        
        center = np.array([s.center.magnitude for s in self])
        direction = np.array([s.direction.magnitude/2 for s in self])
        width = np.array([[s.width/2] for s in self])
        extrema = np.concatenate([center+direction+width,center-direction-width])

        region = np.array([np.min(extrema,axis=0)[:2],np.max(extrema,axis=0)[:2]]).transpose().flatten()

        ax.axis(region)
        
        ax.set_aspect("equal")

    def create_lattice(self, geometry, size, lattice_constant = 1, border = "closed spin", height = None):
        """ 
        Creates a lattice of spins. 
        The geometry can be:
            * "square"
            * "honeycomb"
        The border can be 
            * 'closed spin':
            * 'closed vertex's
            * 'periodic'
        """
        self.clear()
        
        latticeunits = lattice_constant.units

        if geometry == "square":
            center, direction = square_spin_ice_geometry(
                size[0], size[1], lattice_constant.magnitude,
                border = border
            )
        elif geometry == "square3D":
            center, direction = square_spin_ice_geometry3D(
                size[0], size[1], lattice_constant.magnitude,
                height.magnitude, border = border
            )
            #self.height = height            
        elif geometry == "honeycomb":
            center, direction = honeycomb_spin_ice_geometry(
                size[0], size[1], lattice_constant.magnitude,
                border = border
            )
        elif geometry == "triangular":
            center, direction = triangular_spin_ice_geometry(
                size[0], size[1], lattice_constant.magnitude,
                border = border 
            )
        elif geometry == "cairo":
            center, direction = cairo_spin_ice_geometry(
                size[0], size[1], lattice_constant.magnitude,
                border = border 
            )
            
        elif geometry == "shakti":
            center, direction = shakti_spin_ice_geometry(
                size[0], size[1], lattice_constant.magnitude,
                border = border 
            )
            
        else: 
            raise(ValueError(geometry+" is not a supporteed geometry."))
        
        self.__init__(center*latticeunits,direction*latticeunits)
        self.lattice = lattice_constant
        self.size = size
        
    def order_spins(self, ordering):
        """ Modifies de directions of the spins according to a function f(centers,directions,lattice)
        * The function f(centers,directions,lattice) must return an array A of the same length as `directions`, containing logic values where an element `A[i] = True` means the direction of spins[i] is reversed
        """
    
        units = self.lattice.units
    
        centers = np.array([s.center.to(units).magnitude for s in self])*units
        directions = np.array([s.direction.to(units).magnitude for s in self])*units
        
        ordering_array = ordering(centers,directions,self.lattice)
        
        for i,s in enumerate(self):
            if ordering_array[i]:
                s.direction = -s.direction        
                
def random_ordering(centers,directions,lattice):

    order_array = np.array([random.randrange(-1,2,2) for c in centers])
    
    return order_array<0