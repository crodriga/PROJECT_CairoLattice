import sys
sys.path.insert(0, '../icenumerics')
import numpy as np
import icenumerics as ice
ureg = ice.ureg
import pandas as pd
import scipy.spatial as spa


from matplotlib import patches

def extracting_trap_positions(col):
    trap_pos = []

    for c in col:

        trap_pos.append(c.center.magnitude)

    trap_pos = np.array(trap_pos)
    
    return trap_pos

def infer_edges(vertices_pos,trap_pos):
    
    tree = spa.KDTree(vertices_pos)
    dist, edges = tree.query(trap_pos, k = 2)
    data_edges = {'start': edges[:,0], 'stop': edges[:,1]}
    df = pd.DataFrame(data=data_edges)
    return df

def lattice_parameters(theta, d, trap_sep_l, s_l4):
    
    ''' This function generates all the parameters to build the lattices from cairo to shakti. '''

    
    alpha = (np.pi/2-theta)/2
    s_l3 = d*((np.sin(np.pi/2-theta-alpha))/(np.sin(np.pi/2+theta)))
    s_a = d*((np.sin(alpha))/(np.sin(np.pi/2+theta)))
    l = s_l4 + trap_sep_l + s_l3
    a = 2*l*(np.cos(theta)-np.sin(theta))
    trap_sep_a = (a - 2*s_a)
    plaquette_cte = 2*l*np.cos(theta)
    
    return trap_sep_a, s_l3, a, l, plaquette_cte 
    
def Unit_cell_from_Cairo_to_Shakti(theta, trap_sep_l, s_l3, s_l4, a, plaquette_cte):
    
    ureg = ice.ureg

    centers = [[0,0,0], # First plaquette 
                [-(s_l3+trap_sep_l/2)*np.cos(theta),a/2+(s_l3+trap_sep_l/2)*np.sin(theta),0],
                [+(s_l3+trap_sep_l/2)*np.cos(theta),a/2+(s_l3+trap_sep_l/2)*np.sin(theta),0],
                [-(s_l3+trap_sep_l/2)*np.cos(theta),-a/2-(s_l3+trap_sep_l/2)*np.sin(theta),0],
                [+(s_l3+trap_sep_l/2)*np.cos(theta),-a/2-(s_l3+trap_sep_l/2)*np.sin(theta),0],
                
                [plaquette_cte,0,0],# Second plaquette
                [plaquette_cte/2+(s_l4+trap_sep_l/2)*np.sin(theta),(s_l3+trap_sep_l/2)*np.cos(theta),0],
                [plaquette_cte+a/2+(s_l3+trap_sep_l/2)*np.sin(theta),(s_l3+trap_sep_l/2)*np.cos(theta),0],
                [plaquette_cte/2+(s_l4+trap_sep_l/2)*np.sin(theta),-(s_l3+trap_sep_l/2)*np.cos(theta),0],
                [plaquette_cte+a/2+(s_l3+trap_sep_l/2)*np.sin(theta),-(s_l3+trap_sep_l/2)*np.cos(theta),0],
                
                [0,-plaquette_cte,0],# Third plaquette
                [-a/2-(s_l3+trap_sep_l/2)*np.sin(theta),-plaquette_cte/2-(s_l4+trap_sep_l/2)*np.cos(theta),0],
                [+a/2+(s_l3+trap_sep_l/2)*np.sin(theta),-plaquette_cte/2-(s_l4+trap_sep_l/2)*np.cos(theta),0],
                [-a/2-(s_l3+trap_sep_l/2)*np.sin(theta),-plaquette_cte-(s_l3+trap_sep_l/2)*np.cos(theta),0],
                [+a/2+(s_l3+trap_sep_l/2)*np.sin(theta),-plaquette_cte-(s_l3+trap_sep_l/2)*np.cos(theta),0],
                
                [plaquette_cte,-plaquette_cte,0],# Fourth plaquette
                [plaquette_cte/2+(s_l4+trap_sep_l/2)*np.cos(theta),-plaquette_cte/2-(s_l4+trap_sep_l/2)*np.sin(theta),0],
                [plaquette_cte+(s_l3+trap_sep_l/2)*np.cos(theta),-plaquette_cte/2-(s_l4+trap_sep_l/2)*np.sin(theta),0],
                [plaquette_cte/2+(s_l4+trap_sep_l/2)*np.cos(theta),-plaquette_cte-a/2-(s_l3+trap_sep_l/2)*np.sin(theta),0],
                [plaquette_cte+(s_l3+trap_sep_l/2)*np.cos(theta),-plaquette_cte-a/2-(s_l3+trap_sep_l/2)*np.sin(theta),0]
              ]*ureg.um
   
    directions = [[0,2*trap_sep_l,0],# First plaquette
                  [-trap_sep_l*np.cos(theta),trap_sep_l*np.sin(theta),0],
                  [trap_sep_l*np.cos(theta),trap_sep_l*np.sin(theta),0],
                  [trap_sep_l*np.cos(theta),trap_sep_l*np.sin(theta),0],
                  [trap_sep_l*np.cos(theta),-trap_sep_l*np.sin(theta),0],
                  
                  [2*trap_sep_l,0,0],# Second plaquette 
                  [-trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  [trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  [trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  [-trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  
                  [2*trap_sep_l,0,0], #Third plaquette
                  [-trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  [trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  [trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  [-trap_sep_l*np.sin(theta),trap_sep_l*np.cos(theta),0],
                  
                  [0,2*trap_sep_l,0], # Fourth plaquette
                  [-trap_sep_l*np.cos(theta),trap_sep_l*np.sin(theta),0],
                  [trap_sep_l*np.cos(theta),trap_sep_l*np.sin(theta),0],
                  [trap_sep_l*np.cos(theta),trap_sep_l*np.sin(theta),0],
                  [trap_sep_l*np.cos(theta),-trap_sep_l*np.sin(theta),0]
                 ]*ureg.um
    
    
    return centers, directions

def from_Cairo_to_Shakti_vertices(Sx,Sy, theta, a, l, plaquette_cte):
    
    x = np.arange(0,Sx)
    y = np.arange(0,Sy)
    
    four_coord = [[-plaquette_cte/2,plaquette_cte/2],
                  [plaquette_cte/2,plaquette_cte/2],
                  [-plaquette_cte/2,-plaquette_cte/2],
                  [plaquette_cte/2,-plaquette_cte/2]]*ureg.um
    
    three_coord = [[0,a/2],
                  [0,-a/2],
                  [plaquette_cte/2+l*np.sin(theta),0],
                  [plaquette_cte+a/2,0],
                  [-a/2,-plaquette_cte],
                  [+a/2,-plaquette_cte],
                  [plaquette_cte,-plaquette_cte/2-l*np.sin(theta)],
                  [plaquette_cte,-plaquette_cte-a/2]]*ureg.um
    
    tx = 2*plaquette_cte
    ty = 2*plaquette_cte
    
    # < Four-coord> 
    
    four_coordX = []
    four_coordY = []
    
    for i in np.nditer(x):
        for j in range(len(four_coord)):

            four_coordX.append(four_coord[j,0].magnitude+(i*tx))

    new_four_coord = np.zeros((len(four_coordX),3))
    new_four_coord[:,0] = four_coordX
    
    four_coordY = np.tile(four_coord[:,1],(Sx))
    new_four_coord[:,1] = four_coordY
    
    if Sy != 1:
        
        four_coordX = []
        four_coordY = []

        for i in np.nditer(y):
            for j in range(len(new_four_coord)):

                four_coordY.append(new_four_coord[j,1]+(i*ty))

        new_four_coord_def = np.zeros((len(four_coordY),3))
        new_four_coord_def[:,1] = four_coordY

        four_coordX = np.tile(new_four_coord[:,0],(Sy))
        new_four_coord_def[:,0] = four_coordX
        
    # < Three-coord >
        
    three_coordX = []
    three_coordY = []
    
    for i in np.nditer(x):
        for j in range(len(three_coord)):

            three_coordX.append(three_coord[j,0].magnitude+(i*tx))

    new_three_coord = np.zeros((len(three_coordX),3))
    new_three_coord[:,0] = three_coordX
    
    three_coordY = np.tile(three_coord[:,1],(Sx))
    new_three_coord[:,1] = three_coordY
    
    if Sy != 1:
        
        three_coordX = []
        three_coordY = []

        for i in np.nditer(y):
            for j in range(len(new_three_coord)):

                three_coordY.append(new_three_coord[j,1]+(i*ty))

        new_three_coord_def = np.zeros((len(three_coordY),3))
        new_three_coord_def[:,1] = three_coordY

        three_coordX = np.tile(new_three_coord[:,0],(Sy))
        new_three_coord_def[:,0] = three_coordX
    
   
    return new_four_coord_def, new_three_coord_def

def lattices_spin_ice_geometry(Sx,Sy,theta,d,trap_sep_l,s_l4,border):
    
    """ Whit this function we will generate the spin lattices from the cairo (theta = pi/6) to shakti (theta = 0), depending on the value of theta"""
    
    x = np.arange(0,Sx)
    y = np.arange(0,Sy)
    
    trap_sep_a, s_l3, a,l, plaquette_cte = lattice_parameters(theta, d, trap_sep_l, s_l4)
    
    if border == "periodic":
        
        centers, directions = Unit_cell_from_Cairo_to_Shakti(theta, trap_sep_l, s_l3, s_l4, a, plaquette_cte)
        
    else: 
        raise(ValueError(border+" is not a supported border type. Supported borsers are: short trap and periodic"))
        
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

    def create_lattice(self, geometry, size,theta,d,trap_sep_l,s_l4, plaquette_cte, border = "closed spin", height = None):
        """ 
        Creates a lattice of spins. 
        The geometry can be:
            * "square"
            * "honeycomb"
            * "trinagular"
            * "cairo"
            * "shakti"
            * "from_cairo_to_shakti"
            
        The border can be 
            * 'closed spin':
            * 'closed vertex's
            * 'periodic'
        """
        self.clear()
        
        latticeunits = ureg.um

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
            
        elif geometry == "from_cairo_to_shakti":
            center, direction = lattices_spin_ice_geometry(
                size[0], size[1], theta,d,trap_sep_l,s_l4,
                border = border 
            )
            
        else: 
            raise(ValueError(geometry+" is not a supporteed geometry."))
        
        self.__init__(center*latticeunits,direction*latticeunits)
        self.lattice = plaquette_cte
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