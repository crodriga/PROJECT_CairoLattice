import sys
sys.path.insert(0, '../icenumerics')
import numpy as np
import icenumerics as ice
ureg = ice.ureg
import pandas as pd

from matplotlib import patches


def unit_cell_Cairo(a):
    
    """This function generates a unit cell of a Cairo lattice. The input parameter is the size of the shorter side.
    The output of the function is the (x,y) collection of the unit cells (centers) and the directions of the spins."""
    
    
    a = a
    l = 1.37*a
    torad = np.pi/180
    
    centers = np.array([[0,0,0], 
             [-l*(417/890)*np.cos(60*torad)-a/2,l*(417/890)*np.cos(30*torad),0], 
             [-l*(417/890)*np.cos(30*torad),l*np.cos(30*torad)+l*(473/890)*np.cos(60*torad),0], 
             [+l*(417/890)*np.cos(30*torad),l*np.cos(30*torad)+l*(473/890)*np.cos(60*torad),0], 
             [+l*(417/890)*np.cos(60*torad)+a/2,l*(417/890)*np.cos(30*torad),0],
             [+l*(417/890)*np.cos(60*torad)+a/2,-l*(417/890)*np.cos(30*torad),0],
             [a/2+l*np.cos(60*torad)+l*(473/890)*np.cos(30*torad),a/2+l*(417/890)*np.sin(30*torad),0],
#             [a/2+l*np.cos(60*torad)+l/2*np.cos(30*torad),-a/2-l/2*np.sin(30*torad),0],
#             [a/2+l*np.sin(30*torad)+l*np.cos(30*torad),0,0],
#             [a/2+l*np.cos(60*torad)+l/2*np.cos(30*torad)+2*l/2*np.cos(30*torad),a/2+l/2*np.sin(30*torad),0],
#             [a/2+2*l*np.cos(60*torad)+a+l/2*np.sin(30*torad),(l+l/2)*np.sin(60*torad),0],
#             [a/2+2*l*np.cos(60*torad)+a/2,2*l*np.sin(60*torad),0],
             [a/2+(l+l*(473/890))*np.cos(60*torad),(l+l*(473/890))*np.sin(60*torad),0],
#             [a/2+(l+l/2)*np.cos(60*torad),(l+l/2)*np.sin(60*torad)+l*np.sin(60*torad),0],
             [l*(417/890)*np.cos(30*torad),l*np.cos(30*torad)+l*np.sin(30*torad)+a+l*(417/890)*np.sin(30*torad),0],
             [0,l*np.cos(30*torad)+l*np.sin(30*torad)+a/2,0]])*ureg.um
    
    directions =[[1,0,0],[-np.sin(30*torad),np.cos(30*torad),0],
                            [np.sin(60*torad),np.cos(60*torad),0],
                            [-np.sin(60*torad),np.cos(60*torad),0],
                            [np.sin(30*torad),np.cos(30*torad),0],
                            [np.cos(60*torad),-np.sin(60*torad),0],
                            [np.sin(60*torad),-np.cos(60*torad),0],
#                            [np.cos(30*torad),np.sin(30*torad),0],
#                            [0,1,0],
#                            [np.sin(60*torad),np.cos(60*torad),0],
#                            [-np.sin(30*torad),np.cos(30*torad),0],
#                            [1,0,0],
                            [np.sin(30*torad),np.cos(30*torad),0],
#                            [-np.cos(60*torad),np.sin(60*torad),0],
                            [np.cos(30*torad),np.sin(30*torad),0],
                            [0,1,0]]*ureg.um
    
    # < Only with this part of the lattice is difficult to map the entire space. We add a translation of those points in order 
    # to be able to only translate the lattice in x and y to cover the whole space. >
    
    new = np.zeros_like(centers)
    
    centers_toAddX = centers[:,0]+(a+2*l*np.cos(60*np.pi/180))*ureg.um
    centers_toAddY = centers[:,1]+(2*l*-np.sin(60*np.pi/180))*ureg.um
    
    new[:,0] = centers_toAddX
    new[:,1] = centers_toAddY

    new = new*ureg.um

    centers_new = np.concatenate((centers,new))
    directions_new = np.concatenate((directions,directions))
    
    return centers_new, directions_new

def cairo_spin_ice_geometry(Sx,Sy,lattice,border):
    
    """In this function we can build the spin ice cairo lattice choosing the lattice (lenght of the shorter trap), the number of repetitions along the x axis and y axis and the type of border. Until now is only available the "closed spin" one."""
    
    x = np.arange(0,Sx)
    y = np.arange(0,Sy)
    
    if border == "closed spin":
        
        centers, directions = unit_cell_Cairo(lattice)
    
    else: 
        raise(ValueError(border+" is not a supported border type."))
        
    # < From here we will computer the centers of the lattice >
        
    # < tx and ty are the translations in the x and y direccion respectively that are needed to extend the lattice >
    
    tx = 2*(lattice)+4*(1.37*lattice)*np.cos(60*np.pi/180)
    ty = -lattice-2*(1.37*lattice)*(np.cos(30*np.pi/180)+np.cos(60*np.pi/180))
    
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

def get_ice_trj(trj,bounds):
    """ Converts lammps trj to ice trj"""
    # in the trj dataframe, traps and atoms are labeled by different types
    traps = trj[trj.type>=2].copy(deep=True)
    traps = traps.rename(columns = {"mux":"dx","muy":"dy","muz":"dz"})
    atoms = trj[trj.type==1].copy(deep=True)
    atoms = unwrap_trj(atoms.filter(["x","y","z"]),bounds.loc[[0]])
    trj = []

    ## The traps id are ordered (thankfully) in the same order as the particles, but they start consecutively.
    # We keep this order but start at one.
    traps.loc[:,"id"] = traps.index.get_level_values("id").values
    traps.loc[:,"frame"] = traps.index.get_level_values("frame")
    traps.loc[:,"id"] = traps["id"]-min(traps["id"])+1
    traps = traps.set_index(["frame","id"])

    ## create a relative position vector. This goes from the center of the trap to the position of the particle
    colloids = atoms-traps
    colloids = colloids[["x","y","z"]]
    colloids.columns = ["cx","cy","cz"]
    traps = pd.concat([traps,colloids],axis=1)
    colloids = []
    atoms = []

    ## Flip those traps that are not pointing in the  direction of the colloids
    flip = np.sign((traps[["dx","dy","dz"]].values*traps[["cx","cy","cz"]].values).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values*flip[:,np.newaxis]

    ## make the direction vector unitary
    mag = np.sqrt((traps[["dx","dy","dz"]].values**2).sum(axis=1))
    traps[["dx","dy","dz"]] = traps[["dx","dy","dz"]].values/mag[:,np.newaxis]

    #timestep = 10e-3 #sec
    #traps["t"] = traps.index.get_level_values("frame")*timestep

    return traps

def unwrap_trj(trj,bounds):
    """ Unwraps trj around periodic boundaries"""
    trj2 = trj.copy(deep=True)

    def unwrap(p):
        p.iloc[:] = np.unwrap(p,axis=0)
        return p

    for c in trj.columns:
        trj2[c] = (trj2[c] - bounds[c+"_min"].values)/(bounds[c+"_max"].values - bounds[c+"_min"].values)

    trj2 = (trj2*2*np.pi).groupby("id").apply(unwrap)/(2*np.pi)

    for c in trj.columns:
        trj2[c] = trj2[c]*(bounds[c+"_max"].values - bounds[c+"_min"].values) + bounds[c+"_min"].values

    return trj2