import sys
sys.path.insert(0, '../icenumerics')
import numpy as np
import icenumerics as ice
ureg = ice.ureg
import pandas as pd

from matplotlib import patches


def unit_cell_shakti(a):
    
    """ We choose the plaquette size as the Cairo one. Interparticle distances seems to not be the optimal. (DO NOT USE IT)
    Is not the last version of the function. """
    
    torad = np.pi/180

    # <Parameters from the Cairo lattice>
    a = 19.5458
    l = 1.37*a
    trap_sep = 10

    plaquette_cte = 2*l*np.sin(60*torad)
    
    centers = np.array([[0,0,0], # <First plaquette>
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
           [5/4*plaquette_cte, -3/2*plaquette_cte,0]
          ])*ureg.um

    directions = np.array([[0,3*a,0], # <First plaquette>
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
                 ])*ureg.um
    
    return centers, directions

def unit_cell_shakti_def(plaquette_cte):
    
    """ Here, I draw again the lattice but this time the interparticle separation is the same than in the cairo lattice, 3 um.
    This way the vertices will have the same interaction at the same magnetic field value as in the cairo lattice. The trap_sep value will be 10 um. """

    centers = np.array([
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
              ])*ureg.um


    directions = np.array([
                  [0,plaquette_cte,0], # <First plaquette>
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                  [plaquette_cte,0,0], # <Second plaquette>
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [plaquette_cte,0,0], #  <Third plaquette>
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,1,0],
                  [0,plaquette_cte,0], # <Fourth plaquette>
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                  [1,0,0],
                 ])*ureg.um
    
    return centers, directions

def shakti_double_trap(trap_sep, part_d):

    """ Here, I follow the same structure as in the shakti lattice. The lattice is the same but
    I divided the long trap in two short ones."""

    trap_sep = 10
    part_d = 13*np.sqrt(2)
    plaquette_cte = part_d*2+2*trap_sep

    centers = np.array([
               [0,plaquette_cte/4,0], # <First plaquette>
               [0,-plaquette_cte/4,0],

               [-plaquette_cte/4, plaquette_cte/2,0],
               [plaquette_cte/4, plaquette_cte/2,0],
               [-plaquette_cte/4, -plaquette_cte/2,0],
               [plaquette_cte/4, -plaquette_cte/2,0],

               [3/4*plaquette_cte, 0, 0], # <Second plaquette>
               [5/4*plaquette_cte, 0, 0],

               [plaquette_cte/2,plaquette_cte/4, 0],
               [plaquette_cte/2,-plaquette_cte/4, 0],
               [(3/2)*plaquette_cte,plaquette_cte/4, 0],
               [(3/2)*plaquette_cte,-plaquette_cte/4, 0],

               [-1/4*plaquette_cte,-plaquette_cte, 0], # <Third plaquette>
               [+1/4*plaquette_cte,-plaquette_cte, 0],

               [-plaquette_cte/2, -3/4*plaquette_cte,0],
               [-plaquette_cte/2, -5/4*plaquette_cte,0],
               [+plaquette_cte/2, -3/4*plaquette_cte,0],
               [+plaquette_cte/2, -5/4*plaquette_cte,0],

               [plaquette_cte, -3/4*plaquette_cte,0], # <Fourth plaquette>
               [plaquette_cte, -5/4*plaquette_cte,0],

               [3/4*plaquette_cte, -plaquette_cte/2,0],
               [5/4*plaquette_cte, -plaquette_cte/2,0],
               [3/4*plaquette_cte, -3/2*plaquette_cte,0],
               [5/4*plaquette_cte, -3/2*plaquette_cte,0],
              ])*ureg.um

    trap_sep = 10
    directions = np.array([
                  [0,trap_sep,0], # <First plaquette>
                  [0,trap_sep,0],

                  [trap_sep,0,0],
                  [trap_sep,0,0],
                  [trap_sep,0,0],
                  [trap_sep,0,0],

                  [trap_sep,0,0], # <Second plaquette>
                  [trap_sep,0,0],

                  [0,trap_sep,0],
                  [0,trap_sep,0],
                  [0,trap_sep,0],
                  [0,trap_sep,0],

                  [trap_sep,0,0], #  <Third plaquette>
                  [trap_sep,0,0],

                  [0,trap_sep,0],
                  [0,trap_sep,0],
                  [0,trap_sep,0],
                  [0,trap_sep,0],

                  [0,trap_sep,0], # <Fourth plaquette>
                  [0,trap_sep,0], 

                  [trap_sep,0,0],
                  [trap_sep,0,0],
                  [trap_sep,0,0],
                  [trap_sep,0,0],
                 ])*ureg.um
    
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
        
        centers, directions = unit_cell_shakti_def(lattice)
        
    elif border == "short trap":
        
        centers,directions = shakti_double_trap(trap_sep, part_d)
        
    else: 
        raise(ValueError(border+" is not a supported border type. Supported borsers are: closed spin, fixed conf, GS? and periodic"))
        
    # < From here we will compute the centers of the lattice >
        
    # < tx and ty are the translations in the x and y direccion respectively that are needed to extend the lattice >
    
    tx = 2*lattice
    ty = 2*lattice
    
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

############### vertices_shakti #################

# Since 3-coordination vertices are not correclty read I will add the following lines in order to change the flow of the program. 

import scipy.spatial as sptl
import scipy.spatial as spa
import tqdm.auto as tqdm

def vertices_positions(plaquette_cte, Sx, Sy):
    """ This function generate a list/dataframe whit the vertices of the lattice. 
    The input is the plaquette_cte and the size of the colloidal ice. """
    
    # Generate vertices of unit cell.
    vertex_centers = np.array([[0, plaquette_cte/2], # 3-fold vertices.
                  [0, -plaquette_cte/2],
                  [plaquette_cte/2,0],
                  [3/2*plaquette_cte,0],
                  [-plaquette_cte/2,-plaquette_cte],
                  [+plaquette_cte/2,-plaquette_cte],
                  [plaquette_cte,-plaquette_cte/2],
                  [plaquette_cte,-3/2*plaquette_cte],
                  [-plaquette_cte/2,plaquette_cte/2], # 4-fold vertices.
                  [plaquette_cte/2, plaquette_cte/2],
                  [-plaquette_cte/2,-plaquette_cte/2],
                  [plaquette_cte/2,-plaquette_cte/2]])
    
    # Generate all vertices by traslating the points in the x, y axis
    
    x = np.arange(0,Sx)
    y = np.arange(0,Sy)
    
    tx = 2*plaquette_cte
    ty = 2*plaquette_cte
    
    centersX = []
    centersY = []
    
    for i in np.nditer(x):
        for j in range(len(vertex_centers)):

            centersX.append(vertex_centers[j,0]+(i*tx))

    new_centers = np.zeros((len(centersX),2))
    new_centers[:,0] = centersX
    
    centersY = np.tile(vertex_centers[:,1],(Sx))
    new_centers[:,1] = centersY
    
    if Sy != 1:
        
        centersX = []
        centersY = []

        for i in np.nditer(y):
            for j in range(len(new_centers)):

                centersY.append(new_centers[j,1]+(i*ty))

        new_centers_def = np.zeros((len(centersY),2))
        new_centers_def[:,1] = centersY

        centersX = np.tile(new_centers[:,0],(Sy))
        new_centers_def[:,0] = centersX
    
    return new_centers_def

def spin_crossing_point_new_shakti(S1,S2):
    
    trap_sep = 10
    part_d = 13*np.sqrt(2)
    plaquette_cte = part_d*2+2*trap_sep
    
    vertex_centers = vertices_positions(plaquette_cte, 2, 2)
    spins = np.array([[S1[0][0],S1[0][1],0],[S2[0][0],S2[0][1],0]])

    tree = spa.KDTree(vertex_centers)
      
    nearest_dist, nearest_ind = tree.query(spins, k=2, distance_upper_bound= 32)

    index = nearest_ind.flatten()

    u, c = np.unique(index, return_counts=True)
    dup = u[c > 1]
    
    if all(vertex_centers[dup].flatten()):
    
        return [-100,-100]+np.zeros(np.shape(S1[0]))
    
    else:
    
        return [vertex_centers[dup].flatten()[0],vertex_centers[dup].flatten()[1]]
    

def update_edge_directions(edges, spins, positions, verb = False):
    """ Map the 'spins' to the edge directions in 'edges'. """
    
    progress = lambda x, **kargs: x
    if verb:
        progress = tqdm.tqdm
        
    for i,e in progress(edges.iterrows(), total = len(edges)):

        spin_direction = spins["Direction"][e.name]

        if (e<0).any():
            # This happens when a single vertex is assigned to an edge 
            vertex = e[e>=0]
            if vertex.index[0]=="start":
                vertex_join = spins["Center"][e.name]-positions[vertex[0]]
            elif vertex.index[0]=="end":
                vertex_join = positions[vertex[0]]-spins["Center"][e.name]

        else:
            vertex_join = positions[e["end"]]-positions[e["start"]]

        if np.dot(spin_direction,vertex_join)<0:
            ## flip edge
            edges.loc[i,["start","end"]] = edges.loc[i,["end","start"]].values

    return edges

def where_is_edge(e, edge_directory):
    """ What vertex in the edge directory contains the edge 'e'. """
    
    vertices = [i for i in edge_directory if np.isin(e, edge_directory[i])]
    
    if len(vertices)==1:
        vertices.append(-1)
    if len(vertices)!=2:
        print(vertices)
        raise ValueError("edges can only join two vertices")
        
    return vertices

def create_edge_array(edge_directory, spins = None, positions = None):
    """ Retrieve the edge array from the edge_directory. 
    If spins and positions are given they are used to calculate the directions of the edges. 
    """
    
    edge_ids = np.unique(np.array([e for v in tqdm.tqdm(edge_directory) for e in edge_directory[v]]))
    
    edges = np.array([[e,*where_is_edge(e, edge_directory)] 
                      for e in tqdm.tqdm(edge_ids)])
    
    edges = pd.DataFrame(data = edges[:,1:],
                         columns=["start","end"],
                         index=pd.Index(edges[:,0],name="edge"))
    
    if spins is not None and positions is not None:
        edges = update_edge_directions(edges,spins,positions)
    
    return edges

def unique_points(points,tol = 0.1):
    """Returns only the distinct points (with a tolerance)."""
    flatten = lambda lst: [el for l in lst for el in l]

    unique_points = []
    inverse = np.empty(len(points), dtype="uint16")
    copies_assigned = []

    for i,p in enumerate(points):
        if not np.isin(i, flatten(copies_assigned)):

            kdt_points = spa.cKDTree(points)

            same_point_copies = kdt_points.query_ball_point(p, tol)

            copies_assigned.append(same_point_copies)
            unique_points.append(points[same_point_copies].mean(axis=0))
            inverse[same_point_copies] = len(unique_points)-1

    unique_points = np.array(unique_points)
        
    return unique_points, inverse, copies_assigned

def spin_crossing_point(S1,S2):
    # This works well in 2d. In 3d it's triciker
    if not (S1['Direction']==S2['Direction']).all():
        A = np.ones([2,2])
        A[:,0] = S1['Direction']
        A[:,1] = -S2['Direction']

        b = np.array([
            S2['Center'][0]-S1['Center'][0],
            S2['Center'][1]-S1['Center'][1]])

        lam = np.linalg.solve(A,b)
        print("vertex")

        print(S1['Center']+lam[0]*S1['Direction'])
        return S1['Center']+lam[0]*S1['Direction']
    else:
        print(np.Inf+np.zeros(np.shape(S1['Center'])))
        return np.Inf+np.zeros(np.shape(S1['Center']))

def get_vertices_positions(NeighborPairs,spins):
    # From a list of Spins, get neighboring spins, and get the crossing point of each, which defines a vertex.  
    for i,n in enumerate(NeighborPairs):
        
        #NeighborPairs[i]['Vertex'] = spin_crossing_point(spins[n['Pair'][0]],spins[n['Pair'][1]])[0:2] 
        NeighborPairs[i]['Vertex'] = spin_crossing_point_new_shakti(spins[n['Pair'][0]],spins[n['Pair'][1]])[0:2]

    
    return NeighborPairs

def from_neighbors_get_nearest_neighbors(NeighborPairs):
    # This function takes a list of Delaunay Neighbor Pairs and returns only those which are close to the minimum distance.
    NeighborPairs['Distance']=np.around(NeighborPairs['Distance'],decimals=4)
    #print(NeighborPairs['Distance'])
    #NeighborPairs = NeighborPairs[NeighborPairs['Distance']<=np.min(NeighborPairs['Distance'])*1.1]
    NeighborPairs = NeighborPairs[NeighborPairs['Distance']<=31.7351*1.1]

    return NeighborPairs

def ice_to_spins(ice, id_label=None):

    if ice.__class__.__name__ == "colloidal_ice":
        spins = colloidal_ice_vector(ice)
    elif ice.__class__.__name__ == "spins":
        spins = spin_ice_vector(ice)
    elif ice.__class__.__name__ == "DataFrame":
        spins = trj_ice_vector(ice)
    elif ice.__class__.__name__ == "ndarray":
        spins = ice
    
    return spins 

def colloidal_ice_vector(C):
    """Extracts an array of centers and directions from a Colloidal Ice System"""
    Vectors = np.array(np.zeros(len(C)),dtype=[('Center',np.float,(2,)),('Direction',np.float,(2,))])
    i=0
    for c in C:
        Vectors[i] = (c.center[0:2].magnitude,c.direction[0:2])
        i=i+1
    return Vectors

def calculate_neighbor_pairs(Centers):
    """This function makes a list of all the Pairs of Delaunay Neighbors from an array of points"""
    
    tri = sptl.Delaunay(Centers)

    # List all Delaunay neighbors in the system
    NeighborPairs = np.array(np.zeros(2*np.shape(tri.simplices)[0]),
                             dtype=[('Pair',np.int,(2,)),('Distance',np.float),('Vertex',np.float,(2,))])

    i = 0
    for t in tri.simplices:
        NeighborPairs[i]['Pair'] = np.sort(t[0:2])
        NeighborPairs[i]['Distance'] = sptl.distance.euclidean(Centers[t[0]],Centers[t[1]])
        NeighborPairs[i+1]['Pair'] = np.sort(t[1:3])
        NeighborPairs[i+1]['Distance'] = sptl.distance.euclidean(Centers[t[1]],Centers[t[2]])
        i = i+2

    return NeighborPairs

class vertices():
    def __init__(self, positions = None, edges = None, ice = None, id_label = "id", static = True):
        """ Initializes the vertices array.
        Initialization method for the vertices class. 
        Vertices are defined by a set of positions, and a set of directed edges. If any of these are given, then the processing is easier. If they are not given they are inferred from the `input`. If the input is not given, the vertex object is initialized empty, but a topology can be added later by using "colloids_to_vertices", "spins_to_vertices", or "trj_to_vertices". 
        If an object is given to create a vertex array, do so. 
        ---------
        Parameters:
        * positions: Ordered list containing the geometry of the vertices.
        * edges: Ordered list, or disordered set, containing the pairs of vertices that are joined. 
        * ice (colloidal_ice object, trj dataframe, spin_ice object): Initializes the topology, inferring from the input. 
            * colloidal_ice (colloidal_ice object, optional): Initalizes the vertices from a colloidal_ice object
            * spin_ice (spin_ice object, optional): Initializes the vertices from a spin_ice object
            * trj (pd.DataFrame, optional): Initializes the vertices from a pandas array. The pandas array must have the columns [x y z] and [dx dy dz] from which the links direction will be deduced. 
        * id_label (string, optional): If the index of `trj` has more than one level, this is the name of the level that identifies particles. Defaults to "id".
        * static (boolean, True): If the topology of the traps doesn't change, then time can be saved by not recalculating neighbors. Setting this variable to true indicates if static topology can be assumed in case of a MultiIndex. False is not implemented
        
        Attributes: 
        * vertices (DataFrame): contains the possitions of the vertices, plus whatever properties have been calculated. 
        * edges (DataFrame): contains the pairs of vertices that are connected. The edges are directed and go from the first vertex to the second vertex. 
        * edge_directory (dict): indicates which vertices are formed by which edges. The index is the edge number. Each entry contains a list of vertices.
        """
        
        self.vertices = pd.DataFrame({"x":[],"y":[]},
                            index = pd.Index([],name="vertex"))
        self.edges = pd.DataFrame({"start":[],"end":[]},
                             index = pd.Index([],name="edge"))
        self.edge_directory = {}
            
        if positions is not None:
            self.vertices.x = positions[:,0]
            self.vertices.y = positions[:,1]
            
        if edges is not None:
            self.edges.start = edges[:,0]
            self.edges.end = edges[:,1]
                        
    def infer_topology(self, ice, positions=None, method = "voronoi", tolerance = 0.01):
        """ Infer the topology from the spin structure.
        ------------
        Parameters:
        input: object to get the spins from. 
        positions (optional): 
        method (string, "crossings"): Method to infer the positions of the vertices. 
            * "crossings" defines vertices as being in the crossing points of two spins. This is illdefined in more than 2D. 
            * "voronoi" defines vertices as being in the corners of the voronoi tesselation of  
        """
        spins = ice_to_spins(ice)

        neighbor_pairs = calculate_neighbor_pairs(spins['Center'])
        neighbor_pairs = from_neighbors_get_nearest_neighbors(neighbor_pairs)
        neighbor_pairs = get_vertices_positions(neighbor_pairs,spins)
        
        positions, inverse, copies = unique_points(neighbor_pairs['Vertex'])
        self.vertices.x = positions[:,0]
        self.vertices.y = positions[:,1]
        
        self.edge_directory = {i:np.unique(neighbor_pairs[c]["Pair"].flatten()) 
                                for i,c in enumerate(copies)}

        self.edges = create_edge_array(self.edge_directory, spins, positions)   
    
    def update_directions(self, ice):
        """ Updates the directions of the vertices using an ice object """
        positions = self.vertices.loc[:,["x","y"]].values
        spins = ice_to_spins(ice)
        
        self.edges = update_edge_directions(self.edges, spins, positions)
        
    def calculate_coordination(self):
        """ Adds a column to the 'vertices' array with the vertex coordination """
        coordination = [len(self.edge_directory[vertex]) for vertex in self.vertices.index]
        self.vertices["coordination"] = coordination
         
    def calculate_charge(self):
        """ Adds a column to the 'vertices' array with the vertex charge. """
        
        self.vertices["charge"] = 0

        for v_id, vertex in self.vertices.iterrows():
            indegree = (self.edges.loc[self.edge_directory[v_id]].end==v_id).sum()
            outdegree = (self.edges.loc[self.edge_directory[v_id]].start==v_id).sum()
            self.vertices.loc[v_id,"charge"] = indegree-outdegree
    
    def calculate_dipole(self, spins):
        """ Adds two column sto the 'vertices' array with the sum of the directions of the vertex components. """
    
        self.vertices["dx"] = 0
        self.vertices["dy"] = 0

        for v_id, vertex in self.vertices.iterrows():
            self.vertices.loc[v_id,["dx","dy"]] = np.sum(np.array(
                        [spins["Direction"][e] for e in self.edge_directory[v_id]]),
                    axis=0)
            
    def classify_vertices(self, spins):
        
        self.calculate_coordination()
        self.calculate_charge()
        self.calculate_dipole(spins)
         
        return self

    def colloids_to_vertices(self, col):
        """ Uses the col object to infer the topology of the vertices and to classify them."""
        
        spins = ice_to_spins(col)
        self.infer_topology(spins)
        self.classify_vertices(spins)
        
        return self
    
    def trj_to_vertices(self, trj, positions = None, id_label = "id", static = True):
        """ Convert a trj into a vertex array. 
        If trj is a MultiIndex, an array will be saved that has the same internal structure as the passed array, but the identifying column will now refer to vertex numbers. 
        ---------
        Parameters: 
        * trj (pd.DataFrame, optional): Initializes the vertices from a pandas array. The pandas array must have the columns [x y z] and [dx dy dz] from which the links direction will be deduced. 
        * id_label (string, "id"): If the index of `trj` has more than one level, this is the name of the level that identifies particles.
        * static (boolean, True): If the topology of the traps doesn't change, then time can be saved by not recalculating neighbors. Setting this variable to true indicates if static topology can be assumed in case of a MultiIndex.
        """
         
        def trj_to_vertices_single_frame(trj_frame):
            
            spins = ice_to_spins(trj_frame)
            
            if len(self.vertices)==0:
                self.infer_topology(spins, positions=positions)
                self.update_directions(spins) 
                
            else: 
                self.update_directions(spins) 
                
            self.classify_vertices(spins)
            
            return self.vertices.copy(deep=True)
            
        if trj.index.nlevels==1:
                        
            self.vertices = trj_to_vertices_single_frame(trj)
            
            return self
            
        else:
                
            id_i = np.where([n==id_label for n in trj.index.names])
            other_i = list(trj.index.names)
            other_i.remove(other_i[id_i[0][0]])
            
            self.dynamic_array = pd.concat(
                    {o_i:trj_to_vertices_single_frame(trj_oi) for o_i, trj_oi in tqdm.tqdm(trj.groupby(other_i))},
                    names = other_i)
            
            self.vertices = self.dynamic_array
            
            return self
    
    def display(self, ax = None, DspCoord = False, dpl_scale = 1, dpl_width = 5, sl=None):
        
        
        if self.vertices.index.nlevels>1:
            if sl is None:
                sl = self.vertices.index[-1][:-1]
            
        else: 
            sl = slice(None)
        
        vertices = self.vertices.loc[sl]

        if ax is None:
            ax = plt.gca()

        if not DspCoord:
            for i,v in vertices.iterrows():
                if v.charge>0:
                    c = 'r'
                else:
                    c = 'b'
                ax.add_patch(patches.Circle((v.x,v.y),radius = abs(v['charge'])*2,
                    ec='none', fc=c))

                if v.charge==0:
                    X = v.x
                    Y = v.y
                    
                    DX = v['dx']*dpl_scale
                    DY = v['dy']*dpl_scale
                    ax.add_patch(patches.Arrow(X-DX,Y-DY,2*DX,2*DY,width=dpl_width,fc='k'))
                
        if DspCoord: 
            for v in vertices.iterrows:
                if v['charge']>0:
                    c = 'r'
                else:
                    c = 'b'
                    
                ax.add_patch(patches.Circle((v.x,v.y),radius = abs(v['charge'])*2,
                    ec='none', fc=c))
                    
                X = v.x
                Y = v.y
        
        #ax.set_aspect("equal")    
        #plt.axis("equal")
