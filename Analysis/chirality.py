import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def take_trap_pos(col_trj, frame):
    
    """ The following function takes the traps positions of a certain frame from the col_trj dataframe and 
    generates an array with the coordinates """
    
    col_trj_frame = col_trj.loc[frame]
    x = col_trj_frame.x.values
    y = col_trj_frame.y.values
    
    coord = np.stack((np.ravel(x), np.ravel(y)), axis=-1)
    
    return coord, col_trj_frame

def take_trap_pos_ao(col_trj, frame):
    
    """ The following function takes the traps positions of a certain frame from the col_trj dataframe and 
    generates an array with the coordinates """
    
    ## AO slight speedup maybe
    col_trj_frame = col_trj.loc[frame]
    
    coord = col_trj_frame.loc[:,["x","y"]].values
    
    return coord, col_trj_frame

def take_pentagon_coord(df, pentagon):
    
    """ The following function takes one pentagon and put the center positions of the edges in an array of coordinates """
    
    # AO, here you can use the same as in take_trap_pos_ao
    pentagon = df.loc[pentagon]
    x = pentagon.x.values
    y = pentagon.y.values
    
    coord = np.stack((np.ravel(x), np.ravel(y)), axis=-1)
    
    return coord, pentagon

def take_spin_dir(col_trj, df, frame = int ,pentagon_n = int):
    
    """ This function, pairs the traps in col_trj with the corresponding pentagon of df. After that the dx, dy, dz vectors
    from col_trj are copied next to the corresponding pentagon position. In order to couple "experimental" directions with 
    the pentagons forming the system """
    
    coord, col_trj_frame = take_trap_pos(col_trj, frame)
    
    coord_p, pentagon = take_pentagon_coord(df, pentagon_n)
    
    KDTree = scipy.spatial.KDTree(coord)
    dist, nn_index = KDTree.query(coord_p, k=1)
    
    i = 0
    for match in nn_index:
        
        pentagon.iloc[i,6] = col_trj_frame.loc[match].dx
        pentagon.iloc[i,7] = col_trj_frame.loc[match].dy
        pentagon.iloc[i,8] = col_trj_frame.loc[match].dz    
        i = i+1
    
    
    return pentagon

def take_spin_all_pentagons(col_trj, df, frame = int):
    
    """ In this function we will perform the same process as in take_spin_dir() but iterating along all the pentagons
    the output will be a new pentagons dataframe with all the spin information included"""
    
    data = []
    
    for each_pen in np.unique(df.index.values): 
         
        
        data.append(take_spin_dir(col_trj, df, frame = frame, pentagon_n = each_pen))    
   
    pentagons = pd.concat(data)

    return pentagons

def vector_coord( df, col1 = str, col2 = str, col3 = str):
    
    vec_coord = df[[col1, col2, col3]].to_numpy()

    return vec_coord

def plot_pentagon_and_cross_prod(pentagons, n_pentagon = int):
    
    
    """ The following function, plot the pentagon center, the spins, 
    and the cross product of sigma^spin_direction"""
    
    # < Choose one pentagon >
    first_pen = pentagons.loc[n_pentagon]
    
    sigma = []
    spin = []
    
    # < Compute the cross product >
    for i in range(len(first_pen)):
        row = first_pen.iloc[i]
        
        
        sigma.append(vector_coord( row, col1 = 'dz', col2 = 'dz', col3 = 'z_c'))
        #sigma.append(vector_coord( row, col1 = 'y_c', col2 = 'y_c', col3 = 'z_c'))
        spin.append(vector_coord(row, col1 = 'dx', col2 = 'dy', col3 = 'dz'))
        cross = np.cross(sigma, spin)
        
        
    # < Plot results > 
    fig, ax1 = plt.subplots(figsize=(5,5))
    for index, row in first_pen.iterrows():
    
        ax1.add_patch(patches.Arrow(row.x-row.dx,row.y-row.dy,2*row.dx,2*row.dy,width=5,fc='b'))

    plt.plot(first_pen.x, first_pen.y, 'o', color = 'yellow')
    plt.plot(first_pen.x_c, first_pen.y_c, 'o', color = 'green')
    #plt.plot(first_pen.iloc[0].x, first_pen.iloc[0].y, 'o', markersize = 10, color = 'yellow')


    for i in range(len(first_pen)):

        row = first_pen.iloc[i]
    
        ax1.add_patch(patches.Arrow(first_pen.iloc[i].x,first_pen.iloc[i].y,cross[i,0],cross[i,1],width=5,fc='red'))

    plt.axis('equal')  
    
    return cross


def cross_prod(pentagons, n_pentagon = int):
    
    
    """ The following function, plot the pentagon center, the spins, 
    and the cross product of sigma^spin_direction"""
    
    # < Choose one pentagon >
    first_pen = pentagons.loc[n_pentagon]
    
    sigma = []
    spin = []
    
    # < Compute the cross product >
    for i in range(len(first_pen)):
        row = first_pen.iloc[i]
        
        
        sigma.append(vector_coord( row, col1 = 'dz', col2 = 'dz', col3 = 'z_c'))
        #sigma.append(vector_coord( row, col1 = 'y_c', col2 = 'y_c', col3 = 'z_c'))
        spin.append(vector_coord(row, col1 = 'dx', col2 = 'dy', col3 = 'dz'))
        cross = np.cross(sigma, spin)  
    
    return cross

def compute_r_vector(pentagons, n_pentagon = int):
    
    r = []
    
    first_pen = pentagons.loc[n_pentagon]
      
    for i in range(len(first_pen)):
            
        row = first_pen.iloc[i]
    
        r.append([row.x-row.x_c, row.y-row.y_c, row.z])
    
    r = np.array(r)
    
    return r

def chirality_ind_spin(pentagons, n_pentagon = int):
    
    """ This function compute the chirality of each spin in choosen poligon """
    
    cross = cross_prod(pentagons, n_pentagon)
    r = compute_r_vector(pentagons, n_pentagon)
    chi = []
    
    for i in range(0,5):
        
        chi.append(np.dot(cross[i],r[i])/abs(np.dot(cross[i],r[i])))
        
    Chi = np.sum(chi)/np.sum(np.abs(chi))
    
    first_pen = pentagons.loc[n_pentagon]
    
    first_pen['Chi'] = Chi;
    
    
    
    return first_pen

def chirality(pentagons):
   
    df = []
    
    for i in np.unique(pentagons.index.values):
        
        df.append(chirality_ind_spin(pentagons, i))
    
    new_pentagons = pd.concat(df)
        
    return new_pentagons

def show_chirality_frame(new_pentagons,name,frame):
    
    fig, ax = plt.subplots(figsize=(15,15))
    
    for ind, typ in new_pentagons.groupby('pentagon index'):
        
        if (np.unique(typ.Chi)) == 0.2:
            
            ax.plot(typ.x_c, typ.y_c, 'x', color = 'gray', markersize = 15)

        
        elif (np.unique(typ.Chi)) == -0.2:
            
            ax.plot(typ.x_c, typ.y_c, 'x', color = 'gray', markersize = 15)

            
        elif (np.unique(typ.Chi)) == 0.6:
            
            ax.plot(typ.x_c, typ.y_c, 'x', color = 'gray', markersize = 15)

            
        elif (np.unique(typ.Chi)) == -0.6:
            
            ax.plot(typ.x_c, typ.y_c, 'x', color = 'gray', markersize = 15)

            
        elif (np.unique(typ.Chi)) == 1:
            
            ax.plot(typ.x_c, typ.y_c, 'o', color = 'orange', markersize = 20)
            
        elif (np.unique(typ.Chi)) == -1:
            
            ax.plot(typ.x_c, typ.y_c, 'o', color = 'blue', markersize = 20)
            
    
    plt.xlim(0,800)
    plt.ylim(0,-900)
    plt.axis('equal')

    legend_elements = [
                  Line2D([0], [0], marker='o', color='orange', label='$\chi$ = + 1',
                          markerfacecolor='orange', markersize=30),
                  Line2D([0], [0], marker='o', color='blue', label='$\chi$ = - 1',
                          markerfacecolor='blue', markersize=30)]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize = 30)

    plt.axis('off') 
    plt.savefig(name+frame, bbox_inches='tight')
    
def plot_frame(new_pentagons, name , frame):
    


    mpl.rcParams['lines.markersize'] = 20
    mpl.rcParams["font.family"] = "Arial"
    
    
    fig, ax = plt.subplots(figsize=(15,15))

    cm = plt.cm.get_cmap('bwr')

    C = ax.scatter(new_pentagons.x_c, new_pentagons.y_c, c = new_pentagons.Chi, cmap=cm, )


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axis('equal')
    ax.axis('off')
    cbar = plt.colorbar(C, cax =cax)
    cbar.set_label( 'Chirality, $\chi$',rotation=270, size = 30, labelpad=40)
    cbar.ax.tick_params(labelsize=20) 
    
    plt.savefig(name+frame)
    
def show_chirality_unit(new_pentagons):
    
    fig, ax = plt.subplots(figsize=(15,15))
    
    for ind, typ in new_pentagons.groupby('pentagon index'):
            
        if (np.unique(typ.Chi)) == 1:
            
            ax.add_patch(patches.Circle((np.unique(typ.x_c),np.unique(typ.y_c)),radius = 17,
                    ec='none', fc='black'))
            
        elif (np.unique(typ.Chi)) == -1:
            
            ax.add_patch(patches.Circle((np.unique(typ.x_c),np.unique(typ.y_c)),radius = 17,
                    ec='none', fc='gray'))
    plt.legend()
    plt.xlim(0,800)
    plt.ylim(0,-900)
    plt.axis('equal')
        #print(np.unique(typ.Chi))
        #print(typ.Chi)
        
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
                  Line2D([0], [0], marker='o', color='w', label='$\chi$ = + 1',
                          markerfacecolor='black', markersize=30),
                  Line2D([0], [0], marker='o', color='w', label='$\chi$ = - 1',
                          markerfacecolor='gray', markersize=30),]

# Create the figure
#fig, ax = plt.subplots()
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize = 30)

    plt.axis('off')