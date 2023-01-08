# This file contains the Mesh class and Grid class


# --------- TO DO ---------

# Add labelling of mesh in Mesh class (in save function also)
# Create PCA rotation function in Mesh class

# -------------------------

import numpy as np
import math
import scipy.spatial.distance
import time
from operator import add
from scipy import sparse
import nt_tools
import json
import os


class Mesh:

    def __init__(self, filename):

        # Setting name
        self.name = filename[ filename.rfind('/')+1:]

        # creating self.points and self.faces
        if (filename.find(".obj") != -1):
            self.points, self.faces  = [], []
            file = open(filename)
            for line in file:
                line = line.strip().split(' ')
                if line[0]=='v':
                    pt = list(map(float, line[1:]))
                    self.points.append(pt)
                if line[0]=='f':

                    n1 = int (line[1].split('/', 1)[0]) - 1
                    n2 = int ( line[2].split('/', 1)[0]) - 1
                    n3 = int (line[3].split('/', 1)[0]) - 1
                    face = [n1, n2, n3]
                    self.faces.append(face)
            self.points = np.array(self.points)
            self.points = self.points
            self.faces = np.array(self.faces)
            self.faces = self.faces.T
            self.labels = self.extract_labels(filename)

            self.failed = False
        else:
            print ("Failed to load mesh ", filename)
            self.failed = True

        # Setting faces, creating edges (without repetition)
        F = self.faces
        E = np.hstack((F[[0, 1], :], F[[1, 2], :], F[[2, 0], :]))
        E = nt_tools.unique_columns(np.hstack((E, E[[1, 0], :])))
        self.edges = E[:, E[0, :] < E[1, :]]

        # center of gravity:
        self.cog = np.sum(self.points, axis=1)/self.nb_pts()

        # Printing informations
        print ("mesh is ", filename)
        print ("loaded mesh has ", self.nb_pts(), " vertices and ", self.nb_edges(), " edges.")

    def save(self, filename):
        f = open(filename, 'w')

        self.failed_save = False
        # Saving in .obj format (better)
        if (filename.find('.obj') != -1):
            for i in range(self.nb_pts()):
                f.write( 'v ' + ' '.join(list (map(str, self.get_pt(i)))) + '\n' )
            for face in self.faces.T:
                facecopy = face.copy()
                facecopy += [1, 1, 1]
                f.write( 'f ' + ' '.join(list( map (str, facecopy))) + '\n')

        # Saving in .off format
        elif (filename.find('.off') != -1):
            f.write ("OFF\n")
            f.write (str(self.nb_pts()) + ' ' + str(self.nb_faces()) + ' 0\n')
            for pt in self.points.T:
                f.write (' '.join(list (map(str, pt))) + '\n')
            for face in self.faces.T:
                f.write('3 ' + ' '.join(list(map(str, face))) + '\n')
        
        else:
            print ("couldn't save mesh ", self.name)
            self.failed_save = True      
            
    def extract_labels(self, filename):
        json_path = get_json(filename)
        return labels_from_json(json_path)
        
                  
    def barycentric_to_cartesian(self, key):
        if isinstance(self.labels[key][0],list):
            coord = self.labels[key]
            return coord[0][1] * self.points[coord[0][0]] + coord[1][1] * self.points[coord[1][0]] + coord[2][1] * self.points[coord[2][0]]
        
    def change_coord(self):
        for key in self.labels.keys():
            print(key)
            self.labels[key] = self.barycentric_to_cartesian(key)

        
    def compute_PC (self):
        # this functions computes the coordinates of the centered mesh
        # rotated along its principal components

        # Do not compute again if already computed
        if not hasattr(self, 'PC_matrix'):
            centered = (self.points.T - self.cog.T).T
            cov = 1 / (self.nb_pts()-1) * centered @ centered.T
            self.PCvals, self.PCvecs = np.linalg.eig(cov)
            print (centered.shape, self.PCvecs.shape)
            self.points_along_PCs = self.PCvecs.T @ centered

    def compute_grid (self, dimensions, fill='vertices', along_PCs = True):
        # computes and saves the grid for this mesh. For arguments see grid.__init__() function
        # if along_PCs==True, grid will be filled with self.points_along_PCs (see Mesh.compute_PC())
        if not hasattr(self, 'grid'):
            if along_PCs:
                self.compute_PC()
                real_points = self.points
                self.points = self.points_along_PCs
                
            self.grid = Grid(self, dimensions, fill)

            if along_PCs:
                self.points = real_points
            
            

    def __str__(self):
        return "mesh : " + self.name +  "; has " +  str(self.nb_pts()) + " vertices, " + \
               str(self.nb_edges()) +  " edges and " + str(self.nb_faces()) + " faces." 
    
    def hausdorff_distance_detailed (self, other_mesh):
        '''
        :return: (a,b) where a is distance self->other_mesh, b is distance other_mesh->self + vertices of interest
        '''
        return (scipy.spatial.distance.directed_hausdorff(self.points.T, other_mesh.points.T),
                scipy.spatial.distance.directed_hausdorff(other_mesh.points.T, self.points.T))

    def hausdorff_distance (self, other_mesh):
        hdd = self.hausdorff_distance_detailed(other_mesh)
        return max(hdd[0][0], hdd[1][0])

    def L2_distance (self, other_mesh):
        assert self.points.shape==other_mesh.points.shape
        return np.linalg.norm (self.points - other_mesh.points)

    def add_gaussian_noise (self, sigma):
        '''
        :param sigma: WARNING ! This will be multiplied by self.mean_edge_length()
        :return:
        '''
        noise = self.mean_edge_length()*sigma*np.random.randn(self.nb_pts())
        normals = self.normals().T
        for i in range(self.nb_pts()):
            self.set_pt(i, self.get_pt(i) + noise[i]*normals[i])

    def normals(self):
        return nt_tools.compute_normal(self.points, self.faces)
    def nb_pts(self):
        return self.points.shape[1]
    def nb_edges(self):
        return self.edges.shape[1]
    def nb_faces(self):
        return self.faces.shape[1]
    def get_pt(self, i):
        return self.points[:,i]
    def set_pt (self, i, pt):
        self.points[:,i] = pt
    def get_face(self, i):
        return self.faces[:,i]
    def get_edge(self, i):
        return self.edges[:,i]
    def get_cedge(self,i):
        return np.array([self.get_pt(self.get_edge(i)[0]),
                         self.get_pt(self.get_edge(i)[1])]).T

    def get_cface(self, i):
        face = self.get_face(i)
        return np.array([self.get_pt(face[0]),
                         self.get_pt(face[1]),
                         self.get_pt(face[2])]).T
    
    def mean_edge_length(self):
        tot = 0
        for i in range(self.nb_edges()):
            cedge = self.get_cedge(i)
            diff = cedge[:,1] - cedge[:,0]
            tot += np.linalg.norm(diff)
        return tot / self.nb_edges()

    def median_edge_length(self):
        v = self.length_edges()
        v.sort()
        return v[math.floor(np.size(v)/2)]

    def centroid_face (self, i):
        cface = self.get_cface(i)
        return np.sum(cface, axis=1)/3

    def area_face (self, i):
        # This formula is known.
        triangle = self.get_cface(i).T
        a = np.linalg.norm(triangle[1] - triangle[0])
        b = np.linalg.norm(triangle[2] - triangle[1])
        c = np.linalg.norm(triangle[2] - triangle[0])
        s = (a + b + c) / 2
        return (s * (s - a) * (s - b) * (s - c)) ** 0.5

    def length_edge(self, i):
        cedge = self.get_cedge(i)
        diff = cedge[:, 1] - cedge[:, 0]
        return np.linalg.norm(diff)

    def length_edges(self):
        v = []
        for i in range(self.nb_edges()):
            v.append(self.length_edge(i))
        return v

    def is_equal (self, other_mesh):
        return (np.array_equal( self.points, other_mesh.points) and
                np.array_equal(self.faces,other_mesh.faces))



def all_the_values(dictionary):
    for keys , values in dictionary.items():
        if isinstance(values, dict):
            for x in all_the_values(values):
                yield x
        else:
            yield (keys,values)
            
            
def labels_from_json(json_path):
    dic_labels = {}
    with open(json_path) as json_file:
         dic = json.load(json_file)
    for label in all_the_values(dic):
        if isinstance(label[1],list):
            dic_labels[label[0]]=label[1]
    return dic_labels


def get_json(mesh_path):
    json_dir=os.path.join(os.path.dirname(mesh_path),'appData')
    jsons = os.listdir(json_dir)
    if ('left' in mesh_path) and ('patient_left.json' in jsons):
        return os.path.join(json_dir,'patient_left.json')
    if ('right' in mesh_path) and ('patient_right.json' in jsons):
        return os.path.join(json_dir,'patient_right.json')
    if 'patient.json' in jsons:
        return os.path.join(json_dir,'patient.json')
    raise ValueError('json file not found')




class Grid:
    '''
    a 3D grid filled with indices (ie. a 4D list)

    '''

    def __init__(self, mesh, dimensions, fill='vertices'):


        # dimensions: should be a tuple of three natural numbers
        # fill : If 'vertices', vertices fill the grid. If 'faces', centroid of faces fill the grid.

        # finding the bounding box
        self.minbox = np.amin(mesh.points.T, axis=0)
        self.maxbox = np.amax(mesh.points.T, axis=0)
        self.deltabox = self.maxbox - self.minbox
        #print ("bounding box is size ", self.deltabox)
        self.dimensions = np.array(dimensions)
        
        # dimensions of one small cube : 
        [self.rx, self.ry, self.rz] = self.deltabox / self.dimensions

        # Creating  the grid
        self.grid = [[[[] for _ in range(self.dimensions[2])] for _ in range(self.dimensions[1])] for _ in range(self.dimensions[0])]
        # print ("grid size \n", np.shape(np.array(grid)))
        
        # Filling the grid

        self.fill = fill

        if fill=='faces':
            for i in range(mesh.nb_faces()):
                c = mesh.centroid_face(i)
                self._add_pt(c,i)
            self.nb_pts = mesh.nb_faces()
        
        elif fill=='vertices':
            for i in range(mesh.nb_pts()):
                c = mesh.get_pt(i)
                self._add_pt(c,i)
            self.nb_pts = mesh.nb_pts()

    def as_density (self):

        # returns a 3D grid G where G[i,j,k] = proportion of points in grid[i,j,k]
        if not hasattr(self, 'density_grid'):
            # len(self.grid[i,j,k])
            self.density_grid = [[[len(self.grid[i][j][k]) for k in range(self.dimensions[2])] for j in range(self.dimensions[1])] for i in range(self.dimensions[0])]
            self.density_grid = np.array(self.density_grid, dtype='float64')
            self.density_grid /= self.nb_pts
        return self.density_grid

    
    def neighbours (self, new_point):
        nei = []
        for cube in self._where_with_neighbours(new_point):
            #print ("considering cube : ", cube)
            #print ("vertices in this cube : ", self._point_indices_in_cube(cube))
            nei = nei + self._point_indices_in_cube(cube)
        #print ("returning ", nei)
        return nei

    
    def _add_pt(self, pt, index):
        # warning : this point will add points not in the bounding box simply on the extrema of the grid, without error raised.
        # index corresponds to the 'identity' of the vertex
        coor = self._where(pt)
        # putting it "back in the grid"
        coor = np.minimum(np.array(coor), self.dimensions -1)
        coor = np.maximum(np.array(coor), np.zeros_like(coor))
        
        self.grid [coor[0]][coor[1]][coor[2]].append(index)




    def _where (self, new_point):
        dpt = new_point - self.minbox
        ret = [ math.floor(dpt[0]/self.rx), math.floor(dpt[1]/self.ry), math.floor(dpt[2]/self.rz) ]
        #print ("new point is in coordinate ", ret )
        #print ("sizebox is ", self.sizebox)

        return ret

    def _where_with_neighbours (self, new_point):

        mylist = []
        coor = self._where(new_point)

        # ... Add neighbour cubes

        for dx in range((-1)*(coor[0] > 0), 1+(coor[0] < self.sizebox[0]-1)):
            for dy in range(-(coor[1] > 0), 1+(coor[1] < self.sizebox[1] - 1)):
                for dz in range(-(coor[2] > 0), 1+(coor[2] < self.sizebox[2]-1)):
                    mylist.append(list(map(add, coor, [dx, dy, dz])))
        #print ("gridsize :", self.sizebox)
        #print ("cube is ", coor)
        #print ("list of neighbouring coordinates : ", mylist)
        return mylist

    def _point_indices_in_cube (self, cube):
        '''
        :param cube: a 1x3 list or np array with coordinates of the cube
        :return: the points registered inside the cube.
        '''
        return self.grid[cube[0]][cube[1]][cube[2]]




