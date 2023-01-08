
import os
import json
from mesh_and_grid import *

path_mesh_frog='/Users/maximescali/Desktop/PROG_IT/Project_DL/dl22/meshes/frog_body.obj'
path_mesh_leg='/Users/maximescali/Desktop/PROG_IT/Project_DL/DATA/leg_half/2somrjGxGuTb_left.obj'
other = '/Users/maximescali/Desktop/PROG_IT/Project_DL/DATA_raw/KAFO/QxptjDzf8edc/QxptjDzf8edc.obj'

leg=Mesh(other)
print(leg.labels)
leg.change_coord()
print(leg.labels)