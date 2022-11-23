from mesh_and_grid import Mesh, Grid

def test_load_mesh ():
    dragon = Mesh('meshes/dragon.obj')
    assert dragon.name == "dragon.obj"
    assert dragon.failed == False
    assert dragon.nb_pts() == 1190
    assert dragon.nb_faces() == 2564

def test_save_mesh ():
    pass

def test_grid_as_density():
    dragon = Mesh("meshes/dragon.obj")
    grid = Grid (dragon, (10, 10, 10))
    density_grid = grid.as_density()
    assert density_grid.sum() == 1



