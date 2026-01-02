import pyvista as pv
import trimesh
import numpy as np

class Visualizer:
    def __init__(self, faces):
        self.faces = faces
        self.plotter = pv.Plotter(off_screen=False)
        self.mesh_actor = None
        
        # Add a placeholder so the window isn't empty (prevents white screen freeze)
        pv_faces = np.hstack(np.c_[np.full(len(faces), 3), faces])
        self.poly_data = pv.PolyData(np.zeros((5023, 3)), pv_faces)
        self.mesh_actor = self.plotter.add_mesh(self.poly_data, color='cyan')
        
        # Position the camera: [x, y, z] position, [x, y, z] focus, [x, y, z] view-up
        self.plotter.camera_position = [(0, 0, 1.2), (0, 0, 0), (0, 1, 0)]
        self.plotter.show(interactive=False, auto_close=False)

    def update_mesh(self, vertices):
        self.poly_data.points = vertices
        # This keeps the camera from "jumping" while the face moves
        self.plotter.render()