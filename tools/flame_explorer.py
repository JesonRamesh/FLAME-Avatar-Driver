import pickle
import numpy as np
import pyvista as pv

class FlameExpressionExplorer:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f, encoding='latin1')
        
        self.v_template = self.model['v_template']
        self.expression_basis = self.model['shapedirs'][:, :, 300:400]
        self.faces = self.model['f']
        
    def visualize_expression(self, expr_index, intensity=3.0):
        """
        Visualize a single FLAME expression component
        expr_index: 0-99 (which expression to activate)
        intensity: how much to activate it (try 1.0 to 5.0)
        """
        flame_expr = np.zeros(100)
        flame_expr[expr_index] = intensity
        
        # Deform mesh
        v_shaped = self.v_template + np.tensordot(self.expression_basis, flame_expr, axes=[2, 0])
        
        # Create PyVista mesh
        pv_faces = np.hstack(np.c_[np.full(len(self.faces), 3), self.faces])
        mesh = pv.PolyData(v_shaped, pv_faces)
        
        # Plot
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='cyan')
        plotter.camera_position = [(0, 0, 1.5), (0, 0, 0), (0, 1, 0)]
        plotter.add_text(f"Expression {expr_index} @ intensity {intensity}", 
                        position='upper_edge', font_size=12)
        plotter.show()
    
    def find_expressions_by_pattern(self):
        """
        Test multiple expressions to find patterns
        This helps you map MediaPipe blendshapes to FLAME indices
        """
        print("Testing FLAME expressions to identify their effects...")
        print("Common patterns found in FLAME models:")
        print("  0-10: Often jaw/mouth related")
        print("  10-20: Lip movements")
        print("  20-30: Cheeks and smile")
        print("  30-40: Eyes and brows")
        print("  40-50: Nose")
        print("\nTry visualizing expressions 0-50 one by one to map them!")

if __name__ == "__main__":
    explorer = FlameExpressionExplorer('model/generic_model.pkl')
    
    # Example: Visualize expression 0 (usually jaw open)
    print("Showing expression 0 (likely jaw open)...")
    explorer.visualize_expression(4, intensity=3.0)
    
    # Uncomment to test other expressions:
    # explorer.visualize_expression(1, intensity=3.0)  # Test expression 1
    # explorer.visualize_expression(2, intensity=3.0)  # Test expression 2
    # etc...
    
    explorer.find_expressions_by_pattern()