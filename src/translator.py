# import numpy as np
# import pickle

# class FlameTranslator:
#     def __init__(self, model_path):
#         with open(model_path, 'rb') as f:
#             self.model = pickle.load(f, encoding='latin1')

#         self.v_template = self.model['v_template']
#         self.expression_basis = self.model['shapedirs'][:, :, 300:400]

#     def translate(self, mediapipe_scores):
#         """
#         Takes the list of 52 scores and converts them to FLAME expression parameters.
#         """
#         flame_expr = np.zeros(100)

#         for b in mediapipe_scores:
#             if b.category_name == 'jawOpen':
#                 flame_expr[0] = b.score * 3.5  
#             if b.category_name == 'mouthPucker':
#                 flame_expr[2] = b.score * 2.0
#             if b.category_name == 'mouthSmileLeft' or b.category_name == 'mouthSmileRight':
#                 flame_expr[1] = b.score * 2.0
#             if b.category_name == 'eyeBlinkLeft' or b.category_name == 'eyeBlinkRight':
#                 flame_expr[4] = b.score * 2.0 

#         return flame_expr
    
#     def deform_mesh(self, flame_expr):
#         """
#         Deforms the FLAME mesh using the expression parameters.
#         """
#         v_shaped = self.v_template + np.tensordot(self.expression_basis, flame_expr, axes=[2, 0])
#         return v_shaped

import numpy as np
import pickle
import os

class FlameTranslator:
    def __init__(self, model_path, mappings_path='./mappings'):
        """
        Initialize with FLAME model and optional pre-trained mappings.
        
        Args:
            model_path: Path to FLAME model
            mappings_path: Path to folder with mapping .npy files (optional)
        """
        # Load FLAME model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f, encoding='latin1')

        self.v_template = self.model['v_template']
        self.expression_basis = self.model['shapedirs'][:, :, 300:400]
        
        # Try to load pre-trained mappings
        self.use_pretrained = False
        self.has_pose_mapping = False
        self.has_eye_mapping = False
        
        # Try multiple path formats
        possible_paths = [
            mappings_path,
            os.path.abspath(mappings_path),
            './mappings',
            os.path.join(os.getcwd(), 'mappings')
        ]
        
        for path_attempt in possible_paths:
            try:
                bs2exp_path = os.path.join(path_attempt, 'bs2exp.npy')
                
                if os.path.exists(bs2exp_path):
                    # Load expression mapping (required)
                    self.bs2exp = np.load(bs2exp_path)
                    self.use_pretrained = True
                    print(f"✓ Loaded expression mapping from: {os.path.abspath(path_attempt)}")
                    print(f"  - bs2exp shape: {self.bs2exp.shape}")
                    
                    # Try to load pose mapping (optional)
                    bs2pose_path = os.path.join(path_attempt, 'bs2pose.npy')
                    if os.path.exists(bs2pose_path):
                        self.bs2pose = np.load(bs2pose_path)
                        self.has_pose_mapping = True
                        print(f"  - bs2pose shape: {self.bs2pose.shape}")
                    else:
                        print(f"  ⚠ bs2pose.npy not found (using fallback)")
                        # Create a simple fallback for jaw pose
                        self.bs2pose = np.zeros((52, 3))
                        self.bs2pose[25, 0] = 0.5  # jawOpen -> jaw rotation
                    
                    # Try to load eye mapping (optional)
                    bs2eye_path = os.path.join(path_attempt, 'bs2eye.npy')
                    if os.path.exists(bs2eye_path):
                        self.bs2eye = np.load(bs2eye_path)
                        self.has_eye_mapping = True
                        print(f"  - bs2eye shape: {self.bs2eye.shape}")
                    else:
                        print(f"  ⚠ bs2eye.npy not found (using fallback)")
                        # Create a simple fallback for eye pose
                        self.bs2eye = np.zeros((52, 6))
                    
                    break
            except Exception as e:
                continue
        
        if not self.use_pretrained:
            print("⚠ Pre-trained mappings not found, using fallback manual mapping")
            print(f"  Searched in: {[os.path.abspath(p) for p in possible_paths]}")
            print("  To use pre-trained mappings, download bs2exp.npy to: ./mappings/")
            print("  From: https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame")
            
        if self.use_pretrained:
            # MediaPipe blendshape order
            self.blendshape_names = [
                '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 
                'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
                'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
                'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
                'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
                'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward',
                'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft',
                'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
                'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
                'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
                'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
                'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
                'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight'
            ]
        else:
            self.blendshape_names = None

    def mediapipe_to_array(self, mediapipe_scores):
        """Convert MediaPipe scores to ordered numpy array."""
        score_dict = {b.category_name: b.score for b in mediapipe_scores}
        blendshape_array = np.zeros(52)
        for i, name in enumerate(self.blendshape_names):
            if name in score_dict:
                blendshape_array[i] = score_dict[name]
        return blendshape_array

    def translate(self, mediapipe_scores):
        """
        Convert MediaPipe scores to FLAME parameters.
        Returns (expression, jaw_pose, eye_pose) if using pre-trained,
        or just expression if using manual mapping.
        """
        if self.use_pretrained:
            # Use pre-trained linear mappings
            blendshape_array = self.mediapipe_to_array(mediapipe_scores)
            expression = blendshape_array @ self.bs2exp
            jaw_pose = blendshape_array @ self.bs2pose
            eye_pose = blendshape_array @ self.bs2eye
            
            # AMPLIFY EXPRESSIONS for better visibility
            expression = expression * 2.0  # 2x amplification
            
            return expression, jaw_pose, eye_pose
        else:
            # Fallback: Manual mapping with AMPLIFIED multipliers
            flame_expr = np.zeros(100)
            for b in mediapipe_scores:
                if b.category_name == 'jawOpen':
                    flame_expr[0] = b.score * 8.0
                if b.category_name == 'mouthSmileLeft':
                    flame_expr[1] = b.score * 6.0
                if b.category_name == 'mouthSmileRight':
                    flame_expr[2] = b.score * 6.0
                if b.category_name == 'browInnerUp':
                    flame_expr[3] = b.score * 5.0
                if b.category_name == 'eyeBlinkLeft':
                    flame_expr[4] = b.score * 8.0
                if b.category_name == 'eyeBlinkRight':
                    flame_expr[5] = b.score * 8.0
                if b.category_name == 'mouthFunnel':
                    flame_expr[6] = b.score * 6.0
                if b.category_name == 'mouthPucker':
                    flame_expr[7] = b.score * 6.0
            return flame_expr, None, None
    
    def deform_mesh(self, flame_expr, jaw_pose=None):
        """
        Deform FLAME mesh using expression parameters.
        jaw_pose is optional and not yet implemented.
        """
        v_shaped = self.v_template + np.tensordot(self.expression_basis, flame_expr, axes=[2, 0])
        return v_shaped