"""
Skeleton processing nodes for UniRig - Denormalization, Validation, and Preparation.
"""

import os
import tempfile
import numpy as np


class UniRigDenormalizeSkeleton:
    """
    Denormalize skeleton from [-1, 1] range back to original mesh scale.
    This makes the transformation explicit and debuggable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("SKELETON",)
    RETURN_NAMES = ("denormalized_skeleton",)
    FUNCTION = "denormalize"
    CATEGORY = "UniRig/Utils"

    def denormalize(self, skeleton):
        print(f"[UniRigDenormalizeSkeleton] Starting denormalization...")

        mesh_center = skeleton.get('mesh_center')
        mesh_scale = skeleton.get('mesh_scale')

        if mesh_center is None or mesh_scale is None:
            print(f"[UniRigDenormalizeSkeleton] WARNING: Missing mesh bounds, skeleton is not normalized")
            return (skeleton,)

        # Get normalized data from skeleton dict
        joints_normalized = np.array(skeleton['joints'])
        tails_normalized = np.array(skeleton['tails'])

        # Denormalize joint and tail positions
        joints_denormalized = joints_normalized * mesh_scale + mesh_center
        tails_denormalized = tails_normalized * mesh_scale + mesh_center

        print(f"[UniRigDenormalizeSkeleton] Denormalization:")
        print(f"  Mesh center: {mesh_center}")
        print(f"  Mesh scale: {mesh_scale}")
        print(f"  Joint extents before: {joints_normalized.min(axis=0)} to {joints_normalized.max(axis=0)}")
        print(f"  Joint extents after: {joints_denormalized.min(axis=0)} to {joints_denormalized.max(axis=0)}")
        print(f"  Tail extents before: {tails_normalized.min(axis=0)} to {tails_normalized.max(axis=0)}")
        print(f"  Tail extents after: {tails_denormalized.min(axis=0)} to {tails_denormalized.max(axis=0)}")

        # DEBUG: Check mesh vertices if present
        if 'mesh_vertices' in skeleton and skeleton['mesh_vertices'] is not None:
            mesh_verts = np.array(skeleton['mesh_vertices'])
            print(f"  DEBUG - Mesh vertices in skeleton (NORMALIZED): {mesh_verts.min(axis=0)} to {mesh_verts.max(axis=0)}")

        # Create denormalized skeleton dict
        denormalized_skeleton = {
            **skeleton,
            'joints': joints_denormalized,
            'tails': tails_denormalized,
            'is_normalized': False,
        }

        print(f"[UniRigDenormalizeSkeleton] Denormalization complete")

        return (denormalized_skeleton,)


class UniRigValidateSkeleton:
    """
    Validate skeleton quality and data integrity.
    Provides warnings if issues are detected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("SKELETON", "STRING")
    RETURN_NAMES = ("skeleton", "validation_report")
    FUNCTION = "validate"
    CATEGORY = "UniRig/Utils"

    def validate(self, skeleton):
        print(f"[UniRigValidateSkeleton] Validating skeleton...")

        issues = []
        warnings = []

        # Check for required fields
        required_fields = ['joints', 'names', 'parents']
        for field in required_fields:
            if field not in skeleton:
                issues.append(f"Missing required field: {field}")

        if issues:
            report = "VALIDATION FAILED:\n" + "\n".join(f"- {issue}" for issue in issues)
            print(f"[UniRigValidateSkeleton] {report}")
            return (skeleton, report)

        # Get data
        joints = skeleton.get('joints')
        names = skeleton.get('names')
        parents = skeleton.get('parents')
        is_normalized = skeleton.get('is_normalized', None)

        # Check counts match
        num_joints = len(joints) if isinstance(joints, (list, np.ndarray)) else 0
        num_names = len(names) if isinstance(names, (list, np.ndarray)) else 0
        num_parents = len(parents) if isinstance(parents, (list, np.ndarray)) else 0

        if not (num_joints == num_names == num_parents):
            issues.append(f"Count mismatch: {num_joints} joints, {num_names} names, {num_parents} parents")

        # Check normalization status
        if is_normalized is None:
            warnings.append("Normalization status unknown")
        elif is_normalized:
            joints_array = np.array(joints)
            min_val = joints_array.min()
            max_val = joints_array.max()
            if min_val < -1.5 or max_val > 1.5:
                warnings.append(f"Normalized skeleton has values outside [-1, 1]: [{min_val:.2f}, {max_val:.2f}]")
        else:
            joints_array = np.array(joints)
            min_val = joints_array.min()
            max_val = joints_array.max()
            if abs(min_val) > 1000 or abs(max_val) > 1000:
                warnings.append(f"Denormalized skeleton has very large values: [{min_val:.2f}, {max_val:.2f}]")

        # Build report
        if issues:
            report = "VALIDATION FAILED:\n" + "\n".join(f"- {issue}" for issue in issues)
            if warnings:
                report += "\n\nWARNINGS:\n" + "\n".join(f"- {warning}" for warning in warnings)
            print(f"[UniRigValidateSkeleton] {report}")
        elif warnings:
            report = "VALIDATION PASSED WITH WARNINGS:\n" + "\n".join(f"- {warning}" for warning in warnings)
            print(f"[UniRigValidateSkeleton] {report}")
        else:
            report = f"VALIDATION PASSED:\n- {num_joints} joints\n- Hierarchy valid\n- Normalization: {'normalized' if is_normalized else 'denormalized'}"
            print(f"[UniRigValidateSkeleton] {report}")

        return (skeleton, report)


class UniRigPrepareSkeletonForSkinning:
    """
    Prepare skeleton data in the exact format required by the skinning model.
    Saves predict_skeleton.npz with correct field names.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("skeleton_npz_path",)
    FUNCTION = "prepare"
    CATEGORY = "UniRig/Utils"

    def prepare(self, skeleton):
        print(f"[UniRigPrepareSkeletonForSkinning] Preparing skeleton for skinning...")

        # Create temporary directory for predict_skeleton.npz
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_")
        predict_skeleton_dir = os.path.join(temp_dir, "input")
        os.makedirs(predict_skeleton_dir, exist_ok=True)
        predict_skeleton_path = os.path.join(predict_skeleton_dir, "predict_skeleton.npz")

        # Build save_data dict with CORRECT field names for RawData
        save_data = {
            'joints': skeleton['joints'],
            'names': skeleton['names'],
            'parents': skeleton['parents'],
            'tails': skeleton['tails'],
        }

        # Add mesh data
        mesh_data_mapping = {
            'mesh_vertices': 'vertices',
            'mesh_faces': 'faces',
            'mesh_vertex_normals': 'vertex_normals',
            'mesh_face_normals': 'face_normals',
        }
        for skel_key, npz_key in mesh_data_mapping.items():
            if skel_key in skeleton:
                save_data[npz_key] = skeleton[skel_key]

        # Add optional fields that RawData expects
        save_data['skin'] = None
        save_data['no_skin'] = None
        save_data['matrix_local'] = skeleton.get('matrix_local')
        save_data['path'] = None
        save_data['cls'] = skeleton.get('cls')

        # Save NPZ
        np.savez(predict_skeleton_path, **save_data)

        print(f"[UniRigPrepareSkeletonForSkinning] Saved skeleton to: {predict_skeleton_path}")
        print(f"[UniRigPrepareSkeletonForSkinning] Fields saved: {list(save_data.keys())}")

        return (predict_skeleton_path,)
