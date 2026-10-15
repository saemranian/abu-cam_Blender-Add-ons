"""
AbuCam Blender Add-on
---------------------
Native Blender integration for orbital camera rigs, multi-pass rendering
(RGB, 16-bit Depth, Normals), COLMAP dataset generation, and GLB export.
Compatible with Blender 4.2+, 5.1, 5.2 LTS.
"""

bl_info = {
    "name": "AbuCam",
    "author": "Saemranian / AbuCam Lab",
    "version": (0, 0, 1, 9),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AbuCam",
    "description": "Orbital & Volumetric Camera Rig, Environment Sky Sphere, Multi-Pass Renderer & COLMAP Dataset Exporter",
    "doc_url": "https://github.com/saemranian/abu-cam_Blender-Add-ons",
    "tracker_url": "https://github.com/saemranian/abu-cam_Blender-Add-ons/issues",
    "category": "Render",
}

ABUCAM_VERSION = "v0.0.1.9"

import bpy
import bmesh
from bpy.props import FloatProperty, IntProperty, BoolProperty, StringProperty, EnumProperty, PointerProperty, FloatVectorProperty
from mathutils import Vector, Matrix
import math
import time
import numpy as np
import os
import sys
from datetime import datetime
import json
import csv
import struct
import subprocess
import shutil

# -------------------------------------------------------------------------
# Logging System & Diagnostics
# -------------------------------------------------------------------------

def get_abucam_log_dir(scene=None):
    """Returns the persistent diagnostic logs folder for AbuCam."""
    log_dir = None
    if scene and hasattr(scene, 'abucam_props'):
        props = scene.abucam_props
        if props.output_dir:
            try:
                base = bpy.path.abspath(props.output_dir)
                log_dir = os.path.join(base, "logs")
            except Exception:
                pass
    if not log_dir:
        log_dir = os.path.expandvars(r"%APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default\abucam_addon\logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    return log_dir


def abucam_log(msg, level="INFO", scene=None):
    """
    Timestamped diagnostic logger that outputs to stdout and persists to abucam.log on disk.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    entry = f"[{timestamp}] [{level.upper()}] {msg}"
    print(f"[AbuCam] {entry}")
    try:
        log_dir = get_abucam_log_dir(scene)
        log_file = os.path.join(log_dir, "abucam.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass


def format_duration(seconds):
    """Formats seconds into human-readable HH:MM:SS or MM:SS."""
    if seconds is None or seconds <= 0:
        return "00:00"
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_unique_colmap_export_dir(base_dir):
    """
    Determines an isolated destination folder for COLMAP dataset exports so that
    existing files and folders from previous export runs are never overwritten or replaced.
    If 'colmap' or 'sparse' already exists in base_dir, it automatically creates
    and returns a dedicated subfolder (e.g. colmap_001, colmap_002, etc.).
    """
    colmap_dir = os.path.join(base_dir, "colmap")
    sparse_dir = os.path.join(base_dir, "sparse")
    meta_dir = os.path.join(base_dir, "metadata")
    
    # If base directory has never had a COLMAP export, use colmap/
    has_existing = os.path.exists(colmap_dir) or os.path.exists(sparse_dir) or os.path.exists(meta_dir)
    if not has_existing:
        return colmap_dir

    # Find next available non-overlapping folder: colmap_001, colmap_002, ...
    idx = 1
    while True:
        candidate = os.path.join(base_dir, f"colmap_{idx:03d}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


# -------------------------------------------------------------------------
# Default Parameter Mapping for Single & Global Resets
# -------------------------------------------------------------------------

PARAM_DEFAULTS = {
    "azimuth_deg": 0.0,
    "elevation_deg": math.radians(60.0),
    "path_tilt_deg": 0.0,
    "orbit_mode": 'FULL_360',
    "arc_start_deg": math.radians(-90.0),
    "arc_end_deg": math.radians(90.0),
    "show_path_guide": True,
    "orbit_radius": 2.5,
    "active_track_target": True,
    "auto_track_target": True,
    "focal_length": 50.0,
    "total_frames": 81,
    "fps": 24,
    "resolution_x": 1080,
    "resolution_y": 1920,
    "live_update": True,
    "use_ambient_light": True,
    "ambient_intensity": 1.0,
    "ambient_color": (0.92, 0.94, 1.0, 1.0),
    "key_intensity": 45.0,
    "fill_intensity": 18.0,
    "rim_intensity": 32.0,
    # Environment Sky Sphere Defaults
    "use_env_sphere": False,
    "env_sphere_radius": 25.0,
    "env_sphere_mode": 'VORONOI',
    "env_sphere_voronoi_output": 'COLOR',
    "env_sphere_voronoi_scale": 25.0,
    "env_sphere_voronoi_detail": 0.0,
    "env_sphere_voronoi_randomness": 1.0,
    "env_sphere_projection": 'EQUIRECTANGULAR',
    "env_sphere_texture_path": "",
    "env_sphere_emission": 1.0,
    "env_sphere_roughness": 1.0,
    "env_sphere_color": (0.85, 0.90, 1.0, 1.0),
    "env_sphere_zenith_color": (0.15, 0.28, 0.58, 1.0),
    "env_sphere_ground_color": (0.12, 0.13, 0.16, 1.0),
    "env_sphere_sun_elevation": math.radians(35.0),
    "env_sphere_rotation_z": 0.0,
    "env_sphere_scale_u": 1.0,
    "env_sphere_scale_v": 1.0,
    "env_sphere_segments": 64,
    "env_sphere_follow_target": True,
    "env_sphere_visible_camera": True,
    "env_sphere_cast_shadows": False,
    "env_sphere_visible_diffuse": True,
    "env_sphere_visible_glossy": True,
    "env_sphere_exclude_aux_passes": True,
    "renderer": 'BLENDER_EEVEE',
    "eevee_raytracing": False,
    "samples": 32,
    "pass_rgb": True,
    "pass_depth": True,
    "pass_normal": True,
    "pass_wire_rgb": True,
    "wire_color_mode": 'ORANGE',
    "wire_thickness": 1.0,
    "wire_opacity": 0.85,
    "auto_compile_videos": True,
    "is_rendering": False,
    "is_paused": False,
    "show_render_preview_window": True,
    "cancel_render": False,
    "render_status_msg": "",
    "render_progress": 0.0,
    # Volumetric Rig Defaults
    "rig_mode": 'ORBITAL',
    "vol_camera_count": 32,
    "vol_sphere_radius": 2.5,
    "vol_focal_length": 50.0,
    "vol_sensor_width": 36.0,
    "vol_sensor_height": 24.0,
    "vol_distribution": 'FIBONACCI',
    "vol_azimuth_span_deg": math.radians(360.0),
    "vol_elevation_min_deg": math.radians(15.0),
    "vol_elevation_max_deg": math.radians(90.0),
    "vol_azimuth_center_deg": 0.0,
    "vol_show_guide_dome": True,
    "vol_show_camera_lines": True,
    "vol_show_center_lines": False,
    "vol_active_cam_index": 1,
    "frustum_display_size": 0.25,
}

# -------------------------------------------------------------------------
# Helper Functions: RealityCapture / RealityScan Orientation & GPU Acceleration
# -------------------------------------------------------------------------

def extract_reality_capture_ypr(c2w):
    """
    Extracts (yaw, pitch, roll) in degrees matching RealityCapture / RealityScan
    EulerFormat="zyx" convention:
    - Yaw: heading in horizontal plane from North (+Y), clockwise (East = +90, West = -90)
    - Pitch: angle from nadir (-Z), where 0 deg = straight down, 90 deg = horizontal, 180 deg = straight up
    - Roll: tilt/roll around the optical axis (0 deg = upright)
    """
    fx = -float(c2w[0][2])
    fy = -float(c2w[1][2])
    fz = -float(c2w[2][2])
    
    norm = math.sqrt(fx*fx + fy*fy + fz*fz)
    if norm > 1e-8:
        fx /= norm
        fy /= norm
        fz /= norm
    
    horiz = math.sqrt(fx*fx + fy*fy)
    yaw = math.degrees(math.atan2(fx, fy))
    pitch = math.degrees(math.atan2(horiz, -fz))
    
    rx = float(c2w[0][0])
    ry = float(c2w[1][0])
    rz = float(c2w[2][0])
    if horiz > 1e-6:
        zero_rx = fy / horiz
        zero_ry = -fx / horiz
        zero_rz = 0.0
        
        zero_ux = zero_ry * fz - zero_rz * fy
        zero_uy = zero_rz * fx - zero_rx * fz
        zero_uz = zero_rx * fy - zero_ry * fx
        
        cos_roll = rx * zero_rx + ry * zero_ry + rz * zero_rz
        sin_roll = rx * zero_ux + ry * zero_uy + rz * zero_uz
        roll = math.degrees(math.atan2(sin_roll, cos_roll))
    else:
        roll = 0.0
        
    return yaw, pitch, roll


REALITY_CAPTURE_CSV_HEADER = "#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"


def format_reality_capture_csv_line(name, c2w, focal_mm, sensor_width=36.0):
    """
    Formats a single camera pose row matching RealityCapture / RealityScan schema:
    #name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2
    """
    x = float(c2w[0][3])
    y = float(c2w[1][3])
    alt = float(c2w[2][3])
    yaw, pitch, roll = extract_reality_capture_ypr(c2w)
    f_35mm = float(focal_mm) * (36.0 / max(0.1, float(sensor_width)))
    return f"{name},{x:.8f},{y:.8f},{alt:.8f},{yaw:.8f},{pitch:.8f},{roll:.8f},{f_35mm:.4f},0,0,0,0,0,0,0,0\n"


def configure_gpu_accelerated_render(scene, props, pass_type='RGB'):
    """
    Configures Blender for light-speed GPU-accelerated rendering matching Unreal Engine fidelity:
    - EEVEE (Next): GPU rasterization, screen raytracing, temporal reprojection, instant passes.
    - Cycles: OptiX raytracing on NVIDIA RTX GPU, GPU Tensor core AI denoising, persistent VRAM caching,
      adaptive sampling.
    """
    target_engine = props.renderer
    avail_engines = [it.identifier for it in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items]
    if target_engine not in avail_engines:
        if target_engine in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
            if 'BLENDER_EEVEE' in avail_engines:
                target_engine = 'BLENDER_EEVEE'
            elif 'BLENDER_EEVEE_NEXT' in avail_engines:
                target_engine = 'BLENDER_EEVEE_NEXT'
        elif 'CYCLES' in avail_engines:
            target_engine = 'CYCLES'
    
    if scene.render.engine != target_engine:
        try:
            scene.render.engine = target_engine
        except Exception:
            scene.render.engine = 'CYCLES'

    # Enable persistent data so geometry, BVH, and textures stay resident in GPU VRAM across all frames
    try:
        scene.render.use_persistent_data = True
    except Exception:
        pass

    if scene.render.engine == 'CYCLES':
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            if prefs.compute_device_type != 'OPTIX':
                prefs.compute_device_type = 'OPTIX'
                prefs.get_devices()
                has_optix = False
                for d in prefs.devices:
                    if d.type == 'OPTIX':
                        d.use = True
                        has_optix = True
                    else:
                        d.use = False
                if not has_optix:
                    prefs.compute_device_type = 'CUDA'
                    prefs.get_devices()
                    for d in prefs.devices:
                        if d.type == 'CUDA':
                            d.use = True
                        else:
                            d.use = False

            scene.cycles.device = 'GPU'
            
            # Fast GPU OptiX Denoiser using Tensor Cores
            scene.cycles.use_denoising = True
            if hasattr(scene.cycles, 'denoiser'):
                scene.cycles.denoiser = 'OPTIX'
            if hasattr(scene.cycles, 'denoising_input_passes'):
                scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'

            # Adaptive sampling for light-speed convergence
            if hasattr(scene.cycles, 'use_adaptive_sampling'):
                scene.cycles.use_adaptive_sampling = True
                scene.cycles.adaptive_threshold = 0.03
                scene.cycles.adaptive_min_samples = 4

            # Analytical passes (depth/normal/wire mask) require only 4 samples without denoising
            if pass_type in ('DEPTH', 'NORMAL', 'WIRE'):
                scene.cycles.samples = 4
                scene.cycles.use_denoising = False
            else:
                scene.cycles.samples = max(4, props.samples)
                scene.cycles.use_denoising = True
        except Exception as e:
            print(f"[AbuCam] Cycles GPU setup warning: {e}")

    elif scene.render.engine in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
        try:
            if hasattr(scene, 'eevee'):
                ev = scene.eevee
                # Disable screen raytracing during high-speed multi-pass baking unless requested for beauty RGB
                use_rt = getattr(props, 'eevee_raytracing', False) and (pass_type == 'RGB')
                if hasattr(ev, 'use_raytracing'):
                    ev.use_raytracing = use_rt
                if hasattr(ev, 'use_taa_reprojection'):
                    ev.use_taa_reprojection = True
                if hasattr(ev, 'use_fast_gi'):
                    ev.use_fast_gi = True
                if hasattr(ev, 'use_shadows'):
                    ev.use_shadows = True
                
                # Analytical passes need only 1 sample in EEVEE for maximum speed
                if pass_type in ('DEPTH', 'NORMAL', 'WIRE'):
                    if hasattr(ev, 'taa_render_samples'):
                        ev.taa_render_samples = 1
                else:
                    if hasattr(ev, 'taa_render_samples'):
                        ev.taa_render_samples = max(1, props.samples)
        except Exception as e:
            print(f"[AbuCam] EEVEE GPU setup warning: {e}")


# -------------------------------------------------------------------------
# Helper Functions: Pass Materials & Lighting
# -------------------------------------------------------------------------

def get_or_create_depth_material(near=2.0, far=6.0):
    mat = bpy.data.materials.get("__AbuCam_DepthMat__")
    if not mat:
        mat = bpy.data.materials.new(name="__AbuCam_DepthMat__")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    cam_data = nodes.new(type='ShaderNodeCameraData')
    cam_data.location = (-400, 0)

    map_range = nodes.new(type='ShaderNodeMapRange')
    map_range.location = (-180, 0)
    map_range.inputs['From Min'].default_value = float(near)
    map_range.inputs['From Max'].default_value = float(far)
    map_range.inputs['To Min'].default_value = 1.0
    map_range.inputs['To Max'].default_value = 0.0

    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (60, 0)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (260, 0)

    links.new(cam_data.outputs['View Z Depth'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def get_or_create_normal_material():
    mat = bpy.data.materials.get("__AbuCam_NormalMat__")
    if not mat:
        mat = bpy.data.materials.new(name="__AbuCam_NormalMat__")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    geom = nodes.new(type='ShaderNodeNewGeometry')
    geom.location = (-400, 0)

    math_mult = nodes.new(type='ShaderNodeVectorMath')
    math_mult.operation = 'MULTIPLY'
    math_mult.inputs[1].default_value = (0.5, 0.5, 0.5)
    math_mult.location = (-180, 50)

    math_add = nodes.new(type='ShaderNodeVectorMath')
    math_add.operation = 'ADD'
    math_add.inputs[1].default_value = (0.5, 0.5, 0.5)
    math_add.location = (40, 50)

    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (260, 0)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (460, 0)

    links.new(geom.outputs['Normal'], math_mult.inputs[0])
    links.new(math_mult.outputs['Vector'], math_add.inputs[0])
    links.new(math_add.outputs['Vector'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


WIRE_COLORS = {
    'ORANGE': (1.0, 0.55, 0.05),     # Viewport Orange
    'DARK':   (0.05, 0.05, 0.05),     # Dark / Black
    'WHITE':  (0.95, 0.95, 0.95),     # White
    'CYAN':   (0.0, 0.88, 1.0),       # Cyan
}


def get_or_create_wireframe_material(thickness=1.0):
    """
    Creates an occlusion-safe wireframe mask material.
    Polygon faces render pure black (0.0) occluding interior/back faces,
    while polygon edges render pure emission white (1.0).
    """
    mat = bpy.data.materials.get("__AbuCam_WireMat__")
    if not mat:
        mat = bpy.data.materials.new(name="__AbuCam_WireMat__")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    wire = nodes.new(type='ShaderNodeWireframe')
    wire.use_pixel_size = True
    wire.inputs['Size'].default_value = float(max(0.1, thickness))
    wire.location = (-300, 0)

    emit = nodes.new(type='ShaderNodeEmission')
    emit.location = (-50, 0)

    out = nodes.new(type='ShaderNodeOutputMaterial')
    out.location = (200, 0)

    links.new(wire.outputs['Fac'], emit.inputs['Color'])
    links.new(emit.outputs['Emission'], out.inputs['Surface'])
    return mat


def composite_wireframe_on_rgb(rgb_path, wire_mask_path, out_path, color_mode='ORANGE', opacity=0.85):
    """
    Composites a wireframe mask over an RGB beauty pass with customizable line color and opacity.
    Uses numpy and Blender's internal image API without external dependencies.
    """
    if not os.path.exists(rgb_path) or not os.path.exists(wire_mask_path):
        return False
    try:
        img_rgb = bpy.data.images.load(rgb_path, check_existing=False)
        img_wire = bpy.data.images.load(wire_mask_path, check_existing=False)

        w, h = img_rgb.size
        rgb_arr = np.empty(w * h * 4, dtype=np.float32)
        wire_arr = np.empty(w * h * 4, dtype=np.float32)

        img_rgb.pixels.foreach_get(rgb_arr)
        img_wire.pixels.foreach_get(wire_arr)

        rgb_arr = rgb_arr.reshape((h, w, 4))
        wire_arr = wire_arr.reshape((h, w, 4))

        # Wire mask factor (first channel)
        wire_fac = wire_arr[:, :, 0:1]

        c_tuple = WIRE_COLORS.get(color_mode, (1.0, 0.55, 0.05))
        c_rgb = np.array(c_tuple, dtype=np.float32)
        op = float(np.clip(opacity, 0.0, 1.0))

        res_arr = rgb_arr.copy()
        # Blend RGB with wire color:
        # Out = RGB * (1 - wire * op) + WireColor * (wire * op)
        blend_factor = wire_fac * op
        res_arr[:, :, 0:3] = rgb_arr[:, :, 0:3] * (1.0 - blend_factor) + c_rgb * blend_factor
        res_arr[:, :, 3] = 1.0

        out_img = bpy.data.images.new("__AbuCam_WireCompTmp__", width=w, height=h)
        out_img.pixels.foreach_set(res_arr.ravel())
        out_img.filepath_raw = out_path
        out_img.file_format = 'PNG'
        out_img.save()

        bpy.data.images.remove(img_rgb)
        bpy.data.images.remove(img_wire)
        bpy.data.images.remove(out_img)
        return True
    except Exception as e:
        print(f"[AbuCam] Wireframe composite error: {e}")
        return False


# -------------------------------------------------------------------------
# Dynamic Rig & Depsgraph Live Tracking
# -------------------------------------------------------------------------

_is_updating_rig = False
_last_target_matrix = None
_last_target_center = None


@bpy.app.handlers.persistent
def abucam_depsgraph_handler(scene, depsgraph):
    """Live tracking handler: when target object moves in 3D viewport, update orbit center."""
    global _is_updating_rig, _last_target_matrix, _last_target_center
    if _is_updating_rig:
        return
    props = getattr(scene, 'abucam_props', None)
    if not props or getattr(props, 'is_rendering', False) or not getattr(props, 'live_update', True) or not getattr(props, 'auto_track_target', True):
        return
    target_obj = getattr(props, 'target_object', None)
    if not target_obj:
        return

    try:
        cur_mat = target_obj.matrix_world.copy()
    except Exception:
        return

    if _last_target_matrix is not None and cur_mat == _last_target_matrix:
        return

    _last_target_matrix = cur_mat

    try:
        bbox = [cur_mat @ Vector(corner) for corner in target_obj.bound_box]
        cur_center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
    except Exception:
        return

    if _last_target_center is not None:
        delta = cur_center - _last_target_center
        if delta.length > 0.0001:
            _is_updating_rig = True
            try:
                props.target_x += delta.x
                props.target_y += delta.y
                props.target_z += delta.z
                if getattr(props, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
                    update_volumetric_rig(scene)
                else:
                    update_orbital_rig(scene, bake_keyframes=False)
            finally:
                _is_updating_rig = False

    _last_target_center = cur_center


# -------------------------------------------------------------------------
# Dynamic Rig & Viewport Synchronization
# -------------------------------------------------------------------------

def update_orbital_path_guide(scene, props, points, is_cyclic=True):
    """Creates or updates a non-rendering visual 3D guide curve of the camera path."""
    guide_name = "OrbitalPathGuide"
    curve_data_name = "OrbitalPathGuideData"

    if not getattr(props, 'show_path_guide', True):
        old = bpy.data.objects.get(guide_name)
        if old:
            old.hide_viewport = True
        return

    curve_data = bpy.data.curves.get(curve_data_name)
    if not curve_data:
        curve_data = bpy.data.curves.new(curve_data_name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = 0.006
        curve_data.bevel_resolution = 3
        curve_data.use_fill_caps = True

    mat_name = "AbuCam_PathGuide_Mat"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emit = nodes.new(type='ShaderNodeEmission')
        emit.inputs['Color'].default_value = (0.0, 0.85, 1.0, 1.0)
        emit.inputs['Strength'].default_value = 1.8
        out = nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])

    if mat.name not in curve_data.materials:
        curve_data.materials.append(mat)

    curve_data.splines.clear()
    spline = curve_data.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for idx, pt in enumerate(points):
        spline.points[idx].co = (pt[0], pt[1], pt[2], 1.0)
    spline.use_cyclic_u = is_cyclic

    obj = bpy.data.objects.get(guide_name)
    if not obj:
        obj = bpy.data.objects.new(guide_name, curve_data)
        scene.collection.objects.link(obj)
    else:
        obj.data = curve_data
        obj.hide_viewport = False

    obj.hide_render = True
    obj.hide_select = True


def update_orbital_rig(scene, bake_keyframes=False):
    """
    Dynamically positions and configures the OrbitalCamera in real time.
    Calculates 3D spherical coordinates matching AbuCam standards:
    - Azimuth (0-360°): 0° Front, 90° Right, 180° Back, 270° Left.
    - Elevation (0-180°): 0° Top/Zenith, 60° High Angle, 90° Eye Level, 180° Bottom/Nadir.
    - Path Tilt (-90° to +90°): Rotates the circular path plane (0° Horizontal, ±90° Vertical loop).
    Forces active viewport redraw for instant visual feedback without baking keyframes.
    """
    props = getattr(scene, 'abucam_props', None)
    if not props:
        return None

    cam_obj = bpy.data.objects.get("OrbitalCamera")
    if not cam_obj:
        cam_data = bpy.data.cameras.new("OrbitalCamera")
        cam_obj = bpy.data.objects.new("OrbitalCamera", cam_data)
        scene.collection.objects.link(cam_obj)

    scene.camera = cam_obj
    cam = cam_obj.data
    cam.lens = props.focal_length
    cam.sensor_fit = 'AUTO'
    cam.sensor_width = 36.0
    cam.sensor_height = 24.0
    cam.display_size = max(0.005, float(getattr(props, 'frustum_display_size', 0.25)))
    cam.clip_start = 0.05
    cam.clip_end = max(500.0, getattr(props, 'env_sphere_radius', 25.0) * 2.5)

    scene.render.resolution_x = props.resolution_x
    scene.render.resolution_y = props.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.frame_start = 1
    scene.frame_end = max(1, props.total_frames)
    scene.render.fps = max(1, props.fps)

    radius = props.orbit_radius
    az_rad = props.azimuth_deg      # Stored in radians via unit='ROTATION'
    elev_rad = getattr(props, 'elevation_deg', math.radians(90.0))  # 0° Zenith, 60° High Angle, 90° Eye Level, 180° Nadir
    tilt_rad = getattr(props, 'path_tilt_deg', 0.0)  # Rotates path from -90° to +90°
    target_x = getattr(props, 'target_x', 0.0)
    target_y = getattr(props, 'target_y', 0.0)
    target_z = props.target_z
    orbit_center = Vector((target_x, target_y, target_z))

    # Sync Environment Sky Sphere center if follow_target is enabled
    env_obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
    if env_obj and getattr(props, 'env_sphere_follow_target', True):
        env_obj.location = orbit_center

    # Active Track Target in Frame Center:
    # When active_track_target is True and a target_object is selected,
    # the camera aims directly at the geometric center of the target layer mesh,
    # ensuring the subject remains locked dead-center in frame even when
    # Orbit Width (X), Depth (Y), and Height (Z) move the camera orbit ring around.
    active_track = getattr(props, 'active_track_target', True)
    target_mesh = getattr(props, 'target_object', None)
    if active_track and target_mesh:
        try:
            bbox = [target_mesh.matrix_world @ Vector(corner) for corner in target_mesh.bound_box]
            aim_pos = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
        except Exception:
            aim_pos = orbit_center
    else:
        aim_pos = orbit_center

    # Maintain visual target crosshair empty at exact focus center
    target_name = "AbuCam_Target"
    target_empty = bpy.data.objects.get(target_name)
    if not target_empty:
        target_empty = bpy.data.objects.new(target_name, None)
        target_empty.empty_display_type = 'PLAIN_AXES'
        target_empty.empty_display_size = 0.15
        scene.collection.objects.link(target_empty)
    target_empty.location = aim_pos
    target_empty.hide_render = True

    # Ensure Track To constraint on OrbitalCamera to lock target in frame center
    track_c = None
    for c in cam_obj.constraints:
        if c.type == 'TRACK_TO':
            track_c = c
            break
    if not track_c:
        track_c = cam_obj.constraints.new(type='TRACK_TO')
        track_c.track_axis = 'TRACK_NEGATIVE_Z'
        track_c.up_axis = 'UP_Y'
    track_c.target = target_empty

    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)
    cos_e = math.cos(elev_rad)
    sin_e = math.sin(elev_rad)

    def calc_pos(ang):
        # Spherical orbital coordinates with optional path tilt:
        # - elev_rad (0-180°): 0° Zenith (top-down), 60° High Angle, 90° Eye Level (flat), 180° Nadir (bottom-up)
        # - ang (0-360°): Azimuth orbital angle around target
        # - tilt_rad (-90° to +90°): Tilt of the circular path plane
        # - radius: Orbit distance from target / circle radius
        lx = radius * sin_e * math.sin(ang)
        ly = -radius * sin_e * math.cos(ang)
        lz = radius * cos_e
        # Rotate by tilt_rad around local X axis
        tx = lx
        ty = ly * cos_t - lz * sin_t
        tz = ly * sin_t + lz * cos_t
        # Translate to 3D orbit center (X width, Y depth, Z height)
        world_x = tx + target_x
        world_y = ty + target_y
        world_z = tz + target_z
        return Vector((world_x, world_y, world_z))

    cam_pos = calc_pos(az_rad)
    cam_obj.location = cam_pos
    direction = aim_pos - cam_pos
    rot_quat = None
    if direction.length > 0.0001:
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

    # Determine orbit mode & range
    mode = getattr(props, 'orbit_mode', 'FULL_360')
    if mode == 'ARC_180':
        start_ang = math.radians(-90.0)
        end_ang = math.radians(90.0)
        is_cyclic = False
    elif mode == 'CUSTOM':
        start_ang = getattr(props, 'arc_start_deg', math.radians(-90.0))
        end_ang = getattr(props, 'arc_end_deg', math.radians(90.0))
        is_cyclic = False
    else:  # FULL_360
        start_ang = 0.0
        end_ang = 2.0 * math.pi
        is_cyclic = True

    # If keyframe animation exists or explicit bake requested, update the full orbit keyframes
    has_anim = bool(cam_obj.animation_data and cam_obj.animation_data.action)
    if bake_keyframes or has_anim:
        cam_obj.animation_data_clear()
        total_frames = max(1, props.total_frames)
        for i in range(total_frames):
            frame_num = i + 1
            if is_cyclic:
                ang = start_ang + (2.0 * math.pi * i) / total_frames
            else:
                t = i / max(1, total_frames - 1)
                ang = start_ang + t * (end_ang - start_ang)

            pos = calc_pos(ang)
            cam_obj.location = pos
            kdir = aim_pos - pos
            if kdir.length > 0.0001:
                krot = kdir.to_track_quat('-Z', 'Y')
                cam_obj.rotation_euler = krot.to_euler()
            cam_obj.keyframe_insert(data_path="location", frame=frame_num)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame_num)

        # Sync timeline frame to current azimuth angle
        if is_cyclic:
            deg = math.degrees(az_rad) % 360.0
            target_frame = 1 + int(round((deg / 360.0) * max(1, props.total_frames - 1)))
        else:
            s_deg = math.degrees(start_ang)
            e_deg = math.degrees(end_ang)
            cur_deg = math.degrees(az_rad)
            if cur_deg > 180.0:
                cur_deg -= 360.0
            t_norm = (cur_deg - s_deg) / max(0.0001, (e_deg - s_deg))
            t_norm = max(0.0, min(1.0, t_norm))
            target_frame = 1 + int(round(t_norm * max(1, props.total_frames - 1)))

        target_frame = max(1, min(props.total_frames, target_frame))
        if scene.frame_current != target_frame:
            scene.frame_set(target_frame)
        else:
            cam_obj.location = cam_pos
            if rot_quat:
                cam_obj.rotation_euler = rot_quat.to_euler()

    # Generate points for 3D visual guide curve
    sample_count = 64 if is_cyclic else 48
    guide_pts = []
    for s in range(sample_count):
        if is_cyclic:
            a = (2.0 * math.pi * s) / sample_count
        else:
            t = s / max(1, sample_count - 1)
            a = start_ang + t * (end_ang - start_ang)
        guide_pts.append(calc_pos(a))
    update_orbital_path_guide(scene, props, guide_pts, is_cyclic=is_cyclic)

    # Redraw all 3D viewports so user sees live update immediately
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    return cam_obj


# -------------------------------------------------------------------------
# Volumetric Camera Rig Generator & Guide System
# -------------------------------------------------------------------------

def calculate_volumetric_camera_positions(props):
    """
    Calculates 3D positions of N cameras distributed across the spherical surface
    section of radius R, centered at (target_x, target_y, target_z), bounded by
    the 3 angular parameters:
    1. Azimuth Span (0 to 360 deg) centered at Azimuth Heading
    2. Elevation Min (Upper zenith cutoff angle from top pole)
    3. Elevation Max (Lower floor cutoff angle from top pole)
    """
    N = max(1, getattr(props, 'vol_camera_count', 32))
    R = max(0.1, getattr(props, 'vol_sphere_radius', 2.5))
    target_x = getattr(props, 'target_x', 0.0)
    target_y = getattr(props, 'target_y', 0.0)
    target_z = getattr(props, 'target_z', 0.0)
    center = Vector((target_x, target_y, target_z))

    az_span = max(math.radians(1.0), getattr(props, 'vol_azimuth_span_deg', math.radians(360.0)))
    az_center = getattr(props, 'vol_azimuth_center_deg', 0.0)
    az_start = az_center - az_span / 2.0

    el_min = getattr(props, 'vol_elevation_min_deg', math.radians(15.0))
    el_max = getattr(props, 'vol_elevation_max_deg', math.radians(90.0))
    if el_min > el_max:
        el_min, el_max = el_max, el_min
    if el_max - el_min < math.radians(1.0):
        el_max = el_min + math.radians(1.0)

    dist_mode = getattr(props, 'vol_distribution', 'FIBONACCI')
    positions = []

    # Bounding cos(theta) range
    z_top = math.cos(el_min)    # Larger cos value (higher z)
    z_bottom = math.cos(el_max) # Smaller cos value (lower z)

    if dist_mode == 'FIBONACCI':
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~2.39996323 rad (137.508 deg)
        for i in range(N):
            t = (i + 0.5) / float(N)
            z_unit = z_top - t * (z_top - z_bottom)
            z_unit = max(-1.0, min(1.0, z_unit))
            theta = math.acos(z_unit)

            if abs(az_span - 2.0 * math.pi) < 0.01:
                phi = (i * golden_angle) % (2.0 * math.pi)
            else:
                phi_norm = ((i * golden_angle) % (2.0 * math.pi)) / (2.0 * math.pi)
                phi = az_start + phi_norm * az_span

            sin_theta = math.sin(theta)
            lx = R * sin_theta * math.sin(phi)
            ly = -R * sin_theta * math.cos(phi)
            lz = R * z_unit
            positions.append(center + Vector((lx, ly, lz)))

    elif dist_mode == 'RINGS':
        # Concentric Latitude Rings
        num_rings = max(1, int(round(math.sqrt(N * (el_max - el_min) / max(0.01, az_span)))) or 1)
        ring_thetas = [el_min + (r + 0.5) / num_rings * (el_max - el_min) for r in range(num_rings)]
        weights = [max(0.1, math.sin(th)) for th in ring_thetas]
        total_w = sum(weights)
        cams_per_ring = [max(1, int(round(N * (w / total_w)))) for w in weights]
        diff = N - sum(cams_per_ring)
        if diff != 0:
            cams_per_ring[len(cams_per_ring) // 2] += diff

        for r_idx, theta in enumerate(ring_thetas):
            count_in_ring = max(1, cams_per_ring[r_idx])
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            stagger = (az_span / count_in_ring) * 0.5 if (r_idx % 2 == 1 and count_in_ring > 1) else 0.0
            for c in range(count_in_ring):
                if count_in_ring == 1:
                    phi = az_center
                elif abs(az_span - 2.0 * math.pi) < 0.01:
                    phi = az_start + (c / float(count_in_ring)) * az_span + stagger
                else:
                    phi = az_start + (c / float(count_in_ring - 1)) * az_span
                lx = R * sin_theta * math.sin(phi)
                ly = -R * sin_theta * math.cos(phi)
                lz = R * cos_theta
                positions.append(center + Vector((lx, ly, lz)))
                if len(positions) >= N:
                    break
            if len(positions) >= N:
                break
        while len(positions) < N:
            positions.append(positions[-1].copy())

    else:  # RANDOM
        import random
        rng = random.Random(42 + N)
        for i in range(N):
            z_unit = rng.uniform(z_bottom, z_top)
            theta = math.acos(max(-1.0, min(1.0, z_unit)))
            phi = rng.uniform(az_start, az_start + az_span)
            sin_theta = math.sin(theta)
            lx = R * sin_theta * math.sin(phi)
            ly = -R * sin_theta * math.cos(phi)
            lz = R * z_unit
            positions.append(center + Vector((lx, ly, lz)))

    return positions


def set_viewport_relationship_lines(visible):
    """Synchronizes 3D Viewport overlay relationship lines (dashed constraint lines to target)."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D' and hasattr(space, 'overlay'):
                            space.overlay.show_relationship_lines = bool(visible)
                    area.tag_redraw()
    except Exception:
        pass


def update_volumetric_guide(scene, props, camera_positions, aim_pos):
    """Creates or updates a 3D visual guide mesh of the volumetric dome and camera positions."""
    guide_name = "AbuCam_VolumetricGuide"
    mesh_data_name = "AbuCam_VolumetricGuideMesh"

    show_master = getattr(props, 'vol_show_guide_dome', True)
    show_cam_lines = show_master and getattr(props, 'vol_show_camera_lines', True)
    show_ctr_lines = show_master and getattr(props, 'vol_show_center_lines', False)

    set_viewport_relationship_lines(show_ctr_lines)

    if not show_master or (not show_cam_lines and not show_ctr_lines):
        old = bpy.data.objects.get(guide_name)
        if old:
            old.hide_viewport = True
        return

    mesh = bpy.data.meshes.get(mesh_data_name)
    if not mesh:
        mesh = bpy.data.meshes.new(mesh_data_name)

    verts = [aim_pos]
    edges = []
    for i, cp in enumerate(camera_positions):
        verts.append(cp)
        if show_ctr_lines:
            edges.append((0, i + 1))  # Ray from aim center to camera

    if show_cam_lines:
        for i in range(len(camera_positions) - 1):
            edges.append((i + 1, i + 2))  # Line between consecutive cameras

    mesh.clear_geometry()
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    mat_name = "AbuCam_VolGuide_Mat"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emit = nodes.new(type='ShaderNodeEmission')
        emit.inputs['Color'].default_value = (0.2, 0.9, 0.4, 1.0)
        emit.inputs['Strength'].default_value = 2.0
        out = nodes.new(type='ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])

    if mat.name not in mesh.materials:
        mesh.materials.append(mat)

    obj = bpy.data.objects.get(guide_name)
    if not obj:
        obj = bpy.data.objects.new(guide_name, mesh)
        scene.collection.objects.link(obj)
    else:
        obj.data = mesh
        obj.hide_viewport = False

    obj.hide_render = True
    obj.hide_select = True


def update_volumetric_rig(scene):
    """
    Creates, positions, and synchronizes the Volumetric Camera Rig in Blender.
    Maintains an array of N physical cameras targeting AbuCam_Target,
    and synchronizes scene resolution (W, H), timeline frames, FPS, and independent
    horizontal (Sensor W) and vertical (Sensor H) camera frustum dimensions in real time.
    """
    props = getattr(scene, 'abucam_props', None)
    if not props:
        return []

    # Synchronize scene resolution (W, H), timeline frame range, and FPS in real time
    scene.render.resolution_x = props.resolution_x
    scene.render.resolution_y = props.resolution_y
    scene.render.resolution_percentage = 100
    scene.frame_start = 1
    scene.frame_end = max(1, props.total_frames)
    scene.render.fps = max(1, props.fps)

    # Independent horizontal (Sensor W) and vertical (Sensor H) sensor frustum sizing:
    # With cam.sensor_fit = 'HORIZONTAL', horizontal frustum width is proportional to vol_sensor_width,
    # and vertical frustum height is proportional to vol_sensor_height * (H / W).
    # At default 36x24mm, pixel_aspect_x = 1.0 and pixel_aspect_y = 1.0.
    sw = max(1.0, float(props.vol_sensor_width))
    sh = max(1.0, float(props.vol_sensor_height))
    sensor_ratio = (sh / sw) * (36.0 / 24.0)
    if sensor_ratio >= 1.0:
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = min(200.0, sensor_ratio)
    else:
        scene.render.pixel_aspect_x = min(200.0, 1.0 / max(0.005, sensor_ratio))
        scene.render.pixel_aspect_y = 1.0

    coll_name = "AbuCam_Volumetric_Rig"
    coll = bpy.data.collections.get(coll_name)
    if not coll:
        coll = bpy.data.collections.new(coll_name)
        scene.collection.children.link(coll)
    coll.hide_viewport = False

    target_x = getattr(props, 'target_x', 0.0)
    target_y = getattr(props, 'target_y', 0.0)
    target_z = props.target_z
    orbit_center = Vector((target_x, target_y, target_z))

    # Sync Environment Sky Sphere center if follow_target is enabled
    env_obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
    if env_obj and getattr(props, 'env_sphere_follow_target', True):
        env_obj.location = orbit_center

    active_track = getattr(props, 'active_track_target', True)
    target_mesh = getattr(props, 'target_object', None)
    if active_track and target_mesh:
        try:
            bbox = [target_mesh.matrix_world @ Vector(corner) for corner in target_mesh.bound_box]
            aim_pos = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
        except Exception:
            aim_pos = orbit_center
    else:
        aim_pos = orbit_center

    target_name = "AbuCam_Target"
    target_empty = bpy.data.objects.get(target_name)
    if not target_empty:
        target_empty = bpy.data.objects.new(target_name, None)
        target_empty.empty_display_type = 'PLAIN_AXES'
        target_empty.empty_display_size = 0.15
        scene.collection.objects.link(target_empty)
    target_empty.location = aim_pos
    target_empty.hide_render = True

    positions = calculate_volumetric_camera_positions(props)
    N = len(positions)
    far_clip = max(500.0, getattr(props, 'env_sphere_radius', 25.0) * 2.5)

    vol_cameras = []
    for i, pos in enumerate(positions):
        cam_name = f"AbuCam_Vol_{i+1:03d}"
        cam_obj = bpy.data.objects.get(cam_name)
        if not cam_obj:
            cam_data = bpy.data.cameras.new(cam_name)
            cam_obj = bpy.data.objects.new(cam_name, cam_data)
            coll.objects.link(cam_obj)
        else:
            if cam_obj.name not in coll.objects:
                coll.objects.link(cam_obj)
            cam_obj.hide_viewport = False
            cam_obj.hide_render = False

        cam = cam_obj.data
        cam.lens = props.vol_focal_length
        cam.sensor_fit = 'HORIZONTAL'
        cam.sensor_width = sw
        cam.sensor_height = sh
        cam.display_size = max(0.005, float(getattr(props, 'frustum_display_size', 0.25)))
        cam.clip_start = 0.05
        cam.clip_end = far_clip

        cam_obj.location = pos

        track_c = None
        for c in cam_obj.constraints:
            if c.type == 'TRACK_TO':
                track_c = c
                break
        if not track_c:
            track_c = cam_obj.constraints.new(type='TRACK_TO')
            track_c.track_axis = 'TRACK_NEGATIVE_Z'
            track_c.up_axis = 'UP_Y'
        track_c.target = target_empty

        direction = aim_pos - pos
        if direction.length > 0.0001:
            cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

        vol_cameras.append(cam_obj)

    # Remove extra cameras if N was reduced
    for obj in list(coll.objects):
        if obj.name.startswith("AbuCam_Vol_"):
            try:
                num = int(obj.name.split("_")[-1])
                if num > N:
                    cam_data = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    if cam_data and cam_data.users == 0:
                        bpy.data.cameras.remove(cam_data)
            except Exception:
                pass

    update_volumetric_guide(scene, props, positions, aim_pos)

    # Set active view camera
    active_idx = max(1, min(N, getattr(props, 'vol_active_cam_index', 1)))
    if active_idx <= len(vol_cameras):
        scene.camera = vol_cameras[active_idx - 1]
    elif vol_cameras:
        scene.camera = vol_cameras[0]

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    return vol_cameras


def on_camera_param_update(self, context):
    """Callback triggered whenever any camera/rig parameter is modified."""
    if not self.live_update:
        return
    scene = getattr(context, 'scene', None) or bpy.context.scene
    if getattr(self, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
        update_volumetric_rig(scene)
    else:
        update_orbital_rig(scene, bake_keyframes=False)


def on_target_object_update(self, context):
    """Callback triggered whenever target_object is changed."""
    global _last_target_matrix, _last_target_center
    if self.target_object:
        obj = self.target_object
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
        self.target_x = center.x
        self.target_y = center.y
        self.target_z = center.z
        _last_target_matrix = obj.matrix_world.copy()
        _last_target_center = center.copy()
    else:
        _last_target_matrix = None
        _last_target_center = None
    if not self.live_update:
        return
    scene = getattr(context, 'scene', None) or bpy.context.scene
    if getattr(self, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
        update_volumetric_rig(scene)
    else:
        update_orbital_rig(scene, bake_keyframes=False)


def on_rig_mode_update(self, context):
    """Callback triggered when switching between Orbital and Volumetric rig modes."""
    scene = getattr(context, 'scene', None) or bpy.context.scene
    if not scene:
        return
    guide_orb = bpy.data.objects.get("OrbitalPathGuide")
    cam_orb = bpy.data.objects.get("OrbitalCamera")
    vol_coll = bpy.data.collections.get("AbuCam_Volumetric_Rig")
    vol_guide = bpy.data.objects.get("AbuCam_VolumetricGuide")

    if self.rig_mode == 'VOLUMETRIC':
        if guide_orb:
            guide_orb.hide_viewport = True
        if vol_coll:
            vol_coll.hide_viewport = False
        update_volumetric_rig(scene)
    else:
        if vol_guide:
            vol_guide.hide_viewport = True
        if vol_coll:
            vol_coll.hide_viewport = True
        if cam_orb:
            scene.camera = cam_orb
        update_orbital_rig(scene, bake_keyframes=False)


def on_volumetric_param_update(self, context):
    """Callback triggered whenever any volumetric parameter is modified."""
    if not self.live_update or getattr(self, 'rig_mode', 'ORBITAL') != 'VOLUMETRIC':
        return
    scene = getattr(context, 'scene', None) or bpy.context.scene
    update_volumetric_rig(scene)


def on_frustum_size_update(self, context):
    """Fast callback to update 3D Viewport camera frustum display scale across all Orbital & Volumetric cameras."""
    sz = max(0.005, float(getattr(self, 'frustum_display_size', 0.25)))
    for cam_data in bpy.data.cameras:
        if cam_data.name == "OrbitalCamera" or cam_data.name.startswith("AbuCam_Vol_") or cam_data.name.startswith("VolCam_"):
            cam_data.display_size = sz
    scene = getattr(context, 'scene', None) or bpy.context.scene
    if scene:
        for obj in scene.objects:
            if obj.type == 'CAMERA' and (obj.name == "OrbitalCamera" or obj.name.startswith("AbuCam_Vol_") or obj.name.startswith("VolCam_")):
                obj.data.display_size = sz
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def on_guide_lines_update(self, context):
    """Fast callback to immediately toggle 3D guide lines between cameras and center without rebuilding cameras."""
    scene = getattr(context, 'scene', None) or bpy.context.scene
    if not scene:
        return
    target_empty = bpy.data.objects.get("AbuCam_Target")
    if target_empty:
        aim_pos = target_empty.location.copy()
    else:
        aim_pos = Vector((getattr(self, 'target_x', 0.0), getattr(self, 'target_y', 0.0), getattr(self, 'target_z', 0.0)))
    positions = calculate_volumetric_camera_positions(self)
    update_volumetric_guide(scene, self, positions, aim_pos)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def on_active_cam_index_update(self, context):
    """Callback triggered when cycling or selecting active volumetric camera."""
    scene = getattr(context, 'scene', None) or bpy.context.scene
    if not scene:
        return
    idx = self.vol_active_cam_index
    cam_name = f"AbuCam_Vol_{idx:03d}"
    cam_obj = bpy.data.objects.get(cam_name)
    if cam_obj:
        scene.camera = cam_obj
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()


def on_light_param_update(self, context):
    """Callback triggered whenever any studio lighting parameter is modified."""
    scene = getattr(context, 'scene', None) or bpy.context.scene
    props = getattr(scene, 'abucam_props', None)
    if not props:
        return

    radius = max(0.5, props.orbit_radius)
    scale = radius / 0.85

    # Update Key Light
    key = bpy.data.objects.get("KeyLight")
    if key and key.type == 'LIGHT':
        key.data.energy = props.key_intensity * (scale ** 1.5)

    # Update Fill Light
    fill = bpy.data.objects.get("FillLight")
    if fill and fill.type == 'LIGHT':
        fill.data.energy = props.fill_intensity * (scale ** 1.5)

    # Update Rim Light
    rim = bpy.data.objects.get("RimLight")
    if rim and rim.type == 'LIGHT':
        rim.data.energy = props.rim_intensity * (scale ** 1.5)

    # Update World / Ambient Light
    world = scene.world
    if not world:
        world = bpy.data.worlds.new("StudioWorld")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        if props.use_ambient_light:
            bg.inputs['Strength'].default_value = props.ambient_intensity
            bg.inputs['Color'].default_value = (
                props.ambient_color[0],
                props.ambient_color[1],
                props.ambient_color[2],
                1.0
            )
        else:
            bg.inputs['Strength'].default_value = 0.05
            bg.inputs['Color'].default_value = (0.02, 0.02, 0.025, 1.0)

    # Redraw all 3D viewports so user sees lighting changes in real-time
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# -------------------------------------------------------------------------
# Environment Sky Sphere (Pro Feature)
# -------------------------------------------------------------------------

def ensure_voronoi_preset_material(props=None):
    """
    Ensures a standalone preset Voronoi Texture material ('AbuCam_Voronoi_Material') exists
    in bpy.data.materials so it is always available as a default preset in the Material picker.
    """
    preset_name = "AbuCam_Voronoi_Material"
    mat = bpy.data.materials.get(preset_name)
    if not mat:
        mat = bpy.data.materials.new(name=preset_name)
        mat.use_fake_user = True
        mat.use_nodes = True
        try:
            mat.use_backface_culling = False
        except Exception:
            pass
        tree = mat.node_tree
        nodes = tree.nodes
        links = tree.links
        nodes.clear()

        out = nodes.new(type='ShaderNodeOutputMaterial')
        out.location = (550, 0)

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (250, 0)
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = 1.0
        if 'Emission Strength' in bsdf.inputs:
            bsdf.inputs['Emission Strength'].default_value = 1.0

        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-600, 0)

        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-380, 0)

        voronoi = nodes.new(type='ShaderNodeTexVoronoi')
        voronoi.location = (-120, 80)
        try:
            voronoi.voronoi_dimensions = '3D'
            voronoi.feature = 'F1'
        except Exception:
            pass
        if 'Scale' in voronoi.inputs:
            voronoi.inputs['Scale'].default_value = float(getattr(props, 'env_sphere_voronoi_scale', 25.0)) if props else 25.0
        if 'Detail' in voronoi.inputs:
            voronoi.inputs['Detail'].default_value = float(getattr(props, 'env_sphere_voronoi_detail', 0.0)) if props else 0.0
        if 'Randomness' in voronoi.inputs:
            voronoi.inputs['Randomness'].default_value = float(getattr(props, 'env_sphere_voronoi_randomness', 1.0)) if props else 1.0

        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], voronoi.inputs['Vector'])
        links.new(voronoi.outputs['Color'], bsdf.inputs['Base Color'])
        if 'Emission Color' in bsdf.inputs:
            links.new(voronoi.outputs['Color'], bsdf.inputs['Emission Color'])
        elif 'Emission' in bsdf.inputs:
            links.new(voronoi.outputs['Color'], bsdf.inputs['Emission'])
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def build_env_sphere_material(props):
    """
    Creates or updates the 'AbuCam_EnvSkySphere_Mat' material according to props.env_sphere_mode:
    - 'VORONOI': Procedural 3D Voronoi Texture (Multi-Color Cells / Distance / Smooth F1 / Edges) - Default Preset
    - 'HDRI_TEX': Equirectangular or UV Image/HDRI Panorama Texture with Emission & Tint
    - 'PROCEDURAL_SKY': Physically-based atmospheric Sky Texture (Nishita/Hosek)
    - 'GRADIENT': Smooth 3-stop studio cyclorama gradient (Ground -> Horizon -> Zenith)
    - 'SOLID': Uniform solid backdrop color with controllable emission and roughness
    - 'CUSTOM_MAT': Uses user-selected material directly (defaults to AbuCam_Voronoi_Material preset)
    """
    preset_voronoi = ensure_voronoi_preset_material(props)
    if props.env_sphere_mode == 'CUSTOM_MAT':
        if not props.env_sphere_custom_mat:
            try:
                props.env_sphere_custom_mat = preset_voronoi
            except Exception:
                pass
        return props.env_sphere_custom_mat or preset_voronoi

    mat_name = "AbuCam_EnvSkySphere_Mat"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    try:
        mat.use_backface_culling = False
    except Exception:
        pass

    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    out = nodes.new(type='ShaderNodeOutputMaterial')
    out.location = (600, 0)

    # Mix between Principled BSDF (for roughness/specular control) and Emission (for self-lit sky)
    emit_strength = max(0.0, float(props.env_sphere_emission))
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    if 'Roughness' in bsdf.inputs:
        bsdf.inputs['Roughness'].default_value = float(props.env_sphere_roughness)
    if 'Emission Strength' in bsdf.inputs:
        bsdf.inputs['Emission Strength'].default_value = emit_strength

    # Texture Coordinate & Mapping for Z-Rotation and UV Scale
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-650, 0)

    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-430, 0)
    mapping.inputs['Rotation'].default_value = (0.0, 0.0, float(props.env_sphere_rotation_z))
    mapping.inputs['Scale'].default_value = (
        float(props.env_sphere_scale_u),
        float(props.env_sphere_scale_v),
        1.0
    )

    mode = props.env_sphere_mode
    color_socket = None

    if mode == 'VORONOI':
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        voronoi = nodes.new(type='ShaderNodeTexVoronoi')
        voronoi.location = (-180, 100)
        v_out_mode = getattr(props, 'env_sphere_voronoi_output', 'COLOR')
        try:
            voronoi.voronoi_dimensions = '3D'
            if v_out_mode == 'SMOOTH_COLOR':
                voronoi.feature = 'SMOOTH_F1'
                if 'Smoothness' in voronoi.inputs:
                    voronoi.inputs['Smoothness'].default_value = 1.0
            elif v_out_mode == 'EDGES':
                voronoi.feature = 'DISTANCE_TO_EDGE'
            else:
                voronoi.feature = 'F1'
        except Exception:
            pass

        if 'Scale' in voronoi.inputs:
            voronoi.inputs['Scale'].default_value = float(getattr(props, 'env_sphere_voronoi_scale', 25.0))
        if 'Detail' in voronoi.inputs:
            voronoi.inputs['Detail'].default_value = float(getattr(props, 'env_sphere_voronoi_detail', 0.0))
        if 'Randomness' in voronoi.inputs:
            voronoi.inputs['Randomness'].default_value = float(getattr(props, 'env_sphere_voronoi_randomness', 1.0))

        links.new(mapping.outputs['Vector'], voronoi.inputs['Vector'])

        if v_out_mode in ('DISTANCE', 'EDGES'):
            mix_rgb = nodes.new(type='ShaderNodeMix')
            mix_rgb.data_type = 'RGBA'
            mix_rgb.blend_type = 'MULTIPLY'
            mix_rgb.location = (80, 100)
            mix_rgb.inputs['Factor'].default_value = 1.0
            mix_rgb.inputs['B'].default_value = (
                props.env_sphere_color[0],
                props.env_sphere_color[1],
                props.env_sphere_color[2],
                1.0
            )
            links.new(voronoi.outputs['Distance'], mix_rgb.inputs['A'])
            color_socket = mix_rgb.outputs['Result']
        else:
            color_socket = voronoi.outputs['Color']

    elif mode == 'HDRI_TEX':
        # Load image from file path if specified, or from image pointer
        img = props.env_sphere_image
        if props.env_sphere_texture_path:
            abs_p = bpy.path.abspath(props.env_sphere_texture_path)
            if os.path.isfile(abs_p):
                try:
                    img = bpy.data.images.load(abs_p, check_existing=True)
                except Exception:
                    pass

        if props.env_sphere_projection == 'EQUIRECTANGULAR':
            links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            env_tex = nodes.new(type='ShaderNodeTexEnvironment')
            env_tex.location = (-180, 100)
            if img:
                env_tex.image = img
            links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
            tex_out = env_tex.outputs['Color']
        else:
            links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
            img_tex = nodes.new(type='ShaderNodeTexImage')
            img_tex.location = (-180, 100)
            if img:
                img_tex.image = img
            links.new(mapping.outputs['Vector'], img_tex.inputs['Vector'])
            tex_out = img_tex.outputs['Color']

        if img:
            # Multiply texture by Tint Color
            mix_rgb = nodes.new(type='ShaderNodeMix')
            mix_rgb.data_type = 'RGBA'
            mix_rgb.blend_type = 'MULTIPLY'
            mix_rgb.location = (80, 100)
            mix_rgb.inputs['Factor'].default_value = 1.0
            mix_rgb.inputs['B'].default_value = (
                props.env_sphere_color[0],
                props.env_sphere_color[1],
                props.env_sphere_color[2],
                1.0
            )
            links.new(tex_out, mix_rgb.inputs['A'])
            color_socket = mix_rgb.outputs['Result']
        else:
            # Clean procedural sky fallback when no texture file is selected yet
            sky = nodes.new(type='ShaderNodeTexSky')
            sky.location = (-180, -120)
            try:
                sky.sky_type = 'NISHITA'
                sky.sun_elevation = float(props.env_sphere_sun_elevation)
                sky.sun_rotation = float(props.env_sphere_rotation_z)
            except Exception:
                pass
            color_socket = sky.outputs['Color']

    elif mode == 'PROCEDURAL_SKY':
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        sky = nodes.new(type='ShaderNodeTexSky')
        sky.location = (-120, 80)
        try:
            sky.sky_type = 'NISHITA'
            sky.sun_elevation = float(props.env_sphere_sun_elevation)
            sky.sun_rotation = float(props.env_sphere_rotation_z)
        except Exception:
            try:
                sky.sky_type = 'HOSEK_WILKIE'
            except Exception:
                pass
        links.new(mapping.outputs['Vector'], sky.inputs['Vector'])
        color_socket = sky.outputs['Color']

    elif mode == 'GRADIENT':
        links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
        sep = nodes.new(type='ShaderNodeSeparateXYZ')
        sep.location = (-220, 80)
        links.new(mapping.outputs['Vector'], sep.inputs['Vector'])

        ramp = nodes.new(type='ShaderNodeValToRGB')
        ramp.location = (-20, 80)
        ramp.color_ramp.interpolation = 'B_SPLINE'
        # Stop 0: Ground Nadir (Z=0), Stop 1: Horizon (Z=0.5), Stop 2: Zenith (Z=1.0)
        el0 = ramp.color_ramp.elements[0]
        el0.position = 0.0
        el0.color = (
            props.env_sphere_ground_color[0],
            props.env_sphere_ground_color[1],
            props.env_sphere_ground_color[2],
            1.0
        )
        el1 = ramp.color_ramp.elements[1]
        el1.position = 1.0
        el1.color = (
            props.env_sphere_zenith_color[0],
            props.env_sphere_zenith_color[1],
            props.env_sphere_zenith_color[2],
            1.0
        )
        el_mid = ramp.color_ramp.elements.new(0.5)
        el_mid.color = (
            props.env_sphere_color[0],
            props.env_sphere_color[1],
            props.env_sphere_color[2],
            1.0
        )
        links.new(sep.outputs['Z'], ramp.inputs['Fac'])
        color_socket = ramp.outputs['Color']

    else:  # SOLID
        rgb_node = nodes.new(type='ShaderNodeRGB')
        rgb_node.location = (0, 80)
        rgb_node.outputs[0].default_value = (
            props.env_sphere_color[0],
            props.env_sphere_color[1],
            props.env_sphere_color[2],
            1.0
        )
        color_socket = rgb_node.outputs[0]

    if color_socket:
        links.new(color_socket, bsdf.inputs['Base Color'])
        if 'Emission Color' in bsdf.inputs:
            links.new(color_socket, bsdf.inputs['Emission Color'])
        elif 'Emission' in bsdf.inputs:
            links.new(color_socket, bsdf.inputs['Emission'])

    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def update_env_sky_sphere(scene):
    """
    Creates, updates, or hides the 3D Environment Sky Sphere ('AbuCam_EnvSkySphere') in the scene.
    Generates an inward-normal UV sphere mesh at controllable radius with studio ray visibility flags.
    """
    props = getattr(scene, 'abucam_props', None)
    if not props:
        return None

    sphere_name = "AbuCam_EnvSkySphere"
    mesh_name = "AbuCam_EnvSkySphere_Mesh"
    obj = bpy.data.objects.get(sphere_name)

    if not getattr(props, 'use_env_sphere', False):
        if obj:
            obj.hide_viewport = True
            obj.hide_render = True
        return obj

    seg = max(16, min(256, int(getattr(props, 'env_sphere_segments', 64))))
    rings = max(8, seg // 2)

    mesh = bpy.data.meshes.get(mesh_name)
    if not mesh:
        mesh = bpy.data.meshes.new(mesh_name)

    # Rebuild sphere geometry only if vertex count doesn't match current segment resolution
    expected_verts = seg * (rings - 1) + 2
    if len(mesh.vertices) != expected_verts:
        bm = bmesh.new()
        try:
            bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=1.0, calc_uvs=True)
        except TypeError:
            bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, diameter=1.0, calc_uvs=True)
        # Flip normals inward so interior surface faces the cameras
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()
        for poly in mesh.polygons:
            poly.use_smooth = True
        mesh.update()

    if not obj:
        obj = bpy.data.objects.new(sphere_name, mesh)
        scene.collection.objects.link(obj)
    else:
        if obj.name not in scene.collection.objects:
            try:
                scene.collection.objects.link(obj)
            except Exception:
                pass
        obj.data = mesh

    obj.hide_viewport = False
    obj.hide_render = False

    # Position and Radius Scale
    if getattr(props, 'env_sphere_follow_target', True):
        obj.location = (
            getattr(props, 'target_x', 0.0),
            getattr(props, 'target_y', 0.0),
            getattr(props, 'target_z', 0.0)
        )
    else:
        obj.location = (0.0, 0.0, 0.0)

    r = max(0.5, float(getattr(props, 'env_sphere_radius', 25.0)))
    obj.scale = (r, r, r)
    obj.rotation_euler = (0.0, 0.0, float(getattr(props, 'env_sphere_rotation_z', 0.0)))

    # Ray Visibility (Cycles & EEVEE Next)
    try:
        obj.visible_camera = bool(props.env_sphere_visible_camera)
        obj.visible_shadow = bool(props.env_sphere_cast_shadows)
        obj.visible_diffuse = bool(props.env_sphere_visible_diffuse)
        obj.visible_glossy = bool(props.env_sphere_visible_glossy)
    except Exception:
        pass

    # Assign Material
    mat = build_env_sphere_material(props)
    if mat:
        if len(mesh.materials) == 0:
            mesh.materials.append(mat)
        else:
            mesh.materials[0] = mat

    # Ensure active cameras and 3D viewports don't clip out the sky sphere
    far_clip = max(500.0, r * 2.5)
    for cam_o in scene.objects:
        if cam_o.type == 'CAMERA' and (cam_o.name == "OrbitalCamera" or cam_o.name.startswith("AbuCam_Vol_")):
            if cam_o.data.clip_end < far_clip:
                cam_o.data.clip_end = far_clip

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and space.clip_end < far_clip:
                        space.clip_end = far_clip
                area.tag_redraw()

    return obj


def on_env_sphere_update(self, context):
    """Callback triggered whenever any Environment Sky Sphere parameter is modified."""
    scene = getattr(context, 'scene', None) or bpy.context.scene
    if not scene:
        return
    update_env_sky_sphere(scene)


# -------------------------------------------------------------------------
# Properties
# -------------------------------------------------------------------------

class AbuCamProperties(bpy.types.PropertyGroup):
    live_update: BoolProperty(
        name="Live Viewport Update",
        description="Automatically update camera position and viewport in real-time as parameters change",
        default=True
    )
    rig_mode: EnumProperty(
        name="Rig Mode",
        description="Select camera capture system mode",
        items=[
            ('ORBITAL', "Orbital Camera Rig", "Single camera animating along a circular path over timeline frames"),
            ('VOLUMETRIC', "Volumetric Camera Rig", "Multi-camera dome / spherical array capturing synchronous views from multiple angles per frame"),
        ],
        default='ORBITAL',
        update=on_rig_mode_update
    )
    # Volumetric Camera Rig Properties
    vol_camera_count: IntProperty(
        name="Camera Count",
        description="Total number of cameras distributed across the volumetric sphere surface",
        default=32, min=2, max=500,
        update=on_volumetric_param_update
    )
    vol_sphere_radius: FloatProperty(
        name="Sphere Radius (m)",
        description="Radius of the volumetric camera capture dome / sphere",
        default=2.5, min=0.1, max=100.0, step=10, unit='LENGTH',
        update=on_volumetric_param_update
    )
    vol_focal_length: FloatProperty(
        name="Focal Length (mm)",
        description="Focal length in mm for all volumetric cameras",
        default=50.0, min=10.0, max=300.0,
        update=on_volumetric_param_update
    )
    vol_sensor_width: FloatProperty(
        name="Sensor Width (mm)",
        description="Camera sensor width in mm (Full Frame 35mm standard: 36mm)",
        default=36.0, min=1.0, max=100.0,
        update=on_volumetric_param_update
    )
    vol_sensor_height: FloatProperty(
        name="Sensor Height (mm)",
        description="Camera sensor height in mm (Full Frame 35mm standard: 24mm)",
        default=24.0, min=1.0, max=100.0,
        update=on_volumetric_param_update
    )
    frustum_display_size: FloatProperty(
        name="Frustum Scale",
        description="Visual display scale of camera frustum wireframes in the 3D Viewport (does not affect FOV, focal length, or rendered output)",
        default=0.25, min=0.01, max=10.0, step=5,
        update=on_frustum_size_update
    )
    vol_distribution: EnumProperty(
        name="Distribution",
        description="Pattern for distributing cameras across the sphere surface",
        items=[
            ('FIBONACCI', "Fibonacci Spiral (Uniform)", "Golden ratio spherical spiral providing mathematically uniform area distribution without pole crowding"),
            ('RINGS', "Latitude Rings (Tiers)", "Evenly spaced horizontal elevation tiers/rings around the dome"),
            ('RANDOM', "Jittered / Random", "Uniform spherical random distribution"),
        ],
        default='FIBONACCI',
        update=on_volumetric_param_update
    )
    # 3 Angular Surface Limits
    vol_azimuth_span_deg: FloatProperty(
        name="Azimuth Span (0-360°)",
        description="Horizontal arc coverage angle around sphere (360°=Full dome, 180°=Frontal semicircle, etc.)",
        default=math.radians(360.0),
        min=math.radians(10.0),
        max=math.radians(360.0),
        step=100,
        unit='ROTATION',
        update=on_volumetric_param_update
    )
    vol_azimuth_center_deg: FloatProperty(
        name="Azimuth Heading",
        description="Heading orientation of the azimuth arc center (0°=Front, 90°=Right, etc.)",
        default=0.0,
        min=math.radians(-180.0),
        max=math.radians(180.0),
        step=100,
        unit='ROTATION',
        update=on_volumetric_param_update
    )
    vol_elevation_min_deg: FloatProperty(
        name="Elevation Min (Zenith)",
        description="Upper ceiling limit angle from zenith (0°=Direct top-down)",
        default=math.radians(15.0),
        min=0.0,
        max=math.radians(170.0),
        step=100,
        unit='ROTATION',
        update=on_volumetric_param_update
    )
    vol_elevation_max_deg: FloatProperty(
        name="Elevation Max (Floor)",
        description="Lower floor limit angle from zenith (90°=Equator/Ground level, 180°=Direct bottom-up)",
        default=math.radians(90.0),
        min=math.radians(10.0),
        max=math.radians(180.0),
        step=100,
        unit='ROTATION',
        update=on_volumetric_param_update
    )
    vol_show_guide_dome: BoolProperty(
        name="Show 3D Dome Guide",
        description="Master toggle for 3D viewport camera guide lines",
        default=True,
        update=on_guide_lines_update
    )
    vol_show_camera_lines: BoolProperty(
        name="Lines Between Cameras",
        description="Toggle the 3D path line connecting consecutive cameras around the volumetric sphere",
        default=True,
        update=on_guide_lines_update
    )
    vol_show_center_lines: BoolProperty(
        name="Center Target Lines",
        description="Toggle radial sightlines and relationship lines connecting each camera to the target center",
        default=False,
        update=on_guide_lines_update
    )
    vol_active_cam_index: IntProperty(
        name="Active View Camera",
        description="Index of camera to look through in 3D viewport",
        default=1, min=1, max=500,
        update=on_active_cam_index_update
    )
    azimuth_deg: FloatProperty(
        name="Azimuth Angle (0-360°)",
        description="Camera azimuth orbital angle around target (0°=Front, 90°=Right, 180°=Back, 270°=Left)",
        default=0.0,
        min=0.0,
        max=math.radians(360.0),
        step=100,
        unit='ROTATION',
        update=on_camera_param_update
    )
    elevation_deg: FloatProperty(
        name="Elevation Angle (0-180°)",
        description="Camera elevation polar angle (0°=Zenith, 60°=High Angle, 90°=Eye Level, 180°=Nadir)",
        default=math.radians(60.0),
        min=math.radians(1.0),
        max=math.radians(179.0),
        step=100,
        unit='ROTATION',
        update=on_camera_param_update
    )
    path_tilt_deg: FloatProperty(
        name="Path Tilt (-90° to +90°)",
        description="Rotate the circular camera path plane from -90° to +90° (0°=Horizontal, ±90°=Vertical loop)",
        default=0.0,
        min=math.radians(-90.0),
        max=math.radians(90.0),
        step=100,
        unit='ROTATION',
        update=on_camera_param_update
    )
    orbit_mode: EnumProperty(
        name="Orbit Arc Mode",
        description="Coverage of the circular camera path",
        items=[
            ('FULL_360', "Full Orbit (0° to 360°)", "Complete 360-degree turntable orbit around subject"),
            ('ARC_180', "Frontal Arc (-90° to +90°)", "Frontal 180-degree sweep from left profile (-90°) to right profile (+90°)"),
            ('CUSTOM', "Custom Arc", "Custom user-defined start and end angles along circular path"),
        ],
        default='FULL_360',
        update=on_camera_param_update
    )
    arc_start_deg: FloatProperty(
        name="Arc Start",
        description="Starting angle along circular path",
        default=math.radians(-90.0),
        min=math.radians(-180.0),
        max=math.radians(180.0),
        step=100,
        unit='ROTATION',
        update=on_camera_param_update
    )
    arc_end_deg: FloatProperty(
        name="Arc End",
        description="Ending angle along circular path",
        default=math.radians(90.0),
        min=math.radians(-180.0),
        max=math.radians(360.0),
        step=100,
        unit='ROTATION',
        update=on_camera_param_update
    )
    show_path_guide: BoolProperty(
        name="Show 3D Orbit Path Curve",
        description="Display a 3D visual guide curve of the circular camera path in the viewport",
        default=True,
        update=on_camera_param_update
    )
    orbit_radius: FloatProperty(
        name="Orbit Radius (m)",
        description="Distance of camera orbit from target center",
        default=2.5, min=0.1, max=100.0, step=10, unit='LENGTH',
        update=on_camera_param_update
    )
    target_object: PointerProperty(
        name="Target Layer",
        type=bpy.types.Object,
        description="Target layer or object to track and keep centered in frame",
        update=on_target_object_update
    )
    active_track_target: BoolProperty(
        name="Active Track (Keep Target in Center)",
        description="Always keep target layer in the dead center of the frame even when Orbit Width (X), Depth (Y), or Height (Z) change",
        default=True,
        update=on_camera_param_update
    )
    auto_track_target: BoolProperty(
        name="Auto-Follow Target Movement",
        description="Automatically update orbit center and follow target layer when moved in 3D viewport",
        default=True
    )
    target_x: FloatProperty(
        name="Orbit Width (X)",
        description="X coordinate of orbit center (Width / Lateral offset)",
        default=0.0, step=1, unit='LENGTH',
        update=on_camera_param_update
    )
    target_y: FloatProperty(
        name="Orbit Depth (Y)",
        description="Y coordinate of orbit center (Depth / Forward-Back offset)",
        default=0.0, step=1, unit='LENGTH',
        update=on_camera_param_update
    )
    target_z: FloatProperty(
        name="Orbit Height (Z)",
        description="Z coordinate of orbit center (Height / Vertical offset)",
        default=0.0, step=1, unit='LENGTH',
        update=on_camera_param_update
    )
    focal_length: FloatProperty(
        name="Focal Length (mm)",
        description="Camera lens focal length in mm",
        default=50.0, min=10.0, max=300.0,
        update=on_camera_param_update
    )
    total_frames: IntProperty(
        name="Total Orbit Views",
        description="Total frames for full 360 degree orbit (Wan diffusion standard: 81)",
        default=81, min=1, max=1000,
        update=on_camera_param_update
    )
    fps: IntProperty(
        name="FPS",
        description="Animation and video frame rate",
        default=16, min=1, max=120,
        update=on_camera_param_update
    )
    resolution_x: IntProperty(
        name="Width",
        description="Render width in pixels",
        default=832, min=128, max=4096,
        update=on_camera_param_update
    )
    resolution_y: IntProperty(
        name="Height",
        description="Render height in pixels",
        default=480, min=128, max=4096,
        update=on_camera_param_update
    )
    # Studio Lighting Properties
    use_ambient_light: BoolProperty(
        name="Soft Ambient Light",
        description="Enable soft omnidirectional ambient lighting from all directions",
        default=True,
        update=on_light_param_update
    )
    ambient_intensity: FloatProperty(
        name="Ambient Intensity",
        description="Brightness of omnidirectional ambient light from all directions",
        default=1.0, min=0.0, max=10.0,
        update=on_light_param_update
    )
    ambient_color: FloatVectorProperty(
        name="Ambient Tint",
        description="Color tint of ambient light",
        subtype='COLOR', size=4,
        default=(0.92, 0.94, 1.0, 1.0),
        update=on_light_param_update
    )
    key_intensity: FloatProperty(
        name="Key Light (W)",
        description="Power/Intensity of Key Area Light",
        default=45.0, min=0.0, max=500.0,
        update=on_light_param_update
    )
    fill_intensity: FloatProperty(
        name="Fill Light (W)",
        description="Power/Intensity of Fill Area Light",
        default=18.0, min=0.0, max=500.0,
        update=on_light_param_update
    )
    rim_intensity: FloatProperty(
        name="Rim Light (W)",
        description="Power/Intensity of Rim Area Light",
        default=32.0, min=0.0, max=500.0,
        update=on_light_param_update
    )
    # Environment Sky Sphere Properties (Pro Feature)
    use_env_sphere: BoolProperty(
        name="Enable Environment Sky Sphere",
        description="Add a 3D spherical environment dome mesh around the scene with customizable shader, texture, radius, and ray visibility",
        default=False,
        update=on_env_sphere_update
    )
    env_sphere_radius: FloatProperty(
        name="Sky Sphere Radius (m)",
        description="Radius of the 3D environment sky sphere in meters",
        default=25.0, min=0.5, max=5000.0, step=50, unit='LENGTH',
        update=on_env_sphere_update
    )
    env_sphere_mode: EnumProperty(
        name="Material / Sky Mode",
        description="Material and shader type for the environment sky sphere",
        items=[
            ('VORONOI', "Voronoi Texture (Default Preset)", "Procedural 3D Voronoi cell texture preset (ideal for 3DGS / COLMAP / Photogrammetry feature matching)"),
            ('HDRI_TEX', "Texture / HDRI Panorama", "Map an equirectangular HDRI/EXR or image texture onto the sky sphere"),
            ('PROCEDURAL_SKY', "Procedural Physical Sky", "Physically-based atmospheric sky dome (Nishita / Hosek-Wilkie)"),
            ('GRADIENT', "Studio Horizon Gradient", "Smooth 3-stop studio cyclorama gradient (Ground -> Horizon -> Zenith)"),
            ('SOLID', "Solid Backdrop Color", "Uniform solid color emissive/shaded backdrop sphere"),
            ('CUSTOM_MAT', "Custom Scene Material", "Assign any existing Blender material from the scene (includes AbuCam_Voronoi_Material preset)"),
        ],
        default='VORONOI',
        update=on_env_sphere_update
    )
    env_sphere_voronoi_output: EnumProperty(
        name="Voronoi Style",
        description="Voronoi texture feature and color output mode",
        items=[
            ('COLOR', "Multi-Color Cells (F1 Color)", "Distinct random RGB color per Voronoi cell (best for 3DGS / COLMAP background feature matching)"),
            ('SMOOTH_COLOR', "Smooth Multi-Color (Smooth F1)", "Smoothly blended multi-color Voronoi cells"),
            ('DISTANCE', "Cell Distance (Shaded)", "Cellular distance gradient from cell center to border, tinted by Backdrop Color"),
            ('EDGES', "Cell Edges / Cracks", "Distance-to-edge cellular border lines, tinted by Backdrop Color"),
        ],
        default='COLOR',
        update=on_env_sphere_update
    )
    env_sphere_voronoi_scale: FloatProperty(
        name="Voronoi Scale",
        description="Scale / frequency of Voronoi cells across the environment sky sphere",
        default=25.0, min=0.1, max=1000.0, step=50,
        update=on_env_sphere_update
    )
    env_sphere_voronoi_detail: FloatProperty(
        name="Voronoi Detail",
        description="Fractal detail levels of the Voronoi texture (0 = clean cells)",
        default=0.0, min=0.0, max=15.0, step=10,
        update=on_env_sphere_update
    )
    env_sphere_voronoi_randomness: FloatProperty(
        name="Voronoi Randomness",
        description="Randomness of Voronoi cell distribution (1.0 = organic cells, 0.0 = regular grid)",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        update=on_env_sphere_update
    )
    env_sphere_projection: EnumProperty(
        name="Texture Projection",
        description="Coordinate projection method for the environment sphere texture",
        items=[
            ('EQUIRECTANGULAR', "Equirectangular (360° Panorama)", "Spherical lat-long environment panorama projection"),
            ('UV', "Sphere UV Mapping", "Standard UV sphere coordinate mapping"),
        ],
        default='EQUIRECTANGULAR',
        update=on_env_sphere_update
    )
    env_sphere_texture_path: StringProperty(
        name="Texture File",
        description="Path to an HDRI (.hdr, .exr) or image (.png, .jpg) texture file for the sky sphere",
        default="",
        subtype='FILE_PATH',
        update=on_env_sphere_update
    )
    env_sphere_image: PointerProperty(
        name="Loaded Image",
        type=bpy.types.Image,
        description="Select a loaded Blender image or HDRI data-block for the sky sphere",
        update=on_env_sphere_update
    )
    env_sphere_custom_mat: PointerProperty(
        name="Custom Material",
        type=bpy.types.Material,
        description="Select any existing scene material to assign directly to the environment sky sphere",
        update=on_env_sphere_update
    )
    env_sphere_emission: FloatProperty(
        name="Emission / Exposure",
        description="Self-illumination emission strength / exposure of the environment sky sphere",
        default=1.0, min=0.0, max=100.0, step=10,
        update=on_env_sphere_update
    )
    env_sphere_roughness: FloatProperty(
        name="Surface Roughness",
        description="Surface roughness of the environment sky sphere",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        update=on_env_sphere_update
    )
    env_sphere_color: FloatVectorProperty(
        name="Tint / Horizon Color",
        description="Texture color tint, horizon color (Gradient mode), or solid backdrop color",
        subtype='COLOR', size=4,
        default=(0.85, 0.90, 1.0, 1.0),
        update=on_env_sphere_update
    )
    env_sphere_zenith_color: FloatVectorProperty(
        name="Zenith Color",
        description="Upper zenith sky color in Studio Horizon Gradient mode",
        subtype='COLOR', size=4,
        default=(0.15, 0.28, 0.58, 1.0),
        update=on_env_sphere_update
    )
    env_sphere_ground_color: FloatVectorProperty(
        name="Ground Color",
        description="Lower nadir ground color in Studio Horizon Gradient mode",
        subtype='COLOR', size=4,
        default=(0.12, 0.13, 0.16, 1.0),
        update=on_env_sphere_update
    )
    env_sphere_sun_elevation: FloatProperty(
        name="Sun Elevation",
        description="Sun elevation angle for Procedural Physical Sky mode",
        default=math.radians(35.0), min=math.radians(-10.0), max=math.radians(90.0),
        step=100, unit='ROTATION',
        update=on_env_sphere_update
    )
    env_sphere_rotation_z: FloatProperty(
        name="Dome Z-Rotation",
        description="Rotate the environment sky sphere and texture horizontally around the Z axis (0° to 360°)",
        default=0.0, min=math.radians(-360.0), max=math.radians(360.0),
        step=100, unit='ROTATION',
        update=on_env_sphere_update
    )
    env_sphere_scale_u: FloatProperty(
        name="Scale U",
        description="Horizontal UV / texture mapping scale factor",
        default=1.0, min=0.05, max=50.0,
        update=on_env_sphere_update
    )
    env_sphere_scale_v: FloatProperty(
        name="Scale V",
        description="Vertical UV / texture mapping scale factor",
        default=1.0, min=0.05, max=50.0,
        update=on_env_sphere_update
    )
    env_sphere_segments: IntProperty(
        name="Sphere Segments",
        description="Polygon subdivision segments of the sky sphere mesh",
        default=64, min=16, max=256,
        update=on_env_sphere_update
    )
    env_sphere_follow_target: BoolProperty(
        name="Center on Target Rig",
        description="Keep the environment sky sphere centered on the active target/rig coordinates (X, Y, Z)",
        default=True,
        update=on_env_sphere_update
    )
    env_sphere_visible_camera: BoolProperty(
        name="Camera Visibility",
        description="Make the sky sphere directly visible to camera rays in RGB renders",
        default=True,
        update=on_env_sphere_update
    )
    env_sphere_cast_shadows: BoolProperty(
        name="Cast Shadows",
        description="Allow the sky sphere to cast shadows (Keep OFF so external studio lights shine through unimpeded)",
        default=False,
        update=on_env_sphere_update
    )
    env_sphere_visible_diffuse: BoolProperty(
        name="Diffuse GI",
        description="Allow the sky sphere to contribute to diffuse indirect global illumination",
        default=True,
        update=on_env_sphere_update
    )
    env_sphere_visible_glossy: BoolProperty(
        name="Specular Reflections",
        description="Make the sky sphere visible in glossy/specular reflections on scene materials",
        default=True,
        update=on_env_sphere_update
    )
    env_sphere_exclude_aux_passes: BoolProperty(
        name="Auto-Hide in Depth / Normal / Wire Passes",
        description="Automatically hide the sky sphere during Depth, Normal, and Wireframe passes so only the foreground subject is captured",
        default=True
    )
    auto_compile_videos: BoolProperty(
        name="Auto-Compile MP4 Videos",
        description="Automatically compile H.264 MP4 videos after multi-pass rendering finishes",
        default=True
    )
    renderer: EnumProperty(
        name="Engine",
        description="Render engine for multi-pass generation",
        items=[
            ('BLENDER_EEVEE', "Eevee (Real-Time GPU)", "Light-speed real-time rasterization via GPU shaders"),
            ('CYCLES', "Cycles (OptiX RTX Raytracing)", "Full GPU raytracing on RTX 4090 Tensor Cores"),
            ('BLENDER_WORKBENCH', "Workbench (Clean)", "Fast clean wireframe/matcap"),
        ],
        default='BLENDER_EEVEE'
    )
    eevee_raytracing: BoolProperty(
        name="Screen Raytracing",
        description="Enable screen-space raytracing in EEVEE Next (turn off for maximum rendering speed)",
        default=False
    )
    samples: IntProperty(
        name="Samples",
        description="Cycles / Eevee render samples per frame",
        default=32, min=1, max=2048
    )
    pass_rgb: BoolProperty(name="RGB Pass", default=True)
    pass_depth: BoolProperty(name="16-bit Depth Pass", default=True)
    pass_normal: BoolProperty(name="Surface Normal Pass", default=True)
    pass_wire_rgb: BoolProperty(
        name="Wireframe + RGB Pass",
        description="Render RGB shaded model with overlaid wireframe topology lines",
        default=True
    )
    wire_color_mode: EnumProperty(
        name="Wire Color",
        description="Color style for the wireframe lines",
        items=[
            ('ORANGE', "Viewport Orange", "Classic Blender selected wireframe orange"),
            ('DARK', "Dark / Black", "Crisp dark technical CAD wireframe"),
            ('WHITE', "White", "Clean high-contrast white wireframe"),
            ('CYAN', "Cyan", "Futuristic neon cyan wireframe"),
        ],
        default='ORANGE'
    )
    wire_thickness: FloatProperty(
        name="Wire Width (px)",
        description="Thickness of wireframe lines in pixels",
        default=1.0, min=0.2, max=5.0, step=10
    )
    wire_opacity: FloatProperty(
        name="Wire Opacity",
        description="Blending opacity of wireframe over RGB",
        default=0.85, min=0.1, max=1.0, step=5,
        subtype='FACTOR'
    )
    is_rendering: BoolProperty(
        name="Is Rendering",
        description="Indicates whether multi-pass or volumetric rendering is actively running",
        default=False
    )
    is_paused: BoolProperty(
        name="Is Paused",
        description="Temporarily pauses the active render process",
        default=False
    )
    show_render_preview_window: BoolProperty(
        name="Auto-Show Render Window",
        description="Automatically opens the render preview window when rendering begins",
        default=True
    )
    render_frame_current: IntProperty(
        name="Current Render Frame",
        description="The index of the frame currently being rendered",
        default=0
    )
    render_frame_total: IntProperty(
        name="Total Render Frames",
        description="Total number of frames in the current render job",
        default=0
    )
    render_task_current: IntProperty(
        name="Current Task Index",
        description="Current render task index (across all passes/cameras)",
        default=0
    )
    render_task_total: IntProperty(
        name="Total Tasks",
        description="Total number of render tasks in the queue",
        default=0
    )
    render_current_pass: StringProperty(
        name="Current Pass Name",
        description="Name of the pass/camera currently rendering",
        default=""
    )
    render_last_frame_time: FloatProperty(
        name="Last Frame Render Time",
        description="Seconds taken to render the most recent frame/pass",
        default=0.0
    )
    render_avg_frame_time: FloatProperty(
        name="Average Frame Render Time",
        description="Average seconds taken per render task",
        default=0.0
    )
    render_time_elapsed: FloatProperty(
        name="Elapsed Render Time",
        description="Total elapsed seconds since render began",
        default=0.0
    )
    render_time_remaining: FloatProperty(
        name="Estimated Time Remaining",
        description="Estimated seconds remaining based on average frame time",
        default=0.0
    )
    cancel_render: BoolProperty(
        name="Cancel Render Flag",
        description="Flag to signal immediate cancellation of active render loop",
        default=False
    )
    render_status_msg: StringProperty(
        name="Render Status",
        description="Real-time progress message of the active render operation",
        default=""
    )
    render_progress: FloatProperty(
        name="Render Progress",
        description="Fractional progress of current render job (0.0 to 1.0)",
        default=0.0, min=0.0, max=1.0,
        subtype='PERCENTAGE'
    )
    output_dir: StringProperty(
        name="Output Directory",
        description="Folder where passes, dataset, and videos are saved",
        default=r"C:\Users\USer\Documents\Works\Apps\ComfyUI\3DGS_Auto_Workflow_01\AbuCam_Blender_Export",
        subtype='DIR_PATH'
    )


# -------------------------------------------------------------------------
# Operators
# -------------------------------------------------------------------------

class ABUCAM_OT_StopRender(bpy.types.Operator):
    bl_idname = "abucam.stop_render"
    bl_label = "Stop / Cancel Rendering"
    bl_description = "Immediately halts active multi-pass rendering and restores scene state without freezing"

    def execute(self, context):
        props = context.scene.abucam_props
        props.cancel_render = True
        props.is_paused = False
        props.render_status_msg = "Cancelling render..."
        self.report({'WARNING'}, "Render cancellation requested. Halting...")
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


class ABUCAM_OT_TogglePauseRender(bpy.types.Operator):
    bl_idname = "abucam.toggle_pause_render"
    bl_label = "Pause / Resume Rendering"
    bl_description = "Pauses or resumes the active rendering queue"

    def execute(self, context):
        props = context.scene.abucam_props
        props.is_paused = not props.is_paused
        if props.is_paused:
            props.render_status_msg = f"PAUSED at frame {props.render_frame_current}/{props.render_frame_total}"
            abucam_log(f"Render PAUSED by user at task {props.render_task_current}/{props.render_task_total}", level="INFO", scene=context.scene)
            self.report({'INFO'}, "Render PAUSED. Click Resume to continue.")
        else:
            props.render_status_msg = "Resuming render..."
            abucam_log("Render RESUMED by user", level="INFO", scene=context.scene)
            self.report({'INFO'}, "Render RESUMED.")
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for reg in area.regions:
                    if reg.type == 'UI':
                        reg.tag_redraw()
        return {'FINISHED'}


class ABUCAM_OT_ShowRenderWindow(bpy.types.Operator):
    bl_idname = "abucam.show_render_window"
    bl_label = "Open Render Preview Window"
    bl_description = "Opens the Blender Render Result window to monitor frame rendering live"

    def execute(self, context):
        try:
            bpy.ops.render.view_show('INVOKE_DEFAULT')
            self.report({'INFO'}, "Render preview window opened.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'WARNING'}, f"Could not open render window: {e}")
            return {'CANCELLED'}


class ABUCAM_OT_SetupRig(bpy.types.Operator):
    bl_idname = "abucam.setup_rig"
    bl_label = "Setup Orbital Camera Rig"
    bl_description = "Creates and animates a smooth orbital camera around the scene target"

    def execute(self, context):
        props = context.scene.abucam_props
        update_orbital_rig(context.scene, bake_keyframes=True)
        tilt_deg = math.degrees(getattr(props, 'path_tilt_deg', 0.0))
        self.report({'INFO'}, f"Orbital rig configured: {props.total_frames} frames @ {props.fps}fps (Path Tilt: {tilt_deg:+.1f}°, Arc: {props.orbit_mode}).")
        return {'FINISHED'}


class ABUCAM_OT_CenterOnTarget(bpy.types.Operator):
    bl_idname = "abucam.center_on_target"
    bl_label = "Center on Target"
    bl_description = "Calculates geometric center of selected target layer/object and centers orbit (X, Y, Z) on it"

    def execute(self, context):
        props = context.scene.abucam_props
        target_obj = props.target_object or context.active_object
        if not target_obj:
            self.report({'WARNING'}, "No target layer or active object found.")
            return {'CANCELLED'}

        if not props.target_object:
            props.target_object = target_obj

        bbox = [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box]
        center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0

        props.target_x = center.x
        props.target_y = center.y
        props.target_z = center.z

        global _last_target_matrix, _last_target_center
        _last_target_matrix = target_obj.matrix_world.copy()
        _last_target_center = center.copy()

        if getattr(props, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
            update_volumetric_rig(context.scene)
        else:
            update_orbital_rig(context.scene, bake_keyframes=False)
        self.report({'INFO'}, f"Orbit centered on '{target_obj.name}' at ({center.x:.3f}, {center.y:.3f}, {center.z:.3f})")
        return {'FINISHED'}


class ABUCAM_OT_ResetParam(bpy.types.Operator):
    bl_idname = "abucam.reset_param"
    bl_label = "Reset"
    bl_description = "Reset this parameter to its default value"

    param: StringProperty(name="Parameter")

    def execute(self, context):
        props = context.scene.abucam_props
        p = self.param

        if p in ("target_x", "target_y", "target_z"):
            if props.target_object:
                obj = props.target_object
                bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
                if p == "target_x":
                    props.target_x = center.x
                elif p == "target_y":
                    props.target_y = center.y
                elif p == "target_z":
                    props.target_z = center.z
            else:
                setattr(props, p, 0.0)
        elif p in PARAM_DEFAULTS:
            setattr(props, p, PARAM_DEFAULTS[p])
        else:
            try:
                props.property_unset(p)
            except Exception:
                pass

        if p in ("use_ambient_light", "ambient_intensity", "ambient_color", "key_intensity", "fill_intensity", "rim_intensity"):
            on_light_param_update(props, context)
        elif p == "frustum_display_size":
            on_frustum_size_update(props, context)
        elif p.startswith("env_sphere_") or p == "use_env_sphere":
            update_env_sky_sphere(context.scene)
        elif p.startswith("vol_") or getattr(props, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
            update_volumetric_rig(context.scene)
        else:
            update_orbital_rig(context.scene, bake_keyframes=False)

        return {'FINISHED'}


class ABUCAM_OT_ResetAll(bpy.types.Operator):
    bl_idname = "abucam.reset_all"
    bl_label = "Reset All Parameters"
    bl_description = "Reset all AbuCam parameters across all sections to their factory default values"

    def execute(self, context):
        props = context.scene.abucam_props
        for k, v in PARAM_DEFAULTS.items():
            try:
                setattr(props, k, v)
            except Exception:
                pass

        if props.target_object:
            obj = props.target_object
            bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
            props.target_x = center.x
            props.target_y = center.y
            props.target_z = center.z
            global _last_target_matrix, _last_target_center
            _last_target_matrix = obj.matrix_world.copy()
            _last_target_center = center.copy()
        else:
            props.target_x = 0.0
            props.target_y = 0.0
            props.target_z = 0.0

        on_light_param_update(props, context)
        update_env_sky_sphere(context.scene)
        if getattr(props, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
            update_volumetric_rig(context.scene)
        else:
            update_orbital_rig(context.scene, bake_keyframes=False)
        self.report({'INFO'}, "All AbuCam parameters have been reset to default values.")
        return {'FINISHED'}


class ABUCAM_OT_ResetRig(bpy.types.Operator):
    bl_idname = "abucam.reset_rig"
    bl_label = "Reset Rig"
    bl_description = "Reset camera angles, orbit radius, lens, and animation properties to defaults"

    def execute(self, context):
        props = context.scene.abucam_props
        rig_keys = [
            "azimuth_deg", "elevation_deg", "path_tilt_deg",
            "orbit_mode", "arc_start_deg", "arc_end_deg",
            "show_path_guide", "orbit_radius", "focal_length",
            "frustum_display_size", "total_frames", "fps",
            "resolution_x", "resolution_y"
        ]
        for k in rig_keys:
            if k in PARAM_DEFAULTS:
                setattr(props, k, PARAM_DEFAULTS[k])

        update_orbital_rig(context.scene, bake_keyframes=False)
        self.report({'INFO'}, "Camera rig parameters reset to defaults.")
        return {'FINISHED'}


class ABUCAM_OT_ResetVolumetricRig(bpy.types.Operator):
    bl_idname = "abucam.reset_volumetric_rig"
    bl_label = "Reset Volumetric Rig"
    bl_description = "Reset camera count, sphere radius, optics, resolution, and distribution to defaults"

    def execute(self, context):
        props = context.scene.abucam_props
        vol_keys = [
            "vol_camera_count", "vol_sphere_radius", "vol_focal_length",
            "vol_sensor_width", "vol_sensor_height", "frustum_display_size",
            "vol_distribution", "vol_azimuth_span_deg", "vol_azimuth_center_deg",
            "vol_elevation_min_deg", "vol_elevation_max_deg",
            "vol_show_guide_dome", "vol_show_camera_lines", "vol_show_center_lines",
            "vol_active_cam_index",
            "total_frames", "fps", "resolution_x", "resolution_y",
        ]
        for k in vol_keys:
            if k in PARAM_DEFAULTS:
                setattr(props, k, PARAM_DEFAULTS[k])

        update_volumetric_rig(context.scene)
        self.report({'INFO'}, "Volumetric rig parameters reset to defaults.")
        return {'FINISHED'}


class ABUCAM_OT_ResetTargetOffsets(bpy.types.Operator):
    bl_idname = "abucam.reset_target_offsets"
    bl_label = "Reset Target Offsets"
    bl_description = "Reset Orbit Width (X), Depth (Y), and Height (Z) back to target center"

    def execute(self, context):
        props = context.scene.abucam_props
        if props.target_object:
            obj = props.target_object
            bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            center = sum(bbox, Vector((0.0, 0.0, 0.0))) / 8.0
            props.target_x = center.x
            props.target_y = center.y
            props.target_z = center.z
            global _last_target_matrix, _last_target_center
            _last_target_matrix = obj.matrix_world.copy()
            _last_target_center = center.copy()
        else:
            props.target_x = 0.0
            props.target_y = 0.0
            props.target_z = 0.0

        if getattr(props, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
            update_volumetric_rig(context.scene)
        else:
            update_orbital_rig(context.scene, bake_keyframes=False)
        self.report({'INFO'}, "Target orbit offsets reset to center.")
        return {'FINISHED'}


class ABUCAM_OT_ResetLights(bpy.types.Operator):
    bl_idname = "abucam.reset_lights"
    bl_label = "Reset Lights"
    bl_description = "Reset Key, Fill, Rim, and Ambient lighting settings to defaults"

    def execute(self, context):
        props = context.scene.abucam_props
        light_keys = [
            "use_ambient_light", "ambient_intensity", "ambient_color",
            "key_intensity", "fill_intensity", "rim_intensity"
        ]
        for k in light_keys:
            if k in PARAM_DEFAULTS:
                setattr(props, k, PARAM_DEFAULTS[k])

        on_light_param_update(props, context)
        self.report({'INFO'}, "Studio lighting reset to defaults.")
        return {'FINISHED'}


class ABUCAM_OT_SetupEnvSphere(bpy.types.Operator):
    bl_idname = "abucam.setup_env_sphere"
    bl_label = "Add / Update Environment Sky Sphere"
    bl_description = "Creates or updates the 3D Environment Sky Sphere dome mesh around the scene"

    def execute(self, context):
        props = context.scene.abucam_props
        props.use_env_sphere = True
        obj = update_env_sky_sphere(context.scene)
        abucam_log(f"Environment Sky Sphere configured: radius={props.env_sphere_radius}m, mode={props.env_sphere_mode}", level="INFO", scene=context.scene)
        self.report({'INFO'}, f"Environment Sky Sphere active (Radius: {props.env_sphere_radius:.1f}m, Mode: {props.env_sphere_mode}).")
        return {'FINISHED'}


class ABUCAM_OT_RemoveEnvSphere(bpy.types.Operator):
    bl_idname = "abucam.remove_env_sphere"
    bl_label = "Remove Sky Sphere"
    bl_description = "Removes the AbuCam Environment Sky Sphere from the scene"

    def execute(self, context):
        props = context.scene.abucam_props
        props.use_env_sphere = False
        obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
        if obj:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        self.report({'INFO'}, "Environment Sky Sphere removed from scene.")
        return {'FINISHED'}


class ABUCAM_OT_ResetEnvSphere(bpy.types.Operator):
    bl_idname = "abucam.reset_env_sphere"
    bl_label = "Reset Sky Sphere"
    bl_description = "Reset Environment Sky Sphere parameters to factory defaults"

    def execute(self, context):
        props = context.scene.abucam_props
        env_keys = [
            "env_sphere_radius", "env_sphere_mode", "env_sphere_voronoi_output",
            "env_sphere_voronoi_scale", "env_sphere_voronoi_detail",
            "env_sphere_voronoi_randomness", "env_sphere_projection",
            "env_sphere_texture_path", "env_sphere_emission", "env_sphere_roughness",
            "env_sphere_color", "env_sphere_zenith_color", "env_sphere_ground_color",
            "env_sphere_sun_elevation", "env_sphere_rotation_z", "env_sphere_scale_u",
            "env_sphere_scale_v", "env_sphere_segments", "env_sphere_follow_target",
            "env_sphere_visible_camera", "env_sphere_cast_shadows",
            "env_sphere_visible_diffuse", "env_sphere_visible_glossy",
            "env_sphere_exclude_aux_passes"
        ]
        for k in env_keys:
            if k in PARAM_DEFAULTS:
                setattr(props, k, PARAM_DEFAULTS[k])
        props.env_sphere_image = None
        props.env_sphere_custom_mat = ensure_voronoi_preset_material(props)
        update_env_sky_sphere(context.scene)
        self.report({'INFO'}, "Environment Sky Sphere parameters reset to defaults.")
        return {'FINISHED'}


class ABUCAM_OT_SelectEnvSphere(bpy.types.Operator):
    bl_idname = "abucam.select_env_sphere"
    bl_label = "Select Sky Sphere"
    bl_description = "Selects the AbuCam_EnvSkySphere mesh object so you can inspect its material or nodes in the Shader Editor"

    def execute(self, context):
        obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
        if not obj:
            context.scene.abucam_props.use_env_sphere = True
            obj = update_env_sky_sphere(context.scene)
        if obj:
            try:
                bpy.ops.object.select_all(action='DESELECT')
            except Exception:
                pass
            obj.hide_select = False
            obj.select_set(True)
            context.view_layer.objects.active = obj
            self.report({'INFO'}, "Selected AbuCam_EnvSkySphere.")
            return {'FINISHED'}
        self.report({'WARNING'}, "Sky Sphere not found.")
        return {'CANCELLED'}


class ABUCAM_OT_ResetRenderSettings(bpy.types.Operator):
    bl_idname = "abucam.reset_render_settings"
    bl_label = "Reset Render Settings"
    bl_description = "Reset renderer engine, samples, and pass selections to defaults"

    def execute(self, context):
        props = context.scene.abucam_props
        render_keys = [
            "renderer", "eevee_raytracing", "samples", "pass_rgb", "pass_depth", "pass_normal",
            "pass_wire_rgb", "wire_color_mode", "wire_thickness", "wire_opacity",
            "auto_compile_videos"
        ]
        for k in render_keys:
            if k in PARAM_DEFAULTS:
                setattr(props, k, PARAM_DEFAULTS[k])

        self.report({'INFO'}, "Render settings reset to defaults.")
        return {'FINISHED'}


class ABUCAM_OT_SetupLights(bpy.types.Operator):
    bl_idname = "abucam.setup_lights"
    bl_label = "Setup 3-Point Studio Lighting"
    bl_description = "Creates adaptive Key, Fill, and Rim area lights and studio environment"

    def execute(self, context):
        scene = context.scene
        props = scene.abucam_props
        radius = max(0.5, props.orbit_radius)
        target_x = getattr(props, 'target_x', 0.0)
        target_y = getattr(props, 'target_y', 0.0)
        target_z = props.target_z
        scale = radius / 0.85

        for name in ["KeyLight", "FillLight", "RimLight", "StudioFloor"]:
            old = bpy.data.objects.get(name)
            if old:
                bpy.data.objects.remove(old, do_unlink=True)

        target_obj = props.target_object or bpy.data.objects.get("AbuCam_Target") or bpy.data.objects.get("LowPoly_FemaleHead") or context.active_object

        # Key Light (Warm soft studio area light)
        key_data = bpy.data.lights.new("KeyLight", type='AREA')
        key_data.energy = props.key_intensity * (scale ** 1.5)
        key_data.size = 0.6 * scale
        key_data.color = (1.0, 0.96, 0.92)
        key_obj = bpy.data.objects.new("KeyLight", key_data)
        key_obj.location = (target_x + 0.75 * scale, target_y - 0.90 * scale, target_z + 0.55 * scale)
        scene.collection.objects.link(key_obj)
        if target_obj:
            c = key_obj.constraints.new(type='TRACK_TO')
            c.target = target_obj
            c.track_axis = 'TRACK_NEGATIVE_Z'
            c.up_axis = 'UP_Y'

        # Fill Light (Cool soft fill)
        fill_data = bpy.data.lights.new("FillLight", type='AREA')
        fill_data.energy = props.fill_intensity * (scale ** 1.5)
        fill_data.size = 0.8 * scale
        fill_data.color = (0.90, 0.94, 1.0)
        fill_obj = bpy.data.objects.new("FillLight", fill_data)
        fill_obj.location = (target_x - 0.80 * scale, target_y - 0.70 * scale, target_z + 0.30 * scale)
        scene.collection.objects.link(fill_obj)
        if target_obj:
            c = fill_obj.constraints.new(type='TRACK_TO')
            c.target = target_obj
            c.track_axis = 'TRACK_NEGATIVE_Z'
            c.up_axis = 'UP_Y'

        # Rim Light (Crisp contour separation)
        rim_data = bpy.data.lights.new("RimLight", type='AREA')
        rim_data.energy = props.rim_intensity * (scale ** 1.5)
        rim_data.size = 0.5 * scale
        rim_data.color = (1.0, 0.98, 0.95)
        rim_obj = bpy.data.objects.new("RimLight", rim_data)
        rim_obj.location = (target_x + 0.20 * scale, target_y + 0.95 * scale, target_z + 0.60 * scale)
        scene.collection.objects.link(rim_obj)
        if target_obj:
            c = rim_obj.constraints.new(type='TRACK_TO')
            c.target = target_obj
            c.track_axis = 'TRACK_NEGATIVE_Z'
            c.up_axis = 'UP_Y'

        # World Background: Ambient lighting from all directions
        world = scene.world
        if not world:
            world = bpy.data.worlds.new("StudioWorld")
            scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            if props.use_ambient_light:
                bg.inputs['Strength'].default_value = props.ambient_intensity
                bg.inputs['Color'].default_value = (props.ambient_color[0], props.ambient_color[1], props.ambient_color[2], 1.0)
            else:
                bg.inputs['Color'].default_value = (0.025, 0.028, 0.035, 1.0)
                bg.inputs['Strength'].default_value = 0.6

        self.report({'INFO'}, "Studio 3-point area lighting setup installed.")
        return {'FINISHED'}


class ABUCAM_OT_SetupAmbient(bpy.types.Operator):
    bl_idname = "abucam.setup_ambient"
    bl_label = "Add Soft Ambient Light"
    bl_description = "Enables soft omnidirectional ambient lighting from all directions"

    def execute(self, context):
        props = context.scene.abucam_props
        props.use_ambient_light = True
        on_light_param_update(props, context)
        self.report({'INFO'}, f"Soft ambient lighting enabled (Intensity: {props.ambient_intensity:.2f})")
        return {'FINISHED'}


class ABUCAM_OT_OpenOutputDir(bpy.types.Operator):
    bl_idname = "abucam.open_output_dir"
    bl_label = "Open Output Folder"
    bl_description = "Opens the output directory in Windows File Explorer"

    def execute(self, context):
        props = context.scene.abucam_props
        out_dir = bpy.path.abspath(props.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        target_dir = out_dir
        try:
            subdirs = [os.path.join(out_dir, d) for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d)) and d.startswith("abucam_orbit_")]
            if subdirs:
                subdirs.sort(key=os.path.getmtime, reverse=True)
                target_dir = subdirs[0]
        except Exception:
            target_dir = out_dir

        try:
            if sys.platform == "win32":
                os.startfile(target_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target_dir])
            else:
                subprocess.Popen(["xdg-open", target_dir])
            self.report({'INFO'}, f"Opened output folder: {target_dir}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to open folder: {e}")
            return {'CANCELLED'}


class ABUCAM_OT_OpenLogsFolder(bpy.types.Operator):
    bl_idname = "abucam.open_logs_dir"
    bl_label = "Open Logs Folder"
    bl_description = "Opens the directory containing AbuCam diagnostic and performance logs (abucam.log)"

    def execute(self, context):
        log_dir = get_abucam_log_dir(context.scene)
        try:
            if sys.platform == "win32":
                os.startfile(log_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", log_dir])
            else:
                subprocess.Popen(["xdg-open", log_dir])
            self.report({'INFO'}, f"Opened logs directory: {log_dir}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to open logs folder: {e}")
            return {'CANCELLED'}



class ABUCAM_OT_ViewCamera(bpy.types.Operator):
    bl_idname = "abucam.view_camera"
    bl_label = "Look Through Orbital Camera"
    bl_description = "Align active 3D view to the AbuCam orbital camera"

    def execute(self, context):
        cam = bpy.data.objects.get("OrbitalCamera")
        if cam:
            context.scene.camera = cam
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.region_3d.view_perspective = 'CAMERA'
                            return {'FINISHED'}
        self.report({'WARNING'}, "OrbitalCamera not found. Run 'Setup Rig' first.")
        return {'CANCELLED'}


class ABUCAM_OT_SetupVolumetricRig(bpy.types.Operator):
    bl_idname = "abucam.setup_volumetric_rig"
    bl_label = "Setup Volumetric Camera Rig"
    bl_description = "Generates spherical multi-camera dome array targeting subject"

    def execute(self, context):
        props = context.scene.abucam_props
        props.rig_mode = 'VOLUMETRIC'
        scene = context.scene
        scene.frame_start = 1
        scene.frame_end = max(1, props.total_frames)
        scene.render.fps = max(1, props.fps)
        cams = update_volumetric_rig(scene)
        self.report({'INFO'}, f"Volumetric rig configured: {len(cams)} cameras across {props.total_frames} frames @ {props.fps}fps.")
        return {'FINISHED'}


class ABUCAM_OT_ViewVolumetricCamera(bpy.types.Operator):
    bl_idname = "abucam.view_vol_camera"
    bl_label = "View Active Volumetric Camera"
    bl_description = "Align active 3D view to look through the selected volumetric camera"

    def execute(self, context):
        props = context.scene.abucam_props
        idx = props.vol_active_cam_index
        cam_name = f"AbuCam_Vol_{idx:03d}"
        cam_obj = bpy.data.objects.get(cam_name)
        if not cam_obj:
            self.report({'WARNING'}, f"Camera {cam_name} not found. Generating volumetric rig...")
            cams = update_volumetric_rig(context.scene)
            if cams and idx <= len(cams):
                cam_obj = cams[idx - 1]

        if cam_obj:
            context.scene.camera = cam_obj
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.region_3d.view_perspective = 'CAMERA'
            self.report({'INFO'}, f"Viewing through {cam_obj.name}")
            return {'FINISHED'}

        self.report({'ERROR'}, "No volumetric camera found.")
        return {'CANCELLED'}


class ABUCAM_OT_CycleVolumetricCamera(bpy.types.Operator):
    bl_idname = "abucam.cycle_vol_camera"
    bl_label = "Cycle Active Camera"
    bl_description = "Switch to next or previous camera in the volumetric dome"

    direction: IntProperty(default=1)

    def execute(self, context):
        props = context.scene.abucam_props
        N = max(1, props.vol_camera_count)
        cur = props.vol_active_cam_index
        nxt = ((cur - 1 + self.direction) % N) + 1
        props.vol_active_cam_index = nxt
        return {'FINISHED'}


class ABUCAM_OT_ExportGlb(bpy.types.Operator):
    bl_idname = "abucam.export_glb"
    bl_label = "Export GLB Preview"
    bl_description = "Exports active scene as scene_preview.glb for Three.js and web viewports"

    def execute(self, context):
        props = context.scene.abucam_props
        out_dir = bpy.path.abspath(props.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        glb_path = os.path.join(out_dir, "scene_preview.glb")

        guide = bpy.data.objects.get("OrbitalPathGuide")
        prev_guide_hide = None
        if guide:
            prev_guide_hide = guide.hide_viewport
            guide.hide_viewport = True

        try:
            bpy.ops.export_scene.gltf(
                filepath=glb_path,
                export_format='GLB',
                use_selection=False,
                export_apply=True,
                export_materials='EXPORT',
                export_yup=True
            )
            self.report({'INFO'}, f"Exported GLB preview to: {glb_path}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"GLB export failed: {e}")
            return {'CANCELLED'}
        finally:
            if guide and prev_guide_hide is not None:
                guide.hide_viewport = prev_guide_hide


class ABUCAM_OT_RenderPasses(bpy.types.Operator):
    bl_idname = "abucam.render_passes"
    bl_label = "Render Multi-Pass Orbit Frames"
    bl_description = "Renders RGB, Normal, Depth, and Wireframe passes across all frames with non-blocking modal execution and cancellation"

    _timer = None
    _tasks = []
    _task_idx = 0
    _orig_frame = 1
    _orig_camera = None
    _orig_samples = 32
    _orig_filepath = ""
    _orig_override = None
    _session_dir = ""
    _session_name = ""
    _rgb_dir = ""
    _depth_dir = ""
    _normal_dir = ""
    _wire_dir = ""
    _video_dir = ""
    _root_rgb = ""
    _root_depth = ""
    _root_normal = ""
    _root_wire = ""
    _root_video = ""
    _mat_depth = None
    _mat_normal = None
    _mat_wire = None
    _t_start = 0.0
    _frame_times = []
    _pause_t0 = None
    _total_paused_time = 0.0

    def clean_up(self, context):
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        try:
            context.window_manager.progress_end()
        except Exception:
            pass
        try:
            context.scene.view_layers[0].material_override = self._orig_override
            if context.scene.render.engine == 'CYCLES':
                context.scene.cycles.samples = self._orig_samples
            context.scene.frame_set(self._orig_frame)
            if self._orig_camera:
                context.scene.camera = self._orig_camera
            context.scene.render.filepath = self._orig_filepath
        except Exception:
            pass
        props = getattr(context.scene, 'abucam_props', None)
        if props:
            props.is_paused = False
        for area in context.screen.areas:
            area.tag_redraw()

    def finish_render(self, context):
        props = context.scene.abucam_props
        total = props.total_frames
        self.clean_up(context)
        props.is_rendering = False
        props.is_paused = False
        props.render_time_remaining = 0.0
        props.render_progress = 1.0
        props.render_status_msg = "Render complete! Compiling..."

        # Auto-compile MP4 videos into session directory and root video folder
        compiled_videos = []
        if getattr(props, 'auto_compile_videos', True):
            try:
                ffmpeg_bin = shutil.which("ffmpeg") or r"C:\Users\USer\ffmpeg\bin\ffmpeg.exe"
                if os.path.exists(ffmpeg_bin):
                    passes_cfg = [
                        (self._rgb_dir, "frame_%04d.png", os.path.join(self._video_dir, "reference_orbit.mp4")),
                        (self._depth_dir, "depth_%04d.png", os.path.join(self._video_dir, "depth_orbit.mp4")),
                        (self._normal_dir, "normal_%04d.png", os.path.join(self._video_dir, "normal_orbit.mp4")),
                        (self._wire_dir, "wire_%04d.png", os.path.join(self._video_dir, "wire_orbit.mp4")),
                    ]
                    for s_dir, pat, out_vid in passes_cfg:
                        f0 = os.path.join(s_dir, pat.replace("%04d", "0000"))
                        if os.path.exists(f0):
                            cmd = [
                                ffmpeg_bin, "-y",
                                "-framerate", str(props.fps),
                                "-i", os.path.join(s_dir, pat),
                                "-c:v", "libx264",
                                "-pix_fmt", "yuv420p",
                                out_vid
                            ]
                            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            vname = os.path.basename(out_vid)
                            compiled_videos.append(vname)
                            try:
                                shutil.copy2(out_vid, os.path.join(self._root_video, vname))
                            except Exception:
                                pass
            except Exception as e:
                print(f"[AbuCam] Video compilation notice: {e}")

        # Mirror session pass frames to root output passes folder
        for s_dir, r_dir in [
            (self._rgb_dir, self._root_rgb),
            (self._depth_dir, self._root_depth),
            (self._normal_dir, self._root_normal),
            (self._wire_dir, self._root_wire),
        ]:
            if os.path.exists(s_dir) and os.path.exists(r_dir):
                for f_name in os.listdir(s_dir):
                    if f_name.endswith('.png'):
                        try:
                            shutil.copy2(os.path.join(s_dir, f_name), os.path.join(r_dir, f_name))
                        except Exception:
                            pass

        # Automatically open the newly created session folder in File Explorer
        try:
            if sys.platform == "win32":
                os.startfile(self._session_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self._session_dir])
            else:
                subprocess.Popen(["xdg-open", self._session_dir])
        except Exception as e:
            print(f"[AbuCam] Open folder error: {e}")

        # Interactive popup message
        def draw_render_done(s, ctx):
            s.layout.label(text=f"Multi-Pass Render Complete! ({total} frames)", icon='CHECKMARK')
            s.layout.label(text=f"Folder: {self._session_name}", icon='FILE_FOLDER')
            s.layout.separator()
            if props.pass_rgb:
                s.layout.label(text=f"• RGB: {self._rgb_dir}", icon='IMAGE_DATA')
            if props.pass_depth:
                s.layout.label(text=f"• Depth (16b): {self._depth_dir}", icon='IMAGE_DATA')
            if props.pass_normal:
                s.layout.label(text=f"• Normal: {self._normal_dir}", icon='IMAGE_DATA')
            if props.pass_wire_rgb:
                s.layout.label(text=f"• Wire+RGB ({props.wire_color_mode}): {self._wire_dir}", icon='IMAGE_DATA')
            if compiled_videos:
                s.layout.label(text=f"• Videos: {self._video_dir}", icon='FILE_MOVIE')
            s.layout.separator()
            s.layout.operator("abucam.open_output_dir", text="Open Output Folder in Explorer", icon='FILE_FOLDER')

        context.window_manager.popup_menu(draw_render_done, title="AbuCam Render Finished", icon='INFO')
        self.report({'INFO'}, f"Successfully rendered {total} frames to {self._session_dir}")

    def invoke(self, context, event):
        props = context.scene.abucam_props
        if props.is_rendering:
            self.report({'WARNING'}, "A render is already actively running!")
            return {'CANCELLED'}

        if not (props.pass_rgb or props.pass_depth or props.pass_normal or props.pass_wire_rgb):
            self.report({'ERROR'}, "No passes selected to render!")
            return {'CANCELLED'}

        scene = context.scene
        # Ensure orbital rig has full baked keyframe animation
        update_orbital_rig(scene, bake_keyframes=True)
        out_dir = bpy.path.abspath(props.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_name = f"abucam_orbit_{timestamp}"
        self._session_dir = os.path.join(out_dir, self._session_name)

        self._rgb_dir = os.path.join(self._session_dir, "passes", "rgb")
        self._depth_dir = os.path.join(self._session_dir, "passes", "depth")
        self._normal_dir = os.path.join(self._session_dir, "passes", "normal")
        self._wire_dir = os.path.join(self._session_dir, "passes", "wire_rgb")
        self._video_dir = os.path.join(self._session_dir, "video")
        for d in [self._rgb_dir, self._depth_dir, self._normal_dir, self._wire_dir, self._video_dir]:
            os.makedirs(d, exist_ok=True)

        self._root_rgb = os.path.join(out_dir, "passes", "rgb")
        self._root_depth = os.path.join(out_dir, "passes", "depth")
        self._root_normal = os.path.join(out_dir, "passes", "normal")
        self._root_wire = os.path.join(out_dir, "passes", "wire_rgb")
        self._root_video = os.path.join(out_dir, "video")
        for d in [self._root_rgb, self._root_depth, self._root_normal, self._root_wire, self._root_video]:
            os.makedirs(d, exist_ok=True)

        # Light-speed GPU Accelerated Render Configuration (EEVEE Next / Cycles OptiX)
        configure_gpu_accelerated_render(scene, props, pass_type='RGB')

        # Export camera poses CSV matching RealityCapture / RealityScan schema
        cam_obj = bpy.data.objects.get("OrbitalCamera")
        if cam_obj:
            csv_lines = [REALITY_CAPTURE_CSV_HEADER]
            total_f = props.total_frames
            for i in range(total_f):
                f_num = i + 1
                scene.frame_set(f_num)
                depsgraph = context.evaluated_depsgraph_get()
                cam_eval = cam_obj.evaluated_get(depsgraph)
                c2w = cam_eval.matrix_world.copy()
                name = f"frame_{i:04d}.png"
                csv_lines.append(format_reality_capture_csv_line(name, c2w, props.focal_length, sensor_width=36.0))
            
            try:
                with open(os.path.join(self._session_dir, "cameras.csv"), "w", encoding="utf-8") as f:
                    f.writelines(csv_lines)
                with open(os.path.join(out_dir, "cameras.csv"), "w", encoding="utf-8") as f:
                    f.writelines(csv_lines)
            except Exception as e:
                print(f"[AbuCam] Error writing orbit cameras.csv: {e}")

        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'

        half_span = max(0.35, props.orbit_radius * 0.45)
        near_val = max(0.05, props.orbit_radius - half_span)
        far_val = props.orbit_radius + half_span
        self._mat_depth = get_or_create_depth_material(near=near_val, far=far_val)
        self._mat_normal = get_or_create_normal_material()
        self._mat_wire = get_or_create_wireframe_material(thickness=props.wire_thickness)

        self._orig_frame = scene.frame_current
        self._orig_camera = scene.camera
        self._orig_override = scene.view_layers[0].material_override
        self._orig_filepath = scene.render.filepath
        if scene.render.engine == 'CYCLES':
            self._orig_samples = scene.cycles.samples

        # Build task list
        self._tasks = []
        total = props.total_frames
        for i in range(total):
            frame = i + 1
            if props.pass_rgb or props.pass_wire_rgb:
                self._tasks.append(('RGB', i, frame))
            if props.pass_depth:
                self._tasks.append(('DEPTH', i, frame))
            if props.pass_normal:
                self._tasks.append(('NORMAL', i, frame))
            if props.pass_wire_rgb:
                self._tasks.append(('WIRE', i, frame))

        self._task_idx = 0
        self._t_start = time.perf_counter()
        self._frame_times = []
        self._pause_t0 = None
        self._total_paused_time = 0.0

        props.is_rendering = True
        props.is_paused = False
        props.cancel_render = False
        props.render_progress = 0.0
        props.render_frame_current = 0
        props.render_frame_total = total
        props.render_task_current = 0
        props.render_task_total = len(self._tasks)
        props.render_current_pass = ""
        props.render_last_frame_time = 0.0
        props.render_avg_frame_time = 0.0
        props.render_time_elapsed = 0.0
        props.render_time_remaining = 0.0
        props.render_status_msg = f"Starting render (0 / {len(self._tasks)} tasks)..."

        # Automatically pop up Render Preview Window if enabled
        if getattr(props, 'show_render_preview_window', True):
            try:
                bpy.ops.render.view_show('INVOKE_DEFAULT')
            except Exception:
                pass

        context.window_manager.progress_begin(0, len(self._tasks))
        # 40ms timer frequency yields to Windows OS message loop between frames, preventing semi-hang
        self._timer = context.window_manager.event_timer_add(0.04, window=context.window)
        context.window_manager.modal_handler_add(self)

        for area in context.screen.areas:
            area.tag_redraw()

        abucam_log(f"Modal multi-pass render started: {len(self._tasks)} tasks across {total} frames to {self._session_dir}", level="INFO", scene=scene)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        props = context.scene.abucam_props

        # Catch ESC or Stop Button cancel request
        if event.type == 'ESC' or props.cancel_render:
            self.clean_up(context)
            props.is_rendering = False
            props.is_paused = False
            props.cancel_render = False
            props.render_status_msg = f"Cancelled at task {self._task_idx}/{len(self._tasks)}"
            abucam_log(f"Render stopped by user at task {self._task_idx}/{len(self._tasks)}", level="WARNING", scene=context.scene)
            self.report({'WARNING'}, f"Render stopped by user at task {self._task_idx}/{len(self._tasks)}")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # Handle Pause state cleanly
            if props.is_paused:
                if self._pause_t0 is None:
                    self._pause_t0 = time.perf_counter()
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for reg in area.regions:
                            if reg.type == 'UI':
                                reg.tag_redraw()
                return {'PASS_THROUGH'}
            else:
                if self._pause_t0 is not None:
                    self._total_paused_time += (time.perf_counter() - self._pause_t0)
                    self._pause_t0 = None

            if self._task_idx >= len(self._tasks):
                abucam_log(f"Multi-pass render completed: {len(self._tasks)} tasks saved to {self._session_dir}", level="INFO", scene=context.scene)
                self.finish_render(context)
                return {'FINISHED'}

            task_type, i, frame = self._tasks[self._task_idx]
            scene = context.scene
            vl = scene.view_layers[0]
            scene.frame_set(frame)
            task_t0 = time.perf_counter()

            try:
                # Ensure GPU acceleration parameters (OptiX / EEVEE Next rasterization & adaptive samples)
                configure_gpu_accelerated_render(scene, props, pass_type=task_type)

                env_obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
                if env_obj and getattr(props, 'use_env_sphere', False):
                    if task_type in ('DEPTH', 'NORMAL', 'WIRE') and getattr(props, 'env_sphere_exclude_aux_passes', True):
                        env_obj.hide_render = True
                    else:
                        env_obj.hide_render = False

                if task_type == 'RGB':
                    vl.material_override = None
                    scene.render.image_settings.file_format = 'PNG'
                    scene.render.image_settings.color_depth = '8'
                    rgb_base = os.path.join(self._rgb_dir, f"frame_{i:04d}")
                    scene.render.filepath = rgb_base
                    bpy.ops.render.render(write_still=True)

                elif task_type == 'DEPTH':
                    vl.material_override = self._mat_depth
                    scene.render.image_settings.file_format = 'PNG'
                    scene.render.image_settings.color_depth = '16'
                    depth_base = os.path.join(self._depth_dir, f"depth_{i:04d}")
                    scene.render.filepath = depth_base
                    bpy.ops.render.render(write_still=True)

                elif task_type == 'NORMAL':
                    vl.material_override = self._mat_normal
                    scene.render.image_settings.file_format = 'PNG'
                    scene.render.image_settings.color_depth = '8'
                    normal_base = os.path.join(self._normal_dir, f"normal_{i:04d}")
                    scene.render.filepath = normal_base
                    bpy.ops.render.render(write_still=True)

                elif task_type == 'WIRE':
                    rgb_file = os.path.join(self._rgb_dir, f"frame_{i:04d}.png")
                    vl.material_override = self._mat_wire
                    scene.render.image_settings.file_format = 'PNG'
                    scene.render.image_settings.color_depth = '8'
                    tmp_mask_base = os.path.join(self._wire_dir, f"temp_mask_{i:04d}")
                    scene.render.filepath = tmp_mask_base
                    bpy.ops.render.render(write_still=True)

                    tmp_mask_file = f"{tmp_mask_base}.png"
                    out_wire_file = os.path.join(self._wire_dir, f"wire_{i:04d}.png")
                    composite_wireframe_on_rgb(
                        rgb_path=rgb_file,
                        wire_mask_path=tmp_mask_file,
                        out_path=out_wire_file,
                        color_mode=props.wire_color_mode,
                        opacity=props.wire_opacity
                    )
                    if os.path.exists(tmp_mask_file):
                        try:
                            os.remove(tmp_mask_file)
                        except Exception:
                            pass
            except Exception as e:
                err_msg = f"Error during render task {task_type} frame {frame}: {e}"
                print(f"[AbuCam] {err_msg}")
                abucam_log(err_msg, level="ERROR", scene=scene)
            finally:
                env_obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
                if env_obj and getattr(props, 'use_env_sphere', False):
                    env_obj.hide_render = False

            task_dt = time.perf_counter() - task_t0
            self._frame_times.append(task_dt)
            abucam_log(f"Frame {frame} ({task_type}) rendered in {task_dt:.3f}s", level="DEBUG", scene=scene)

            self._task_idx += 1
            total_tasks = len(self._tasks)
            props.render_task_current = self._task_idx
            props.render_task_total = total_tasks
            props.render_frame_current = frame
            props.render_frame_total = props.total_frames
            props.render_current_pass = task_type
            props.render_last_frame_time = task_dt

            avg_dt = sum(self._frame_times) / max(1, len(self._frame_times))
            props.render_avg_frame_time = avg_dt

            elapsed = max(0.0, time.perf_counter() - self._t_start - self._total_paused_time)
            props.render_time_elapsed = elapsed

            remaining_tasks = total_tasks - self._task_idx
            props.render_time_remaining = remaining_tasks * avg_dt

            props.render_progress = float(self._task_idx) / float(total_tasks)
            props.render_status_msg = f"Frame {frame}/{props.total_frames} ({task_type}) — Last: {task_dt:.2f}s | ETA: {format_duration(props.render_time_remaining)}"
            context.window_manager.progress_update(self._task_idx)

            # Signal Render Result update so open render windows refresh frame immediately
            try:
                for img in bpy.data.images:
                    if img.type == 'RENDER_RESULT':
                        img.update()
            except Exception:
                pass

            # Targeted UI-only redraw avoids stalling the GPU with 3D viewport re-rasterization
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for reg in area.regions:
                        if reg.type == 'UI':
                            reg.tag_redraw()

        return {'PASS_THROUGH'}

    def execute(self, context):
        return self.invoke(context, None)


class ABUCAM_OT_RenderVolumetric(bpy.types.Operator):
    bl_idname = "abucam.render_volumetric"
    bl_label = "Render Volumetric Multi-Camera Frames"
    bl_description = "Renders all volumetric cameras frame-by-frame into structured folders with non-blocking modal execution and cancellation"

    _timer = None
    _tasks = []
    _task_idx = 0
    _orig_frame = 1
    _orig_camera = None
    _orig_samples = 32
    _orig_filepath = ""
    _orig_override = None
    _session_dir = ""
    _session_name = ""
    _mat_depth = None
    _mat_normal = None
    _mat_wire = None
    _transforms_by_frame = {}
    _csv_by_frame = {}
    _all_csv_rows = []
    _total_timeline_frames = 1
    _num_cams = 0
    _total_operations = 0
    _t_start = 0.0
    _frame_times = []
    _pause_t0 = None
    _total_paused_time = 0.0

    def clean_up(self, context):
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        try:
            context.window_manager.progress_end()
        except Exception:
            pass
        try:
            context.scene.view_layers[0].material_override = self._orig_override
            if context.scene.render.engine == 'CYCLES':
                context.scene.cycles.samples = self._orig_samples
            context.scene.frame_set(self._orig_frame)
            if self._orig_camera:
                context.scene.camera = self._orig_camera
            context.scene.render.filepath = self._orig_filepath
        except Exception:
            pass
        props = getattr(context.scene, 'abucam_props', None)
        if props:
            props.is_paused = False
        for area in context.screen.areas:
            area.tag_redraw()

    def finish_render(self, context):
        props = context.scene.abucam_props
        self.clean_up(context)
        props.is_rendering = False
        props.is_paused = False
        props.render_time_remaining = 0.0
        props.render_progress = 1.0
        props.render_status_msg = "Volumetric render complete!"

        # Write full-session cameras.csv matching RealityCapture / RealityScan schema
        if self._all_csv_rows:
            try:
                with open(os.path.join(self._session_dir, "cameras.csv"), "w", encoding="utf-8") as f_csv:
                    f_csv.write(REALITY_CAPTURE_CSV_HEADER)
                    f_csv.writelines(self._all_csv_rows)
                out_dir = bpy.path.abspath(props.output_dir)
                with open(os.path.join(out_dir, "cameras.csv"), "w", encoding="utf-8") as f_csv:
                    f_csv.write(REALITY_CAPTURE_CSV_HEADER)
                    f_csv.writelines(self._all_csv_rows)
            except Exception as e:
                print(f"[AbuCam] Error writing volumetric session cameras.csv: {e}")

        # Open session folder
        try:
            if sys.platform == "win32":
                os.startfile(self._session_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self._session_dir])
            else:
                subprocess.Popen(["xdg-open", self._session_dir])
        except Exception as e:
            print(f"[AbuCam] Open folder error: {e}")

        # Interactive popup
        def draw_vol_render_done(s, ctx):
            s.layout.label(text="Volumetric Multi-Camera Render Complete!", icon='CHECKMARK')
            s.layout.label(text=f"{self._total_timeline_frames} frame(s) × {self._num_cams} cameras = {self._total_operations} renders", icon='INFO')
            s.layout.label(text=f"Folder: {self._session_name}", icon='FILE_FOLDER')
            s.layout.separator()
            s.layout.label(text="Structure: frame_0001/cam_001.png... + transforms.json", icon='FILE_TEXT')
            s.layout.separator()
            s.layout.operator("abucam.open_output_dir", text="Open Output Folder in Explorer", icon='FILE_FOLDER')

        context.window_manager.popup_menu(draw_vol_render_done, title="AbuCam Volumetric Render Finished", icon='INFO')
        self.report({'INFO'}, f"Successfully rendered {self._total_operations} images across {self._total_timeline_frames} frame(s) to {self._session_dir}")

    def invoke(self, context, event):
        scene = context.scene
        props = scene.abucam_props
        if props.is_rendering:
            self.report({'WARNING'}, "A render is already actively running!")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(props.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        vol_cams = update_volumetric_rig(scene)
        if not vol_cams:
            self.report({'ERROR'}, "No volumetric cameras found in rig. Click 'Setup Volumetric Camera Rig' first.")
            return {'CANCELLED'}

        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_name = f"abucam_volumetric_{timestamp}"
        self._session_dir = os.path.join(out_dir, self._session_name)
        os.makedirs(self._session_dir, exist_ok=True)

        # Light-speed GPU Accelerated Render Configuration (EEVEE Next / Cycles OptiX)
        configure_gpu_accelerated_render(scene, props, pass_type='RGB')

        self._csv_by_frame = {}
        self._all_csv_rows = []

        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'

        half_span = max(0.35, props.vol_sphere_radius * 0.45)
        near_val = max(0.05, props.vol_sphere_radius - half_span)
        far_val = props.vol_sphere_radius + half_span
        self._mat_depth = get_or_create_depth_material(near=near_val, far=far_val)
        self._mat_normal = get_or_create_normal_material()
        self._mat_wire = get_or_create_wireframe_material(thickness=props.wire_thickness)

        scene.frame_start = 1
        scene.frame_end = max(1, props.total_frames)
        scene.render.fps = max(1, props.fps)
        start_frame = 1
        end_frame = max(1, props.total_frames)
        self._total_timeline_frames = max(1, props.total_frames)
        self._num_cams = len(vol_cams)
        self._total_operations = self._total_timeline_frames * self._num_cams

        self._orig_cam = scene.camera
        self._orig_frame = scene.frame_current
        self._orig_override = scene.view_layers[0].material_override
        self._orig_filepath = scene.render.filepath
        if scene.render.engine == 'CYCLES':
            self._orig_samples = scene.cycles.samples

        fl_x = (props.vol_focal_length / props.vol_sensor_width) * props.resolution_x
        fl_y = (props.vol_focal_length / props.vol_sensor_height) * props.resolution_y
        cx = props.resolution_x / 2.0
        cy = props.resolution_y / 2.0
        camera_angle_x = 2.0 * math.atan((props.vol_sensor_width / 2.0) / props.vol_focal_length)

        self._transforms_by_frame = {}
        self._tasks = []

        for f in range(start_frame, end_frame + 1):
            frame_folder_name = f"frame_{f:04d}"
            frame_dir = os.path.join(self._session_dir, frame_folder_name)
            os.makedirs(frame_dir, exist_ok=True)

            self._transforms_by_frame[f] = {
                "camera_angle_x": camera_angle_x,
                "camera_angle_y": 2.0 * math.atan((props.vol_sensor_height / 2.0) / props.vol_focal_length),
                "fl_x": fl_x, "fl_y": fl_y,
                "cx": cx, "cy": cy,
                "w": props.resolution_x, "h": props.resolution_y,
                "frames": []
            }

            for cam_idx, cam_obj in enumerate(vol_cams, start=1):
                cam_name = f"cam_{cam_idx:03d}"
                if props.pass_rgb or props.pass_wire_rgb:
                    self._tasks.append(('RGB', f, cam_idx, cam_obj, cam_name, frame_dir))
                if props.pass_depth:
                    self._tasks.append(('DEPTH', f, cam_idx, cam_obj, cam_name, frame_dir))
                if props.pass_normal:
                    self._tasks.append(('NORMAL', f, cam_idx, cam_obj, cam_name, frame_dir))
                if props.pass_wire_rgb:
                    self._tasks.append(('WIRE', f, cam_idx, cam_obj, cam_name, frame_dir))
                # Camera transform metadata
                is_last_cam = (cam_idx == len(vol_cams))
                self._tasks.append(('META', f, cam_idx, cam_obj, cam_name, frame_dir, is_last_cam))

        self._task_idx = 0
        self._t_start = time.perf_counter()
        self._frame_times = []
        self._pause_t0 = None
        self._total_paused_time = 0.0

        props.is_rendering = True
        props.is_paused = False
        props.cancel_render = False
        props.render_progress = 0.0
        props.render_frame_current = 0
        props.render_frame_total = self._total_timeline_frames
        props.render_task_current = 0
        props.render_task_total = len(self._tasks)
        props.render_current_pass = ""
        props.render_last_frame_time = 0.0
        props.render_avg_frame_time = 0.0
        props.render_time_elapsed = 0.0
        props.render_time_remaining = 0.0
        props.render_status_msg = f"Starting volumetric render: {len(self._tasks)} tasks across {self._total_timeline_frames} frame(s)..."

        # Automatically pop up Render Preview Window if enabled
        if getattr(props, 'show_render_preview_window', True):
            try:
                bpy.ops.render.view_show('INVOKE_DEFAULT')
            except Exception:
                pass

        context.window_manager.progress_begin(0, len(self._tasks))
        # 40ms timer frequency yields to Windows OS message loop between frames, preventing semi-hang
        self._timer = context.window_manager.event_timer_add(0.04, window=context.window)
        context.window_manager.modal_handler_add(self)

        for area in context.screen.areas:
            area.tag_redraw()

        abucam_log(f"Modal volumetric render started: {len(self._tasks)} tasks across {self._total_timeline_frames} frame(s) to {self._session_dir}", level="INFO", scene=scene)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        props = context.scene.abucam_props

        if event.type == 'ESC' or props.cancel_render:
            self.clean_up(context)
            props.is_rendering = False
            props.is_paused = False
            props.cancel_render = False
            props.render_status_msg = f"Volumetric render cancelled at task {self._task_idx}/{len(self._tasks)}"
            abucam_log(f"Volumetric render stopped by user at task {self._task_idx}/{len(self._tasks)}", level="WARNING", scene=context.scene)
            self.report({'WARNING'}, f"Volumetric render stopped by user at task {self._task_idx}/{len(self._tasks)}")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # Handle Pause state cleanly
            if props.is_paused:
                if self._pause_t0 is None:
                    self._pause_t0 = time.perf_counter()
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for reg in area.regions:
                            if reg.type == 'UI':
                                reg.tag_redraw()
                return {'PASS_THROUGH'}
            else:
                if self._pause_t0 is not None:
                    self._total_paused_time += (time.perf_counter() - self._pause_t0)
                    self._pause_t0 = None

            if self._task_idx >= len(self._tasks):
                abucam_log(f"Volumetric render completed: {self._total_operations} images across {self._total_timeline_frames} frame(s) in {self._session_name}", level="INFO", scene=context.scene)
                self.finish_render(context)
                return {'FINISHED'}

            task = self._tasks[self._task_idx]
            task_type = task[0]
            f = task[1]
            cam_idx = task[2]
            cam_obj = task[3]
            cam_name = task[4]
            frame_dir = task[5]

            scene = context.scene
            vl = scene.view_layers[0]
            task_t0 = time.perf_counter()

            if task_type == 'META':
                is_last_cam = task[6]
                depsgraph = context.evaluated_depsgraph_get()
                cam_eval = cam_obj.evaluated_get(depsgraph)
                c2w = cam_eval.matrix_world.copy()
                nerf_matrix = [
                    [float(c2w[0][0]), float(-c2w[0][1]), float(-c2w[0][2]), float(c2w[0][3])],
                    [float(c2w[1][0]), float(-c2w[1][1]), float(-c2w[1][2]), float(c2w[1][3])],
                    [float(c2w[2][0]), float(-c2w[2][1]), float(-c2w[2][2]), float(c2w[2][3])],
                    [0.0, 0.0, 0.0, 1.0]
                ]
                self._transforms_by_frame[f]["frames"].append({
                    "file_path": f"{cam_name}.png",
                    "transform_matrix": nerf_matrix,
                    "camera_id": cam_idx,
                    "camera_name": cam_obj.name
                })
                # RealityCapture / RealityScan schema camera row
                rc_line = format_reality_capture_csv_line(
                    f"{cam_name}.png", c2w, props.vol_focal_length, sensor_width=props.vol_sensor_width
                )
                if f not in self._csv_by_frame:
                    self._csv_by_frame[f] = []
                self._csv_by_frame[f].append(rc_line)
                self._all_csv_rows.append(rc_line)

                if is_last_cam:
                    with open(os.path.join(frame_dir, "transforms.json"), "w") as tf:
                        json.dump(self._transforms_by_frame[f], tf, indent=2)
                    try:
                        with open(os.path.join(frame_dir, "cameras.csv"), "w", encoding="utf-8") as f_csv:
                            f_csv.write(REALITY_CAPTURE_CSV_HEADER)
                            f_csv.writelines(self._csv_by_frame[f])
                    except Exception as e:
                        print(f"[AbuCam] Error writing frame {f} cameras.csv: {e}")

            else:
                scene.frame_set(f)
                scene.camera = cam_obj

                try:
                    # Ensure GPU acceleration parameters (OptiX / EEVEE Next rasterization & adaptive samples)
                    configure_gpu_accelerated_render(scene, props, pass_type=task_type)

                    env_obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
                    if env_obj and getattr(props, 'use_env_sphere', False):
                        if task_type in ('DEPTH', 'NORMAL', 'WIRE') and getattr(props, 'env_sphere_exclude_aux_passes', True):
                            env_obj.hide_render = True
                        else:
                            env_obj.hide_render = False

                    if task_type == 'RGB':
                        vl.material_override = None
                        scene.render.image_settings.file_format = 'PNG'
                        scene.render.image_settings.color_depth = '8'
                        scene.render.filepath = os.path.join(frame_dir, f"{cam_name}.png")
                        bpy.ops.render.render(write_still=True)

                    elif task_type == 'DEPTH':
                        vl.material_override = self._mat_depth
                        scene.render.image_settings.file_format = 'PNG'
                        scene.render.image_settings.color_depth = '16'
                        scene.render.filepath = os.path.join(frame_dir, f"{cam_name}_depth.png")
                        bpy.ops.render.render(write_still=True)

                    elif task_type == 'NORMAL':
                        vl.material_override = self._mat_normal
                        scene.render.image_settings.file_format = 'PNG'
                        scene.render.image_settings.color_depth = '8'
                        scene.render.filepath = os.path.join(frame_dir, f"{cam_name}_normal.png")
                        bpy.ops.render.render(write_still=True)

                    elif task_type == 'WIRE':
                        rgb_file = os.path.join(frame_dir, f"{cam_name}.png")
                        vl.material_override = self._mat_wire
                        scene.render.image_settings.file_format = 'PNG'
                        scene.render.image_settings.color_depth = '8'
                        tmp_mask_base = os.path.join(frame_dir, f"temp_{cam_name}_mask")
                        scene.render.filepath = tmp_mask_base
                        bpy.ops.render.render(write_still=True)

                        tmp_mask_file = f"{tmp_mask_base}.png"
                        out_wire_file = os.path.join(frame_dir, f"{cam_name}_wire.png")
                        composite_wireframe_on_rgb(
                            rgb_path=rgb_file,
                            wire_mask_path=tmp_mask_file,
                            out_path=out_wire_file,
                            color_mode=props.wire_color_mode,
                            opacity=props.wire_opacity
                        )
                        if os.path.exists(tmp_mask_file):
                            try:
                                os.remove(tmp_mask_file)
                            except Exception:
                                pass
                except Exception as e:
                    err_msg = f"Error during volumetric render {cam_name} frame {f}: {e}"
                    print(f"[AbuCam] {err_msg}")
                    abucam_log(err_msg, level="ERROR", scene=scene)
                finally:
                    env_obj = bpy.data.objects.get("AbuCam_EnvSkySphere")
                    if env_obj and getattr(props, 'use_env_sphere', False):
                        env_obj.hide_render = False

                task_dt = time.perf_counter() - task_t0
                self._frame_times.append(task_dt)
                abucam_log(f"Volumetric Frame {f} | {cam_name} ({task_type}) rendered in {task_dt:.3f}s", level="DEBUG", scene=scene)
                props.render_last_frame_time = task_dt
                props.render_current_pass = f"{cam_name} ({task_type})"

            self._task_idx += 1
            total_tasks = len(self._tasks)
            props.render_task_current = self._task_idx
            props.render_task_total = total_tasks
            props.render_frame_current = f
            props.render_frame_total = self._total_timeline_frames

            avg_dt = sum(self._frame_times) / max(1, len(self._frame_times))
            props.render_avg_frame_time = avg_dt

            elapsed = max(0.0, time.perf_counter() - self._t_start - self._total_paused_time)
            props.render_time_elapsed = elapsed

            remaining_tasks = total_tasks - self._task_idx
            props.render_time_remaining = remaining_tasks * avg_dt

            props.render_progress = float(self._task_idx) / float(total_tasks)
            if task_type != 'META':
                props.render_status_msg = f"Frame {f}/{self._total_timeline_frames} | {cam_name} ({task_type}) — Last: {task_dt:.2f}s | ETA: {format_duration(props.render_time_remaining)}"
            context.window_manager.progress_update(self._task_idx)

            # Signal Render Result update so open render windows show new frame immediately
            try:
                for img in bpy.data.images:
                    if img.type == 'RENDER_RESULT':
                        img.update()
            except Exception:
                pass

            # Targeted UI-only redraw avoids stalling the GPU with 3D viewport re-rasterization
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for reg in area.regions:
                        if reg.type == 'UI':
                            reg.tag_redraw()

        return {'PASS_THROUGH'}

    def execute(self, context):
        return self.invoke(context, None)


class ABUCAM_OT_ExportDataset(bpy.types.Operator):
    bl_idname = "abucam.export_dataset"
    bl_label = "Export COLMAP & Camera Poses"
    bl_description = "Calculates exact pinhole camera projection matrices and exports cameras.txt, images.txt, points3D.ply, camera_poses.csv, transforms.json"

    def execute(self, context):
        props = context.scene.abucam_props
        scene = context.scene
        base_out_dir = bpy.path.abspath(props.output_dir)
        os.makedirs(base_out_dir, exist_ok=True)

        # Non-destructive destination: if colmap/sparse already exists, create a new uniquely named directory (colmap_001, colmap_002, etc.)
        target_export_dir = get_unique_colmap_export_dir(base_out_dir)
        meta_dir = os.path.join(target_export_dir, "metadata")
        colmap_dir = os.path.join(target_export_dir, "colmap")
        sparse_dir = os.path.join(target_export_dir, "sparse", "0")
        for d in [target_export_dir, meta_dir, colmap_dir, sparse_dir]:
            os.makedirs(d, exist_ok=True)

        width = props.resolution_x
        height = props.resolution_y
        is_volumetric = (getattr(props, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC')

        if is_volumetric:
            vol_cams = [obj for obj in scene.objects if obj.type == 'CAMERA' and obj.name.startswith(("AbuCam_Vol_", "VolCam_"))]
            if not vol_cams:
                vol_cams = update_volumetric_rig(scene)
            vol_cams = sorted(vol_cams, key=lambda c: c.name)
            if not vol_cams:
                self.report({'ERROR'}, "No volumetric cameras found in rig. Setup rig first.")
                return {'CANCELLED'}

            focal_mm = props.vol_focal_length
            sensor_w = props.vol_sensor_width
            sensor_h = props.vol_sensor_height
        else:
            update_orbital_rig(context.scene, bake_keyframes=True)
            cam_obj = bpy.data.objects.get("OrbitalCamera")
            if not cam_obj:
                self.report({'ERROR'}, "OrbitalCamera not found.")
                return {'CANCELLED'}

            focal_mm = props.focal_length
            sensor_w = 36.0
            sensor_h = 24.0

        fl_x = (focal_mm / sensor_w) * width
        fl_y = (focal_mm / sensor_h) * height
        cx = width / 2.0
        cy = height / 2.0

        rc_csv_lines = []
        colmap_images = []
        transforms_frames = []

        depsgraph = context.evaluated_depsgraph_get()

        if is_volumetric:
            for idx, cam_o in enumerate(vol_cams, 1):
                cam_eval = cam_o.evaluated_get(depsgraph)
                c2w = cam_eval.matrix_world.copy()
                pos = c2w.translation
                w2c = c2w.inverted()
                rot_w2c = w2c.to_3x3()
                q = rot_w2c.to_quaternion()
                t = w2c.translation

                name = f"cam_{idx:03d}.png"
                rc_csv_lines.append(format_reality_capture_csv_line(name, c2w, focal_mm, sensor_width=sensor_w))

                colmap_images.append({
                    "image_id": idx,
                    "qw": q.w, "qx": q.x, "qy": q.y, "qz": q.z,
                    "tx": t.x, "ty": t.y, "tz": t.z,
                    "camera_id": 1,
                    "name": name,
                    "points2d": []
                })

                nerf_matrix = [
                    [float(c2w[0][0]), float(-c2w[0][1]), float(-c2w[0][2]), float(c2w[0][3])],
                    [float(c2w[1][0]), float(-c2w[1][1]), float(-c2w[1][2]), float(c2w[1][3])],
                    [float(c2w[2][0]), float(-c2w[2][1]), float(-c2w[2][2]), float(c2w[2][3])],
                    [0.0, 0.0, 0.0, 1.0]
                ]
                transforms_frames.append({
                    "file_path": name,
                    "transform_matrix": nerf_matrix,
                    "camera_id": idx,
                    "camera_name": cam_o.name
                })
        else:
            total = props.total_frames
            for i in range(total):
                frame = i + 1
                scene.frame_set(frame)
                cam_eval = cam_obj.evaluated_get(depsgraph)
                c2w = cam_eval.matrix_world.copy()

                pos = c2w.translation
                w2c = c2w.inverted()
                rot_w2c = w2c.to_3x3()
                q = rot_w2c.to_quaternion()
                t = w2c.translation

                name = f"frame_{i:04d}.png"
                rc_csv_lines.append(format_reality_capture_csv_line(name, c2w, focal_mm, sensor_width=sensor_w))

                colmap_images.append({
                    "image_id": i + 1,
                    "qw": q.w, "qx": q.x, "qy": q.y, "qz": q.z,
                    "tx": t.x, "ty": t.y, "tz": t.z,
                    "camera_id": 1,
                    "name": name,
                    "points2d": []
                })

                transforms_frames.append({
                    "file_path": f"passes/rgb/{name}",
                    "transform_matrix": [[c2w[r][c] for c in range(4)] for r in range(4)]
                })

        # Save RealityCapture / RealityScan schema CSV files
        for csv_p in [
            os.path.join(target_export_dir, "cameras.csv"),
            os.path.join(meta_dir, "cameras.csv"),
            os.path.join(meta_dir, "camera_poses.csv"),
            os.path.join(colmap_dir, "cameras.csv")
        ]:
            try:
                with open(csv_p, "w", encoding="utf-8") as f_csv:
                    f_csv.write(REALITY_CAPTURE_CSV_HEADER)
                    f_csv.writelines(rc_csv_lines)
            except Exception as e:
                print(f"[AbuCam] Error writing {csv_p}: {e}")

        with open(os.path.join(meta_dir, "transforms.json"), "w", encoding="utf-8") as f:
            json.dump({"fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": width, "h": height, "frames": transforms_frames}, f, indent=2)
        with open(os.path.join(target_export_dir, "transforms.json"), "w", encoding="utf-8") as f:
            json.dump({"fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy, "w": width, "h": height, "frames": transforms_frames}, f, indent=2)

        for target in [colmap_dir, sparse_dir, target_export_dir]:
            with open(os.path.join(target, "cameras.txt"), "w", encoding="utf-8") as f:
                f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                f.write(f"1 PINHOLE {width} {height} {fl_x:.4f} {fl_y:.4f} {cx:.4f} {cy:.4f}\n")

            with open(os.path.join(target, "images.txt"), "w", encoding="utf-8") as f:
                f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[]\n")
                for img in colmap_images:
                    f.write(f"{img['image_id']} {img['qw']:.9f} {img['qx']:.9f} {img['qy']:.9f} {img['qz']:.9f} {img['tx']:.9f} {img['ty']:.9f} {img['tz']:.9f} {img['camera_id']} {img['name']}\n\n")

            # Generate sparse point cloud from actual scene geometry
            ply_file = os.path.join(target, "points3D.ply")
            points = []
            target_obj = bpy.data.objects.get("LowPoly_FemaleHead") or context.active_object
            if target_obj and target_obj.type == 'MESH' and target_obj.name != "AbuCam_EnvSkySphere":
                depsgraph = context.evaluated_depsgraph_get()
                eval_obj = target_obj.evaluated_get(depsgraph)
                mesh_eval = eval_obj.to_mesh()
                mw = target_obj.matrix_world
                for v in mesh_eval.vertices:
                    w_co = mw @ v.co
                    w_no = (mw.to_3x3() @ v.normal).normalized()
                    points.append((w_co.x, w_co.y, w_co.z, w_no.x, w_no.y, w_no.z, 232, 200, 184))
                eval_obj.to_mesh_clear()

            if len(points) == 0:
                # Fallback spherical distribution
                N = 2500
                for p_idx in range(N):
                    u = (p_idx % 50) / 49.0
                    v = (p_idx // 50) / 49.0
                    theta = u * 2.0 * math.pi
                    phi = (v - 0.5) * math.pi
                    px = 0.5 * math.cos(phi) * math.cos(theta)
                    py = 0.5 * math.cos(phi) * math.sin(theta)
                    pz = props.target_z + 0.5 * math.sin(phi)
                    points.append((px, py, pz, 0.0, 0.0, 1.0, 40, 220, 130))

            with open(ply_file, "wb") as f:
                hdr = (
                    f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property float nx\nproperty float ny\nproperty float nz\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                )
                f.write(hdr.encode('ascii'))
                for pt in points:
                    f.write(struct.pack('<6f3B', pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7], pt[8]))

        abucam_log(f"COLMAP dataset & camera poses exported to new folder: {target_export_dir}", level="INFO", scene=scene)
        self.report({'INFO'}, f"COLMAP dataset exported to: {target_export_dir}")
        try:
            if sys.platform == "win32":
                os.startfile(target_export_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target_export_dir])
            else:
                subprocess.Popen(["xdg-open", target_export_dir])
        except Exception:
            pass
        return {'FINISHED'}


class ABUCAM_OT_CompileVideos(bpy.types.Operator):
    bl_idname = "abucam.compile_videos"
    bl_label = "Compile H.264 MP4 Video Passes"
    bl_description = "Assembles rendered frame sequences into high quality reference_orbit.mp4, depth_orbit.mp4, normal_orbit.mp4 using FFmpeg"

    def execute(self, context):
        props = context.scene.abucam_props
        out_dir = bpy.path.abspath(props.output_dir)
        vid_dir = os.path.join(out_dir, "video")
        os.makedirs(vid_dir, exist_ok=True)

        ffmpeg_bin = shutil.which("ffmpeg") or r"C:\Users\USer\ffmpeg\bin\ffmpeg.exe"
        if not os.path.exists(ffmpeg_bin):
            self.report({'ERROR'}, "FFmpeg executable not found in PATH or standard location.")
            return {'CANCELLED'}

        passes = [
            ("passes/rgb", "frame_%04d.png", os.path.join(vid_dir, "reference_orbit.mp4")),
            ("passes/depth", "depth_%04d.png", os.path.join(vid_dir, "depth_orbit.mp4")),
            ("passes/normal", "normal_%04d.png", os.path.join(vid_dir, "normal_orbit.mp4")),
            ("passes/wire_rgb", "wire_%04d.png", os.path.join(vid_dir, "wire_orbit.mp4")),
        ]

        compiled = 0
        for p_dir, pat, out_vid in passes:
            seq_dir = os.path.join(out_dir, p_dir)
            first_frame = os.path.join(seq_dir, pat.replace("%04d", "0000"))
            if not os.path.exists(first_frame):
                continue

            cmd = [
                ffmpeg_bin, "-y",
                "-framerate", str(props.fps),
                "-i", os.path.join(seq_dir, pat),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                out_vid
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0:
                compiled += 1

        self.report({'INFO'}, f"Compiled {compiled} video pass(es) to: {vid_dir}")
        return {'FINISHED'}


class ABUCAM_OT_OneClickBake(bpy.types.Operator):
    bl_idname = "abucam.one_click_bake"
    bl_label = "ONE-CLICK FULL ABUCAM BAKE"
    bl_description = "Executes Rig Setup -> Lighting -> Multi-Pass Render -> COLMAP Export -> Video Encoding -> GLB Export"

    def execute(self, context):
        props = context.scene.abucam_props
        if getattr(props, 'rig_mode', 'ORBITAL') == 'VOLUMETRIC':
            bpy.ops.abucam.setup_volumetric_rig()
            bpy.ops.abucam.setup_lights()
            bpy.ops.abucam.render_volumetric()
        else:
            bpy.ops.abucam.setup_rig()
            bpy.ops.abucam.setup_lights()
            bpy.ops.abucam.render_passes()
            bpy.ops.abucam.export_dataset()
            bpy.ops.abucam.compile_videos()
            bpy.ops.abucam.export_glb()
        self.report({'INFO'}, "Full AbuCam Bake completed successfully!")
        return {'FINISHED'}


# -------------------------------------------------------------------------
# UI Panel
# -------------------------------------------------------------------------

class VIEW3D_PT_AbuCamPanel(bpy.types.Panel):
    bl_label = f"AbuCam Production Suite ({ABUCAM_VERSION})"
    bl_idname = "VIEW3D_PT_abucam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'AbuCam'

    def draw(self, context):
        layout = self.layout
        props = context.scene.abucam_props

        # Active Render Progress & Cancel Box (Sticky at top when rendering)
        if props.is_rendering:
            alert_box = layout.box()
            alert_box.alert = True
            row_stop = alert_box.row(align=True)
            row_stop.scale_y = 1.6
            row_stop.operator("abucam.stop_render", text="STOP / CANCEL RENDERING", icon='CANCEL')
            
            row_info = alert_box.row(align=True)
            row_info.label(text=props.render_status_msg or "Rendering in progress...", icon='TIME')
            
            row_prog = alert_box.row(align=True)
            row_prog.prop(props, "render_progress", text="Progress", slider=True)
            alert_box.label(text="Press ESC on keyboard or click button above to stop.", icon='INFO')
            layout.separator()

        # Top Action Button
        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("abucam.one_click_bake", icon='RENDER_ANIMATION')

        # Main Reset Option (Resets ALL parameters across all sections at the same time)
        row_reset = layout.row(align=True)
        row_reset.scale_y = 1.15
        row_reset.operator("abucam.reset_all", text="Reset All Parameters (Main Reset)", icon='LOOP_BACK')

        # Rig Mode Switcher Tabs
        row_mode = layout.row(align=True)
        row_mode.scale_y = 1.25
        row_mode.prop(props, "rig_mode", expand=True)

        if props.rig_mode == 'ORBITAL':
            # Orbital Camera Rig Box
            box = layout.box()
            row_rh = box.row()
            row_rh.label(text="Orbital Camera Rig", icon='CAMERA_DATA')
            row_rh.operator("abucam.reset_rig", text="Reset Rig", icon='LOOP_BACK')

            # Live Viewport Update Checkbox
            box.prop(props, "live_update", icon='RESTRICT_VIEW_OFF')

            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(props, "azimuth_deg", slider=True)
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "azimuth_deg"

            row = col.row(align=True)
            row.prop(props, "elevation_deg", slider=True)
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "elevation_deg"

            row = col.row(align=True)
            row.prop(props, "path_tilt_deg", slider=True, text="Orbit Path Tilt (-90° to +90°)")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "path_tilt_deg"

            row_m = col.row(align=True)
            row_m.prop(props, "orbit_mode", text="Arc")
            if props.orbit_mode == 'CUSTOM':
                row_arc = col.row(align=True)
                row_arc.prop(props, "arc_start_deg", text="Start", slider=True)
                row_arc.prop(props, "arc_end_deg", text="End", slider=True)

            col.prop(props, "show_path_guide")
            col.separator()
            row = col.row(align=True)
            row.prop(props, "orbit_radius", slider=True, text="Orbit Radius / Diameter (m)")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "orbit_radius"

            # Target Tracking & 3D Center
            box_target = col.box()
            row_th = box_target.row()
            row_th.label(text="Target Tracking & Orbit Center", icon='TRACKING')
            row_th.operator("abucam.reset_target_offsets", text="Reset XYZ", icon='LOOP_BACK')

            row_target = box_target.row(align=True)
            row_target.prop(props, "target_object", text="Target Layer")
            row_target.operator("abucam.center_on_target", text="Center", icon='CURSOR')

            # Active Track & Auto-Follow toggles
            box_target.prop(props, "active_track_target", text="Active Track (Keep Target in Center)", icon='CAMERA_DATA')
            box_target.prop(props, "auto_track_target", text="Auto-Follow Target Movement", icon='CONSTRAINT')

            col_xyz = box_target.column(align=True)
            row_x = col_xyz.row(align=True)
            row_x.prop(props, "target_x", text="Orbit Width (X)")
            row_x.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "target_x"

            row_y = col_xyz.row(align=True)
            row_y.prop(props, "target_y", text="Orbit Depth (Y)")
            row_y.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "target_y"

            row_z = col_xyz.row(align=True)
            row_z.prop(props, "target_z", text="Orbit Height (Z)")
            row_z.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "target_z"

            col.separator()
            row_fl = col.row(align=True)
            row_fl.prop(props, "focal_length")
            row_fl.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "focal_length"

            row_fs = col.row(align=True)
            row_fs.prop(props, "frustum_display_size", slider=True, text="Frustum Scale")
            row_fs.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "frustum_display_size"

            row_tf = col.row(align=True)
            row_tf.prop(props, "total_frames")
            row_tf.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "total_frames"

            row_fps = col.row(align=True)
            row_fps.prop(props, "fps")
            row_fps.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "fps"

            row = box.row(align=True)
            row.prop(props, "resolution_x", text="W")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "resolution_x"
            row.prop(props, "resolution_y", text="H")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "resolution_y"

            row = box.row(align=True)
            row.operator("abucam.setup_rig", icon='OUTLINER_OB_CAMERA')
            row.operator("abucam.view_camera", icon='VIEW_CAMERA')

        else:
            # Volumetric Camera Rig Box
            box = layout.box()
            row_vh = box.row()
            row_vh.label(text="Volumetric Camera Rig", icon='VIEW_PAN')
            row_vh.operator("abucam.reset_volumetric_rig", text="Reset Rig", icon='LOOP_BACK')

            # Live Viewport Update Checkbox
            box.prop(props, "live_update", icon='RESTRICT_VIEW_OFF')

            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(props, "vol_camera_count")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_camera_count"

            row = col.row(align=True)
            row.prop(props, "vol_sphere_radius", slider=True, text="Sphere Radius (m)")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_sphere_radius"

            row = col.row(align=True)
            row.prop(props, "vol_distribution", text="Pattern")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_distribution"

            # 3 Angular Degrees to limit surface section
            box_ang = col.box()
            box_ang.label(text="Surface Sector Limits (3 Angular Degrees)", icon='CONE')

            row_az = box_ang.row(align=True)
            row_az.prop(props, "vol_azimuth_span_deg", slider=True, text="1. Azimuth Span")
            row_az.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_azimuth_span_deg"

            row_hd = box_ang.row(align=True)
            row_hd.prop(props, "vol_azimuth_center_deg", slider=True, text="Heading Offset")
            row_hd.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_azimuth_center_deg"

            row_emin = box_ang.row(align=True)
            row_emin.prop(props, "vol_elevation_min_deg", slider=True, text="2. Elevation Min (Zenith)")
            row_emin.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_elevation_min_deg"

            row_emax = box_ang.row(align=True)
            row_emax.prop(props, "vol_elevation_max_deg", slider=True, text="3. Elevation Max (Floor)")
            row_emax.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_elevation_max_deg"

            # Camera Sensor & Optics
            box_opt = col.box()
            box_opt.label(text="Camera Optics & Sensor Size", icon='CAMERA_DATA')
            row_fl = box_opt.row(align=True)
            row_fl.prop(props, "vol_focal_length")
            row_fl.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_focal_length"

            row_sens = box_opt.row(align=True)
            row_sens.prop(props, "vol_sensor_width", text="Sensor W (mm)")
            row_sens.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_sensor_width"
            row_sens.prop(props, "vol_sensor_height", text="Sensor H (mm)")
            row_sens.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "vol_sensor_height"

            row_vfs = box_opt.row(align=True)
            row_vfs.prop(props, "frustum_display_size", slider=True, text="Frustum Scale")
            row_vfs.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "frustum_display_size"

            # Target Tracking & Rig Center
            box_target = col.box()
            row_th = box_target.row()
            row_th.label(text="Target Tracking & Rig Center", icon='TRACKING')
            row_th.operator("abucam.reset_target_offsets", text="Reset XYZ", icon='LOOP_BACK')

            row_target = box_target.row(align=True)
            row_target.prop(props, "target_object", text="Target Layer")
            row_target.operator("abucam.center_on_target", text="Center", icon='CURSOR')

            box_target.prop(props, "active_track_target", text="Active Track (Keep Target in Center)", icon='CAMERA_DATA')
            box_target.prop(props, "auto_track_target", text="Auto-Follow Target Movement", icon='CONSTRAINT')

            col_xyz = box_target.column(align=True)
            row_x = col_xyz.row(align=True)
            row_x.prop(props, "target_x", text="Center X (Width)")
            row_x.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "target_x"

            row_y = col_xyz.row(align=True)
            row_y.prop(props, "target_y", text="Center Y (Depth)")
            row_y.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "target_y"

            row_z = col_xyz.row(align=True)
            row_z.prop(props, "target_z", text="Center Z (Height)")
            row_z.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "target_z"

            col.prop(props, "vol_show_guide_dome", text="Show 3D Dome Guide Lines")
            if props.vol_show_guide_dome:
                row_gl = col.row(align=True)
                row_gl.prop(props, "vol_show_camera_lines", text="Lines Between Cameras", toggle=True)
                row_gl.prop(props, "vol_show_center_lines", text="Center Target Lines", toggle=True)

            col.separator()
            row_tf = col.row(align=True)
            row_tf.prop(props, "total_frames", text="Total Frames")
            row_tf.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "total_frames"

            row_fps = col.row(align=True)
            row_fps.prop(props, "fps", text="FPS")
            row_fps.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "fps"

            row_view = col.row(align=True)
            row_view.prop(props, "vol_active_cam_index", text="Active Camera")
            op_prev = row_view.operator("abucam.cycle_vol_camera", text="", icon='TRIA_LEFT')
            op_prev.direction = -1
            op_next = row_view.operator("abucam.cycle_vol_camera", text="", icon='TRIA_RIGHT')
            op_next.direction = 1

            row = box.row(align=True)
            row.prop(props, "resolution_x", text="W")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "resolution_x"
            row.prop(props, "resolution_y", text="H")
            row.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "resolution_y"

            row_btns = box.row(align=True)
            row_btns.operator("abucam.setup_volumetric_rig", icon='VIEW_PAN')
            row_btns.operator("abucam.view_vol_camera", icon='VIEW_CAMERA')

            if props.is_rendering:
                row_vrend = box.row(align=True)
                row_vrend.scale_y = 1.35
                if props.is_paused:
                    row_vrend.operator("abucam.toggle_pause_render", text="RESUME RENDER", icon='PLAY')
                else:
                    row_vrend.operator("abucam.toggle_pause_render", text="PAUSE RENDER", icon='PAUSE')
                row_vrend.operator("abucam.stop_render", text="STOP / CANCEL", icon='CANCEL')
            else:
                row_vrend = box.row(align=True)
                row_vrend.scale_y = 1.35
                row_vrend.operator("abucam.render_volumetric", text="Render Volumetric Multi-Camera Frames", icon='RENDER_ANIMATION')

        # Environment & Lighting
        box = layout.box()
        row_lh = box.row()
        row_lh.label(text="Studio Environment & Lighting", icon='LIGHT')
        row_lh.operator("abucam.reset_lights", text="Reset Lights", icon='LOOP_BACK')

        row = box.row(align=True)
        row.operator("abucam.setup_lights", icon='LIGHT_SUN', text="Setup 3-Point Studio Lighting")
        row.operator("abucam.setup_ambient", icon='WORLD', text="Add Soft Ambient Light")

        col = box.column(align=True)
        col.prop(props, "use_ambient_light", text="Soft Ambient Light (All Directions)")
        if props.use_ambient_light:
            row_amb = col.row(align=True)
            row_amb.prop(props, "ambient_intensity", text="Ambient Intensity", slider=True)
            row_amb.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "ambient_intensity"
            row_amb.prop(props, "ambient_color", text="")

        col.separator()
        col.label(text="Light Intensities (Direct Control):", icon='LIGHT_DATA')
        row_k = col.row(align=True)
        row_k.prop(props, "key_intensity", text="Key Light (W)", slider=True)
        row_k.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "key_intensity"

        row_f = col.row(align=True)
        row_f.prop(props, "fill_intensity", text="Fill Light (W)", slider=True)
        row_f.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "fill_intensity"

        row_r = col.row(align=True)
        row_r.prop(props, "rim_intensity", text="Rim Light (W)", slider=True)
        row_r.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "rim_intensity"

        # Environment Sky Sphere (Pro Feature)
        box_env = box.box()
        row_eh = box_env.row(align=True)
        row_eh.prop(props, "use_env_sphere", text="Environment Sky Sphere (Pro)", icon='SHADING_RENDERED')
        row_eh.operator("abucam.reset_env_sphere", text="", icon='LOOP_BACK')
        if bpy.data.objects.get("AbuCam_EnvSkySphere"):
            row_eh.operator("abucam.select_env_sphere", text="", icon='RESTRICT_SELECT_OFF')
            row_eh.operator("abucam.remove_env_sphere", text="", icon='TRASH')

        row_ebtn = box_env.row(align=True)
        row_ebtn.operator("abucam.setup_env_sphere", text="Add / Update Sky Sphere", icon='MESH_UVSPHERE')

        if props.use_env_sphere:
            col_e = box_env.column(align=True)

            row_rad = col_e.row(align=True)
            row_rad.prop(props, "env_sphere_radius", slider=True, text="Sphere Radius (m)")
            row_rad.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_radius"

            row_mode = col_e.row(align=True)
            row_mode.prop(props, "env_sphere_mode", text="Mode")
            row_mode.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_mode"

            if props.env_sphere_mode == 'VORONOI':
                row_vo = col_e.row(align=True)
                row_vo.prop(props, "env_sphere_voronoi_output", text="Style")
                row_vo.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_voronoi_output"

                row_vs = col_e.row(align=True)
                row_vs.prop(props, "env_sphere_voronoi_scale", slider=True, text="Voronoi Scale")
                row_vs.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_voronoi_scale"

                row_vd = col_e.row(align=True)
                row_vd.prop(props, "env_sphere_voronoi_detail", slider=True, text="Detail")
                row_vd.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_voronoi_detail"

                row_vr = col_e.row(align=True)
                row_vr.prop(props, "env_sphere_voronoi_randomness", slider=True, text="Randomness")
                row_vr.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_voronoi_randomness"

                if props.env_sphere_voronoi_output in ('DISTANCE', 'EDGES'):
                    row_vc = col_e.row(align=True)
                    row_vc.prop(props, "env_sphere_color", text="Tint Color")
                    row_vc.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_color"

            elif props.env_sphere_mode == 'HDRI_TEX':
                row_proj = col_e.row(align=True)
                row_proj.prop(props, "env_sphere_projection", text="Map")
                col_e.prop(props, "env_sphere_texture_path", text="File")
                col_e.prop(props, "env_sphere_image", text="Image")
                row_tc = col_e.row(align=True)
                row_tc.prop(props, "env_sphere_color", text="Tint Color")
                row_tc.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_color"

            elif props.env_sphere_mode == 'PROCEDURAL_SKY':
                row_sun = col_e.row(align=True)
                row_sun.prop(props, "env_sphere_sun_elevation", slider=True, text="Sun Elevation")
                row_sun.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_sun_elevation"

            elif props.env_sphere_mode == 'GRADIENT':
                row_g1 = col_e.row(align=True)
                row_g1.prop(props, "env_sphere_zenith_color", text="Zenith")
                row_g1.prop(props, "env_sphere_color", text="Horizon")
                row_g1.prop(props, "env_sphere_ground_color", text="Ground")

            elif props.env_sphere_mode == 'SOLID':
                row_sc = col_e.row(align=True)
                row_sc.prop(props, "env_sphere_color", text="Backdrop Color")
                row_sc.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_color"

            elif props.env_sphere_mode == 'CUSTOM_MAT':
                col_e.prop(props, "env_sphere_custom_mat", text="Material")

            col_e.separator()
            row_em = col_e.row(align=True)
            row_em.prop(props, "env_sphere_emission", slider=True, text="Emission / Exposure")
            row_em.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_emission"

            row_rg = col_e.row(align=True)
            row_rg.prop(props, "env_sphere_roughness", slider=True, text="Surface Roughness")
            row_rg.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_roughness"

            row_rot = col_e.row(align=True)
            row_rot.prop(props, "env_sphere_rotation_z", slider=True, text="Z-Rotation (Heading)")
            row_rot.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_rotation_z"

            row_uv = col_e.row(align=True)
            row_uv.prop(props, "env_sphere_scale_u", text="Scale U")
            row_uv.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_scale_u"
            row_uv.prop(props, "env_sphere_scale_v", text="Scale V")
            row_uv.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_scale_v"

            row_seg = col_e.row(align=True)
            row_seg.prop(props, "env_sphere_segments", text="Mesh Segments")
            row_seg.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "env_sphere_segments"

            col_e.prop(props, "env_sphere_follow_target", text="Lock Center to Target Rig (X, Y, Z)")

            box_ray = col_e.box()
            box_ray.label(text="Ray Visibility & Pass Masking:", icon='HIDE_OFF')
            row_rv1 = box_ray.row(align=True)
            row_rv1.prop(props, "env_sphere_visible_camera", text="Camera")
            row_rv1.prop(props, "env_sphere_visible_diffuse", text="Diffuse GI")
            row_rv2 = box_ray.row(align=True)
            row_rv2.prop(props, "env_sphere_visible_glossy", text="Reflections")
            row_rv2.prop(props, "env_sphere_cast_shadows", text="Cast Shadows")
            box_ray.prop(props, "env_sphere_exclude_aux_passes", text="Auto-Hide in Depth / Normal / Wire Passes")

        # Multi-Pass Render Box
        box = layout.box()
        row_mh = box.row()
        row_mh.label(text="Multi-Pass Reference Engine", icon='SCENE')
        row_mh.operator("abucam.reset_render_settings", text="Reset Render", icon='LOOP_BACK')

        col = box.column(align=True)
        col.prop(props, "renderer")
        row_s = col.row(align=True)
        row_s.prop(props, "samples", text="Render Samples (GPU)")
        row_s.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "samples"

        if props.renderer in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
            row_rt = col.row(align=True)
            row_rt.prop(props, "eevee_raytracing", text="Screen Raytracing (Slower)")
            row_rt.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "eevee_raytracing"

        box_gpu = col.box()
        row_gpu = box_gpu.row(align=True)
        if props.renderer == 'CYCLES':
            row_gpu.label(text="GPU Acceleration: Active (RTX 4090 | OptiX Compute)", icon='AUTO')
        else:
            row_gpu.label(text="GPU Acceleration: Active (RTX 4090 | EEVEE Rasterizer)", icon='AUTO')

        row = box.row(align=True)
        row.prop(props, "pass_rgb", text="RGB")
        row.prop(props, "pass_depth", text="Depth (16b)")
        row.prop(props, "pass_normal", text="Normal")
        row.prop(props, "pass_wire_rgb", text="Wire + RGB")

        if props.pass_wire_rgb:
            box_wire = box.box()
            box_wire.label(text="Wireframe Overlay Settings", icon='SNAP_EDGE')

            row_wcol = box_wire.row(align=True)
            row_wcol.prop(props, "wire_color_mode", text="Wire Color")
            row_wcol.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "wire_color_mode"

            row_wth = box_wire.row(align=True)
            row_wth.prop(props, "wire_thickness", slider=True, text="Line Width (px)")
            row_wth.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "wire_thickness"

            row_wop = box_wire.row(align=True)
            row_wop.prop(props, "wire_opacity", slider=True, text="Opacity")
            row_wop.operator("abucam.reset_param", text="", icon='LOOP_BACK').param = "wire_opacity"

        col = box.column(align=True)
        col.prop(props, "auto_compile_videos", text="Auto-Compile MP4 Videos After Render")
        col.prop(props, "show_render_preview_window", text="Auto-Open Preview Window on Render")

        if props.is_rendering:
            box_act = box.box()
            
            # Header with Pause / Active status
            if props.is_paused:
                box_act.alert = True
                row_h = box_act.row(align=True)
                row_h.label(text="⏸️ RENDER PAUSED", icon='PAUSE')
            else:
                box_act.alert = False
                row_h = box_act.row(align=True)
                row_h.label(text="🔴 RENDERING IN PROGRESS", icon='RENDER_STILL')

            # Progress Bar & Percentage
            box_act.prop(props, "render_progress", text="Progress", slider=True)

            # Detailed Frame & Task Counter
            col_info = box_act.column(align=True)
            row_f = col_info.row(align=True)
            if props.rig_mode == 'VOLUMETRIC':
                row_f.label(text=f"Frame: {props.render_frame_current} / {props.render_frame_total}", icon='CAMERA_DATA')
                row_f.label(text=f"Task: {props.render_task_current} / {props.render_task_total} ({props.render_current_pass})")
            else:
                row_f.label(text=f"Frame: {props.render_frame_current} / {props.render_frame_total}", icon='CAMERA_DATA')
                row_f.label(text=f"Pass: {props.render_current_pass} ({props.render_task_current}/{props.render_task_total})")

            # Timing & Performance Metrics
            box_metrics = box_act.box()
            col_m = box_metrics.column(align=True)
            
            row_m1 = col_m.row(align=True)
            row_m1.label(text=f"Last: {props.render_last_frame_time:.2f}s", icon='TIME')
            row_m1.label(text=f"Avg: {props.render_avg_frame_time:.2f}s", icon='SORTTIME')
            
            row_m2 = col_m.row(align=True)
            row_m2.label(text=f"Elapsed: {format_duration(props.render_time_elapsed)}", icon='PREVIEW_RANGE')
            row_m2.label(text=f"ETA: {format_duration(props.render_time_remaining)}", icon='FORWARD')

            # Action Controls (Pause / Resume / Stop / View Window)
            col_ctrl = box_act.column(align=True)
            row_btn = col_ctrl.row(align=True)
            row_btn.scale_y = 1.3
            if props.is_paused:
                row_btn.operator("abucam.toggle_pause_render", text="RESUME RENDER", icon='PLAY')
            else:
                row_btn.operator("abucam.toggle_pause_render", text="PAUSE RENDER", icon='PAUSE')
            row_btn.operator("abucam.stop_render", text="STOP / CANCEL", icon='CANCEL')

            row_win = col_ctrl.row(align=True)
            row_win.operator("abucam.show_render_window", text="Open Render Preview Window", icon='WINDOW')
        else:
            if props.rig_mode == 'VOLUMETRIC':
                box.operator("abucam.render_volumetric", icon='OUTPUT', text="Render Volumetric Multi-Camera Frames")
            else:
                box.operator("abucam.render_passes", icon='OUTPUT', text="Render Multi-Pass Orbit Frames")

        # Dataset & Export Box
        box = layout.box()
        box.label(text="Dataset & Video Exporter", icon='EXPORT')
        row = box.row(align=True)
        row.prop(props, "output_dir", text="")
        row.operator("abucam.open_output_dir", icon='FILE_FOLDER', text="Open Folder")
        row.operator("abucam.open_logs_dir", icon='TEXT', text="Logs")

        row = box.row(align=True)
        row.operator("abucam.export_dataset", text="Export Cameras CSV & COLMAP", icon='FILE_TEXT')
        row.operator("abucam.compile_videos", icon='FILE_MOVIE')

        box.operator("abucam.export_glb", icon='FILE_3D')


# -------------------------------------------------------------------------
# Registration
# -------------------------------------------------------------------------

classes = (
    AbuCamProperties,
    ABUCAM_OT_StopRender,
    ABUCAM_OT_TogglePauseRender,
    ABUCAM_OT_ShowRenderWindow,
    ABUCAM_OT_SetupRig,
    ABUCAM_OT_SetupVolumetricRig,
    ABUCAM_OT_CenterOnTarget,
    ABUCAM_OT_ResetParam,
    ABUCAM_OT_ResetAll,
    ABUCAM_OT_ResetRig,
    ABUCAM_OT_ResetVolumetricRig,
    ABUCAM_OT_ResetTargetOffsets,
    ABUCAM_OT_ResetLights,
    ABUCAM_OT_SetupEnvSphere,
    ABUCAM_OT_RemoveEnvSphere,
    ABUCAM_OT_ResetEnvSphere,
    ABUCAM_OT_SelectEnvSphere,
    ABUCAM_OT_ResetRenderSettings,
    ABUCAM_OT_SetupLights,
    ABUCAM_OT_SetupAmbient,
    ABUCAM_OT_OpenOutputDir,
    ABUCAM_OT_OpenLogsFolder,
    ABUCAM_OT_ViewCamera,
    ABUCAM_OT_ViewVolumetricCamera,
    ABUCAM_OT_CycleVolumetricCamera,
    ABUCAM_OT_ExportGlb,
    ABUCAM_OT_RenderPasses,
    ABUCAM_OT_RenderVolumetric,
    ABUCAM_OT_ExportDataset,
    ABUCAM_OT_CompileVideos,
    ABUCAM_OT_OneClickBake,
    VIEW3D_PT_AbuCamPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.abucam_props = PointerProperty(type=AbuCamProperties)

    # Clean up any existing handler instance and register
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post
        if getattr(h, '__name__', '') != 'abucam_depsgraph_handler'
    ]
    bpy.app.handlers.depsgraph_update_post.append(abucam_depsgraph_handler)

def unregister():
    try:
        bpy.app.handlers.depsgraph_update_post[:] = [
            h for h in bpy.app.handlers.depsgraph_update_post
            if getattr(h, '__name__', '') != 'abucam_depsgraph_handler'
        ]
    except Exception:
        pass
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    try:
        del bpy.types.Scene.abucam_props
    except Exception:
        pass

if __name__ == '__main__':
    register()
