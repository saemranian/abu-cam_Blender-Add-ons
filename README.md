# AbuCam — Production Camera Rig, Environment Sky Sphere, Multi-Pass Renderer & 3DGS/COLMAP Exporter for Blender

[![Version](https://img.shields.io/badge/Version-v0.0.1.8-blue.svg)](https://github.com/saemranian/abu-cam_Blender-Add-ons/releases)
[![Blender](https://img.shields.io/badge/Blender-4.2%2B%20%7C%205.1%20%7C%205.2%20LTS-orange.svg)](https://www.blender.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0%20%2F%20GPL_3.0-green.svg)](LICENSE)

**AbuCam** is an all-in-one production suite for **Blender 4.2+, 5.1, and 5.2 LTS** designed for **3D Gaussian Splatting (3DGS)**, **NeRF**, **Photogrammetry (RealityCapture / RealityScan / Postshot)**, **AI Video / ControlNet Reference Generation (Wan / ComfyUI)**, and **Look-Dev Turntable Rendering**.

---

## ✨ Key Features

### 1. Dual Camera Capture Systems (`Orbital Rig` & `Volumetric Rig`)
- **Orbital Camera Rig**:
  - Smooth animated turntable camera orbiting around any target mesh or custom 3D center `(X, Y, Z)`.
  - **Coverage Modes**: Full 360° Orbit, Frontal 180° Arc (`-90°` to `+90°`), or Custom Start/End Angles.
  - **Orbit Path Tilt (`-90°` to `+90°`)**: Tilt the orbital ring plane from horizontal to vertical loops with a real-time 3D viewport curve guide.
  - **Active Target Tracking & Auto-Follow**: Locks the camera aim to the geometric bounding-box center of the selected target layer and automatically follows it when moved in the 3D viewport.
- **Volumetric Multi-Camera Dome Rig**:
  - Generates a spherical array of $N$ physical cameras (`AbuCam_Vol_001` … `AbuCam_Vol_N`) capturing synchronous multi-angle views across single or animated timeline frames.
  - **Distribution Patterns**:
    - **Fibonacci Spiral (Uniform)**: Golden-ratio spherical distribution with zero pole crowding.
    - **Latitude Rings (Tiers)**: Concentric horizontal elevation rings around the subject.
    - **Jittered / Random**: Stratified random spherical coverage.
  - **3 Angular Sector Limits**: Precisely constrain the camera dome using **Azimuth Span (`10°–360°`)**, **Heading Offset (`-180°` to `+180°`)**, **Elevation Min (Zenith Cutoff)**, and **Elevation Max (Floor Cutoff)**.
  - **Independent Sensor W & Sensor H Optics + 3D Frustum Scale**: Real-time control over **Focal Length (`mm`)**, **Sensor Width (`mm`)**, **Sensor Height (`mm`)**, and **3D Viewport Frustum Scale (`0.01 – 10.0`, default `0.25`)**, plus live synchronization of render resolution (**W × H**), **Total Frames**, and **FPS**.

---

### 2. Pro Environment Sky Sphere (`AbuCam_EnvSkySphere`)
Create a finite 3D spherical environment dome mesh around your scene with inward-facing smooth normals and studio lighting controls:
- **Controllable Sphere Radius (`0.5m – 5000m`)**: Automatically adjusts camera far-clip planes so the sky dome is never clipped.
- **6 Material & Shader Modes (Default Preset: Voronoi Texture)**:
  1. **Voronoi Texture (`VORONOI` — Default Preset)**: Procedural 3D Voronoi cellular texture with 4 styles (**Multi-Color Cells (`F1 Color`)**, **Smooth Multi-Color (`Smooth F1`)**, **Cell Distance (`Shaded`)**, and **Cell Edges / Cracks**) plus real-time **Voronoi Scale**, **Detail**, and **Randomness** controls—ideal for 3DGS / COLMAP / RealityScan background feature matching. Also registers a standalone `AbuCam_Voronoi_Material` preset in Blender's Material list.
  2. **Texture / HDRI Panorama (`HDRI_TEX`)**: Map any `.hdr`, `.exr`, `.png`, or `.jpg` panorama onto the sphere using **Equirectangular (360° Lat-Long)** or **Sphere UV** projection, with color tinting and an automatic procedural sky fallback.
  3. **Procedural Physical Sky (`PROCEDURAL_SKY`)**: Physically-based Nishita/Hosek-Wilkie atmospheric sky with real-time **Sun Elevation** and **Z-Rotation (Heading)** controls.
  4. **Studio Horizon Gradient (`GRADIENT`)**: Smooth 3-stop B-Spline cyclorama gradient (**Ground Nadir → Horizon → Zenith**).
  5. **Solid Backdrop Color (`SOLID`)**: Clean emissive/shaded solid studio backdrop.
  6. **Custom Scene Material (`CUSTOM_MAT`)**: Assign any existing Blender material from your project directly to the sky sphere (pre-populated with `AbuCam_Voronoi_Material`).
- **Pro Transform, Grading & Ray Visibility Controls**:
  - **Emission / Exposure Strength**, **Surface Roughness**, **Z-Rotation (`-360°` to `+360°`)**, **UV Scale U / V**, and **Mesh Segments (`16–256`)**.
  - **Ray Visibility Toggles**: Independently toggle **Camera Visibility**, **Diffuse GI**, **Specular Reflections**, and **Cast Shadows** (defaults to `OFF` so external 3-point studio lights shine through unimpeded).
  - **Auto-Hide in Depth / Normal / Wire Passes**: Automatically hides the sky sphere during auxiliary pass renders so your 16-bit depth maps, surface normals, and wireframe overlays capture only the foreground subject.

---

### 3. GPU-Accelerated Multi-Pass Reference Engine
- **Hardware Acceleration**:
  - **EEVEE / EEVEE Next (Real-Time GPU)**: Configured for high-speed GPU rasterization with optional Screen Raytracing toggle.
  - **Cycles (OptiX RTX Raytracing)**: Automatically enables NVIDIA RTX OptiX compute, GPU Tensor Core OpenImageDenoise, adaptive sampling, and `use_persistent_data` so geometry and textures remain resident in VRAM across frames.
- **4 Simultaneous Render Passes**:
  - **RGB Pass (`8-bit PNG`)**: Full beauty/shaded pass.
  - **Depth Pass (`16-bit PNG`)**: High-precision normalized camera-space linear depth map (`65,535` depth levels) for ControlNet / 3DGS depth regularization.
  - **Surface Normal Pass (`8-bit PNG`)**: Camera-space surface normal pass.
  - **Wireframe + RGB Pass (`8-bit PNG`)**: Clean anti-aliased topology wireframe composited over the RGB beauty render with customizable **Wire Color** (`Viewport Orange`, `Dark / Black`, `White`, `Cyan`), **Line Width (`px`)**, and **Opacity**.

---

### 4. Non-Blocking Live Render Monitor & Diagnostic Logging
- **Responsive UI with Live Preview**: Uses a non-blocking modal timer queue that yields to the OS window manager between frames and automatically opens Blender's **Render Result Preview Window**.
- **Pause / Resume & Instant Stop**: Pause rendering at any time to free GPU resources, resume seamlessly without losing progress, or cancel immediately via the UI button or `ESC` key.
- **Real-Time Telemetry**: Displays current **Frame / Task Counter**, **Last Frame Time (`s`)**, **Average Frame Time (`s`)**, **Elapsed Time**, and **Estimated Time Remaining (ETA)**.
- **Persistent Logs (`abucam.log`)**: Writes timestamped diagnostic and per-frame timing logs to `<Output Directory>/logs/abucam.log`, accessible with one click via the **Logs** button.

---

### 5. COLMAP, RealityScan / RealityCapture CSV & Video Exporter
- **Non-Destructive Versioned COLMAP Exports**: Clicking **Export Cameras CSV & COLMAP** never overwrites previous exports—if `colmap/` already exists, AbuCam automatically creates `colmap_001/`, `colmap_002/`, etc.
- **Multi-Format Camera Pose & Point Cloud Output**:
  - `cameras.csv` & `camera_poses.csv`: Matches the exact **RealityCapture / RealityScan** schema (`#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2`).
  - `cameras.txt`, `images.txt`, `points3D.ply`: Standard **COLMAP `PINHOLE`** sparse reconstruction files with a binary PLY point cloud sampled directly from the target mesh surface and normals.
  - `transforms.json`: **NeRF / 3DGS** camera intrinsics (`fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`) and $4\times4$ camera-to-world matrices.
- **FFmpeg H.264 MP4 Video Compiler & GLB Exporter**: Automatically encodes rendered pass sequences into `reference_orbit.mp4`, `depth_orbit.mp4`, `normal_orbit.mp4`, and `wire_orbit.mp4`, and exports `scene_preview.glb`.

---

## 🚀 Installation Guide

### Method 1: Install from ZIP (Recommended for Blender 4.2+ / 5.1 / 5.2 LTS)
1. Download the latest release ZIP (`abucam_addon_v0.0.1.8.zip`) from the [`releases/`](releases/) folder or GitHub Releases page (or download this repository as a ZIP).
2. Open **Blender** and go to **Edit → Preferences → Add-ons** (or **Get Extensions**).
3. Click the top-right dropdown arrow `⌄` and select **Install from Disk...**.
4. Select `abucam_addon_v0.0.1.8.zip` and enable **AbuCam**.
5. In the **3D Viewport**, press `N` to open the right-hand Sidebar and click the **AbuCam** tab.

### Method 2: Manual Extension Folder Copy
Copy `__init__.py` and `blender_manifest.toml` into your Blender user extensions directory:
```text
%APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default\abucam_addon\
```

---

## 🛠️ Quick Start Workflow

1. **Select Your Subject**: Select your 3D character or asset in the 3D Viewport.
2. **Choose Rig Mode**:
   - Select **Orbital Camera Rig** for a 360° turntable animation, or **Volumetric Camera Rig** for a multi-camera spherical dome.
   - Click **Center** next to `Target Layer` to lock the rig onto your mesh's bounding-box center.
   - Click **Setup Orbital Camera Rig** (or **Setup Volumetric Camera Rig**).
3. **Configure Studio Lighting & Sky Sphere**:
   - Click **Setup 3-Point Studio Lighting** and **Add Soft Ambient Light**.
   - Enable **Environment Sky Sphere (Pro)** to add a 3D backdrop sphere (defaults to **Voronoi Texture** preset), adjust **Sphere Radius (m)**, or switch between **Voronoi Texture**, **Texture / HDRI**, **Procedural Sky**, **Studio Horizon Gradient**, **Solid**, or **Custom Material**.
4. **Render Passes & Export Dataset**:
   - Choose **Eevee (Real-Time GPU)** or **Cycles (OptiX RTX Raytracing)**, select your desired passes (**RGB**, **Depth**, **Normal**, **Wire + RGB**), and click **Render Multi-Pass Orbit Frames** (or **Render Volumetric Multi-Camera Frames**).
   - Click **Export Cameras CSV & COLMAP** to generate non-destructive COLMAP (`cameras.txt`, `images.txt`, `points3D.ply`), RealityScan (`cameras.csv`), and NeRF (`transforms.json`) datasets.

---

## 📋 Changelog

### `v0.0.1.8` (2026-10-16)
- **Added 3D Viewport Camera Frustum Display Scale (`Frustum Scale`)**:
  - Added a real-time **`Frustum Scale` (`frustum_display_size`)** slider (`0.01 – 10.0`, default `0.25`) with a Reset button in both **Volumetric Camera Rig** (*Camera Optics & Sensor Size*) and **Orbital Camera Rig**, allowing instant scaling of camera frustum wireframes in the 3D Viewport (even across 500 Volumetric cameras) without altering focal length, sensor size, or rendered output.

### `v0.0.1.7` (2026-10-15)
- **Added Voronoi Texture Material as Default Preset for Environment Sky Sphere**:
  - Added **`Voronoi Texture (Default Preset)`** (`VORONOI`) as the default shader mode for `AbuCam_EnvSkySphere` with 4 styles (**Multi-Color Cells (`F1 Color`)**, **Smooth Multi-Color (`Smooth F1`)**, **Cell Distance (`Shaded`)**, **Cell Edges / Cracks**) and live **Voronoi Scale**, **Detail**, and **Randomness** controls.
  - Automatically registers a standalone **`AbuCam_Voronoi_Material`** preset data-block in `bpy.data.materials` for immediate selection in `Custom Scene Material` mode.

### `v0.0.1.6` (2026-10-15)
- **Fixed UI Panel Draw Truncation When Environment Sky Sphere Is Enabled**:
  - Replaced deprecated `VISIBLE_IPO_ON` icon on the *Ray Visibility & Pass Masking* box header with Blender 5.2-compatible `HIDE_OFF`, preventing `UILayout.label()` from raising an exception that truncated the **Multi-Pass Reference Engine** and **Dataset & Video Export** sections below it.

### `v0.0.1.5` (2026-10-15)
- **Added Pro Environment Sky Sphere (`AbuCam_EnvSkySphere`)**:
  - 3D inward-normal UV sphere with real-time **Sphere Radius (`m`)**, **Z-Rotation**, **UV Scale U/V**, **Emission/Exposure**, **Roughness**, and **Mesh Segments**.
  - 5 shader modes: **Texture / HDRI Panorama** (Equirectangular & UV), **Procedural Physical Sky** (Nishita), **Studio Horizon Gradient** (Zenith / Horizon / Ground), **Solid Backdrop**, and **Custom Scene Material**.
  - Studio ray visibility controls (`Camera`, `Diffuse GI`, `Specular Reflections`, `Cast Shadows`) and **Auto-Hide in Depth / Normal / Wire Passes**.
- **Fixed Volumetric Rig Optics & Resolution Live Sync**:
  - `Sensor W (mm)` and `Sensor H (mm)` now independently control horizontal and vertical camera frustum dimensions in real time with dedicated reset buttons.
  - `W` (`resolution_x`), `H` (`resolution_y`), `Total Frames`, and `FPS` now synchronize immediately with Blender's Scene Output & Timeline settings in Volumetric Rig mode.
- **Updated Extension Manifest & Repository Metadata**:
  - Updated `blender_manifest.toml` and `bl_info` website/documentation links to `https://github.com/saemranian/abu-cam_Blender-Add-ons`.
  - Standardized versioning starting at `v0.0.1.5`.

---

## 📄 License
Licensed under the [Apache License 2.0](LICENSE) / GPL-3.0-or-later for Blender Extension compatibility.