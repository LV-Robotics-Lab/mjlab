"""Interactive RTX viewer backed by NVIDIA ovrtx, streamed to viser.

ovrtx is a headless offscreen path tracer: it renders to a framebuffer, it does
not open a window. So we drive it from the running sim and present the frames
through a viser web canvas (works headless / over SSH, no OpenGL needed).

Per displayed frame we write the live geom world transforms
(``geom_xpos``/``geom_xmat``, straight from the mujoco-warp GPU arrays) into
ovrtx via a zero-copy ``bind_attribute`` mapping plus a Warp kernel, update a
follow camera, render, and push the result to ``viser`` as a background image.
Rendering is pipelined: each frame fetches the previous async render and kicks
the next, so the GPU render overlaps the viewer's physics stepping.

The scene geometry is produced once with ``mujoco.usd.exporter`` (which handles
the full MjModel -> USD conversion) and loaded into ovrtx; only transforms move
afterwards.
"""

from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol

_RENDER_PRODUCT = "/Render/Camera"
_CAM_PRIM = "/World/RenderCam"


@wp.kernel(enable_backward=False)
def _write_geom_xforms(
  transforms: wp.array(dtype=wp.mat44d),
  xpos: wp.array2d(dtype=wp.vec3),
  xmat: wp.array2d(dtype=wp.mat33),
  gids: wp.array(dtype=wp.int32),
  world: int,
):
  """Write each geom's world pose as a USD row-major mat44d (Rᵀ, translate row3)."""
  i = wp.tid()
  g = gids[i]
  r = xmat[world, g]
  t = xpos[world, g]
  z = wp.float64(0.0)
  one = wp.float64(1.0)
  transforms[i] = wp.mat44d(
    wp.float64(r[0, 0]),
    wp.float64(r[1, 0]),
    wp.float64(r[2, 0]),
    z,
    wp.float64(r[0, 1]),
    wp.float64(r[1, 1]),
    wp.float64(r[2, 1]),
    z,
    wp.float64(r[0, 2]),
    wp.float64(r[1, 2]),
    wp.float64(r[2, 2]),
    z,
    wp.float64(t[0]),
    wp.float64(t[1]),
    wp.float64(t[2]),
    one,
  )


def _camera_matrix(
  eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None
) -> np.ndarray:
  """Row-major USD camera-to-world transform (camera looks along local -Z)."""
  if up_hint is None:
    up_hint = np.array([0.0, 0.0, 1.0])
  fwd = target - eye
  fwd /= np.linalg.norm(fwd)
  right = np.cross(fwd, up_hint)
  right /= np.linalg.norm(right)
  up = np.cross(right, fwd)
  m = np.eye(4, dtype=np.float64)
  m[0, :3] = right
  m[1, :3] = up
  m[2, :3] = -fwd
  m[3, :3] = eye
  return m


def _parse_geom_prims(stage) -> tuple[list[str], list[int]]:
  """Return (rel_paths, geom_ids) for every exported per-geom Xform prim.

  Each prim's name ends in ``_id<geom_id>_geom``. The robot's geoms are named
  ``Mesh_Xform_None_id*_geom`` but named entities (e.g. a ``cube`` object) get a
  different prefix like ``cube_geom_id57_geom`` and may be nested, so we match on
  the id suffix and prim type rather than a name prefix, and return each prim's
  path relative to ``/World``. Paths (not names) because prims may be nested.
  """
  from pxr import Usd

  paths: list[str] = []
  ids: list[int] = []
  pat = re.compile(r"_id(\d+)_geom$")
  prefix = "/World/"
  for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
    if prim.GetTypeName() != "Xform":  # the Xform container, not its Mesh child
      continue
    m = pat.search(prim.GetName())
    if m is None:
      continue
    path = prim.GetPath().pathString
    if not path.startswith(prefix):
      continue
    paths.append(path[len(prefix) :])
    ids.append(int(m.group(1)))
  return paths, ids


def _set_material_opacity(stage, opacity: float) -> None:
  """Make all exported materials translucent (for the ghost).

  Also zero out metallic: the exporter defaults materials to metallic=0.5, and a
  metallic surface renders opaque (conductors don't transmit), so the opacity
  would otherwise be ignored by the RTX renderer.
  """
  from pxr import Sdf, UsdShade

  materials = stage.GetPrimAtPath("/World/_materials")
  if not materials:
    return
  for mat in materials.GetChildren():
    sh = stage.GetPrimAtPath(mat.GetPath().AppendChild("Principled_BSDF"))
    if not sh:
      continue
    sh = UsdShade.Shader(sh)
    for name, value in (("opacity", float(opacity)), ("metallic", 0.0)):
      inp = sh.GetInput(name) or sh.CreateInput(name, Sdf.ValueTypeNames.Float)
      inp.DisconnectSource()
      inp.Set(value)


def _geom_xforms_numpy(geom_xpos, geom_xmat, geom_ids) -> np.ndarray:
  """Build USD row-major (N,4,4) transforms from MjData geom world poses."""
  ids = np.asarray(geom_ids, dtype=np.int64)
  pos = np.asarray(geom_xpos)[ids]
  rot = np.asarray(geom_xmat)[ids].reshape(-1, 3, 3)
  m = np.zeros((len(ids), 4, 4), dtype=np.float64)
  m[:, :3, :3] = np.transpose(rot, (0, 2, 1))  # Rᵀ in the upper 3x3
  m[:, 3, :3] = pos
  m[:, 3, 3] = 1.0
  return m


class OvrtxViewer(BaseViewer):
  """Path-traced viewer that streams ovrtx frames to a viser web canvas."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 30.0,
    width: int | None = None,
    height: int | None = None,
    port: int = 8080,
    verbosity: int = 0,
  ) -> None:
    super().__init__(env, policy, frame_rate=frame_rate, verbosity=verbosity)
    # Default to 1080p; the env's ViewerConfig size is tuned for small
    # training-video thumbnails, which is too low for an interactive RTX viewer.
    self._width = int(width or 1920)
    self._height = int(height or 1080)
    self._port = port
    # Task-agnostic follow-cam: a fixed 3/4 direction at an auto-computed distance,
    # aimed at the centroid of the (non-ground) geometry. Works for a floating-base
    # humanoid and a fixed-base manipulator alike. Filled in during setup.
    self._cam_dir = np.array([0.62, -0.67, 0.41])
    self._cam_dir = self._cam_dir / np.linalg.norm(self._cam_dir)
    self._cam_dist = 4.0
    self._cam_target: np.ndarray | None = None  # smoothed look-at
    self._track_geom_ids: list[int] = []
    self._clients: list[Any] = []
    self._track = True  # default; the live value comes from _track_handle
    self._track_handle: Any = None

    # Frames are encoded + pushed to viser on a worker thread so the JPEG encode
    # overlaps the next GPU render instead of blocking the main loop. Newest frame
    # wins; if the encoder falls behind, intermediate frames are dropped.
    self._latest_rgb: np.ndarray | None = None
    self._frame_lock = threading.Lock()
    self._frame_ready = threading.Event()
    self._stop = threading.Event()
    self._frame_thread: threading.Thread | None = None

    self._renderer: Any = None
    self._server: Any = None
    self._binding: Any = None
    self._pending: Any = None
    self._gids_wp: Any = None
    self._geom_ids: list[int] = []
    self._workdir = Path(tempfile.mkdtemp(prefix="mjlab_ovrtx_"))

    # Tracking-task "ghost" of the reference motion (built lazily in setup).
    self._motion_cmd: Any = None
    self._ghost_model: Any = None
    self._ghost_data: Any = None
    self._ghost_binding: Any = None
    self._ghost_geom_ids: list[int] = []
    self._ghost_usd: Path | None = None
    self._ghost_prim_names: list[str] = []

  # Lifecycle.

  def setup(self) -> None:
    import viser

    wp.init()

    # Export the USD scene (mujoco + pxr/usd-core) BEFORE importing ovrtx. ovrtx
    # bundles its own USD build; importing it first makes the two USD schema
    # registries clash ("alias ... already set") when the exporter opens a stage.
    usd_path, prim_paths, self._geom_ids = self._export_scene()
    self.log(f"[ovrtx] scene: {len(prim_paths)} geoms -> {usd_path}")

    from ovrtx import Device, PrimMode, Renderer, RendererConfig

    self._Device = Device
    self.log("[ovrtx] creating renderer (first run compiles shaders) ...")
    self._renderer = Renderer(RendererConfig(active_cuda_gpus="0"))
    self._renderer.open_usd(str(usd_path))

    device = str(self.env.unwrapped.device)
    self._gids_wp = wp.array(
      np.asarray(self._geom_ids, dtype=np.int32), dtype=wp.int32, device=device
    )
    self._binding = self._renderer.bind_attribute(
      prim_paths=prim_paths,
      attribute_name="omni:xform",
      dtype="float64",
      shape=(4, 4),
      prim_mode=PrimMode.MUST_EXIST,
    )

    # Reference-motion ghost: a translucent second robot, added as a USD reference
    # and driven each frame from the motion command. Best-effort: never let a ghost
    # problem take down the viewer.
    if self._ghost_usd is not None:
      try:
        self._renderer.add_usd_reference(str(self._ghost_usd), "/World/Ghost")
        ghost_paths = [f"/World/Ghost/{n}" for n in self._ghost_prim_names]
        self._ghost_binding = self._renderer.bind_attribute(
          prim_paths=ghost_paths,
          attribute_name="omni:xform",
          dtype="float64",
          shape=(4, 4),
          prim_mode=PrimMode.MUST_EXIST,
        )
        self.log(f"[ovrtx] ghost: {len(ghost_paths)} geoms")
      except Exception as e:  # noqa: BLE001
        self.log(f"[ovrtx] ghost disabled ({e})")
        self._ghost_binding = None

    self._server = viser.ViserServer(port=self._port)
    self._server.scene.set_up_direction("+z")  # match MuJoCo's Z-up world
    self._add_gui()

    @self._server.on_client_connect
    def _(client: Any) -> None:
      # Frame the scene, then let the user orbit/pan/zoom (used when tracking off).
      target = self._cam_target if self._cam_target is not None else np.zeros(3)
      client.camera.position = tuple((target + self._cam_dir * self._cam_dist).tolist())
      client.camera.look_at = tuple(target.tolist())
      self._clients.append(client)

    @self._server.on_client_disconnect
    def _(client: Any) -> None:
      if client in self._clients:
        self._clients.remove(client)

    print(f"\n[ovrtx] open the viewer at  http://localhost:{self._port}\n")

    self._frame_thread = threading.Thread(target=self._frame_worker, daemon=True)
    self._frame_thread.start()

    # Prime the async pipeline with the initial state.
    self._write_state()
    self._pending = self._renderer.step_async(
      render_products={_RENDER_PRODUCT}, delta_time=self.frame_time
    )

  def sync_env_to_viewer(self) -> None:
    if self._renderer is None:
      return
    # Read back the finished render (consuming its buffer, required before the
    # next step) and hand the pixels to the encoder thread, then immediately kick
    # the next render so the JPEG encode overlaps it.
    if self._pending is not None:
      rgb = self._fetch_rgb(self._pending.wait().fetch())
      if rgb is not None:
        with self._frame_lock:
          self._latest_rgb = rgb
        self._frame_ready.set()
    self._write_state()
    self._pending = self._renderer.step_async(
      render_products={_RENDER_PRODUCT}, delta_time=self.frame_time
    )

  def _frame_worker(self) -> None:
    """Encode + push the newest frame to viser, off the main render loop."""
    while not self._stop.is_set():
      if not self._frame_ready.wait(timeout=0.1):
        continue
      self._frame_ready.clear()
      with self._frame_lock:
        rgb = self._latest_rgb
        self._latest_rgb = None
      if rgb is None or self._server is None:
        continue
      try:
        self._server.scene.set_background_image(rgb, format="jpeg", jpeg_quality=92)
      except Exception:  # noqa: BLE001
        pass

  def sync_viewer_to_env(self) -> None:
    """No viewer-driven perturbations to push back to the sim."""

  def is_running(self) -> bool:
    return True

  def close(self) -> None:
    self._stop.set()
    self._frame_ready.set()
    if self._frame_thread is not None:
      self._frame_thread.join(timeout=1.0)
      self._frame_thread = None
    if self._pending is not None:
      try:
        self._pending.wait().fetch()
      except Exception:
        pass
      self._pending = None
    for attr in ("_binding", "_ghost_binding"):
      b = getattr(self, attr)
      if b is not None:
        try:
          b.unbind()
        except Exception:
          pass
        setattr(self, attr, None)
    if self._server is not None:
      self._server.stop()
      self._server = None
    self._renderer = None

  # Internals.

  def _write_state(self) -> None:
    """Push live geom transforms (GPU) and the current camera pose into ovrtx."""
    from ovrtx import Semantic

    sim = self.env.unwrapped.sim
    wp_data = sim._wp_data
    device = str(self.env.unwrapped.device)
    with self._binding.map(device=self._Device.CUDA, device_id=0) as mapping:
      tr = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
      wp.launch(
        _write_geom_xforms,
        dim=len(self._geom_ids),
        inputs=[tr, wp_data.geom_xpos, wp_data.geom_xmat, self._gids_wp, 0],
        device=device,
      )
      wp.synchronize_device(device)
      mapping.unmap()

    self._write_ghost(Semantic)
    self._renderer.write_attribute(
      prim_paths=[_CAM_PRIM],
      attribute_name="omni:xform",
      tensor=self._camera_xform()[None],
      semantic=Semantic.XFORM_MAT4x4,
    )

  def _write_ghost(self, semantic) -> None:
    """Pose the ghost at the motion command's reference pose for this frame."""
    if self._ghost_binding is None:
      return
    import mujoco

    cmd = self._motion_cmd
    entity = self.env.unwrapped.scene[cmd.cfg.entity_name]
    free_adr = entity.indexing.free_joint_q_adr.cpu().numpy()
    joint_adr = entity.indexing.joint_q_adr.cpu().numpy()

    qpos = np.zeros(self._ghost_model.nq)
    qpos[free_adr[0:3]] = cmd.body_pos_w[0, 0].cpu().numpy()
    qpos[free_adr[3:7]] = cmd.body_quat_w[0, 0].cpu().numpy()
    qpos[joint_adr] = cmd.joint_pos[0].cpu().numpy()
    self._ghost_data.qpos[:] = qpos
    mujoco.mj_forward(self._ghost_model, self._ghost_data)

    xforms = _geom_xforms_numpy(
      self._ghost_data.geom_xpos, self._ghost_data.geom_xmat, self._ghost_geom_ids
    )
    self._renderer.write_attribute(
      prim_paths=[f"/World/Ghost/{n}" for n in self._ghost_prim_names],
      attribute_name="omni:xform",
      tensor=xforms,
      semantic=semantic.XFORM_MAT4x4,
    )

  def _camera_xform(self) -> np.ndarray:
    """Render-camera transform.

    "Track robot" on (default): a locked follow-cam that keeps the robot framed.
    Off: the user's viser camera, freely orbitable with the mouse. We never write
    back to the client camera (doing so fights viser's own controls and oscillates).
    """
    tracking = self._track_handle.value if self._track_handle else self._track
    if tracking or not self._clients:
      target = self._track_target()
      return _camera_matrix(target + self._cam_dir * self._cam_dist, target)

    cam = self._clients[-1].camera
    return _camera_matrix(
      np.asarray(cam.position, dtype=np.float64),
      np.asarray(cam.look_at, dtype=np.float64),
      np.asarray(cam.up_direction, dtype=np.float64),
    )

  def _track_target(self) -> np.ndarray:
    """Smoothed centroid of the tracked (non-ground) geometry in world space."""
    gpos = self.env.unwrapped.sim._wp_data.geom_xpos.numpy()[0]
    centroid = gpos[np.asarray(self._track_geom_ids)].mean(axis=0).astype(np.float64)
    if self._cam_target is None:
      self._cam_target = centroid
    else:
      self._cam_target = 0.85 * self._cam_target + 0.15 * centroid
    return self._cam_target

  def _fetch_rgb(self, products) -> np.ndarray | None:
    for _, product in products.items():
      for frame in product.frames:
        v = frame.render_vars["LdrColor"].map(device=self._Device.CPU)
        # Copy off the mapped buffer: it is reused by the next render, and the
        # encoder thread reads this asynchronously.
        return np.ascontiguousarray(np.from_dlpack(v)[..., :3])
    return None

  def _export_scene(self) -> tuple[Path, list[str], list[int]]:
    """Export geometry once and author a camera, lights, and render product."""
    import mujoco
    from mujoco.usd import exporter as usd_exporter
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    sim = self.env.unwrapped.sim
    mj_model = sim.mj_model
    mj_model.vis.global_.offwidth = max(int(mj_model.vis.global_.offwidth), self._width)
    mj_model.vis.global_.offheight = max(
      int(mj_model.vis.global_.offheight), self._height
    )

    mj_data = mujoco.MjData(mj_model)
    d = sim.data
    mj_data.qpos[:] = d.qpos[0].cpu().numpy()
    mj_data.qvel[:] = d.qvel[0].cpu().numpy()
    if mj_model.nmocap > 0:
      mj_data.mocap_pos[:] = d.mocap_pos[0].cpu().numpy()
      mj_data.mocap_quat[:] = d.mocap_quat[0].cpu().numpy()
    mujoco.mj_forward(mj_model, mj_data)

    exp = usd_exporter.USDExporter(
      model=mj_model,
      height=self._height,
      width=self._width,
      output_directory="scene",
      output_directory_root=str(self._workdir),
      camera_names=[],
      light_intensity=150,
      verbose=False,
    )
    exp.update_scene(mj_data)
    stage = exp.stage

    names, geom_ids = _parse_geom_prims(stage)
    prim_paths = [f"/World/{n}" for n in names]

    # Auto-frame the scene: aim at the centroid of the non-ground geometry, at a
    # distance scaled to its spread. Works for both floating-base and fixed-base.
    plane = mujoco.mjtGeom.mjGEOM_PLANE
    track_ids = [g for g in geom_ids if mj_model.geom_type[g] != plane]
    self._track_geom_ids = track_ids or list(geom_ids)
    gpos = np.asarray(mj_data.geom_xpos)[np.asarray(self._track_geom_ids)]
    centroid = gpos.mean(axis=0)
    radius = (
      float(np.linalg.norm(gpos - centroid, axis=1).max()) if len(gpos) > 1 else 0.5
    )
    self._cam_dist = max(1.2, 2.6 * radius)
    self._cam_target = centroid.astype(np.float64)

    target = Gf.Vec3d(*centroid.tolist())
    eye = Gf.Vec3d(*(centroid + self._cam_dir * self._cam_dist).tolist())
    cam = UsdGeom.Camera.Define(stage, _CAM_PRIM)
    cam.CreateProjectionAttr("perspective")
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0 * self._height / self._width)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(view.GetInverse())

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(350.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    sun.CreateIntensityAttr(2400.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))
    sun.CreateAngleAttr(0.53)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 40.0))

    stage.DefinePrim("/Render", "Scope")
    var = stage.DefinePrim("/Render/Vars/LdrColor", "RenderVar")
    var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
    product = stage.DefinePrim(_RENDER_PRODUCT, "RenderProduct")
    product.CreateRelationship("camera").SetTargets([_CAM_PRIM])
    product.CreateRelationship("orderedVars").SetTargets(["/Render/Vars/LdrColor"])
    product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(
      Gf.Vec2i(self._width, self._height)
    )
    product.CreateAttribute(
      "omni:rtx:background:source:type", Sdf.ValueTypeNames.Token
    ).Set("sky")

    exp.save_scene("usda")
    usd_path = self._workdir / "scene" / "frames" / f"frame_{exp.frame_count}.usda"

    self._export_ghost(mj_model, mj_data)
    return usd_path, prim_paths, geom_ids

  def _export_ghost(self, mj_model, mj_data) -> None:
    """If this is a tracking task, export a translucent ghost of the robot.

    The ghost is the same robot geometry recolored translucent; it is driven each
    frame from the motion command's reference pose. Mirrors MotionCommand's own
    ghost setup (collision geoms hidden, visual geoms tinted).
    """
    import copy

    import mujoco
    from mujoco.usd import exporter as usd_exporter

    cmd = self._detect_motion_command()
    if cmd is None:
      return

    color = np.asarray(cmd.cfg.viz.ghost_color, dtype=np.float32)
    ghost_model = copy.deepcopy(mj_model)
    for gi in range(ghost_model.ngeom):
      if ghost_model.geom_contype[gi] != 0 or ghost_model.geom_conaffinity[gi] != 0:
        ghost_model.geom_rgba[gi, 3] = 0.0
      else:
        ghost_model.geom_rgba[gi] = color

    exp = usd_exporter.USDExporter(
      model=ghost_model,
      height=self._height,
      width=self._width,
      output_directory="ghost",
      output_directory_root=str(self._workdir),
      camera_names=[],
      light_intensity=150,
      verbose=False,
    )
    exp.update_scene(mj_data)
    _set_material_opacity(exp.stage, float(color[3]))
    names, geom_ids = _parse_geom_prims(exp.stage)
    exp.save_scene("usda")

    self._motion_cmd = cmd
    self._ghost_model = ghost_model
    self._ghost_data = mujoco.MjData(ghost_model)
    self._ghost_prim_names = names
    self._ghost_geom_ids = geom_ids
    self._ghost_usd = (
      self._workdir / "ghost" / "frames" / f"frame_{exp.frame_count}.usda"
    )

  def _detect_motion_command(self) -> Any:
    """Return the tracking MotionCommand if present and set to 'ghost' viz."""
    cm = getattr(self.env.unwrapped, "command_manager", None)
    terms = getattr(cm, "_terms", {}) if cm is not None else {}
    cmd = terms.get("motion")
    viz = getattr(getattr(cmd, "cfg", None), "viz", None)
    if cmd is None or getattr(viz, "mode", None) != "ghost":
      return None
    return cmd

  def _add_gui(self) -> None:
    """A few viser controls wired to the BaseViewer action queue."""
    gui = self._server.gui
    # Read the checkbox live each frame (via _is_tracking) rather than through a
    # callback, so untracking takes effect immediately and can't get stale.
    self._track_handle = gui.add_checkbox("Track robot", initial_value=self._track)

    pause = gui.add_button("Pause / Resume")
    pause.on_click(lambda _: self.request_toggle_pause())
    reset = gui.add_button("Reset")
    reset.on_click(lambda _: self.request_reset())
    faster = gui.add_button("Speed +")
    faster.on_click(lambda _: self.request_speed_up())
    slower = gui.add_button("Speed -")
    slower.on_click(lambda _: self.request_speed_down())
