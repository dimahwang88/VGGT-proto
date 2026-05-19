// Three.js point-cloud + camera-frustum viewer.
//
// The GLB point cloud and the frustums are both built in VGGT world
// coordinates and added under one group, so they stay mutually aligned
// (absolute orientation doesn't matter — OrbitControls lets the user spin).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const DEFAULT_COLOR = 0x2b6cff;
const HILITE_COLOR = 0xff9100;

export class Viewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x101114);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.001, 5000);
    this.camera.position.set(0, 0, 3);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.zoomSpeed = 2.5;
    this.controls.zoomToCursor = true;

    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.up = new THREE.Vector3(0, 1, 0);  // scene up; refined from camera poses

    this.cameras = [];          // [{ group, line }] per camera, index-aligned
    this.selected = -1;
    this.onCameraClick = null;  // (idx) => void
    this.raycaster = new THREE.Raycaster();

    // Distinguish a click-to-select from an OrbitControls drag.
    let down = null;
    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
    el.addEventListener("click", (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down[0], e.clientY - down[1]);
      down = null;
      if (moved > 5) return;            // it was an orbit drag, not a pick
      this._pickCamera(e);
    });

    this._resize();
    window.addEventListener("resize", () => this._resize());
    this._animate();
  }

  _pickCamera(e) {
    if (!this.cameras.length) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObjects(
      this.cameras.map((c) => c.line),
      false
    );
    if (!hits.length) return;
    const idx = hits[0].object.userData.cameraIndex;
    this.highlightCamera(idx);
    this.onCameraClick?.(idx);
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  clear() {
    this.root.clear();
    this.cameras = [];
    this.selected = -1;
  }

  // Manual 180° pitch flip (y→-y, z→-z): a true vertical flip with no
  // left-right mirror. VGGT world orientation is arbitrary per scene, so
  // this is a user toggle rather than an automatic correction.
  flipVertical() {
    this.root.rotation.x += Math.PI;
    this._frameToContent();
  }

  async loadPointCloud(url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    gltf.scene.traverse((o) => {
      if (o.isPoints) {
        o.material.size = 0.012;
        o.material.sizeAttenuation = true;
        o.material.vertexColors = true;
      }
    });
    this.root.add(gltf.scene);
    this._frameToContent();
  }

  // idx: camera index; mat4: row-major 4x4 camera-to-world; intr: 3x3;
  // w,h: frame pixels. Builds frustum + RGB orientation axes + index label.
  addCamera(idx, mat4, intr, w, h, scale) {
    const m = new THREE.Matrix4().set(...mat4.flat());
    const fx = intr[0][0], fy = intr[1][1], cx = intr[0][2], cy = intr[1][2];
    const d = scale;
    const corner = (px, py) =>
      new THREE.Vector3(((px - cx) / fx) * d, ((py - cy) / fy) * d, d).applyMatrix4(m);

    const o = new THREE.Vector3().setFromMatrixPosition(m);
    const c = [corner(0, 0), corner(w, 0), corner(w, h), corner(0, h)];
    const pts = [
      o, c[0], o, c[1], o, c[2], o, c[3],
      c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0],
    ];
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.LineSegments(
      g,
      new THREE.LineBasicMaterial({ color: DEFAULT_COLOR })
    );
    line.userData.cameraIndex = idx;

    // RGB triad showing camera orientation (extrinsic rotation).
    const xa = new THREE.Vector3(), ya = new THREE.Vector3(), za = new THREE.Vector3();
    m.extractBasis(xa, ya, za);
    const axis = (dir, color) => {
      const end = o.clone().add(dir.clone().normalize().multiplyScalar(scale));
      const ag = new THREE.BufferGeometry().setFromPoints([o, end]);
      return new THREE.Line(ag, new THREE.LineBasicMaterial({ color }));
    };

    const label = this._makeLabelSprite(`#${idx}`, scale);
    label.position.copy(o);

    const group = new THREE.Group();
    group.add(line, axis(xa, 0xff3b30), axis(ya, 0x34c759), axis(za, 0x0a84ff), label);
    this.root.add(group);
    this.cameras.push({ group, line });
  }

  _makeLabelSprite(text, scale) {
    const cv = document.createElement("canvas");
    cv.width = 128;
    cv.height = 64;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.beginPath();
    ctx.roundRect(4, 14, 120, 36, 8);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 28px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 64, 33);

    const tex = new THREE.CanvasTexture(cv);
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: tex, depthTest: false })
    );
    sprite.scale.set(scale * 1.8, scale * 0.9, 1);
    return sprite;
  }

  highlightCamera(idx) {
    this.cameras.forEach((c, i) => {
      c.line.material.color.setHex(i === idx ? HILITE_COLOR : DEFAULT_COLOR);
    });
    this.selected = idx;
  }

  // VGGT cameras are OpenCV convention: cam_to_world column 1 is the camera's
  // local +Y (image-down) axis in world space, so world-up ≈ mean of the
  // negated column-1 vectors. Used as the OrbitControls azimuth axis so
  // left/right drag is a turntable spin regardless of VGGT's arbitrary frame.
  setUpFromCameras(cameras) {
    const up = new THREE.Vector3();
    for (const cam of cameras) {
      const m = cam.cam_to_world;
      up.add(new THREE.Vector3(-m[0][1], -m[1][1], -m[2][1]));
    }
    if (up.lengthSq() > 1e-9) {
      up.normalize();
      this.up.copy(up);
      this.camera.up.copy(up);
    }
  }

  _frameToContent() {
    const box = new THREE.Box3().setFromObject(this.root);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() * 0.5 || 1;

    // View direction must not be parallel to the up axis.
    const seed =
      Math.abs(this.up.dot(new THREE.Vector3(0, 0, 1))) > 0.9
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(0, 0, 1);
    const dir = seed
      .sub(this.up.clone().multiplyScalar(seed.dot(this.up)))
      .normalize();

    this.controls.target.copy(c);
    this.camera.position.copy(c).add(dir.multiplyScalar(r * 2.2));
    this.raycaster.params.Line.threshold = r * 0.02;
    this.camera.near = r / 100;
    this.camera.far = r * 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  // Average distance between camera centers — a sane frustum size.
  frustumScale(cameras) {
    if (cameras.length < 2) return 0.2;
    const ctr = cameras.map((cam) => {
      const m = cam.cam_to_world;
      return [m[0][3], m[1][3], m[2][3]];
    });
    let s = 0, n = 0;
    for (let i = 1; i < ctr.length; i++) {
      s += Math.hypot(
        ctr[i][0] - ctr[i - 1][0],
        ctr[i][1] - ctr[i - 1][1],
        ctr[i][2] - ctr[i - 1][2]
      );
      n++;
    }
    return (s / Math.max(n, 1)) * 0.4 || 0.2;
  }
}
