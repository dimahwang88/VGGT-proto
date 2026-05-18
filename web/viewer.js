// Three.js point-cloud + camera-frustum viewer.
//
// The GLB point cloud and the frustums are both built in VGGT world
// coordinates and added under one group, so they stay mutually aligned
// (absolute orientation doesn't matter — OrbitControls lets the user spin).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

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

    this.root = new THREE.Group();
    this.scene.add(this.root);

    this._resize();
    window.addEventListener("resize", () => this._resize());
    this._animate();
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

  // mat4: row-major 4x4 camera-to-world; intr: 3x3; w,h: frame pixels.
  addFrustum(mat4, intr, w, h, scale) {
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
    this.root.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x2b6cff })));
  }

  _frameToContent() {
    const box = new THREE.Box3().setFromObject(this.root);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() * 0.5 || 1;
    this.controls.target.copy(c);
    this.camera.position.copy(c).add(new THREE.Vector3(0, 0, r * 2.2));
    this.camera.near = r / 100;
    this.camera.far = r * 100;
    this.camera.updateProjectionMatrix();
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
