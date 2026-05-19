// Learn-mode: an interactive VGGT explainer driven by the loaded scene.
// Content source: "VGGT: Visual Geometry Grounded Transformer", Wang et al.,
// CVPR 2025, and the public github.com/facebookresearch/vggt source.

import mermaid from "mermaid";
import katex from "katex";
import Chart from "chart.js/auto";

const $ = (id) => document.getElementById(id);
let built = false;
let ctx = null; // { state, viewer }
let mixChart = null;

mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });

// --- KaTeX helper ----------------------------------------------------------
function kx(tex, display = true) {
  return katex.renderToString(tex, { displayMode: display, throwOnError: false });
}

// --- Architecture content --------------------------------------------------
const ARCH_GRAPH = `flowchart TD
  IMG["Input images<br/>N x 3 x H x W"] --> PATCH["Patchify + DINOv2 ViT-L<br/>+ camera & register tokens"]
  PATCH --> AGG["Alternating-Attention Aggregator<br/>frame-wise &harr; global x L"]
  AGG --> CAM["Camera head<br/>iterative x4 -> pose enc"]
  AGG --> DEPTH["DPT depth head<br/>+ confidence"]
  AGG --> POINT["DPT point head<br/>+ confidence"]
  AGG --> TRACK["Track head<br/>CoTracker-style"]
  CAM --> OUT["Extrinsics [R|t] + intrinsics K"]
  DEPTH --> OUT
  click IMG call __vggtArch("img")
  click PATCH call __vggtArch("patch")
  click AGG call __vggtArch("agg")
  click CAM call __vggtArch("cam")
  click DEPTH call __vggtArch("depth")
  click POINT call __vggtArch("point")
  click TRACK call __vggtArch("track")
  click OUT call __vggtArch("out")`;

const ARCH = {
  img: {
    t: "Input & coordinate frame",
    h: `<p>VGGT ingests <b>1–N</b> RGB views jointly in a single forward pass.
    All predictions live in the coordinate frame of the <b>first camera</b>
    (camera&nbsp;1 = world). That is exactly why this app's tracking has a
    fast path when the reference frame is index&nbsp;0.</p>`,
  },
  patch: {
    t: "Tokenization (DINOv2)",
    h: `<p><b>Refresher — ViT patching:</b> each image is split into
    14×14 patches; a linear projection turns every patch into a token. VGGT
    initializes this backbone from <b>DINOv2 ViT-L</b> (self-supervised
    features). Per image it appends one learnable <b>camera token</b> and
    4 <b>register tokens</b> (register tokens absorb artefacts — from
    "Vision Transformers Need Registers").</p>
    <p>Token count per image: ${kx(String.raw`S = 1_{\text{cam}} + 4_{\text{reg}} + \tfrac{H}{14}\cdot\tfrac{W}{14}`, false)} patches.</p>
    <p><code>vggt/models/aggregator.py</code></p>`,
  },
  agg: {
    t: "Alternating-Attention Aggregator (the core idea)",
    h: `<p><b>Refresher — attention:</b> ${kx(String.raw`\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V`)}</p>
    <p>The aggregator stacks <b>L</b> blocks that <b>alternate</b> between:</p>
    <ul>
      <li><b>Frame-wise</b> self-attention — tokens attend only within their
      own image (per-view structure, cheap: cost ${kx(String.raw`O(N\,S^2)`, false)}).</li>
      <li><b>Global</b> self-attention — all tokens of all images attend to
      each other (cross-view fusion: cost ${kx(String.raw`O((NS)^2)`, false)}).</li>
    </ul>
    <p>Alternating gives multi-view reasoning without the full cost of
    always-global attention. The <b>Attention</b> tab visualizes this: the
    "cross-frame mixing" of a query token grows as you step deeper —
    that rise <i>is</i> global attention fusing the views.</p>
    <p><code>vggt/models/aggregator.py</code></p>`,
  },
  cam: {
    t: "Camera head",
    h: `<p>Reads the per-image <b>camera token</b> and applies a small trunk
    <b>iteratively (4 passes)</b>, each refining a 9-D <b>pose encoding</b>:
    ${kx(String.raw`\;[\,\mathbf{t}\in\mathbb{R}^3,\; \mathbf{q}\in\mathbb{R}^4,\; \text{FOV}\in\mathbb{R}^2\,]`, false)}.</p>
    <p>Decoded to extrinsics/intrinsics by
    <code>pose_encoding_to_extri_intri</code> — the same util this app calls
    in <code>vggt_runner.reconstruct</code>. Principal point is the image
    center; focal comes from FOV (see the <b>Geometry math</b> tab).</p>
    <p><code>vggt/heads/camera_head.py</code>, <code>vggt/utils/pose_enc.py</code></p>`,
  },
  depth: {
    t: "DPT depth head",
    h: `<p><b>Refresher — DPT:</b> a Dense Prediction Transformer reassembles
    tokens from several aggregator layers into a multi-scale feature pyramid,
    then convolutionally decodes a dense per-pixel map.</p>
    <p>Outputs per-pixel <b>depth</b> + an <b>aleatoric confidence</b> σ. The
    point cloud you see is depth <i>unprojected</i> to 3D (more stable than
    the direct point head) — see <code>unproject_depth_map_to_point_map</code>.
    The <b>Depth: on</b> toggle shows this head's output.</p>
    <p><code>vggt/heads/dpt_head.py</code></p>`,
  },
  point: {
    t: "DPT point head",
    h: `<p>A second DPT head regresses a <b>point map</b> (3D point per pixel
    in the camera-1 frame) + confidence. VGGT supervises it, but the paper
    finds depth→unprojection more accurate for the final cloud, which is the
    convention this app follows.</p>
    <p><code>vggt/heads/dpt_head.py</code></p>`,
  },
  track: {
    t: "Track head",
    h: `<p>A <b>CoTracker</b>-style module: given dense features + query
    points it iteratively correlates and predicts 2D <b>tracks</b> across all
    frames plus <b>visibility</b>. This powers the Point-tracking panel; query
    points are expected on frame&nbsp;0, hence the reorder logic in
    <code>vggt_runner.track</code>.</p>
    <p><code>vggt/heads/track_head.py</code></p>`,
  },
  out: {
    t: "Outputs",
    h: `<p>Per view: extrinsics ${kx(String.raw`[R|t]`, false)} (world→cam),
    intrinsics ${kx(String.raw`K`, false)}, dense depth + point map with
    confidence, and tracks. All consistent in the camera-1 world frame, so
    the cloud and frustums in the 3D view share one coordinate system.</p>`,
  },
};

window.__vggtArch = (key) => {
  const n = ARCH[key];
  if (n) $("archInfo").innerHTML = `<h4 style="margin:.2em 0">${n.t}</h4>${n.h}`;
};

// --- Geometry math (with refreshers) ---------------------------------------
function geomHtml() {
  return `
  <h4 style="margin:.2em 0">Pinhole projection</h4>
  <p><b>Refresher.</b> A world point ${kx(String.raw`\mathbf{X}_w`, false)} maps to a pixel via
  extrinsics then intrinsics:</p>
  ${kx(String.raw`\mathbf{X}_c = R\,\mathbf{X}_w + \mathbf{t}, \qquad
     \begin{bmatrix}u\\v\\1\end{bmatrix} \sim K\,\mathbf{X}_c,\quad
     K=\begin{bmatrix} f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1\end{bmatrix}`)}
  <p>VGGT puts the principal point at the image center
  ${kx(String.raw`(c_x,c_y)=(W/2,\,H/2)`, false)} and derives focal from FOV:</p>
  ${kx(String.raw`f_x=\dfrac{W/2}{\tan(\text{FOV}_x/2)}`)}
  <h4 style="margin:.6em 0 .2em">Rotation: quaternion → R</h4>
  <p><b>Refresher.</b> The pose encoding stores rotation as a unit quaternion
  ${kx(String.raw`\mathbf{q}=(w,x,y,z)`, false)}; it expands to</p>
  ${kx(String.raw`R=\begin{bmatrix}
   1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy)\\
   2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx)\\
   2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)\end{bmatrix}`)}
  <h4 style="margin:.6em 0 .2em">Depth unprojection (what builds the cloud)</h4>
  <p>For pixel ${kx(String.raw`(u,v)`, false)} with predicted depth
  ${kx(String.raw`d`, false)}, the world point is the inverse pipeline:</p>
  ${kx(String.raw`\mathbf{X}_w = R^{\top}\!\left(d\,K^{-1}\begin{bmatrix}u\\v\\1\end{bmatrix} - \mathbf{t}\right)`)}
  <p class="hint">This is exactly what
  <code>unproject_depth_map_to_point_map</code> does. Move the sliders to see
  the FOV change the frustum and the principal ray's reach in the 3D view.</p>`;
}

// --- Training (conceptual, from the paper) ---------------------------------
const DATASETS = [
  ["Co3Dv2", 18], ["BlendedMVS", 14], ["MegaDepth", 12], ["ScanNet++", 11],
  ["ARKitScenes", 10], ["Hypersim (synth)", 9], ["Habitat/HM3D (synth)", 9],
  ["TartanAir (synth)", 8], ["WildRGB-D / others", 9],
];

function trainHtml() {
  return `
  <h4 style="margin:.2em 0">Multi-task loss</h4>
  <p>VGGT is trained end-to-end with a weighted sum over the heads:</p>
  ${kx(String.raw`\mathcal{L}=\mathcal{L}_{\text{cam}}+\mathcal{L}_{\text{depth}}+\mathcal{L}_{\text{point}}+\lambda\,\mathcal{L}_{\text{track}}`)}
  <p><b>Refresher — aleatoric (Laplace) confidence weighting.</b> The dense
  heads predict an uncertainty σ; the loss down-weights uncertain pixels and
  pays a small price for being uncertain (a learned, per-pixel robustness):</p>
  ${kx(String.raw`\mathcal{L}_{\text{depth}}=\big|\,\hat{y}-y\,\big|\cdot e^{-\sigma}+\alpha\,\sigma`)}
  <p>Camera loss: Huber on the pose-encoding components
  ${kx(String.raw`(\mathbf{t},\mathbf{q},\text{FOV})`, false)}. Track loss:
  L1 on tracks + BCE on visibility (CoTracker-style).</p>
  <h4 style="margin:.6em 0 .2em">Recipe (from the paper)</h4>
  <ul>
    <li>Backbone initialized from <b>DINOv2</b>.</li>
    <li>Variable views per sample (≈1–24) and variable resolution —
    teaches view-count generalization.</li>
    <li>AdamW + cosine schedule; large-scale (paper reports ≈64×A100).</li>
    <li>Ground truth = real + synthetic datasets with metric depth/cameras.</li>
  </ul>
  <p class="hint">⚠ Training code &amp; the exact data mixture are <b>not
  publicly released</b>. The chart is an indicative composition and the
  recipe is summarized from the paper — nothing here is reproduced.</p>`;
}

// --- Attention tab ---------------------------------------------------------
let attnQuery = [0, 0]; // model-pixel coords on the picked frame
let fetchTimer = null;

function drawPick() {
  const { state } = ctx;
  const f = +$("attnFrame").value;
  const img = state.frameImgs[f];
  if (!img) return;
  const cv = $("attnPick");
  const w = cv.clientWidth || cv.parentElement.clientWidth || 280;
  const h = Math.round((state.fh / state.fw) * w);
  cv.width = w;
  cv.height = h;
  const c = cv.getContext("2d");
  c.drawImage(img, 0, 0, w, h);
  // marker
  c.strokeStyle = "#ffd400";
  c.lineWidth = 2;
  const mx = (attnQuery[0] / state.fw) * w;
  const my = (attnQuery[1] / state.fh) * h;
  c.beginPath();
  c.arc(mx, my, 6, 0, Math.PI * 2);
  c.stroke();
}

function scheduleFetch() {
  clearTimeout(fetchTimer);
  fetchTimer = setTimeout(fetchIntrospect, 140);
}

async function fetchIntrospect() {
  const { state } = ctx;
  if (!state.session) return;
  const f = +$("attnFrame").value;
  const l = +$("attnLayer").value;
  $("attnFrameV").textContent = f;
  $("attnLayerV").textContent = l;
  const u = `/api/introspect/${state.session}?frame=${f}&layer=${l}` +
    `&qx=${attnQuery[0]}&qy=${attnQuery[1]}`;
  try {
    const r = await fetch(u);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    $("attnSim").src = d.attention_png;
    $("attnNorm").src = d.tokennorm_png;
    $("attnLayer").max = String(d.num_layers - 1);
    $("attnKind").innerHTML =
      `Captured layer <b>${d.layer}</b>/${d.num_layers - 1} · ` +
      `heuristic kind: <b>${d.layer_kind}</b> · patch grid ` +
      `${d.grid[0]}×${d.grid[1]} · query patch [${d.query_patch}]`;
    drawMix(d.cross_frame_mix, f);
    $("attnNote").textContent =
      "Similarity is a robust proxy for attention (VGGT's fused SDPA " +
      "weights aren't exposed without patching the model). Watch " +
      "cross-frame mixing rise across layers — that is global attention.";
  } catch (e) {
    $("attnKind").innerHTML = `<span class="err">${e.message}</span>`;
  }
}

function drawMix(mix, picked) {
  const cv = $("attnMix");
  const c = cv.getContext("2d");
  c.clearRect(0, 0, cv.width, cv.height);
  const n = mix.length;
  const bw = cv.width / n;
  for (let i = 0; i < n; i++) {
    const v = (mix[i] + 1) / 2; // [-1,1] -> [0,1]
    const bh = v * (cv.height - 16);
    c.fillStyle = i === picked ? "#ffd400" : "#2b6cff";
    c.fillRect(i * bw + 2, cv.height - bh - 14, bw - 4, bh);
    c.fillStyle = "#9aa0a6";
    c.font = "10px system-ui";
    c.fillText(`#${i}`, i * bw + 3, cv.height - 2);
  }
}

// --- Geometry tab ----------------------------------------------------------
function sceneScale() {
  const { viewer, state } = ctx;
  return viewer.frustumScale(state.cameras) * 4 || 1;
}

function applyGeom() {
  const { state, viewer } = ctx;
  const cam = state.cameras[state.refIdx] || state.cameras[0];
  if (!cam) return;
  const fov = +$("gFov").value;
  const depth = (+$("gDepth").value / 100) * sceneScale();
  $("gFovV").textContent = fov;
  $("gDepthV").textContent = depth.toFixed(2);
  viewer.setTeachingOverlay({
    mat4: cam.cam_to_world,
    w: state.fw,
    h: state.fh,
    fovDeg: fov,
    depth,
  });
}

// --- Tabs ------------------------------------------------------------------
function showSec(sec) {
  for (const b of document.querySelectorAll("#learnBox .tabs button"))
    b.classList.toggle("active", b.dataset.sec === sec);
  for (const s of document.querySelectorAll(".learn-sec"))
    s.classList.toggle("active", s.id === `sec-${sec}`);
  if (sec === "attn") { drawPick(); fetchIntrospect(); }
  if (sec === "geom") applyGeom();
  else ctx.viewer.clearTeaching?.();
}

// --- Public entry ----------------------------------------------------------
export async function initLearn(state, viewer) {
  ctx = { state, viewer };
  $("learnBox").hidden = false;

  if (!built) {
    built = true;

    for (const b of document.querySelectorAll("#learnBox .tabs button"))
      b.addEventListener("click", () => showSec(b.dataset.sec));

    try {
      const { svg, bindFunctions } = await mermaid.render("vggtArchSvg", ARCH_GRAPH);
      $("archGraph").innerHTML = svg;
      bindFunctions?.($("archGraph")); // wires the `click ... call __vggtArch` handlers
    } catch (e) {
      $("archGraph").textContent = "diagram failed: " + e.message;
    }
    __vggtArch("agg");

    $("geomMath").innerHTML = geomHtml();
    $("trainNotes").innerHTML = trainHtml();

    mixChart = new Chart($("dataMix"), {
      type: "doughnut",
      data: {
        labels: DATASETS.map((d) => d[0]),
        datasets: [{ data: DATASETS.map((d) => d[1]) }],
      },
      options: {
        plugins: {
          legend: { position: "right", labels: { color: "#e6e6e6", font: { size: 10 } } },
          title: { display: true, text: "Indicative training-data mixture (not official)", color: "#9aa0a6" },
        },
      },
    });

    $("attnPick").addEventListener("click", (ev) => {
      const cv = $("attnPick");
      const r = cv.getBoundingClientRect();
      attnQuery = [
        ((ev.clientX - r.left) / r.width) * ctx.state.fw,
        ((ev.clientY - r.top) / r.height) * ctx.state.fh,
      ];
      drawPick();
      fetchIntrospect();
    });
    $("attnFrame").addEventListener("input", () => { drawPick(); scheduleFetch(); });
    $("attnLayer").addEventListener("input", scheduleFetch);
    for (const id of ["gFov", "gDepth"])
      $(id).addEventListener("input", applyGeom);
  }

  // Per-reconstruction refresh.
  const n = state.num_images || state.frameImgs.length;
  $("attnFrame").max = String(Math.max(0, n - 1));
  $("attnFrame").value = String(Math.min(+$("attnFrame").value, n - 1));
  attnQuery = [state.fw / 2, state.fh / 2];
  drawPick();
}
