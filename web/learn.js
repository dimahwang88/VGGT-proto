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
    h: `<p>VGGT ingests a batched stack of ${kx(String.raw`N`, false)} views
    as one tensor:</p>
    ${kx(String.raw`\mathbf{I}\in\mathbb{R}^{B\times N\times 3\times H\times W},\quad B=1,\quad H,W\equiv 0\bmod 14`)}
    <p>Preprocessing (<code>vggt/utils/load_fn.py</code> →
    <code>load_and_preprocess_images</code>): resize so the longer side is the
    configured target, pad to a multiple of the patch size (14), normalize to
    DINOv2 mean/std. EXIF orientation is applied <i>by this app</i> before the
    call (see <code>backend/app.py</code> — the model itself doesn't honor
    EXIF).</p>
    <p><b>Coordinate frame.</b> All extrinsics, depth, and point predictions
    are expressed in the world frame defined as <b>camera 1 = identity</b>.
    The first camera token is special-cased inside the aggregator (it
    anchors the global frame). This is the exact reason the tracker's
    "fast path" works only when the reference frame is index&nbsp;0 — the
    pre-computed tokens already live in that frame.</p>`,
  },

  patch: {
    t: "Tokenization (DINOv2 ViT-L/14 + special tokens)",
    h: `<p><b>1. Patch embedding.</b> A single conv strides over the image:</p>
    ${kx(String.raw`\text{PatchEmbed}: \;\text{Conv2d}(3\to C,\;k=14,\;s=14),\quad C\approx 1024`)}
    <p>so an image (3, H, W) becomes patch tokens
    ${kx(String.raw`\big(\tfrac{H}{14}\!\cdot\!\tfrac{W}{14}=P,\;C\big)`, false)}.
    Weights init from <b>DINOv2 ViT-L/14</b> (self-supervised, registers
    variant).</p>
    <p><b>2. Special tokens (per image).</b> The aggregator prepends learned
    tokens to every image:</p>
    <ul>
      <li><b>1× camera token</b> — later read by the camera head.</li>
      <li><b>4× register tokens</b> — absorb high-norm artefacts (Darcet et al.
      2023, "Vision Transformers Need Registers"); they exist so patch tokens
      stay clean.</li>
    </ul>
    <p>Total length per image:
    ${kx(String.raw`S = \underbrace{1}_{\text{cam}} + \underbrace{4}_{\text{reg}} + P`, false)}.
    The aggregator returns the <b>patch start index</b>
    ${kx(String.raw`\texttt{ps\_idx}=5`, false)} so heads know where patch
    tokens begin. The batched layout the aggregator works on is</p>
    ${kx(String.raw`\mathbf{X}\in\mathbb{R}^{B\times N\times S\times C}`)}
    <p><b>3. Positional info.</b> 2-D sinusoidal / DINOv2 pos-embed is added
    to patch tokens; camera & register tokens get their own learned position
    embeddings. Source: <code>vggt/models/aggregator.py</code>.</p>`,
  },

  agg: {
    t: "Alternating-Attention Aggregator (the engine)",
    h: `<p><b>Refresher — pre-LN transformer block:</b></p>
    ${kx(String.raw`\begin{aligned}
       \mathbf{x} &\leftarrow \mathbf{x} + \mathrm{MSA}(\mathrm{LN}(\mathbf{x}))\\
       \mathbf{x} &\leftarrow \mathbf{x} + \mathrm{MLP}(\mathrm{LN}(\mathbf{x}))
       \end{aligned}`)}
    <p>with ${kx(String.raw`\mathrm{MSA}(x)=\bigoplus_{h=1}^{H}\mathrm{Attn}(xW^Q_h, xW^K_h, xW^V_h)\,W^O,\;\;
       \mathrm{Attn}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_k})V`)}
       and ${kx(String.raw`\mathrm{MLP}(x)=W_2\,\mathrm{GELU}(W_1 x)`, false)}, MLP ratio = 4.</p>
    <p><b>The VGGT twist.</b> Each aggregator block contains <b>two</b>
    such sub-blocks back-to-back; the only difference between them is which
    tokens are allowed to attend to which. Given input
    ${kx(String.raw`\mathbf{X}\in\mathbb{R}^{B\times N\times S\times C}`, false)}:</p>
    <pre style="background:#0d0e10;padding:8px;border-radius:4px;font-size:12px;line-height:1.4">
<span style="color:#9ad">def</span> aa_block(X):                          <span style="color:#7a8">// shape (B,N,S,C)</span>
    <span style="color:#7a8"># --- frame-wise sub-block ---</span>
    Xf = X.reshape(B*N, S, C)              <span style="color:#7a8">// each image isolated</span>
    Xf = Xf + MSA<sub>frame</sub>(LN(Xf))            <span style="color:#7a8">// mask = block-diagonal</span>
    Xf = Xf + MLP<sub>frame</sub>(LN(Xf))
    X  = Xf.reshape(B, N, S, C)
    <span style="color:#7a8"># --- global sub-block ---</span>
    Xg = X.reshape(B, N*S, C)              <span style="color:#7a8">// all tokens, all images</span>
    Xg = Xg + MSA<sub>global</sub>(LN(Xg))           <span style="color:#7a8">// full attention</span>
    Xg = Xg + MLP<sub>global</sub>(LN(Xg))
    <span style="color:#9ad">return</span> Xg.reshape(B, N, S, C)</pre>
    <p><b>Costs.</b> Frame-wise:
    ${kx(String.raw`\mathcal{O}(N\cdot S^2 \cdot C)`, false)};
    global: ${kx(String.raw`\mathcal{O}((NS)^2 \cdot C)`, false)}. Alternating
    avoids paying the global cost twice as often as needed while still fusing
    views every block.</p>
    <p><b>Stack.</b> The public 1B checkpoint uses ${kx(String.raw`L\approx 24`, false)}
    AA blocks at ViT-L/14 dims ${kx(String.raw`(C\!=\!1024,\;\text{heads}\!=\!16,\;\text{MLP}\!\times\!4)`, false)}
    initialized from DINOv2. Exact numbers for your installed checkpoint live
    in <code>vggt/models/aggregator.py</code>.</p>
    <p>The block-diagram in the <b>Attention</b> tab visualizes the two masks
    and the reshape steps end-to-end.</p>`,
  },

  cam: {
    t: "Camera head — iterative pose regression",
    h: `<p>Reads only the ${kx(String.raw`N`, false)} camera tokens (one per
    image), pulled from the aggregator at <code>ps_idx − 5</code>:</p>
    ${kx(String.raw`\mathbf{c}\in\mathbb{R}^{B\times N\times C}`)}
    <p><b>1. Cross-view trunk.</b> A small transformer attends <i>across the N
    camera tokens</i> — so the cameras "talk to each other" and the predicted
    poses become mutually consistent (you can't pick a per-image pose
    independently; the trunk enforces a joint solution).</p>
    <p><b>2. Iterative refinement.</b> The trunk is run ${kx(String.raw`T=4`, false)}
    times, each pass predicting a residual update
    ${kx(String.raw`\Delta\boldsymbol{\theta}_t`, false)} on top of the previous
    pose-encoding estimate ${kx(String.raw`\boldsymbol{\theta}_{t-1}`, false)}
    (a learned unrolled optimizer, in the spirit of GANs/Plucker-net heads):</p>
    ${kx(String.raw`\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \Delta\boldsymbol{\theta}_t,\quad
                    \boldsymbol{\theta}\in\mathbb{R}^{9}=[\,\mathbf{t}\;|\;\mathbf{q}\;|\;\text{FOV}_x,\text{FOV}_y\,]`)}
    <p>Camera 1 is gauge-fixed to identity (its prediction is overwritten by
    the canonical pose). The official wrapper returns a length-T list and the
    repo uses <code>[-1]</code> — see <code>vggt_runner.reconstruct</code>.</p>
    <p><b>3. Decoding to ${kx(String.raw`(R,\mathbf{t},K)`, false)}.</b>
    The quaternion is normalized then expanded (formula in the Geometry tab);
    intrinsics: principal point at ${kx(String.raw`(W/2,H/2)`, false)}, focal
    from FOV: ${kx(String.raw`f_x=(W/2)/\tan(\text{FOV}_x/2)`, false)}. Done by
    <code>vggt/utils/pose_enc.py::pose_encoding_to_extri_intri</code>.</p>
    <p>Source: <code>vggt/heads/camera_head.py</code>.</p>`,
  },

  depth: {
    t: "DPT depth head — token reassembly",
    h: `<p><b>Refresher — DPT</b> (Ranftl et al. 2021). Take patch tokens from
    several transformer layers, project them, reshape (P, C) ↦
    ${kx(String.raw`(C',\tfrac{H}{14},\tfrac{W}{14})`, false)}, and use
    transposed/regular convs to produce a multi-scale feature pyramid that an
    FPN-style fusion module ("RefineNet") upsamples to the input resolution.</p>
    <p>VGGT taps <b>4 aggregator layers</b> (e.g., layers {4, 11, 17, 23} —
    early to late). For each, per image:</p>
    ${kx(String.raw`
       \underbrace{(P,C)}_{\text{patch tokens at layer }\ell}\;\xrightarrow{\text{1×1 conv}}\;
       (C'_\ell,P)\;\xrightarrow{\text{reshape}}\;
       (C'_\ell,\tfrac{H}{14},\tfrac{W}{14})\;\xrightarrow{\text{up/down}}\;
       (C'_\ell,\tfrac{H}{s_\ell},\tfrac{W}{s_\ell})`)}
    <p>with strides ${kx(String.raw`s_\ell\in\{4,8,16,32\}`, false)}. RefineNet
    fuses them coarse→fine into a single ${kx(String.raw`(C_o,H,W)`, false)} map,
    finally projected to two channels:</p>
    ${kx(String.raw`\hat{d}(u,v)\in\mathbb{R}_{>0},\qquad \hat{\sigma}(u,v)\in\mathbb{R}_{>0}`)}
    <p><b>Aleatoric confidence.</b> ${kx(String.raw`\hat{\sigma}`, false)} is the
    Laplacian scale used at training: pixels VGGT is uncertain about (textureless
    regions, occlusions) are down-weighted in the loss
    ${kx(String.raw`\mathcal{L}=|\hat{d}-d|\cdot e^{-\hat{\sigma}}+\alpha\hat{\sigma}`, false)}
    (see Training tab). At inference, ${kx(String.raw`\hat{\sigma}`, false)} is what the
    confidence-floor in <code>vggt_runner.reconstruct</code> filters on to keep
    only trustworthy points in the GLB.</p>
    <p>Source: <code>vggt/heads/dpt_head.py</code>.</p>`,
  },

  point: {
    t: "DPT point head (the 'direct' 3D)",
    h: `<p>Identical DPT decoder structure to the depth head but with a 3-channel
    output ${kx(String.raw`\hat{\mathbf{X}}(u,v)\in\mathbb{R}^3`, false)}
    expressed in the <b>camera-1 world frame</b> (no extrinsic chain needed at
    inference). Plus a confidence channel.</p>
    <p>The paper finds <b>depth → unproject</b> more accurate for clean clouds,
    so this app uses depth + ${kx(String.raw`K,R,\mathbf{t}`, false)}:</p>
    ${kx(String.raw`\mathbf{X}_w=R^{\!\top}\!\bigl(\hat{d}\,K^{-1}\![u,v,1]^{\!\top}-\mathbf{t}\bigr)`)}
    <p>(<code>vggt/utils/geometry.py::unproject_depth_map_to_point_map</code>).
    The point head still gets supervised — it stabilizes training. Source:
    <code>vggt/heads/dpt_head.py</code>.</p>`,
  },

  track: {
    t: "Track head — CoTracker-style temporal refinement",
    h: `<p>Inputs: dense features (built from aggregator tokens of the
    target layers), the original images, ${kx(String.raw`\texttt{ps\_idx}`, false)},
    and a query tensor ${kx(String.raw`\mathbf{Q}\in\mathbb{R}^{B\times Q\times 2}`, false)}
    of query pixels in frame 0. (When the user picks frame ${kx(String.raw`k\!\neq\!0`, false)},
    this app reorders the batch so ${kx(String.raw`k`, false)} is first and
    re-runs the aggregator — see <code>vggt_runner.track</code>.)</p>
    <p><b>1. Feature pyramids.</b> Build per-frame dense feature maps
    ${kx(String.raw`\mathbf{F}_n`, false)} via DPT-like projections of the same
    aggregator tokens used by depth (no separate backbone).</p>
    <p><b>2. Correlation / cost volumes.</b> For each query
    ${kx(String.raw`\mathbf{q}`, false)} encode its feature
    ${kx(String.raw`\phi(\mathbf{q})`, false)} and compute, per frame
    ${kx(String.raw`n`, false)}, a local cost volume around the current track
    estimate ${kx(String.raw`\hat{\mathbf{p}}_n`, false)}:</p>
    ${kx(String.raw`\mathbf{c}_n(\Delta)=\langle\phi(\mathbf{q}),\,\mathbf{F}_n[\hat{\mathbf{p}}_n+\Delta]\rangle`)}
    <p><b>3. Iterative state updates</b> (CoTracker style, ${kx(String.raw`T\!=\!4`, false)} iters):
    a small transformer treats each query as a temporal state across frames and
    produces residuals on track and visibility logits:</p>
    ${kx(String.raw`(\hat{\mathbf{p}}_n,\;\hat{v}_n)\;\leftarrow\;
                    (\hat{\mathbf{p}}_n,\;\hat{v}_n)\;+\;\text{Update}(\mathbf{c}_n,\;\hat{\mathbf{p}}_n,\;\hat{v}_n)`)}
    <p><b>Outputs.</b> Final tracks ${kx(String.raw`\hat{\mathbf{p}}\in\mathbb{R}^{N\times Q\times 2}`, false)}
    and visibilities ${kx(String.raw`\hat{v}\in[0,1]^{N\times Q}`, false)} —
    that's what the Point-tracking panel renders.</p>
    <p>Source: <code>vggt/heads/track_head.py</code>.</p>`,
  },

  out: {
    t: "Outputs in a single forward pass",
    h: `<p>Per view, all consistent in the camera-1 world frame:</p>
    <ul>
      <li>${kx(String.raw`R\in SO(3),\;\mathbf{t}\in\mathbb{R}^3`, false)} extrinsics,
          ${kx(String.raw`K\in\mathbb{R}^{3\times 3}`, false)} intrinsics.</li>
      <li>Dense ${kx(String.raw`\hat{d}(u,v)`, false)} + ${kx(String.raw`\hat{\sigma}_d`, false)}.</li>
      <li>Dense ${kx(String.raw`\hat{\mathbf{X}}(u,v)`, false)} + ${kx(String.raw`\hat{\sigma}_p`, false)}.</li>
      <li>Tracks ${kx(String.raw`\hat{\mathbf{p}}`, false)} +
          visibilities ${kx(String.raw`\hat{v}`, false)} (only when queries given).</li>
    </ul>
    <p>No iterative reconstruction, no bundle adjustment — one feed-forward.
    The whole frustum-and-cloud picture you see in the 3D view shares this
    single coordinate system, which is why the camera frustums and the points
    line up without further alignment.</p>`,
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

// --- Attention tab: step-by-step block anatomy -----------------------------
function tokenLayoutSvg() {
  const tokens = [
    { l: "cam", c: "#ff5252" },
    { l: "r0", c: "#34373d" }, { l: "r1", c: "#34373d" },
    { l: "r2", c: "#34373d" }, { l: "r3", c: "#34373d" },
  ];
  for (let i = 0; i < 10; i++) tokens.push({ l: `p${i}`, c: "#1b5fff" });
  tokens.push({ l: "…", c: "#1b5fff", faded: true });
  const w = 36, gap = 2;
  const cells = tokens.map((t, i) => `
    <g transform="translate(${i * (w + gap)},0)">
      <rect width="${w}" height="${w}" rx="4" fill="${t.c}" opacity="${t.faded ? 0.4 : 1}" />
      <text x="${w / 2}" y="${w / 2 + 4}" text-anchor="middle" fill="#fff" font-size="11">${t.l}</text>
    </g>`).join("");
  return `<svg height="42" width="${tokens.length * (w + gap)}">${cells}</svg>`;
}

function attentionMaskSvg(kind, N = 3, S = 6) {
  const cell = 9, NS = N * S, size = NS * cell;
  let rects = "";
  for (let r = 0; r < NS; r++) {
    for (let c = 0; c < NS; c++) {
      const fr = (r / S) | 0, fc = (c / S) | 0;
      const on = kind === "global" || fr === fc;
      rects += `<rect x="${c * cell}" y="${r * cell}" width="${cell - 1}" height="${cell - 1}" fill="${on ? "#2b6cff" : "#1c1e22"}"/>`;
    }
  }
  // frame-boundary guide lines
  let lines = "";
  for (let i = 1; i < N; i++) {
    const p = i * S * cell;
    lines += `<line x1="${p}" y1="0" x2="${p}" y2="${size}" stroke="#9aa0a6" stroke-dasharray="3 3" stroke-width="0.6"/>`;
    lines += `<line x1="0" y1="${p}" x2="${size}" y2="${p}" stroke="#9aa0a6" stroke-dasharray="3 3" stroke-width="0.6"/>`;
  }
  return `<svg width="${size}" height="${size}" style="background:#101114;border:1px solid #2a2c31;border-radius:4px">${rects}${lines}</svg>`;
}

// ----- Matrix-shape SVG helpers for the three MSA equations ----------------
function mBox(x, y, w, h, fill, label, rowLbl, colLbl) {
  return `
    <text x="${x + w / 2}" y="${y - 5}" text-anchor="middle" fill="#9aa0a6" font-size="10">${colLbl}</text>
    <text x="${x - 6}" y="${y + h / 2 + 4}" text-anchor="end" fill="#9aa0a6" font-size="10">${rowLbl}</text>
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${fill}" stroke="#2a2c31"/>
    <text x="${x + w / 2}" y="${y + h / 2 + 5}" text-anchor="middle" fill="#fff" font-size="13" font-weight="600">${label}</text>`;
}
const mSym = (x, y, s) =>
  `<text x="${x}" y="${y}" text-anchor="middle" fill="#cfd2d6" font-size="22" font-weight="600">${s}</text>`;

// Eq 1:  Q = x W^Q   (and analogously K, V)
function eqProjSvg() {
  const xW = 130, xH = 30, wW = 130, wH = 130;
  const xPos = 60, wPos = xPos + xW + 50, qPos = wPos + wW + 50;
  const xY = 60 + (wH - xH) / 2;
  const midY = xY + xH / 2 + 7;
  return `
  <svg viewBox="0 0 ${qPos + xW + 60} 230" width="100%" style="max-width:760px;background:#0d0e10;border-radius:6px;padding:8px">
    ${mBox(xPos, xY, xW, xH, "#34373d", "x", "S", "C")}
    ${mSym(xPos + xW + 25, midY, "·")}
    ${mBox(wPos, 60, wW, wH, "#2b6cff", "Wᴽ", "C", "C")}
    ${mSym(wPos + wW + 25, midY, "=")}
    ${mBox(qPos, xY, xW, xH, "#ff5252", "Q", "S", "C")}
    <text x="${(xPos + qPos + xW) / 2}" y="215" text-anchor="middle" fill="#9aa0a6" font-size="11">
      same shape for K = xWᴷ and V = xWⱽ (different weights, identical structure)
    </text>
  </svg>`;
}

// Eq 2:  Attn_i(Q,K,V) = softmax( Q_i K_iᵀ / √d_k ) V_i  — per-head attention.
// Includes a "slice indicator" above Q_i showing it is the i-th column-slice
// of the full Q from Eq 1.
function eqAttnSvg() {
  const S = 50, dk = 18;
  // Slice indicator (mini wide Q split into h column slices, slice i highlit).
  const sliceY = 22, sliceH = 18, sliceW = 160, hShown = 8, sliceI = 3;
  const cellW = sliceW / hShown;
  // Row 1: Q_i · K_iᵀ = A_i — shifted down to make room for indicator.
  const r1y = 80;
  const qX = 100, kX = qX + dk + 60, aX = kX + S + 60;
  // centre the slice indicator on the Q_i column
  const indX = qX + dk / 2 - sliceW / 2;
  const hiX = indX + sliceI * cellW;
  let slices = "";
  for (let i = 0; i < hShown; i++) {
    const fill = i === sliceI ? "#ff5252" : "#23272f";
    slices += `<rect x="${indX + i * cellW}" y="${sliceY}" width="${cellW - 0.7}" height="${sliceH}" fill="${fill}" stroke="#2a2c31" stroke-width="0.5"/>`;
  }
  // Row 2: softmax(A_i / √d_k) = w_i,  then  w_i · V_i = out
  const r2y = 190;
  const wX = 100, vX = wX + S + 60, oX = vX + dk + 60;
  const wHeatmap = Array.from({ length: 6 }, (_, i) =>
    `<rect x="${wX}" y="${r2y + (i + 0.5) * (S / 7) - 1}" width="${S}" height="${(S / 8) | 0}" fill="#ffd400" opacity="${0.15 + 0.08 * i}"/>`
  ).join("");
  return `
  <svg viewBox="0 0 ${oX + dk + 80} 290" width="100%" style="max-width:760px;background:#0d0e10;border-radius:6px;padding:8px">
    <defs>
      <marker id="ax" markerWidth="8" markerHeight="8" refX="6" refY="4"
              orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,8 L8,4 z" fill="#ffd400"/>
      </marker>
      <marker id="qs" markerWidth="8" markerHeight="8" refX="6" refY="4"
              orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,8 L8,4 z" fill="#ff5252"/>
      </marker>
    </defs>

    <!-- Slice indicator: Q split into h column-strips, slice i highlighted -->
    <text x="${indX + sliceW / 2}" y="${sliceY - 6}" text-anchor="middle"
          fill="#9aa0a6" font-size="10">
      full Q from Eq 1 (width C)  =  [ Q₁ | Q₂ | … | Qₕ ]   — column-slices of width dₖ
    </text>
    ${slices}
    <line x1="${hiX + cellW / 2}" y1="${sliceY + sliceH + 2}"
          x2="${qX + dk / 2}" y2="${r1y - 4}"
          stroke="#ff5252" stroke-width="1.4" marker-end="url(#qs)"/>
    <text x="${hiX + cellW + 4}" y="${(sliceY + sliceH + r1y) / 2 + 4}"
          fill="#ff5252" font-size="11">slice i  →  Qᵢ</text>

    <!-- Row 1: scores -->
    ${mBox(qX, r1y, dk, S, "#ff5252", "Qᵢ", "S", "dₖ")}
    ${mSym(qX + dk + 30, r1y + S / 2 + 7, "·")}
    ${mBox(kX, r1y, S, dk, "#4caf50", "Kᵢᵀ", "dₖ", "S")}
    ${mSym(kX + S + 30, r1y + S / 2 + 7, "=")}
    ${mBox(aX, r1y, S, S, "#23272f", "Aᵢ", "S", "S")}
    <text x="${aX + S + 6}" y="${r1y + S / 2 + 4}" fill="#9aa0a6" font-size="11">
      "scores"  Aᵢ = QᵢKᵢᵀ
    </text>

    <!-- softmax arrow -->
    <line x1="${aX + S / 2}" y1="${r1y + S + 6}" x2="${wX + S / 2}" y2="${r2y - 6}"
          stroke="#ffd400" stroke-width="1.4" marker-end="url(#ax)"/>
    <text x="${(aX + S / 2 + wX + S / 2) / 2 + 4}" y="${(r1y + S + r2y) / 2}"
          fill="#ffd400" font-size="11">softmax( · / √dₖ )</text>

    <!-- Row 2: weighted V -->
    ${mBox(wX, r2y, S, S, "#23272f", "wᵢ", "S", "S")}
    ${wHeatmap}
    ${mSym(wX + S + 30, r2y + S / 2 + 7, "·")}
    ${mBox(vX, r2y, dk, S, "#2b6cff", "Vᵢ", "S", "dₖ")}
    ${mSym(vX + dk + 30, r2y + S / 2 + 7, "=")}
    ${mBox(oX, r2y, dk, S, "#ff9100", "Attnᵢ", "S", "dₖ")}
    <text x="${(wX + oX + dk) / 2}" y="${r2y + S + 28}" text-anchor="middle"
          fill="#9aa0a6" font-size="11">
      analogous slices: Kᵢ from K, Vᵢ from V (same column-slice index i)
    </text>
  </svg>`;
}

// Eq 3:  MSA(x) = [Attn_1 || … || Attn_h] · W^O
function eqConcatSvg() {
  // 4 visible slices each (S × d_k) → concat into (S × C); then · W^O = out.
  const S = 60, dk = 18, gap = 6;
  const visible = 4;
  const C = visible * dk + (visible - 1) * gap + 30; // visual C width incl '…'
  const cy = 40;
  let s = "";
  const startX = 60;
  // Four head slices + "…" + label for total
  const heads = ["Attn₁", "Attn₂", "…", "Attnₕ"];
  let cur = startX;
  heads.forEach((name, i) => {
    if (i === 2) {
      s += `<text x="${cur + dk / 2}" y="${cy + S / 2 + 6}" text-anchor="middle" fill="#cfd2d6" font-size="20">…</text>`;
    } else {
      s += mBox(cur, cy, dk, S, "#ff9100", name, i === 0 ? "S" : "", "dₖ");
    }
    cur += dk + gap;
  });
  // Annotation bracket below the slices
  const concatEndX = cur - gap;
  s += `<path d="M${startX - 4},${cy + S + 8} l0,6 L${concatEndX + 4},${cy + S + 14} l0,-6"
            stroke="#9aa0a6" fill="none"/>`;
  s += `<text x="${(startX + concatEndX) / 2}" y="${cy + S + 32}" text-anchor="middle"
            fill="#9aa0a6" font-size="11">concat on channel axis  →  (S, h·dₖ = C)</text>`;
  // · W^O = output
  const woX = concatEndX + 60;
  const woW = 100, woH = 100;
  const opMidY = cy + S / 2 + 7;
  s += mSym(concatEndX + 25, opMidY, "·");
  s += mBox(woX, cy + (S - woH) / 2, woW, woH, "#2b6cff", "Wᴼ", "C", "C");
  s += mSym(woX + woW + 25, opMidY, "=");
  const outX = woX + woW + 50;
  const outW = 130;
  s += mBox(outX, cy + (S - 30) / 2, outW, 30, "#34373d", "MSA(x)", "S", "C");
  return `
  <svg viewBox="0 0 ${outX + outW + 60} 180" width="100%" style="max-width:760px;background:#0d0e10;border-radius:6px;padding:8px">
    ${s}
  </svg>`;
}

// SVG of multi-head self-attention: x -> Q,K,V projections -> h heads ->
// per-head softmax(QKᵀ/√d_k)V -> concat -> W^O. Compact and self-labelled.
function msaDiagramSvg() {
  const heads = ["head 1", "head 2", "head 3", "…", "head h=16"];
  const W = 760, H = 500;
  const box = (x, y, w, h, fill, label, sub = "") => `
    <g>
      <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6"
            fill="${fill}" stroke="#2a2c31" />
      <text x="${x + w / 2}" y="${y + h / 2 - 2}" text-anchor="middle"
            fill="#fff" font-size="12" font-weight="600">${label}</text>
      ${sub ? `<text x="${x + w / 2}" y="${y + h / 2 + 13}" text-anchor="middle"
            fill="#9aa0a6" font-size="10">${sub}</text>` : ""}
    </g>`;
  const arrow = (x1, y1, x2, y2) => `
    <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
          stroke="#9aa0a6" stroke-width="1.2" marker-end="url(#a)" />`;
  const headRows = heads.map((name, i) => {
    const y = 215 + i * 28;
    return `
      ${box(60, y, 640, 22, i === 3 ? "#101114" : "#1c2434",
        i === 3 ? "…" :
        `${name}:  Attn(Q${i + 1}, K${i + 1}, V${i + 1}) = softmax(Q${i + 1}K${i + 1}ᵀ / √d_k) V${i + 1}     →   (S, d_k=64)`)}
    `;
  }).join("");
  return `
  <svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:760px;background:#0d0e10;border-radius:6px;padding:6px">
    <defs>
      <marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="4"
              orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,8 L8,4 z" fill="#9aa0a6"/>
      </marker>
    </defs>
    ${box(280, 8, 200, 30, "#34373d", "input  x", "(S, C) — one image's tokens")}
    ${arrow(380, 38, 150, 70)} ${arrow(380, 38, 380, 70)} ${arrow(380, 38, 610, 70)}
    ${box(60, 70, 180, 32, "#2b6cff", "Q = x W^Q", "(S, C)")}
    ${box(290, 70, 180, 32, "#2b6cff", "K = x W^K", "(S, C)")}
    ${box(520, 70, 180, 32, "#2b6cff", "V = x W^V", "(S, C)")}
    ${arrow(150, 102, 150, 130)} ${arrow(380, 102, 380, 130)} ${arrow(610, 102, 610, 130)}
    ${box(60, 130, 640, 32, "#23272f",
      "reshape  →  h = 16 heads, each Q_i, K_i, V_i ∈ ℝ^{S × d_k}     (d_k = C / h = 1024 / 16 = 64)")}
    ${arrow(380, 162, 380, 210)}
    ${headRows}
    ${(() => {
      const yEnd = 215 + heads.length * 28;
      return `
        ${arrow(380, yEnd, 380, yEnd + 30)}
        ${box(60, yEnd + 35, 640, 28, "#23272f",
          "concat heads on channel axis  →  (S, h·d_k = C)")}
        ${arrow(380, yEnd + 63, 380, yEnd + 93)}
        ${box(60, yEnd + 98, 640, 28, "#2b6cff",
          "output  =  (concat) · W^O    ∈  ℝ^{S × C}")}
      `;
    })()}
  </svg>`;
}

// SVG of the MLP / FFN: per-token C -> 4C -> GELU -> C bottleneck.
function mlpDiagramSvg() {
  const u = 24;
  const c = 4 * u; // bar height ∝ width
  const ay = (h) => 70 - h / 2;
  const W = 760;
  const box = (x, w, h, fill, label, sub) => `
    <g>
      <rect x="${x}" y="${ay(h)}" width="${w}" height="${h}" rx="5"
            fill="${fill}" stroke="#2a2c31"/>
      <text x="${x + w / 2}" y="${ay(h) + h + 16}" text-anchor="middle"
            fill="#fff" font-size="12" font-weight="600">${label}</text>
      <text x="${x + w / 2}" y="${ay(h) + h + 30}" text-anchor="middle"
            fill="#9aa0a6" font-size="10">${sub}</text>
    </g>`;
  const arr = (x) => `<line x1="${x}" y1="70" x2="${x + 30}" y2="70"
            stroke="#9aa0a6" stroke-width="1.2" marker-end="url(#b)" />`;
  return `
  <svg viewBox="0 0 ${W} 150" width="100%" style="max-width:760px;background:#0d0e10;border-radius:6px;padding:10px 6px">
    <defs>
      <marker id="b" markerWidth="8" markerHeight="8" refX="6" refY="4"
              orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,8 L8,4 z" fill="#9aa0a6"/>
      </marker>
    </defs>
    ${box(40, 40, c, "#34373d", "x", "ℝ^C   (C = 1024)")}
    ${arr(80)}
    ${box(140, 80, c, "#2b6cff", "W₁ x", "ℝ^{4C} (= 4096)")}
    <text x="220" y="${ay(c) + c + 50}" text-anchor="middle"
          fill="#cfd2d6" font-size="11">↑ 4× expansion</text>
    ${arr(280)}
    ${box(330, 80, c, "#ffd400", "GELU", "elementwise")}
    ${arr(440)}
    ${box(490, 80, c, "#2b6cff", "W₂ (·)", "ℝ^C")}
    ${arr(600)}
    ${box(650, 40, c, "#34373d", "MLP(x)", "ℝ^C")}
  </svg>`;
}

// SVG of the one-patch → one-token mapping: highlight a 14×14 tile in an
// image grid, flatten + project, add a position embedding, become a token.
function patchToTokenSvg() {
  const imgX = 20, imgY = 40, imgW = 150, imgH = 150;
  const grid = 9;
  const cell = imgW / grid;
  let gridLines = "";
  for (let i = 0; i <= grid; i++) {
    const o = i * cell;
    gridLines += `<line x1="${imgX + o}" y1="${imgY}" x2="${imgX + o}" y2="${imgY + imgH}" stroke="#2a2c31" stroke-width="0.5"/>`;
    gridLines += `<line x1="${imgX}" y1="${imgY + o}" x2="${imgX + imgW}" y2="${imgY + o}" stroke="#2a2c31" stroke-width="0.5"/>`;
  }
  const hr = 4, hc = 5; // highlighted cell
  const hl = `<rect x="${imgX + hc * cell}" y="${imgY + hr * cell}" width="${cell}" height="${cell}" fill="#ff5252" opacity="0.75"/>`;
  const tileX = 230, tileY = 60, tileW = 80;
  let tile = `<rect x="${tileX}" y="${tileY}" width="${tileW}" height="${tileW}" fill="#ff5252" opacity="0.5"/>`;
  const sub = 4;
  for (let i = 0; i <= sub; i++) {
    const o = (i * tileW) / sub;
    tile += `<line x1="${tileX + o}" y1="${tileY}" x2="${tileX + o}" y2="${tileY + tileW}" stroke="#2a2c31" stroke-width="0.5"/>`;
    tile += `<line x1="${tileX}" y1="${tileY + o}" x2="${tileX + tileW}" y2="${tileY + o}" stroke="#2a2c31" stroke-width="0.5"/>`;
  }
  return `
  <svg viewBox="0 0 760 250" width="100%" style="max-width:760px;background:#0d0e10;border-radius:6px;padding:8px">
    <defs>
      <marker id="pa" markerWidth="8" markerHeight="8" refX="6" refY="4"
              orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,8 L8,4 z" fill="#9aa0a6"/>
      </marker>
    </defs>
    <text x="${imgX + imgW / 2}" y="${imgY - 8}" text-anchor="middle" fill="#9aa0a6" font-size="11">image (3, H, W)</text>
    <rect x="${imgX}" y="${imgY}" width="${imgW}" height="${imgH}" fill="#1c2434"/>
    ${gridLines}${hl}
    <text x="${imgX + imgW / 2}" y="${imgY + imgH + 16}" text-anchor="middle" fill="#9aa0a6" font-size="11">grid of P = (H/14)(W/14) patches</text>

    <line x1="${imgX + (hc + 1) * cell + 4}" y1="${imgY + (hr + 0.5) * cell}"
          x2="${tileX - 8}" y2="${tileY + tileW / 2}"
          stroke="#9aa0a6" stroke-width="1.2" marker-end="url(#pa)"/>

    <text x="${tileX + tileW / 2}" y="${tileY - 8}" text-anchor="middle" fill="#9aa0a6" font-size="11">one patch — 14×14×3 = 588 values</text>
    ${tile}

    <line x1="${tileX + tileW + 4}" y1="${tileY + tileW / 2}"
          x2="380" y2="${tileY + tileW / 2}"
          stroke="#9aa0a6" stroke-width="1.2" marker-end="url(#pa)"/>

    <rect x="380" y="${tileY + tileW / 2 - 22}" width="150" height="44" rx="5"
          fill="#2b6cff" stroke="#2a2c31"/>
    <text x="455" y="${tileY + tileW / 2 - 4}" text-anchor="middle" fill="#fff" font-size="12" font-weight="600">flatten · Wₚₐₜcₕ</text>
    <text x="455" y="${tileY + tileW / 2 + 12}" text-anchor="middle" fill="#cfd2d6" font-size="10">Wₚₐₜcₕ ∈ ℝ^{C × 588}</text>

    <line x1="535" y1="${tileY + tileW / 2}" x2="580" y2="${tileY + tileW / 2}"
          stroke="#9aa0a6" stroke-width="1.2" marker-end="url(#pa)"/>

    <text x="595" y="${imgY - 8}" text-anchor="middle" fill="#9aa0a6" font-size="11">token</text>
    <rect x="585" y="${imgY}" width="20" height="${imgH}" rx="3" fill="#ff5252" stroke="#2a2c31"/>
    <text x="595" y="${imgY + imgH + 16}" text-anchor="middle" fill="#9aa0a6" font-size="11">ℝ^C, C=1024</text>

    <text x="623" y="${tileY + tileW / 2 + 5}" fill="#cfd2d6" font-size="16">+</text>

    <text x="660" y="${imgY - 8}" text-anchor="middle" fill="#9aa0a6" font-size="11">pos embed</text>
    <rect x="650" y="${imgY}" width="20" height="${imgH}" rx="3" fill="#34373d" stroke="#2a2c31"/>

    <text x="685" y="${tileY + tileW / 2 + 5}" fill="#cfd2d6" font-size="16">=</text>

    <text x="725" y="${imgY - 8}" text-anchor="middle" fill="#9aa0a6" font-size="11">patch token</text>
    <rect x="715" y="${imgY}" width="20" height="${imgH}" rx="3" fill="#1b5fff" stroke="#2a2c31"/>
  </svg>`;
}

function flowStep(title, body, color = "#2b6cff") {
  return `
  <div style="border-left:3px solid ${color};padding:6px 10px;margin:6px 0;background:#1c1e22;border-radius:4px">
    <div style="font-weight:600;font-size:13px">${title}</div>
    <div style="font-size:12px;color:#cfd2d6;margin-top:3px">${body}</div>
  </div>`;
}

function renderAnatomy() {
  const inn = `
  <h3 style="margin:0 0 6px">Inside one Alternating-Attention block</h3>
  <p>Each AA block takes a token grid
     ${kx(String.raw`\mathbf{X}\in\mathbb{R}^{B\times N\times S\times C}`, false)}
     and runs <b>two</b> pre-LN transformer sub-blocks back-to-back — frame-wise
     first, then global. The only difference between them is the reshape
     applied before the multi-head self-attention, which changes <i>which
     tokens are allowed to attend to which</i>. Stack
     ${kx(String.raw`L\!\approx\!24`, false)} of these to get the aggregator.</p>
  <table style="border-collapse:collapse;margin:8px 0;font-size:12px">
    <tr><th style="text-align:left;padding:3px 10px;border:1px solid #2a2c31;background:#1c1e22">symbol</th>
        <th style="text-align:left;padding:3px 10px;border:1px solid #2a2c31;background:#1c1e22">meaning</th>
        <th style="text-align:left;padding:3px 10px;border:1px solid #2a2c31;background:#1c1e22">typical value</th></tr>
    <tr><td style="padding:3px 10px;border:1px solid #2a2c31">${kx(String.raw`B`, false)}</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31"><b>batch</b> — number of scenes per forward pass</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31">1 in this app</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #2a2c31">${kx(String.raw`N`, false)}</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31"><b>number of views / images</b> in the scene — what you uploaded</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31">≈ 1–50</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #2a2c31">${kx(String.raw`S`, false)}</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31"><b>tokens per image</b>: ${kx(String.raw`1_{\text{cam}}+4_{\text{reg}}+P`, false)} where ${kx(String.raw`P=\tfrac{H}{14}\!\cdot\!\tfrac{W}{14}`, false)}</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31">e.g. 5 + 1369 = 1374 at 518×518</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #2a2c31">${kx(String.raw`C`, false)}</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31"><b>hidden dim</b> — width of every token vector</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31">1024 (ViT-L/14)</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #2a2c31">MSA</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31"><b>Multi-head Self-Attention</b>: ${kx(String.raw`h`, false)} parallel heads of ${kx(String.raw`\mathrm{softmax}(QK^\top/\sqrt{d_k})V`, false)}, concatenated then linearly projected</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31">${kx(String.raw`h\!=\!16`, false)} heads, ${kx(String.raw`d_k\!=\!C/h\!=\!64`, false)}</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #2a2c31">MLP</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31"><b>Multi-Layer Perceptron</b> (a.k.a. FFN): per-token 2-layer feed-forward ${kx(String.raw`W_2\,\mathrm{GELU}(W_1 x)`, false)}</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31">MLP ratio 4: ${kx(String.raw`C\!\to\!4C\!\to\!C`, false)}</td></tr>
    <tr><td style="padding:3px 10px;border:1px solid #2a2c31">LN</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31"><b>LayerNorm</b> — normalizes each token vector to zero mean / unit variance, then scales & shifts with learned ${kx(String.raw`\gamma,\beta`, false)}</td>
        <td style="padding:3px 10px;border:1px solid #2a2c31">"pre-LN" = LN inside the residual branch</td></tr>
  </table>

  <h4 style="margin:18px 0 6px">Refresher — LayerNorm (LN)</h4>
  <p>For a single token vector ${kx(String.raw`\mathbf{x}\in\mathbb{R}^{C}`, false)} (one of the
     ${kx(String.raw`B\!\cdot\!N\!\cdot\!S`, false)} tokens in the grid), LN computes
     per-token statistics <i>across the channel axis</i>:</p>
  ${kx(String.raw`\mu=\tfrac{1}{C}\!\sum_{i=1}^{C}\!x_i,\qquad
                  \sigma^2=\tfrac{1}{C}\!\sum_{i=1}^{C}(x_i-\mu)^2`)}
  <p>then normalizes and applies a learned per-channel affine
     (${kx(String.raw`\gamma,\beta\in\mathbb{R}^{C}`, false)}):</p>
  ${kx(String.raw`\mathrm{LN}(\mathbf{x})_i=\gamma_i\cdot\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta_i`)}
  <p>LN is applied <b>independently per token</b> — it never mixes information across
     tokens or across the batch. (Contrast with BatchNorm, which normalizes across the
     batch and would couple tokens across scenes.) That's why the same LN module works
     unchanged for both AA sub-blocks: the frame-wise reshape
     ${kx(String.raw`(B\!\cdot\!N,\,S,\,C)`, false)} and the global reshape
     ${kx(String.raw`(B,\,N\!\cdot\!S,\,C)`, false)} both leave the per-token
     ${kx(String.raw`C`, false)} axis intact.</p>

  <h5 style="margin:10px 0 4px">Pre-LN vs post-LN — and why VGGT uses pre-LN</h5>
  <p>Look at where LN sits in the residual:</p>
  ${kx(String.raw`\underbrace{\mathbf{x}\leftarrow\mathbf{x}+\mathrm{MSA}\!\big(\mathrm{LN}(\mathbf{x})\big)}_{\textbf{pre-LN — VGGT}}
     \quad\text{vs.}\quad
     \underbrace{\mathbf{x}\leftarrow\mathrm{LN}\!\big(\mathbf{x}+\mathrm{MSA}(\mathbf{x})\big)}_{\text{post-LN — original Transformer}}`)}
  <p>In pre-LN, LN is <i>inside</i> the residual branch — the residual path
     ${kx(String.raw`\mathbf{x}+\cdots`, false)} carries the un-normalized signal
     straight through. Pre-LN became standard because, with deep stacks (24+ blocks like the aggregator),
     it gives noticeably better gradient flow and trains <b>without learning-rate warmup</b>
     (Xiong et al. 2020). MSA / MLP see inputs with controlled scale so logits don't
     explode, while the residual stream can accumulate features across all
     ${kx(String.raw`L`, false)} layers without LN squashing them at every step —
     important for an aggregator that has to carry geometry information end-to-end.
     Every modern ViT/LLM (DINOv2, Llama, GPT-NeoX, …) is pre-LN; VGGT inherits this
     from ViT-L/14.</p>

  <h4 style="margin:18px 0 6px">Refresher — Multi-head Self-Attention (MSA)</h4>
  <p>Three learned linear projections turn each token into a <b>query</b>,
     <b>key</b>, and <b>value</b>. The set of all queries attend to the set of
     all keys (softmax-normalized) to produce weights, which then aggregate the
     values. Doing this in ${kx(String.raw`h`, false)} parallel <i>heads</i>
     (each operating on a ${kx(String.raw`d_k=C/h=64`, false)}-dim slice) lets
     the model attend to several different relations at once:</p>
  <p><b>1. Projections</b> — three learned linear maps split each input token
     into a query, key, and value:</p>
  ${kx(String.raw`Q=xW^Q,\qquad K=xW^K,\qquad V=xW^V`)}
  ${eqProjSvg()}

  <p><b>2. Per-head attention</b> — within a single head ${kx(String.raw`i`, false)},
     queries score against keys, softmax-normalize, then take a weighted sum of values:</p>

  <div style="background:#1c2434;border-left:3px solid #ffd400;padding:10px 14px;border-radius:4px;margin:8px 0;font-size:13px">
    <div style="font-weight:600;margin-bottom:4px">${kx(String.raw`C`, false)} vs ${kx(String.raw`d_k`, false)} — easy to confuse, important to keep straight</div>
    <table style="border-collapse:collapse;font-size:12px;margin-top:4px">
      <tr>
        <td style="padding:3px 12px 3px 0">${kx(String.raw`C`, false)}</td>
        <td style="padding:3px 12px 3px 0">full hidden dim — the model width</td>
        <td style="padding:3px 0;color:#9ad">1024</td>
      </tr>
      <tr>
        <td style="padding:3px 12px 3px 0">${kx(String.raw`d_k`, false)}</td>
        <td style="padding:3px 12px 3px 0"><b>per-head</b> dim — a slice of ${kx(String.raw`C`, false)}: ${kx(String.raw`d_k=C/h`, false)}</td>
        <td style="padding:3px 0;color:#9ad">1024/16 = <b>64</b></td>
      </tr>
    </table>
    <p style="margin:6px 0 0;color:#cfd2d6">
      Eq 1 (projection) and Eq 3 (concat + ${kx(String.raw`W^O`, false)}) operate
      at <b>full width ${kx(String.raw`C`, false)}</b>. Eq 2 (per-head
      attention) operates at the <b>narrower ${kx(String.raw`d_k`, false)}</b>
      — that's why ${kx(String.raw`Q_i, K_i, V_i`, false)} are drawn as thin
      bars below: each head sees only ${kx(String.raw`64`, false)} of the
      ${kx(String.raw`1024`, false)} channels. The score matrix
      ${kx(String.raw`A_i\in\mathbb{R}^{S\times S}`, false)} has no ${kx(String.raw`C`, false)} or
      ${kx(String.raw`d_k`, false)} in its shape at all — ${kx(String.raw`d_k`, false)}
      survives only inside the ${kx(String.raw`1/\sqrt{d_k}`, false)}
      temperature scaling.
    </p>
  </div>

  ${kx(String.raw`\mathrm{Attn}_i(Q,K,V)=\mathrm{softmax}\!\Big(\frac{Q_iK_i^{\!\top}}{\sqrt{d_k}}\Big)\,V_i`)}
  ${eqAttnSvg()}
  <p class="hint">The grey ${kx(String.raw`S\!\times\!S`, false)} block is the
     score matrix ${kx(String.raw`A_i`, false)}; softmax along rows normalizes
     it into the attention <i>weights</i> matrix
     ${kx(String.raw`w_i`, false)} (faint yellow bands). Multiplying
     ${kx(String.raw`w_i\cdot V_i`, false)} produces one head's output
     ${kx(String.raw`\mathrm{Attn}_i\in\mathbb{R}^{S\times d_k}`, false)}.
     The ${kx(String.raw`1/\sqrt{d_k}`, false)} factor keeps the softmax temperature
     well-behaved as ${kx(String.raw`d_k`, false)} grows.</p>

  <p><b>3. Concatenate heads and project</b>:</p>
  ${kx(String.raw`\mathrm{MSA}(x)=\big[\mathrm{Attn}_1\;\Vert\;\dots\;\Vert\;\mathrm{Attn}_h\big]\,W^O`)}
  ${eqConcatSvg()}

  <h5 style="margin:12px 0 4px">Putting it together (the full block flow)</h5>
  ${msaDiagramSvg()}
  <p><b>${kx(String.raw`W^Q,W^K,W^V,W^O\in\mathbb{R}^{C\times C}`, false)}</b> are
     the four learned linear layers of MSA. ${kx(String.raw`W^O`, false)} is the
     <b>output projection</b>: after the ${kx(String.raw`h`, false)} heads each
     produce a ${kx(String.raw`(S,d_k)`, false)} slice, they're concatenated on the
     channel axis to ${kx(String.raw`(S,C)`, false)} and multiplied by
     ${kx(String.raw`W^O`, false)}. Without it the heads would never mix — each
     would only write to its own ${kx(String.raw`d_k`, false)} contiguous
     channels — so ${kx(String.raw`W^O`, false)} blends them into a single
     coherent vector before it gets added back into the residual stream.</p>
  <p class="hint">The frame-wise sub-block applies this with a block-diagonal
     mask over ${kx(String.raw`S`, false)} tokens; the global sub-block applies
     the exact same MSA but over ${kx(String.raw`N\!\cdot\!S`, false)} tokens
     with no mask. Same module, different reshape.</p>

  <h4 style="margin:18px 0 6px">Refresher — MLP (FFN inside each block)</h4>
  <p>Applied <b>independently to every token</b> (no token mixing — that job
     belongs to MSA). It's a 2-layer feed-forward net with a 4× channel
     expansion and a GELU non-linearity:</p>
  ${kx(String.raw`\mathrm{MLP}(x)=W_2\,\mathrm{GELU}(W_1 x),\qquad W_1\in\mathbb{R}^{4C\times C},\;W_2\in\mathbb{R}^{C\times 4C}`)}
  ${mlpDiagramSvg()}
  <p class="hint">Roughly two-thirds of a transformer block's parameters live
     in this expansion (${kx(String.raw`2\!\cdot\!C\!\cdot\!4C=8C^2`, false)} for MLP
     vs ${kx(String.raw`4C^2`, false)} for Q,K,V,W^O combined). GELU adds a
     smooth gating non-linearity that empirically beats ReLU for transformers.</p>

  <h4 style="margin:18px 0 6px">One patch → one token (the 1-to-1 mapping)</h4>
  <p>The image is tiled by the patch-embed conv (kernel = stride = 14) into
     <b>non-overlapping</b> 14×14×3 tiles. Each tile — a single
     ${kx(String.raw`14\!\cdot\!14\!\cdot\!3=588`, false)}-number vector — is
     compressed by one learned linear projection into a single
     ${kx(String.raw`C`, false)}-dim <b>token</b>, and a position embedding is
     added so the model knows <i>which</i> patch in the grid it came from:</p>
  ${kx(String.raw`\underbrace{\text{patch}\;\in\mathbb{R}^{588}}_{14\times 14\times 3}\;\xrightarrow{W_{\text{patch}}\in\mathbb{R}^{C\times 588}}\;\text{token}\in\mathbb{R}^{C}\;\xrightarrow{+\,\text{pos embed}}\;\text{patch token}\in\mathbb{R}^{C}`)}
  ${patchToTokenSvg()}
  <p>Stacking all ${kx(String.raw`P=(H/14)(W/14)`, false)} patch tokens gives
     the patch part of the per-image sequence: a
     ${kx(String.raw`(P,\,C)`, false)} matrix. The aggregator then prepends
     <b>5 extra tokens that don't come from any patch</b> — 1 camera + 4
     register tokens (learned vectors); together they make the
     ${kx(String.raw`S = 1\!+\!4\!+\!P`, false)} per-image sequence.</p>
  <p class="hint">Two practical consequences you already see in this app:
   (i) the Attention picker's similarity heatmap is exactly
   ${kx(String.raw`(H/14)\!\times\!(W/14)`, false)} — one cell per patch token,
   not per pixel — and your click snaps to the nearest patch cell.
   (ii) ${kx(String.raw`H,W`, false)} must be multiples of 14 (the backend
   preprocesses to that), otherwise patches don't tile the image cleanly.</p>

  <h4 style="margin:18px 0 6px">Per-image token layout (one row of length S)</h4>
  <div style="overflow-x:auto">${tokenLayoutSvg()}</div>
  <p class="hint">${kx(String.raw`\texttt{cam}`, false)} = camera token (read by
   the camera head). ${kx(String.raw`\texttt{r0..r3}`, false)} = register tokens
   (absorb high-norm artefacts; <code>ps_idx = 5</code> tells heads patches start
   here). ${kx(String.raw`p_i`, false)} = patch tokens, row-major over
   ${kx(String.raw`P=\tfrac{H}{14}\!\cdot\!\tfrac{W}{14}`, false)}.</p>

  <h4 style="margin:18px 0 6px">Step-by-step flow</h4>
  ${flowStep("Step 0 — Input",
    `${kx(String.raw`\mathbf{X}\in\mathbb{R}^{B\times N\times S\times C}`, false)},
     with ${kx(String.raw`C\!\approx\!1024`, false)} (ViT-L/14 width).`)}
  ${flowStep("Step 1 — Frame-wise reshape",
    `Flatten frames into the batch:
     ${kx(String.raw`(B,N,S,C)\to(B\!\cdot\!N,\,S,\,C)`, false)}.
     Each image is now an independent sequence of length ${kx(String.raw`S`, false)}.`)}
  ${flowStep("Step 2 — MSA<sub>frame</sub> + residual",
    `${kx(String.raw`\mathbf{X}\leftarrow\mathbf{X}+\mathrm{MSA}\!\big(\mathrm{LN}(\mathbf{X})\big)`, false)},
     16 heads, mask = <b>block-diagonal</b> (see grid on the right). Cost
     ${kx(String.raw`\mathcal{O}(N S^2 C)`, false)}.`)}
  ${flowStep("Step 3 — MLP + residual",
    `${kx(String.raw`\mathbf{X}\leftarrow\mathbf{X}+\mathrm{MLP}\!\big(\mathrm{LN}(\mathbf{X})\big)`, false)},
     MLP ratio 4: ${kx(String.raw`C\!\to\!4C\!\to\!C`, false)} with GELU.
     Then reshape back to ${kx(String.raw`(B,N,S,C)`, false)}.`)}
  ${flowStep("Step 4 — Global reshape",
    `Concatenate all images' tokens:
     ${kx(String.raw`(B,N,S,C)\to(B,\,N\!\cdot\!S,\,C)`, false)}.
     Tokens of different frames are now in one sequence.`, "#ffd400")}
  ${flowStep("Step 5 — MSA<sub>global</sub> + residual",
    `Same MSA but with <b>full</b> attention over ${kx(String.raw`NS`, false)}
     positions. Cost ${kx(String.raw`\mathcal{O}((NS)^2 C)`, false)}.
     <b>This is where view fusion happens.</b>`, "#ffd400")}
  ${flowStep("Step 6 — MLP + residual, reshape back",
    `As Step 3; output again ${kx(String.raw`(B,N,S,C)`, false)} — ready for the
     next AA block.`, "#ffd400")}

  <h4 style="margin:18px 0 6px">Attention-mask comparison (N=3, S=6 demo)</h4>
  <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">
    <div>
      <div style="font-size:12px;color:#9aa0a6;margin-bottom:4px">Frame-wise — block-diagonal</div>
      ${attentionMaskSvg("frame")}
      <div class="hint" style="margin-top:4px;max-width:170px">Each ${kx(String.raw`S\!\times\!S`, false)} block on the diagonal is one image; off-diagonal blocks are zero.</div>
    </div>
    <div>
      <div style="font-size:12px;color:#9aa0a6;margin-bottom:4px">Global — full ${kx(String.raw`NS\!\times\!NS`, false)}</div>
      ${attentionMaskSvg("global")}
      <div class="hint" style="margin-top:4px;max-width:170px">Every token can attend to every other token of every frame.</div>
    </div>
  </div>

  <p class="hint" style="margin-top:14px">The live picker below uses cosine
   similarity of the chosen patch's residual-stream feature to every other
   patch as a robust proxy for "what does this patch attend to?" — VGGT's fused
   SDPA attention weights aren't exposed without patching the model. The
   <b>cross-frame mixing</b> bars rise with depth: that rise <i>is</i> the
   global sub-block of each AA block doing its job.</p>`;
  $("attnAnatomy").innerHTML = inn;
}

// --- Attention tab: live similarity picker ---------------------------------
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
  for (const b of document.querySelectorAll("#learnView .tabs button"))
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
  // Visibility of #learnView is controlled by the Learn-VGGT toggle button
  // in app.js — initLearn just builds/refreshes content.

  if (!built) {
    built = true;

    for (const b of document.querySelectorAll("#learnView .tabs button"))
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
    renderAnatomy();

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
