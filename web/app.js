import { Viewer } from "/viewer.js?v=2";

const $ = (id) => document.getElementById(id);
const viewer = new Viewer($("viewer"));

const state = {
  session: null,
  frameUrls: [],
  frameImgs: [],   // loaded HTMLImageElement per frame
  fw: 0,
  fh: 0,
  cameras: [],
  refIdx: 0,
  queryPts: [],    // [[x,y], ...] in model pixel coords (on refIdx frame)
  tracks: null,    // tracks[q][frame] = [x,y,visible]
};

function setStatus(msg, isErr = false) {
  const el = $("status");
  el.textContent = msg;
  el.className = isErr ? "err" : "";
}

$("run").addEventListener("click", reconstruct);
$("flip").addEventListener("click", () => viewer.flipVertical());
$("track").addEventListener("click", runTracking);
$("clearPts").addEventListener("click", () => {
  state.queryPts = [];
  state.tracks = null;
  renderTracking();
});
$("dlGlb").addEventListener("click", () => {
  if (state.session) window.open(`/api/asset/${state.session}.glb`, "_blank");
});
$("dlCam").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(state.cameras, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "cameras.json";
  a.click();
});

async function reconstruct() {
  const files = $("files").files;
  if (!files.length) {
    setStatus("Select at least one image.", true);
    return;
  }
  const fd = new FormData();
  for (const f of files) fd.append("images", f);

  $("run").disabled = true;
  setStatus(`Reconstructing ${files.length} image(s)…`);
  try {
    const res = await fetch("/api/reconstruct", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();

    state.session = data.session_id;
    state.frameUrls = data.frame_urls;
    state.fw = data.frame_width;
    state.fh = data.frame_height;
    state.cameras = data.cameras;
    state.refIdx = 0;
    state.queryPts = [];
    state.tracks = null;

    viewer.clear();
    await viewer.loadPointCloud(data.pointcloud_url);
    const scale = viewer.frustumScale(data.cameras);
    for (const cam of data.cameras) {
      viewer.addFrustum(cam.cam_to_world, cam.intrinsic, state.fw, state.fh, scale);
    }

    await loadFrameImages();
    $("trackBox").hidden = false;   // must be visible so #refCanvas can measure its width
    renderTracking();
    renderCameraTable();

    $("camBox").hidden = false;
    $("flip").disabled = false;
    $("dlGlb").disabled = false;
    $("dlCam").disabled = false;
    $("track").disabled = false;
    $("clearPts").disabled = false;
    setStatus(`Done — ${data.num_images} cameras, point cloud loaded.`);
  } catch (e) {
    setStatus(`Reconstruction failed: ${e.message}`, true);
  } finally {
    $("run").disabled = false;
  }
}

function loadFrameImages() {
  return Promise.all(
    state.frameUrls.map(
      (url) =>
        new Promise((resolve) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.src = url;
        })
    )
  ).then((imgs) => {
    state.frameImgs = imgs;
  });
}

const COLORS = ["#ff5252", "#4caf50", "#ffc107", "#e040fb", "#00e5ff", "#ff9100"];

function renderTracking() {
  drawThumbGrid();
  drawRefCanvas();
  drawTrackStrip();
}

// Small 3-col grid — used only to pick the reference frame.
function drawThumbGrid() {
  const wrap = $("thumbs");
  wrap.innerHTML = "";
  state.frameImgs.forEach((img, i) => {
    const div = document.createElement("div");
    div.className = "thumb" + (i === state.refIdx ? " sel" : "");

    const cv = document.createElement("canvas");
    const dispW = 100;
    const dispH = Math.round((state.fh / state.fw) * dispW);
    cv.width = dispW;
    cv.height = dispH;
    cv.getContext("2d").drawImage(img, 0, 0, dispW, dispH);

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = i === state.refIdx ? `#${i} ref` : `#${i}`;
    div.appendChild(cv);
    div.appendChild(badge);

    div.addEventListener("click", () => {
      if (i === state.refIdx) return;
      state.refIdx = i;
      state.queryPts = [];
      state.tracks = null;
      renderTracking();
    });

    wrap.appendChild(div);
  });
}

// Enlarged reference frame — where the user clicks query points.
function drawRefCanvas() {
  const cv = $("refCanvas");
  const img = state.frameImgs[state.refIdx];
  if (!img) return;

  const dispW = cv.clientWidth || cv.parentElement.clientWidth || 300;
  const dispH = Math.round((state.fh / state.fw) * dispW);
  cv.width = dispW;
  cv.height = dispH;
  const ctx = cv.getContext("2d");
  ctx.drawImage(img, 0, 0, dispW, dispH);

  const sx = dispW / state.fw;
  const sy = dispH / state.fh;

  state.queryPts.forEach(([x, y], qi) => {
    dot(ctx, x * sx, y * sy, COLORS[qi % COLORS.length], true);
  });
  if (state.tracks) {
    state.tracks.forEach((perFrame, qi) => {
      const [x, y, vis] = perFrame[state.refIdx];
      dot(ctx, x * sx, y * sy, COLORS[qi % COLORS.length], vis >= 0.5);
    });
  }

  cv.onclick = (ev) => {
    const rect = cv.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * state.fw;
    const py = ((ev.clientY - rect.top) / rect.height) * state.fh;
    state.queryPts.push([px, py]);
    state.tracks = null;
    renderTracking();
  };
}

// Full-width bottom strip — all frames concatenated horizontally with the
// tracked points drawn (filled = visible, hollow = occluded).
function drawTrackStrip() {
  const imgs = state.frameImgs;
  if (!imgs.length) return;

  const H = 220;
  const fdW = Math.round((state.fw / state.fh) * H);
  const cv = $("stripCanvas");
  cv.width = fdW * imgs.length;
  cv.height = H;
  const ctx = cv.getContext("2d");

  const sx = fdW / state.fw;
  const sy = H / state.fh;

  imgs.forEach((img, i) => {
    const ox = i * fdW;
    ctx.drawImage(img, ox, 0, fdW, H);

    if (state.tracks) {
      state.tracks.forEach((perFrame, qi) => {
        const [x, y, vis] = perFrame[i];
        dot(ctx, ox + x * sx, y * sy, COLORS[qi % COLORS.length], vis >= 0.5);
      });
    }

    ctx.fillStyle = "#000a";
    ctx.fillRect(ox + 2, 2, 26, 16);
    ctx.fillStyle = "#fff";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "top";
    ctx.fillText(`#${i}`, ox + 5, 4);
  });

  const stripEl = $("trackStrip");
  const firstShow = stripEl.hidden;
  stripEl.hidden = false;
  // Strip just appeared → its grid row shrinks the viewer; trigger a resize.
  if (firstShow) window.dispatchEvent(new Event("resize"));
}

function dot(ctx, x, y, color, filled) {
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, Math.PI * 2);
  if (filled) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

async function runTracking() {
  if (!state.queryPts.length) {
    setStatus("Click at least one point on the reference frame.", true);
    return;
  }
  $("track").disabled = true;
  setStatus(`Tracking ${state.queryPts.length} point(s)…`);
  try {
    const res = await fetch("/api/track", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.session,
        ref_image_id: state.refIdx,
        query_points: state.queryPts,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    state.tracks = (await res.json()).tracks;
    renderTracking();
    setStatus("Tracking done — filled = visible, hollow = occluded.");
  } catch (e) {
    setStatus(`Tracking failed: ${e.message}`, true);
  } finally {
    $("track").disabled = false;
  }
}

function renderCameraTable() {
  const fmt = (v) => v.toFixed(3);
  let html =
    "<table><tr><th>img</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th></tr>";
  for (const c of state.cameras) {
    const k = c.intrinsic;
    html += `<tr><td>${c.image_id}</td><td>${fmt(k[0][0])}</td><td>${fmt(
      k[1][1]
    )}</td><td>${fmt(k[0][2])}</td><td>${fmt(k[1][2])}</td></tr>`;
  }
  html += "</table>";
  $("cams").innerHTML = html;
}
