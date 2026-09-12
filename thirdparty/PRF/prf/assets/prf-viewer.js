import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";

const BIN_COLORS = [
  [8, 24, 107],
  [0, 72, 168],
  [0, 140, 170],
  [218, 158, 0],
  [215, 72, 0],
  [135, 0, 34],
];
const OBS_DETAIL_COLORS = [
  [8, 24, 107],
  [0, 59, 143],
  [0, 104, 201],
  [0, 151, 178],
  [224, 160, 0],
  [217, 74, 0],
  [135, 0, 34],
];
const BOTTLENECK_COLORS = [
  [244, 109, 67],
  [69, 117, 180],
  [80, 200, 80],
  [120, 120, 120],
];
const BOTTLENECK_NAMES = ["KERNEL", "SPACING", "OBS", "INVALID"];
const EDGES = [0.5, 1, 2, 5, 10];
const OBS_DETAIL_EDGES = [0.35, 0.5, 0.75, 1, 1.5, 2];

function resolutionBin(value, edges = EDGES) {
  if (!(value >= 0) || !Number.isFinite(value)) return -1;
  for (let i = 0; i < edges.length; i += 1) {
    if (value <= edges[i]) return i;
  }
  return edges.length;
}

function colorFor(metric, value, bottleneck) {
  if (metric === "bottleneck") {
    return BOTTLENECK_COLORS[bottleneck] || BOTTLENECK_COLORS[3];
  }
  const detail = metric === "R_obs";
  const bin = resolutionBin(value, detail ? OBS_DETAIL_EDGES : EDGES);
  return bin < 0 ? [120, 120, 120] : (detail ? OBS_DETAIL_COLORS : BIN_COLORS)[bin];
}

async function fetchChecked(url, kind = "arrayBuffer") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} 加载失败：HTTP ${response.status}`);
  return response[kind]();
}

function parsePatches(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || new TextDecoder().decode(buffer.slice(0, 4)) !== "PRF2") {
    throw new Error("patches.bin 不是 PRF2 数据");
  }
  let o = 4;
  const n = view.getUint32(o, true); o += 4;
  const nHull = view.getUint32(o, true); o += 4;
  const nMem = view.getUint32(o, true); o += 4;
  const nView = view.getUint32(o, true); o += 4;
  const take = (Typed, count, bytes) => {
    const end = o + count * bytes;
    if (end > buffer.byteLength) throw new Error("patches.bin 数据被截断");
    const arr = new Typed(buffer.slice(o, end));
    o = end;
    return arr;
  };
  const patchId = take(Int32Array, n, 4);
  const center = take(Float32Array, n * 3, 4);
  const normal = take(Float32Array, n * 3, 4);
  const t1 = take(Float32Array, n * 3, 4);
  const t2 = take(Float32Array, n * 3, 4);
  const radius = take(Float32Array, n, 4);
  const area = take(Float32Array, n, 4);
  const R_obs = take(Float32Array, n, 4);
  const R_kernel = take(Float32Array, n, 4);
  const R_spacing = take(Float32Array, n, 4);
  const R_phys = take(Float32Array, n, 4);
  const bottleneck = take(Uint8Array, n, 1);
  const nVisible = take(Uint16Array, n, 2);
  const pairAngle = take(Float32Array, n, 4);
  const hullOffsets = take(Int32Array, n + 1, 4);
  const hullUv = take(Float32Array, nHull * 2, 4);
  const memberOffsets = take(Int32Array, n + 1, 4);
  const memberXyz = take(Float32Array, nMem * 3, 4);
  const rView = take(Float32Array, n * nView, 4);
  const inside = take(Uint8Array, n * nView, 1);
  if (o !== buffer.byteLength) throw new Error(`patches.bin 长度异常：多出 ${buffer.byteLength - o} 字节`);
  if (hullOffsets[0] !== 0 || hullOffsets[n] !== nHull) throw new Error("hull_offsets 与 hull_uv 不一致");
  if (memberOffsets[0] !== 0 || memberOffsets[n] !== nMem) throw new Error("member_offsets 与 member 数据不一致");
  return {
    n, nView, patchId, center, normal, t1, t2, radius, area,
    R_obs, R_kernel, R_spacing, R_phys, bottleneck, nVisible, pairAngle,
    hullOffsets, hullUv, memberOffsets, memberXyz, rView, inside,
  };
}

function parseGaussians(buffer) {
  const magic = new TextDecoder().decode(buffer.slice(0, 4));
  if (buffer.byteLength < 8 || !["GSC1", "GSC2", "GSC3", "GSF1", "GSF2"].includes(magic)) {
    throw new Error("gaussian_context.bin 不是受支持的 GSC 数据");
  }
  const n = new DataView(buffer).getUint32(4, true);
  const xyzEnd = 8 + n * 3 * 4;
  if (xyzEnd > buffer.byteLength) throw new Error("gaussian_context.bin 数据被截断");
  if (magic === "GSC2" && xyzEnd !== buffer.byteLength) throw new Error("GSC2 数据长度异常");
  let rgb = null;
  let codes = null;
  let bottleneck = null;
  if (magic === "GSC3") {
    const rgbEnd = xyzEnd + n * 3;
    if (rgbEnd !== buffer.byteLength) throw new Error("GSC3 RGB 数据长度异常");
    rgb = new Uint8Array(buffer, xyzEnd, n * 3);
  } else if (magic === "GSF1" || magic === "GSF2") {
    let offset = xyzEnd;
    const takeBytes = (count) => {
      const end = offset + count;
      if (end > buffer.byteLength) throw new Error(`${magic} 数据被截断`);
      const result = new Uint8Array(buffer, offset, count);
      offset = end;
      return result;
    };
    rgb = takeBytes(n * 3);
    codes = {
      R_obs: takeBytes(n),
      R_kernel: takeBytes(n),
      R_spacing: takeBytes(n),
      R_phys: takeBytes(n),
    };
    if (magic === "GSF2") codes.R_obs_detail = takeBytes(n);
    bottleneck = takeBytes(n);
    if (offset !== buffer.byteLength) throw new Error(`${magic} 数据长度异常`);
  }
  return { n, xyz: new Float32Array(buffer, 8, n * 3), rgb, codes, bottleneck };
}

function vec3(src, i) {
  return new THREE.Vector3(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]);
}

function hullWorld(patches, i) {
  const a = patches.hullOffsets[i];
  const b = patches.hullOffsets[i + 1];
  const c = vec3(patches.center, i);
  const t1 = vec3(patches.t1, i);
  const t2 = vec3(patches.t2, i);
  const points = [];
  for (let k = a; k < b; k += 1) {
    const u = patches.hullUv[k * 2];
    const v = patches.hullUv[k * 2 + 1];
    points.push(c.clone().addScaledVector(t1, u).addScaledVector(t2, v));
  }
  return points;
}

function project(view, point) {
  const m = view.w2c;
  const x = m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3];
  const y = m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7];
  const z = m[8] * point.x + m[9] * point.y + m[10] * point.z + m[11];
  if (!(z > 1e-6)) return null;
  return { u: view.fx * x / z + view.cx, v: view.fy * y / z + view.cy, z };
}

function buildPatchSurface(patches) {
  let triangleCount = 0;
  let edgeCount = 0;
  for (let i = 0; i < patches.n; i += 1) {
    const size = patches.hullOffsets[i + 1] - patches.hullOffsets[i];
    triangleCount += Math.max(size - 2, 0);
    edgeCount += size;
  }
  const positions = new Float32Array(triangleCount * 9);
  const colors = new Float32Array(triangleCount * 9);
  const trianglePatch = new Uint32Array(triangleCount);
  const outlinePositions = new Float32Array(edgeCount * 6);
  let triangle = 0;
  let edge = 0;
  for (let i = 0; i < patches.n; i += 1) {
    const points = hullWorld(patches, i);
    for (let k = 1; k < points.length - 1; k += 1) {
      const vertices = [points[0], points[k], points[k + 1]];
      for (let q = 0; q < 3; q += 1) {
        positions[(triangle * 9) + (q * 3)] = vertices[q].x;
        positions[(triangle * 9) + (q * 3) + 1] = vertices[q].y;
        positions[(triangle * 9) + (q * 3) + 2] = vertices[q].z;
      }
      trianglePatch[triangle] = i;
      triangle += 1;
    }
    for (let k = 0; k < points.length; k += 1) {
      const a = points[k];
      const b = points[(k + 1) % points.length];
      outlinePositions.set([a.x, a.y, a.z, b.x, b.y, b.z], edge * 6);
      edge += 1;
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.50,
      side: THREE.DoubleSide,
      depthWrite: true,
    }),
  );
  const outlineGeometry = new THREE.BufferGeometry();
  outlineGeometry.setAttribute("position", new THREE.BufferAttribute(outlinePositions, 3));
  const outlines = new THREE.LineSegments(
    outlineGeometry,
    new THREE.LineBasicMaterial({ color: 0x1b1b1b, transparent: true, opacity: 0.52 }),
  );
  return { mesh, outlines, trianglePatch };
}

function formatResolution(value, invalidLabel = "∞") {
  return value < 0 || !Number.isFinite(value) ? invalidLabel : `${value.toFixed(3)} m`;
}

async function main() {
  const status = document.getElementById("status");
  status.textContent = "正在加载 patch 与 Gaussian 上下文…";
  const [meta, patchBuf, gaussBuf] = await Promise.all([
    fetchChecked("./metadata.json", "json"),
    fetchChecked("./patches.bin"),
    fetchChecked("./gaussian_context.bin"),
  ]);
  if (meta.schema_version !== 2) throw new Error(`metadata schema_version=${meta.schema_version}，期望 2`);
  const patches = parsePatches(patchBuf);
  const gaussians = parseGaussians(gaussBuf);
  if (patches.n !== meta.counts.patches || gaussians.n !== meta.counts.gaussians) {
    throw new Error("metadata.json 与二进制计数不一致");
  }

  const canvas = document.getElementById("c3d");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111315);
  const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 10000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;

  const gaussGeom = new THREE.BufferGeometry();
  gaussGeom.setAttribute("position", new THREE.BufferAttribute(gaussians.xyz, 3));
  const gaussianDisplayRgb = new Uint8Array(gaussians.rgb || gaussians.n * 3);
  gaussGeom.setAttribute("color", new THREE.Uint8BufferAttribute(gaussianDisplayRgb, 3, true));
  const gaussPoints = new THREE.Points(
    gaussGeom,
    new THREE.PointsMaterial({
      size: 1.15,
      color: 0xffffff,
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      sizeAttenuation: false,
      depthWrite: false,
    }),
  );
  gaussPoints.renderOrder = -1;
  scene.add(gaussPoints);

  const patchSurface = buildPatchSurface(patches);
  const patchGroup = new THREE.Group();
  patchGroup.add(patchSurface.mesh, patchSurface.outlines);
  patchGroup.visible = document.getElementById("show-patches").checked;
  scene.add(patchGroup);

  const members = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({ size: 2.6, color: 0xffffff, sizeAttenuation: true, depthTest: false }),
  );
  members.renderOrder = 4;
  scene.add(members);
  const normalHelper = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 8, 0xffdd57);
  normalHelper.visible = false;
  scene.add(normalHelper);
  const selectedOutline = new THREE.LineLoop(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xffe08a, depthTest: false }),
  );
  selectedOutline.visible = false;
  selectedOutline.renderOrder = 5;
  scene.add(selectedOutline);

  const sphere = patchSurface.mesh.geometry.boundingSphere || new THREE.Sphere(new THREE.Vector3(), 1);
  const radius = Math.max(sphere.radius, 1);
  const fitDistance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov * 0.5)) * 1.12;
  camera.near = Math.max(radius / 2000, 0.01);
  camera.far = Math.max(radius * 30, 1000);
  const cameraStateKey = `prf-camera:${meta.source_ply || window.location.pathname}`;
  function saveCameraState() {
    localStorage.setItem(cameraStateKey, JSON.stringify({
      position: camera.position.toArray(),
      up: camera.up.toArray(),
      target: controls.target.toArray(),
    }));
  }
  function restoreCameraState() {
    try {
      const state = JSON.parse(localStorage.getItem(cameraStateKey));
      const values = [...state.position, ...state.up, ...state.target];
      if (values.length !== 9 || !values.every(Number.isFinite)) return false;
      camera.position.fromArray(state.position);
      camera.up.fromArray(state.up);
      controls.target.fromArray(state.target);
      camera.updateProjectionMatrix();
      controls.update();
      return true;
    } catch {
      return false;
    }
  }
  function fitView(direction, topDown = false) {
    camera.up.set(0, topDown ? 1 : 0, topDown ? 0 : 1);
    controls.target.copy(sphere.center);
    camera.position.copy(sphere.center).addScaledVector(direction.clone().normalize(), fitDistance);
    camera.updateProjectionMatrix();
    controls.update();
    saveCameraState();
  }
  if (!restoreCameraState()) fitView(new THREE.Vector3(0, 0, 1), true);
  let cameraSaveFrame = 0;
  controls.addEventListener("change", () => {
    cancelAnimationFrame(cameraSaveFrame);
    cameraSaveFrame = requestAnimationFrame(saveCameraState);
  });

  let selected = -1;
  let metric = "R_phys";

  function paintGaussianField() {
    const colors = gaussGeom.attributes.color.array;
    if (metric === "rgb" || !gaussians.codes) {
      if (gaussians.rgb) colors.set(gaussians.rgb);
      gaussGeom.attributes.color.needsUpdate = true;
      return;
    }
    const codeKey = metric === "R_obs" && gaussians.codes.R_obs_detail
      ? "R_obs_detail"
      : (metric === "R_obs_absolute" ? "R_obs" : metric);
    const field = metric === "bottleneck" ? gaussians.bottleneck : gaussians.codes[codeKey];
    const strength = Number(document.getElementById("field-opacity").value) / 100;
    for (let i = 0; i < gaussians.n; i += 1) {
      const code = field[i];
      const offset = i * 3;
      if (code === 254 || (metric === "bottleneck" && code === 4)) {
        colors[offset] = gaussians.rgb[offset];
        colors[offset + 1] = gaussians.rgb[offset + 1];
        colors[offset + 2] = gaussians.rgb[offset + 2];
        continue;
      }
      const rgb = metric === "bottleneck"
        ? (BOTTLENECK_COLORS[code] || BOTTLENECK_COLORS[3])
        : (code === 255 ? [120, 120, 120] : (metric === "R_obs" ? OBS_DETAIL_COLORS : BIN_COLORS)[code]);
      colors[offset] = Math.round(rgb[0] * strength + gaussians.rgb[offset] * (1 - strength));
      colors[offset + 1] = Math.round(rgb[1] * strength + gaussians.rgb[offset + 1] * (1 - strength));
      colors[offset + 2] = Math.round(rgb[2] * strength + gaussians.rgb[offset + 2] * (1 - strength));
    }
    gaussGeom.attributes.color.needsUpdate = true;
  }

  function paint() {
    const colorArray = patchSurface.mesh.geometry.attributes.color.array;
    for (let triangle = 0; triangle < patchSurface.trianglePatch.length; triangle += 1) {
      const patchIndex = patchSurface.trianglePatch[triangle];
      const patchMetric = metric === "R_obs_absolute" ? "R_obs" : metric;
      const rgb = metric === "rgb"
        ? [190, 190, 190]
        : colorFor(metric, patches[patchMetric]?.[patchIndex], patches.bottleneck[patchIndex]);
      const mix = patchIndex === selected ? 0.36 : 0;
      const r = (rgb[0] * (1 - mix) + 255 * mix) / 255;
      const g = (rgb[1] * (1 - mix) + 255 * mix) / 255;
      const b = (rgb[2] * (1 - mix) + 255 * mix) / 255;
      for (let q = 0; q < 3; q += 1) colorArray.set([r, g, b], triangle * 9 + q * 3);
    }
    patchSurface.mesh.geometry.attributes.color.needsUpdate = true;
    paintGaussianField();
    document.getElementById("resolution-legend").hidden = metric === "R_obs" || metric === "bottleneck" || metric === "rgb";
    document.getElementById("obs-detail-legend").hidden = metric !== "R_obs";
    document.getElementById("bottleneck-legend").hidden = metric !== "bottleneck";
  }

  const viewElement = document.getElementById("view");
  const resize = () => {
    const rect = viewElement.getBoundingClientRect();
    renderer.setSize(Math.max(rect.width, 1), Math.max(rect.height, 1), false);
    camera.aspect = rect.width / Math.max(rect.height, 1);
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(viewElement);
  resize();

  const camSelect = document.getElementById("camera");
  meta.views.forEach((view, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = view.id;
    camSelect.appendChild(opt);
  });
  camSelect.disabled = meta.views.length === 0;

  const previewWrap = document.getElementById("preview-wrap");
  const preview = document.getElementById("preview");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  let previewZoom = 1;

  function resetPreviewZoom() {
    previewZoom = 1;
    overlay.style.transformOrigin = "50% 50%";
    overlay.style.transform = "scale(1)";
  }

  previewWrap.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = previewWrap.getBoundingClientRect();
    const originX = 100 * (event.clientX - rect.left) / Math.max(rect.width, 1);
    const originY = 100 * (event.clientY - rect.top) / Math.max(rect.height, 1);
    previewZoom = Math.min(8, Math.max(1, previewZoom * Math.exp(-event.deltaY * 0.0015)));
    overlay.style.transformOrigin = `${originX}% ${originY}%`;
    overlay.style.transform = `scale(${previewZoom})`;
  }, { passive: false });

  function currentView() {
    return meta.views[Number(camSelect.value) || 0];
  }

  function drawImagePanel() {
    const view = currentView();
    const info = document.getElementById("image-info");
    if (!view) {
      info.innerHTML = "<dt>训练相机</dt><dd>未导出</dd>";
      preview.removeAttribute("src");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      return;
    }
    previewWrap.style.aspectRatio = `${view.w} / ${view.h}`;
    if (view.preview && !preview.src.endsWith(view.preview)) preview.src = view.preview;
    const sourceW = Math.max(view.preview_w || 512, 1);
    const sourceH = Math.max(view.preview_h || Math.round(sourceW * view.h / view.w), 1);
    const w = overlay.width = sourceW;
    const h = overlay.height = sourceH;
    ctx.clearRect(0, 0, w, h);
    if (preview.complete && preview.naturalWidth > 0) {
      ctx.drawImage(preview, 0, 0, sourceW, sourceH);
    }
    if (selected < 0) {
      info.innerHTML = "";
      return;
    }
    const sx = w / view.w;
    const sy = h / view.h;
    const projected = hullWorld(patches, selected).map((point) => project(view, point));
    const allInFront = projected.every(Boolean);
    const allInFrame = allInFront && projected.every((point) => (
      point.u >= 0 && point.u < view.w && point.v >= 0 && point.v < view.h
    ));
    const center = project(view, vec3(patches.center, selected));
    ctx.strokeStyle = "#ffe08a";
    ctx.lineWidth = Math.max(2, w / 256);
    if (allInFront && projected.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(projected[0].u * sx, projected[0].v * sy);
      projected.slice(1).forEach((point) => ctx.lineTo(point.u * sx, point.v * sy));
      ctx.closePath();
      ctx.stroke();
    }
    if (center) {
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(center.u * sx, center.v * sy, Math.max(3, w / 170), 0, Math.PI * 2);
      ctx.fill();
    }
    const vIndex = Number(camSelect.value) || 0;
    const centerInside = patches.inside[selected * patches.nView + vIndex];
    const ri = patches.rView[selected * patches.nView + vIndex];
    const scale = center && Number.isFinite(center.z) ? (view.fx / center.z).toFixed(3) : "—";
    info.innerHTML = `
      <dt>中心在图内</dt><dd>${centerInside ? "是" : "否"}</dd>
      <dt>完整 hull 在图内</dt><dd>${allInFrame ? "是" : "否"}</dd>
      <dt>投影倍率</dt><dd>${scale} px/m</dd>
      <dt>单视图 r_i</dt><dd>${ri >= 0 ? `${ri.toFixed(3)} m` : "不可见"}</dd>
      <dt>可见性</dt><dd>frustum-only</dd>
    `;
  }
  preview.addEventListener("load", drawImagePanel);

  function showPatch(i) {
    if (!(i >= 0 && i < patches.n)) return;
    selected = i;
    const info = document.getElementById("info");
    info.hidden = false;
    document.getElementById("hint").hidden = true;
    const pair = [meta.pair_a[i], meta.pair_b[i]].filter(Boolean).join(" / ") || "无";
    const memberCount = patches.memberOffsets[i + 1] - patches.memberOffsets[i];
    info.innerHTML = `
      <dt>patch_id</dt><dd>${patches.patchId[i]}</dd>
      <dt>面积</dt><dd>${patches.area[i].toFixed(2)} m²</dd>
      <dt>半径</dt><dd>${patches.radius[i].toFixed(2)} m</dd>
      <dt>成员 Gaussian</dt><dd>${memberCount}</dd>
      <dt>R_obs</dt><dd>${formatResolution(patches.R_obs[i], "∞ / INVALID")}</dd>
      <dt>R_kernel</dt><dd>${formatResolution(patches.R_kernel[i])}</dd>
      <dt>R_spacing</dt><dd>${formatResolution(patches.R_spacing[i])}</dd>
      <dt>R_phys</dt><dd>${formatResolution(patches.R_phys[i])}</dd>
      <dt>瓶颈</dt><dd>${BOTTLENECK_NAMES[patches.bottleneck[i]]}</dd>
      <dt>可见视图数</dt><dd>${patches.nVisible[i]}</dd>
      <dt>最佳相机对</dt><dd>${pair}</dd>
      <dt>相机对夹角</dt><dd>${patches.pairAngle[i].toFixed(2)}°</dd>
    `;
    const normal = vec3(patches.normal, i).normalize();
    normalHelper.position.copy(vec3(patches.center, i));
    normalHelper.setDirection(normal);
    normalHelper.setLength(Math.max(patches.radius[i], 2) * 1.4);
    normalHelper.visible = true;
    const a = patches.memberOffsets[i];
    const b = patches.memberOffsets[i + 1];
    members.geometry.setAttribute("position", new THREE.BufferAttribute(patches.memberXyz.slice(a * 3, b * 3), 3));
    members.geometry.computeBoundingSphere();
    selectedOutline.geometry.dispose();
    selectedOutline.geometry = new THREE.BufferGeometry().setFromPoints(hullWorld(patches, i));
    selectedOutline.visible = true;
    const bestIndex = meta.views.findIndex((view) => view.id === meta.pair_a[i]);
    if (bestIndex >= 0) camSelect.value = String(bestIndex);
    paint();
    drawImagePanel();
  }

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerStart = null;
  canvas.addEventListener("pointerdown", (event) => {
    pointerStart = { x: event.clientX, y: event.clientY };
  });
  canvas.addEventListener("pointerup", (event) => {
    const start = pointerStart;
    pointerStart = null;
    if (!start || Math.hypot(event.clientX - start.x, event.clientY - start.y) > 5) return;
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(patchSurface.mesh, false)[0];
    if (hit && Number.isInteger(hit.faceIndex)) showPatch(patchSurface.trianglePatch[hit.faceIndex]);
  });

  document.getElementById("metric").addEventListener("change", (event) => {
    metric = event.target.value;
    paint();
  });
  document.getElementById("show-gaussians").addEventListener("change", (event) => {
    gaussPoints.visible = event.target.checked;
  });
  document.getElementById("show-patches").addEventListener("change", (event) => {
    patchGroup.visible = event.target.checked;
  });
  document.getElementById("field-opacity").addEventListener("input", (event) => {
    document.getElementById("field-opacity-value").value = `${event.target.value}%`;
    paintGaussianField();
  });
  document.getElementById("patch-opacity").addEventListener("input", (event) => {
    const opacity = Number(event.target.value) / 100;
    patchSurface.mesh.material.opacity = opacity;
    document.getElementById("patch-opacity-value").value = `${event.target.value}%`;
  });
  document.getElementById("view-top").addEventListener("click", () => {
    fitView(new THREE.Vector3(0, 0, 1), true);
  });
  document.getElementById("view-oblique").addEventListener("click", () => {
    fitView(new THREE.Vector3(1.2, -1.2, 0.85));
  });
  camSelect.addEventListener("change", () => {
    resetPreviewZoom();
    drawImagePanel();
  });
  document.getElementById("best-a").addEventListener("click", () => {
    if (selected < 0) return;
    const i = meta.views.findIndex((view) => view.id === meta.pair_a[selected]);
    if (i >= 0) { camSelect.value = String(i); resetPreviewZoom(); drawImagePanel(); }
  });
  document.getElementById("best-b").addEventListener("click", () => {
    if (selected < 0) return;
    const i = meta.views.findIndex((view) => view.id === meta.pair_b[selected]);
    if (i >= 0) { camSelect.value = String(i); resetPreviewZoom(); drawImagePanel(); }
  });

  paint();
  drawImagePanel();
  status.textContent = `${patches.n.toLocaleString()} patches · ${gaussians.n.toLocaleString()} 原始 PLY Gaussian · ${meta.views.length} 张训练图`;
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

main().catch((error) => {
  const status = document.getElementById("status");
  status.textContent = `加载失败：${error.message || error}`;
  status.classList.add("error");
  document.getElementById("hint").textContent = "请确认页面通过 HTTP 服务打开，且 metadata / binary / vendor 资源完整。";
  console.error(error);
});
