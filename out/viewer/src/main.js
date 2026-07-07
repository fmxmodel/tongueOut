// ARKit avatar viewer/driver — new stack (ICT-FaceKit-based head_arkit_v2.glb).
//
// Loads the rigged head GLB (out/head_arkit_v2.glb → served from public/), a single
// mesh `HeadARKit` split into THREE opaque primitives (HeadMat + EyeMat + RestMat),
// each carrying the SAME 52 ARKit-named morph targets. Runs MediaPipe FaceLandmarker
// in VIDEO mode with outputFaceBlendshapes and drives morphTargetInfluences by EXACT,
// case-sensitive name via each mesh's own morphTargetDictionary. Supports live webcam
// or a supplied video file, with influence smoothing and optional head-pose rotation.
//
// INVARIANTS (CLAUDE.md):
//   #1 driving is its own problem — we only read the GLB's morphTargetDictionary.
//   #2 the 52 names are a contract — resolved by exact name, never remapped/aliased.
//   #4 prove by measurement — the coverage panel and scripts/verify-names.mjs both
//      resolve names against the real GLB and log any MediaPipe categoryName that
//      finds no morph target (naming drift for qa-verifier), never silently dropped.
//
// The name->influence mapping is the PURE module src/driver.js (also used by the
// static verifier). No shape order is assumed. Because the head is multi-primitive,
// every frame is pushed to EVERY morph-bearing mesh, each resolving indices by name.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

import { CONFIG } from './config.js';
import { applyBlendshapes, resolveNames } from './driver.js';
import { ARKIT_52, MP_EMITTED, MP_NOT_EMITTED, MP_SKIP } from './arkit52.js';

// ----------------------------------------------------------------------------- DOM
const el = (id) => document.getElementById(id);
const appEl = el('app');
const statusEl = el('status');
const fpsEl = el('fps');
const coverageEl = el('coverage');
const unresolvedEl = el('unresolved');
const smoothingInput = el('smoothing');
const smoothingLabel = el('smoothingLabel');
const startBtn = el('startBtn');
const stopBtn = el('stopBtn');
const sourceSel = el('source');
const fileInput = el('fileInput');
const mirrorChk = el('mirror');
const headPoseChk = el('headPose');
const tongueSlider = el('tongueSlider');
const tongueLabel = el('tongueLabel');
const manualList = el('manualList');
const resetBtn = el('resetBtn');
const video = el('video');

function setStatus(msg, kind = 'info') {
  statusEl.textContent = msg;
  statusEl.dataset.kind = kind;
  // eslint-disable-next-line no-console
  console[kind === 'error' ? 'error' : 'log']('[viewer]', msg);
}

// ----------------------------------------------------------------------------- three.js scene
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
appEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x11131a);

const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0.03, 0.55);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.minDistance = 0.12;
controls.maxDistance = 3;

// Lighting: hemisphere ambient + key/fill/rim so the opaque head reads solid from all
// angles even without an environment map (kept dependency-light — no RoomEnvironment).
scene.add(new THREE.HemisphereLight(0xffffff, 0x30343f, 1.15));
const key = new THREE.DirectionalLight(0xffffff, 1.7);
key.position.set(0.6, 0.9, 1.2);
scene.add(key);
const fill = new THREE.DirectionalLight(0xdfe6ff, 0.5);
fill.position.set(-1.0, 0.2, 0.8);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffffff, 0.7);
rim.position.set(-0.3, 0.4, -1.2);
scene.add(rim);

// headRoot receives optional MediaPipe head-pose rotation; the GLB is recentered into
// it so rotation pivots about the head's own centre.
const headRoot = new THREE.Group();
scene.add(headRoot);

// Placeholder shown until a real GLB loads (or if it never does).
const placeholder = new THREE.Mesh(
  new THREE.IcosahedronGeometry(0.09, 2),
  new THREE.MeshStandardMaterial({ color: 0x4fd1c5, flatShading: true, wireframe: true }),
);
scene.add(placeholder);

// ----------------------------------------------------------------------------- state
/** Every mesh in the GLB that carries morph targets (here: the 3 head primitives).
 *  Each has its own morphTargetDictionary (name->index) and morphTargetInfluences.
 *  We drive them in lockstep by pushing each frame to every one, resolved by name. */
let drivenMeshes = [];
let headDict = {};          // union name->(first-seen index), for coverage + slider build
let faceLandmarker = null;  // created lazily on Start
let running = false;
let lastVideoTime = -1;
let smoothing = CONFIG.smoothingDefault;
let applyHeadPose = CONFIG.applyHeadPoseDefault;
const unresolvedSeen = new Map(); // categoryName -> hit count (naming-drift log)

// FPS (inference cadence) tracking.
let fpsEma = 0;
let lastInferMs = 0;

// ----------------------------------------------------------------------------- GLB load
function collectMorphMeshes(root) {
  const meshes = [];
  root.traverse((o) => {
    if (o.isMesh && o.morphTargetDictionary && o.morphTargetInfluences) {
      o.morphTargetInfluences.fill(0);
      hardenOpaque(o);
      meshes.push(o);
    }
  });
  return meshes;
}

// The GLB is alphaMode OPAQUE on every material; three.js already respects that. This
// is a belt-and-braces pass so nothing downstream (a stray transparent flag, alphaTest,
// depthWrite=false) can make the head render see-through. We do NOT flip doubleSided.
function hardenOpaque(mesh) {
  const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (const m of mats) {
    if (!m) continue;
    m.transparent = false;
    m.opacity = 1;
    m.alphaTest = 0;
    m.depthWrite = true;
    m.depthTest = true;
    if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
    m.needsUpdate = true;
  }
}

function buildHeadDict(meshes) {
  const dict = {};
  for (const mesh of meshes) {
    for (const name of Object.keys(mesh.morphTargetDictionary)) {
      if (!(name in dict)) dict[name] = mesh.morphTargetDictionary[name];
    }
  }
  return dict;
}

function frameOn(root) {
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  // Recenter the model at the origin so head-pose rotation pivots about the head.
  root.position.sub(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 0.2;
  controls.target.set(0, 0, 0);
  camera.position.set(0, size.y * 0.04, maxDim * 1.65);
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

let glbLoadPromise = null;
function ensureGlbLoaded() {
  if (!glbLoadPromise) glbLoadPromise = loadGlb();
  return glbLoadPromise;
}

async function loadGlb() {
  setStatus(`Loading GLB from ${CONFIG.glbUrl} …`);
  const loader = new GLTFLoader();
  try {
    const gltf = await loader.loadAsync(CONFIG.glbUrl);
    scene.remove(placeholder);
    headRoot.add(gltf.scene);
    drivenMeshes = collectMorphMeshes(gltf.scene);
    headDict = buildHeadDict(drivenMeshes);
    frameOn(gltf.scene);
    if (drivenMeshes.length === 0) {
      setStatus(
        'GLB loaded but NO mesh has morph targets — rig/export upstream is missing ARKit shapes. Cannot drive.',
        'error',
      );
    } else {
      const perMesh = drivenMeshes
        .map((m) => `${m.name || 'mesh'}:${Object.keys(m.morphTargetDictionary).length}`)
        .join(', ');
      setStatus(
        `GLB loaded: ${drivenMeshes.length} morph primitive(s) [${perMesh}], ` +
          `${Object.keys(headDict).length} distinct morph target(s). Ready to drive.`,
      );
      buildManualSliders();
    }
  } catch (err) {
    setStatus(
      `GLB not available at ${CONFIG.glbUrl} — run \`npm run copy-model\` (or npm run dev, which ` +
        `does it automatically) to stage out/head_arkit_v2.glb into public/. (${err?.message || err})`,
      'error',
    );
  }
  refreshCoverage();
}

// ----------------------------------------------------------------------------- coverage / reporting
function refreshCoverage() {
  const dict = headDict;
  const n = Object.keys(dict).length;

  if (!drivenMeshes.length) {
    coverageEl.innerHTML =
      `<b>Coverage:</b> waiting for GLB. Contract lists <b>${ARKIT_52.length}</b> ARKit names.`;
    refreshUnresolvedLog();
    return;
  }

  // Which of the 52 contract names are present in the GLB (drivable by name)?
  const { resolved, unresolved: missing } = resolveNames(dict, ARKIT_52);
  // Which webcam-emitted (MediaPipe) names resolve?  Should be all 51.
  const webcamDrivable = MP_EMITTED.filter((nm) => dict[nm] !== undefined);
  // Present in the GLB but MediaPipe never emits it → mappable, exercised via slider.
  const presentNotWebcam = [...MP_NOT_EMITTED].filter((nm) => dict[nm] !== undefined);
  // Any GLB morph name outside the ARKit-52 contract → unexpected extra (report it).
  const extra = Object.keys(dict).filter((nm) => !ARKIT_52.includes(nm));

  coverageEl.innerHTML =
    `<b>Coverage (name-resolution vs the ARKit-52 contract):</b> ` +
    `<span class="ok">${resolved.length}/52 present in GLB</span> · ` +
    `<span class="ok">${webcamDrivable.length}/51 webcam-drivable</span>` +
    (presentNotWebcam.length
      ? ` · <span class="hint">present but not webcam-driven (manual): ${presentNotWebcam.join(', ')}</span>`
      : '') +
    (missing.length
      ? ` · <span class="warn">MISSING from GLB (naming drift → qa-verifier): ${missing.join(', ')}</span>`
      : '') +
    (extra.length
      ? ` · <span class="warn">GLB has non-contract morphs: ${extra.join(', ')}</span>`
      : '') +
    ` <span class="hint">· GLB exposes ${n} named morph targets across ${drivenMeshes.length} primitives</span>`;
  refreshUnresolvedLog();
}

function refreshUnresolvedLog() {
  if (unresolvedSeen.size === 0) {
    unresolvedEl.innerHTML =
      '<b>Unresolved MediaPipe names:</b> none — every emitted categoryName resolved to a morph target.';
    return;
  }
  const rows = [...unresolvedSeen.entries()]
    .map(([nm, c]) => {
      const tag = MP_NOT_EMITTED.has(nm)
        ? '<span class="hint">(expected — not a webcam shape)</span>'
        : '<span class="warn">(UNEXPECTED — naming drift → qa-verifier)</span>';
      return `<li><code>${nm}</code> ×${c} ${tag}</li>`;
    })
    .join('');
  unresolvedEl.innerHTML =
    `<b class="warn">Unresolved MediaPipe categoryNames (no matching morph target):</b><ul>${rows}</ul>`;
}

// ----------------------------------------------------------------------------- manual sliders
// Build one slider per ARKit-52 name that the GLB actually carries. tongueOut is pulled
// out into its own always-visible slider (it is the one webcam-undriven morph); the rest
// live in a collapsible panel so any morph can be exercised without a camera.
function buildManualSliders() {
  // tongueOut dedicated slider.
  if (headDict.tongueOut !== undefined) {
    tongueSlider.disabled = false;
    tongueSlider.value = '0';
    tongueLabel.textContent = '0.00';
    tongueSlider.oninput = () => {
      const v = Number(tongueSlider.value);
      tongueLabel.textContent = v.toFixed(2);
      setInfluenceByName('tongueOut', v);
    };
  }
  // Full list (skip tongueOut — it has its own control above).
  manualList.innerHTML = '';
  for (const name of ARKIT_52) {
    if (name === 'tongueOut') continue;
    if (headDict[name] === undefined) continue;
    const row = document.createElement('label');
    row.className = 'mrow';
    const span = document.createElement('span');
    span.className = 'mname';
    span.textContent = name;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = '0';
    input.max = '1';
    input.step = '0.01';
    input.value = '0';
    input.dataset.name = name;
    input.addEventListener('input', () => setInfluenceByName(name, Number(input.value)));
    row.append(span, input);
    manualList.appendChild(row);
  }
}

// Set a morph influence on every primitive that carries it, by exact name.
function setInfluenceByName(name, value) {
  const v = Math.max(0, Math.min(1, value));
  for (const mesh of drivenMeshes) {
    const idx = mesh.morphTargetDictionary[name];
    if (idx !== undefined) mesh.morphTargetInfluences[idx] = v;
  }
}

function resetAllMorphs() {
  for (const mesh of drivenMeshes) mesh.morphTargetInfluences.fill(0);
  tongueSlider.value = '0';
  tongueLabel.textContent = '0.00';
  manualList.querySelectorAll('input[type="range"]').forEach((i) => (i.value = '0'));
}

// ----------------------------------------------------------------------------- MediaPipe
async function resolveModelPath() {
  try {
    const head = await fetch(CONFIG.modelAssetPath, { method: 'HEAD' });
    if (head.ok) return CONFIG.modelAssetPath;
  } catch {
    /* fall through to CDN */
  }
  setStatus('Local face_landmarker.task not found — using Google CDN model.');
  return CONFIG.modelAssetPathCdn;
}

async function ensureLandmarker() {
  if (faceLandmarker) return faceLandmarker;
  setStatus('Initialising MediaPipe FaceLandmarker …');
  const fileset = await FilesetResolver.forVisionTasks(CONFIG.wasmBase);
  const modelAssetPath = await resolveModelPath();
  const make = (delegate) =>
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  try {
    faceLandmarker = await make('GPU');
    setStatus('MediaPipe ready (GPU delegate).');
  } catch {
    setStatus('GPU delegate unavailable — falling back to CPU delegate.');
    faceLandmarker = await make('CPU');
  }
  return faceLandmarker;
}

// ----------------------------------------------------------------------------- video sources
async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  video.removeAttribute('src');
  await video.play();
}

async function startFile(file) {
  stopVideo();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = true;
  await video.play();
}

async function startFileUrl(url) {
  stopVideo();
  video.srcObject = null;
  video.src = url;
  video.loop = true;
  await video.play();
}

function stopVideo() {
  const s = video.srcObject;
  if (s && s.getTracks) s.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  if (video.src) {
    URL.revokeObjectURL(video.src);
    video.removeAttribute('src');
  }
  video.pause();
}

// ----------------------------------------------------------------------------- driver loop
function driveFrame(categories) {
  let lastResult = null;
  for (const mesh of drivenMeshes) {
    lastResult = applyBlendshapes(mesh, categories, {
      smoothing,
      skipNames: CONFIG.skipNames,
    });
  }
  const unresolved = lastResult?.unresolved ?? [];
  if (unresolved.length) {
    for (const nm of unresolved) unresolvedSeen.set(nm, (unresolvedSeen.get(nm) || 0) + 1);
    refreshUnresolvedLog();
  }
  return lastResult;
}

// Apply MediaPipe head pose (rotation only) to the head group. The 4x4 matrix is
// column-major; we decompose to a quaternion and, when the preview is mirrored,
// reflect it about X so the on-screen head turns the same way as the operator.
const _m = new THREE.Matrix4();
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();
function applyPose(matrices) {
  if (!applyHeadPose) return;
  const data = matrices?.[0]?.data;
  if (!data || data.length !== 16) return;
  _m.fromArray(data);
  _m.decompose(_p, _q, _s);
  if (mirrorChk.checked) {
    _q.set(_q.x, -_q.y, -_q.z, _q.w); // reflect rotation about X (horizontal mirror)
  }
  headRoot.quaternion.slerp(_q, 0.5); // ease toward the tracked pose
}

async function tick() {
  if (!running) return;
  if (faceLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const now = performance.now();
      const result = faceLandmarker.detectForVideo(video, now);
      const categories = result.faceBlendshapes?.[0]?.categories ?? [];
      if (categories.length && drivenMeshes.length) driveFrame(categories);
      applyPose(result.facialTransformationMatrixes);
      // FPS = smoothed inference cadence.
      if (lastInferMs) {
        const inst = 1000 / Math.max(1, now - lastInferMs);
        fpsEma = fpsEma ? fpsEma * 0.9 + inst * 0.1 : inst;
        fpsEl.textContent = `${fpsEma.toFixed(0)} fps`;
      }
      lastInferMs = now;
    } catch (err) {
      setStatus(`detectForVideo error: ${err?.message || err}`, 'error');
    }
  }
  requestAnimationFrame(tick);
}

async function start() {
  try {
    startBtn.disabled = true;
    if (sourceSel.value === 'webcam') {
      await startWebcam();
    } else {
      const file = fileInput.files?.[0];
      if (!file) {
        setStatus('Choose a video file first (or switch source to Webcam).', 'error');
        startBtn.disabled = false;
        return;
      }
      await startFile(file);
    }
    await ensureLandmarker();
    running = true;
    lastVideoTime = -1;
    lastInferMs = 0;
    fpsEma = 0;
    stopBtn.disabled = false;
    setStatus('Driving — tracking face and writing morphTargetInfluences by name.');
    tick();
  } catch (err) {
    setStatus(`Failed to start: ${err?.message || err}`, 'error');
    startBtn.disabled = false;
  }
}

function stop() {
  running = false;
  stopVideo();
  startBtn.disabled = false;
  stopBtn.disabled = true;
  fpsEl.textContent = '— fps';
  headRoot.quaternion.identity();
  setStatus('Stopped.');
}

// ----------------------------------------------------------------------------- UI wiring
smoothingInput.value = String(smoothing);
smoothingLabel.textContent = smoothing.toFixed(2);
smoothingInput.addEventListener('input', () => {
  smoothing = Number(smoothingInput.value);
  smoothingLabel.textContent = smoothing.toFixed(2);
});
sourceSel.addEventListener('change', () => {
  fileInput.style.display = sourceSel.value === 'file' ? '' : 'none';
});
fileInput.style.display = sourceSel.value === 'file' ? '' : 'none';
startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);
resetBtn.addEventListener('click', resetAllMorphs);

headPoseChk.checked = applyHeadPose;
headPoseChk.addEventListener('change', () => {
  applyHeadPose = headPoseChk.checked;
  if (!applyHeadPose) headRoot.quaternion.identity();
});

video.style.transform = mirrorChk.checked ? 'scaleX(-1)' : 'none';
mirrorChk.addEventListener('change', () => {
  video.style.transform = mirrorChk.checked ? 'scaleX(-1)' : 'none';
});

// Credits / Licenses panel.
const creditsBtn = el('creditsBtn');
const creditsPanel = el('credits');
const creditsClose = el('creditsClose');
const creditsBackdrop = el('creditsBackdrop');
function setCredits(open) {
  creditsPanel.classList.toggle('open', open);
  creditsBackdrop.classList.toggle('open', open);
}
creditsBtn?.addEventListener('click', () => setCredits(true));
creditsClose?.addEventListener('click', () => setCredits(false));
creditsBackdrop?.addEventListener('click', () => setCredits(false));
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') setCredits(false);
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update();
  if (scene.children.includes(placeholder)) placeholder.rotation.y += 0.008;
  renderer.render(scene, camera);
});

// ----------------------------------------------------------------------------- headless / verification hook
// A screenshot/verification harness (e.g. Puppeteer) can drive the rig WITHOUT a webcam
// or the MediaPipe model by injecting blendshape frames directly — it uses the exact
// same name-resolution driver the live loop uses, so a screenshot proves the mapping.
window.__viewer = {
  ready: ensureGlbLoaded(),
  get state() {
    return {
      glbLoaded: drivenMeshes.length > 0,
      morphMeshCount: drivenMeshes.length,
      morphTargets: Object.keys(headDict),
      morphTargetCount: Object.keys(headDict).length,
      contract52: ARKIT_52,
      webcamNotEmitted: [...MP_NOT_EMITTED],
      running,
      smoothing,
      unresolved: Object.fromEntries(unresolvedSeen),
    };
  },
  /** Inject one frame of [{categoryName, score}] straight into the rig (all primitives). */
  injectBlendshapes(categories) {
    return driveFrame(categories || []);
  },
  /** Set a single morph target by name to a value (manual poke / QA). */
  setInfluence(name, value) {
    setInfluenceByName(name, value);
    return { name, value };
  },
  loadVideoUrl: startFileUrl,
  renderer,
  scene,
  camera,
};

// ----------------------------------------------------------------------------- boot
(async function boot() {
  smoothing = CONFIG.smoothingDefault;
  await ensureGlbLoaded();

  // URL params for automation: ?source=webcam|file  ?video=<url>  ?autostart=1  ?smoothing=0.5
  const q = new URLSearchParams(location.search);
  if (q.has('smoothing')) {
    smoothing = Number(q.get('smoothing'));
    smoothingInput.value = String(smoothing);
    smoothingLabel.textContent = smoothing.toFixed(2);
  }
  if (q.get('source') === 'webcam') sourceSel.value = 'webcam';
  if (q.get('headpose') === '1') {
    applyHeadPose = true;
    headPoseChk.checked = true;
  }
  if (q.get('video')) {
    sourceSel.value = 'file';
    await startFileUrl(q.get('video'));
  }
  if (q.get('autostart') === '1') start();
})();
