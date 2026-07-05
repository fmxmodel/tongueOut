// Track B1 (FLAME 2023 Open) viewer/driver — Phase 5.
//
// Loads the rigged head GLB (out/head_arkit.glb, ARKit-named morph targets — the
// FLAME-supported subset of Apple's 52), runs MediaPipe FaceLandmarker in VIDEO mode
// with outputFaceBlendshapes, and drives morphTargetInfluences by EXACT case-sensitive
// name via morphTargetDictionary. Supports live webcam OR a supplied video file, with
// influence smoothing.
//
// NOTE: this file wires the runtime. The actual name->influence mapping is the pure
// module src/driver.js (also used by the static verifier). No shape order is assumed;
// nothing is renamed. The loaded GLB's morphTargetDictionary is the SOLE source of
// truth for what can be driven; every MediaPipe categoryName that does not resolve is
// logged (never dropped silently) — for B1 that honestly covers the a-priori
// unsupported names (tongueOut is not emitted by MediaPipe; cheekPuff / cheekSquint*
// are emitted but not carried) plus any shape the pod demoted. The manifest contract
// (out/shapes/arkit_manifest.json) is read ONLY to LABEL those no-ops as expected vs
// real naming drift — it never remaps anything.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

import { CONFIG } from './config.js';
import { applyBlendshapes, resolveNames } from './driver.js';

// ----------------------------------------------------------------------------- DOM
const el = (id) => document.getElementById(id);
const appEl = el('app');
const statusEl = el('status');
const coverageEl = el('coverage');
const unresolvedEl = el('unresolved');
const smoothingInput = el('smoothing');
const smoothingLabel = el('smoothingLabel');
const startBtn = el('startBtn');
const stopBtn = el('stopBtn');
const sourceSel = el('source');
const fileInput = el('fileInput');
const mirrorChk = el('mirror');
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
appEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0420);

const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0.03, 0.55);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.02, 0);
controls.enableDamping = true;
controls.minDistance = 0.15;
controls.maxDistance = 3;

scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 1.4);
key.position.set(0.5, 0.8, 1.2);
scene.add(key);

// Placeholder shown until a real GLB loads (or if it never does).
const placeholder = new THREE.Mesh(
  new THREE.IcosahedronGeometry(0.09, 2),
  new THREE.MeshStandardMaterial({ color: 0x4fd1c5, flatShading: true, wireframe: true }),
);
scene.add(placeholder);

// ----------------------------------------------------------------------------- state
/** Every mesh in the GLB that carries morph targets. The face mesh plus any groom
 *  cards (eyebrows/beard) share the same ARKit shape-key names (the FLAME-supported
 *  subset), so we drive them in lockstep by pushing each frame to every one. */
let drivenMeshes = [];
let contract = null;       // raw contract JSON (B1 manifest, or legacy map), if reachable
let contractNames = null;  // normalized: the canonical ARKit names the contract lists
let contractUnsupported = new Set(); // names the contract declares unsupported (expected no-ops)
let contractRunState = null;         // 'DEFERRED-pod-run' | 'measured-on-pod' | null
let faceLandmarker = null; // created lazily on Start
let running = false;
let lastVideoTime = -1;
let smoothing = CONFIG.smoothingDefault;
const unresolvedSeen = new Map(); // categoryName -> hit count (naming-drift log)

// ----------------------------------------------------------------------------- GLB load
async function loadContract() {
  try {
    const res = await fetch(CONFIG.contractUrl, { cache: 'no-cache' });
    if (res.ok) {
      contract = await res.json();
      normalizeContract(contract);
    }
  } catch {
    /* optional; coverage panel just shows less */
  }
}

// Normalize either the B1 arkit-rigger manifest ({ shapes:{name:{supported,...}},
// run_state }) OR the legacy B2 51<->52 map ({ apple_canonical_52, unbaked_name })
// into { contractNames, contractUnsupported, contractRunState }. This is used ONLY to
// LABEL coverage — never to remap or drive.
function normalizeContract(raw) {
  if (!raw) return;
  if (raw.shapes && typeof raw.shapes === 'object') {
    const keys = Object.keys(raw.shapes);
    contractNames = keys;
    contractRunState = raw.run_state || null;
    const measured = contractRunState === 'measured-on-pod';
    // Deferred: unsupported == explicitly supported:false (the a-priori four).
    // Measured: unsupported == anything not measured supported:true (folds in demotions).
    contractUnsupported = new Set(
      keys.filter((k) => {
        const s = raw.shapes[k] || {};
        return measured ? s.supported !== true : s.supported === false;
      }),
    );
  } else if (Array.isArray(raw.apple_canonical_52)) {
    contractNames = raw.apple_canonical_52;
    contractUnsupported = new Set(raw.unbaked_name ? [raw.unbaked_name] : []);
    contractRunState = null;
  }
}

function collectMorphMeshes(root) {
  const meshes = [];
  root.traverse((o) => {
    if (o.isMesh && o.morphTargetDictionary && o.morphTargetInfluences) {
      o.morphTargetInfluences.fill(0);
      meshes.push(o);
    }
  });
  return meshes;
}

function frameOn(root) {
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 0.2;
  controls.target.copy(center);
  camera.position.set(center.x, center.y + size.y * 0.05, center.z + maxDim * 1.8);
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadGlb() {
  setStatus(`Loading GLB from ${CONFIG.glbUrl} …`);
  const loader = new GLTFLoader();
  try {
    const gltf = await loader.loadAsync(CONFIG.glbUrl);
    scene.remove(placeholder);
    scene.add(gltf.scene);
    drivenMeshes = collectMorphMeshes(gltf.scene);
    frameOn(gltf.scene);
    if (drivenMeshes.length === 0) {
      setStatus(
        'GLB loaded but NO mesh has morph targets — rig/export upstream is missing ARKit shapes. Cannot drive.',
        'error',
      );
    } else {
      const total = drivenMeshes.reduce((n, m) => n + Object.keys(m.morphTargetDictionary).length, 0);
      setStatus(
        `GLB loaded: ${drivenMeshes.length} morph mesh(es), ${total} morph target(s) total. Ready to drive.`,
      );
    }
  } catch (err) {
    // GRACEFUL DEGRADE: the real GLB is produced later on the GPU/Windows box.
    setStatus(
      `GLB not available at ${CONFIG.glbUrl} — this is expected before the GPU/real-artifact stage. ` +
        `Drop the exported head at out/head_arkit.glb and reload. (${err?.message || err})`,
      'error',
    );
  }
  refreshCoverage();
}

// ----------------------------------------------------------------------------- coverage / reporting panel
function currentDict() {
  // The face mesh is the source of truth; fall back to the first morph mesh.
  return drivenMeshes[0]?.morphTargetDictionary || null;
}

function refreshCoverage() {
  const names = contractNames; // canonical ARKit names from the B1 manifest (or legacy map)
  const dict = currentDict();
  const stateNote = contractRunState ? ` <span class="hint">· manifest ${contractRunState}</span>` : '';

  if (!dict) {
    coverageEl.innerHTML = names
      ? `<b>Coverage:</b> waiting for GLB. Contract lists <b>${names.length}</b> ARKit names` +
        (contractUnsupported.size
          ? ` · <span class="warn">unsupported (expected no-op): ${[...contractUnsupported].join(', ')}</span>`
          : '') +
        stateNote
      : `<b>Coverage:</b> waiting for GLB and contract.`;
  } else if (names) {
    const { resolved, unresolved } = resolveNames(dict, names);
    // Split "not in this GLB" into EXPECTED (contract says unsupported / pod-demoted)
    // vs UNEXPECTED (contract expected it — a build gap / naming drift for qa-verifier).
    const expected = unresolved.filter((n) => contractUnsupported.has(n));
    const drift = unresolved.filter((n) => !contractUnsupported.has(n));
    coverageEl.innerHTML =
      `<b>Coverage (name-resolution vs contract's ${names.length}):</b> ` +
      `<span class="ok">${resolved.length} drivable</span> / ${names.length}` +
      (expected.length ? ` · <span class="warn">unsupported (expected): ${expected.join(', ')}</span>` : '') +
      (drift.length ? ` · <span class="warn">UNEXPECTED not-in-GLB (naming drift → qa-verifier): ${drift.join(', ')}</span>` : '') +
      stateNote;
  } else {
    const n = Object.keys(dict).length;
    coverageEl.innerHTML = `<b>Coverage:</b> GLB exposes <b>${n}</b> named morph targets.`;
  }
  refreshUnresolvedLog();
}

function refreshUnresolvedLog() {
  if (unresolvedSeen.size === 0) {
    unresolvedEl.innerHTML = '<b>Unresolved MediaPipe names:</b> none';
    return;
  }
  const rows = [...unresolvedSeen.entries()]
    .map(([n, c]) => {
      // Tag each honest no-op: EXPECTED (contract declares it unsupported / pod-demoted)
      // vs UNEXPECTED (contract expected it — real naming drift for qa-verifier).
      const tag = contractUnsupported.has(n)
        ? '<span class="hint">(expected unsupported, B1)</span>'
        : '<span class="warn">(UNEXPECTED — naming drift)</span>';
      return `<li><code>${n}</code> ×${c} ${tag}</li>`;
    })
    .join('');
  unresolvedEl.innerHTML =
    `<b class="warn">Unresolved MediaPipe categoryNames (no matching morph target — for qa-verifier):</b><ul>${rows}</ul>`;
}

// ----------------------------------------------------------------------------- MediaPipe
async function resolveModelPath() {
  // Offline-first: prefer the locally fetched model; fall back to CDN if absent.
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
      outputFacialTransformationMatrixes: false,
    });
  try {
    faceLandmarker = await make('GPU');
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
  // Log unresolved names once-with-count (they reveal upstream naming drift).
  const unresolved = lastResult?.unresolved ?? [];
  if (unresolved.length) {
    for (const n of unresolved) unresolvedSeen.set(n, (unresolvedSeen.get(n) || 0) + 1);
    refreshUnresolvedLog();
  }
  return lastResult;
}

async function tick() {
  if (!running) return;
  if (faceLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const result = faceLandmarker.detectForVideo(video, performance.now());
      const categories = result.faceBlendshapes?.[0]?.categories ?? [];
      if (categories.length && drivenMeshes.length) driveFrame(categories);
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

// --- Credits / Licenses affordance (CC-BY-4.0 requires the FLAME credit be reasonably
//     visible in the product, not only in a repo file). Toggles the in-app panel that
//     surfaces the FLAME/MediaPipe/three.js attributions and links to THIRD-PARTY-NOTICES.md.
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
video.style.transform = mirrorChk.checked ? 'scaleX(-1)' : 'none';
mirrorChk.addEventListener('change', () => {
  video.style.transform = mirrorChk.checked ? 'scaleX(-1)' : 'none';
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
// A screenshot/verification harness (e.g. Puppeteer) can drive the rig WITHOUT a
// webcam or the MediaPipe model by injecting blendshape frames directly. See
// out/viewer_report.md ("Headless / verification path"). This uses the exact same
// name-resolution driver the live loop uses — so a screenshot proves the mapping.
window.__viewer = {
  ready: Promise.all([loadContract().then(loadGlb)]),
  get state() {
    return {
      glbLoaded: drivenMeshes.length > 0,
      morphMeshCount: drivenMeshes.length,
      morphTargets: currentDict() ? Object.keys(currentDict()) : [],
      running,
      smoothing,
      unresolved: Object.fromEntries(unresolvedSeen),
      // Contract-declared honest no-ops (a-priori + pod-demoted), so qa-verifier can
      // tell an EXPECTED unresolved name from real naming drift.
      expectedUnsupported: [...contractUnsupported],
      contractRunState,
    };
  },
  /** Inject one frame of [{categoryName, score}] straight into the rig. */
  injectBlendshapes(categories) {
    return driveFrame(categories || []);
  },
  /** Set a single morph target by name to a value (manual poke / QA). */
  setInfluence(name, value) {
    return driveFrame([{ categoryName: name, score: value }]);
  },
  loadVideoUrl: startFileUrl,
  renderer,
  scene,
  camera,
};

// ----------------------------------------------------------------------------- boot
(async function boot() {
  smoothing = CONFIG.smoothingDefault;
  await loadContract();
  await loadGlb();

  // URL params for automation: ?source=webcam|file  ?video=<url>  ?autostart=1  ?smoothing=0.5
  const q = new URLSearchParams(location.search);
  if (q.has('smoothing')) {
    smoothing = Number(q.get('smoothing'));
    smoothingInput.value = String(smoothing);
    smoothingLabel.textContent = smoothing.toFixed(2);
  }
  if (q.get('source') === 'webcam') sourceSel.value = 'webcam';
  if (q.get('video')) {
    sourceSel.value = 'file';
    await startFileUrl(q.get('video'));
  }
  if (q.get('autostart') === '1') start();
})();
