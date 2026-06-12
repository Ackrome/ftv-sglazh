const $ = (id) => document.getElementById(id);

const state = {
  gl: null,
  model: null,
  programs: {},
  buffers: {},
  textures: {},
  mode: 1,
  mix: 1,
  verticalScale: 2.4,
  showBase: true,
  yaw: -0.62,
  pitch: 0.72,
  distance: 2.25,
  target: [0, 0, 0],
  drag: null,
  autoRotate: false,
  lastFrameTime: 0,
};

const terrainVertexShader = `#version 300 es
precision highp float;
in vec2 aUv;
uniform sampler2D uBefore;
uniform sampler2D uAfter;
uniform sampler2D uMask;
uniform mat4 uViewProjection;
uniform float uMix;
uniform float uVerticalScale;
uniform float uSpanM;
uniform float uWidthN;
uniform float uDepthN;
uniform float uCenterElevation;
out vec3 vNormal;
out float vElevation;
out float vResidual;
out float vMask;

float maskAt(ivec2 p) {
  ivec2 size = textureSize(uMask, 0);
  return texelFetch(uMask, clamp(p, ivec2(0), size - 1), 0).r;
}

float beforeAt(ivec2 p) {
  ivec2 size = textureSize(uBefore, 0);
  return texelFetch(uBefore, clamp(p, ivec2(0), size - 1), 0).r;
}

float afterAt(ivec2 p) {
  ivec2 size = textureSize(uAfter, 0);
  return texelFetch(uAfter, clamp(p, ivec2(0), size - 1), 0).r;
}

float heightAt(ivec2 p) {
  return mix(beforeAt(p), afterAt(p), uMix);
}

float neighborHeight(ivec2 p, float centerHeight) {
  return maskAt(p) > 0.5 ? heightAt(p) : centerHeight;
}

void main() {
  ivec2 size = textureSize(uBefore, 0);
  ivec2 p = ivec2(round(aUv * vec2(size - 1)));
  float heightM = heightAt(p);
  float leftM = neighborHeight(p + ivec2(-1, 0), heightM);
  float rightM = neighborHeight(p + ivec2(1, 0), heightM);
  float downM = neighborHeight(p + ivec2(0, -1), heightM);
  float upM = neighborHeight(p + ivec2(0, 1), heightM);
  float stepX = uWidthN / max(float(size.x - 1), 1.0);
  float stepZ = uDepthN / max(float(size.y - 1), 1.0);
  vec3 dx = vec3(2.0 * stepX, (rightM - leftM) / uSpanM * uVerticalScale, 0.0);
  vec3 dz = vec3(0.0, (upM - downM) / uSpanM * uVerticalScale, 2.0 * stepZ);
  vNormal = normalize(cross(dz, dx));
  vElevation = heightM;
  vResidual = afterAt(p) - beforeAt(p);
  vMask = maskAt(p);
  vec3 position = vec3(
    (aUv.x - 0.5) * uWidthN,
    (heightM - uCenterElevation) / uSpanM * uVerticalScale,
    (aUv.y - 0.5) * uDepthN
  );
  gl_Position = uViewProjection * vec4(position, 1.0);
}`;

const terrainFragmentShader = `#version 300 es
precision highp float;
in vec3 vNormal;
in float vElevation;
in float vResidual;
in float vMask;
uniform float uElevationMin;
uniform float uElevationMax;
uniform int uMode;
out vec4 outColor;

vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 a = vec3(0.05, 0.28, 0.22);
  vec3 b = vec3(0.28, 0.55, 0.36);
  vec3 c = vec3(0.73, 0.70, 0.45);
  vec3 d = vec3(0.72, 0.52, 0.40);
  vec3 e = vec3(0.90, 0.91, 0.88);
  if (t < 0.25) return mix(a, b, t * 4.0);
  if (t < 0.55) return mix(b, c, (t - 0.25) / 0.30);
  if (t < 0.82) return mix(c, d, (t - 0.55) / 0.27);
  return mix(d, e, (t - 0.82) / 0.18);
}

vec3 residualRamp(float value) {
  float strength = clamp(abs(value) / 4.0, 0.0, 1.0);
  vec3 neutral = vec3(0.76, 0.80, 0.82);
  vec3 negative = vec3(0.20, 0.54, 0.85);
  vec3 positive = vec3(0.90, 0.35, 0.29);
  return mix(neutral, value < 0.0 ? negative : positive, strength);
}

void main() {
  if (vMask < 0.5) discard;
  float t = (vElevation - uElevationMin) / max(uElevationMax - uElevationMin, 0.001);
  vec3 color = uMode == 3 ? residualRamp(vResidual) : ramp(t);
  vec3 light = normalize(vec3(-0.42, 0.84, 0.34));
  float diffuse = max(dot(normalize(vNormal), light), 0.0);
  float lighting = 0.42 + 0.58 * diffuse;
  outColor = vec4(color * lighting, 1.0);
}`;

const baseVertexShader = `#version 300 es
precision highp float;
in vec3 aPosition;
uniform mat4 uViewProjection;
void main() {
  gl_Position = uViewProjection * vec4(aPosition, 1.0);
}`;

const baseFragmentShader = `#version 300 es
precision highp float;
out vec4 outColor;
void main() {
  outColor = vec4(0.075, 0.12, 0.16, 1.0);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "Shader compilation failed");
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "Program linking failed");
  }
  return program;
}

function multiply4(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row] * b[column * 4] +
        a[4 + row] * b[column * 4 + 1] +
        a[8 + row] * b[column * 4 + 2] +
        a[12 + row] * b[column * 4 + 3];
    }
  }
  return out;
}

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function normalize3(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function subtract3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function lookAt(eye, center, up) {
  const z = normalize3(subtract3(eye, center));
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
    -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
    -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
    1,
  ]);
}

function createTexture(gl, width, height, internalFormat, format, type, data) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
  const error = gl.getError();
  if (error !== gl.NO_ERROR) throw new Error(`Texture upload failed with WebGL error ${error}`);
  return texture;
}

function createTerrainGeometry(gl, rows, columns) {
  const vertices = new Float32Array(rows * columns * 2);
  let vertexOffset = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      vertices[vertexOffset++] = column / (columns - 1);
      vertices[vertexOffset++] = row / (rows - 1);
    }
  }
  const indices = new Uint32Array((rows - 1) * (columns - 1) * 6);
  let indexOffset = 0;
  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const a = row * columns + column;
      const b = a + 1;
      const c = a + columns;
      const d = c + 1;
      indices[indexOffset++] = a;
      indices[indexOffset++] = c;
      indices[indexOffset++] = b;
      indices[indexOffset++] = b;
      indices[indexOffset++] = c;
      indices[indexOffset++] = d;
    }
  }
  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  return { vertexBuffer, indexBuffer, indexCount: indices.length };
}

function createBaseGeometry(gl, widthN, depthN, floorY) {
  const x = widthN / 2;
  const z = depthN / 2;
  const vertices = new Float32Array([
    -x, floorY, -z,  x, floorY, -z,  -x, floorY, z,
    -x, floorY, z,   x, floorY, -z,   x, floorY, z,
  ]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  return buffer;
}

function bindTexture(gl, texture, unit, uniform, program) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(gl.getUniformLocation(program, uniform), unit);
}

function resetView() {
  state.yaw = -0.62;
  state.pitch = 0.72;
  state.distance = 2.25;
  state.target = [0, 0, 0];
}

function cameraViewProjection(canvas) {
  const cosPitch = Math.cos(state.pitch);
  const eye = [
    state.target[0] + state.distance * cosPitch * Math.sin(state.yaw),
    state.target[1] + state.distance * Math.sin(state.pitch),
    state.target[2] + state.distance * cosPitch * Math.cos(state.yaw),
  ];
  const projection = perspective(48 * Math.PI / 180, canvas.width / canvas.height, 0.02, 20);
  return multiply4(projection, lookAt(eye, state.target, [0, 1, 0]));
}

function resizeCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(2, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(2, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function setUniform1f(gl, program, name, value) {
  gl.uniform1f(gl.getUniformLocation(program, name), value);
}

function renderTerrain(viewProjection) {
  const gl = state.gl;
  const model = state.model;
  const program = state.programs.terrain;
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.buffers.terrain.vertexBuffer);
  const uvLocation = gl.getAttribLocation(program, "aUv");
  gl.enableVertexAttribArray(uvLocation);
  gl.vertexAttribPointer(uvLocation, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state.buffers.terrain.indexBuffer);
  bindTexture(gl, state.textures.before, 0, "uBefore", program);
  bindTexture(gl, state.textures.after, 1, "uAfter", program);
  bindTexture(gl, state.textures.mask, 2, "uMask", program);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, "uViewProjection"), false, viewProjection);
  setUniform1f(gl, program, "uMix", state.mix);
  setUniform1f(gl, program, "uVerticalScale", state.verticalScale);
  setUniform1f(gl, program, "uSpanM", model.terrain_span_m);
  setUniform1f(gl, program, "uWidthN", model.span_x_m / model.terrain_span_m);
  setUniform1f(gl, program, "uDepthN", model.span_z_m / model.terrain_span_m);
  setUniform1f(gl, program, "uCenterElevation", (model.elevation_min_m + model.elevation_max_m) / 2);
  setUniform1f(gl, program, "uElevationMin", model.elevation_min_m);
  setUniform1f(gl, program, "uElevationMax", model.elevation_max_m);
  gl.uniform1i(gl.getUniformLocation(program, "uMode"), state.mode);
  gl.drawElements(gl.TRIANGLES, state.buffers.terrain.indexCount, gl.UNSIGNED_INT, 0);
}

function renderBase(viewProjection) {
  if (!state.showBase) return;
  const gl = state.gl;
  const program = state.programs.base;
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, state.buffers.base);
  const positionLocation = gl.getAttribLocation(program, "aPosition");
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
  gl.uniformMatrix4fv(gl.getUniformLocation(program, "uViewProjection"), false, viewProjection);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function updateViewBadge() {
  const yaw = ((state.yaw * 180 / Math.PI) % 360 + 360) % 360;
  const pitch = state.pitch * 180 / Math.PI;
  $("viewBadge").textContent = `Yaw ${yaw.toFixed(0)} deg | Elevation ${pitch.toFixed(0)} deg`;
}

function renderFrame(time) {
  if (!state.gl || !state.model) return;
  if (state.autoRotate && state.lastFrameTime) {
    state.yaw += (time - state.lastFrameTime) * 0.00022;
  }
  state.lastFrameTime = time;
  const canvas = $("terrainCanvas");
  resizeCanvas(canvas);
  const gl = state.gl;
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  gl.clearColor(0.071, 0.082, 0.13, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const viewProjection = cameraViewProjection(canvas);
  renderBase(viewProjection);
  renderTerrain(viewProjection);
  updateViewBadge();
  window.requestAnimationFrame(renderFrame);
}

function setMode(mode) {
  const names = { before: "Before FTV", after: "After FTV", blend: "Blend layers", difference: "Correction map" };
  state.mode = { before: 0, after: 1, blend: 2, difference: 3 }[mode];
  if (mode === "before") state.mix = 0;
  if (mode === "after" || mode === "difference") state.mix = 1;
  if (mode === "blend") state.mix = Number($("mixSlider").value) / 100;
  $("mixSlider").disabled = mode !== "blend";
  $("mixValue").textContent = `${Math.round(state.mix * 100)}%`;
  $("modeBadge").textContent = names[mode];
}

function bindInteractions() {
  document.querySelectorAll('input[name="layerMode"]').forEach((radio) => {
    radio.addEventListener("change", (event) => setMode(event.target.value));
  });
  $("mixSlider").addEventListener("input", (event) => {
    state.mix = Number(event.target.value) / 100;
    $("mixValue").textContent = `${event.target.value}%`;
  });
  $("verticalSlider").addEventListener("input", (event) => {
    state.verticalScale = Number(event.target.value) / 100;
    $("verticalValue").textContent = `${state.verticalScale.toFixed(2)}x`;
  });
  $("baseToggle").addEventListener("change", (event) => { state.showBase = event.target.checked; });
  $("resetButton").addEventListener("click", resetView);
  $("autoRotateButton").addEventListener("click", (event) => {
    state.autoRotate = !state.autoRotate;
    event.target.classList.toggle("is-active", state.autoRotate);
  });

  const viewport = $("terrainViewport");
  viewport.addEventListener("pointerdown", (event) => {
    state.drag = {
      x: event.clientX,
      y: event.clientY,
      yaw: state.yaw,
      pitch: state.pitch,
      target: [...state.target],
      pan: event.shiftKey || event.button === 2,
    };
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    const dx = event.clientX - state.drag.x;
    const dy = event.clientY - state.drag.y;
    if (state.drag.pan) {
      const scale = state.distance * 0.0015;
      state.target[0] = state.drag.target[0] - dx * scale;
      state.target[2] = state.drag.target[2] + dy * scale;
    } else {
      state.yaw = state.drag.yaw - dx * 0.008;
      state.pitch = Math.max(-1.48, Math.min(1.48, state.drag.pitch + dy * 0.008));
    }
  });
  viewport.addEventListener("pointerup", () => { state.drag = null; });
  viewport.addEventListener("pointercancel", () => { state.drag = null; });
  viewport.addEventListener("contextmenu", (event) => event.preventDefault());
  viewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.distance = Math.max(0.38, Math.min(8, state.distance * Math.exp(event.deltaY * 0.001)));
  }, { passive: false });
  viewport.addEventListener("dblclick", resetView);
  viewport.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") state.yaw += 0.08;
    if (event.key === "ArrowRight") state.yaw -= 0.08;
    if (event.key === "ArrowUp") state.pitch = Math.min(1.48, state.pitch + 0.08);
    if (event.key === "ArrowDown") state.pitch = Math.max(-1.48, state.pitch - 0.08);
  });
}

function fillMetadata(model, triangleCount) {
  $("sourceLabel").textContent = model.source;
  $("sceneLabel").textContent = model.scene;
  $("gridValue").textContent = `${model.grid_shape[0]} x ${model.grid_shape[1]}`;
  $("sourceGridValue").textContent = `${model.source_shape[0]} x ${model.source_shape[1]}`;
  $("spanValue").textContent = `${(model.span_x_m / 1000).toFixed(1)} x ${(model.span_z_m / 1000).toFixed(1)} km`;
  $("coverageValue").textContent = `${(model.source_valid_fraction * 100).toFixed(1)}%`;
  $("triangleLabel").textContent = `${triangleCount.toLocaleString()} triangles`;
  $("verticalSlider").value = String(Math.round(model.vertical_exaggeration_default * 100));
  $("verticalValue").textContent = `${model.vertical_exaggeration_default.toFixed(2)}x`;
}

async function loadBinary(path, Type) {
  const revision = encodeURIComponent(state.model.asset_revision || state.model.schema_version);
  const response = await fetch(`${path}?revision=${revision}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load ${path}: ${response.status}`);
  return new Type(await response.arrayBuffer());
}

async function init() {
  const canvas = $("terrainCanvas");
  const gl = canvas.getContext("webgl2", { antialias: true });
  if (!gl) throw new Error("WebGL2 is not available in this browser");
  state.gl = gl;
  const response = await fetch("terrain-model.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load terrain-model.json: ${response.status}`);
  state.model = await response.json();
  if (state.model.viewer_type !== "webgl-terrain-model") {
    throw new Error("This directory does not contain a browser-native terrain model");
  }
  const [before, after, mask] = await Promise.all([
    loadBinary(state.model.files.before, Float32Array),
    loadBinary(state.model.files.after, Float32Array),
    loadBinary(state.model.files.mask, Uint8Array),
  ]);
  const [rows, columns] = state.model.grid_shape;
  if (before.length !== rows * columns || after.length !== rows * columns || mask.length !== rows * columns) {
    throw new Error("Terrain binary sizes do not match the exported grid");
  }
  state.programs.terrain = createProgram(gl, terrainVertexShader, terrainFragmentShader);
  state.programs.base = createProgram(gl, baseVertexShader, baseFragmentShader);
  state.textures.before = createTexture(gl, columns, rows, gl.R32F, gl.RED, gl.FLOAT, before);
  state.textures.after = createTexture(gl, columns, rows, gl.R32F, gl.RED, gl.FLOAT, after);
  state.textures.mask = createTexture(gl, columns, rows, gl.R8, gl.RED, gl.UNSIGNED_BYTE, mask);
  state.buffers.terrain = createTerrainGeometry(gl, rows, columns);
  const widthN = state.model.span_x_m / state.model.terrain_span_m;
  const depthN = state.model.span_z_m / state.model.terrain_span_m;
  const center = (state.model.elevation_min_m + state.model.elevation_max_m) / 2;
  const maximumVerticalScale = Number($("verticalSlider").max) / 100;
  const floorY = (state.model.elevation_min_m - center) / state.model.terrain_span_m * maximumVerticalScale - 0.012;
  state.buffers.base = createBaseGeometry(gl, widthN, depthN, floorY);
  state.verticalScale = state.model.vertical_exaggeration_default;
  fillMetadata(state.model, state.buffers.terrain.indexCount / 3);
  bindInteractions();
  setMode("after");
  resetView();
  $("loadingState").hidden = true;
  $("statusLabel").textContent = "Live full-area mesh ready";
  window.__terrainViewerReady = true;
  window.__terrainViewerState = state;
  window.requestAnimationFrame(renderFrame);
}

init().catch((error) => {
  console.error(error);
  $("loadingState").textContent = error.message;
  $("statusLabel").textContent = "Viewer failed to load";
  $("rendererLabel").textContent = "WebGL2 unavailable";
});
