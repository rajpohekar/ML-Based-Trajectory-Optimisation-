const state = {
  data: null,
  frame: 0,
  liveStep: 0,
  playing: true,
  ended: false,
  speed: 1,
  lastTick: 0,
};

const sceneCanvas = document.getElementById("scene3d");
const sceneCtx = sceneCanvas.getContext("2d");
const miniCanvas = document.getElementById("trajectoryMini");
const miniCtx = miniCanvas.getContext("2d");
const coverageCanvas = document.getElementById("coverageMap");
const coverageCtx = coverageCanvas.getContext("2d");
const altitudeCanvas = document.getElementById("altitudeChart");
const altitudeCtx = altitudeCanvas.getContext("2d");
const donutCanvas = document.getElementById("losDonut");
const donutCtx = donutCanvas.getContext("2d");

function fitCanvas(canvas, ctx) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function project(point, rect, world) {
  const [x, y, z] = point;
  const cx = rect.width * 0.53;
  const cy = rect.height * 0.58;
  const sx = rect.width / world;
  const sy = rect.height / world;
  const isoX = (x - y) * 0.82 * sx;
  const isoY = (x + y) * 0.30 * sy - z * 1.42;
  return [cx + isoX, cy + isoY];
}

function signalColor(value, alpha = 1) {
  const t = Math.max(0, Math.min(1, (value + 120) / 85));
  const stops = [
    [35, 16, 75],
    [25, 125, 160],
    [50, 205, 112],
    [255, 226, 39],
  ];
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const c = stops[i].map((v, idx) => Math.round(v + (stops[i + 1][idx] - v) * f));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

function drawGrid(ctx, rect, world) {
  ctx.strokeStyle = "rgba(130, 170, 200, 0.18)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const a = (world / 8) * i;
    const p1 = project([a, 0, 0], rect, world);
    const p2 = project([a, world, 0], rect, world);
    const p3 = project([0, a, 0], rect, world);
    const p4 = project([world, a, 0], rect, world);
    ctx.beginPath();
    ctx.moveTo(...p1);
    ctx.lineTo(...p2);
    ctx.moveTo(...p3);
    ctx.lineTo(...p4);
    ctx.stroke();
  }
  const corners = [[0, 0, 0], [world, 0, 0], [world, world, 0], [0, world, 0]];
  ctx.strokeStyle = "rgba(200, 230, 250, 0.32)";
  ctx.beginPath();
  corners.forEach((p, i) => {
    const q = project(p, rect, world);
    if (i === 0) ctx.moveTo(...q);
    else ctx.lineTo(...q);
  });
  ctx.closePath();
  ctx.stroke();
}

function drawGroundHeat(ctx, rect, data) {
  const world = data.meta.worldM;
  const map = data.coverageMap.signalDbm;
  const size = data.coverageMap.size;
  for (let gy = 0; gy < size - 1; gy += 2) {
    for (let gx = 0; gx < size - 1; gx += 2) {
      const x = (gx / (size - 1)) * world;
      const y = (gy / (size - 1)) * world;
      const p = project([x, y, 1], rect, world);
      const radius = Math.max(8, rect.width / 70);
      const glow = ctx.createRadialGradient(p[0], p[1], 1, p[0], p[1], radius);
      glow.addColorStop(0, signalColor(map[gy][gx], 0.62));
      glow.addColorStop(1, signalColor(map[gy][gx], 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawMarker(ctx, x, y, type, size = 9) {
  ctx.save();
  if (type === "base") {
    ctx.fillStyle = "#21e7ff";
    ctx.strokeStyle = "#02131a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? size * 1.3 : size * 0.55;
      const a = -Math.PI / 2 + i * Math.PI / 5;
      ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  } else if (type === "uav") {
    ctx.fillStyle = "#ffd51f";
    ctx.strokeStyle = "#0b0c0e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y - size * 1.4);
    ctx.lineTo(x + size * 1.25, y + size);
    ctx.lineTo(x - size * 1.25, y + size);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.fillStyle = "rgba(245,248,255,0.95)";
    ctx.strokeStyle = "rgba(40,50,60,0.9)";
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawScene() {
  fitCanvas(sceneCanvas, sceneCtx);
  const rect = sceneCanvas.getBoundingClientRect();
  const { data } = state;
  const world = data.meta.worldM;
  const frame = data.frames[state.frame];
  const trajectory = data.trajectory.slice(0, state.frame + 1);
  const users = frame.users || data.users;

  const grad = sceneCtx.createLinearGradient(0, 0, 0, rect.height);
  grad.addColorStop(0, "#07111d");
  grad.addColorStop(1, "#02070c");
  sceneCtx.fillStyle = grad;
  sceneCtx.fillRect(0, 0, rect.width, rect.height);

  drawGroundHeat(sceneCtx, rect, data);
  drawGrid(sceneCtx, rect, world);

  const base = project(data.baseStation, rect, world);
  const uav = project(frame.uav, rect, world);

  users.forEach((user, idx) => {
    const p = project([user[0], user[1], 0], rect, world);
    if (idx % 3 === 0 || frame.losMask[idx]) {
      sceneCtx.strokeStyle = frame.losMask[idx] ? "rgba(115,223,80,0.55)" : "rgba(247,167,40,0.5)";
      sceneCtx.setLineDash([7, 6]);
      sceneCtx.beginPath();
      sceneCtx.moveTo(uav[0], uav[1]);
      sceneCtx.lineTo(p[0], p[1]);
      sceneCtx.stroke();
      sceneCtx.setLineDash([]);
    }
  });

  sceneCtx.strokeStyle = "rgba(33,231,255,0.8)";
  sceneCtx.setLineDash([8, 7]);
  sceneCtx.beginPath();
  sceneCtx.moveTo(base[0], base[1]);
  sceneCtx.lineTo(uav[0], uav[1]);
  sceneCtx.stroke();
  sceneCtx.setLineDash([]);

  sceneCtx.strokeStyle = "#ff2c2c";
  sceneCtx.lineWidth = 4;
  sceneCtx.beginPath();
  trajectory.forEach((pt, idx) => {
    const p = project(pt, rect, world);
    if (idx === 0) sceneCtx.moveTo(...p);
    else sceneCtx.lineTo(...p);
  });
  sceneCtx.stroke();

  trajectory.forEach((pt, idx) => {
    if (idx % 5 !== 0 && idx !== trajectory.length - 1) return;
    const p = project(pt, rect, world);
    sceneCtx.fillStyle = "#ff2c2c";
    sceneCtx.beginPath();
    sceneCtx.arc(p[0], p[1], 4, 0, Math.PI * 2);
    sceneCtx.fill();
  });

  users.forEach((user) => {
    const p = project([user[0], user[1], 0], rect, world);
    drawMarker(sceneCtx, p[0], p[1], "user", 4);
  });
  drawMarker(sceneCtx, base[0], base[1], "base", 11);
  drawMarker(sceneCtx, uav[0], uav[1], "uav", 12);
}

function drawCoverageMap() {
  fitCanvas(coverageCanvas, coverageCtx);
  const rect = coverageCanvas.getBoundingClientRect();
  const { data } = state;
  const frame = data.frames[state.frame];
  const users = frame.users || data.users;
  const map = data.coverageMap.probability;
  const size = data.coverageMap.size;
  const pad = 48;
  const plot = Math.min(rect.width - pad - 18, rect.height - pad - 14);
  const x0 = pad;
  const y0 = 42;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = map[size - 1 - y][x];
      coverageCtx.fillStyle = signalColor(-120 + v * 85, 1);
      coverageCtx.fillRect(x0 + (x / size) * plot, y0 + (y / size) * plot, plot / size + 1, plot / size + 1);
    }
  }
  coverageCtx.strokeStyle = "rgba(220,235,245,.55)";
  coverageCtx.strokeRect(x0, y0, plot, plot);

  coverageCtx.save();
  coverageCtx.strokeStyle = "#ff2c2c";
  coverageCtx.lineWidth = 2.5;
  coverageCtx.shadowColor = "rgba(255, 44, 44, 0.75)";
  coverageCtx.shadowBlur = 8;
  coverageCtx.beginPath();
  data.trajectory.slice(0, state.frame + 1).forEach((pt, idx) => {
    const x = x0 + (pt[0] / data.meta.worldM) * plot;
    const y = y0 + plot - (pt[1] / data.meta.worldM) * plot;
    if (idx === 0) coverageCtx.moveTo(x, y);
    else coverageCtx.lineTo(x, y);
  });
  coverageCtx.stroke();
  coverageCtx.restore();

  users.forEach((u) => {
    const x = x0 + (u[0] / data.meta.worldM) * plot;
    const y = y0 + plot - (u[1] / data.meta.worldM) * plot;
    drawMarker(coverageCtx, x, y, "user", 2.6);
  });
  const base = data.baseStation;
  drawMarker(coverageCtx, x0 + (base[0] / data.meta.worldM) * plot, y0 + plot - (base[1] / data.meta.worldM) * plot, "base", 9);
  const uav = data.frames[state.frame].uav;
  drawMarker(coverageCtx, x0 + (uav[0] / data.meta.worldM) * plot, y0 + plot - (uav[1] / data.meta.worldM) * plot, "uav", 9);
  drawAxes(coverageCtx, x0, y0, plot, "X (m)", "Y (m)");
}

function drawAxes(ctx, x0, y0, size, xLabel, yLabel) {
  ctx.fillStyle = "#dce8f2";
  ctx.font = "13px Segoe UI";
  ctx.fillText(xLabel, x0 + size / 2 - 16, y0 + size + 33);
  ctx.save();
  ctx.translate(x0 - 34, y0 + size / 2 + 16);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawMiniTrajectory() {
  fitCanvas(miniCanvas, miniCtx);
  const rect = miniCanvas.getBoundingClientRect();
  const { data } = state;
  const world = data.meta.worldM;
  const pad = 28;
  miniCtx.fillStyle = "rgba(7,15,24,0.96)";
  miniCtx.fillRect(0, 0, rect.width, rect.height);
  drawGrid(miniCtx, rect, world);
  miniCtx.strokeStyle = "#ff2c2c";
  miniCtx.lineWidth = 3;
  miniCtx.beginPath();
  data.trajectory.slice(0, state.frame + 1).forEach((pt, idx) => {
    const p = project(pt, rect, world);
    p[0] = Math.max(pad, Math.min(rect.width - pad, p[0]));
    p[1] = Math.max(38, Math.min(rect.height - pad, p[1]));
    if (idx === 0) miniCtx.moveTo(...p);
    else miniCtx.lineTo(...p);
  });
  miniCtx.stroke();
  const base = project(data.baseStation, rect, world);
  const uav = project(data.frames[state.frame].uav, rect, world);
  drawMarker(miniCtx, base[0], base[1], "base", 10);
  drawMarker(miniCtx, uav[0], uav[1], "uav", 10);
}

function drawLineChart() {
  fitCanvas(altitudeCanvas, altitudeCtx);
  const rect = altitudeCanvas.getBoundingClientRect();
  const data = state.data;
  const values = data.trajectory.map((p) => p[2]);
  const pad = 48;
  const x0 = pad;
  const y0 = 38;
  const w = rect.width - pad - 18;
  const h = rect.height - pad - 28;
  altitudeCtx.strokeStyle = "rgba(220,235,245,.28)";
  altitudeCtx.strokeRect(x0, y0, w, h);
  altitudeCtx.strokeStyle = "rgba(160,190,210,.25)";
  altitudeCtx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = y0 + (h / 4) * i;
    altitudeCtx.beginPath();
    altitudeCtx.moveTo(x0, y);
    altitudeCtx.lineTo(x0 + w, y);
    altitudeCtx.stroke();
  }
  altitudeCtx.strokeStyle = "#73df50";
  altitudeCtx.lineWidth = 3;
  altitudeCtx.beginPath();
  values.forEach((v, idx) => {
    const x = x0 + (idx / (values.length - 1)) * w;
    const y = y0 + h - ((v - 0) / 130) * h;
    if (idx === 0) altitudeCtx.moveTo(x, y);
    else altitudeCtx.lineTo(x, y);
  });
  altitudeCtx.stroke();
  const fx = x0 + (state.frame / (values.length - 1)) * w;
  const fy = y0 + h - (values[state.frame] / 130) * h;
  altitudeCtx.strokeStyle = "rgba(255,255,255,.5)";
  altitudeCtx.setLineDash([7, 6]);
  altitudeCtx.beginPath();
  altitudeCtx.moveTo(fx, y0);
  altitudeCtx.lineTo(fx, y0 + h);
  altitudeCtx.stroke();
  altitudeCtx.setLineDash([]);
  altitudeCtx.fillStyle = "#73df50";
  altitudeCtx.beginPath();
  altitudeCtx.arc(fx, fy, 6, 0, Math.PI * 2);
  altitudeCtx.fill();
  altitudeCtx.fillStyle = "#dce8f2";
  altitudeCtx.font = "13px Segoe UI";
  altitudeCtx.fillText("Altitude (m)", x0, 22);
  altitudeCtx.fillText("Time Step", x0 + w / 2 - 28, rect.height - 10);
}

function drawDonut() {
  const frame = state.data.frames[state.frame];
  const total = frame.losCount + frame.nlosCount;
  const pct = frame.losCount / total;
  donutCtx.clearRect(0, 0, donutCanvas.width, donutCanvas.height);
  donutCtx.lineWidth = 22;
  donutCtx.strokeStyle = "#f7a728";
  donutCtx.beginPath();
  donutCtx.arc(85, 82, 58, 0, Math.PI * 2);
  donutCtx.stroke();
  donutCtx.strokeStyle = "#2ab84d";
  donutCtx.beginPath();
  donutCtx.arc(85, 82, 58, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * pct);
  donutCtx.stroke();
}

function updateText() {
  const frame = state.data.frames[state.frame];
  const totalSteps = state.data.meta.maxSteps;
  const totalUsers = frame.losCount + frame.nlosCount;
  document.getElementById("stepText").textContent = totalSteps ? `${frame.step} / ${totalSteps}` : `${state.liveStep} / live`;
  document.getElementById("posText").textContent = `(${frame.uav.map((v) => v.toFixed(1)).join(", ")}) m`;
  document.getElementById("altText").textContent = `${frame.uav[2].toFixed(1)} m`;
  document.getElementById("sinrValue").textContent = `${frame.avgSnrDb.toFixed(2)} dB`;
  document.getElementById("rateValue").textContent = `${frame.avgRateMbps.toFixed(2)} Mbps`;
  document.getElementById("throughputValue").textContent = `${(frame.sumRateMbps / 1000).toFixed(2)} Gbps`;
  document.getElementById("latencyValue").textContent = `${frame.avgLatencyMs.toFixed(2)} ms`;
  document.getElementById("coverageValue").textContent = `${frame.coverageProbability.toFixed(2)} (${Math.round(frame.coverageProbability * 100)}%)`;
  document.getElementById("fairnessValue").textContent = frame.fairnessIndex.toFixed(2);
  document.getElementById("backhaulValue").textContent = `${Math.round(frame.backhaulQuality * 100)}%`;
  document.getElementById("losPct").textContent = `${Math.round((frame.losCount / totalUsers) * 100)}%`;
  document.getElementById("losText").textContent = `${frame.losCount} / ${totalUsers}`;
  document.getElementById("losCount").textContent = frame.losCount;
  document.getElementById("nlosCount").textContent = frame.nlosCount;
  document.getElementById("batteryText").textContent = `${(frame.batteryRemaining * 100).toFixed(1)}%`;
  document.getElementById("batteryFill").style.height = `${frame.batteryRemaining * 100}%`;
  document.getElementById("energyBar").style.width = `${(1 - frame.batteryRemaining) * 100}%`;
  document.getElementById("totalPower").textContent = `${(70 + (1 - frame.batteryRemaining) * 65).toFixed(1)} W`;
  document.getElementById("rewardValue").textContent = frame.reward.toFixed(1);
  document.getElementById("epsilonValue").textContent = state.data.training.epsilon[state.frame].toFixed(2);
  document.getElementById("tradeKnob").style.transform = `translate(${Math.round(frame.coverageProbability * 100)}%, -2px)`;
}

function render() {
  drawScene();
  drawCoverageMap();
  drawMiniTrajectory();
  drawLineChart();
  drawDonut();
  updateText();
}

function tick(ts) {
  if (!state.lastTick) state.lastTick = ts;
  if (state.playing && !state.ended && ts - state.lastTick > 260 / state.speed) {
    state.liveStep += 1;
    state.frame = state.liveStep % state.data.frames.length;
    state.lastTick = ts;
    render();
  }
  requestAnimationFrame(tick);
}

async function init() {
  const response = await fetch("./data/dashboard_data.json");
  state.data = await response.json();
  document.getElementById("envText").textContent = `${state.data.meta.worldM} m x ${state.data.meta.worldM} m`;
  document.getElementById("playPause").addEventListener("click", () => {
    if (state.ended) {
      state.ended = false;
    }
    state.playing = !state.playing;
    document.getElementById("playPause").textContent = state.playing ? "Pause" : "Play";
  });
  document.getElementById("resetBtn").addEventListener("click", () => {
    state.frame = 0;
    state.liveStep = 0;
    state.ended = false;
    state.playing = true;
    document.getElementById("playPause").textContent = "Pause";
    render();
  });
  document.getElementById("speedBtn").addEventListener("click", () => {
    state.speed = state.speed === 1 ? 2 : state.speed === 2 ? 4 : 1;
    document.getElementById("speedBtn").textContent = state.speed === 1 ? "Faster" : `${state.speed}x`;
  });
  document.getElementById("endBtn").addEventListener("click", () => {
    state.ended = true;
    state.playing = false;
    document.getElementById("playPause").textContent = "Play";
  });
  window.addEventListener("resize", render);
  render();
  requestAnimationFrame(tick);
}

init();
