"use strict";

const LEFT_BUTTON_DOWN = 0x0001;
const FIXED_DISPLAY_PROFILE = {
  name: "24_inch_full_hd",
  physicalSizeCm: [53.13, 29.89],
  resolutionPx: [1920, 1080],
  refreshHz: 240,
};

const els = {
  setup: document.querySelector("#setup"),
  experiment: document.querySelector("#experiment"),
  form: document.querySelector("#participant-form"),
  connection: document.querySelector("#connection"),
  start: document.querySelector("#start-button"),
  setupError: document.querySelector("#setup-error"),
  displayCheck: document.querySelector("#display-check"),
  configuredTrialCount: document.querySelector("#configured-trial-count"),
  canvas: document.querySelector("#task-canvas"),
  overlay: document.querySelector("#overlay"),
  overlayKicker: document.querySelector("#overlay-kicker"),
  overlayTitle: document.querySelector("#overlay-title"),
  overlayCopy: document.querySelector("#overlay-copy"),
  overlayButton: document.querySelector("#overlay-button"),
  overlaySecondaryButton: document.querySelector("#overlay-secondary-button"),
  overlayDangerButton: document.querySelector("#overlay-danger-button"),
  uiCursor: document.querySelector("#ui-cursor"),
  fatal: document.querySelector("#fatal"),
};

const ctx = els.canvas.getContext("2d", { alpha: false, desynchronized: true });

// Native 32 px frame extracted losslessly from the user-provided aero_arrow.cur.
// The CUR hotspot is (0, 0), matching the virtual cursor coordinates.
const aeroArrowCursor = new Image(32, 32);
aeroArrowCursor.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAA/klEQVR42u3VsWoCQRgE4Pl/XQgIt3dCTslbWN1T5BXzErlSLewCaYJw2kqKEyshJzFYjIWaiCTGwtsF2Sm3mY9hlwUkKtWkGbxFolI0HvhDSFSSpD/EHuAPcQTwgzgBuEf8AnCL+APgDnEGsP7aOECcAZDkcvlRM+IfAEnO54saERcASHI2e6doPLh2ffP0YLVao9W6AwBoIxk6fQVV9Ukgqp7z/uEV9LV5/+gEcChXk2Yidvw9u9hx7YDjcgAQTfLR6GW/QpLXusLuYv2UA4CaNIPYCUkWxZQi9s39J6lJXhTT3QoSvzoHNEz3AWInou0nNZ0eQkJCQkJCbjVbxl9Gb4rWTdMAAAAASUVORK5CYII=";
els.uiCursor.src = aeroArrowCursor.src;

const state = {
  ws: null,
  connected: false,
  session: null,
  experiment: null,
  display: null,
  appearance: null,
  templates: [],
  cpiOrder: [],
  blockIndex: 0,
  phase: "practice",
  phaseIndex: 0,
  attempt: 1,
  current: null,
  waitingForSave: false,
  intertrialMotionEnabled: false,
  pointerLockRawRequested: false,
  pointerLockRawGranted: false,
  initialTrialCursorSet: false,
  overlayActions: { primary: null, secondary: null, danger: null },
  cursorCommitRequests: new Map(),
  nextTrialTimer: null,
  paused: false,
  pauseAfterInvalidation: false,
  displayProfileOk: false,
  background: "rgb(24, 27, 33)",
  displayCalibration: null,
  inputBackend: "windows_raw_input",
  cpiConfirmed: false,
  cpiSettingMethod: "physical_button",
  cursorPresentationMode: "virtual_raw_input",
  clientInputQueue: [],
  clientInputFlushTimer: null,
  lastCursorX: 0,
  lastCursorY: 0,
  lastUiPointer: null,
  nativePointerVirtual: null,
};

function send(payload) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
    throw new Error("The local recorder is not connected.");
  }
  state.ws.send(JSON.stringify(payload));
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws`);
  state.ws = ws;
  ws.addEventListener("open", () => {
    state.connected = true;
    els.connection.classList.remove("waiting");
    els.connection.classList.add("connected");
    els.connection.lastElementChild.textContent = "Local recorder connected";
    updateDisplayCheck();
    refreshServerProtocolSummary();
    els.start.disabled = false;
  });
  ws.addEventListener("close", () => {
    state.connected = false;
    els.start.disabled = true;
    if (state.session) showFatal("The local recorder disconnected. This session cannot continue.");
    else {
      els.connection.classList.remove("connected");
      els.connection.lastElementChild.textContent = "Local recorder disconnected";
    }
  });
  ws.addEventListener("error", () => {
    els.setupError.textContent = "Could not connect to the Python recorder. Keep the server window open.";
  });
  ws.addEventListener("message", event => handleServerMessage(JSON.parse(event.data)));
}

async function refreshServerProtocolSummary() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) return;
    const health = await response.json();
    const recordedPerCpi = Number(health.measured_trials_per_cpi);
    const cpiConditions = Number(health.cpi_condition_count);
    if (Number.isFinite(recordedPerCpi) && Number.isFinite(cpiConditions)) {
      const total = recordedPerCpi * cpiConditions;
      els.configuredTrialCount.textContent = health.ab_pilot
        ? `${total} total (A/B pilot)`
        : `${total} total`;
    }
  } catch (_) {}
}

function optionalNumber(selector) {
  const value = document.querySelector(selector).value;
  return value === "" ? null : Number(value);
}

function requiredPositiveNumber(selector, label) {
  const value = Number(document.querySelector(selector).value);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive number.`);
  }
  return value;
}

function requiredPositiveInteger(selector, label) {
  const value = requiredPositiveNumber(selector, label);
  if (!Number.isInteger(value)) {
    throw new Error(`${label} must be an integer.`);
  }
  return value;
}

function browserEstimatedResolution() {
  return [
    Math.round(screen.width * devicePixelRatio),
    Math.round(screen.height * devicePixelRatio),
  ];
}

function screenWindowSnapshot() {
  const rect = els.canvas.getBoundingClientRect();
  const visual = window.visualViewport;
  return {
    screen_width_css_px: screen.width,
    screen_height_css_px: screen.height,
    screen_avail_width_css_px: screen.availWidth,
    screen_avail_height_css_px: screen.availHeight,
    inner_width_css_px: innerWidth,
    inner_height_css_px: innerHeight,
    outer_width_css_px: outerWidth,
    outer_height_css_px: outerHeight,
    visual_viewport_width_css_px: visual ? visual.width : null,
    visual_viewport_height_css_px: visual ? visual.height : null,
    color_depth: screen.colorDepth,
    pixel_depth: screen.pixelDepth,
    canvas_css_width_px: rect.width,
    canvas_css_height_px: rect.height,
    canvas_bounding_left_px: rect.left,
    canvas_bounding_top_px: rect.top,
    canvas_bounding_width_px: rect.width,
    canvas_bounding_height_px: rect.height,
  };
}

function fixedDisplayMismatchMessage(estimatedResolution) {
  const expectedResolution = FIXED_DISPLAY_PROFILE.resolutionPx;
  if (
    Math.abs(estimatedResolution[0] - expectedResolution[0]) <= 2
    && Math.abs(estimatedResolution[1] - expectedResolution[1]) <= 2
  ) {
    return "";
  }
  return (
    `Warning: this protocol is designed for a 1920 x 1080 Full HD display, `
    + `but this browser reports approximately ${estimatedResolution[0]} x ${estimatedResolution[1]} px. `
    + "You can still start the experiment, but the physical display size may not match the recommended 24 inch profile."
  );
}

function updateDisplayCheck() {
  const estimatedResolution = browserEstimatedResolution();
  const message = fixedDisplayMismatchMessage(estimatedResolution);
  state.displayProfileOk = message === "";
  els.displayCheck.classList.toggle("ok", state.displayProfileOk);
  els.displayCheck.classList.toggle("warning", !state.displayProfileOk);
  els.displayCheck.textContent = state.displayProfileOk
    ? `Display check OK: browser reports ${estimatedResolution[0]} x ${estimatedResolution[1]} px.`
    : message;
}

function displayCalibrationPayload() {
  const widthCm = requiredPositiveNumber("#screen-width-cm", "Screen width");
  const heightCm = requiredPositiveNumber("#screen-height-cm", "Screen height");
  const widthM = widthCm / 100;
  const heightM = heightCm / 100;
  const estimatedResolution = browserEstimatedResolution();
  const resolution = [
    requiredPositiveInteger("#resolution-width-px", "Resolution width"),
    requiredPositiveInteger("#resolution-height-px", "Resolution height"),
  ];
  const refreshHz = requiredPositiveNumber("#refresh-hz", "Refresh rate");
  return {
    source: "fixed_protocol_profile",
    physical_size_cm: [widthCm, heightCm],
    physical_size_m: [widthM, heightM],
    resolution_px: resolution,
    refresh_hz: refreshHz,
    px_per_m: [resolution[0] / widthM, resolution[1] / heightM],
    diagonal_in: Math.hypot(widthCm, heightCm) / 2.54,
    browser_screen_css_px: [screen.width, screen.height],
    browser_screen_avail_css_px: [screen.availWidth, screen.availHeight],
    browser_viewport_css_px: [innerWidth, innerHeight],
    browser_outer_window_css_px: [outerWidth, outerHeight],
    browser_visual_viewport_css_px: [
      window.visualViewport ? window.visualViewport.width : null,
      window.visualViewport ? window.visualViewport.height : null,
    ],
    browser_device_pixel_ratio: devicePixelRatio,
    browser_estimated_resolution_px: estimatedResolution,
    fixed_profile_warning: fixedDisplayMismatchMessage(estimatedResolution),
    fixed_profile_name: FIXED_DISPLAY_PROFILE.name,
    notes: document.querySelector("#display-notes").value.trim(),
  };
}

function applyFixedDisplayProfile() {
  document.querySelector("#screen-width-cm").value = String(FIXED_DISPLAY_PROFILE.physicalSizeCm[0]);
  document.querySelector("#screen-height-cm").value = String(FIXED_DISPLAY_PROFILE.physicalSizeCm[1]);
  document.querySelector("#resolution-width-px").value = String(FIXED_DISPLAY_PROFILE.resolutionPx[0]);
  document.querySelector("#resolution-height-px").value = String(FIXED_DISPLAY_PROFILE.resolutionPx[1]);
  document.querySelector("#refresh-hz").value = String(FIXED_DISPLAY_PROFILE.refreshHz);
}

function participantPayload() {
  const displayCalibration = displayCalibrationPayload();
  state.displayCalibration = displayCalibration;
  state.cpiSettingMethod = document.querySelector("#cpi-setting-method").value;
  state.cursorPresentationMode = document.querySelector("#cursor-presentation-mode").value;
  return {
    type: "hello",
    initial_cursor: {
      source: "awaiting_first_fullscreen_start_click",
      cursor_x_m: 0,
      cursor_y_m: 0,
    },
    participant: {
      id: document.querySelector("#participant-id").value.trim(),
      age: optionalNumber("#age"),
      sex: document.querySelector("#sex").value,
      dominant_hand: document.querySelector("#dominant-hand").value,
      gaming_hours_per_week: optionalNumber("#gaming-hours"),
      notes: document.querySelector("#notes").value.trim(),
    },
    hardware: {
      mouse_model: document.querySelector("#mouse-model").value.trim() || "not_reported",
      declared_mouse_polling_hz: requiredPositiveInteger("#mouse-polling-hz", "Effective Raw Input rate"),
      mouse_polling_rate_basis: "independent_windows_raw_input_diagnostic",
      cursor_presentation_mode: state.cursorPresentationMode,
      pointer_acceleration: document.querySelector("#pointer-acceleration").value,
      windows_pointer_speed: document.querySelector("#pointer-speed").value.trim() || "not_reported",
      cpi_setting_method: state.cpiSettingMethod,
      declared_resolution_px: displayCalibration.resolution_px,
      declared_refresh_hz: displayCalibration.refresh_hz,
      display_calibration: displayCalibration,
    },
    browser: {
      user_agent: navigator.userAgent,
      language: navigator.language,
      screen_css_px: [screen.width, screen.height],
      screen_avail_css_px: [screen.availWidth, screen.availHeight],
      viewport_css_px: [innerWidth, innerHeight],
      outer_window_css_px: [outerWidth, outerHeight],
      visual_viewport_css_px: [
        window.visualViewport ? window.visualViewport.width : null,
        window.visualViewport ? window.visualViewport.height : null,
      ],
      device_pixel_ratio: devicePixelRatio,
      color_depth: screen.colorDepth,
      pixel_depth: screen.pixelDepth,
      time_origin_ms: performance.timeOrigin,
    },
  };
}

async function enterExperimentFullscreen() {
  if (document.fullscreenElement === els.experiment) {
    return true;
  }
  if (!els.experiment.requestFullscreen) {
    return false;
  }
  els.experiment.classList.remove("hidden");
  resizeCanvas();
  try {
    await els.experiment.requestFullscreen();
    resizeCanvas();
    return true;
  } catch (_) {
    return false;
  }
}

function restoreSetupView() {
  if (document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  }
  els.experiment.classList.add("hidden");
  els.uiCursor.classList.add("hidden");
  els.setup.classList.remove("hidden");
}

function overlayPointerTrackingIsActive() {
  return Boolean(
    state.session
    && !els.experiment.classList.contains("hidden")
    && !els.overlay.classList.contains("hidden")
    && document.pointerLockElement !== els.canvas
  );
}

function overlayCursorIsActive() {
  return overlayPointerTrackingIsActive() && !usesNativeWindowsCursorMode();
}

function positionCustomCursor(clientX, clientY) {
  els.uiCursor.style.transform = `translate3d(${clientX}px, ${clientY}px, 0)`;
  els.uiCursor.classList.remove("hidden");
}

function virtualCursorClientPosition() {
  const rect = els.canvas.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return null;
  const point = pointToCanvas(state.lastCursorX, state.lastCursorY);
  return {
    clientX: rect.left + point[0] * rect.width / els.canvas.width,
    clientY: rect.top + point[1] * rect.height / els.canvas.height,
  };
}

function clientPointToVirtual(clientX, clientY) {
  const rect = els.canvas.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return null;
  const canvasX = (clientX - rect.left) * els.canvas.width / rect.width;
  const canvasY = (clientY - rect.top) * els.canvas.height / rect.height;
  const transform = taskTransform();
  return {
    x: clamp(
      (canvasX - transform.offsetX) / transform.scale - transform.widthM / 2,
      -transform.widthM / 2,
      transform.widthM / 2
    ),
    y: clamp(
      transform.heightM / 2 - (canvasY - transform.offsetY) / transform.scale,
      -transform.heightM / 2,
      transform.heightM / 2
    ),
  };
}

function updateCustomCursorVisibility() {
  if (!overlayCursorIsActive()) {
    els.uiCursor.classList.add("hidden");
    return;
  }
  const position = !state.initialTrialCursorSet && state.lastUiPointer
    ? state.lastUiPointer
    : virtualCursorClientPosition();
  if (position) positionCustomCursor(position.clientX, position.clientY);
}

function updateOverlayCursorFromClientPoint(clientX, clientY) {
  if (!overlayPointerTrackingIsActive()) return;
  const rect = els.canvas.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return;
  const canvasX = (clientX - rect.left) * els.canvas.width / rect.width;
  const canvasY = (clientY - rect.top) * els.canvas.height / rect.height;
  const transform = taskTransform();
  state.lastCursorX = clamp(
    (canvasX - transform.offsetX) / transform.scale - transform.widthM / 2,
    -transform.widthM / 2,
    transform.widthM / 2
  );
  state.lastCursorY = clamp(
    transform.heightM / 2 - (canvasY - transform.offsetY) / transform.scale,
    -transform.heightM / 2,
    transform.heightM / 2
  );
  if (!usesNativeWindowsCursorMode()) positionCustomCursor(clientX, clientY);
}

document.addEventListener("pointerdown", event => {
  if (document.pointerLockElement === els.canvas) return;
  updateOverlayCursorFromClientPoint(event.clientX, event.clientY);
  state.lastUiPointer = {
    clientX: event.clientX,
    clientY: event.clientY,
    screenX: event.screenX,
    screenY: event.screenY,
  };
  if (usesNativeWindowsCursorMode()) {
    state.nativePointerVirtual = clientPointToVirtual(event.clientX, event.clientY);
  }
}, true);

document.addEventListener("pointermove", event => {
  if (document.pointerLockElement === els.canvas) return;
  updateOverlayCursorFromClientPoint(event.clientX, event.clientY);
  state.lastUiPointer = {
    clientX: event.clientX,
    clientY: event.clientY,
    screenX: event.screenX,
    screenY: event.screenY,
  };
  if (usesNativeWindowsCursorMode() && state.current?.recording) {
    state.nativePointerVirtual = clientPointToVirtual(event.clientX, event.clientY);
  }
}, true);

els.form.addEventListener("submit", async event => {
  event.preventDefault();
  els.setupError.textContent = "";
  if (!state.connected) return;
  try {
    const payload = participantPayload();
    const fullscreenOk = await enterExperimentFullscreen();
    if (!fullscreenOk) {
      throw new Error("The browser refused fullscreen mode. Please click Start experiment again or allow fullscreen.");
    }
    send(payload);
    els.start.disabled = true;
  } catch (error) {
    restoreSetupView();
    els.setupError.textContent = `Could not start the session: ${error.message}`;
  }
});

function handleServerMessage(message) {
  switch (message.type) {
    case "session_ready":
      state.session = message.session_id;
      state.experiment = message.experiment;
      state.display = message.display;
      state.appearance = message.appearance;
      state.templates = message.templates;
      state.cpiOrder = message.cpi_order;
      state.inputBackend = message.input_backend || "windows_raw_input";
      state.cursorPresentationMode = message.cursor_presentation_mode || state.cursorPresentationMode;
      els.experiment.classList.toggle(
        "native-windows-cursor",
        usesNativeWindowsCursorMode()
      );
      state.cpiConfirmed = false;
      state.background = `rgb(${state.appearance.background_rgb.join(",")})`;
      state.lastCursorX = Number.isFinite(message.cursor_x_m) ? message.cursor_x_m : 0;
      state.lastCursorY = Number.isFinite(message.cursor_y_m) ? message.cursor_y_m : 0;
      els.setup.classList.add("hidden");
      els.experiment.classList.remove("hidden");
      resizeCanvas();
      showIntroduction();
      break;
    case "trial_ready":
      onTrialReady(message);
      break;
    case "trial_active":
      if (state.current && state.current.trialId === message.trial_id) {
        state.current.active = true;
      }
      break;
    case "raw_batch":
      onRawBatch(message);
      break;
    case "intertrial_cursor":
      onIntertrialCursor(message);
      break;
    case "overlay_cursor_committed":
      onOverlayCursorCommitted(message);
      break;
    case "trial_result":
      onTrialResult(message);
      break;
    case "trial_saved":
      onTrialSaved(message);
      break;
    case "session_saved":
      showFinished(message);
      break;
    case "error":
      showServerError(message.message);
      break;
  }
}

function showServerError(message) {
  if (!state.session) {
    els.setupError.textContent = message;
    els.start.disabled = !state.connected;
    restoreSetupView();
    return;
  }
  showFatal(message);
}

function configureOverlayButton(buttonEl, label, action) {
  if (!label) {
    buttonEl.classList.add("hidden");
    buttonEl.textContent = "";
    return null;
  }
  buttonEl.textContent = label;
  buttonEl.classList.remove("hidden");
  return action || null;
}

function showOverlay({
  kicker = "",
  title,
  paragraphs,
  button = "Continue",
  action,
  secondaryButton = null,
  secondaryAction = null,
  dangerButton = null,
  dangerAction = null,
}) {
  state.overlayActions.primary = configureOverlayButton(els.overlayButton, button, action);
  state.overlayActions.secondary = configureOverlayButton(
    els.overlaySecondaryButton, secondaryButton, secondaryAction
  );
  state.overlayActions.danger = configureOverlayButton(
    els.overlayDangerButton, dangerButton, dangerAction
  );
  els.overlayKicker.textContent = kicker;
  els.overlayTitle.textContent = title;
  els.overlayCopy.replaceChildren(...paragraphs.map(text => {
    const p = document.createElement("p");
    p.textContent = text;
    return p;
  }));
  els.overlay.classList.remove("hidden");
  clearCanvas();
  updateCustomCursorVisibility();
}

els.overlayButton.addEventListener("click", async () => {
  const action = state.overlayActions.primary;
  state.overlayActions.primary = null;
  if (action) await action();
});

els.overlaySecondaryButton.addEventListener("click", async () => {
  const action = state.overlayActions.secondary;
  state.overlayActions.secondary = null;
  if (action) await action();
});

els.overlayDangerButton.addEventListener("click", async () => {
  const action = state.overlayActions.danger;
  state.overlayActions.danger = null;
  if (action) await action();
});

async function toggleExperimentFullscreen() {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      const ok = await enterExperimentFullscreen();
      if (!ok) throw new Error("The browser refused fullscreen mode.");
    }
    return null;
  } catch (error) {
    return `Fullscreen could not be changed: ${error.message}`;
  }
}

document.addEventListener("keydown", event => {
  if (!state.session) return;
  if (event.key === "p" || event.key === "P" || event.key === "Escape") {
    event.preventDefault();
    if (
      !els.overlay.classList.contains("hidden")
      && !state.current?.active
      && !state.waitingForSave
    ) {
      return;
    }
    requestPause();
  }
});

document.addEventListener("mousemove", event => {
  if (!usesBrowserPointerLockBackend()) return;
  if (document.pointerLockElement !== els.canvas) return;
  if (state.current?.active) {
    queueClientInputEvent({
      movementX: event.movementX || 0,
      movementY: event.movementY || 0,
    });
  } else {
    queueClientIntertrialMovement(event.movementX || 0, event.movementY || 0);
  }
});

document.addEventListener("mousedown", event => {
  if (!usesBrowserPointerLockBackend()) return;
  if (document.pointerLockElement !== els.canvas) return;
  if (!state.current?.active) return;
  if (event.button !== 0) return;
  event.preventDefault();
  queueClientInputEvent({
    movementX: 0,
    movementY: 0,
    buttonFlags: LEFT_BUTTON_DOWN,
    immediate: true,
  });
});

function showIntroduction() {
  const backendNotice = usesBrowserPointerLockBackend()
    ? "This Mac/Linux-compatible pilot mode uses browser Pointer Lock movement, not Windows Raw Input."
    : "This Windows mode uses the local Raw Input recorder for low-level mouse deltas.";
  const cursorNotice = usesNativeWindowsCursorMode()
    ? "A/B condition: the native Windows cursor is visible and authoritative for pointing; Raw Input counts are recorded separately."
    : "A/B condition: the controlled virtual Windows-style cursor is visible.";
  const startTargetEnabled = Boolean(state.experiment?.start_target_enabled);
  showOverlay({
    kicker: "Instructions",
    title: startTargetEnabled
      ? "Use the red start target, then point to the light-orange target"
      : "Point to the filled target",
    paragraphs: [
      backendNotice,
      cursorNotice,
      startTargetEnabled
        ? "Each task begins with a red start target. Click it to set the exact sampled starting position."
        : "The random target is light orange. Your cursor is the mouse arrow.",
      startTargetEnabled
        ? "After the red target is clicked, point to the light-orange task target. Your cursor is the mouse arrow."
        : "There is no center reference target: you may move freely between trials, and each new target starts from the cursor's current position.",
      "Move as naturally as possible. The first left click ends each trial, even when it misses.",
      "The experiment starts in fullscreen for the 24 inch Full HD profile. Press Escape, P, or Pause to open the menu.",
    ],
    button: "View the first CPI block",
    action: showBlockStart,
  });
}

function showBlockStart() {
  disableIntertrialMotion("block_screen");
  document.exitPointerLock?.();
  const cpi = state.cpiOrder[state.blockIndex];
  state.cpiConfirmed = false;
  showOverlay({
    kicker: `Block ${state.blockIndex + 1} of ${state.cpiOrder.length}`,
    title: `Set the mouse to ${cpi} CPI`,
    paragraphs: [
      `Use the physical CPI button or mouse software to select exactly ${cpi} CPI.`,
      "Do not change the Windows pointer speed or acceleration setting between blocks.",
      `${state.experiment.practice_trials_per_cpi} practice trials will come first.`,
    ],
    button: `Confirm ${cpi} CPI and start`,
    action: async () => {
      await acquirePointerLock();
      await commitOverlayCursorPosition("block_start_button_click");
      state.cpiConfirmed = true;
      send({
        type: "confirm_cpi",
        cpi,
        block_index: state.blockIndex + 1,
        cpi_setting_method: state.cpiSettingMethod,
        browser_ms: performance.now(),
      });
      state.phase = "practice";
      state.phaseIndex = 0;
      state.attempt = 1;
      els.overlay.classList.add("hidden");
      scheduleTrial();
    },
  });
}

function logPauseEvent(eventName) {
  try {
    send({
      type: "pause_event",
      event: eventName,
      phase: state.phase,
      trial_stage: state.current?.trialStage ?? null,
      block_index: state.blockIndex + 1,
      phase_index: state.phaseIndex,
      trial_id: state.current?.trialId ?? null,
      browser_ms: performance.now(),
    });
  } catch (_) {}
}

function clearScheduledTrial() {
  if (state.nextTrialTimer != null) {
    clearTimeout(state.nextTrialTimer);
    state.nextTrialTimer = null;
  }
}

function requestPause() {
  if (!state.session || state.paused) return;
  state.paused = true;
  logPauseEvent("pause_start");
  clearScheduledTrial();
  if (state.current?.active && !state.waitingForSave) {
    state.pauseAfterInvalidation = true;
    try { send({ type: "invalidate_trial", reason: "participant_paused" }); } catch (_) {}
    showOverlay({
      kicker: "Pause requested",
      title: "Saving the current trial state",
      paragraphs: [
        "The active trial is being marked as invalid so it can be repeated after the pause.",
      ],
      button: null,
    });
    return;
  }
  if (state.waitingForSave) {
    state.pauseAfterInvalidation = true;
    showOverlay({
      kicker: "Pause requested",
      title: "Waiting for the trial to be saved",
      paragraphs: ["The pause menu will open as soon as the current trial is written."],
      button: null,
    });
    return;
  }
  showPauseMenu();
}

function showPauseMenu(statusMessage = null) {
  state.paused = true;
  disableIntertrialMotion("pause_menu");
  clearScheduledTrial();
  exitPointerLockPreservingCursor("pause_menu");
  const fullscreenLabel = document.fullscreenElement ? "Exit fullscreen" : "Enter fullscreen";
  const paragraphs = [
    "No trial is currently being recorded.",
    "Resume continues from the same block and trial.",
    "You can also change fullscreen mode from this menu.",
  ];
  if (statusMessage) {
    paragraphs.unshift(statusMessage);
  }
  showOverlay({
    kicker: "Main menu",
    title: "Experiment paused",
    paragraphs,
    button: "Resume experiment",
    action: resumeFromPause,
    secondaryButton: fullscreenLabel,
    secondaryAction: async () => {
      const fullscreenError = await toggleExperimentFullscreen();
      showPauseMenu(fullscreenError);
    },
    dangerButton: "Save and stop",
    dangerAction: abortSessionFromMenu,
  });
}

async function resumeFromPause() {
  state.paused = false;
  state.pauseAfterInvalidation = false;
  logPauseEvent("pause_end");
  await acquirePointerLock();
  await commitOverlayCursorPosition("resume_button_click");
  els.overlay.classList.add("hidden");
  scheduleTrial();
}

function abortSessionFromMenu() {
  clearScheduledTrial();
  state.paused = true;
  document.exitPointerLock?.();
  send({ type: "abort_session", reason: "participant_menu_stop", browser_ms: performance.now() });
}

function usesBrowserPointerLockBackend() {
  return state.inputBackend === "browser_pointer_lock";
}

function usesNativeWindowsCursorMode() {
  return state.cursorPresentationMode === "native_windows";
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function requestOsCursorSync(reason) {
  if (!state.session || !state.connected || usesNativeWindowsCursorMode()) return;
  const dpr = window.devicePixelRatio || 1;
  try {
    send({
      type: "sync_os_cursor_to_virtual",
      reason,
      screen_origin_x_px: Math.round(window.screenX * dpr),
      screen_origin_y_px: Math.round(window.screenY * dpr),
      browser_ms: performance.now(),
    });
  } catch (_) {}
}

function nativeScreenMappings() {
  if (!usesNativeWindowsCursorMode()) return null;
  const rect = els.canvas.getBoundingClientRect();
  const transform = taskTransform();
  const dpr = window.devicePixelRatio || 1;
  const pointer = state.lastUiPointer;
  const viewportOriginX = pointer
    ? pointer.screenX - pointer.clientX
    : window.screenX;
  const viewportOriginY = pointer
    ? pointer.screenY - pointer.clientY
    : window.screenY;
  const logicalOriginX = viewportOriginX + rect.left;
  const logicalOriginY = viewportOriginY + rect.top;
  return {
    logical: {
      basis: "logical",
      screen_origin_x_px: logicalOriginX,
      screen_origin_y_px: logicalOriginY,
      task_offset_x_px: transform.offsetX / dpr,
      task_offset_y_px: transform.offsetY / dpr,
      task_scale_px_per_m: transform.scale / dpr,
    },
    physical: {
      basis: "physical",
      screen_origin_x_px: logicalOriginX * dpr,
      screen_origin_y_px: logicalOriginY * dpr,
      task_offset_x_px: transform.offsetX,
      task_offset_y_px: transform.offsetY,
      task_scale_px_per_m: transform.scale,
    },
  };
}

async function commitOverlayCursorPosition(reason) {
  if (!Number.isFinite(state.lastCursorX) || !Number.isFinite(state.lastCursorY)) return false;
  const requestId = `cursor-commit-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const acknowledgement = new Promise(resolve => {
    const timeout = setTimeout(() => {
      state.cursorCommitRequests.delete(requestId);
      resolve(false);
    }, 1500);
    state.cursorCommitRequests.set(requestId, { resolve, timeout });
  });
  send({
    type: "commit_overlay_cursor",
    request_id: requestId,
    cursor_x_m: state.lastCursorX,
    cursor_y_m: state.lastCursorY,
    source: reason,
    native_screen_mappings: nativeScreenMappings(),
    browser_ms: performance.now(),
  });
  const acknowledged = await acknowledgement;
  requestOsCursorSync(reason);
  return acknowledged;
}

function exitPointerLockPreservingCursor(reason) {
  if (document.pointerLockElement !== els.canvas) {
    document.exitPointerLock?.();
    return;
  }
  let synchronized = false;
  const synchronize = () => {
    if (synchronized || document.pointerLockElement === els.canvas) return;
    synchronized = true;
    document.removeEventListener("pointerlockchange", synchronize);
    requestOsCursorSync(reason);
  };
  document.addEventListener("pointerlockchange", synchronize);
  document.exitPointerLock();
  setTimeout(synchronize, 100);
}

function setInitialCursorFromFullscreenStartClick() {
  if (state.initialTrialCursorSet) return;
  const pointer = state.lastUiPointer;
  if (!pointer || !state.session || !state.display || !state.appearance) return;
  const rect = els.canvas.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return;
  const canvasX = (pointer.clientX - rect.left) * (els.canvas.width / rect.width);
  const canvasY = (pointer.clientY - rect.top) * (els.canvas.height / rect.height);
  const transform = taskTransform();
  const cursorX = clamp(
    (canvasX - transform.offsetX) / transform.scale - transform.widthM / 2,
    -transform.widthM / 2,
    transform.widthM / 2
  );
  const cursorY = clamp(
    transform.heightM / 2 - (canvasY - transform.offsetY) / transform.scale,
    -transform.heightM / 2,
    transform.heightM / 2
  );
  state.lastCursorX = cursorX;
  state.lastCursorY = cursorY;
  state.initialTrialCursorSet = true;
  drawInterTrialCursor();
  send({
    type: "set_initial_cursor",
    cursor_x_m: cursorX,
    cursor_y_m: cursorY,
    source: "first_fullscreen_start_button",
    browser_ms: performance.now(),
  });
}

async function acquirePointerLock() {
  setInitialCursorFromFullscreenStartClick();
  if (usesNativeWindowsCursorMode()) {
    state.pointerLockRawRequested = false;
    state.pointerLockRawGranted = false;
    return;
  }
  state.pointerLockRawRequested = true;
  state.pointerLockRawGranted = false;
  try {
    const result = els.canvas.requestPointerLock({ unadjustedMovement: true });
    if (result && typeof result.then === "function") await result;
    state.pointerLockRawGranted = document.pointerLockElement === els.canvas;
  } catch (_) {
    const fallback = els.canvas.requestPointerLock();
    if (fallback && typeof fallback.then === "function") await fallback;
  }
  if (document.pointerLockElement !== els.canvas) {
    throw new Error("Pointer lock is required for the experiment.");
  }
}

function flushClientInputEvents() {
  if (state.clientInputFlushTimer != null) {
    clearTimeout(state.clientInputFlushTimer);
    state.clientInputFlushTimer = null;
  }
  if (!usesBrowserPointerLockBackend() || !state.current || state.clientInputQueue.length === 0) {
    state.clientInputQueue = [];
    return;
  }
  const events = state.clientInputQueue.splice(0);
  try {
    send({
      type: "client_input_batch",
      trial_id: state.current.trialId,
      events,
    });
  } catch (_) {
    state.clientInputQueue.unshift(...events);
  }
}

function queueClientInputEvent({ movementX = 0, movementY = 0, buttonFlags = 0, buttonData = 0, immediate = false }) {
  if (!usesBrowserPointerLockBackend()) return;
  const trial = state.current;
  if (!trial || !trial.active || state.paused || state.waitingForSave) return;
  const transform = taskTransform();
  const dxM = movementX / transform.scale;
  const dyM = -movementY / transform.scale;
  trial.cursorX = clamp(trial.cursorX + dxM, -transform.widthM / 2, transform.widthM / 2);
  trial.cursorY = clamp(trial.cursorY + dyM, -transform.heightM / 2, transform.heightM / 2);
  trial.rawSequence += 1;
  trial.rawSinceFrame += 1;
  if ((movementX !== 0 || movementY !== 0) && trial.firstMovementMs == null) {
    trial.firstMovementMs = performance.now();
  }
  if (buttonFlags & LEFT_BUTTON_DOWN) {
    trial.clickMs = performance.now();
  }
  state.lastCursorX = trial.cursorX;
  state.lastCursorY = trial.cursorY;
  state.clientInputQueue.push({
    sequence: trial.rawSequence,
    browser_ms: performance.now(),
    dx_counts: movementX,
    dy_counts: movementY,
    button_flags: buttonFlags,
    button_data: buttonData,
    cursor_x_m: trial.cursorX,
    cursor_y_m: trial.cursorY,
  });
  if (immediate) {
    flushClientInputEvents();
  } else if (state.clientInputFlushTimer == null) {
    state.clientInputFlushTimer = setTimeout(flushClientInputEvents, 4);
  }
}

function queueClientIntertrialMovement(movementX, movementY) {
  if (
    !usesBrowserPointerLockBackend()
    || !state.intertrialMotionEnabled
    || state.paused
    || !state.cpiConfirmed
  ) return;
  if (movementX === 0 && movementY === 0) return;
  const transform = taskTransform();
  state.lastCursorX = clamp(
    state.lastCursorX + movementX / transform.scale,
    -transform.widthM / 2,
    transform.widthM / 2
  );
  state.lastCursorY = clamp(
    state.lastCursorY - movementY / transform.scale,
    -transform.heightM / 2,
    transform.heightM / 2
  );
  drawInterTrialCursor();
  send({
    type: "intertrial_client_input",
    dx_counts: movementX,
    dy_counts: movementY,
    cursor_x_m: state.lastCursorX,
    cursor_y_m: state.lastCursorY,
    browser_ms: performance.now(),
  });
}

function disableIntertrialMotion(reason) {
  state.intertrialMotionEnabled = false;
  if (!state.session || !state.connected) return;
  try {
    send({ type: "set_intertrial_motion", enabled: false, reason });
  } catch (_) {}
}

function phaseTemplates() {
  const practiceCount = state.experiment.practice_trials_per_cpi;
  if (state.phase === "practice") return state.templates.slice(0, practiceCount);
  return state.templates.slice(practiceCount, practiceCount + state.experiment.measured_trials_per_cpi);
}

function scheduleTrial() {
  if (state.paused) return;
  clearScheduledTrial();
  drawInterTrialCursor();
  state.nextTrialTimer = setTimeout(() => {
    state.nextTrialTimer = null;
    const initialStage = state.experiment?.start_target_enabled ? "start" : "task";
    beginTrial(initialStage);
  }, state.experiment.inter_trial_s * 1000);
}

function beginTrial(trialStage = "task") {
  if (state.paused) return;
  if (!state.cpiConfirmed) {
    showFatal("The CPI condition was not confirmed for this block.");
    return;
  }
  const templates = phaseTemplates();
  const template = templates[state.phaseIndex];
  if (!template) return endPhase();
  send({
    type: "start_trial",
    template_id: template.template_id,
    phase: state.phase,
    trial_stage: trialStage,
    cpi: state.cpiOrder[state.blockIndex],
    cpi_confirmed: state.cpiConfirmed,
    cpi_setting_method: state.cpiSettingMethod,
    block_index: state.blockIndex + 1,
    attempt: state.attempt,
  });
}

function onTrialReady(message) {
  const template = message.template;
  state.clientInputQueue = [];
  state.intertrialMotionEnabled = false;
  if (state.clientInputFlushTimer != null) {
    clearTimeout(state.clientInputFlushTimer);
    state.clientInputFlushTimer = null;
  }
  const cursorX = Number.isFinite(message.cursor_x_m) ? message.cursor_x_m : state.lastCursorX;
  const cursorY = Number.isFinite(message.cursor_y_m) ? message.cursor_y_m : state.lastCursorY;
  state.current = {
    trialId: message.trial_id,
    trialStage: message.trial_stage || "task",
    template,
    cursorX,
    cursorY,
    active: false,
    recording: true,
    frames: [],
    rawSinceFrame: 0,
    rawSequence: 0,
    lastFrameMs: null,
    onsetMs: null,
    firstMovementMs: null,
    clickMs: null,
    endMs: null,
    outcome: null,
  };
  state.lastCursorX = cursorX;
  state.lastCursorY = cursorY;
  requestAnimationFrame(timestamp => {
    if (!state.current) return;
    state.current.onsetMs = timestamp;
    drawTask();
    send({ type: "target_onset", trial_id: message.trial_id, browser_ms: timestamp });
    requestAnimationFrame(animationFrame);
  });
}

function animationFrame(timestamp) {
  const trial = state.current;
  if (!trial || !trial.recording) return;
  drawTask();
  const interval = trial.lastFrameMs == null ? 0 : timestamp - trial.lastFrameMs;
  const nativePointer = usesNativeWindowsCursorMode() ? state.nativePointerVirtual : null;
  const nativeAlignmentError = nativePointer
    ? Math.hypot(nativePointer.x - trial.cursorX, nativePointer.y - trial.cursorY)
    : null;
  trial.frames.push({
    browser_ms: timestamp,
    interval_ms: interval,
    cursor_x_m: trial.cursorX,
    cursor_y_m: trial.cursorY,
    raw_events: trial.rawSinceFrame,
    raw_sequence: trial.rawSequence,
    focused: document.hasFocus(),
    visible: document.visibilityState === "visible",
    pointer_locked: document.pointerLockElement === els.canvas,
    native_cursor_x_m: nativePointer?.x ?? null,
    native_cursor_y_m: nativePointer?.y ?? null,
    native_alignment_error_m: nativeAlignmentError,
  });
  trial.rawSinceFrame = 0;
  trial.lastFrameMs = timestamp;
  requestAnimationFrame(animationFrame);
}

function onRawBatch(message) {
  const trial = state.current;
  if (!trial || trial.trialId !== message.trial_id) return;
  for (const event of message.events) {
    const moved = event.cursor_x_m !== trial.cursorX || event.cursor_y_m !== trial.cursorY;
    trial.cursorX = event.cursor_x_m;
    trial.cursorY = event.cursor_y_m;
    state.lastCursorX = trial.cursorX;
    state.lastCursorY = trial.cursorY;
    trial.rawSequence = event.sequence;
    trial.rawSinceFrame += 1;
    if (moved && trial.firstMovementMs == null) trial.firstMovementMs = performance.now();
    if (event.button_flags & LEFT_BUTTON_DOWN) trial.clickMs = performance.now();
  }
}

function onIntertrialCursor(message) {
  if (!Number.isFinite(message.cursor_x_m) || !Number.isFinite(message.cursor_y_m)) return;
  state.lastCursorX = message.cursor_x_m;
  state.lastCursorY = message.cursor_y_m;
  if (!state.current?.active) drawInterTrialCursor();
}

function onOverlayCursorCommitted(message) {
  onIntertrialCursor(message);
  const pending = state.cursorCommitRequests.get(message.request_id);
  if (!pending) return;
  clearTimeout(pending.timeout);
  state.cursorCommitRequests.delete(message.request_id);
  pending.resolve(true);
}

function onTrialResult(message) {
  const trial = state.current;
  if (!trial || trial.trialId !== message.trial_id) return;
  trial.active = false;
  trial.recording = false;
  trial.outcome = message.outcome;
  trial.valid = message.valid;
  trial.trialStage = message.trial_stage || trial.trialStage || "task";
  trial.endMs = performance.now();
  state.intertrialMotionEnabled = trial.trialStage === "task";
  trial.cursorX = Number.isFinite(message.cursor_x_m) ? message.cursor_x_m : trial.cursorX;
  trial.cursorY = Number.isFinite(message.cursor_y_m) ? message.cursor_y_m : trial.cursorY;
  state.lastCursorX = trial.cursorX;
  state.lastCursorY = trial.cursorY;
  drawInterTrialCursor();
  const geometry = renderedTargetGeometry(trial.template);
  const screenWindow = screenWindowSnapshot();
  state.waitingForSave = true;
  send({
    type: "trial_client_data",
    trial_id: trial.trialId,
    frames: trial.frames,
    client: {
      input_backend: state.inputBackend,
      cursor_presentation_mode: state.cursorPresentationMode,
      time_origin_ms: performance.timeOrigin,
      first_movement_ms: trial.firstMovementMs,
      click_ms: trial.clickMs,
      end_ms: trial.endMs,
      pointer_lock_raw_requested: state.pointerLockRawRequested,
      pointer_lock_raw_granted: state.pointerLockRawGranted,
      fullscreen: document.fullscreenElement === els.experiment,
      width_px: els.canvas.width,
      height_px: els.canvas.height,
      device_pixel_ratio: devicePixelRatio,
      display_calibration: state.displayCalibration,
      screen_window: screenWindow,
      rendered_target_radius_px: geometry.radiusPx,
      rendered_target_radius_m: geometry.radiusM,
      render_scale_px_per_virtual_m: geometry.scalePxPerVirtualM,
      rendered_task_width_px: geometry.renderedTaskWidthPx,
      rendered_task_height_px: geometry.renderedTaskHeightPx,
      rendered_task_width_physical_m: geometry.renderedTaskWidthPhysicalM,
      rendered_task_height_physical_m: geometry.renderedTaskHeightPhysicalM,
      rendered_target_radius_physical_m: geometry.renderedTargetRadiusPhysicalM,
    },
  });
}

function onTrialSaved(message) {
  const trial = state.current;
  if (!trial || trial.trialId !== message.trial_id) return;
  state.waitingForSave = false;
  if (Number.isFinite(message.cursor_x_m) && Number.isFinite(message.cursor_y_m)) {
    state.lastCursorX = message.cursor_x_m;
    state.lastCursorY = message.cursor_y_m;
  } else {
    state.lastCursorX = trial.cursorX;
    state.lastCursorY = trial.cursorY;
  }
  drawInterTrialCursor();
  const valid = Boolean(trial.valid);
  const feedbackDelayMs = (
    trial.trialStage === "start" && valid && trial.outcome === "success"
  ) ? 0 : state.experiment.feedback_s * 1000;
  setTimeout(() => {
    if (trial.trialStage === "start") {
      state.current = null;
      if (state.pauseAfterInvalidation || trial.outcome === "participant_paused") {
        state.pauseAfterInvalidation = false;
        showPauseMenu();
        return;
      }
      if (valid && trial.outcome === "success") {
        beginTrial("task");
        return;
      }
      state.attempt += 1;
      if (!valid) {
        exitPointerLockPreservingCursor("invalid_start_trial");
        showOverlay({
          kicker: "Trial invalidated",
          title: "The trial will be repeated",
          paragraphs: ["The page lost focus, visibility, or pointer lock during the trial."],
          button: "Repeat trial",
          action: async () => {
            await acquirePointerLock();
            await commitOverlayCursorPosition("repeat_start_trial_button_click");
            els.overlay.classList.add("hidden");
            scheduleTrial();
          },
        });
      } else {
        scheduleTrial();
      }
      return;
    }
    if (valid) {
      state.phaseIndex += 1;
      state.attempt = 1;
    } else {
      state.attempt += 1;
    }
    state.current = null;
    if (state.pauseAfterInvalidation || trial.outcome === "participant_paused") {
      state.pauseAfterInvalidation = false;
      showPauseMenu();
      return;
    }
    if (!valid) {
      exitPointerLockPreservingCursor("invalid_task_trial");
      showOverlay({
        kicker: "Trial invalidated",
        title: "The trial will be repeated",
        paragraphs: ["The page lost focus, visibility, or pointer lock during the trial."],
        button: "Repeat trial",
        action: async () => {
          await acquirePointerLock();
          await commitOverlayCursorPosition("repeat_task_trial_button_click");
          els.overlay.classList.add("hidden");
          scheduleTrial();
        },
      });
    } else {
      scheduleTrial();
    }
  }, feedbackDelayMs);
}

function endPhase() {
  disableIntertrialMotion("phase_complete");
  exitPointerLockPreservingCursor("phase_complete");
  if (state.phase === "practice") {
    showOverlay({
      kicker: `${state.cpiOrder[state.blockIndex]} CPI`,
      title: "Practice is complete",
      paragraphs: [
        `The ${state.experiment.measured_trials_per_cpi} recorded trials will now begin.`,
        "Take a short break if needed, but keep the CPI setting unchanged.",
        "The same cursor remains active on this screen. The next trial starts exactly where you click the button.",
      ],
      button: "Begin recorded trials",
      action: async () => {
        await acquirePointerLock();
        await commitOverlayCursorPosition("measured_phase_start_button_click");
        state.phase = "measured";
        state.phaseIndex = 0;
        state.attempt = 1;
        els.overlay.classList.add("hidden");
        scheduleTrial();
      },
    });
  } else {
    state.blockIndex += 1;
    if (state.blockIndex < state.cpiOrder.length) showBlockStart();
    else send({ type: "session_complete" });
  }
}

function showFinished(message = {}) {
  disableIntertrialMotion("session_complete");
  document.exitPointerLock?.();
  clearScheduledTrial();
  const aborted = Boolean(message.aborted);
  const savedSession = message.session_id || state.session;
  showOverlay({
    kicker: aborted ? "Session stopped" : "Session saved",
    title: aborted ? "The partial session was saved" : "The experiment is complete. Thank you!",
    paragraphs: [
      `Session: ${savedSession}`,
      aborted
        ? "The data collected before stopping was saved by the local recorder."
        : "All raw mouse events, rendered frames, clicks, and trial summaries were saved by the local recorder.",
      "You can now return to the setup screen for a new participant, or close the browser tab.",
    ],
    button: "Back to setup",
    action: async () => {
      if (document.fullscreenElement) await document.exitFullscreen();
      resetToSetup();
    },
  });
}

function resetToSetup() {
  document.exitPointerLock?.();
  clearScheduledTrial();
  els.form.reset();
  applyFixedDisplayProfile();
  updateDisplayCheck();
  Object.assign(state, {
    session: null,
    experiment: null,
    display: null,
    appearance: null,
    templates: [],
    cpiOrder: [],
    blockIndex: 0,
    phase: "practice",
    phaseIndex: 0,
    attempt: 1,
    current: null,
    waitingForSave: false,
    intertrialMotionEnabled: false,
    pointerLockRawRequested: false,
    pointerLockRawGranted: false,
    initialTrialCursorSet: false,
    overlayActions: { primary: null, secondary: null, danger: null },
    cursorCommitRequests: new Map(),
    nextTrialTimer: null,
    paused: false,
    pauseAfterInvalidation: false,
    background: "rgb(24, 27, 33)",
    displayCalibration: null,
    inputBackend: "windows_raw_input",
    cursorPresentationMode: "virtual_raw_input",
    clientInputQueue: [],
    clientInputFlushTimer: null,
    lastCursorX: 0,
    lastCursorY: 0,
    lastUiPointer: null,
    nativePointerVirtual: null,
  });
  els.fatal.classList.add("hidden");
  els.fatal.textContent = "";
  els.setupError.textContent = "";
  els.overlay.classList.add("hidden");
  els.uiCursor.classList.add("hidden");
  els.experiment.classList.remove("native-windows-cursor");
  els.experiment.classList.add("hidden");
  els.setup.classList.remove("hidden");
  els.start.disabled = !state.connected;
}

function showFatal(message) {
  els.fatal.textContent = message;
  els.fatal.classList.remove("hidden");
  if (state.current?.active) {
    try { send({ type: "invalidate_trial", reason: "client_error" }); } catch (_) {}
  }
}

function resizeCanvas() {
  const [renderWidthPx, renderHeightPx] = FIXED_DISPLAY_PROFILE.resolutionPx;
  const dpr = window.devicePixelRatio || 1;
  if (document.fullscreenElement === els.experiment) {
    els.canvas.width = Math.max(1, Math.round(window.innerWidth * dpr));
    els.canvas.height = Math.max(1, Math.round(window.innerHeight * dpr));
    els.canvas.style.width = "100vw";
    els.canvas.style.height = "100vh";
  } else {
    els.canvas.width = renderWidthPx;
    els.canvas.height = renderHeightPx;
    els.canvas.style.width = `${renderWidthPx}px`;
    els.canvas.style.height = `${renderHeightPx}px`;
  }
  const shouldKeepIdleCursorVisible = (
    state.session &&
    state.display &&
    state.appearance &&
    !state.current?.recording
  );
  if (shouldKeepIdleCursorVisible) {
    drawInterTrialCursor();
  } else {
    clearCanvas();
  }
}

function taskTransform() {
  const [widthM, heightM] = state.display.virtual_size_m;
  const fullscreen = document.fullscreenElement === els.experiment;
  const [fixedWidthPx, fixedHeightPx] = state.display.resolution_px;
  const renderWidthPx = fullscreen ? els.canvas.width : fixedWidthPx;
  const renderHeightPx = fullscreen ? els.canvas.height : fixedHeightPx;
  const scaleX = renderWidthPx / widthM;
  const scaleY = renderHeightPx / heightM;
  const scale = Math.min(scaleX, scaleY);
  const taskWidthPx = widthM * scale;
  const taskHeightPx = heightM * scale;
  return {
    scale,
    offsetX: (els.canvas.width - taskWidthPx) / 2,
    offsetY: (els.canvas.height - taskHeightPx) / 2,
    widthM,
    heightM,
    renderWidthPx: taskWidthPx,
    renderHeightPx: taskHeightPx,
    fullscreen,
  };
}

function pointToCanvas(xM, yM) {
  const t = taskTransform();
  return [
    t.offsetX + (xM + t.widthM / 2) * t.scale,
    t.offsetY + (t.heightM / 2 - yM) * t.scale,
  ];
}

function renderedTargetGeometry(template) {
  const t = taskTransform();
  const radiusPx = Math.max(1, Math.round(template.target_radius_m * t.scale));
  const taskWidthPx = t.renderWidthPx;
  const taskHeightPx = t.renderHeightPx;
  const calibration = state.displayCalibration;
  const pxPerM = calibration?.px_per_m ?? null;
  const physicalPxPerM = pxPerM ? (Number(pxPerM[0]) + Number(pxPerM[1])) / 2 : null;
  return {
    radiusPx,
    radiusM: radiusPx / t.scale,
    scalePxPerVirtualM: t.scale,
    renderedTaskWidthPx: taskWidthPx,
    renderedTaskHeightPx: taskHeightPx,
    renderedTaskWidthPhysicalM: pxPerM ? taskWidthPx / Number(pxPerM[0]) : null,
    renderedTaskHeightPhysicalM: pxPerM ? taskHeightPx / Number(pxPerM[1]) : null,
    renderedTargetRadiusPhysicalM: physicalPxPerM ? radiusPx / physicalPxPerM : null,
  };
}

function clearCanvas() {
  ctx.fillStyle = state.background;
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
}

function phaseTrialProgress() {
  if (!state.experiment || !["practice", "measured"].includes(state.phase)) return null;
  const total = Number(
    state.phase === "practice"
      ? state.experiment.practice_trials_per_cpi
      : state.experiment.measured_trials_per_cpi
  );
  if (!Number.isFinite(total) || total <= 0) return null;
  return {
    current: Math.min(total, Math.max(1, Number(state.phaseIndex) + 1)),
    total,
  };
}

function drawPhaseTrialCounter() {
  const progress = phaseTrialProgress();
  if (!progress || !state.appearance) return;
  const dpr = window.devicePixelRatio || 1;
  const fontSize = Number(state.appearance.phase_counter_font_px ?? 14) * dpr;
  const margin = Number(state.appearance.phase_counter_margin_px ?? 18) * dpr;
  const color = state.appearance.phase_counter_rgb || [230, 232, 236];
  const alpha = Number(state.appearance.phase_counter_alpha ?? 0.48);
  ctx.save();
  ctx.globalAlpha = Math.min(1, Math.max(0, alpha));
  ctx.fillStyle = `rgb(${color.join(",")})`;
  ctx.font = `400 ${fontSize}px "Segoe UI", sans-serif`;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`${progress.current} / ${progress.total}`, els.canvas.width - margin, margin);
  ctx.restore();
}

function drawCursor(cursorX, cursorY) {
  if (usesNativeWindowsCursorMode()) return;
  const cursor = pointToCanvas(cursorX, cursorY);
  const dpr = window.devicePixelRatio || 1;
  const style = state.appearance.cursor_style ?? "arrow";
  const x = cursor[0];
  const y = cursor[1];

  if (style === "aero_arrow" && aeroArrowCursor.complete) {
    const configuredSize = state.appearance.cursor_bitmap_size_px ?? [32, 32];
    const width = Math.max(1, Number(configuredSize[0] ?? 32) * dpr);
    const height = Math.max(1, Number(configuredSize[1] ?? 32) * dpr);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(aeroArrowCursor, x, y, width, height);
    ctx.restore();
    return;
  }

  if (style === "arrow" || style === "aero_arrow") {
    const width = Math.max(6, Number(state.appearance.cursor_arrow_width_px ?? 12) * dpr);
    const height = Math.max(10, Number(state.appearance.cursor_arrow_height_px ?? 18) * dpr);
    const outline = Math.max(1, Number(state.appearance.cursor_arrow_outline_px ?? 1) * dpr);
    // Compact classic Windows cursor. The first point remains the cursor
    // hotspot, independently of the visible polygon.
    const points = [
      [x, y],
      [x, y + height * 0.78],
      [x + width * 0.28, y + height * 0.62],
      [x + width * 0.53, y + height],
      [x + width * 0.75, y + height * 0.91],
      [x + width * 0.48, y + height * 0.58],
      [x + width, y + height * 0.58],
    ];

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
    ctx.closePath();
    ctx.fillStyle = `rgb(${state.appearance.cursor_rgb.join(",")})`;
    ctx.strokeStyle = `rgb(${(state.appearance.cursor_outline_rgb || [32, 32, 32]).join(",")})`;
    ctx.lineWidth = outline;
    ctx.lineJoin = "round";
    const shadow = state.appearance.cursor_shadow_rgb || [0, 0, 0];
    const shadowAlpha = Number(state.appearance.cursor_shadow_alpha ?? 80) / 255;
    const shadowOffset = Number(state.appearance.cursor_shadow_offset_px ?? 1) * dpr;
    ctx.shadowColor = `rgba(${shadow.join(",")},${shadowAlpha})`;
    ctx.shadowOffsetX = shadowOffset;
    ctx.shadowOffsetY = shadowOffset;
    ctx.shadowBlur = 1.25 * dpr;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    return;
  }

  const gap = Number(state.appearance.cursor_crosshair_gap_px ?? 18) * dpr;
  const length = Number(state.appearance.cursor_crosshair_length_px ?? 60) * dpr;
  const thickness = Math.max(1, Number(state.appearance.cursor_line_width_px ?? 8) * dpr);
  const centerSize = Math.max(1, Number(state.appearance.cursor_crosshair_center_px ?? thickness) * dpr);
  const rects = [
    [x - gap - length, y - thickness / 2, length, thickness],
    [x + gap, y - thickness / 2, length, thickness],
    [x - thickness / 2, y - gap - length, thickness, length],
    [x - thickness / 2, y + gap, thickness, length],
    [x - centerSize / 2, y - centerSize / 2, centerSize, centerSize],
  ];
  ctx.fillStyle = `rgb(${state.appearance.cursor_rgb.join(",")})`;
  for (const [rx, ry, rw, rh] of rects) {
    ctx.fillRect(Math.round(rx), Math.round(ry), Math.round(rw), Math.round(rh));
  }
}

function drawInterTrialCursor() {
  clearCanvas();
  if (!els.overlay.classList.contains("hidden")) return;
  drawPhaseTrialCounter();
  if (!state.appearance || !Number.isFinite(state.lastCursorX) || !Number.isFinite(state.lastCursorY)) {
    return;
  }
  drawCursor(state.lastCursorX, state.lastCursorY);
}

function drawTask(feedback = null) {
  const trial = state.current;
  if (!trial) return drawInterTrialCursor();
  clearCanvas();
  const target = pointToCanvas(trial.template.target_x_m, trial.template.target_y_m);
  const geometry = renderedTargetGeometry(trial.template);
  const targetColor = trial.trialStage === "start"
    ? (state.appearance.start_target_rgb || [235, 64, 52])
    : state.appearance.target_rgb;
  ctx.fillStyle = `rgb(${targetColor.join(",")})`;
  ctx.beginPath();
  ctx.arc(target[0], target[1], geometry.radiusPx, 0, Math.PI * 2);
  ctx.fill();
  drawPhaseTrialCounter();
  drawCursor(trial.cursorX, trial.cursorY);
}

function invalidateIfActive(reason) {
  if (state.current?.active && !state.waitingForSave && !state.paused) {
    send({ type: "invalidate_trial", reason });
  }
}

document.addEventListener("pointerlockchange", () => {
  updateCustomCursorVisibility();
  if (document.pointerLockElement !== els.canvas && state.current?.active && !state.paused) {
    requestPause();
  }
});
document.addEventListener("fullscreenchange", () => {
  if (state.current?.active && !state.paused) {
    requestPause();
  }
  resizeCanvas();
  updateCustomCursorVisibility();
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") invalidateIfActive("visibility_lost");
});
window.addEventListener("blur", () => invalidateIfActive("focus_lost"));
window.addEventListener("resize", () => {
  if (state.current?.active) invalidateIfActive("window_resized");
  resizeCanvas();
  updateCustomCursorVisibility();
});

applyFixedDisplayProfile();
updateDisplayCheck();
connect();
