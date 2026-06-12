const $ = (selector) => document.querySelector(selector);

const state = {
  jobs: [],
  selectedJobId: null,
  eventSource: null,
  language: "en",
  config: null,
  storage: null,
  terrain: {
    currentKey: null,
    generation: 0,
    viewers: [],
    roiControllers: [],
    roiObserver: null,
    maxActiveRoiViewers: 8,
    fullViewer: null,
    camera: null,
    verticalScale: 2.4,
  },
};

const metricSpec = [
  ["optimizer_iterations", "metricIterations", (value) => formatNumber(value, 0)],
  ["optimizer_elapsed_seconds", "metricOptimizer", formatSeconds],
  ["artifact_fraction_of_valid", "metricArtifacts", formatPercent],
  ["residual_p99_abs_m", "metricP99Residual", (value) => `${formatNumber(value, 2)} m`],
  ["rmse_before_after_m", "metricRmseDelta", (value) => `${formatNumber(value, 2)} m`],
  ["slope_rmse_before_after_deg", "metricSlopeRmse", (value) => `${formatNumber(value, 2)}°`],
  ["slope_p95_abs_delta_deg", "metricSlopeP95", (value) => `${formatNumber(value, 2)}°`],
  ["slope_correlation_before_after", "metricSlopeCorrelation", (value) => formatNumber(value, 3)],
  ["curvature_variance_ratio", "metricCurvatureRatio", (value) => formatNumber(value, 3)],
  ["pipeline_elapsed_seconds_before_serialization", "metricPipeline", formatSeconds],
  ["rss_peak_observed_bytes", "metricPeakRss", formatBytes],
];

const PARAMETER_PRESETS = window.FTV_PARAMETER_PRESETS || {};

const I18N = {
  en: {
    appTitle: "FTV Calculation Console",
    environmentPreparing: "Preparing runtime",
    jobsLabel: "Jobs",
    language: "Language",
    refresh: "Refresh saved runs",
    runParameters: "Run parameters",
    checkingInput: "Checking input",
    checkingCompute: "Checking compute",
    inputReady: "Input ready",
    inputMissing: "Input missing",
    selectedBackend: "Selected backend",
    cudaStack: "CUDA stack",
    detectingCompute: "Detecting available compute backend",
    computeUnknown: "Compute unknown",
    computeUnavailable: "Compute diagnostics unavailable",
    gpuReady: "GPU ready",
    gpuBlocked: "GPU blocked",
    cpuSelected: "CPU selected",
    cupyMissing: "CuPy missing",
    cudaDevices: "CUDA devices",
    noneVisible: "None visible",
    usable: "usable",
    blocked: "blocked",
    memoryUnknown: "memory unknown",
    free: "free",
    fieldInputNc: "Input NetCDF",
    fieldCrop: "Crop",
    fieldPreset: "Preset",
    fieldAlpha: "Alpha",
    fieldKSize: "K size",
    fieldLambda: "Lambda",
    fieldMaxIter: "Max iter",
    fieldTolerance: "Tolerance",
    fieldMsaThreshold: "MSA threshold",
    fieldBackend: "Backend",
    fieldConvolution: "Convolution",
    fieldInterpolation: "Interpolation",
    fieldPreviewDpi: "Preview DPI",
    fieldWebglSide: "WebGL side",
    fieldVerticalScale: "Vertical scale",
    fieldRoiSamples: "ROI samples",
    fieldPostprocess: "Postprocess",
    fieldSaveSarp: "Save SARP",
    presetCustom: "Custom",
    presetFast: "Fast preview",
    presetBalanced: "Balanced",
    presetConservative: "Conservative",
    presetHighQuality: "High quality",
    presetAggressive: "Aggressive smoothing",
    optAuto: "Auto",
    optCpu: "CPU",
    optGpu: "GPU",
    optDirect: "Direct",
    optFft: "FFT",
    optAfter: "After",
    optBefore: "Before",
    optBlend: "Blend",
    optDifference: "Difference",
    runButton: "Queue or load cached",
    result: "Result",
    noSavedRun: "No saved run selected",
    inline3d: "Inline 3D",
    ready: "Ready",
    savedResultsAppear: "Saved results will appear here",
    cancel: "Cancel",
    retry: "Retry",
    delete: "Delete",
    report: "Report",
    cleanup: "Cleanup",
    storageChecking: "Storage: checking",
    storageLine: "Storage: {size} in {count} result folders",
    cleaningStorage: "Cleaning storage",
    jobCancelled: "Job cancelled",
    jobDeleted: "Job deleted",
    retryingJob: "Retrying job",
    slopeComparison: "Slope comparison",
    slopeComparisonMeta: "Slope maps before and after FTV plus slope delta",
    interactiveTerrain: "Interactive 3D terrain",
    fullTerrainModel: "Full exported terrain model",
    layer: "Layer",
    vertical: "Vertical",
    fit: "Fit",
    waiting3d: "Waiting for 3D result",
    loading3d: "Loading 3D terrain",
    terrainHint: "Drag to orbit | Shift + drag to pan | Wheel to zoom",
    changedAreas: "Most changed areas",
    changedAreasMeta: "Before/after examples ranked by mean absolute correction",
    roiMeta: "{count} areas ranked by mean absolute correction",
    savedRuns: "Saved runs",
    noJobs: "No jobs",
    fullGrid: "full grid",
    crop: "crop",
    mean: "mean",
    segments: "segments",
    segmentsOne: "segment",
    queueingJob: "Queueing job",
    refreshingJobs: "Refreshing jobs",
    streamDisconnected: "Progress stream disconnected",
    waitingForResult: "Waiting for result",
    calculationFailed: "Calculation failed",
    roiHeadArea: "Area",
    roiHead2dBefore: "2D before",
    roiHead2dAfter: "2D after",
    roiHead3dBefore: "3D before",
    roiHead3dAfter: "3D after",
    before3d: "Before 3D",
    after3d: "After 3D",
    load3d: "Load 3D",
    loading: "Loading",
    mesh: "mesh",
    metricIterations: "Iterations",
    metricOptimizer: "Optimizer",
    metricArtifacts: "Artifacts",
    metricP99Residual: "P99 residual",
    metricRmseDelta: "RMSE delta",
    metricSlopeRmse: "Slope RMSE",
    metricSlopeP95: "Slope P95 delta",
    metricSlopeCorrelation: "Slope correlation",
    metricCurvatureRatio: "Curvature ratio",
    metricPipeline: "Pipeline",
    metricPeakRss: "Peak RSS",
  },
  ru: {
    appTitle: "Консоль расчетов FTV",
    environmentPreparing: "Подготовка окружения",
    jobsLabel: "Задачи",
    language: "Язык",
    refresh: "Обновить сохраненные запуски",
    runParameters: "Параметры запуска",
    checkingInput: "Проверка входа",
    checkingCompute: "Проверка вычислений",
    inputReady: "Вход готов",
    inputMissing: "Вход не найден",
    selectedBackend: "Выбранный backend",
    cudaStack: "Стек CUDA",
    detectingCompute: "Определение доступного backend",
    computeUnknown: "Backend неизвестен",
    computeUnavailable: "Диагностика вычислений недоступна",
    gpuReady: "GPU готов",
    gpuBlocked: "GPU недоступен",
    cpuSelected: "Выбран CPU",
    cupyMissing: "CuPy не установлен",
    cudaDevices: "CUDA устройства",
    noneVisible: "Не видны",
    usable: "доступен",
    blocked: "недоступен",
    memoryUnknown: "память неизвестна",
    free: "свободно",
    fieldInputNc: "Входной NetCDF",
    fieldCrop: "Обрезка",
    fieldPreset: "Пресет",
    fieldAlpha: "Alpha",
    fieldKSize: "Размер K",
    fieldLambda: "Lambda",
    fieldMaxIter: "Макс. итераций",
    fieldTolerance: "Допуск",
    fieldMsaThreshold: "Порог MSA",
    fieldBackend: "Backend",
    fieldConvolution: "Свертка",
    fieldInterpolation: "Интерполяция",
    fieldPreviewDpi: "DPI превью",
    fieldWebglSide: "Сторона WebGL",
    fieldVerticalScale: "Вертикальный масштаб",
    fieldRoiSamples: "ROI участки",
    fieldPostprocess: "Постобработка",
    fieldSaveSarp: "Сохранить SARP",
    presetCustom: "Свои значения",
    presetFast: "Быстрое превью",
    presetBalanced: "Сбалансированный",
    presetConservative: "Консервативный",
    presetHighQuality: "Высокое качество",
    presetAggressive: "Сильное сглаживание",
    optAuto: "Авто",
    optCpu: "CPU",
    optGpu: "GPU",
    optDirect: "Прямая",
    optFft: "FFT",
    optAfter: "После",
    optBefore: "До",
    optBlend: "Смешивание",
    optDifference: "Разница",
    runButton: "Поставить в очередь или открыть кеш",
    result: "Результат",
    noSavedRun: "Сохраненный запуск не выбран",
    inline3d: "3D внутри",
    ready: "Готово",
    savedResultsAppear: "Здесь появятся сохраненные результаты",
    cancel: "Отменить",
    retry: "Повторить",
    delete: "Удалить",
    report: "Отчет",
    cleanup: "Очистить",
    storageChecking: "Хранилище: проверка",
    storageLine: "Хранилище: {size} в {count} папках результатов",
    cleaningStorage: "Очистка хранилища",
    jobCancelled: "Задача отменена",
    jobDeleted: "Задача удалена",
    retryingJob: "Повторный запуск",
    slopeComparison: "Сравнение наклона",
    slopeComparisonMeta: "Карты наклона до и после FTV плюс дельта наклона",
    interactiveTerrain: "Интерактивный 3D-рельеф",
    fullTerrainModel: "Полная экспортированная 3D-модель",
    layer: "Слой",
    vertical: "Вертикаль",
    fit: "Вписать",
    waiting3d: "Ожидание 3D результата",
    loading3d: "Загрузка 3D-рельефа",
    terrainHint: "Перетащите: поворот | Shift + перетаскивание: сдвиг | колесо: масштаб",
    changedAreas: "Самые измененные участки",
    changedAreasMeta: "Примеры до/после, ранжированные по средней абсолютной коррекции",
    roiMeta: "{count} участков по средней абсолютной коррекции",
    savedRuns: "Сохраненные запуски",
    noJobs: "Нет задач",
    fullGrid: "вся сетка",
    crop: "обрезка",
    mean: "среднее",
    segments: "сегментов",
    segmentsOne: "сегмент",
    queueingJob: "Постановка задачи в очередь",
    refreshingJobs: "Обновление задач",
    streamDisconnected: "Поток прогресса отключен",
    waitingForResult: "Ожидание результата",
    calculationFailed: "Расчет не выполнен",
    roiHeadArea: "Участок",
    roiHead2dBefore: "2D до",
    roiHead2dAfter: "2D после",
    roiHead3dBefore: "3D до",
    roiHead3dAfter: "3D после",
    before3d: "3D до",
    after3d: "3D после",
    load3d: "Загрузить 3D",
    loading: "Загрузка",
    mesh: "сетка",
    metricIterations: "Итерации",
    metricOptimizer: "Оптимизатор",
    metricArtifacts: "Артефакты",
    metricP99Residual: "P99 невязки",
    metricRmseDelta: "Дельта RMSE",
    metricSlopeRmse: "RMSE наклона",
    metricSlopeP95: "P95 дельты наклона",
    metricSlopeCorrelation: "Корреляция наклона",
    metricCurvatureRatio: "Отн. кривизны",
    metricPipeline: "Pipeline",
    metricPeakRss: "Пик RSS",
  },
};

const HELP_TEXT = {
  en: {
    input_nc: "Source dataset with reprojected_dem and derivative layers.",
    crop: "Optional row and column slice for a smaller run. Empty value processes the full grid.",
    preset: "Fill the form with a named parameter set. Custom keeps the current values.",
    alpha: "Fractional derivative order. Lower values smooth more; higher values preserve sharper relief structures.",
    k_size: "Short-memory window for fractional derivatives. Larger windows use more time and memory.",
    lambda_base: "Base fidelity weight. Lower values smooth more strongly; higher values keep the DEM closer to the input.",
    max_iter: "Maximum optimizer iterations. More iterations can improve convergence but increase runtime.",
    tol: "Early-stop threshold for relative optimizer change. Smaller values demand tighter convergence.",
    msa_threshold: "Slope anomaly threshold for artifact detection. Higher values mark fewer cells as artifacts.",
    backend: "Compute backend. Auto probes GPU first and falls back to CPU; GPU is strict.",
    convolution_method: "Derivative implementation. Direct is best for short windows; FFT can help with large windows.",
    interpolation_iterations: "NaN-fill diffusion passes before optimization. More passes smooth larger missing regions.",
    visualization_dpi: "Resolution for generated PNG previews. Higher values produce larger image files.",
    webgl_max_side: "Maximum side length of the exported 3D mesh. Higher values preserve detail but increase viewer size.",
    vertical_exaggeration: "Vertical exaggeration for the WebGL terrain viewer. It changes visualization only.",
    roi_sample_count: "Number of most-changed areas to export as 2D and 3D before/after examples. Use 0 to disable.",
    postprocess: "Run morphological cleanup after FTV to reduce isolated correction artifacts.",
    save_sarp: "Store the adaptive SARP fidelity field in the output NetCDF for later inspection.",
  },
  ru: {
    input_nc: "Исходный датасет со слоями reprojected_dem и производными.",
    crop: "Необязательная обрезка строк и столбцов для малого запуска. Пустое значение обрабатывает всю сетку.",
    preset: "Заполняет форму именованным набором параметров. Свои значения оставляют текущие поля без изменений.",
    alpha: "Порядок дробной производной. Меньшие значения сильнее сглаживают, большие лучше сохраняют резкие формы рельефа.",
    k_size: "Окно короткой памяти для дробных производных. Большее окно требует больше времени и памяти.",
    lambda_base: "Базовый вес близости к данным. Меньше значение дает сильнее сглаживание, большее держит DEM ближе к входу.",
    max_iter: "Максимальное число итераций оптимизатора. Больше итераций может улучшить сходимость, но увеличивает время.",
    tol: "Порог ранней остановки по относительному изменению. Меньшие значения требуют более строгой сходимости.",
    msa_threshold: "Порог локальной аномалии уклона для поиска артефактов. Большее значение отмечает меньше ячеек.",
    backend: "Вычислительный backend. Авто сначала проверяет GPU и падает на CPU; GPU-режим строгий.",
    convolution_method: "Реализация производных. Прямая быстрее на коротких окнах; FFT может помочь на больших окнах.",
    interpolation_iterations: "Число проходов заполнения NaN перед оптимизацией. Больше проходов сглаживает крупные пропуски.",
    visualization_dpi: "Разрешение PNG-превью. Большее значение создает более крупные файлы.",
    webgl_max_side: "Максимальная сторона экспортируемой 3D-сетки. Больше значение сохраняет детализацию, но утяжеляет viewer.",
    vertical_exaggeration: "Вертикальное преувеличение для WebGL-рельефа. Меняет только отображение.",
    roi_sample_count: "Сколько самых измененных участков экспортировать как 2D и 3D примеры до/после. 0 отключает экспорт.",
    postprocess: "Включает морфологическую очистку после FTV, чтобы уменьшить изолированные артефакты коррекции.",
    save_sarp: "Сохраняет адаптивное поле близости SARP в выходной NetCDF для последующего анализа.",
  },
};

function t(key, replacements = {}) {
  const dictionary = I18N[state.language] || I18N.en;
  let text = dictionary[key] || I18N.en[key] || key;
  Object.entries(replacements).forEach(([name, value]) => {
    text = text.replace(`{${name}}`, String(value));
  });
  return text;
}

function formatNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatSeconds(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "-";
  if (seconds < 90) return `${formatNumber(seconds, 1)} s`;
  return `${formatNumber(seconds / 60, 1)} min`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${formatNumber(number * 100, 2)}%`;
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "-";
  const gib = bytes / 1024 / 1024 / 1024;
  return `${formatNumber(gib, 2)} GiB`;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function setStatus(message, kind = "") {
  const status = $("#statusBox");
  status.textContent = message;
  status.className = `status-box ${kind ? `is-${kind}` : ""}`.trim();
}

function translateStatus(status) {
  if (state.language !== "ru") return status;
  return {
    queued: "в очереди",
    running: "выполняется",
    completed: "готово",
    failed: "ошибка",
    cancelled: "отменено",
  }[status] || status;
}

function translateStage(stage) {
  if (state.language !== "ru" || !stage) return stage;
  const replacements = [
    ["Loaded from saved result", "Загружено из сохраненного результата"],
    ["Preparing calculation", "Подготовка расчета"],
    ["Optimizing FTV", "Оптимизация FTV"],
    ["Rendering slope comparison PNG", "Построение PNG сравнения наклона"],
    ["Exporting WebGL model", "Экспорт WebGL модели"],
    ["Exporting changed-area samples", "Экспорт участков изменений"],
    ["Queued on CPU worker", "В очереди CPU worker"],
    ["Queued on GPU worker", "В очереди GPU worker"],
    ["Completed", "Готово"],
    ["Queued", "В очереди"],
    ["Failed", "Ошибка"],
  ];
  let text = stage;
  replacements.forEach(([source, target]) => {
    text = text.replace(source, target);
  });
  return text;
}

function setProgress(value) {
  const progress = Math.max(0, Math.min(100, Number(value) || 0));
  $("#progressBar").style.width = `${progress}%`;
}

function setText(selector, text) {
  const element = $(selector);
  if (element) element.textContent = text;
}

function setFieldLabel(name, text) {
  const field = $("#runForm")?.elements[name];
  const label = field?.closest("label")?.querySelector(".field-label, .toggle-label");
  if (!label) return;
  const textNode = Array.from(label.childNodes).find((node) => node.nodeType === Node.TEXT_NODE);
  if (textNode) textNode.nodeValue = `\n                ${text}\n                `;
}

function setSelectOptionText(selector, values) {
  const select = $(selector);
  if (!select) return;
  Array.from(select.options).forEach((option) => {
    if (values[option.value]) option.textContent = values[option.value];
  });
}

function applyHelpText() {
  const help = HELP_TEXT[state.language] || HELP_TEXT.en;
  const form = $("#runForm");
  Object.entries(help).forEach(([name, text]) => {
    const field = form.elements[name];
    const tip = field?.closest("label")?.querySelector(".help-tip");
    if (!tip) return;
    tip.title = text;
    tip.setAttribute("aria-label", text);
    tip.dataset.tip = text;
  });
}

function applyLanguage() {
  document.documentElement.lang = state.language;
  $("#languageSelect").value = state.language;
  document.title = t("appTitle");
  setText(".brand h1", t("appTitle"));
  setText(".language-control span", t("language"));
  const refresh = $("#refreshButton");
  refresh.title = t("refresh");
  refresh.setAttribute("aria-label", t("refresh"));
  setText("#controlsTitle", t("runParameters"));
  setText("#inputState", t("checkingInput"));
  setText("#computeState", t("checkingCompute"));
  setText(".compute-summary-grid div:nth-child(1) span", t("selectedBackend"));
  setText(".compute-summary-grid div:nth-child(2) span", t("cudaStack"));
  setText("#computeReason", t("detectingCompute"));
  setText("#runButton span", t("runButton"));
  setText("#resultTitle", t("result"));
  setText("#resultSubtitle", t("noSavedRun"));
  setText("#viewerState span", t("inline3d"));
  setText("#cancelJobButton", t("cancel"));
  setText("#retryJobButton", t("retry"));
  setText("#deleteJobButton", t("delete"));
  setText("#reportLink span", t("report"));
  setText("#statusBox", t("ready"));
  setText("#emptyPreview", t("savedResultsAppear"));
  setText("#slopeSection h3", t("slopeComparison"));
  setText("#slopeMeta", t("slopeComparisonMeta"));
  setText("#terrainSection h3", t("interactiveTerrain"));
  setText("#terrainMeta", t("fullTerrainModel"));
  setText(".terrain-controls label:nth-child(1) span", t("layer"));
  setText(".terrain-controls label:nth-child(2) span", t("vertical"));
  setText("#terrainResetButton", t("fit"));
  setText("#fullTerrainLoading", t("waiting3d"));
  setText(".terrain-hint", t("terrainHint"));
  setText("#roiSection h3", t("changedAreas"));
  setText("#roiMeta", t("changedAreasMeta"));
  setText("#historyTitle", t("savedRuns"));
  setText("#cleanupButton", t("cleanup"));
  setText("#storageState", state.storage ? t("storageLine", {
    size: formatBytes(state.storage.total_size_bytes),
    count: state.storage.result_dir_count,
  }) : t("storageChecking"));

  setFieldLabel("input_nc", t("fieldInputNc"));
  setFieldLabel("crop", t("fieldCrop"));
  setFieldLabel("preset", t("fieldPreset"));
  setFieldLabel("alpha", t("fieldAlpha"));
  setFieldLabel("k_size", t("fieldKSize"));
  setFieldLabel("lambda_base", t("fieldLambda"));
  setFieldLabel("max_iter", t("fieldMaxIter"));
  setFieldLabel("tol", t("fieldTolerance"));
  setFieldLabel("msa_threshold", t("fieldMsaThreshold"));
  setFieldLabel("backend", t("fieldBackend"));
  setFieldLabel("convolution_method", t("fieldConvolution"));
  setFieldLabel("interpolation_iterations", t("fieldInterpolation"));
  setFieldLabel("visualization_dpi", t("fieldPreviewDpi"));
  setFieldLabel("webgl_max_side", t("fieldWebglSide"));
  setFieldLabel("vertical_exaggeration", t("fieldVerticalScale"));
  setFieldLabel("roi_sample_count", t("fieldRoiSamples"));
  setFieldLabel("postprocess", t("fieldPostprocess"));
  setFieldLabel("save_sarp", t("fieldSaveSarp"));

  setSelectOptionText('select[name="backend"]', {
    auto: t("optAuto"),
    cpu: t("optCpu"),
    gpu: t("optGpu"),
  });
  setSelectOptionText('select[name="convolution_method"]', {
    auto: t("optAuto"),
    direct: t("optDirect"),
    fft: t("optFft"),
  });
  setSelectOptionText("#presetSelect", {
    custom: t("presetCustom"),
    fast: t("presetFast"),
    balanced: t("presetBalanced"),
    conservative: t("presetConservative"),
    high_quality: t("presetHighQuality"),
    aggressive: t("presetAggressive"),
  });
  setSelectOptionText("#terrainModeSelect", {
    after: t("optAfter"),
    before: t("optBefore"),
    blend: t("optBlend"),
    difference: t("optDifference"),
  });
  applyHelpText();
  if (state.config) renderConfigState(state.config);
}

function setLanguage(language) {
  state.language = language === "ru" ? "ru" : "en";
  localStorage.setItem("ftvLanguage", state.language);
  applyLanguage();
  renderHistory();
  const selected = state.jobs.find((job) => job.id === state.selectedJobId);
  if (selected) {
    state.terrain.currentKey = null;
    renderJob(selected);
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `Request failed with ${response.status}`);
  }
  return payload;
}

function applyFieldSpecs(fieldSpecs) {
  const form = $("#runForm");
  Object.entries(fieldSpecs || {}).forEach(([name, spec]) => {
    const field = form.elements[name];
    if (!field || field.type !== "number") return;
    [
      ["min", spec.min],
      ["max", spec.max],
      ["step", spec.step],
    ].forEach(([attribute, value]) => {
      if (value === null || value === undefined) {
        field.removeAttribute(attribute);
      } else {
        field.setAttribute(attribute, String(value));
      }
    });
  });
}

function setFormDefaults(defaults) {
  const form = $("#runForm");
  Object.entries(defaults).forEach(([key, value]) => {
    const field = form.elements[key];
    if (!field) return;
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
    } else {
      field.value = value ?? "";
    }
  });
  snapNumberFields(form);
}

function applyPreset(name) {
  const preset = PARAMETER_PRESETS[name];
  if (!preset) return;
  const form = $("#runForm");
  Object.entries(preset).forEach(([key, value]) => {
    const field = form.elements[key];
    if (!field) return;
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
    } else {
      field.value = value ?? "";
    }
  });
  snapNumberFields(form);
  loadCompute(form.elements.backend.value);
}

function decimalPlaces(value) {
  const text = String(value || "");
  if (!text.includes(".")) return 0;
  return text.split(".")[1].replace(/0+$/, "").length;
}

function formatSnappedValue(value, digits) {
  if (!Number.isFinite(value)) return "";
  if (digits <= 0) return String(Math.round(value));
  return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function snapNumberField(field) {
  if (!field || field.type !== "number" || field.value === "") return;
  const parsed = Number(String(field.value).replace(",", "."));
  if (!Number.isFinite(parsed)) return;
  const min = field.min === "" ? -Infinity : Number(field.min);
  const max = field.max === "" ? Infinity : Number(field.max);
  const stepText = field.step || "1";
  let value = Math.min(Math.max(parsed, min), max);
  let digits = 0;
  if (stepText !== "any") {
    const step = Number(stepText);
    if (Number.isFinite(step) && step > 0) {
      const base = Number.isFinite(min) ? min : 0;
      value = base + Math.round((value - base) / step) * step;
      value = Math.min(Math.max(value, min), max);
      digits = Math.max(decimalPlaces(stepText), decimalPlaces(field.min));
    }
  }
  field.value = formatSnappedValue(value, digits);
}

function snapNumberFields(root) {
  root.querySelectorAll('input[type="number"]').forEach((field) => snapNumberField(field));
}

function bindNumberSnapping() {
  document.querySelectorAll('input[type="number"]').forEach((field) => {
    field.addEventListener("blur", () => snapNumberField(field));
    field.addEventListener("change", () => snapNumberField(field));
  });
}

function computeRuntimeLine(gpu) {
  if (!gpu?.cupy_available) return t("cupyMissing");
  const parts = [`CuPy ${gpu.cupy_version || "unknown"}`];
  if (gpu.cuda_runtime_version) parts.push(`CUDA ${gpu.cuda_runtime_version}`);
  if (gpu.cuda_driver_version) parts.push(`driver ${gpu.cuda_driver_version}`);
  return parts.join(" | ");
}

function renderCompute(compute) {
  const statePill = $("#computeState");
  const backend = $("#computeBackend");
  const runtime = $("#computeRuntime");
  const reason = $("#computeReason");
  const devices = $("#computeDevices");
  statePill.className = "state-pill";
  devices.replaceChildren();

  if (!compute) {
    statePill.textContent = t("computeUnknown");
    backend.textContent = "-";
    runtime.textContent = "-";
    reason.textContent = t("computeUnavailable");
    return;
  }

  const selectedGpu = Boolean(compute.selected_gpu);
  const gpu = compute.gpu || {};
  statePill.textContent = selectedGpu
    ? t("gpuReady")
    : compute.requested === "gpu"
      ? t("gpuBlocked")
      : t("cpuSelected");
  statePill.classList.add(selectedGpu || compute.requested !== "gpu" ? "is-ok" : "is-error");
  backend.textContent = compute.selected_backend || "numba-cpu";
  runtime.textContent = computeRuntimeLine(gpu);
  reason.textContent = compute.reason || gpu.reason || "No diagnostics message";

  const gpuDevices = gpu.devices || [];
  if (gpuDevices.length === 0) {
    const empty = document.createElement("div");
    empty.className = "compute-device";
    const label = document.createElement("span");
    label.textContent = t("cudaDevices");
    const value = document.createElement("strong");
    value.textContent = t("noneVisible");
    empty.append(label, value);
    devices.append(empty);
    return;
  }

  gpuDevices.forEach((device) => {
    const card = document.createElement("div");
    card.className = `compute-device ${device.usable ? "is-usable" : "is-blocked"}`;
    const label = document.createElement("span");
    const memory = device.memory_total_bytes
      ? `${formatBytes(device.memory_free_bytes)} ${t("free")} / ${formatBytes(device.memory_total_bytes)}`
      : t("memoryUnknown");
    label.textContent = `cuda:${device.id} | ${device.usable ? t("usable") : t("blocked")}`;
    const value = document.createElement("strong");
    const cc = device.compute_capability ? ` | cc ${device.compute_capability}` : "";
    value.textContent = `${device.name}${cc} | ${memory}`;
    const note = document.createElement("p");
    note.textContent = device.warning || device.reason || "";
    card.append(label, value);
    if (note.textContent) card.append(note);
    devices.append(card);
  });
}

async function loadCompute(requested = "auto") {
  try {
    const compute = await fetchJson(`/api/compute?requested=${encodeURIComponent(requested)}`);
    renderCompute(compute);
  } catch (error) {
    renderCompute({
      requested,
      selected_backend: "numba-cpu",
      selected_gpu: false,
      reason: error.message,
      gpu: { cupy_available: false, reason: error.message, devices: [] },
    });
  }
}

function renderConfigState(payload) {
  renderCompute(payload.compute);
  $("#environmentLabel").textContent = `${t("jobsLabel")}: ${payload.jobs_db}`;
  const inputState = $("#inputState");
  inputState.classList.remove("is-ok", "is-error");
  if (payload.input_exists) {
    inputState.textContent = t("inputReady");
    inputState.classList.add("is-ok");
  } else {
    inputState.textContent = t("inputMissing");
    inputState.classList.add("is-error");
  }
}

function renderStorage(payload) {
  state.storage = payload;
  $("#storageState").textContent = t("storageLine", {
    size: formatBytes(payload?.total_size_bytes),
    count: payload?.result_dir_count ?? 0,
  });
}

async function loadStorage() {
  try {
    renderStorage(await fetchJson("/api/storage"));
  } catch (error) {
    $("#storageState").textContent = error.message;
  }
}

function readForm() {
  const form = $("#runForm");
  const payload = {};
  Array.from(form.elements).forEach((field) => {
    if (!field.name) return;
    if (field.type === "checkbox") {
      payload[field.name] = field.checked;
    } else {
      payload[field.name] = field.value;
    }
  });
  return payload;
}

function parameterLine(item) {
  const params = item.parameters || {};
  const crop = params.crop ? `${t("crop")} ${params.crop}` : t("fullGrid");
  return `alpha ${params.alpha} | lambda ${params.lambda_base} | k ${params.k_size} | ${crop}`;
}

function upsertJob(job) {
  const index = state.jobs.findIndex((item) => item.id === job.id);
  if (index >= 0) {
    state.jobs[index] = job;
  } else {
    state.jobs.unshift(job);
  }
}

function statusKind(job) {
  if (job.status === "completed") return "ok";
  if (job.status === "failed" || job.status === "cancelled") return "error";
  if (job.status === "queued" || job.status === "running") return "running";
  return "";
}

function renderHistory() {
  const list = $("#historyList");
  list.replaceChildren();
  $("#runCount").textContent = String(state.jobs.length);
  if (state.jobs.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-preview";
    empty.textContent = t("noJobs");
    list.append(empty);
    return;
  }

  state.jobs.forEach((job) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item ${job.id === state.selectedJobId ? "is-selected" : ""}`.trim();
    button.addEventListener("click", () => selectJob(job));

    const header = document.createElement("header");
    const title = document.createElement("b");
    title.textContent = job.cache_key;
    const time = document.createElement("time");
    time.textContent = `${translateStatus(job.status)} | ${formatDate(job.updated_at)}`;
    header.append(title, time);

    const params = document.createElement("p");
    params.textContent = `${formatNumber(job.progress_percent, 0)}% | ${parameterLine(job)}`;
    button.append(header, params);
    list.append(button);
  });
}

function renderMetrics(result) {
  const grid = $("#metricsGrid");
  grid.replaceChildren();
  const metrics = result?.metrics || {};
  metricSpec.forEach(([key, labelKey, formatter]) => {
    if (!(key in metrics)) return;
    const card = document.createElement("div");
    card.className = "metric";
    const name = document.createElement("span");
    name.textContent = t(labelKey);
    const value = document.createElement("strong");
    value.textContent = formatter(metrics[key]);
    card.append(name, value);
    grid.append(card);
  });
}

function renderJobActions(job) {
  const cancel = $("#cancelJobButton");
  const retry = $("#retryJobButton");
  const deleteButton = $("#deleteJobButton");
  cancel.hidden = !job?.can_cancel;
  retry.hidden = !job?.can_retry;
  deleteButton.hidden = !job?.terminal;
  cancel.disabled = !job?.can_cancel;
  retry.disabled = !job?.can_retry;
  deleteButton.disabled = !job?.terminal;
}

function disposeTerrainViewers() {
  state.terrain.viewers.forEach((viewer) => viewer.dispose());
  state.terrain.viewers = [];
  state.terrain.roiControllers = [];
  if (state.terrain.roiObserver) {
    state.terrain.roiObserver.disconnect();
    state.terrain.roiObserver = null;
  }
  state.terrain.fullViewer = null;
  state.terrain.camera = null;
}

function clearTerrainResult() {
  state.terrain.generation += 1;
  state.terrain.currentKey = null;
  disposeTerrainViewers();
  $("#terrainSection").hidden = true;
  $("#roiSection").hidden = true;
  $("#roiTable").replaceChildren();
  $("#viewerState").classList.add("is-disabled");
}

function syncTerrainCamera(source) {
  const camera = source.getCamera();
  state.terrain.camera = camera;
  state.terrain.viewers.forEach((viewer) => {
    if (viewer !== source) viewer.setCamera(camera, true);
  });
}

function setSharedVerticalScale(value) {
  const scale = Math.max(0.1, Number(value) || 2.4);
  state.terrain.verticalScale = scale;
  $("#terrainVerticalSlider").value = String(Math.round(scale * 100));
  $("#terrainVerticalInput").value = formatSnappedValue(scale, 1);
  $("#terrainVerticalValue").textContent = `${scale.toFixed(2)}x`;
  state.terrain.viewers.forEach((viewer) => viewer.setVerticalScale(scale));
}

function createTerrainViewer(canvas, modelUrl, mode, options = {}) {
  if (!window.FTVTerrainViewer) {
    throw new Error("Embedded terrain viewer is not available");
  }
  canvas.tabIndex = 0;
  const viewer = new window.FTVTerrainViewer(canvas, {
    mode,
    verticalScale: state.terrain.verticalScale,
    distance: options.distance,
    interactive: true,
    onCameraChange: syncTerrainCamera,
  });
  if (state.terrain.camera) viewer.setCamera(state.terrain.camera, true);
  return viewer.load(modelUrl).then(() => {
    state.terrain.viewers.push(viewer);
    return viewer;
  });
}

function addRoiHeaders(table) {
  [
    t("roiHeadArea"),
    t("roiHead2dBefore"),
    t("roiHead2dAfter"),
    t("roiHead3dBefore"),
    t("roiHead3dAfter"),
  ].forEach((label) => {
    const header = document.createElement("div");
    header.className = "roi-head";
    header.textContent = label;
    table.append(header);
  });
}

function appendRoiImageCell(table, url, label, jobId) {
  const cell = document.createElement("div");
  cell.className = "roi-cell";
  const image = document.createElement("img");
  image.alt = label;
  image.src = `${url}?job=${jobId}`;
  cell.append(image);
  table.append(cell);
}

function disposeRoiController(controller) {
  if (!controller.viewer) return;
  controller.viewer.dispose();
  state.terrain.viewers = state.terrain.viewers.filter((viewer) => viewer !== controller.viewer);
  controller.viewer = null;
  controller.loaded = false;
  controller.loading = false;
  controller.frame.classList.add("is-pending");
  controller.canvas.replaceWith(controller.canvas.cloneNode(false));
  controller.canvas = controller.frame.querySelector("canvas");
  controller.canvas.setAttribute("aria-label", controller.ariaLabel);
  controller.button.hidden = false;
  controller.button.disabled = false;
  controller.button.textContent = t("load3d");
  controller.badge.textContent = controller.mode === "before" ? t("before3d") : t("after3d");
}

function enforceRoiContextLimit(exceptController) {
  const active = state.terrain.roiControllers
    .filter((controller) => controller.viewer && controller !== exceptController)
    .sort((a, b) => a.lastUsed - b.lastUsed);
  while (active.length >= state.terrain.maxActiveRoiViewers) {
    disposeRoiController(active.shift());
  }
}

function loadRoiController(controller) {
  if (controller.viewer || controller.loading || controller.generation !== state.terrain.generation) {
    return;
  }
  enforceRoiContextLimit(controller);
  controller.loading = true;
  controller.button.disabled = true;
  controller.button.textContent = t("loading");
  controller.badge.textContent = t("loading");
  createTerrainViewer(controller.canvas, controller.modelUrl, controller.mode, { distance: 2.0 })
    .then((viewer) => {
      if (controller.generation !== state.terrain.generation) {
        viewer.dispose();
        return;
      }
      controller.viewer = viewer;
      controller.loaded = true;
      controller.loading = false;
      controller.lastUsed = performance.now();
      controller.frame.classList.remove("is-pending");
      controller.button.hidden = true;
      controller.badge.textContent = controller.mode === "before" ? t("before3d") : t("after3d");
    })
    .catch((error) => {
      if (controller.generation !== state.terrain.generation) return;
      controller.loading = false;
      controller.button.disabled = false;
      controller.button.textContent = t("load3d");
      controller.badge.textContent = error.message;
    });
}

function appendRoiTerrainCell(table, sample, mode, generation) {
  const cell = document.createElement("div");
  cell.className = "roi-cell";
  const frame = document.createElement("div");
  frame.className = "roi-terrain is-pending";
  const canvas = document.createElement("canvas");
  const ariaLabel = `${sample.id} ${mode} 3D terrain`;
  canvas.setAttribute("aria-label", ariaLabel);
  const badge = document.createElement("span");
  badge.className = "roi-rank";
  badge.textContent = mode === "before" ? t("before3d") : t("after3d");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "roi-load-button";
  button.textContent = t("load3d");
  frame.append(canvas, button, badge);
  cell.append(frame);
  table.append(cell);

  const controller = {
    frame,
    canvas,
    button,
    badge,
    mode,
    modelUrl: sample.urls.webgl_model,
    generation,
    viewer: null,
    loading: false,
    loaded: false,
    lastUsed: 0,
    ariaLabel,
  };
  button.addEventListener("click", () => loadRoiController(controller));
  frame.addEventListener("pointerenter", () => {
    if (controller.viewer) controller.lastUsed = performance.now();
  });
  state.terrain.roiControllers.push(controller);
  if (!state.terrain.roiObserver && "IntersectionObserver" in window) {
    state.terrain.roiObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) loadRoiController(entry.target.__roiController);
      });
    }, { root: $("#roiSection"), rootMargin: "120px" });
  }
  frame.__roiController = controller;
  state.terrain.roiObserver?.observe(frame);
}

function renderRoiSamples(result, jobId, generation) {
  const section = $("#roiSection");
  const table = $("#roiTable");
  table.replaceChildren();
  const samples = (result?.roi_samples?.samples || []).filter(
    (sample) => sample?.urls?.before_png && sample?.urls?.after_png && sample?.urls?.webgl_model,
  );
  if (samples.length === 0) {
    section.hidden = true;
    return;
  }

  addRoiHeaders(table);
  samples.forEach((sample) => {
    const rank = document.createElement("div");
    rank.className = "roi-rank-cell";
    const title = document.createElement("strong");
    title.textContent = `#${sample.rank}`;
    const change = document.createElement("span");
    change.textContent = `${formatNumber(sample.mean_abs_change_m, 2)} m ${t("mean")}`;
    const segments = document.createElement("span");
    segments.className = "roi-segments";
    const segmentCount = Number(sample.change_segment_count || 0);
    segments.textContent = `${segmentCount} ${segmentCount === 1 ? t("segmentsOne") : t("segments")}`;
    const location = document.createElement("span");
    location.textContent = `r ${sample.row_start}:${sample.row_stop} | c ${sample.col_start}:${sample.col_stop}`;
    rank.append(title, change, segments, location);
    table.append(rank);

    appendRoiImageCell(table, sample.urls.before_png, `${sample.id} before FTV`, jobId);
    appendRoiImageCell(table, sample.urls.after_png, `${sample.id} after FTV`, jobId);
    appendRoiTerrainCell(table, sample, "before", generation);
    appendRoiTerrainCell(table, sample, "after", generation);
  });
  $("#roiMeta").textContent = t("roiMeta", { count: samples.length });
  section.hidden = false;
}

function renderTerrainResult(job) {
  const result = job.result;
  const key = result?.cache_key || null;
  if (!key || !result?.urls?.webgl_model) {
    clearTerrainResult();
    return;
  }
  if (state.terrain.currentKey === key) return;

  clearTerrainResult();
  state.terrain.currentKey = key;
  const generation = state.terrain.generation;
  const defaultScale = Number(
    job.parameters?.vertical_exaggeration
    || result.webgl?.vertical_exaggeration_default
    || 2.4,
  );
  setSharedVerticalScale(defaultScale);
  $("#terrainSection").hidden = false;
  $("#viewerState").classList.remove("is-disabled");
  $("#fullTerrainLoading").hidden = false;
  $("#fullTerrainLoading").textContent = t("loading3d");
  $("#terrainMeta").textContent = result.webgl?.scene || t("fullTerrainModel");

  createTerrainViewer($("#fullTerrainCanvas"), result.urls.webgl_model, $("#terrainModeSelect").value)
    .then((viewer) => {
      if (generation !== state.terrain.generation) {
        viewer.dispose();
        return;
      }
      state.terrain.fullViewer = viewer;
      $("#fullTerrainLoading").hidden = true;
      $("#terrainMeta").textContent = `${viewer.model.grid_shape[0]} x ${viewer.model.grid_shape[1]} ${t("mesh")}`;
    })
    .catch((error) => {
      if (generation !== state.terrain.generation) return;
      $("#fullTerrainLoading").hidden = false;
      $("#fullTerrainLoading").textContent = error.message;
    });

  renderRoiSamples(result, job.id, generation);
}

function renderSlopeResult(result, jobId) {
  const section = $("#slopeSection");
  const image = $("#slopeImage");
  if (result?.urls?.slope_comparison_png) {
    section.hidden = false;
    image.src = `${result.urls.slope_comparison_png}?job=${jobId}`;
  } else {
    section.hidden = true;
    image.removeAttribute("src");
  }
}

function renderReportLink(result) {
  const link = $("#reportLink");
  if (result?.urls?.validation_report_md) {
    link.href = result.urls.validation_report_md;
    link.hidden = false;
    link.classList.remove("is-disabled");
  } else {
    link.href = "#";
    link.hidden = true;
    link.classList.add("is-disabled");
  }
}

function renderJob(job) {
  state.selectedJobId = job.id;
  $("#resultSubtitle").textContent = parameterLine(job);
  setProgress(job.progress_percent);
  setStatus(`${translateStage(job.stage)} (${formatNumber(job.progress_percent, 0)}%)`, statusKind(job));
  renderJobActions(job);

  const result = job.result;
  const image = $("#comparisonImage");
  const empty = $("#emptyPreview");

  if (result?.urls?.comparison_png) {
    image.style.display = "block";
    image.src = `${result.urls.comparison_png}?job=${job.id}`;
    empty.style.display = "none";
  } else {
    image.style.display = "none";
    image.removeAttribute("src");
    empty.style.display = "block";
    empty.textContent = job.status === "failed" ? job.error || t("calculationFailed") : t("waitingForResult");
  }

  renderTerrainResult(job);
  renderSlopeResult(result, job.id);
  renderReportLink(result);
  renderMetrics(result);
  renderHistory();
}

function clearResultPanel() {
  state.selectedJobId = null;
  $("#resultSubtitle").textContent = t("noSavedRun");
  setProgress(0);
  setStatus(t("ready"));
  renderJobActions(null);
  $("#comparisonImage").style.display = "none";
  $("#comparisonImage").removeAttribute("src");
  $("#emptyPreview").style.display = "block";
  $("#emptyPreview").textContent = t("savedResultsAppear");
  $("#slopeSection").hidden = true;
  $("#slopeImage").removeAttribute("src");
  renderReportLink(null);
  clearTerrainResult();
  renderMetrics(null);
  renderHistory();
}

function selectJob(job) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  renderJob(job);
  if (!job.terminal) {
    subscribeToJob(job.id);
  }
}

function subscribeToJob(jobId) {
  if (state.eventSource) state.eventSource.close();
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = source;
  source.onmessage = (event) => {
    const job = JSON.parse(event.data);
    upsertJob(job);
    renderJob(job);
    if (job.terminal) {
      source.close();
      state.eventSource = null;
      loadJobs(job.id);
    }
  };
  source.onerror = () => {
    source.close();
    state.eventSource = null;
    setStatus(t("streamDisconnected"), "error");
  };
}

async function loadJobs(preferredJobId = null) {
  const payload = await fetchJson("/api/jobs");
  state.jobs = payload.jobs || [];
  renderHistory();
  const selected = state.jobs.find((item) => item.id === preferredJobId)
    || state.jobs.find((item) => !item.terminal)
    || state.jobs[0];
  if (selected) {
    selectJob(selected);
  } else {
    clearResultPanel();
  }
}

async function loadConfig() {
  const payload = await fetchJson("/api/config");
  state.config = payload;
  applyFieldSpecs(payload.field_specs || {});
  setFormDefaults(payload.defaults || {});
  renderConfigState(payload);
}

async function submitRun(event) {
  event.preventDefault();
  const button = $("#runButton");
  snapNumberFields($("#runForm"));
  $("#presetSelect").value = "custom";
  button.disabled = true;
  setStatus(t("queueingJob"), "running");
  setProgress(0);
  try {
    const job = await fetchJson("/api/jobs", {
      method: "POST",
      body: JSON.stringify(readForm()),
    });
    setFormDefaults(job.parameters || readForm());
    upsertJob(job);
    selectJob(job);
    loadStorage();
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function refresh() {
  setStatus(t("refreshingJobs"));
  await loadJobs(state.selectedJobId);
  await loadStorage();
}

async function cancelSelectedJob() {
  if (!state.selectedJobId) return;
  const job = await fetchJson(`/api/jobs/${state.selectedJobId}/cancel`, { method: "POST" });
  upsertJob(job);
  selectJob(job);
  setStatus(t("jobCancelled"), "error");
}

async function retrySelectedJob() {
  if (!state.selectedJobId) return;
  setStatus(t("retryingJob"), "running");
  const job = await fetchJson(`/api/jobs/${state.selectedJobId}/retry`, { method: "POST" });
  upsertJob(job);
  selectJob(job);
}

async function deleteSelectedJob() {
  if (!state.selectedJobId) return;
  const deletedJobId = state.selectedJobId;
  await fetchJson(`/api/jobs/${deletedJobId}`, { method: "DELETE" });
  state.jobs = state.jobs.filter((job) => job.id !== deletedJobId);
  state.selectedJobId = null;
  setStatus(t("jobDeleted"), "ok");
  await loadJobs();
  await loadStorage();
}

async function cleanupStorage() {
  const button = $("#cleanupButton");
  button.disabled = true;
  setStatus(t("cleaningStorage"), "running");
  try {
    const payload = await fetchJson("/api/cleanup", { method: "POST" });
    renderStorage(payload.storage);
    await loadJobs(state.selectedJobId);
  } finally {
    button.disabled = false;
  }
}

async function init() {
  const savedLanguage = localStorage.getItem("ftvLanguage");
  state.language = savedLanguage || (navigator.language?.toLowerCase().startsWith("ru") ? "ru" : "en");
  applyLanguage();
  bindNumberSnapping();
  $("#runForm").addEventListener("submit", submitRun);
  $("#refreshButton").addEventListener("click", refresh);
  $("#cleanupButton").addEventListener("click", () => {
    cleanupStorage().catch((error) => setStatus(error.message, "error"));
  });
  $("#cancelJobButton").addEventListener("click", () => {
    cancelSelectedJob().catch((error) => setStatus(error.message, "error"));
  });
  $("#retryJobButton").addEventListener("click", () => {
    retrySelectedJob().catch((error) => setStatus(error.message, "error"));
  });
  $("#deleteJobButton").addEventListener("click", () => {
    deleteSelectedJob().catch((error) => setStatus(error.message, "error"));
  });
  $("#languageSelect").addEventListener("change", (event) => {
    setLanguage(event.target.value);
  });
  $("#presetSelect").addEventListener("change", (event) => {
    applyPreset(event.target.value);
  });
  $("#runForm").elements.backend.addEventListener("change", (event) => {
    loadCompute(event.target.value);
  });
  $("#terrainModeSelect").addEventListener("change", (event) => {
    if (state.terrain.fullViewer) state.terrain.fullViewer.setMode(event.target.value);
  });
  $("#terrainVerticalSlider").addEventListener("input", (event) => {
    setSharedVerticalScale(Number(event.target.value) / 100);
  });
  $("#terrainVerticalInput").addEventListener("input", (event) => {
    setSharedVerticalScale(event.target.value);
  });
  $("#terrainVerticalInput").addEventListener("blur", (event) => {
    snapNumberField(event.target);
    setSharedVerticalScale(event.target.value);
  });
  $("#terrainResetButton").addEventListener("click", () => {
    if (!state.terrain.fullViewer) return;
    state.terrain.fullViewer.resetView();
  });
  await loadConfig();
  await loadStorage();
  await loadJobs();
  if (state.jobs.length === 0) {
    setStatus(t("ready"));
    setProgress(0);
  }
}

init().catch((error) => {
  setStatus(error.message, "error");
});
