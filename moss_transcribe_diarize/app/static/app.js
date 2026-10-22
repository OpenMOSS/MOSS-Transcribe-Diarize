import {
  getLocale,
  initI18n,
  localizedError,
  setLocale,
  t,
  tp,
} from './i18n.js';

const RUNNING_STATES = new Set(['queued', 'loading_model', 'transcribing', 'postprocessing', 'rendering']);
const EDIT_STATES = new Set(['waiting_review', 'done']);
const TERMINAL_STATES = new Set(['waiting_review', 'done', 'failed', 'cancelled']);
const fileInput = document.querySelector('#file');
const importTitleEl = document.querySelector('#importTitle');
const rerunSourceEl = document.querySelector('#rerunSource');
const promptInput = document.querySelector('#prompt');
const advancedDetails = document.querySelector('.advanced');
const maxNewTokensInput = document.querySelector('#maxNewTokens');
const maxLenInput = document.querySelector('#maxLen');
const decodingSelect = document.querySelector('#decoding');
const temperatureInput = document.querySelector('#temperature');
const uploadBtn = document.querySelector('#upload');
const newTaskBtn = document.querySelector('#newTask');
const refreshJobsBtn = document.querySelector('#refreshJobs');
const sidebarToggleBtn = document.querySelector('#sidebarToggle');
const deleteCurrentBtn = document.querySelector('#deleteCurrent');
const openNewBtn = document.querySelector('#openNew');
const saveBtn = document.querySelector('#save');
const renderBtn = document.querySelector('#render');
const rerunBtn = document.querySelector('#rerun');
const saveStatusEl = document.querySelector('#saveStatus');
const importView = document.querySelector('#importView');
const processingView = document.querySelector('#processingView');
const workbench = document.querySelector('#workbench');
const runtimeEl = document.querySelector('#runtime');
const jobListEl = document.querySelector('#jobList');
const jobCountEl = document.querySelector('#jobCount');
const importErrorEl = document.querySelector('#importError');
const processTitleEl = document.querySelector('#processTitle');
const processNameEl = document.querySelector('#processName');
const processMetaEl = document.querySelector('#processMeta');
const processBarEl = document.querySelector('#processBar');
const processErrorEl = document.querySelector('#processError');
const selectedNameEl = document.querySelector('#selectedName');
const taskStatusEl = document.querySelector('#taskStatus');
const taskUsageEl = document.querySelector('#taskUsage');
const taskParamsEl = document.querySelector('#taskParams');
const taskNoticeEl = document.querySelector('#taskNotice');
const modelInfoEl = document.querySelector('#modelinfo');
const tbody = document.querySelector('#segments');
const speakerMapEl = document.querySelector('#speakerMap');
const videoStage = document.querySelector('#videoStage');
const videoShell = document.querySelector('.video-shell');
const preview = document.querySelector('#preview');
const subtitleOverlay = document.querySelector('#subtitleOverlay');
const downloads = document.querySelector('#downloads');
const localeSelect = document.querySelector('#localeSelect');
let jobs = [];
let currentJob = null;
let rerunDraftJob = null;
let pollTimer = null;
let ffmpegAvailable = false;
let activeSegmentIndex = -1;
let activeSegmentIndexesKey = null;
let assPlayRes = { x: 1920, y: 1080 };
let layoutFitFrame = 0;
let editorDirty = false;
let saveStatusTimer = 0;
let speakerNameMap = {};
let previewSyncRequest = null;
let previewSyncRequestKind = null;
let runtimeState = 'checking';
let currentSaveState = 'saved';
let currentSaveMessageKey = 'save.saved';
let currentSaveMessageParams = {};
let importErrorDescriptor = null;
let taskNoticeDescriptor = null;
const assFontLineHeightFactor = 1.448;
const speakerPalette = ['#ffffff', '#ffe75b', '#8ff286', '#ffa7bb', '#ffd700', '#6bb5ff', '#db8eff', '#d8d8d8'];

function apiUrl(path) {
  const clean = String(path).replace(/^\/+/, '');
  const basePath = window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/';
  return new URL(clean, window.location.origin + basePath).toString();
}

async function refreshRuntime() {
  try {
    const res = await fetch(apiUrl('api/runtime'), { cache: 'no-store' });
    if (!res.ok) throw new Error('runtime status ' + res.status);
    const data = await res.json();
    ffmpegAvailable = !!(data.ffmpeg && data.ffmpeg.available);
    runtimeState = ffmpegAvailable ? 'available' : 'unavailable';
    renderRuntimeStatus();
    if (importErrorDescriptor && importErrorDescriptor.fallbackKey === 'runtime.connectionHelp') clearImportError();
    renderBtn.disabled = !ffmpegAvailable;
    applyInferenceDefaults(data.inference || {});
    renderModelInfo(data.model || {});
  } catch (err) {
    ffmpegAvailable = false;
    runtimeState = 'apiUnavailable';
    renderRuntimeStatus();
    renderBtn.disabled = true;
    setImportError(null, 'runtime.connectionHelp');
  }
}

function renderRuntimeStatus() {
  const key = `runtime.${runtimeState}`;
  runtimeEl.textContent = t(key);
  runtimeEl.className = 'pill ' + (runtimeState === 'available' ? 'ok' : runtimeState === 'checking' ? '' : 'bad');
}

function setImportError(data, fallbackKey) {
  importErrorDescriptor = { data, fallbackKey };
  renderImportError();
}

function clearImportError() {
  importErrorDescriptor = null;
  importErrorEl.textContent = '';
}

function renderImportError() {
  if (!importErrorDescriptor) return;
  importErrorEl.textContent = localizedError(importErrorDescriptor.data, importErrorDescriptor.fallbackKey);
}

function applyInferenceDefaults(defaults) {
  if (!promptInput.value && defaults.prompt) promptInput.value = defaults.prompt;
  if (defaults.max_new_tokens) maxNewTokensInput.value = defaults.max_new_tokens;
  if (defaults.max_length) maxLenInput.value = defaults.max_length;
  if (defaults.decoding) decodingSelect.value = defaults.decoding;
  if (defaults.temperature) temperatureInput.value = defaults.temperature;
  updateDecodingControls();
}

function renderModelInfo(model) {
  const parts = [];
  if (model.path) {
    const pathParts = String(model.path).split('/');
    parts.push(pathParts.slice(-2).join('/'));
  }
  if (model.device) parts.push(model.device);
  if (model.dtype) parts.push(model.dtype);
  const processor = model.processor || {};
  if (processor.time_marker_every_seconds) parts.push('time marker ' + processor.time_marker_every_seconds + 's');
  modelInfoEl.textContent = parts.join(' · ');
}

function updateDecodingControls() {
  temperatureInput.disabled = decodingSelect.value !== 'sample';
}

function scheduleLayoutFit() {
  if (layoutFitFrame) cancelAnimationFrame(layoutFitFrame);
  layoutFitFrame = requestAnimationFrame(() => {
    layoutFitFrame = 0;
    fitVideoStageToMedia();
  });
}

function setSidebarCollapsed(collapsed, persist = true) {
  document.body.classList.toggle('sidebar-collapsed', collapsed);
  const label = t(collapsed ? 'sidebar.expand' : 'sidebar.collapse');
  sidebarToggleBtn.setAttribute('aria-label', label);
  sidebarToggleBtn.title = label;
  if (persist) {
    try {
      localStorage.setItem('mtdSidebarCollapsed', collapsed ? '1' : '0');
    } catch (err) {}
  }
  scheduleLayoutFit();
}

function restoreSidebarState() {
  try {
    setSidebarCollapsed(localStorage.getItem('mtdSidebarCollapsed') === '1', false);
  } catch (err) {
    setSidebarCollapsed(false);
  }
}

function setSaveState(state, messageKey, messageParams = {}) {
  if (saveStatusTimer) {
    clearTimeout(saveStatusTimer);
    saveStatusTimer = 0;
  }
  currentSaveState = state;
  currentSaveMessageKey = messageKey;
  currentSaveMessageParams = messageParams;
  saveStatusEl.className = 'save-status ' + state;
  saveStatusEl.textContent = t(messageKey, messageParams);
  const showButton = state === 'dirty' || state === 'saving' || state === 'error';
  saveBtn.classList.toggle('is-hidden', !showButton);
  saveBtn.classList.toggle('primary', showButton);
  saveBtn.classList.toggle('saved', false);
  saveBtn.disabled = state === 'saving' || !currentJob;
  if (state === 'saving') saveBtn.textContent = t('actions.saving');
  else if (state === 'error') saveBtn.textContent = t('actions.retrySave');
  else saveBtn.textContent = t('actions.saveChanges');
}

function setEditorDirty(dirty) {
  editorDirty = dirty;
  if (dirty) setSaveState('dirty', 'save.dirty');
  else setSaveState('saved', 'save.saved');
}

function markEditorDirty() {
  if (!currentJob) return;
  setEditorDirty(true);
}

decodingSelect.addEventListener('change', updateDecodingControls);
localeSelect.addEventListener('change', async () => {
  localeSelect.disabled = true;
  try {
    await setLocale(localeSelect.value);
    refreshLocalizedUI();
  } finally {
    localeSelect.disabled = false;
  }
});

newTaskBtn.addEventListener('click', () => showImportView({ clearDraft: true }));
openNewBtn.addEventListener('click', () => showImportView({ clearDraft: true }));
refreshJobsBtn.addEventListener('click', () => refreshJobs());
sidebarToggleBtn.addEventListener('click', () => {
  setSidebarCollapsed(!document.body.classList.contains('sidebar-collapsed'));
});
deleteCurrentBtn.addEventListener('click', async () => {
  if (currentJob) await deleteJob(currentJob.id);
});

jobListEl.addEventListener('click', async (event) => {
  const deleteButton = event.target.closest('[data-delete-id]');
  if (deleteButton) {
    event.stopPropagation();
    await deleteJob(deleteButton.dataset.deleteId);
    return;
  }
  const item = event.target.closest('[data-job-id]');
  if (item) await selectJob(item.dataset.jobId);
});

fileInput.addEventListener('change', () => {
  if (rerunDraftJob) resetImportMode();
  const file = fileInput.files[0];
  if (file) {
    stopPreviewSync();
    preview.src = URL.createObjectURL(file);
    resetVideoStage();
  }
});

uploadBtn.addEventListener('click', async () => {
  if (rerunDraftJob) {
    await startRerunDraft();
    return;
  }
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  form.append('prompt', promptInput.value);
  if (maxNewTokensInput.value) form.append('max_new_tokens', maxNewTokensInput.value);
  if (maxLenInput.value) form.append('max_len', maxLenInput.value);
  form.append('decoding', decodingSelect.value);
  if (temperatureInput.value) form.append('temperature', temperatureInput.value);
  uploadBtn.disabled = true;
  advancedDetails.open = false;
  clearImportError();
  showProcessingPlaceholder(file.name);
  const res = await fetch(apiUrl('api/jobs'), { method: 'POST', body: form });
  const job = await res.json();
  uploadBtn.disabled = false;
  if (!res.ok) {
    setImportError(job, 'errors.uploadFailed');
    showImportView({ preserveError: true });
    return;
  }
  currentJob = job;
  await refreshJobs({ keepSelection: true });
  await selectJob(job.id);
});

saveBtn.addEventListener('click', async () => {
  await saveSegments();
});

renderBtn.addEventListener('click', async () => {
  if (!currentJob || !ffmpegAvailable) return;
  const saved = await saveSegments();
  if (!saved) return;
  const style = collectSubtitleStyle();
  const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/render`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ style })
  });
  const data = await res.json();
  if (!res.ok) setLocalizedTaskNotice(data, 'errors.burnFailed');
  else {
    currentJob = data;
    renderCurrentJob(data);
    await refreshJobs({ keepSelection: true });
  }
});

rerunBtn.addEventListener('click', () => {
  if (currentJob) showRerunDraft(currentJob);
});

preview.addEventListener('timeupdate', syncActiveSegment);
preview.addEventListener('seeked', syncActiveSegment);
preview.addEventListener('play', schedulePreviewSync);
preview.addEventListener('pause', () => {
  stopPreviewSync();
  syncActiveSegment();
});
preview.addEventListener('ended', () => {
  stopPreviewSync();
  syncActiveSegment();
});
preview.addEventListener('loadedmetadata', () => {
  fitVideoStageToMedia();
  syncActiveSegment();
});
window.addEventListener('resize', () => {
  scheduleLayoutFit();
});
if ('ResizeObserver' in window) {
  const layoutObserver = new ResizeObserver(scheduleLayoutFit);
  for (const element of [videoShell, document.querySelector('.content'), document.querySelector('.editor-grid')]) {
    if (element) layoutObserver.observe(element);
  }
}
tbody.addEventListener('input', (event) => {
  markEditorDirty();
  if (event.target.classList.contains('text')) {
    const tr = event.target.closest('tr');
    resizeSegmentTextarea(event.target, tr && tr.classList.contains('active'));
  }
  if (event.target.classList.contains('start') || event.target.classList.contains('end')) {
    syncActiveSegment(undefined, true);
  }
  else {
    if (event.target.classList.contains('speaker')) renderSpeakerMap(collectSegments());
    const tr = event.target.closest('tr');
    const rowIndex = tr ? Number(tr.dataset.index) : -1;
    const segments = collectSegments();
    updateSubtitlePreview(segments, rowIndex === activeSegmentIndex ? [rowIndex] : undefined);
  }
});
tbody.addEventListener('change', markEditorDirty);
speakerMapEl.addEventListener('input', () => {
  syncSpeakerNameInputs();
  markEditorDirty();
  updateSubtitlePreview();
});
tbody.addEventListener('focusin', (event) => {
  const tr = event.target.closest('tr');
  if (!tr) return;
  const rowIndex = Number(tr.dataset.index);
  setActiveSegment(rowIndex, false, [rowIndex]);
  resizeSegmentRow(tr, true);
  updateSubtitlePreview(collectSegments(), [rowIndex]);
});
for (const id of ['fontSize', 'marginV', 'showSpeaker', 'speakerColors']) {
  document.querySelector('#' + id).addEventListener('input', () => {
    markEditorDirty();
    updateSubtitlePreview();
  });
  document.querySelector('#' + id).addEventListener('change', () => {
    markEditorDirty();
    updateSubtitlePreview();
  });
}

async function refreshJobs(options = {}) {
  const res = await fetch(apiUrl('api/jobs'), { cache: 'no-store' });
  if (!res.ok) return;
  const data = await res.json();
  jobs = data.jobs || [];
  renderJobList();
  if (currentJob) {
    const fresh = jobs.find((job) => job.id === currentJob.id);
    if (fresh) {
      const wasEditable = EDIT_STATES.has(currentJob.status);
      currentJob = fresh;
      if (options.background && wasEditable && EDIT_STATES.has(fresh.status)) {
        updateEditorChrome(fresh);
      } else {
        renderCurrentJob(fresh, { skipSegments: options.skipSegments || editorDirty });
      }
    } else {
      currentJob = null;
      showImportView();
    }
  } else if (!options.keepSelection && jobs.length && options.selectLatest) {
    await selectJob(jobs[0].id);
  }
  ensurePolling();
}

function renderJobList() {
  jobCountEl.textContent = tp('jobs.count', jobs.length);
  if (!jobs.length) {
    jobListEl.innerHTML = `<div class="meta" style="padding:10px">${escapeHtml(t('jobs.empty'))}</div>`;
    return;
  }
  jobListEl.innerHTML = jobs.map((job) => {
    const active = currentJob && currentJob.id === job.id ? ' active' : '';
    const canDelete = !RUNNING_STATES.has(job.status);
    const percent = Math.round((job.progress || 0) * 100);
    const warning = truncationWarning(job);
    return `
      <div class="task-item${active}" data-job-id="${escapeHtml(job.id)}">
        <div class="task-row">
          <div class="task-name">${escapeHtml(job.media_name || 'input.media')}</div>
          <span class="${statusClass(job.status)}">${statusLabel(job.status)}</span>
        </div>
        <div class="task-id meta">${escapeHtml(job.id)}</div>
        <div class="meta">${escapeHtml(tokenUsageSummary(job))}</div>
        ${warning ? `<div class="warning">${escapeHtml(warning)}</div>` : ''}
        <div class="task-foot">
          <div class="progress task-progress"><div class="bar" style="width:${percent}%"></div></div>
          ${canDelete ? `<button class="small ghost" data-delete-id="${escapeHtml(job.id)}">${escapeHtml(t('actions.deleteJob'))}</button>` : ''}
        </div>
      </div>`;
  }).join('');
}

async function selectJob(jobId) {
  const local = jobs.find((job) => job.id === jobId);
  currentJob = local || currentJob;
  renderJobList();
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { cache: 'no-store' });
  if (!res.ok) {
    await refreshJobs();
    return;
  }
  currentJob = await res.json();
  renderCurrentJob(currentJob);
}

function renderCurrentJob(job, options = {}) {
  renderJobList();
  if (EDIT_STATES.has(job.status)) showEditor(job, options);
  else showProcessing(job);
}

function showImportView(options = {}) {
  if (options.clearDraft !== false) resetImportMode();
  currentJob = null;
  setEditorDirty(false);
  fileInput.value = '';
  if (!options.preserveError) clearImportError();
  setVisible(importView);
  renderJobList();
}

function resetImportMode() {
  rerunDraftJob = null;
  importTitleEl.textContent = t('actions.importMedia');
  rerunSourceEl.textContent = '';
  fileInput.disabled = false;
  uploadBtn.textContent = t('actions.startTranscription');
}

function showProcessingPlaceholder(name) {
  currentJob = null;
  processTitleEl.textContent = t('processing.creating');
  processNameEl.textContent = name;
  processMetaEl.textContent = t('processing.preparing');
  processBarEl.style.width = '2%';
  processErrorEl.textContent = '';
  setVisible(processingView);
}

function showProcessing(job) {
  processTitleEl.textContent = t(job.status === 'failed' ? 'processing.failed' : 'processing.transcribing');
  processNameEl.textContent = job.media_name || 'input.media';
  processMetaEl.textContent = jobSummary(job);
  processBarEl.style.width = `${Math.round((job.progress || 0) * 100)}%`;
  processErrorEl.textContent = job.error || truncationWarning(job);
  deleteCurrentBtn.disabled = RUNNING_STATES.has(job.status);
  setVisible(processingView);
}

async function showEditor(job, options = {}) {
  applySubtitleStyle(job.subtitle_style || {});
  updateEditorChrome(job);
  setVisible(workbench);
  const mediaUrl = apiUrl(`api/jobs/${job.id}/media`);
  if (preview.dataset.jobId !== job.id) {
    stopPreviewSync();
    preview.dataset.jobId = job.id;
    preview.src = mediaUrl;
    resetVideoStage();
  }
  renderDownloads(job.status);
  if (!options.skipSegments) await loadSegments(job.id);
  fitVideoStageToMedia();
}

function updateEditorChrome(job, options = {}) {
  selectedNameEl.textContent = job.media_name || 'input.media';
  taskStatusEl.textContent = statusLabel(job.status);
  taskStatusEl.className = statusClass(job.status);
  taskUsageEl.textContent = tokenUsageSummary(job);
  taskParamsEl.textContent = parameterSummary(job);
  if (!options.preserveNotice) {
    if (job.error) setTaskNotice(job.error, 'error');
    else if (truncationWarning(job)) setTaskNoticeKey('notice.truncatedShort', 'warning');
    else setTaskNotice('', '');
  }
  renderBtn.disabled = !ffmpegAvailable || job.status === 'rendering';
  renderBtn.textContent = job.status === 'rendering'
    ? t('actions.burning')
    : ffmpegAvailable ? t('actions.burnVideo') : t('actions.ffmpegUnavailable');
  updateRerunAction(job);
  setSaveState(editorDirty ? 'dirty' : 'saved', editorDirty ? 'save.dirty' : 'save.saved');
  renderDownloads(job.status);
}

function setTaskNotice(message, kind) {
  taskNoticeDescriptor = message ? { message, kind: kind || '' } : null;
  renderTaskNotice();
}

function setTaskNoticeKey(messageKey, kind, params = {}) {
  taskNoticeDescriptor = { messageKey, params, kind: kind || '' };
  renderTaskNotice();
}

function setLocalizedTaskNotice(data, fallbackKey, kind = 'error') {
  taskNoticeDescriptor = { data, fallbackKey, kind };
  renderTaskNotice();
}

function renderTaskNotice() {
  const descriptor = taskNoticeDescriptor;
  const message = !descriptor
    ? ''
    : descriptor.messageKey
      ? t(descriptor.messageKey, descriptor.params)
      : descriptor.fallbackKey
        ? localizedError(descriptor.data, descriptor.fallbackKey)
        : descriptor.message;
  taskNoticeEl.textContent = message || '';
  taskNoticeEl.className = 'task-notice ' + (descriptor ? descriptor.kind : '');
  taskNoticeEl.classList.toggle('is-hidden', !message);
}

function updateRerunAction(job) {
  rerunBtn.disabled = RUNNING_STATES.has(job.status);
  rerunBtn.textContent = t('actions.rerun');
}

function showRerunDraft(job) {
  const usage = job.usage || {};
  const inference = job.inference || {};
  const currentMax = Number(usage.max_new_tokens || inference.max_new_tokens || 0);
  rerunDraftJob = job;
  currentJob = null;
  importTitleEl.textContent = t('rerun.title');
  fileInput.value = '';
  fileInput.disabled = true;
  rerunSourceEl.textContent = t('rerun.source', { name: job.media_name || 'input.media' });
  promptInput.value = inference.prompt || '';
  maxNewTokensInput.value = usage.possibly_truncated && currentMax > 0
    ? Math.max(currentMax * 2, currentMax + 512)
    : currentMax || '';
  maxLenInput.value = inference.max_length || '';
  decodingSelect.value = inference.decoding || 'greedy';
  temperatureInput.value = inference.temperature == null ? '1.0' : inference.temperature;
  updateDecodingControls();
  advancedDetails.open = true;
  uploadBtn.textContent = t('actions.startRerun');
  clearImportError();
  setVisible(importView);
  renderJobList();
}

async function startRerunDraft() {
  if (!rerunDraftJob) return;
  const source = rerunDraftJob;
  const payload = {
    prompt: promptInput.value,
    max_new_tokens: Number(maxNewTokensInput.value || 0),
    max_len: Number(maxLenInput.value || 0),
    decoding: decodingSelect.value,
  };
  if (temperatureInput.value) payload.temperature = Number(temperatureInput.value);
  uploadBtn.disabled = true;
  advancedDetails.open = false;
  clearImportError();
  showProcessingPlaceholder(source.media_name || 'input.media');
  const res = await fetch(apiUrl(`api/jobs/${source.id}/rerun`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  uploadBtn.disabled = false;
  if (!res.ok) {
    setImportError(data, 'errors.rerunFailed');
    showImportView({ clearDraft: false, preserveError: true });
    return;
  }
  resetImportMode();
  currentJob = data;
  await refreshJobs({ keepSelection: true });
  await selectJob(data.id);
}

function setVisible(view) {
  importView.classList.toggle('is-hidden', view !== importView);
  processingView.classList.toggle('is-hidden', view !== processingView);
  workbench.classList.toggle('is-hidden', view !== workbench);
}

async function deleteJob(jobId) {
  const job = jobs.find((item) => item.id === jobId);
  if (job && RUNNING_STATES.has(job.status)) return;
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { method: 'DELETE' });
  if (!res.ok) return;
  if (currentJob && currentJob.id === jobId) {
    currentJob = null;
    stopPreviewSync();
    preview.removeAttribute('src');
    preview.removeAttribute('data-job-id');
    preview.load();
    tbody.innerHTML = '';
    downloads.innerHTML = '';
    setEditorDirty(false);
    showImportView();
  }
  await refreshJobs({ keepSelection: true });
}

async function saveSegments() {
  if (!currentJob) return false;
  if (!editorDirty) return true;
  setSaveState('saving', 'save.saving');
  const segments = collectSegments();
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/segments`), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments, style: collectSubtitleStyle() })
    });
    const data = await res.json();
    if (!res.ok) {
      setLocalizedTaskNotice(data, 'errors.saveFailed');
      setSaveState('error', 'save.failed');
      saveBtn.disabled = false;
      return false;
    }
    setTaskNotice('', '');
    renderSegments(data.segments);
    setEditorDirty(false);
    saveStatusEl.textContent = t('save.saved');
    saveStatusTimer = setTimeout(() => {
      if (!editorDirty) saveStatusEl.textContent = t('save.saved');
    }, 1200);
    await selectJob(currentJob.id);
    return true;
  } catch (err) {
    setTaskNoticeKey('save.failedWithDetail', 'error', { detail: err.message });
    setSaveState('error', 'save.failed');
    saveBtn.disabled = false;
    return false;
  }
}

async function loadSegments(jobId) {
  const res = await fetch(apiUrl(`api/jobs/${jobId}/segments`));
  const data = await res.json();
  renderSegments(data.segments || []);
  setEditorDirty(false);
}

function ensurePolling() {
  const shouldPoll = jobs.some((job) => RUNNING_STATES.has(job.status));
  if (shouldPoll && !pollTimer) pollTimer = setInterval(() => refreshJobs({ keepSelection: true, background: true }), 1500);
  if (!shouldPoll && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function collectSubtitleStyle() {
  return {
    font_size: Number(document.querySelector('#fontSize').value || 48),
    margin_v: Number(document.querySelector('#marginV').value || 56),
    show_speaker: document.querySelector('#showSpeaker').value === 'true',
    speaker_colors: document.querySelector('#speakerColors').value === 'true',
    speaker_names: collectSpeakerNames()
  };
}

function applySubtitleStyle(style) {
  if (!style || editorDirty) return;
  if (style.font_size != null) document.querySelector('#fontSize').value = style.font_size;
  if (style.margin_v != null) document.querySelector('#marginV').value = style.margin_v;
  if (style.show_speaker != null) document.querySelector('#showSpeaker').value = String(!!style.show_speaker);
  if (style.speaker_colors != null) document.querySelector('#speakerColors').value = String(!!style.speaker_colors);
  speakerNameMap = {};
  speakerMapEl.innerHTML = '';
  const names = style.speaker_names || {};
  for (const [speaker, name] of Object.entries(names)) {
    if (String(name).trim()) speakerNameMap[String(speaker)] = String(name).trim();
  }
}

function collectSpeakerNames() {
  const names = {};
  for (const input of speakerMapEl.querySelectorAll('input[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    const name = input.value.trim();
    if (speaker && name) names[speaker] = name;
  }
  return names;
}

function syncSpeakerNameInputs() {
  for (const input of speakerMapEl.querySelectorAll('input[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    if (!speaker) continue;
    const name = input.value.trim();
    if (name) speakerNameMap[speaker] = name;
    else delete speakerNameMap[speaker];
  }
}

function renderSpeakerMap(segments) {
  syncSpeakerNameInputs();
  const speakers = [...new Set(segments.map((segment) => segment.speaker).filter(Boolean))].sort();
  if (!speakers.length) {
    speakerMapEl.innerHTML = `<div class="meta">${escapeHtml(t('speaker.none'))}</div>`;
    return;
  }
  speakerMapEl.innerHTML = speakers.map((speaker) => {
    const name = speakerNameMap[speaker] || '';
    return `
      <div class="speaker-map-row">
        <div class="speaker-tag">${escapeHtml(speaker)}</div>
        <input type="text" data-speaker="${escapeHtml(speaker)}" value="${escapeHtml(name)}" placeholder="${escapeHtml(t('speaker.displayName'))}">
      </div>`;
  }).join('');
}

function speakerDisplayName(speaker) {
  const names = collectSpeakerNames();
  return names[speaker] || speakerNameMap[speaker] || speaker;
}

function renderSegments(segments) {
  tbody.innerHTML = '';
  activeSegmentIndex = -1;
  activeSegmentIndexesKey = null;
  for (const [index, segment] of segments.entries()) {
    const tr = document.createElement('tr');
    tr.dataset.id = segment.id;
    tr.dataset.index = String(index);
    tr.innerHTML = `
      <td><input class="start" type="number" min="0" step="0.01" value="${segment.start}"></td>
      <td><input class="end" type="number" min="0" step="0.01" value="${segment.end}"></td>
      <td><input class="speaker" type="text" value="${escapeHtml(segment.speaker)}"></td>
      <td><textarea class="text" rows="1">${escapeHtml(segment.text)}</textarea></td>
    `;
    tr.addEventListener('click', (event) => {
      if (event.target.closest('input, textarea')) return;
      const rowIndex = Number(tr.dataset.index);
      const start = Number(tr.querySelector('.start').value);
      if (Number.isFinite(start)) preview.currentTime = Math.max(0, start);
      const latestSegments = collectSegments();
      const activeIndexes = activeSegmentIndexesAt(Number(preview.currentTime || 0), latestSegments);
      setActiveSegment(rowIndex, false, activeIndexes.includes(rowIndex) ? activeIndexes : [rowIndex]);
      updateSubtitlePreview(latestSegments, activeIndexes.includes(rowIndex) ? activeIndexes : [rowIndex]);
    });
    tbody.appendChild(tr);
    resizeSegmentRow(tr, false);
  }
  renderSpeakerMap(segments);
  syncActiveSegment();
}

function resizeSegmentTextarea(textarea, expanded) {
  if (!textarea) return;
  const maxHeight = expanded ? 112 : 48;
  textarea.style.height = 'auto';
  const naturalHeight = textarea.scrollHeight;
  const nextHeight = Math.max(30, Math.min(naturalHeight, maxHeight));
  textarea.style.height = nextHeight + 'px';
  textarea.style.overflowY = naturalHeight > maxHeight ? 'auto' : 'hidden';
}

function resizeSegmentRow(tr, expanded) {
  resizeSegmentTextarea(tr && tr.querySelector('textarea.text'), expanded);
}

function collectSegments() {
  return [...tbody.querySelectorAll('tr')].map((tr, index) => ({
    id: tr.dataset.id || `seg_${String(index + 1).padStart(4, '0')}`,
    start: Number(tr.querySelector('.start').value),
    end: Number(tr.querySelector('.end').value),
    speaker: tr.querySelector('.speaker').value,
    text: tr.querySelector('.text').value
  }));
}

function resetVideoStage() {
  assPlayRes = { x: 1920, y: 1080 };
  videoStage.style.width = '';
  videoStage.style.height = '';
  videoStage.style.aspectRatio = assPlayRes.x + ' / ' + assPlayRes.y;
}

function fitVideoStageToMedia() {
  const videoWidth = Number(preview.videoWidth || 0);
  const videoHeight = Number(preview.videoHeight || 0);
  const shell = videoStage.parentElement;
  if (!shell || videoWidth <= 0 || videoHeight <= 0) {
    resetVideoStage();
    updateSubtitlePreview();
    return;
  }
  assPlayRes = { x: videoWidth, y: videoHeight };
  const maxWidth = shell.clientWidth || videoWidth;
  const maxHeight = Math.max(180, Math.floor(window.innerHeight * 0.48));
  const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight);
  videoStage.style.width = Math.max(1, Math.floor(videoWidth * scale)) + 'px';
  videoStage.style.height = Math.max(1, Math.floor(videoHeight * scale)) + 'px';
  videoStage.style.aspectRatio = videoWidth + ' / ' + videoHeight;
  updateSubtitlePreview();
}

function assScriptScale() {
  const playResY = Number(assPlayRes.y || preview.videoHeight || 0);
  if (playResY <= 0) return 1;
  return (videoStage.clientHeight || playResY) / playResY;
}

function schedulePreviewSync() {
  if (previewSyncRequest !== null || preview.paused || preview.ended) return;
  if (typeof preview.requestVideoFrameCallback === 'function') {
    previewSyncRequestKind = 'video';
    previewSyncRequest = preview.requestVideoFrameCallback((_now, metadata) => {
      previewSyncRequest = null;
      previewSyncRequestKind = null;
      const mediaTime = metadata && Number.isFinite(metadata.mediaTime) ? metadata.mediaTime : undefined;
      syncActiveSegment(mediaTime);
      schedulePreviewSync();
    });
    return;
  }
  previewSyncRequestKind = 'animation';
  previewSyncRequest = requestAnimationFrame(() => {
    previewSyncRequest = null;
    previewSyncRequestKind = null;
    syncActiveSegment();
    schedulePreviewSync();
  });
}

function stopPreviewSync() {
  if (previewSyncRequest === null) return;
  if (previewSyncRequestKind === 'video' && typeof preview.cancelVideoFrameCallback === 'function') {
    preview.cancelVideoFrameCallback(previewSyncRequest);
  } else {
    cancelAnimationFrame(previewSyncRequest);
  }
  previewSyncRequest = null;
  previewSyncRequestKind = null;
}

function syncActiveSegment(timeOverride, forceRender = false) {
  const time = Number.isFinite(timeOverride) ? timeOverride : Number(preview.currentTime || 0);
  const segments = collectSegments();
  const activeIndexes = activeSegmentIndexesAt(time, segments);
  const index = activeIndexes.length ? activeIndexes[0] : -1;
  const activeSetChanged = setActiveSegment(index, true, activeIndexes);
  if (activeSetChanged || forceRender) updateSubtitlePreview(segments, activeIndexes);
}

function setActiveSegment(index, shouldScroll, activeIndexes) {
  const normalizedIndexes = activeIndexes || (index >= 0 ? [index] : []);
  const nextIndexesKey = normalizedIndexes.join(',');
  if (index === activeSegmentIndex && nextIndexesKey === activeSegmentIndexesKey) return false;
  const activeSet = new Set(normalizedIndexes);
  activeSegmentIndex = index;
  activeSegmentIndexesKey = nextIndexesKey;
  for (const tr of tbody.querySelectorAll('tr')) {
    const rowIndex = Number(tr.dataset.index);
    const active = rowIndex === index;
    const overlap = activeSet.has(rowIndex);
    tr.classList.toggle('active', active);
    tr.classList.toggle('overlap', overlap && !active);
    resizeSegmentRow(tr, active);
    if (active && shouldScroll) scrollSegmentRowIntoView(tr);
  }
  return true;
}

function activeSegmentIndexesAt(time, segments) {
  const indexes = [];
  for (const [index, segment] of segments.entries()) {
    if (segmentContainsTime(segment, time)) indexes.push(index);
  }
  return indexes;
}

function segmentContainsTime(segment, time) {
  const start = Number(segment.start);
  const end = Number(segment.end);
  return Number.isFinite(start) && Number.isFinite(end) && start <= time && time < end;
}

function assignOverlapLanes(segments) {
  const lanes = new Array(segments.length).fill(0);
  const laneEnds = [];
  const indexed = segments.map((segment, index) => ({ segment, index })).sort((a, b) => {
    const startDiff = Number(a.segment.start) - Number(b.segment.start);
    if (startDiff) return startDiff;
    const endDiff = Number(a.segment.end) - Number(b.segment.end);
    if (endDiff) return endDiff;
    return a.index - b.index;
  });

  for (const item of indexed) {
    const start = Number(item.segment.start);
    const end = Math.max(start, Number(item.segment.end));
    let lane = laneEnds.findIndex((laneEnd) => laneEnd <= start);
    if (lane < 0) {
      lane = laneEnds.length;
      laneEnds.push(end);
    } else {
      laneEnds[lane] = end;
    }
    lanes[item.index] = lane;
  }
  return lanes;
}

function scrollSegmentRowIntoView(tr) {
  const container = tr.closest('.table-wrap');
  if (!container) return;
  const stickyHeaderHeight = 30;
  const rowTop = tr.offsetTop;
  const rowBottom = rowTop + tr.offsetHeight;
  const viewTop = container.scrollTop + stickyHeaderHeight;
  const viewBottom = container.scrollTop + container.clientHeight;
  if (rowTop < viewTop) {
    container.scrollTop = Math.max(0, rowTop - stickyHeaderHeight - 4);
  } else if (rowBottom > viewBottom) {
    container.scrollTop = rowBottom - container.clientHeight + 8;
  }
}

function updateSubtitlePreview(segments, activeIndexes) {
  segments = segments || collectSegments();
  activeIndexes = activeIndexes || activeSegmentIndexesAt(Number(preview.currentTime || 0), segments);
  const visibleIndexes = activeIndexes.filter((index) => segments[index] && segments[index].text);
  if (!visibleIndexes.length) {
    subtitleOverlay.classList.remove('visible');
    subtitleOverlay.replaceChildren();
    return;
  }
  const showSpeaker = document.querySelector('#showSpeaker').value === 'true';
  const useSpeakerColors = document.querySelector('#speakerColors').value === 'true';
  const fontSize = Math.max(12, Number(document.querySelector('#fontSize').value || 48));
  const marginV = Math.max(0, Number(document.querySelector('#marginV').value || 56));
  const scale = assScriptScale();
  subtitleOverlay.style.fontSize = Math.max(10, fontSize * scale / assFontLineHeightFactor) + 'px';
  subtitleOverlay.style.lineHeight = String(assFontLineHeightFactor);
  subtitleOverlay.style.webkitTextStroke = subtitleTextStroke(scale);
  subtitleOverlay.style.textShadow = subtitleTextShadow(scale);
  subtitleOverlay.replaceChildren();
  const lanes = assignOverlapLanes(segments);
  const laneStep = Math.max(1, fontSize) * scale;
  const baseMargin = Math.max(0, marginV * scale);
  visibleIndexes
    .sort((left, right) => lanes[left] - lanes[right] || Number(segments[left].start) - Number(segments[right].start) || left - right)
    .forEach((index) => {
      const segment = segments[index];
      const line = document.createElement('div');
      line.className = 'subtitle-line';
      line.textContent = showSpeaker && segment.speaker ? speakerDisplayName(segment.speaker) + ': ' + segment.text : segment.text;
      const color = useSpeakerColors ? speakerColor(segment.speaker, segments) : '#ffffff';
      line.style.color = color;
      line.style.webkitTextFillColor = color;
      line.style.bottom = baseMargin + lanes[index] * laneStep + 'px';
      line.dataset.lane = String(lanes[index]);
      subtitleOverlay.appendChild(line);
    });
  subtitleOverlay.classList.add('visible');
}

function subtitleTextStroke(scale) {
  return Math.max(1, 3 * scale) + 'px #000';
}

function subtitleTextShadow(scale) {
  const shadow = Math.max(0.5, 1 * scale);
  const blur = Math.max(1, 3 * scale);
  return `0 ${shadow}px ${blur}px rgba(0, 0, 0, 0.65)`;
}

function speakerColor(speaker, segments) {
  const speakers = [];
  for (const segment of segments) {
    if (segment.speaker && !speakers.includes(segment.speaker)) speakers.push(segment.speaker);
  }
  speakers.sort();
  const index = Math.max(0, speakers.indexOf(speaker || ''));
  return speakerPalette[index % speakerPalette.length];
}

function renderDownloads(status) {
  if (!currentJob) return;
  const links = [
    ['json', 'JSON'],
    ['srt', 'SRT'],
    ['ass', 'ASS'],
    ['transcript', t('outputs.transcript')]
  ];
  if (status === 'done') links.push(['mp4', 'MP4']);
  downloads.innerHTML = links.map(([kind, label]) =>
    `<a href="${apiUrl(`api/jobs/${currentJob.id}/download?kind=${kind}`)}" target="_blank">${label}</a>`
  ).join('');
}

function jobSummary(job) {
  const inference = job.inference || {};
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return tokenUsageSummary(job) + ' · max_len ' + inference.max_length + ' · ' + inference.decoding + temp;
}

function parameterSummary(job) {
  const inference = job.inference || {};
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return 'max_len ' + inference.max_length + ' · ' + inference.decoding + temp;
}

function tokenUsageSummary(job) {
  const usage = job.usage || {};
  const inference = job.inference || {};
  const maxNewTokens = usage.max_new_tokens || inference.max_new_tokens || 0;
  if (usage.generated_tokens == null) return t('usage.pending', { max: maxNewTokens });
  const prompt = usage.prompt_tokens == null ? '' : t('usage.prompt', { prompt: usage.prompt_tokens });
  return t('usage.generated', { generated: usage.generated_tokens, max: maxNewTokens, prompt });
}

function truncationWarning(job) {
  const usage = job.usage || {};
  if (!usage.possibly_truncated) return '';
  return t('notice.truncatedLong');
}

function statusClass(status) {
  return 'pill ' + (status === 'failed' ? 'bad' : status === 'done' ? 'ok' : '');
}

function statusLabel(status) {
  const key = `status.${status}`;
  const label = t(key);
  return label === key ? status : label;
}

function refreshLocalizedUI() {
  localeSelect.value = getLocale();
  renderRuntimeStatus();
  setSidebarCollapsed(document.body.classList.contains('sidebar-collapsed'), false);
  renderJobList();
  setSaveState(currentSaveState, currentSaveMessageKey, currentSaveMessageParams);

  if (rerunDraftJob) {
    importTitleEl.textContent = t('rerun.title');
    rerunSourceEl.textContent = t('rerun.source', { name: rerunDraftJob.media_name || 'input.media' });
    uploadBtn.textContent = t('actions.startRerun');
  } else {
    importTitleEl.textContent = t('actions.importMedia');
    uploadBtn.textContent = t('actions.startTranscription');
  }

  if (currentJob) {
    if (EDIT_STATES.has(currentJob.status)) {
      updateEditorChrome(currentJob, { preserveNotice: true });
      renderSpeakerMap(collectSegments());
      updateSubtitlePreview();
    } else {
      showProcessing(currentJob);
    }
  } else if (!processingView.classList.contains('is-hidden') && processNameEl.textContent) {
    processTitleEl.textContent = t('processing.creating');
    processMetaEl.textContent = t('processing.preparing');
  }
  renderImportError();
  renderTaskNotice();
}

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

await initI18n();
localeSelect.value = getLocale();
restoreSidebarState();
renderJobList();
setSaveState('saved', 'save.saved');
refreshRuntime();
refreshJobs({ selectLatest: true });
