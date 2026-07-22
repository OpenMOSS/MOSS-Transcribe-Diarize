export const SUPPORTED_LOCALES = ['zh-CN', 'en'];

const FALLBACK_LOCALE = 'en';
const STORAGE_KEY = 'mtdUiLocale';
const messageCache = new Map();
let currentLocale = FALLBACK_LOCALE;
let currentMessages = {};

export function normalizeLocale(locale) {
  const value = String(locale || '').trim().toLowerCase();
  if (value === 'zh' || value.startsWith('zh-')) return 'zh-CN';
  if (value === 'en' || value.startsWith('en-')) return 'en';
  return null;
}

export function preferredLocale() {
  try {
    const saved = normalizeLocale(localStorage.getItem(STORAGE_KEY));
    if (saved) return saved;
  } catch (err) {}
  const candidates = Array.isArray(navigator.languages) && navigator.languages.length
    ? navigator.languages
    : [navigator.language];
  for (const candidate of candidates) {
    const locale = normalizeLocale(candidate);
    if (locale) return locale;
  }
  return FALLBACK_LOCALE;
}

async function loadMessages(locale) {
  if (messageCache.has(locale)) return messageCache.get(locale);
  const url = new URL(`locales/${locale}.json`, import.meta.url);
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Unable to load locale ${locale}: ${response.status}`);
  const messages = await response.json();
  messageCache.set(locale, messages);
  return messages;
}

export async function initI18n() {
  return setLocale(preferredLocale(), false);
}

export async function setLocale(locale, persist = true) {
  let resolved = normalizeLocale(locale) || FALLBACK_LOCALE;
  try {
    currentMessages = await loadMessages(resolved);
  } catch (err) {
    console.error(err);
    resolved = FALLBACK_LOCALE;
    currentMessages = await loadMessages(resolved);
  }
  currentLocale = resolved;
  document.documentElement.lang = resolved;
  applyDocumentTranslations();
  if (persist) {
    try {
      localStorage.setItem(STORAGE_KEY, resolved);
    } catch (err) {}
  }
  return resolved;
}

export function getLocale() {
  return currentLocale;
}

export function hasMessage(key) {
  return Object.prototype.hasOwnProperty.call(currentMessages, key);
}

export function t(key, params = {}) {
  const template = hasMessage(key) ? String(currentMessages[key]) : key;
  return template.replace(/\{([A-Za-z0-9_]+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : match
  ));
}

export function tp(key, count, params = {}) {
  const category = new Intl.PluralRules(currentLocale).select(Number(count));
  const candidate = `${key}.${category}`;
  const resolvedKey = hasMessage(candidate) ? candidate : `${key}.other`;
  return t(resolvedKey, { ...params, count });
}

export function applyDocumentTranslations(root = document) {
  for (const element of root.querySelectorAll('[data-i18n]')) {
    element.textContent = t(element.dataset.i18n);
  }
  for (const element of root.querySelectorAll('[data-i18n-placeholder]')) {
    element.setAttribute('placeholder', t(element.dataset.i18nPlaceholder));
  }
  for (const element of root.querySelectorAll('[data-i18n-title]')) {
    element.setAttribute('title', t(element.dataset.i18nTitle));
  }
  for (const element of root.querySelectorAll('[data-i18n-aria-label]')) {
    element.setAttribute('aria-label', t(element.dataset.i18nAriaLabel));
  }
}

export function localizedError(data, fallbackKey) {
  if (data && data.code && hasMessage(`errors.${data.code}`)) return t(`errors.${data.code}`);
  if (data && typeof data.detail === 'string' && data.detail) return data.detail;
  return t(fallbackKey);
}
