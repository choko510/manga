(function () {
    const STORAGE_KEYS = {
        hiddenTags: 'manga_hidden_tags',
        liked: 'manga_liked_galleries',
        history: 'manga_view_history',
        theme: 'manga_theme',
        tagUsage: 'manga_tag_usage'
    };

    let translationPromise = null;
    let translations = {};
    const aliasIndex = new Map();
    const canonicalTags = new Map();
    let hiddenTagSet = null;
    let likedSet = null;
    let tagUsageMap = null;
    const galleryThumbnailCache = new Map();
    const TAG_USAGE_LIMIT = 200;

    function normaliseTag(tag) {
        return (tag || '').toString().trim().toLowerCase();
    }

    async function loadTranslations() {
        // 毎回最新の tag-translations.json を取得するため、リクエスト毎にキャッシュを回避
        translationPromise = fetch('/static/tag-translations.json?ts=' + Date.now(), { cache: 'no-store' })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to load translations');
                }
                return response.json();
            })
            .then((data) => {
                translations = {};
                aliasIndex.clear();
                canonicalTags.clear();
                Object.entries(data || {}).forEach(([key, value]) => {
                    const norm = normaliseTag(key);
                    if (!norm) {
                        return;
                    }
                    const record = parseTranslationRecord(value);
                    translations[norm] = record;
                    canonicalTags.set(norm, key);
                    registerAlias(key, key);
                    if (record.translation) {
                        registerAlias(record.translation, key);
                    }
                    if (Array.isArray(record.aliases)) {
                        record.aliases.forEach((alias) => registerAlias(alias, key));
                    }
                });
                return translations;
            })
            .catch(() => {
                translations = {};
                aliasIndex.clear();
                canonicalTags.clear();
                return translations;
            });
        return translationPromise;
    }

    function registerAlias(alias, canonicalTag) {
        const norm = normaliseTag(alias);
        if (!norm) {
            return;
        }
        let targets = aliasIndex.get(norm);
        if (!targets) {
            targets = new Set();
            aliasIndex.set(norm, targets);
        }
        targets.add(canonicalTag);
    }

    function parseTranslationRecord(value) {
        if (value && typeof value === 'object') {
            const translation = typeof value.translation === 'string' ? value.translation : '';
            const description = typeof value.description === 'string' ? value.description : '';
            const aliases = [];
            if (Array.isArray(value.aliases)) {
                const seen = new Set();
                value.aliases.forEach((alias) => {
                    if (typeof alias !== 'string') {
                        return;
                    }
                    const trimmed = alias.trim();
                    if (!trimmed) {
                        return;
                    }
                    const norm = normaliseTag(trimmed);
                    if (!norm || seen.has(norm)) {
                        return;
                    }
                    seen.add(norm);
                    aliases.push(trimmed);
                });
            }
            return { translation, description, aliases };
        }
        if (typeof value === 'string') {
            return { translation: value, description: '', aliases: [] };
        }
        return { translation: '', description: '', aliases: [] };
    }

    function getTagMetadata(tag) {
        if (!tag) {
            return { translation: '', description: '', aliases: [] };
        }
        const entry = translations[normaliseTag(tag)];
        if (entry && typeof entry === 'object') {
            return {
                translation: typeof entry.translation === 'string' ? entry.translation : '',
                description: typeof entry.description === 'string' ? entry.description : '',
                aliases: Array.isArray(entry.aliases) ? entry.aliases : [],
            };
        }
        return { translation: '', description: '', aliases: [] };
    }

    function translateTag(tag) {
        if (!tag) return '';
        const metadata = getTagMetadata(tag);
        return metadata.translation || tag;
    }

    function getTagDescription(tag) {
        if (!tag) {
            return '';
        }
        const metadata = getTagMetadata(tag);
        return metadata.description || '';
    }

    function getTagAliases(tag) {
        if (!tag) {
            return [];
        }
        const metadata = getTagMetadata(tag);
        return Array.isArray(metadata.aliases) ? metadata.aliases : [];
    }

    function resolveTagKeyword(keyword) {
        if (!keyword) {
            return [];
        }
        const norm = normaliseTag(keyword);
        if (!norm) {
            return [];
        }
        const results = new Set();
        const direct = canonicalTags.get(norm);
        if (direct) {
            results.add(direct);
        }
        const aliases = aliasIndex.get(norm);
        if (aliases) {
            aliases.forEach((value) => results.add(value));
        }
        return Array.from(results);
    }

    function resolveTagQueryString(query) {
        if (!query) {
            return '';
        }
        const segments = query.split(/[,\s]+/).filter((segment) => segment.length > 0);
        const resolved = [];
        const seen = new Set();
        segments.forEach((segment) => {
            const isNegated = segment.startsWith('-');
            const base = isNegated ? segment.slice(1) : segment;
            const replacements = resolveTagKeyword(base);
            if (!replacements.length) {
                if (!seen.has(segment)) {
                    seen.add(segment);
                    resolved.push(segment);
                }
                return;
            }
            replacements.forEach((replacement) => {
                const token = isNegated ? `-${replacement}` : replacement;
                if (seen.has(token)) {
                    return;
                }
                seen.add(token);
                resolved.push(token);
            });
        });
        return resolved.join(' ');
    }

    function getHiddenTags() {
        if (hiddenTagSet !== null) {
            return Array.from(hiddenTagSet);
        }
        const raw = localStorage.getItem(STORAGE_KEYS.hiddenTags);
        const tags = new Set();
        if (raw) {
            raw.split(',').forEach((item) => {
                const norm = normaliseTag(item);
                if (norm) {
                    tags.add(norm);
                }
            });
        }
        hiddenTagSet = tags;
        return Array.from(hiddenTagSet);
    }

    function saveHiddenTags(tags) {
        hiddenTagSet = new Set(tags.map(normaliseTag).filter((tag) => tag));
        localStorage.setItem(STORAGE_KEYS.hiddenTags, Array.from(hiddenTagSet).join(','));
        document.dispatchEvent(new CustomEvent('manga:hidden-tags-change', {
            detail: { tags: Array.from(hiddenTagSet) }
        }));
    }

    function isTagHidden(tag) {
        if (hiddenTagSet === null) {
            getHiddenTags();
        }
        return hiddenTagSet.has(normaliseTag(tag));
    }

    function getLikedSet() {
        if (likedSet !== null) {
            return likedSet;
        }
        try {
            const raw = localStorage.getItem(STORAGE_KEYS.liked);
            const parsed = raw ? JSON.parse(raw) : [];
            likedSet = new Set(parsed);
        } catch (error) {
            likedSet = new Set();
        }
        return likedSet;
    }

    function persistLikes() {
        localStorage.setItem(STORAGE_KEYS.liked, JSON.stringify(Array.from(getLikedSet())));
        document.dispatchEvent(new CustomEvent('manga:likes-change', {
            detail: { likes: Array.from(getLikedSet()) }
        }));
    }

    function replaceLikes(ids) {
        if (Array.isArray(ids)) {
            likedSet = new Set(ids);
        } else {
            likedSet = new Set();
        }
        persistLikes();
    }

    function isLiked(galleryId) {
        return getLikedSet().has(galleryId);
    }

    function toggleLike(galleryId) {
        const likes = getLikedSet();
        if (likes.has(galleryId)) {
            likes.delete(galleryId);
        } else {
            likes.add(galleryId);
        }
        persistLikes();
        return likes.has(galleryId);
    }

    function sanitiseHistoryEntry(entry) {
        if (!entry || typeof entry !== 'object') {
            return null;
        }
        const { image_urls, thumbnail_url, ...rest } = entry;
        return { ...rest };
    }

    function sanitiseHistory(entries) {
        if (!Array.isArray(entries)) {
            return [];
        }
        return entries
            .map((item) => sanitiseHistoryEntry(item))
            .filter((item) => item !== null);
    }

    function getHistory() {
        try {
            const raw = localStorage.getItem(STORAGE_KEYS.history);
            if (!raw) {
                return [];
            }
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                const sanitised = sanitiseHistory(parsed);
                const hadSensitiveFields = parsed.some((item) =>
                    item && typeof item === 'object' && ('image_urls' in item || 'thumbnail_url' in item)
                );
                if (hadSensitiveFields) {
                    localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(sanitised));
                }
                return sanitised;
            }
            return [];
        } catch (error) {
            return [];
        }
    }

    function saveHistory(entries) {
        const sanitised = sanitiseHistory(entries);
        localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(sanitised));
        document.dispatchEvent(new CustomEvent('manga:history-change', { detail: { history: sanitised } }));
    }

    const HISTORY_LIMIT = 20;

    function findFirstImageUrl(source) {
        if (Array.isArray(source)) {
            for (const value of source) {
                if (typeof value === 'string' && value.trim().length > 0) {
                    return value.trim();
                }
            }
        } else if (typeof source === 'string' && source.trim().length > 0) {
            return source.trim();
        }
        return '';
    }

    function cacheThumbnailFromEntry(entry) {
        if (!entry || typeof entry !== 'object') {
            return;
        }
        const galleryId = entry.gallery_id;
        const firstImage = findFirstImageUrl(entry.image_urls);
        if (!galleryId || !firstImage) {
            return;
        }
        const cached = galleryThumbnailCache.get(galleryId);
        if (typeof cached === 'string' && cached) {
            return;
        }
        galleryThumbnailCache.set(galleryId, firstImage);
    }

    function getCachedGalleryThumbnailUrl(galleryId) {
        const cached = galleryThumbnailCache.get(galleryId);
        return typeof cached === 'string' ? cached : '';
    }

    function getThumbnailUrl(entry) {
        if (!entry || typeof entry !== 'object') {
            return '';
        }
        const galleryId = entry.gallery_id;
        let raw = '';
        if (galleryId) {
            raw = getCachedGalleryThumbnailUrl(galleryId);
            if (!raw) {
                const fromEntry = findFirstImageUrl(entry.image_urls);
                if (fromEntry) {
                    galleryThumbnailCache.set(galleryId, fromEntry);
                    raw = fromEntry;
                }
            }
        } else {
            raw = findFirstImageUrl(entry.image_urls);
        }
        if (!raw) {
            return '';
        }
        return raw.startsWith('/proxy/') ? raw : `/proxy/${raw}`;
    }

    function buildThumbnailStyle(entry) {
        const url = getThumbnailUrl(entry);
        return url ? `background-image: url(${url});` : '';
    }

    function fetchGalleryThumbnail(galleryId) {
        if (!galleryId) {
            return Promise.resolve('');
        }
        const cached = galleryThumbnailCache.get(galleryId);
        if (typeof cached === 'string') {
            return Promise.resolve(cached);
        }
        if (cached && typeof cached.then === 'function') {
            return cached;
        }
        const request = fetch(`/gallery/${galleryId}`)
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to fetch gallery');
                }
                return response.json();
            })
            .then((data) => {
                const firstImage = findFirstImageUrl(data?.image_urls);
                const normalised = firstImage || '';
                galleryThumbnailCache.set(galleryId, normalised);
                return normalised;
            })
            .catch(() => {
                galleryThumbnailCache.set(galleryId, '');
                return '';
            });
        galleryThumbnailCache.set(galleryId, request);
        return request;
    }

    function addHistoryEntry(entry) {
        if (!entry || !entry.gallery_id) {
            return;
        }
        const currentHistory = getHistory();
        const existingEntry = currentHistory.find((item) => item.gallery_id === entry.gallery_id);
        const history = currentHistory.filter((item) => item.gallery_id !== entry.gallery_id);
        cacheThumbnailFromEntry(entry);
        const pageCount = Number.isFinite(entry.page_count) && entry.page_count >= 0
            ? entry.page_count
            : Number.isFinite(existingEntry?.page_count)
                ? existingEntry.page_count
                : 0;
        const lastPageIndex = Number.isFinite(entry.last_page_index)
            ? Math.max(0, entry.last_page_index)
            : Number.isFinite(existingEntry?.last_page_index)
                ? Math.max(0, existingEntry.last_page_index)
                : 0;
        const lastPage = Number.isFinite(entry.last_page)
            ? Math.max(1, entry.last_page)
            : Number.isFinite(existingEntry?.last_page)
                ? Math.max(1, existingEntry.last_page)
                : Math.min(lastPageIndex + 1, Math.max(1, pageCount));
        const completed = typeof entry.completed === 'boolean'
            ? entry.completed
            : existingEntry?.completed ?? (pageCount > 0 && lastPage >= pageCount);
        history.unshift({
            gallery_id: entry.gallery_id,
            japanese_title: entry.japanese_title || '',
            viewed_at: Date.now(),
            updated_at: Date.now(),
            page_count: pageCount,
            last_page_index: Math.min(lastPageIndex, Math.max(0, pageCount - 1)),
            last_page: Math.min(lastPage, Math.max(1, pageCount || lastPage)),
            completed,
        });
        const trimmed = history.slice(0, HISTORY_LIMIT);
        saveHistory(trimmed);
    }

    function updateHistoryProgress(galleryId, pageIndex, totalPages) {
        if (!galleryId || !Number.isFinite(pageIndex)) {
            return;
        }

        const history = getHistory();
        const targetIndex = history.findIndex((item) => item.gallery_id === galleryId);
        if (targetIndex === -1) {
            return;
        }

        const entry = { ...history[targetIndex] };
        const pages = Number.isFinite(totalPages) && totalPages > 0
            ? Math.max(1, Math.floor(totalPages))
            : Number.isFinite(entry.page_count) && entry.page_count > 0
                ? entry.page_count
                : null;

        const clampedIndex = Math.max(0, pages ? Math.min(pageIndex, pages - 1) : Math.floor(pageIndex));
        entry.last_page_index = clampedIndex;
        entry.last_page = clampedIndex + 1;
        if (pages) {
            entry.page_count = pages;
            entry.completed = entry.last_page >= pages;
        }
        entry.updated_at = Date.now();

        history.splice(targetIndex, 1);
        history.unshift(entry);
        const trimmed = history.slice(0, HISTORY_LIMIT);
        saveHistory(trimmed);
    }

    function clearHistory() {
        saveHistory([]);
    }

    function removeHistoryEntries(ids) {
        if (!Array.isArray(ids) || ids.length === 0) {
            return;
        }
        const targetIds = new Set(
            ids
                .map((id) => {
                    if (typeof id === 'number' || typeof id === 'string') {
                        return String(id);
                    }
                    if (id && typeof id === 'object' && 'gallery_id' in id) {
                        return String(id.gallery_id);
                    }
                    return null;
                })
                .filter((value) => typeof value === 'string' && value.length > 0)
        );
        if (!targetIds.size) {
            return;
        }

        const history = getHistory();
        const filtered = history.filter((item) => {
            if (!item || typeof item !== 'object') {
                return true;
            }
            const id = item.gallery_id;
            const str = (typeof id === 'number' || typeof id === 'string') ? String(id) : '';
            if (!str) {
                return true;
            }
            return !targetIds.has(str);
        });

        if (filtered.length === history.length) {
            return;
        }
        saveHistory(filtered);
    }

    function removeHistoryEntry(id) {
        if (typeof id === 'undefined' || id === null) {
            return;
        }
        removeHistoryEntries([id]);
    }

    function getPreferredTheme() {
        const stored = localStorage.getItem(STORAGE_KEYS.theme);
        if (stored === 'dark' || stored === 'light') {
            return stored;
        }
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        return prefersDark ? 'dark' : 'light';
    }

    function applyTheme(doc = document, theme = null) {
        const targetTheme = theme || getPreferredTheme();
        doc.body.classList.toggle('dark-mode', targetTheme === 'dark');
        doc.documentElement.setAttribute('data-theme', targetTheme);
        document.dispatchEvent(new CustomEvent('manga:theme-change', { detail: { theme: targetTheme } }));
    }

    function toggleTheme() {
        const nextTheme = getPreferredTheme() === 'dark' ? 'light' : 'dark';
        localStorage.setItem(STORAGE_KEYS.theme, nextTheme);
        applyTheme(document, nextTheme);
        return nextTheme;
    }

    function getTagUsageMap() {
        if (tagUsageMap && typeof tagUsageMap === 'object') {
            return tagUsageMap;
        }
        try {
            const raw = localStorage.getItem(STORAGE_KEYS.tagUsage);
            if (!raw) {
                tagUsageMap = {};
            } else {
                const parsed = JSON.parse(raw);
                tagUsageMap = parsed && typeof parsed === 'object' ? pruneTagUsageMap(parsed) : {};
            }
        } catch (error) {
            tagUsageMap = {};
        }
        return tagUsageMap;
    }

    function pruneTagUsageMap(map) {
        const entries = Object.entries(map || {}).filter(([, value]) => {
            const numeric = Number.isFinite(value) ? value : Number.parseInt(value, 10);
            return Number.isFinite(numeric) && numeric > 0;
        });
        entries.sort((a, b) => {
            const valueA = Number.isFinite(a[1]) ? a[1] : Number.parseInt(a[1], 10) || 0;
            const valueB = Number.isFinite(b[1]) ? b[1] : Number.parseInt(b[1], 10) || 0;
            return valueB - valueA;
        });
        const limited = entries.slice(0, TAG_USAGE_LIMIT);
        const sanitised = {};
        limited.forEach(([key, value]) => {
            const numeric = Number.isFinite(value) ? value : Number.parseInt(value, 10) || 0;
            if (numeric > 0) {
                sanitised[key] = numeric;
            }
        });
        return sanitised;
    }

    function persistTagUsage(map) {
        tagUsageMap = map;
        try {
            localStorage.setItem(STORAGE_KEYS.tagUsage, JSON.stringify(map));
        } catch (error) {
            /* noop */
        }
    }

    function dispatchTagUsageChange(map) {
        document.dispatchEvent(new CustomEvent('manga:tag-usage-change', {
            detail: { usage: { ...map } }
        }));
    }

    function recordTagUsage(tags) {
        const values = Array.isArray(tags) ? tags : [tags];
        if (!values.length) {
            return;
        }
        const baseMap = { ...getTagUsageMap() };
        let changed = false;
        values.forEach((value) => {
            const norm = normaliseTag(value);
            if (!norm) {
                return;
            }
            const currentRaw = baseMap[norm];
            const numeric = Number.isFinite(currentRaw) ? currentRaw : Number.parseInt(currentRaw, 10);
            const current = Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
            baseMap[norm] = current + 1;
            changed = true;
        });
        if (!changed) {
            return;
        }
        const pruned = pruneTagUsageMap(baseMap);
        persistTagUsage(pruned);
        dispatchTagUsageChange(pruned);
    }

    function getTagUsageCounts() {
        const map = getTagUsageMap();
        const result = {};
        Object.entries(map || {}).forEach(([key, value]) => {
            const numeric = Number.isFinite(value) ? value : Number.parseInt(value, 10);
            if (Number.isFinite(numeric) && numeric > 0) {
                result[key] = numeric;
            }
        });
        return result;
    }

    function replaceTagUsage(map) {
        if (!map || typeof map !== 'object') {
            persistTagUsage({});
            dispatchTagUsageChange({});
            return;
        }
        const base = {};
        Object.entries(map).forEach(([key, value]) => {
            const norm = normaliseTag(key);
            const numeric = Number.isFinite(value) ? value : Number.parseInt(value, 10);
            if (norm && Number.isFinite(numeric) && numeric > 0) {
                base[norm] = numeric;
            }
        });
        const pruned = pruneTagUsageMap(base);
        persistTagUsage(pruned);
        dispatchTagUsageChange(pruned);
    }

    function exportUserData() {
        return {
            history: getHistory(),
            hidden_tags: getHiddenTags(),
            likes: Array.from(getLikedSet()),
            tag_usage: getTagUsageCounts(),
        };
    }

    function importUserData(data) {
        if (!data || typeof data !== 'object') {
            return false;
        }
        let changed = false;
        if (Array.isArray(data.history)) {
            saveHistory(data.history);
            changed = true;
        }
        if (Array.isArray(data.hidden_tags)) {
            saveHiddenTags(data.hidden_tags);
            changed = true;
        }
        if (Array.isArray(data.likes)) {
            replaceLikes(data.likes);
            changed = true;
        }
        if (data.tag_usage && typeof data.tag_usage === 'object') {
            replaceTagUsage(data.tag_usage);
            changed = true;
        }
        return changed;
    }

    window.MangaApp = {
        ensureTranslations: loadTranslations,
        translateTag,
        getTagDescription,
        getTagMetadata,
        getTagAliases,
        resolveTagKeyword,
        resolveTagQueryString,
        getHiddenTags,
        saveHiddenTags,
        isTagHidden,
        toggleLike,
        isLiked,
        getLikedGalleries: () => Array.from(getLikedSet()),
        getHistory,
        addHistoryEntry,
        updateHistoryProgress,
        clearHistory,
        removeHistoryEntries,
        removeHistoryEntry,
        applyThemeToDocument: applyTheme,
        toggleTheme,
        getPreferredTheme,
        getThumbnailUrl,
        buildThumbnailStyle,
        fetchGalleryThumbnail,
        recordTagUsage,
        getTagUsageCounts,
        exportUserData,
        importUserData,
    };
})();
