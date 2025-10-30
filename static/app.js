(function () {
    const STORAGE_KEYS = {
        hiddenTags: 'manga_hidden_tags',
        liked: 'manga_liked_galleries',
        history: 'manga_view_history',
        theme: 'manga_theme'
    };

    let translationPromise = null;
    let translations = {};
    let hiddenTagSet = null;
    let likedSet = null;
    const galleryThumbnailCache = new Map();

    function normaliseTag(tag) {
        return (tag || '').toString().trim().toLowerCase();
    }

    async function loadTranslations() {
        if (translationPromise) {
            return translationPromise;
        }
        translationPromise = fetch('/static/tag-translations.json')
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to load translations');
                }
                return response.json();
            })
            .then((data) => {
                translations = {};
                Object.keys(data || {}).forEach((key) => {
                    translations[normaliseTag(key)] = data[key];
                });
                return translations;
            })
            .catch(() => {
                translations = {};
                return translations;
            });
        return translationPromise;
    }

    function translateTag(tag) {
        if (!tag) return '';
        const key = normaliseTag(tag);
        if (translations[key]) {
            return translations[key];
        }
        return tag;
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

    window.MangaApp = {
        ensureTranslations: loadTranslations,
        translateTag,
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
        applyThemeToDocument: applyTheme,
        toggleTheme,
        getPreferredTheme,
        getThumbnailUrl,
        buildThumbnailStyle,
        fetchGalleryThumbnail,
    };
})();
