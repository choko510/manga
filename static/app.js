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

    function getHistory() {
        try {
            const raw = localStorage.getItem(STORAGE_KEYS.history);
            if (!raw) {
                return [];
            }
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                return parsed;
            }
            return [];
        } catch (error) {
            return [];
        }
    }

    function saveHistory(entries) {
        localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(entries));
        document.dispatchEvent(new CustomEvent('manga:history-change', { detail: { history: entries } }));
    }

    function addHistoryEntry(entry) {
        if (!entry || !entry.gallery_id) {
            return;
        }
        const history = getHistory().filter((item) => item.gallery_id !== entry.gallery_id);
        history.unshift({
            gallery_id: entry.gallery_id,
            japanese_title: entry.japanese_title || '',
            image_urls: entry.image_urls || [],
            viewed_at: Date.now(),
            page_count: entry.page_count || 0
        });
        const limit = 20;
        const trimmed = history.slice(0, limit);
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
        clearHistory,
        applyThemeToDocument: applyTheme,
        toggleTheme,
        getPreferredTheme,
    };
})();
