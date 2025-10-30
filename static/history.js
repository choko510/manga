(function () {
    function formatRelativeTime(timestamp) {
        if (!timestamp) {
            return '';
        }
        const diff = Date.now() - timestamp;
        const seconds = Math.floor(diff / 1000);
        if (seconds < 60) {
            return `${seconds}秒前`;
        }
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) {
            return `${minutes}分前`;
        }
        const hours = Math.floor(minutes / 60);
        if (hours < 24) {
            return `${hours}時間前`;
        }
        const days = Math.floor(hours / 24);
        if (days < 30) {
            return `${days}日前`;
        }
        const months = Math.floor(days / 30);
        if (months < 12) {
            return `${months}か月前`;
        }
        const years = Math.floor(months / 12);
        return `${years}年前`;
    }

    function applyThumbnail(element, entry) {
        if (!element) {
            return;
        }

        const url = typeof MangaApp.getThumbnailUrl === 'function'
            ? MangaApp.getThumbnailUrl(entry)
            : '';

        if (url) {
            element.style.backgroundImage = `url(${url})`;
            return;
        }

        if (!entry?.gallery_id || typeof MangaApp.fetchGalleryThumbnail !== 'function') {
            element.style.backgroundImage = '';
            return;
        }

        const galleryId = entry.gallery_id;
        element.dataset.galleryId = String(galleryId);
        MangaApp.fetchGalleryThumbnail(galleryId)
            .then((fetchedUrl) => {
                if (!fetchedUrl) {
                    return;
                }
                if (element.dataset.galleryId !== String(galleryId)) {
                    return;
                }
                const resolved = fetchedUrl.startsWith('/proxy/') ? fetchedUrl : `/proxy/${fetchedUrl}`;
                element.style.backgroundImage = `url(${resolved})`;
            })
            .catch(() => {
                element.style.backgroundImage = '';
            });
    }

    function calculateProgress(entry) {
        const total = Number.isFinite(entry?.page_count) && entry.page_count > 0 ? entry.page_count : null;
        const last = Number.isFinite(entry?.last_page) ? Math.max(1, entry.last_page) : 1;
        if (!total) {
            return { percent: 0, label: `第${last}ページ` };
        }
        const clampedLast = Math.min(last, total);
        const percent = Math.min(100, Math.round((clampedLast / total) * 100));
        return {
            percent,
            label: `${clampedLast} / ${total} ページ`
        };
    }

    function createContinueCard(entry) {
        const anchor = document.createElement('a');
        anchor.href = `/viewer?id=${entry.gallery_id}`;
        anchor.className = 'continue-card';

        const thumbnail = document.createElement('div');
        thumbnail.className = 'continue-thumbnail';
        applyThumbnail(thumbnail, entry);
        anchor.appendChild(thumbnail);

        const info = document.createElement('div');
        info.className = 'continue-info';

        const title = document.createElement('h3');
        title.className = 'continue-title';
        title.textContent = entry.japanese_title || '無題';
        info.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'continue-meta';
        const progress = calculateProgress(entry);
        meta.textContent = `続き: ${progress.label}`;
        info.appendChild(meta);

        const button = document.createElement('span');
        button.className = 'ghost-button';
        button.textContent = '続きを読む';
        info.appendChild(button);

        anchor.appendChild(info);
        return anchor;
    }

    function createHistoryCard(entry) {
        const anchor = document.createElement('a');
        anchor.href = `/viewer?id=${entry.gallery_id}`;
        anchor.className = 'history-card';

        const thumb = document.createElement('div');
        thumb.className = 'history-thumbnail';
        applyThumbnail(thumb, entry);
        anchor.appendChild(thumb);

        const title = document.createElement('h3');
        title.className = 'history-title';
        title.textContent = entry.japanese_title || '無題';
        anchor.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'history-meta';

        const progress = calculateProgress(entry);
        const progressWrapper = document.createElement('div');
        progressWrapper.className = 'progress-bar';
        const progressInner = document.createElement('span');
        progressInner.style.transform = `scaleX(${progress.percent / 100})`;
        progressWrapper.appendChild(progressInner);
        meta.appendChild(progressWrapper);

        const progressText = document.createElement('span');
        progressText.textContent = entry.completed ? '読了済み' : `進捗: ${progress.label}`;
        meta.appendChild(progressText);

        const updatedText = document.createElement('span');
        updatedText.textContent = `更新: ${formatRelativeTime(entry.updated_at || entry.viewed_at)}`;
        meta.appendChild(updatedText);

        anchor.appendChild(meta);
        return anchor;
    }

    function renderHistory(elements) {
        const history = MangaApp.getHistory();
        renderContinueSection(elements, history);
        renderHistoryList(elements, history);
    }

    function renderContinueSection(elements, history) {
        if (!elements.continueSection || !elements.continueContainer) {
            return;
        }
        const continueEntry = (history || []).find((item) => {
            if (!item) return false;
            const total = Number.isFinite(item.page_count) && item.page_count > 0 ? item.page_count : null;
            if (!total) return false;
            const lastIndex = Number.isFinite(item.last_page_index) ? item.last_page_index : (Number.isFinite(item.last_page) ? item.last_page - 1 : 0);
            return !item.completed && lastIndex < total - 1;
        });

        elements.continueContainer.innerHTML = '';
        if (!continueEntry) {
            elements.continueSection.classList.remove('active');
            return;
        }

        elements.continueSection.classList.add('active');
        elements.continueContainer.appendChild(createContinueCard(continueEntry));
    }

    function renderHistoryList(elements, history) {
        if (!elements.historyList || !elements.emptyState) {
            return;
        }
        const list = history || [];
        if (!list.length) {
            elements.historyList.innerHTML = '';
            elements.emptyState.hidden = false;
            return;
        }

        elements.emptyState.hidden = true;
        elements.historyList.innerHTML = '';
        const fragment = document.createDocumentFragment();
        list.forEach((entry) => {
            fragment.appendChild(createHistoryCard(entry));
        });
        elements.historyList.appendChild(fragment);
    }

    function updateThemeToggle(themeToggle, theme) {
        if (!themeToggle) {
            return;
        }
        themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    document.addEventListener('DOMContentLoaded', () => {
        const elements = {
            continueSection: document.getElementById('continueSection'),
            continueContainer: document.getElementById('continueCardContainer'),
            historyList: document.getElementById('historyList'),
            emptyState: document.getElementById('historyEmpty'),
            clearButton: document.getElementById('historyClearButton'),
            themeToggle: document.getElementById('themeToggle'),
        };

        MangaApp.applyThemeToDocument(document);
        MangaApp.ensureTranslations().catch(() => { /* ignore */ });

        const currentTheme = MangaApp.getPreferredTheme ? MangaApp.getPreferredTheme() : 'light';
        updateThemeToggle(elements.themeToggle, currentTheme);

        if (elements.themeToggle) {
            elements.themeToggle.addEventListener('click', () => {
                const nextTheme = MangaApp.toggleTheme();
                updateThemeToggle(elements.themeToggle, nextTheme);
            });
        }

        if (elements.clearButton) {
            elements.clearButton.addEventListener('click', () => {
                if (confirm('履歴をクリアしますか？')) {
                    MangaApp.clearHistory();
                }
            });
        }

        const render = () => renderHistory(elements);
        render();

        document.addEventListener('manga:history-change', render);
        document.addEventListener('manga:theme-change', (event) => {
            updateThemeToggle(elements.themeToggle, event.detail?.theme);
        });
    });
})();
