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

    function toHistoryId(entry) {
        if (!entry) {
            return '';
        }
        const id = entry.gallery_id;
        if (typeof id === 'number' || typeof id === 'string') {
            const str = String(id);
            return str.length > 0 ? str : '';
        }
        return '';
    }

    function createHistoryCard(entry, state, handlers) {
        const container = document.createElement('div');
        container.className = 'history-card';
        const id = toHistoryId(entry);
        if (id) {
            container.dataset.galleryId = id;
        }

        const isSelected = id && state.selected.has(id);
        if (isSelected) {
            container.classList.add('selected');
        }

        const selectControl = document.createElement('label');
        selectControl.className = 'history-select-control';
        selectControl.title = 'この履歴を選択';
        selectControl.setAttribute('aria-label', 'この履歴を選択');
        selectControl.addEventListener('click', (event) => {
            event.stopPropagation();
        });

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'history-select-checkbox';
        checkbox.checked = isSelected;
        checkbox.setAttribute('aria-label', 'この履歴を選択');
        checkbox.addEventListener('click', (event) => {
            event.stopPropagation();
        });
        checkbox.addEventListener('change', (event) => {
            const target = event.target;
            handlers.onToggle(id, Boolean(target.checked));
        });
        selectControl.appendChild(checkbox);

        const indicator = document.createElement('span');
        indicator.className = 'history-select-indicator';
        selectControl.appendChild(indicator);

        container.appendChild(selectControl);

        const anchor = document.createElement('a');
        anchor.href = id ? `/viewer?id=${id}` : '#';
        anchor.className = 'history-card-link';
        if (!id) {
            anchor.addEventListener('click', (event) => {
                event.preventDefault();
            });
        }

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
        container.appendChild(anchor);
        return container;
    }

    function renderHistory(elements, state) {
        const history = MangaApp.getHistory();
        renderContinueSection(elements, history);
        renderHistoryList(elements, history, state);
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

    function renderHistoryList(elements, history, state) {
        if (!elements.historyList || !elements.emptyState) {
            return;
        }
        const list = history || [];
        if (!list.length) {
            elements.historyList.innerHTML = '';
            elements.emptyState.hidden = false;
            state.selected.clear();
            updateSelectionUI(elements, state);
            return;
        }

        const availableIds = new Set(list.map((item) => toHistoryId(item)).filter((id) => id));
        Array.from(state.selected).forEach((id) => {
            if (!availableIds.has(id)) {
                state.selected.delete(id);
            }
        });

        elements.emptyState.hidden = true;
        elements.historyList.innerHTML = '';
        const fragment = document.createDocumentFragment();
        list.forEach((entry) => {
            fragment.appendChild(createHistoryCard(entry, state, {
                onToggle: (id, checked) => {
                    toggleSelection(state, id, checked);
                    updateSelectionUI(elements, state);
                },
            }));
        });
        elements.historyList.appendChild(fragment);
        updateSelectionUI(elements, state);
    }

    function toggleSelection(state, id, checked) {
        if (!id) {
            return;
        }
        if (checked) {
            state.selected.add(id);
        } else {
            state.selected.delete(id);
        }
    }

    function updateSelectionUI(elements, state) {
        if (!elements.historyList) {
            return;
        }
        const cards = elements.historyList.querySelectorAll('.history-card');
        cards.forEach((card) => {
            const id = card.dataset.galleryId || '';
            const selected = id && state.selected.has(id);
            card.classList.toggle('selected', Boolean(selected));
            const checkbox = card.querySelector('.history-select-checkbox');
            if (checkbox) {
                checkbox.checked = Boolean(selected);
            }
        });

        if (elements.deleteSelectedButton) {
            elements.deleteSelectedButton.disabled = state.selected.size === 0;
        }

        if (elements.selectAllButton) {
            const total = cards.length;
            const selectedCount = state.selected.size;
            elements.selectAllButton.disabled = total === 0;
            const shouldDeselect = total > 0 && selectedCount === total;
            elements.selectAllButton.textContent = shouldDeselect ? '選択解除' : '全て選択';
        }
    }

    function updateThemeToggle(themeToggle, theme) {
        if (!themeToggle) {
            return;
        }
        themeToggle.innerHTML = theme === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }

    function normaliseTransferCode(value) {
        return (value || '').toString().trim().toLowerCase();
    }

    function setTransferStatus(elements, message, type = 'info') {
        const status = elements.transferStatus;
        if (!status) {
            return;
        }
        status.textContent = message || '';
        status.classList.remove('success', 'error');
        if (type === 'success') {
            status.classList.add('success');
        } else if (type === 'error') {
            status.classList.add('error');
        }
    }

    function clearTransferResult(elements) {
        if (elements.transferResult) {
            elements.transferResult.hidden = true;
        }
        if (elements.transferCodeDisplay) {
            elements.transferCodeDisplay.textContent = '';
        }
        if (elements.transferUrlDisplay) {
            elements.transferUrlDisplay.textContent = '';
        }
        if (elements.transferExpiryDisplay) {
            elements.transferExpiryDisplay.textContent = '';
        }
        if (elements.copyTransferUrlButton) {
            elements.copyTransferUrlButton.disabled = true;
            elements.copyTransferUrlButton.dataset.url = '';
        }
        setTransferStatus(elements, '');
    }

    function formatExpiry(value) {
        if (!value) {
            return '---';
        }
        try {
            const date = new Date(value);
            if (!Number.isNaN(date.getTime())) {
                return date.toLocaleString('ja-JP');
            }
        } catch (error) {
            /* ignore */
        }
        return value;
    }

    function showTransferResult(elements, data) {
        if (!elements.transferResult) {
            return;
        }
        const code = data?.code || '';
        const restoreUrl = data?.restore_url || (code ? `${window.location.origin}/history?transfer=${encodeURIComponent(code)}` : '');
        const expiry = data?.expires_at || '';

        if (elements.transferCodeDisplay) {
            elements.transferCodeDisplay.textContent = code;
        }
        if (elements.transferUrlDisplay) {
            elements.transferUrlDisplay.textContent = restoreUrl;
        }
        if (elements.transferExpiryDisplay) {
            elements.transferExpiryDisplay.textContent = formatExpiry(expiry);
        }
        if (elements.copyTransferUrlButton) {
            elements.copyTransferUrlButton.disabled = !restoreUrl;
            elements.copyTransferUrlButton.dataset.url = restoreUrl;
        }
        elements.transferResult.hidden = !(code || restoreUrl);
    }

    function toggleTransferLoading(elements, isLoading) {
        const targets = [
            elements.generateTransferButton,
            elements.restoreFromCodeButton,
            elements.copyTransferUrlButton,
        ];
        targets.forEach((button) => {
            if (button) {
                button.disabled = Boolean(isLoading);
            }
        });
        if (!isLoading && elements.copyTransferUrlButton) {
            const hasUrl = Boolean(elements.transferUrlDisplay?.textContent);
            elements.copyTransferUrlButton.disabled = !hasUrl;
        }
    }

    async function copyTransferUrl(elements) {
        if (!elements.copyTransferUrlButton) {
            return;
        }
        const url = elements.transferUrlDisplay?.textContent?.trim();
        if (!url) {
            setTransferStatus(elements, 'コピーできるURLがありません', 'error');
            return;
        }
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(url);
            } else {
                const temp = document.createElement('textarea');
                temp.value = url;
                temp.style.position = 'fixed';
                temp.style.opacity = '0';
                document.body.appendChild(temp);
                temp.select();
                document.execCommand('copy');
                document.body.removeChild(temp);
            }
            setTransferStatus(elements, 'URLをコピーしました', 'success');
        } catch (error) {
            console.error('copy error', error);
            setTransferStatus(elements, 'URLのコピーに失敗しました', 'error');
        }
    }

    async function generateTransferSnapshot(elements) {
        if (typeof MangaApp.exportUserData !== 'function') {
            setTransferStatus(elements, 'エクスポート機能が利用できません', 'error');
            return;
        }
        try {
            toggleTransferLoading(elements, true);
            clearTransferResult(elements);
            setTransferStatus(elements, '引き継ぎデータを作成しています…');
            const payload = MangaApp.exportUserData();
            const response = await fetch('/api/history/snapshots', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                throw new Error('引き継ぎコードの発行に失敗しました');
            }
            const data = await response.json();
            const code = data?.code || '';
            if (elements.transferCodeInput) {
                elements.transferCodeInput.value = code;
            }
            showTransferResult(elements, {
                code,
                restore_url: data?.restore_url,
                expires_at: data?.expires_at,
            });
            setTransferStatus(elements, '引き継ぎコードを発行しました', 'success');
        } catch (error) {
            console.error('snapshot error', error);
            setTransferStatus(elements, error.message || '引き継ぎコードの発行に失敗しました', 'error');
            clearTransferResult(elements);
        } finally {
            toggleTransferLoading(elements, false);
        }
    }

    async function restoreFromCode(elements, state, code, options = {}) {
        const normalised = normaliseTransferCode(code || elements.transferCodeInput?.value);
        if (!normalised) {
            setTransferStatus(elements, 'コードを入力してください', 'error');
            return;
        }
        try {
            toggleTransferLoading(elements, true);
            setTransferStatus(elements, 'データを復元しています…');
            const response = await fetch(`/api/history/snapshots/${encodeURIComponent(normalised)}`);
            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error('指定されたコードが見つかりませんでした');
                }
                if (response.status === 410) {
                    throw new Error('スナップショットの有効期限が切れています');
                }
                throw new Error('データの復元に失敗しました');
            }
            const data = await response.json();
            if (typeof MangaApp.importUserData === 'function') {
                MangaApp.importUserData(data);
            }
            if (elements.transferCodeInput) {
                elements.transferCodeInput.value = normalised;
            }
            showTransferResult(elements, {
                code: normalised,
                restore_url: `${window.location.origin}/history?transfer=${encodeURIComponent(normalised)}`,
                expires_at: data?.expires_at,
            });
            state.selected.clear();
            updateSelectionUI(elements, state);
            renderHistory(elements, state);
            const historyCount = Array.isArray(data?.history) ? data.history.length : 0;
            const message = historyCount
                ? `データを復元しました（履歴 ${historyCount} 件）`
                : 'データを復元しました';
            setTransferStatus(elements, message, 'success');
        } catch (error) {
            console.error('restore error', error);
            setTransferStatus(elements, error.message || 'データの復元に失敗しました', 'error');
        } finally {
            toggleTransferLoading(elements, false);
        }
    }

    function setupTransferHandlers(elements, state) {
        if (elements.generateTransferButton) {
            elements.generateTransferButton.addEventListener('click', () => {
                generateTransferSnapshot(elements);
            });
        }
        if (elements.restoreFromCodeButton) {
            elements.restoreFromCodeButton.addEventListener('click', () => {
                restoreFromCode(elements, state, elements.transferCodeInput?.value);
            });
        }
        if (elements.transferCodeInput) {
            elements.transferCodeInput.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    restoreFromCode(elements, state, elements.transferCodeInput.value);
                }
            });
        }
        if (elements.copyTransferUrlButton) {
            elements.copyTransferUrlButton.addEventListener('click', () => {
                copyTransferUrl(elements);
            });
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        const state = {
            selected: new Set(),
        };

        const elements = {
            continueSection: document.getElementById('continueSection'),
            continueContainer: document.getElementById('continueCardContainer'),
            historyList: document.getElementById('historyList'),
            emptyState: document.getElementById('historyEmpty'),
            clearButton: document.getElementById('historyClearButton'),
            selectAllButton: document.getElementById('historySelectAllButton'),
            deleteSelectedButton: document.getElementById('historyDeleteSelectedButton'),
            themeToggle: document.getElementById('themeToggle'),
            generateTransferButton: document.getElementById('generateTransferButton'),
            copyTransferUrlButton: document.getElementById('copyTransferUrlButton'),
            transferCodeInput: document.getElementById('transferCodeInput'),
            restoreFromCodeButton: document.getElementById('restoreFromCodeButton'),
            transferStatus: document.getElementById('transferStatus'),
            transferResult: document.getElementById('transferResult'),
            transferCodeDisplay: document.getElementById('transferCodeDisplay'),
            transferUrlDisplay: document.getElementById('transferUrlDisplay'),
            transferExpiryDisplay: document.getElementById('transferExpiryDisplay'),
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
                if (confirm('履歴を全て削除しますか？')) {
                    MangaApp.clearHistory();
                    state.selected.clear();
                    updateSelectionUI(elements, state);
                }
            });
        }

        if (elements.selectAllButton) {
            elements.selectAllButton.addEventListener('click', () => {
                const cards = elements.historyList?.querySelectorAll('.history-card') ?? [];
                const total = cards.length;
                const shouldDeselect = total > 0 && state.selected.size === total;
                if (shouldDeselect) {
                    state.selected.clear();
                } else {
                    cards.forEach((card) => {
                        const id = card.dataset.galleryId;
                        if (id) {
                            state.selected.add(id);
                        }
                    });
                }
                updateSelectionUI(elements, state);
            });
        }

        if (elements.deleteSelectedButton) {
            elements.deleteSelectedButton.addEventListener('click', () => {
                if (!state.selected.size) {
                    return;
                }
                const count = state.selected.size;
                const message = count === 1
                    ? '選択した履歴を削除しますか？'
                    : `選択した${count}件の履歴を削除しますか？`;
                if (confirm(message)) {
                    MangaApp.removeHistoryEntries(Array.from(state.selected));
                    state.selected.clear();
                    updateSelectionUI(elements, state);
                }
            });
        }

        const render = () => renderHistory(elements, state);
        render();

        setupTransferHandlers(elements, state);
        clearTransferResult(elements);

        const params = new URLSearchParams(window.location.search);
        const transferCode = params.get('transfer');
        if (transferCode) {
            if (elements.transferCodeInput) {
                elements.transferCodeInput.value = transferCode;
            }
            restoreFromCode(elements, state, transferCode).finally(() => {
                params.delete('transfer');
                const newQuery = params.toString();
                const newUrl = `${window.location.pathname}${newQuery ? `?${newQuery}` : ''}${window.location.hash}`;
                window.history.replaceState({}, '', newUrl);
            });
        }

        document.addEventListener('manga:history-change', render);
        document.addEventListener('manga:theme-change', (event) => {
            updateThemeToggle(elements.themeToggle, event.detail?.theme);
        });
    });
})();
