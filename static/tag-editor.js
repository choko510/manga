(() => {
    const translationsTableBody = document.getElementById('translationsTableBody');
    const translationsEmptyState = document.getElementById('translationsEmptyState');
    const translationSearchInput = document.getElementById('translationSearch');
    const translationCount = document.getElementById('translationCount');
    const statusBar = document.getElementById('statusBar');
    const currentVersionBadge = document.getElementById('currentVersionBadge');
    const autoSaveStateText = document.getElementById('autoSaveStateText');
    const versionList = document.getElementById('versionList');
    const versionsEmptyState = document.getElementById('versionsEmptyState');
    const versionHistoryModal = document.getElementById('versionHistoryModal');
    const closeVersionHistoryModal = document.getElementById('closeVersionHistoryModal');
    const openVersionHistoryButton = document.getElementById('openVersionHistoryButton');

    const addTranslationButton = document.getElementById('addTranslationButton');
    const autoAddButton = document.getElementById('autoAddButton');
    const saveTranslationsButton = document.getElementById('saveTranslationsButton');
    const refreshVersionsButton = document.getElementById('refreshVersionsButton');

    const AUTO_SAVE_DELAY = 2000;
    const UPDATE_RETRY_DELAY = 5000;

    const state = {
        translations: [],
        popularTagsOffset: 0,
        popularTagsLimit: 10,
        currentVersion: null,
        autoSaveTimer: null,
        isSaving: false,
        pendingChanges: false,
        updateListenerActive: false,
        versions: [],
    };
    function showStatus(message, type = 'success', timeout = 3000) {
        if (!statusBar) return;
        statusBar.textContent = message;
        statusBar.classList.remove('success', 'error', 'visible');
        statusBar.classList.add(type === 'error' ? 'error' : 'success');
        requestAnimationFrame(() => {
            statusBar.classList.add('visible');
        });
        if (timeout > 0) {
            setTimeout(() => {
                statusBar.classList.remove('visible');
            }, timeout);
        }
    }

    function showProgress(current, total, message = '処理中...') {
        if (!statusBar) return;
        const percentage = total > 0 ? Math.round((current / total) * 100) : 0;
        statusBar.textContent = `${message} (${current}/${total} - ${percentage}%)`;
        statusBar.classList.remove('success', 'error', 'visible');
        statusBar.classList.add('visible');
    }

    function setSyncStatus({ tone = 'success', message }) {
        if (autoSaveStateText) {
            autoSaveStateText.textContent = message || '';
        }
        if (currentVersionBadge) {
            currentVersionBadge.classList.remove('success', 'warning', 'error');
            if (tone) {
                currentVersionBadge.classList.add(tone);
            }
        }
    }

    function updateVersionBadge(version) {
        if (currentVersionBadge) {
            currentVersionBadge.textContent = version ? `バージョン: ${version}` : 'バージョン: -';
        }
    }

    function normaliseTag(value) {
        return (value || '').toString().trim().toLowerCase();
    }

    function parseAliases(value) {
        if (!value) return [];
        const seen = new Set();
        const result = [];
        value
            .split(/[\n,]+/)
            .map((item) => item.trim())
            .filter((item) => item)
            .forEach((item) => {
                const norm = normaliseTag(item);
                if (!norm || seen.has(norm)) return;
                seen.add(norm);
                result.push(item);
            });
        return result;
    }

    function markDirty() {
        state.pendingChanges = true;
        setSyncStatus({ tone: 'warning', message: '変更を自動保存します…' });
        scheduleAutoSave();
    }

    function scheduleAutoSave() {
        if (state.autoSaveTimer) {
            clearTimeout(state.autoSaveTimer);
        }
        state.autoSaveTimer = window.setTimeout(() => {
            state.autoSaveTimer = null;
            if (state.pendingChanges) {
                saveTranslations({ auto: true });
            }
        }, AUTO_SAVE_DELAY);
    }

    function updateTranslationCount() {
        if (!translationCount) return;
        const visibleRows = translationsTableBody ? translationsTableBody.querySelectorAll('tr:not([hidden])').length : 0;
        translationCount.textContent = `${visibleRows} 件表示 / 全 ${state.translations.length} 件`;
    }

    function createTranslationRow(entry = { tag: '', translation: '', description: '', aliases: [] }) {
        const tr = document.createElement('tr');
        tr.dataset.tag = entry.tag || '';

        const tagCell = document.createElement('td');
        const tagInput = document.createElement('input');
        tagInput.type = 'text';
        tagInput.className = 'table-input';
        tagInput.value = entry.tag || '';
        tagInput.placeholder = 'タグ (英語)';
        tagInput.addEventListener('input', () => {
            tr.dataset.tag = tagInput.value;
            entry.tag = tagInput.value;
            markDirty();
        });
        tagCell.appendChild(tagInput);

        const translationCell = document.createElement('td');
        const translationInput = document.createElement('input');
        translationInput.type = 'text';
        translationInput.className = 'table-input';
        translationInput.value = entry.translation || '';
        translationInput.placeholder = '翻訳 (日本語)';
        translationInput.addEventListener('input', () => {
            entry.translation = translationInput.value;
            markDirty();
        });
        translationCell.appendChild(translationInput);

        // 説明列
        const descriptionCell = document.createElement('td');
        const descriptionInput = document.createElement('input');
        descriptionInput.type = 'text';
        descriptionInput.className = 'table-input';
        descriptionInput.value = entry.description || '';
        descriptionInput.placeholder = '説明 (任意)';
        descriptionInput.addEventListener('input', () => {
            entry.description = descriptionInput.value;
            markDirty();
        });
        descriptionCell.appendChild(descriptionInput);

        // 優先度列
        const priorityCell = document.createElement('td');
        const priorityInput = document.createElement('input');
        priorityInput.type = 'number';
        priorityInput.className = 'table-input';
        priorityInput.value = typeof entry.priority === 'number' ? entry.priority : 0;
        priorityInput.placeholder = '0';
        priorityInput.style.minWidth = '60px';
        priorityInput.addEventListener('input', () => {
            const val = parseInt(priorityInput.value, 10);
            entry.priority = isNaN(val) ? 0 : val;
            markDirty();
        });
        priorityCell.appendChild(priorityInput);

        // あいまい検索キーワード列
        const aliasesCell = document.createElement('td');
        const aliasesInput = document.createElement('input');
        aliasesInput.type = 'text';
        aliasesInput.className = 'table-input';
        aliasesInput.value = Array.isArray(entry.aliases) ? entry.aliases.join(', ') : '';
        aliasesInput.placeholder = '例: 代替タグ, ニックネーム';
        aliasesInput.addEventListener('input', () => {
            entry.aliases = parseAliases(aliasesInput.value);
            markDirty();
        });
        aliasesCell.appendChild(aliasesInput);

        const actionCell = document.createElement('td');
        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'button danger';
        deleteButton.textContent = '削除';
        deleteButton.addEventListener('click', () => {
            tr.remove();
            state.translations = state.translations.filter((item) => item !== entry);
            updateEmptyStates();
            updateTranslationCount();
            markDirty();
        });
        actionCell.appendChild(deleteButton);

        tr.appendChild(tagCell);
        tr.appendChild(translationCell);
        tr.appendChild(descriptionCell);
        tr.appendChild(priorityCell);
        tr.appendChild(aliasesCell);
        tr.appendChild(actionCell);

        tr._inputs = { tagInput, translationInput, aliasesInput, descriptionInput, priorityInput };
        return tr;
    }

    function renderTranslations(translations) {
        if (!translationsTableBody) return;
        translationsTableBody.innerHTML = '';
        const sorted = [...translations].sort((a, b) => {
            const aUntranslated = !(a.translation || '').toString().trim();
            const bUntranslated = !(b.translation || '').toString().trim();

            if (aUntranslated !== bUntranslated) {
                return aUntranslated ? -1 : 1;
            }

            return a.tag.localeCompare(b.tag);
        });
        sorted.forEach((entry) => {
            const row = createTranslationRow(entry);
            translationsTableBody.appendChild(row);
        });
        updateEmptyStates();
        updateTranslationCount();
        filterTranslations();
    }

    function formatTimestamp(value) {
        if (!value) return '';
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) {
            return value;
        }
        return parsed.toLocaleString('ja-JP', { hour12: false });
    }

    function formatVersionReason(entry) {
        if (!entry) return '';
        const reason = entry.reason || '';
        if (reason === 'auto-save') return '自動保存';
        if (reason === 'manual') return '手動保存';
        if (reason === 'rollback') return 'ロールバック';
        if (reason === 'initial') return '初期状態';
        return reason;
    }

    function renderVersions(versions) {
        if (!versionList) return;
        versionList.innerHTML = '';
        if (!Array.isArray(versions) || versions.length === 0) {
            if (versionsEmptyState) {
                versionsEmptyState.hidden = false;
            }
            return;
        }
        if (versionsEmptyState) {
            versionsEmptyState.hidden = true;
        }
        versions.forEach((entry) => {
            const item = document.createElement('div');
            item.className = 'version-item';

            const meta = document.createElement('div');
            meta.className = 'version-meta';
            const title = document.createElement('strong');
            title.textContent = entry.version || '-';
            if (entry.version && entry.version === state.currentVersion) {
                title.textContent += '（現在）';
            }
            const subtitle = document.createElement('span');
            const timestamp = formatTimestamp(entry.created_at);
            const details = [];
            const reasonLabel = formatVersionReason(entry);
            if (reasonLabel) {
                details.push(`種別: ${reasonLabel}`);
            }
            if (entry.restored_from) {
                details.push(`復元元: ${entry.restored_from}`);
            }
            if (entry.parent_version && entry.parent_version !== entry.restored_from) {
                details.push(`前バージョン: ${entry.parent_version}`);
            }
            subtitle.textContent = `保存: ${timestamp || '---'}${details.length ? ` / ${details.join(' / ')}` : ''}`;

            meta.appendChild(title);
            meta.appendChild(subtitle);

            const actions = document.createElement('div');
            actions.className = 'version-actions';

            const rollbackButton = document.createElement('button');
            rollbackButton.type = 'button';
            rollbackButton.className = 'button secondary';
            if (entry.version === state.currentVersion) {
                rollbackButton.textContent = '現在のバージョン';
                rollbackButton.disabled = true;
            } else {
                rollbackButton.textContent = 'このバージョンに戻す';
                rollbackButton.addEventListener('click', () => rollbackToVersion(entry.version));
            }
            actions.appendChild(rollbackButton);

            item.appendChild(meta);
            item.appendChild(actions);
            versionList.appendChild(item);
        });
    }

    function openVersionHistory() {
        if (!versionHistoryModal) return;
        versionHistoryModal.hidden = false;
        versionHistoryModal.style.display = 'flex';
        versionHistoryModal.setAttribute('aria-modal', 'true');
        versionHistoryModal.setAttribute('role', 'dialog');
        // モーダルを開くたびに最新状態を取得
        refreshVersions({ silent: true }).catch(() => { });
    }

    function closeVersionHistory() {
        if (!versionHistoryModal) return;
        versionHistoryModal.hidden = true;
        versionHistoryModal.style.display = 'none';
        versionHistoryModal.removeAttribute('aria-modal');
        versionHistoryModal.removeAttribute('role');
    }

    async function refreshVersions({ silent = false } = {}) {
        if (!versionList) return;
        try {
            const response = await fetch('/api/tag-translations/versions');
            if (!response.ok) {
                throw new Error('バージョン履歴の取得に失敗しました');
            }
            const data = await response.json();
            state.versions = Array.isArray(data.versions) ? data.versions : [];
            renderVersions(state.versions);
        } catch (error) {
            console.error(error);
            if (!silent) {
                showStatus(error.message || 'バージョン履歴の取得に失敗しました', 'error', 5000);
            }
        }
    }

    async function rollbackToVersion(version) {
        if (!version) return;
        if (!window.confirm(`バージョン ${version} にロールバックします。現在の変更は失われます。よろしいですか？`)) {
            return;
        }
        setSyncStatus({ tone: 'warning', message: 'ロールバック中…' });
        try {
            const response = await fetch('/api/tag-translations/rollback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ version }),
            });
            if (!response.ok) {
                throw new Error('ロールバックに失敗しました');
            }
            const data = await response.json();
            state.translations = mapTranslationsPayload(data.translations || {});
            state.currentVersion = data.version || version;
            updateVersionBadge(state.currentVersion);
            state.pendingChanges = false;
            renderTranslations(state.translations);
            setSyncStatus({ tone: 'success', message: '指定したバージョンを復元しました' });
            showStatus(`バージョン ${version} を復元しました`);
            await refreshVersions();
        } catch (error) {
            console.error(error);
            setSyncStatus({ tone: 'error', message: error.message || 'ロールバックに失敗しました' });
            showStatus(error.message || 'ロールバックに失敗しました', 'error', 6000);
        }
    }

    function setupVersionHistoryModalEvents() {
        if (!versionHistoryModal) {
            return;
        }

        // 初期状態は必ず非表示にしておく（テンプレ上の hidden が無視されても防御）
        versionHistoryModal.hidden = true;
        versionHistoryModal.style.display = 'none';

        if (openVersionHistoryButton) {
            openVersionHistoryButton.addEventListener('click', () => {
                openVersionHistory();
            });
        }

        if (closeVersionHistoryModal) {
            closeVersionHistoryModal.addEventListener('click', () => {
                closeVersionHistory();
            });
        }

        // 背景クリックで閉じる
        versionHistoryModal.addEventListener('click', (event) => {
            if (event.target === versionHistoryModal) {
                closeVersionHistory();
            }
        });

        // ESCキーで閉じる
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && !versionHistoryModal.hidden) {
                closeVersionHistory();
            }
        });
    }

    // 初期化: バージョン履歴モーダルのイベントをセットアップ
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupVersionHistoryModalEvents);
    } else {
        setupVersionHistoryModalEvents();
    }

    async function startUpdateListener() {
        if (state.updateListenerActive) return;
        state.updateListenerActive = true;
        while (true) {
            try {
                const params = new URLSearchParams();
                if (state.currentVersion) {
                    params.set('since', state.currentVersion);
                }
                const query = params.toString();
                const url = query ? `/api/tag-translations/updates?${query}` : '/api/tag-translations/updates';
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error('更新の監視に失敗しました');
                }
                const data = await response.json();
                const { version, changed } = data;
                if (changed && version && version !== state.currentVersion) {
                    state.currentVersion = version;
                    updateVersionBadge(state.currentVersion);
                    setSyncStatus({ tone: 'warning', message: '他のユーザーの変更を取得しています…' });
                    await loadData({ withVersions: true });
                    setSyncStatus({ tone: 'success', message: '最新の状態です' });
                    showStatus('他のユーザーの変更を反映しました');
                }
            } catch (error) {
                console.error('update listener error', error);
                setSyncStatus({ tone: 'warning', message: '変更の監視を再試行しています…' });
                await new Promise((resolve) => setTimeout(resolve, UPDATE_RETRY_DELAY));
            }
        }
    }



    function updateEmptyStates() {
        if (translationsEmptyState) {
            const hasRows = translationsTableBody && translationsTableBody.querySelector('tr');
            translationsEmptyState.hidden = !!hasRows;
        }
    }

    function filterTranslations() {
        const keyword = (translationSearchInput?.value || '').toLowerCase();
        if (!translationsTableBody) return;
        translationsTableBody.querySelectorAll('tr').forEach((row) => {
            const { tagInput, translationInput, aliasesInput, descriptionInput } = row._inputs || {};
            if (!tagInput || !translationInput) return;
            const tagText = tagInput.value.toLowerCase();
            const translationText = translationInput.value.toLowerCase();
            const aliasesText = aliasesInput ? aliasesInput.value.toLowerCase() : '';
            const descriptionText = descriptionInput ? descriptionInput.value.toLowerCase() : '';
            const match =
                !keyword ||
                tagText.includes(keyword) ||
                translationText.includes(keyword) ||
                descriptionText.includes(keyword) ||
                aliasesText.includes(keyword);
            row.hidden = !match;
        });
        updateTranslationCount();
    }

    function mapTranslationsPayload(map) {
        if (!map || typeof map !== 'object') return [];
        return Object.entries(map).map(([tag, value]) => {
            const entry = value && typeof value === 'object' ? value : {};
            const translation = typeof entry.translation === 'string' ? entry.translation : typeof value === 'string' ? value : '';
            const description = typeof entry.description === 'string' ? entry.description : '';
            const priority = typeof entry.priority === 'number' ? entry.priority : 0;
            const aliases = Array.isArray(entry.aliases) ? entry.aliases : [];
            return { tag, translation, description, priority, aliases };
        });
    }

    function validateDuplicateTagsPayload(payload) {
        if (!payload || typeof payload !== 'object') return;

        const seen = new Map();
        for (const [tag] of Object.entries(payload)) {
            const rawTag = (tag || '').trim();
            if (!rawTag) continue;
            const norm = normaliseTag(rawTag);
            if (!norm) continue;

            if (seen.has(norm)) {
                const first = seen.get(norm);
                // 同じキー名での衝突は通常起こらないはずだが、安全側でチェック
                throw new Error(`重複しているタグがあります: "${rawTag}" と "${first}"`);
            }
            seen.set(norm, rawTag);
        }
    }

    async function loadData({ withVersions = true } = {}) {
        try {
            const translationsResponse = await fetch('/api/tag-translations');
            if (!translationsResponse.ok) {
                throw new Error('翻訳の取得に失敗しました');
            }
            const translationsData = await translationsResponse.json();
            state.translations = mapTranslationsPayload(translationsData.translations || {});
            state.currentVersion = translationsData.version || null;
            updateVersionBadge(state.currentVersion);
            state.pendingChanges = false;
            setSyncStatus({ tone: 'success', message: '最新の状態です' });
            renderTranslations(state.translations);
            if (withVersions) {
                await refreshVersions({ silent: true });
            }
        } catch (error) {
            console.error(error);
            showStatus(error.message || 'データの読み込みに失敗しました', 'error', 5000);
        }
    }

    function collectTranslations() {
        if (!translationsTableBody) return {};
        const rows = Array.from(translationsTableBody.querySelectorAll('tr'));
        const result = {};
        const duplicates = new Map();
        const aliasDuplicates = new Map();
        for (const row of rows) {
            const { tagInput, translationInput, aliasesInput, descriptionInput, priorityInput } = row._inputs || {};
            if (!tagInput || !translationInput) continue;
            const rawTag = tagInput.value.trim();
            const rawTranslation = translationInput.value.trim();
            const rawDescription = descriptionInput ? descriptionInput.value.trim() : '';
            const rawPriority = priorityInput ? parseInt(priorityInput.value, 10) : 0;
            const rawAliases = aliasesInput ? parseAliases(aliasesInput.value) : [];
            if (!rawTag) {
                continue;
            }
            const normalised = normaliseTag(rawTag);
            if (duplicates.has(normalised)) {
                throw new Error(`重複しているタグがあります: "${rawTag}" と "${duplicates.get(normalised)}"`);
            }
            duplicates.set(normalised, rawTag);

            rawAliases.forEach((alias) => {
                const aliasNorm = normaliseTag(alias);
                if (!aliasNorm) return;
                if (aliasDuplicates.has(aliasNorm) && aliasDuplicates.get(aliasNorm) !== rawTag) {
                    throw new Error(`検索キーワードが重複しています: "${alias}"`);
                }
                aliasDuplicates.set(aliasNorm, rawTag);
            });

            const entry = {
                translation: rawTranslation,
                description: rawDescription,
                priority: isNaN(rawPriority) ? 0 : rawPriority,
            };
            if (rawAliases.length) {
                entry.aliases = rawAliases;
            }
            result[rawTag] = entry;
        }
        return result;
    }



    async function saveTranslations({ auto = false } = {}) {
        if (state.autoSaveTimer) {
            clearTimeout(state.autoSaveTimer);
            state.autoSaveTimer = null;
        }
        if (state.isSaving) {
            state.pendingChanges = true;
            return;
        }
        let translations;
        try {
            translations = collectTranslations();
        } catch (error) {
            console.error(error);
            setSyncStatus({ tone: 'error', message: error.message || '入力内容を確認してください' });
            showStatus(error.message || '入力内容を確認してください', 'error', 6000);
            return;
        }

        state.isSaving = true;
        if (auto) {
            setSyncStatus({ tone: 'warning', message: '自動保存中…' });
        } else {
            setSyncStatus({ tone: 'warning', message: '保存中…' });
            showStatus('保存中…', 'success', 0);
        }

        try {
            const response = await fetch('/api/tag-translations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    translations,
                    base_version: state.currentVersion,
                    auto_save: auto,
                }),
            });

            if (response.status === 409) {
                const payload = await response.json().catch(() => ({}));
                state.pendingChanges = false;
                state.translations = mapTranslationsPayload(payload.translations || {});
                renderTranslations(state.translations);
                state.currentVersion = payload.version || state.currentVersion;
                updateVersionBadge(state.currentVersion);
                setSyncStatus({ tone: 'warning', message: '他のユーザーの変更を反映しました' });
                showStatus('他のユーザーによる変更が検出されたため内容を更新しました', 'error', 6000);
                await refreshVersions({ silent: true });
                return;
            }

            if (!response.ok) {
                throw new Error('翻訳の保存に失敗しました');
            }

            const data = await response.json();
            state.currentVersion = data.version || state.currentVersion;
            updateVersionBadge(state.currentVersion);
            state.pendingChanges = false;
            if (auto) {
                setSyncStatus({ tone: 'success', message: '最新の状態です' });
                await refreshVersions();
            } else {
                setSyncStatus({ tone: 'success', message: '保存が完了しました' });
                showStatus('翻訳を保存しました');
                await loadData({ withVersions: true });
            }
        } catch (error) {
            console.error(error);
            state.pendingChanges = true;
            setSyncStatus({ tone: 'error', message: error.message || '保存に失敗しました' });
            showStatus(error.message || '翻訳の保存に失敗しました', 'error', 6000);
            scheduleAutoSave();
        } finally {
            state.isSaving = false;
        }
    }



    addTranslationButton?.addEventListener('click', () => {
        const entry = { tag: '', translation: '', description: '', priority: 0, aliases: [] };
        state.translations.push(entry);
        const row = createTranslationRow(entry);
        translationsTableBody?.prepend(row);
        row.querySelector('input')?.focus();
        updateEmptyStates();
        updateTranslationCount();
        markDirty();
    });

    // データベースからタグを自動的に追加する機能
    autoAddButton?.addEventListener('click', async () => {
        try {
            showStatus('データベースからタグを取得中...', 'success', 0);

            // 人気タグを取得（既存の翻訳を除外、オフセットとリミットを指定）
            const response = await fetch(`/api/popular-tags?limit=${state.popularTagsLimit}&offset=${state.popularTagsOffset}&exclude_existing=true`);
            if (!response.ok) {
                throw new Error('人気タグの取得に失敗しました');
            }

            const data = await response.json();
            const popularTags = data.tags || [];

            if (popularTags.length === 0) {
                showStatus('追加可能なタグがありません', 'error', 3000);
                return;
            }

            // 既存のタグと重複チェック
            const existingTags = new Set(state.translations.map(t => normaliseTag(t.tag)));

            // 重複しないタグのみをフィルタリング
            const newTags = popularTags.filter(tagInfo => !existingTags.has(normaliseTag(tagInfo.tag)));

            if (newTags.length === 0) {
                showStatus('すべてのタグが既に登録されています', 'error', 3000);
                return;
            }

            // 新しいタグを翻訳リストに追加（進捗表示付き）
            let addedCount = 0;
            showProgress(0, newTags.length, 'タグを追加中');

            for (let i = 0; i < newTags.length; i++) {
                const tagInfo = newTags[i];
                const entry = {
                    tag: tagInfo.tag,
                    translation: '',
                    description: '',
                    priority: 0,
                    aliases: [],
                };
                state.translations.push(entry);
                const row = createTranslationRow(entry);
                translationsTableBody?.prepend(row);
                addedCount++;

                // 進捗を更新
                showProgress(i + 1, newTags.length, 'タグを追加中');

                // 少し遅延を入れてUIが更新されるようにする
                await new Promise(resolve => setTimeout(resolve, 10));
            }

            // オフセットを更新（次回取得時に使用）
            state.popularTagsOffset += state.popularTagsLimit;

            updateEmptyStates();
            updateTranslationCount();
            showStatus(`${addedCount}件のタグを追加しました`, 'success', 3000);
            if (addedCount > 0) {
                markDirty();
            }

        } catch (error) {
            console.error(error);
            showStatus(error.message || 'タグの追加に失敗しました', 'error', 5000);
        }
    });

    saveTranslationsButton?.addEventListener('click', () => {
        saveTranslations({ auto: false });
    });

    refreshVersionsButton?.addEventListener('click', () => {
        refreshVersions();
    });



    translationSearchInput?.addEventListener('input', () => {
        filterTranslations();
    });

    loadData()
        .then(() => {
            startUpdateListener();
        })
        .catch(() => {
            startUpdateListener();
        });
})();
