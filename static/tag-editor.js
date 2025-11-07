(() => {
    const translationsTableBody = document.getElementById('translationsTableBody');
    const translationsEmptyState = document.getElementById('translationsEmptyState');
    const categoriesContainer = document.getElementById('categoriesContainer');
    const categoriesEmptyState = document.getElementById('categoriesEmptyState');
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
    const addCategoryButton = document.getElementById('addCategoryButton');
    const saveCategoriesButton = document.getElementById('saveCategoriesButton');
    const refreshVersionsButton = document.getElementById('refreshVersionsButton');

    const AUTO_SAVE_DELAY = 2000;
    const UPDATE_RETRY_DELAY = 5000;

    const state = {
        translations: [],
        categories: [],
        popularTagsOffset: 0, // 追加: 人気タグの取得オフセット
        popularTagsLimit: 10, // 追加: 人気タグの取得数
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
        tr.appendChild(aliasesCell);
        tr.appendChild(actionCell);

        tr._inputs = { tagInput, translationInput, aliasesInput, descriptionInput };
        return tr;
    }

    function renderTranslations(translations) {
        if (!translationsTableBody) return;
        translationsTableBody.innerHTML = '';
        const sorted = [...translations].sort((a, b) => a.tag.localeCompare(b.tag));
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
        refreshVersions({ silent: true }).catch(() => {});
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

    function createCategoryCard(category = { id: '', label: '', tags: [] }) {
        const card = document.createElement('div');
        card.className = 'category-card';
        card.dataset.categoryId = category.id || '';

        // ドラッグで並び替え可能にする
        card.draggable = true;

        card.addEventListener('dragstart', (event) => {
            card.classList.add('dragging');
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', card.dataset.categoryId || '');
        });

        card.addEventListener('dragend', () => {
            card.classList.remove('dragging');
            syncCategoryOrderFromDOM();
        });

        card.addEventListener('dragover', (event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = 'move';
            const dragging = categoriesContainer.querySelector('.category-card.dragging');
            if (!dragging || dragging === card) return;

            const bounding = card.getBoundingClientRect();
            const offset = event.clientY - bounding.top;
            const shouldInsertAfter = offset > bounding.height / 2;

            if (shouldInsertAfter) {
                if (card.nextSibling !== dragging) {
                    card.after(dragging);
                }
            } else {
                if (card.previousSibling !== dragging) {
                    card.before(dragging);
                }
            }
        });

        card.addEventListener('drop', (event) => {
            event.preventDefault();
            const dragging = categoriesContainer.querySelector('.category-card.dragging');
            if (dragging && dragging !== card) {
                // dragover 内ですでに DOM の位置は調整されているので、ここでは順序同期のみ
                dragging.classList.remove('dragging');
                syncCategoryOrderFromDOM();
            }
        });

        const header = document.createElement('div');
        header.className = 'category-header';

        const idInput = document.createElement('input');
        idInput.type = 'text';
        idInput.className = 'table-input id-input';
        idInput.placeholder = 'ID (英数)';
        idInput.value = category.id || '';

        const labelInput = document.createElement('input');
        labelInput.type = 'text';
        labelInput.className = 'table-input';
        labelInput.placeholder = '表示名';
        labelInput.value = category.label || '';

        header.appendChild(idInput);
        header.appendChild(labelInput);

        // タグ入力エリアを改善
        const tagInputContainer = document.createElement('div');
        tagInputContainer.className = 'tag-input-container';
        
        const tagInput = document.createElement('input');
        tagInput.type = 'text';
        tagInput.className = 'table-input tag-input';
        tagInput.placeholder = 'タグを入力（オートコンプリート対応）';
        
        const addTagButton = document.createElement('button');
        addTagButton.type = 'button';
        addTagButton.className = 'button secondary add-tag-btn';
        addTagButton.textContent = '追加';
        
        tagInputContainer.appendChild(tagInput);
        tagInputContainer.appendChild(addTagButton);

        // タグ表示エリア
        const tagDisplayContainer = document.createElement('div');
        tagDisplayContainer.className = 'tag-display-container';
        
        // 既存のタグを表示
        const existingTags = Array.isArray(category.tags) ? category.tags : [];
        existingTags.forEach(tag => {
            const tagElement = createTagElement(tag);
            tagDisplayContainer.appendChild(tagElement);
        });

        const buttonRow = document.createElement('div');
        buttonRow.className = 'section-actions';

        // 上下移動ボタン
        const moveUpButton = document.createElement('button');
        moveUpButton.type = 'button';
        moveUpButton.className = 'button secondary';
        moveUpButton.textContent = '↑ 上へ';
        moveUpButton.addEventListener('click', () => {
            const prev = card.previousElementSibling;
            if (prev && prev.classList.contains('category-card')) {
                categoriesContainer.insertBefore(card, prev);
                syncCategoryOrderFromDOM();
            }
        });

        const moveDownButton = document.createElement('button');
        moveDownButton.type = 'button';
        moveDownButton.className = 'button secondary';
        moveDownButton.textContent = '↓ 下へ';
        moveDownButton.addEventListener('click', () => {
            const next = card.nextElementSibling;
            if (next && next.classList.contains('category-card')) {
                categoriesContainer.insertBefore(next, card);
                syncCategoryOrderFromDOM();
            }
        });

        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'button danger';
        deleteButton.textContent = 'カテゴリを削除';
        deleteButton.addEventListener('click', () => {
            card.remove();
            state.categories = state.categories.filter((item) => item !== category);
            updateEmptyStates();
        });
        buttonRow.appendChild(deleteButton);

        card.appendChild(header);
        card.appendChild(tagInputContainer);
        card.appendChild(tagDisplayContainer);
        card.appendChild(buttonRow);

        // オートコンプリート機能
        setupTagAutocomplete(tagInput, tagDisplayContainer);
        
        // 追加ボタンのイベント
        addTagButton.addEventListener('click', () => {
            const tagValue = tagInput.value.trim();
            if (tagValue) {
                addTagToCategory(tagValue, tagDisplayContainer, category);
                tagInput.value = '';
            }
        });
        
        // Enterキーでタグ追加
        tagInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const tagValue = tagInput.value.trim();
                if (tagValue) {
                    addTagToCategory(tagValue, tagDisplayContainer, category);
                    tagInput.value = '';
                }
            }
        });

        card._inputs = { idInput, labelInput, tagInput, tagDisplayContainer };
        return card;
    }

    function getTagMetadata(tag) {
        const normalised = normaliseTag(tag);
        const entry = state.translations.find((t) => normaliseTag(t.tag) === normalised);
        if (entry) {
            return {
                translation: entry.translation || '',
                description: entry.description || '',
            };
        }
        return { translation: '', description: '' };
    }

    function createTagElement(tag) {
        const tagElement = document.createElement('div');
        tagElement.className = 'tag-element';

        const tagText = document.createElement('span');
        tagText.className = 'tag-text';

        const metadata = getTagMetadata(tag);
        const translation = metadata.translation;
        const description = metadata.description;
        if (translation) {
            tagText.textContent = `${translation} (${tag})`;
            tagElement.title = description ? `${tag} - ${translation}\n${description}` : `${tag} - ${translation}`;
        } else {
            tagText.textContent = tag;
            tagElement.title = description ? `${tag}\n${description}` : tag;
        }
        
        const removeButton = document.createElement('button');
        removeButton.type = 'button';
        removeButton.className = 'tag-remove-btn';
        removeButton.textContent = '×';
        removeButton.addEventListener('click', () => {
            tagElement.remove();
        });
        
        tagElement.appendChild(tagText);
        tagElement.appendChild(removeButton);
        return tagElement;
    }

    function addTagToCategory(tag, container, category) {
        // 重複チェック
        const existingTags = Array.from(container.querySelectorAll('.tag-text')).map(el =>
            el.textContent.split(' (')[0] // 翻訳がある場合は翻訳部分のみ取得
        );

        const metadata = getTagMetadata(tag);
        const translation = metadata.translation;
        const displayText = translation ? `${translation} (${tag})` : tag;
        
        if (existingTags.includes(displayText)) {
            showStatus('このタグは既に追加されています', 'error', 3000);
            return;
        }
        
        const tagElement = createTagElement(tag);
        container.appendChild(tagElement);
        
        // カテゴリのタグリストを更新
        if (!category.tags) {
            category.tags = [];
        }
        category.tags.push(tag);
    }

    function setupTagAutocomplete(input, container) {
        let currentFocus = -1;
        
        input.addEventListener('input', function() {
            const value = this.value.trim();
            closeAllLists();

            if (!value) return;

            // 利用可能なタグ（既に追加されているものを除く）
            const existingTags = Array.from(container.querySelectorAll('.tag-text')).map(el =>
                el.textContent.split(' (')[1]?.replace(')', '') || el.textContent
            );

            const searchLower = value.toLowerCase();
            const availableTags = state.translations
                .filter(t => !existingTags.includes(t.tag))
                .filter(t => {
                    const translation = (t.translation || '').toLowerCase();
                    const description = (t.description || '').toLowerCase();
                    return t.tag.toLowerCase().includes(searchLower) ||
                        translation.includes(searchLower) ||
                        description.includes(searchLower);
                })
                .slice(0, 10); // 最大10件まで表示
                
            if (availableTags.length === 0) return;
            
            const listContainer = document.createElement('div');
            listContainer.setAttribute('id', 'autocomplete-list');
            listContainer.className = 'autocomplete-items';
            
            availableTags.forEach((item, index) => {
                const itemElement = document.createElement('div');
                itemElement.className = 'autocomplete-item';

                const primary = item.translation || item.tag;
                const description = item.description ? `<div class="autocomplete-desc">${item.description}</div>` : '';
                itemElement.innerHTML = `<strong>${primary}</strong> (${item.tag})${description}`;
                
                itemElement.addEventListener('click', function() {
                    addTagToCategory(item.tag, container, container.closest('.category-card')._category);
                    input.value = '';
                    closeAllLists();
                });
                
                itemElement.addEventListener('mouseover', function() {
                    currentFocus = index;
                    addActive(listContainer.getElementsByClassName('autocomplete-item'));
                });
                
                listContainer.appendChild(itemElement);
            });
            
            this.parentNode.appendChild(listContainer);
        });
        
        input.addEventListener('keydown', function(e) {
            const listContainer = document.getElementById('autocomplete-list');
            if (!listContainer) return;
            
            const items = listContainer.getElementsByClassName('autocomplete-item');
            if (items.length === 0) return;
            
            if (e.keyCode === 40) { // 下矢印
                currentFocus++;
                addActive(items);
            } else if (e.keyCode === 38) { // 上矢印
                currentFocus--;
                addActive(items);
            } else if (e.keyCode === 13) { // Enter
                e.preventDefault();
                if (currentFocus > -1) {
                    items[currentFocus].click();
                }
            } else if (e.keyCode === 27) { // Esc
                closeAllLists();
            }
        });
        
        function addActive(items) {
            if (!items) return;
            removeActive(items);
            if (currentFocus >= 0 && currentFocus < items.length) {
                items[currentFocus].classList.add('autocomplete-active');
            }
        }
        
        function removeActive(items) {
            for (let i = 0; i < items.length; i++) {
                items[i].classList.remove('autocomplete-active');
            }
        }
        
        function closeAllLists() {
            const lists = document.getElementsByClassName('autocomplete-items');
            for (let i = 0; i < lists.length; i++) {
                lists[i].parentNode.removeChild(lists[i]);
            }
            currentFocus = -1;
        }
        
        // クリックでリストを閉じる
        document.addEventListener('click', function(e) {
            if (e.target !== input) {
                closeAllLists();
            }
        });
    }

    function renderCategories(categories) {
        if (!categoriesContainer) return;
        categoriesContainer.innerHTML = '';
        categories.forEach((category) => {
            const card = createCategoryCard(category);
            // カテゴリオブジェクトへの参照を保存
            card._category = category;
            categoriesContainer.appendChild(card);
        });
        updateEmptyStates();
    }

    function updateEmptyStates() {
        if (translationsEmptyState) {
            const hasRows = translationsTableBody && translationsTableBody.querySelector('tr');
            translationsEmptyState.hidden = !!hasRows;
        }
        if (categoriesEmptyState) {
            const hasCategories = categoriesContainer && categoriesContainer.querySelector('.category-card');
            categoriesEmptyState.hidden = !!hasCategories;
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
            const aliases = Array.isArray(entry.aliases) ? entry.aliases : [];
            return { tag, translation, description, aliases };
        });
    }

    async function loadData({ withVersions = true } = {}) {
        try {
            const [translationsResponse, categoriesResponse] = await Promise.all([
                fetch('/api/tag-translations'),
                fetch('/api/tag-categories'),
            ]);
            if (!translationsResponse.ok) {
                throw new Error('翻訳の取得に失敗しました');
            }
            if (!categoriesResponse.ok) {
                throw new Error('カテゴリの取得に失敗しました');
            }
            const translationsData = await translationsResponse.json();
            const categoriesData = await categoriesResponse.json();
            state.translations = mapTranslationsPayload(translationsData.translations || {});
            state.currentVersion = translationsData.version || null;
            updateVersionBadge(state.currentVersion);
            state.pendingChanges = false;
            setSyncStatus({ tone: 'success', message: '最新の状態です' });
            state.categories = Array.isArray(categoriesData.categories)
                ? categoriesData.categories.map((item) => ({
                      id: item.id || '',
                      label: item.label || '',
                      tags: Array.isArray(item.tags) ? item.tags : [],
                  }))
                : [];
            renderTranslations(state.translations);
            renderCategories(state.categories);
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
            const { tagInput, translationInput, aliasesInput, descriptionInput } = row._inputs || {};
            if (!tagInput || !translationInput) continue;
            const rawTag = tagInput.value.trim();
            const rawTranslation = translationInput.value.trim();
            const rawDescription = descriptionInput ? descriptionInput.value.trim() : '';
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
            };
            if (rawAliases.length) {
                entry.aliases = rawAliases;
            }
            result[rawTag] = entry;
        }
        return result;
    }

    function collectCategories() {
        if (!categoriesContainer) return [];
        const cards = Array.from(categoriesContainer.querySelectorAll('.category-card'));
        return cards
            .map((card) => {
                const { idInput, labelInput, tagDisplayContainer } = card._inputs || {};
                if (!idInput || !labelInput || !tagDisplayContainer) {
                    return null;
                }
                const id = idInput.value.trim();
                const label = labelInput.value.trim();
                
                // タグ表示エリアからタグを収集
                const tagElements = tagDisplayContainer.querySelectorAll('.tag-text');
                const tags = Array.from(tagElements).map(el => {
                    const text = el.textContent;
                    // 翻訳がある場合は元のタグ名を抽出
                    const match = text.match(/\(([^)]+)\)$/);
                    return match ? match[1] : text;
                });
                
                if (!id && tags.length === 0 && !label) {
                    return null;
                }
                if (!id) {
                    throw new Error('カテゴリIDを入力してください');
                }
                if (!label) {
                    throw new Error(`カテゴリ「${id}」の表示名を入力してください`);
                }
                return { id, label, tags };
            })
            .filter(Boolean);
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

    async function saveCategories() {
        try {
            const categories = collectCategories();
            const response = await fetch('/api/tag-categories', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ categories }),
            });
            if (!response.ok) {
                throw new Error('カテゴリの保存に失敗しました');
            }
            showStatus('カテゴリを保存しました');
            await loadData();
        } catch (error) {
            console.error(error);
            showStatus(error.message || 'カテゴリの保存に失敗しました', 'error', 5000);
        }
    }

    addTranslationButton?.addEventListener('click', () => {
        const entry = { tag: '', translation: '', description: '', aliases: [] };
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

    addCategoryButton?.addEventListener('click', () => {
        const category = { id: '', label: '', tags: [] };
        state.categories.push(category);
        const card = createCategoryCard(category);
        // カテゴリオブジェクトへの参照を保存
        card._category = category;
        categoriesContainer?.prepend(card);
        card.querySelector('input')?.focus();
        updateEmptyStates();
    });

    saveCategoriesButton?.addEventListener('click', () => {
        saveCategories();
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
