(() => {
    const translationsTableBody = document.getElementById('translationsTableBody');
    const translationsEmptyState = document.getElementById('translationsEmptyState');
    const categoriesContainer = document.getElementById('categoriesContainer');
    const categoriesEmptyState = document.getElementById('categoriesEmptyState');
    const translationSearchInput = document.getElementById('translationSearch');
    const translationCount = document.getElementById('translationCount');
    const statusBar = document.getElementById('statusBar');

    const addTranslationButton = document.getElementById('addTranslationButton');
    const autoAddButton = document.getElementById('autoAddButton');
    const saveTranslationsButton = document.getElementById('saveTranslationsButton');
    const addCategoryButton = document.getElementById('addCategoryButton');
    const saveCategoriesButton = document.getElementById('saveCategoriesButton');

    const state = {
        translations: [],
        categories: [],
        popularTagsOffset: 0, // 追加: 人気タグの取得オフセット
        popularTagsLimit: 10, // 追加: 人気タグの取得数
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

    function normaliseTag(value) {
        return (value || '').toString().trim().toLowerCase();
    }

    function updateTranslationCount() {
        if (!translationCount) return;
        const visibleRows = translationsTableBody ? translationsTableBody.querySelectorAll('tr:not([hidden])').length : 0;
        translationCount.textContent = `${visibleRows} 件表示 / 全 ${state.translations.length} 件`;
    }

    function createTranslationRow(entry = { tag: '', translation: '', description: '' }) {
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
        });
        translationCell.appendChild(translationInput);

        const descriptionCell = document.createElement('td');
        const descriptionInput = document.createElement('input');
        descriptionInput.type = 'text';
        descriptionInput.className = 'table-input';
        descriptionInput.value = entry.description || '';
        descriptionInput.placeholder = '説明 (任意)';
        descriptionInput.addEventListener('input', () => {
            entry.description = descriptionInput.value;
        });
        descriptionCell.appendChild(descriptionInput);

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
        });
        actionCell.appendChild(deleteButton);

        tr.appendChild(tagCell);
        tr.appendChild(translationCell);
        tr.appendChild(descriptionCell);
        tr.appendChild(actionCell);

        tr._inputs = { tagInput, translationInput, descriptionInput };
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
    }

    function createCategoryCard(category = { id: '', label: '', tags: [] }) {
        const card = document.createElement('div');
        card.className = 'category-card';

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
            const { tagInput, translationInput, descriptionInput } = row._inputs || {};
            if (!tagInput || !translationInput) return;
            const tagText = tagInput.value.toLowerCase();
            const translationText = translationInput.value.toLowerCase();
            const descriptionText = descriptionInput ? descriptionInput.value.toLowerCase() : '';
            const match = !keyword || tagText.includes(keyword) || translationText.includes(keyword) || descriptionText.includes(keyword);
            row.hidden = !match;
        });
        updateTranslationCount();
    }

    async function loadData() {
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
            state.translations = Object.entries(translationsData.translations || {}).map(([tag, value]) => {
                let translation = '';
                let description = '';
                if (value && typeof value === 'object') {
                    translation = value.translation || '';
                    description = value.description || '';
                } else if (typeof value === 'string') {
                    translation = value;
                }
                return { tag, translation, description };
            });
            state.categories = Array.isArray(categoriesData.categories)
                ? categoriesData.categories.map((item) => ({
                      id: item.id || '',
                      label: item.label || '',
                      tags: Array.isArray(item.tags) ? item.tags : [],
                  }))
                : [];
            renderTranslations(state.translations);
            renderCategories(state.categories);
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
        for (const row of rows) {
            const { tagInput, translationInput, descriptionInput } = row._inputs || {};
            if (!tagInput || !translationInput) continue;
            const rawTag = tagInput.value.trim();
            const rawTranslation = translationInput.value.trim();
            const rawDescription = descriptionInput ? descriptionInput.value.trim() : '';
            if (!rawTag) {
                continue;
            }
            const normalised = normaliseTag(rawTag);
            if (duplicates.has(normalised)) {
                throw new Error(`重複しているタグがあります: "${rawTag}" と "${duplicates.get(normalised)}"`);
            }
            duplicates.set(normalised, rawTag);
            result[rawTag] = {
                translation: rawTranslation,
                description: rawDescription,
            };
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

    async function saveTranslations() {
        try {
            const translations = collectTranslations();
            const response = await fetch('/api/tag-translations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ translations }),
            });
            if (!response.ok) {
                throw new Error('翻訳の保存に失敗しました');
            }
            showStatus('翻訳を保存しました');
            await loadData();
        } catch (error) {
            console.error(error);
            showStatus(error.message || '翻訳の保存に失敗しました', 'error', 5000);
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
        const entry = { tag: '', translation: '', description: '' };
        state.translations.push(entry);
        const row = createTranslationRow(entry);
        translationsTableBody?.prepend(row);
        row.querySelector('input')?.focus();
        updateEmptyStates();
        updateTranslationCount();
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
            
        } catch (error) {
            console.error(error);
            showStatus(error.message || 'タグの追加に失敗しました', 'error', 5000);
        }
    });

    saveTranslationsButton?.addEventListener('click', () => {
        saveTranslations();
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

    loadData();
})();
