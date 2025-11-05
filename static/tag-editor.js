(() => {
    const translationsTableBody = document.getElementById('translationsTableBody');
    const translationsEmptyState = document.getElementById('translationsEmptyState');
    const categoriesContainer = document.getElementById('categoriesContainer');
    const categoriesEmptyState = document.getElementById('categoriesEmptyState');
    const translationSearchInput = document.getElementById('translationSearch');
    const translationCount = document.getElementById('translationCount');
    const statusBar = document.getElementById('statusBar');

    const addTranslationButton = document.getElementById('addTranslationButton');
    const saveTranslationsButton = document.getElementById('saveTranslationsButton');
    const addCategoryButton = document.getElementById('addCategoryButton');
    const saveCategoriesButton = document.getElementById('saveCategoriesButton');

    const state = {
        translations: [],
        categories: [],
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

    function normaliseTag(value) {
        return (value || '').toString().trim().toLowerCase();
    }

    function updateTranslationCount() {
        if (!translationCount) return;
        const visibleRows = translationsTableBody ? translationsTableBody.querySelectorAll('tr:not([hidden])').length : 0;
        translationCount.textContent = `${visibleRows} 件表示 / 全 ${state.translations.length} 件`;
    }

    function createTranslationRow(entry = { tag: '', translation: '' }) {
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
        });
        tagCell.appendChild(tagInput);

        const translationCell = document.createElement('td');
        const translationInput = document.createElement('input');
        translationInput.type = 'text';
        translationInput.className = 'table-input';
        translationInput.value = entry.translation || '';
        translationInput.placeholder = '翻訳 (日本語)';
        translationCell.appendChild(translationInput);

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
        tr.appendChild(actionCell);

        tr._inputs = { tagInput, translationInput };
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

        const textarea = document.createElement('textarea');
        textarea.className = 'tag-textarea';
        textarea.placeholder = 'タグを1行ずつ入力';
        textarea.value = Array.isArray(category.tags) ? category.tags.join('\n') : '';

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
        card.appendChild(textarea);
        card.appendChild(buttonRow);

        card._inputs = { idInput, labelInput, textarea };
        return card;
    }

    function renderCategories(categories) {
        if (!categoriesContainer) return;
        categoriesContainer.innerHTML = '';
        categories.forEach((category) => {
            const card = createCategoryCard(category);
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
            const { tagInput, translationInput } = row._inputs || {};
            if (!tagInput || !translationInput) return;
            const tagText = tagInput.value.toLowerCase();
            const translationText = translationInput.value.toLowerCase();
            const match = !keyword || tagText.includes(keyword) || translationText.includes(keyword);
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
            state.translations = Object.entries(translationsData.translations || {}).map(([tag, translation]) => ({
                tag,
                translation: translation ?? '',
            }));
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
            const { tagInput, translationInput } = row._inputs || {};
            if (!tagInput || !translationInput) continue;
            const rawTag = tagInput.value.trim();
            const rawTranslation = translationInput.value.trim();
            if (!rawTag) {
                continue;
            }
            const normalised = normaliseTag(rawTag);
            if (duplicates.has(normalised)) {
                throw new Error(`重複しているタグがあります: "${rawTag}" と "${duplicates.get(normalised)}"`);
            }
            duplicates.set(normalised, rawTag);
            result[rawTag] = rawTranslation;
        }
        return result;
    }

    function collectCategories() {
        if (!categoriesContainer) return [];
        const cards = Array.from(categoriesContainer.querySelectorAll('.category-card'));
        return cards
            .map((card) => {
                const { idInput, labelInput, textarea } = card._inputs || {};
                if (!idInput || !labelInput || !textarea) {
                    return null;
                }
                const id = idInput.value.trim();
                const label = labelInput.value.trim();
                const tags = textarea.value
                    .split(/\r?\n/)
                    .map((tag) => tag.trim())
                    .filter((tag) => tag.length > 0);
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
        const entry = { tag: '', translation: '' };
        state.translations.push(entry);
        const row = createTranslationRow(entry);
        translationsTableBody?.prepend(row);
        row.querySelector('input')?.focus();
        updateEmptyStates();
        updateTranslationCount();
    });

    saveTranslationsButton?.addEventListener('click', () => {
        saveTranslations();
    });

    addCategoryButton?.addEventListener('click', () => {
        const category = { id: '', label: '', tags: [] };
        state.categories.push(category);
        const card = createCategoryCard(category);
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
