(function () {
    const LIMIT = 30;
    let currentPage = 1;
    let currentQuery = '';
    let currentResolvedQuery = '';
    let currentMinPages = 0;
    let currentMaxPages = null;
    let isLoading = false;
    let cardObserver = null;
    const searchCache = new Map();

    function logSearchTracking(query, resolvedQuery) {
        try {
            if (!window.Tracking || !window.Tracking.search || typeof window.Tracking.search.logQuery !== 'function') {
                return;
            }
            const q = (query || '').toString();
            const resolved = (resolvedQuery || '').toString();
            // クエリからタグらしきトークンを抽出（スペース/カンマ区切り）
            const tokens = (resolved || q)
                .split(/[,\s]+/)
                .map((v) => v.trim())
                .filter((v) => v.length > 0);
            window.Tracking.search.logQuery({
                query: q,
                tags: tokens
            });
        } catch (e) {
            console.error('Tracking.search.logQuery error', e);
        }
    }

    function extractTagsFromQuery(query) {
        if (!query) {
            return [];
        }
        return query
            .split(/[,\s]+/)
            .map((value) => value.trim())
            .filter((value) => value.length > 0)
            .map((value) => (value.startsWith('-') ? value.slice(1) : value));
    }

    document.addEventListener('DOMContentLoaded', () => {
        const elements = {
            searchInput: document.getElementById('searchInput'),
            searchButton: document.getElementById('searchButton'),
            sortBySelect: document.getElementById('sortBySelect'),
            minPagesSelect: document.getElementById('minPagesSelect'),
            maxPagesSelect: document.getElementById('maxPagesSelect'),
            cardGrid: document.getElementById('cardGrid'),
            loadingIndicator: document.getElementById('loadingIndicator'),
            historySection: document.getElementById('historySection'),
            historyGrid: document.getElementById('historyGrid'),
            resultMeta: document.getElementById('resultMeta'),
            themeToggle: document.getElementById('themeToggle'),
            settingsToggle: document.getElementById('settingsToggle'),
            settingsPanel: document.getElementById('settingsPanel'),
            settingsOverlay: document.getElementById('settingsOverlay'),
            hiddenTagsInput: document.getElementById('hiddenTagsInput'),
            hiddenTagsSummary: document.getElementById('hiddenTagsSummary'),
            clearHistoryButton: document.getElementById('clearHistoryButton'),
            closeSettingsButton: document.getElementById('closeSettingsButton'),
            resetSettingsButton: document.getElementById('resetSettingsButton'),
            filtersToggle: document.getElementById('filtersToggle'),
            searchOptions: document.getElementById('searchOptions'),
        };

        MangaApp.applyThemeToDocument(document);
        MangaApp.ensureTranslations().catch(() => { /* ignore */ });

        const urlParams = new URLSearchParams(window.location.search);
        const initialTag = urlParams.get('tag');
        if (initialTag) {
            elements.searchInput.value = initialTag;
            currentQuery = initialTag;
        }

        // URLからページ番号を読み取る
        const initialPage = parseInt(urlParams.get('page'), 10);
        if (initialPage && initialPage > 0) {
            currentPage = initialPage;
        }

        setupObservers();
        attachEventHandlers(elements);
        updateHiddenTagsUI(elements);
        renderHistory(elements);
        // 初回は reset=false でURLのページを維持
        performSearch(elements, currentPage === 1);

        // ブラウザの戻る/進むボタン対応
        window.addEventListener('popstate', () => {
            const params = new URLSearchParams(window.location.search);
            const page = parseInt(params.get('page'), 10) || 1;
            const tag = params.get('tag') || '';
            currentPage = page;
            currentQuery = tag;
            elements.searchInput.value = tag;
            performSearch(elements, false);
        });
    });

    function setupObservers() {
        if ('IntersectionObserver' in window) {
            cardObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    const card = entry.target;
                    const img = card.querySelector('img[data-src]');
                    if (img && !img.src) {
                        img.src = img.dataset.src;
                        img.onload = () => {
                            img.style.display = 'block';
                            const placeholder = card.querySelector('.image-placeholder');
                            if (placeholder) {
                                placeholder.remove();
                            }
                        };
                    }
                    cardObserver.unobserve(card);
                });
            }, { rootMargin: '200px 0px' });
        }
    }

    function attachEventHandlers(elements) {
        // デフォルトの最小ページ数を 10 に設定
        if (elements.minPagesSelect) {
            elements.minPagesSelect.value = '10';
            currentMinPages = 10;
        }

        elements.searchButton.addEventListener('click', () => performSearch(elements, true));
        elements.searchInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                performSearch(elements, true);
            }
        });

        // Fuse.js + 共通タグサーチによるオートコンプリート
        if (window.TagSearch && elements.searchInput) {
            TagSearch.initAutocomplete(elements.searchInput, {
                minLength: 1,
                limit: 12,
                onSelect(tag, meta) {
                    // 選択されたタグを検索クエリに追記し、即検索実行
                    const current = elements.searchInput.value.trim();
                    const label = meta && (meta.translation || meta.tag)
                        ? `${meta.translation || ''} (${tag})`
                        : tag;
                    const next = current ? `${current} ${label}` : label;
                    elements.searchInput.value = next;
                    performSearch(elements, true);
                },
            });
        }

        elements.minPagesSelect.addEventListener('change', () => {
            currentMinPages = parseInt(elements.minPagesSelect.value, 10) || 0;
            ensureValidPageRange(elements);
            performSearch(elements, true);
        });
        elements.maxPagesSelect.addEventListener('change', () => {
            currentMaxPages = parseInt(elements.maxPagesSelect.value, 10);
            if (Number.isNaN(currentMaxPages)) {
                currentMaxPages = null;
            }
            ensureValidPageRange(elements);
            performSearch(elements, true);
        });

        // 並び順の変更イベントリスナーを追加
        elements.sortBySelect.addEventListener('change', () => {
            performSearch(elements, true);
        });
        elements.themeToggle.addEventListener('click', () => {
            const theme = MangaApp.toggleTheme();
            updateThemeToggleIcon(elements.themeToggle, theme);
        });
        updateThemeToggleIcon(elements.themeToggle, MangaApp.getPreferredTheme());

        elements.settingsToggle.addEventListener('click', () => toggleSettings(elements, true));
        elements.settingsOverlay.addEventListener('click', () => toggleSettings(elements, false));
        elements.closeSettingsButton.addEventListener('click', () => toggleSettings(elements, false));
        elements.resetSettingsButton.addEventListener('click', () => {
            MangaApp.saveHiddenTags([]);
            updateHiddenTagsUI(elements);
            performSearch(elements, true);
        });
        elements.hiddenTagsInput.addEventListener('input', debounce(() => {
            const tags = elements.hiddenTagsInput.value
                .split(',')
                .map((tag) => tag.trim())
                .filter((tag) => tag);
            MangaApp.saveHiddenTags(tags);
            updateHiddenTagsSummary(elements);
        }, 400));

        if (elements.clearHistoryButton) {
            elements.clearHistoryButton.addEventListener('click', () => {
                MangaApp.clearHistory();
                renderHistory(elements);
            });
        }

        document.addEventListener('manga:history-change', () => renderHistory(elements));
        document.addEventListener('manga:hidden-tags-change', () => {
            updateHiddenTagsUI(elements);
            performSearch(elements, true);
        });

        setupFiltersToggle(elements);
    }

    function setupFiltersToggle(elements) {
        const { filtersToggle, searchOptions } = elements;
        if (!filtersToggle || !searchOptions) {
            return;
        }

        const mobileMediaQuery = window.matchMedia('(max-width: 767px)');

        const handleViewportChange = (event) => {
            if (!event.matches) {
                filtersToggle.setAttribute('aria-expanded', 'false');
                searchOptions.classList.remove('active');
            }
        };

        if (mobileMediaQuery.addEventListener) {
            mobileMediaQuery.addEventListener('change', handleViewportChange);
        } else if (mobileMediaQuery.addListener) {
            mobileMediaQuery.addListener(handleViewportChange);
        }

        filtersToggle.addEventListener('click', () => {
            const isActive = searchOptions.classList.toggle('active');
            filtersToggle.setAttribute('aria-expanded', isActive.toString());
        });

        handleViewportChange(mobileMediaQuery);
    }

    function toggleSettings(elements, show) {
        const isActive = show ?? !elements.settingsPanel.classList.contains('active');
        elements.settingsPanel.classList.toggle('active', isActive);
        elements.settingsOverlay.classList.toggle('active', isActive);
        elements.settingsPanel.setAttribute('aria-hidden', (!isActive).toString());
        if (isActive) {
            elements.hiddenTagsInput.focus();
        }
    }

    function ensureValidPageRange(elements) {
        const minVal = parseInt(elements.minPagesSelect.value, 10) || 0;
        let maxVal = parseInt(elements.maxPagesSelect.value, 10);
        if (!Number.isFinite(maxVal)) {
            maxVal = null;
        }
        if (maxVal !== null && maxVal < minVal) {
            elements.maxPagesSelect.value = '';
            currentMaxPages = null;
        }
        currentMinPages = minVal;
        currentMaxPages = maxVal;
    }

    function updateThemeToggleIcon(button, theme) {
        button.innerHTML = theme === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }

    function updateHiddenTagsUI(elements) {
        const hiddenTags = MangaApp.getHiddenTags();
        elements.hiddenTagsInput.value = hiddenTags.join(', ');
        updateHiddenTagsSummary(elements);
    }

    function updateHiddenTagsSummary(elements) {
        const hiddenTags = MangaApp.getHiddenTags();
        if (!hiddenTags.length) {
            elements.hiddenTagsSummary.textContent = '除外タグなし';
        } else {
            elements.hiddenTagsSummary.textContent = `除外タグ: ${hiddenTags.join(', ')}`;
        }
    }

    async function performSearch(elements, reset) {
        if (isLoading) return;

        if (reset) {
            currentPage = 1;
            currentQuery = elements.searchInput.value.trim();
            currentResolvedQuery = typeof MangaApp.resolveTagQueryString === 'function'
                ? MangaApp.resolveTagQueryString(currentQuery)
                : currentQuery;

            // 新トラッキング: 検索実行を記録
            if (currentQuery) {
                logSearchTracking(currentQuery, currentResolvedQuery);
            }
            currentMinPages = parseInt(elements.minPagesSelect.value, 10) || 0;
            const maxVal = parseInt(elements.maxPagesSelect.value, 10);
            currentMaxPages = Number.isFinite(maxVal) ? maxVal : null;
            elements.cardGrid.innerHTML = '';
            elements.resultMeta.textContent = '';
        }

        // 並び順の値を取得
        const sortBy = elements.sortBySelect ? elements.sortBySelect.value : 'created_at';

        showLoading(elements, true);

        try {
            const data = await fetchSearchResults(
                currentResolvedQuery,
                currentQuery,
                currentPage,
                currentMinPages,
                currentMaxPages,
                sortBy
            );
            const { results, totalPages, totalCount } = data;

            if (reset && typeof MangaApp.recordTagUsage === 'function' && currentQuery) {
                const tags = extractTagsFromQuery(currentResolvedQuery || currentQuery);
                if (tags.length) {
                    MangaApp.recordTagUsage(tags);
                }
            }

            if (reset && results.length === 0) {
                elements.cardGrid.innerHTML = '<p>検索結果が見つかりませんでした。</p>';
                elements.resultMeta.textContent = '0 件を表示中';
            } else {
                renderResults(elements, results, reset);
            }

            // ページネーションを表示
            if (data.totalPages !== undefined) {
                renderPagination(elements, currentPage, data.totalPages, totalCount);
            }

            // URLにページ番号を反映
            updateUrlWithPage(currentPage, currentQuery);

            if (results.length) {
                const hiddenTags = MangaApp.getHiddenTags();
                const startItem = (currentPage - 1) * LIMIT + 1;
                const endItem = Math.min(currentPage * LIMIT, totalCount);
                elements.resultMeta.textContent = `${startItem}-${endItem} 件を表示中 / 合計 ${totalCount} 件${hiddenTags.length ? '（除外タグ反映）' : ''}`;
            }
        } catch (error) {
            console.error('検索エラー', error);
            if (reset) {
                elements.cardGrid.innerHTML = '<p>検索中にエラーが発生しました。</p>';
            }
        } finally {
            showLoading(elements, false);
        }
    }
    async function fetchSearchResults(resolvedQuery, userQuery, page, minPages, maxPages, sortBy = 'created_at') {
        const cacheKey = `${resolvedQuery}|${userQuery}|${page}|${minPages ?? 0}|${maxPages ?? ''}|${sortBy}|${MangaApp.getHiddenTags().join(',')}`;
        if (searchCache.has(cacheKey)) {
            return searchCache.get(cacheKey);
        }

        const params = new URLSearchParams();
        params.append('limit', LIMIT.toString());
        params.append('page', page.toString());
        const queryForRequest = resolvedQuery || userQuery;
        if (queryForRequest) {
            params.append('tag', queryForRequest);
        }
        if (typeof minPages === 'number' && minPages > 0) {
            params.append('min_pages', minPages.toString());
        }
        if (typeof maxPages === 'number' && maxPages > 0) {
            params.append('max_pages', maxPages.toString());
        }
        const hiddenTags = MangaApp.getHiddenTags();
        if (hiddenTags.length) {
            params.append('exclude_tag', hiddenTags.join(','));
        }

        // 統合されたsearchエンドポイントを使用
        // sort_byパラメータでランキング機能も対応
        if (sortBy !== 'created_at' && ['daily', 'weekly', 'monthly', 'yearly', 'all_time'].includes(sortBy)) {
            params.append('sort_by', sortBy);
        }

        const response = await fetch(`/search?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        const data = await response.json();

        // 統合されたsearchエンドポイントのレスポンス形式に対応
        let filtered, totalPages, totalCount;

        // APIレスポンス形式は統一されているため、常に同じ処理を適用
        filtered = data.results || [];
        totalPages = data.total_pages || 1;
        totalCount = data.total_count || filtered.length;

        const result = {
            results: filtered,
            totalPages: totalPages,
            totalCount: totalCount,
        };
        searchCache.set(cacheKey, result);
        if (searchCache.size > 60) {
            const firstKey = searchCache.keys().next().value;
            searchCache.delete(firstKey);
        }
        return result;
    }

    function renderResults(elements, results, reset) {
        const fragment = document.createDocumentFragment();
        results.forEach((gallery) => {
            const card = createGalleryCard(gallery);
            if (cardObserver) {
                cardObserver.observe(card);
            }
            fragment.appendChild(card);
        });

        // ページネーション対応: ページ変更時も常にクリアして入れ替える
        elements.cardGrid.innerHTML = '';
        elements.cardGrid.appendChild(fragment);
    }

    function createGalleryCard(gallery) {
        const card = document.createElement('a');
        card.href = `/viewer?id=${gallery.gallery_id}`;
        card.className = 'card';
        card.dataset.galleryId = gallery.gallery_id;

        const thumbnail = document.createElement('div');
        thumbnail.className = 'card-thumbnail';
        const placeholder = document.createElement('div');
        placeholder.className = 'image-placeholder';
        thumbnail.appendChild(placeholder);
        const img = document.createElement('img');
        let firstImage = '';
        // ランキングAPIと検索APIで画像URLの形式が異なる可能性があるため、両方に対応
        if (Array.isArray(gallery.image_urls) && gallery.image_urls.length > 0) {
            firstImage = gallery.image_urls[0];
        } else if (typeof gallery.image_urls === 'string' && gallery.image_urls) {
            firstImage = gallery.image_urls;
        } else if (gallery.thumbnail_url) {
            firstImage = gallery.thumbnail_url;
        }

        if (firstImage) {
            const resolved = firstImage.startsWith('/proxy/') ? firstImage : `/proxy/${firstImage}`;
            img.dataset.src = resolved;
            if (!cardObserver) {
                img.src = resolved;
                img.onload = () => {
                    const placeholder = thumbnail.querySelector('.image-placeholder');
                    if (placeholder) {
                        placeholder.remove();
                    }
                };
            }
        }
        thumbnail.appendChild(img);

        const likeButton = document.createElement('button');
        likeButton.className = 'like-button';
        likeButton.type = 'button';
        likeButton.setAttribute('aria-label', 'いいね');
        const liked = MangaApp.isLiked(gallery.gallery_id);
        likeButton.classList.toggle('active', liked);
        likeButton.innerHTML = liked ? '<i class="fas fa-heart"></i>' : '<i class="far fa-heart"></i>';
        likeButton.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            const isLiked = MangaApp.toggleLike(gallery.gallery_id);
            likeButton.classList.toggle('active', isLiked);
            likeButton.innerHTML = isLiked ? '<i class="fas fa-heart"></i>' : '<i class="far fa-heart"></i>';
        });

        const info = document.createElement('div');
        info.className = 'card-info';

        const title = document.createElement('h3');
        title.className = 'card-title';
        title.textContent = gallery.japanese_title || '無題';

        const meta = document.createElement('div');
        meta.className = 'card-meta';
        if (gallery.page_count) {
            const pageInfo = document.createElement('span');
            pageInfo.textContent = `${gallery.page_count}ページ`;
            meta.appendChild(pageInfo);
        }

        const tagsList = document.createElement('ul');
        tagsList.className = 'card-tags';
        const tagArray = safeParseArray(gallery.tags);
        tagArray.slice(0, 5).forEach((tag) => {
            const tagItem = document.createElement('li');
            const chip = document.createElement('a');
            chip.href = `/tags?tag=${encodeURIComponent(tag)}`;
            chip.className = 'tag-chip';
            const jp = document.createElement('span');
            jp.className = 'tag-jp';
            jp.textContent = MangaApp.translateTag(tag);
            chip.appendChild(jp);
            tagItem.appendChild(chip);
            tagsList.appendChild(tagItem);
        });

        info.appendChild(title);
        info.appendChild(meta);
        info.appendChild(tagsList);

        card.appendChild(thumbnail);
        card.appendChild(likeButton);
        card.appendChild(info);

        card.addEventListener('click', () => {
            MangaApp.addHistoryEntry(gallery);
        });

        return card;
    }

    function safeParseArray(source) {
        if (Array.isArray(source)) {
            return source;
        }
        if (typeof source === 'string') {
            try {
                const parsed = JSON.parse(source);
                return Array.isArray(parsed) ? parsed : [];
            } catch (error) {
                return source.split(/[\s,]+/).filter(Boolean);
            }
        }
        return [];
    }

    function applyHistoryThumbnail(element, entry) {
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

    function renderHistory(elements) {
        if (!elements.historySection || !elements.historyGrid) {
            return;
        }
        const history = MangaApp.getHistory();
        if (!history.length) {
            elements.historySection.classList.remove('active');
            elements.historyGrid.innerHTML = '';
            return;
        }
        elements.historySection.classList.add('active');
        elements.historyGrid.innerHTML = '';
        const fragment = document.createDocumentFragment();
        history.slice(0, 8).forEach((item) => {
            const card = document.createElement('a');
            card.href = `/viewer?id=${item.gallery_id}`;
            card.className = 'history-card';
            applyHistoryThumbnail(card, item);
            const label = document.createElement('span');
            label.textContent = item.japanese_title || '無題';
            card.appendChild(label);
            fragment.appendChild(card);
        });
        elements.historyGrid.appendChild(fragment);
    }

    function showLoading(elements, visible) {
        isLoading = visible;
        elements.loadingIndicator.style.display = visible ? 'block' : 'none';
    }

    // ページネーションをレンダリングする関数
    function renderPagination(elements, page, totalPages, totalCount) {
        // 既存のページネーションを削除
        const existingPaginationTop = document.getElementById('paginationContainerTop');
        const existingPaginationBottom = document.getElementById('paginationContainerBottom');
        if (existingPaginationTop) {
            existingPaginationTop.remove();
        }
        if (existingPaginationBottom) {
            existingPaginationBottom.remove();
        }

        // デバッグ情報をコンソールに出力
        console.log('renderPagination called:', { page, totalPages, totalCount });

        if (totalPages <= 1) {
            console.log('Total pages <= 1, not showing pagination');
            return; // 1ページのみの場合はページネーションを表示しない
        }

        // ページネーションコンテナを生成する関数
        function createPaginationContainer(id) {
            const paginationContainer = document.createElement('div');
            paginationContainer.id = id;
            paginationContainer.className = 'pagination-container';
            paginationContainer.style.cssText = `
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 8px;
                margin: 24px 0;
                flex-wrap: wrap;
            `;

            // 前のページボタン
            const prevButton = createPaginationButton('« 前へ', page > 1, () => {
                if (page > 1) {
                    currentPage = page - 1;
                    performSearch(elements, false);
                    window.scrollTo({ top: 0, behavior: 'instant' });
                }
            });
            paginationContainer.appendChild(prevButton);

            // ページ番号ボタン
            const maxButtons = 5;
            let startPage = Math.max(1, page - Math.floor(maxButtons / 2));
            let endPage = Math.min(totalPages, startPage + maxButtons - 1);

            if (endPage - startPage < maxButtons - 1) {
                startPage = Math.max(1, endPage - maxButtons + 1);
            }

            if (startPage > 1) {
                paginationContainer.appendChild(createPaginationButton('1', true, () => {
                    currentPage = 1;
                    performSearch(elements, false);
                    window.scrollTo({ top: 0, behavior: 'instant' });
                }));
                if (startPage > 2) {
                    const dots = document.createElement('span');
                    dots.textContent = '...';
                    dots.style.cssText = 'padding: 0 8px; color: var(--text-secondary);';
                    paginationContainer.appendChild(dots);
                }
            }

            for (let i = startPage; i <= endPage; i++) {
                const pageButton = createPaginationButton(i.toString(), true, () => {
                    currentPage = i;
                    performSearch(elements, false);
                    window.scrollTo({ top: 0, behavior: 'instant' });
                });
                if (i === page) {
                    pageButton.style.cssText += `
                        background: var(--accent);
                        color: white;
                        border-color: var(--accent);
                    `;
                }
                paginationContainer.appendChild(pageButton);
            }

            if (endPage < totalPages) {
                if (endPage < totalPages - 1) {
                    const dots = document.createElement('span');
                    dots.textContent = '...';
                    dots.style.cssText = 'padding: 0 8px; color: var(--text-secondary);';
                    paginationContainer.appendChild(dots);
                }
                paginationContainer.appendChild(createPaginationButton(totalPages.toString(), true, () => {
                    currentPage = totalPages;
                    performSearch(elements, false);
                    window.scrollTo({ top: 0, behavior: 'instant' });
                }));
            }

            // 次のページボタン
            const nextButton = createPaginationButton('次へ »', page < totalPages, () => {
                if (page < totalPages) {
                    currentPage = page + 1;
                    performSearch(elements, false);
                    window.scrollTo({ top: 0, behavior: 'instant' });
                }
            });
            paginationContainer.appendChild(nextButton);

            return paginationContainer;
        }

        // 上下にページネーションを挿入
        const cardGrid = document.getElementById('cardGrid');
        if (cardGrid) {
            // 上のページネーション
            const topPagination = createPaginationContainer('paginationContainerTop');
            cardGrid.parentNode.insertBefore(topPagination, cardGrid);

            // 下のページネーション
            const bottomPagination = createPaginationContainer('paginationContainerBottom');
            cardGrid.parentNode.insertBefore(bottomPagination, cardGrid.nextSibling);

            console.log('Pagination containers added to DOM (top and bottom)');
        } else {
            console.error('cardGrid not found');
        }
    }

    function createPaginationButton(text, enabled, onClick) {
        const button = document.createElement('button');
        button.textContent = text;
        button.className = 'pagination-button';
        button.style.cssText = `
            padding: 8px 12px;
            border: 1px solid rgba(148, 163, 184, 0.3);
            background: var(--bg-button);
            color: var(--text-secondary);
            border-radius: 6px;
            cursor: ${enabled ? 'pointer' : 'not-allowed'};
            opacity: ${enabled ? '1' : '0.5'};
            transition: all 0.2s ease;
            font-size: 14px;
            min-width: 40px;
        `;

        if (enabled) {
            button.addEventListener('click', onClick);
            button.addEventListener('mouseenter', () => {
                button.style.borderColor = 'var(--accent)';
                button.style.color = 'var(--accent)';
            });
            button.addEventListener('mouseleave', () => {
                button.style.borderColor = 'rgba(148, 163, 184, 0.3)';
                button.style.color = 'var(--text-secondary)';
            });
        }

        return button;
    }

    function debounce(fn, delay) {
        let timer = null;
        return function (...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    }

    // URLにページ番号を反映する関数
    function updateUrlWithPage(page, query) {
        const url = new URL(window.location);

        // ページ番号を設定（1ページ目の場合は削除してURLをきれいに）
        if (page > 1) {
            url.searchParams.set('page', page.toString());
        } else {
            url.searchParams.delete('page');
        }

        // タグを設定
        if (query) {
            url.searchParams.set('tag', query);
        } else {
            url.searchParams.delete('tag');
        }

        // URLを更新（履歴に追加）
        window.history.replaceState({ page, query }, '', url.toString());
    }
})();
