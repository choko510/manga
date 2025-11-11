(function () {
    const LIMIT = 20;
    let currentAfterCreatedAt = null;
    let currentQuery = '';
    let currentResolvedQuery = '';
    let currentMinPages = 0;
    let currentMaxPages = null;
    let currentRankingOffset = 0;
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
            loadMoreButton: document.getElementById('loadMoreButton'),
            loadMoreContainer: document.getElementById('loadMoreContainer'),
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

        setupObservers();
        attachEventHandlers(elements);
        updateHiddenTagsUI(elements);
        renderHistory(elements);
        performSearch(elements, true);
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

            // 無限スクロール用のオブザーバー
            const loadMoreContainer = document.getElementById('loadMoreContainer');
            if (loadMoreContainer) {
                const infiniteScrollObserver = new IntersectionObserver((entries) => {
                    entries.forEach((entry) => {
                        if (entry.isIntersecting && !isLoading) {
                            const loadMoreButton = document.getElementById('loadMoreButton');
                            if (loadMoreButton && loadMoreContainer.style.display !== 'none') {
                                loadMoreButton.click();
                            }
                        }
                    });
                }, { rootMargin: '300px 0px' });
                
                infiniteScrollObserver.observe(loadMoreContainer);
            }
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
        elements.loadMoreButton.addEventListener('click', () => performSearch(elements, false));
        
        // 初期状態で無限スクロールを有効化
        toggleInfiniteScroll(true);
        
        // 設定パネルに無限スクロールのオン/オフ切り替えを追加
        const infiniteScrollToggle = document.createElement('div');
        infiniteScrollToggle.innerHTML = `
            <label style="display: flex; align-items: center; gap: 8px; margin-top: 16px;">
                <input type="checkbox" id="infiniteScrollCheckbox" checked>
                <span>無限スクロールを有効にする</span>
            </label>
        `;
        
        const settingsPanel = document.getElementById('settingsPanel');
        if (settingsPanel) {
            settingsPanel.insertBefore(infiniteScrollToggle, settingsPanel.querySelector('.settings-actions'));
            
            // チェックボックスのイベントリスナー
            document.getElementById('infiniteScrollCheckbox').addEventListener('change', (e) => {
                toggleInfiniteScroll(e.target.checked);
            });
        }
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
            currentAfterCreatedAt = null;
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
            currentRankingOffset = 0;
            if (window.MangaApp && typeof MangaApp.setLastSearchQuery === 'function') {
                MangaApp.setLastSearchQuery(currentQuery, currentResolvedQuery);
            }
        }

        // 並び順の値を取得
        const sortBy = elements.sortBySelect ? elements.sortBySelect.value : 'created_at';
        const isRankingSort = sortBy !== 'created_at' && ['daily', 'weekly', 'monthly', 'yearly', 'all_time'].includes(sortBy);

        showLoading(elements, true);

        try {
            const data = await fetchSearchResults(
                currentResolvedQuery,
                currentQuery,
                currentAfterCreatedAt,
                currentMinPages,
                currentMaxPages,
                sortBy,
                currentRankingOffset
            );
            const { results, hasMore, nextAfterCreatedAt, nextRankingOffset } = data;

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

            if (isRankingSort) {
                currentAfterCreatedAt = null;
                if (typeof nextRankingOffset === 'number') {
                    currentRankingOffset = nextRankingOffset;
                } else if (reset) {
                    currentRankingOffset = results.length;
                } else {
                    currentRankingOffset += results.length;
                }
            } else {
                currentAfterCreatedAt = nextAfterCreatedAt;
                currentRankingOffset = 0;
            }

            // 無限スクロールが有効な場合はボタンを非表示に、そうでなければ表示
            if (infiniteScrollEnabled) {
                elements.loadMoreContainer.style.display = hasMore ? 'block' : 'none';
                const loadMoreButton = document.getElementById('loadMoreButton');
                if (loadMoreButton) {
                    loadMoreButton.style.display = 'none';
                }
            } else {
                elements.loadMoreContainer.style.display = hasMore ? 'block' : 'none';
                const loadMoreButton = document.getElementById('loadMoreButton');
                if (loadMoreButton) {
                    loadMoreButton.style.display = 'block';
                }
            }
            
            if (results.length) {
                const hiddenTags = MangaApp.getHiddenTags();
                elements.resultMeta.textContent = `表示件数: ${results.length}${hasMore ? ' / さらに表示可能' : ''}${hiddenTags.length ? '（除外タグ反映）' : ''}`;
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
async function fetchSearchResults(resolvedQuery, userQuery, afterCreatedAt, minPages, maxPages, sortBy = 'created_at', rankingOffset = 0) {
    const cacheKey = `${resolvedQuery}|${userQuery}|${afterCreatedAt ?? 'start'}|${minPages ?? 0}|${maxPages ?? ''}|${sortBy}|${rankingOffset}|${MangaApp.getHiddenTags().join(',')}`;
    if (searchCache.has(cacheKey)) {
        return searchCache.get(cacheKey);
    }

    const params = new URLSearchParams();
    params.append('limit', LIMIT.toString());
    const queryForRequest = resolvedQuery || userQuery;
    if (queryForRequest) {
        params.append('tag', queryForRequest);
    }
    if (afterCreatedAt) {
        params.append('after_created_at', afterCreatedAt);
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

    // ランキングソートの場合:
    // - タグまたはクエリ入力あり: 通常の検索API(/search)を使用し sort_by パラメータで並び替え
    // - タグ無し: ランキングAPI(/api/rankings)を使用
    let response;
    if (sortBy !== 'created_at' && ['daily', 'weekly', 'monthly', 'yearly', 'all_time'].includes(sortBy)) {
        if (queryForRequest) {
            // タグ検索（クエリあり）の場合は通常の検索APIを使用し、パラメータにソート順を追加
            params.append('sort_by', sortBy);
            params.append('offset', rankingOffset.toString());
            response = await fetch(`/search?${params.toString()}`);
        } else {
            // タグなしの場合はランキングAPIを使用
            const rankingParams = new URLSearchParams();
            rankingParams.append('ranking_type', sortBy);
            rankingParams.append('limit', LIMIT.toString());
            rankingParams.append('offset', rankingOffset.toString());
            if (typeof minPages === 'number' && minPages > 0) {
                rankingParams.append('min_pages', minPages.toString());
            }
            if (typeof maxPages === 'number' && maxPages > 0) {
                rankingParams.append('max_pages', maxPages.toString());
            }
            if (hiddenTags.length) {
                rankingParams.append('exclude_tag', hiddenTags.join(','));
            }
            response = await fetch(`/api/rankings?${rankingParams.toString()}`);
        }
    } else {
        // 通常の新着順検索
        response = await fetch(`/search?${params.toString()}`);
    }
    if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
    }
    const data = await response.json();

    // ランキングAPIと検索APIでレスポンス形式が異なるため、それぞれに対応
    let filtered, hasMore, nextAfterCreatedAt;
    let nextRankingOffset = null;

    if (sortBy !== 'created_at' && ['daily', 'weekly', 'monthly', 'yearly', 'all_time'].includes(sortBy)) {
        if (queryForRequest) {
            // タグ/クエリありランキングソート:
            // /search のレスポンス形式 { results, has_more, ... } をそのまま利用
            filtered = data.results || [];
            hasMore = Boolean(data.has_more);
            nextAfterCreatedAt = null;
            if (typeof data.next_offset === 'number') {
                nextRankingOffset = data.next_offset;
            } else {
                nextRankingOffset = rankingOffset + filtered.length;
            }
        } else {
            // タグ無しランキングソート:
            // /api/rankings のレスポンス形式 { rankings, has_more, ... } に対応
            filtered = data.rankings || [];
            hasMore = Boolean(data.has_more);
            const lastItem = filtered[filtered.length - 1];
            nextAfterCreatedAt = null; // ランキングでは使用しない
            if (typeof data.next_offset === 'number') {
                nextRankingOffset = data.next_offset;
            } else {
                nextRankingOffset = rankingOffset + filtered.length;
            }
        }
    } else {
        // 通常の検索APIレスポンスを処理
        filtered = data.results || [];
        hasMore = Boolean(data.has_more);
        const lastItem = filtered[filtered.length - 1];
        nextAfterCreatedAt = lastItem ? lastItem.created_at : null;
    }
    
    const result = {
        results: filtered,
        hasMore: hasMore,
        nextAfterCreatedAt: nextAfterCreatedAt,
        nextRankingOffset: nextRankingOffset,
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

        if (reset) {
            elements.cardGrid.innerHTML = '';
        }
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
        if (visible) {
            elements.loadMoreContainer.style.display = 'none';
        }
    }

    // 無限スクロールの状態を管理
    let infiniteScrollEnabled = true;
    let lastScrollPosition = 0;
    let scrollThreshold = 100; // ページ下部からの距離（ピクセル）

    // スクロールイベントリスナーを追加
    document.addEventListener('scroll', () => {
        if (!infiniteScrollEnabled || isLoading) return;

        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        
        // 下部に近づいたら自動読み込み
        if (scrollTop + windowHeight >= documentHeight - scrollThreshold) {
            const loadMoreButton = document.getElementById('loadMoreButton');
            const loadMoreContainer = document.getElementById('loadMoreContainer');
            
            if (loadMoreButton && loadMoreContainer && loadMoreContainer.style.display !== 'none') {
                loadMoreButton.click();
            }
        }
        
        lastScrollPosition = scrollTop;
    });

    // 無限スクロールのオン/オフを切り替える関数
    function toggleInfiniteScroll(enabled) {
        infiniteScrollEnabled = enabled;
        const loadMoreButton = document.getElementById('loadMoreButton');
        const loadMoreContainer = document.getElementById('loadMoreContainer');
        
        if (loadMoreButton && loadMoreContainer) {
            if (enabled) {
                loadMoreButton.textContent = 'さらに表示（自動読み込み有効）';
                loadMoreButton.style.display = 'none'; // 自動読み込み時はボタンを非表示
            } else {
                loadMoreButton.textContent = 'さらに表示';
                loadMoreButton.style.display = 'block'; // 手動読み込み時はボタンを表示
            }
        }
    }

    function debounce(fn, delay) {
        let timer = null;
        return function (...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    }
})();
