(function () {
    const LIMIT = 30;
    const SEARCH_CACHE_MAX_SIZE = 60;
    const PAGINATION_MAX_BUTTONS = 5;
    const PRIORITY_IMAGE_COUNT = 6;
    const VISIBLE_TAGS_COUNT = 5;
    const HISTORY_DISPLAY_COUNT = 8;
    const RECOMMENDATION_LIMIT = 20;
    const DEBOUNCE_DELAY_MS = 400;
    const TRACKING_MAX_TAGS = 50;
    const OBSERVER_ROOT_MARGIN = '600px 0px';
    let currentPage = 1;
    let currentQuery = '';
    let currentResolvedQuery = '';
    let currentUnifiedQuery = '';
    let currentMinPages = 10;
    let currentMaxPages = null;
    let isLoading = false;

    /**
     * ギャラリーオブジェクトから最初の画像URLを取得する。
     * image_urls (配列/文字列) と thumbnail_url の両方に対応。
     */
    function resolveFirstImageUrl(gallery) {
        if (Array.isArray(gallery.image_urls) && gallery.image_urls.length > 0) {
            return gallery.image_urls[0];
        }
        if (typeof gallery.image_urls === 'string' && gallery.image_urls) {
            return gallery.image_urls;
        }
        if (gallery.thumbnail_url) {
            return gallery.thumbnail_url;
        }
        return '';
    }

    /**
     * Observer が無い場合のフォールバック: 低解像度→高解像度の段階的読み込みを設定する。
     */
    function setupFallbackImageLoad(img, thumbnailEl) {
        const initialSrc = img.dataset.lowRes || img.dataset.src;
        img.src = initialSrc;
        img.onload = () => {
            img.style.display = 'block';
            const ph = thumbnailEl.querySelector('.image-placeholder');
            if (ph) ph.remove();

            if (img.dataset.lowRes && img.src.includes('thumbnail=true')) {
                const originalImg = new Image();
                originalImg.src = img.dataset.src;
                originalImg.onload = () => {
                    img.src = originalImg.src;
                };
            }
        };
    }

    /**
     * いいねボタンを作成する共通関数。
     */
    function createLikeButton(galleryId) {
        const likeButton = document.createElement('button');
        likeButton.className = 'like-button';
        likeButton.type = 'button';
        likeButton.setAttribute('aria-label', 'いいね');
        const liked = MangaApp.isLiked(galleryId);
        likeButton.classList.toggle('active', liked);
        likeButton.innerHTML = liked ? '<i class="fas fa-heart"></i>' : '<i class="far fa-heart"></i>';
        likeButton.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            const isLiked = MangaApp.toggleLike(galleryId);
            likeButton.classList.toggle('active', isLiked);
            likeButton.innerHTML = isLiked ? '<i class="fas fa-heart"></i>' : '<i class="far fa-heart"></i>';
        });
        return likeButton;
    }

    let cardObserver = null;
    let isPageLoaded = false;
    const deferredToObserve = []; // 読み込み完了後にobserveする画像を一時保存
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
            sortBySelectMobile: document.getElementById('sortBySelectMobile'),
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
        const initialQ = urlParams.get('q');

        if (initialQ) {
            elements.searchInput.value = initialQ;
            currentQuery = initialQ;
            currentUnifiedQuery = initialQ;
            currentResolvedQuery = typeof MangaApp.resolveTagQueryString === 'function'
                ? MangaApp.resolveTagQueryString(initialQ)
                : initialQ;
        } else if (initialTag) {
            elements.searchInput.value = initialTag;
            currentQuery = initialTag;
            currentResolvedQuery = typeof MangaApp.resolveTagQueryString === 'function'
                ? MangaApp.resolveTagQueryString(initialTag)
                : initialTag;
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

        // パーソナライズおすすめセクションを初期化
        loadRecommendations(elements);

        // 初回は reset=false でURLのページを維持
        performSearch(elements, currentPage === 1);

        // ブラウザの戻る/進むボタン対応
        window.addEventListener('popstate', () => {
            const params = new URLSearchParams(window.location.search);
            const page = parseInt(params.get('page'), 10) || 1;
            const tag = params.get('tag') || '';
            const q = params.get('q') || '';
            currentPage = page;
            currentQuery = tag || q;
            currentUnifiedQuery = q;
            currentResolvedQuery = tag;
            elements.searchInput.value = tag || q;
            performSearch(elements, false);
        });

        // ページ全体の読み込みが完了したら遅延読み込みを開始
        if (document.readyState === 'complete') {
            isPageLoaded = true;
        } else {
            window.addEventListener('load', () => {
                isPageLoaded = true;
                // Deferされていた画像をobserve開始
                if (cardObserver && deferredToObserve.length > 0) {
                    deferredToObserve.forEach((card) => {
                        cardObserver.observe(card);
                    });
                    deferredToObserve.length = 0; // 配列を空にする
                }
            });
        }
    });

    function setupObservers() {
        if ('IntersectionObserver' in window) {
            cardObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    const card = entry.target;
                    const img = card.querySelector('img[data-src]');
                    if (img && !img.src) {
                        const lowRes = img.dataset.lowRes;
                        const finalSrc = img.dataset.src;

                        // PCならまず軽量版をセット
                        img.src = lowRes || finalSrc;

                        img.onload = () => {
                            img.style.display = 'block';
                            const placeholder = card.querySelector('.image-placeholder');
                            if (placeholder) {
                                placeholder.remove();
                            }

                            // 軽量版から高画質版への切り替え（PCのみ）
                            if (lowRes) {
                                const originalImg = new Image();
                                originalImg.src = finalSrc;
                                originalImg.onload = () => {
                                    img.src = finalSrc;
                                };
                            }
                        };
                    }
                    cardObserver.unobserve(card);
                });
            }, { rootMargin: OBSERVER_ROOT_MARGIN });
        }
    }

    function attachEventHandlers(elements) {
        // 保存された設定を復元
        const savedSettings = typeof MangaApp.getSearchSettings === 'function'
            ? MangaApp.getSearchSettings()
            : { sortBy: 'weekly', minPages: 10, maxPages: null };

        // 並び順を復元
        if (savedSettings.sortBy) {
            if (elements.sortBySelect) elements.sortBySelect.value = savedSettings.sortBy;
            if (elements.sortBySelectMobile) elements.sortBySelectMobile.value = savedSettings.sortBy;
        }

        // 最小ページ数を復元
        if (elements.minPagesSelect) {
            const minPages = savedSettings.minPages ?? 10;
            elements.minPagesSelect.value = String(minPages);
            currentMinPages = minPages;
        }

        // 最大ページ数を復元
        if (elements.maxPagesSelect) {
            const maxPages = savedSettings.maxPages;
            elements.maxPagesSelect.value = maxPages !== null ? String(maxPages) : '';
            currentMaxPages = maxPages;
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
            ensureValidPageRange(elements);
            // 設定を保存
            if (typeof MangaApp.saveSearchSettings === 'function') {
                MangaApp.saveSearchSettings({
                    minPages: currentMinPages,
                    maxPages: currentMaxPages
                });
            }
            performSearch(elements, true);
        });
        elements.maxPagesSelect.addEventListener('change', () => {
            ensureValidPageRange(elements);
            // 設定を保存
            if (typeof MangaApp.saveSearchSettings === 'function') {
                MangaApp.saveSearchSettings({
                    minPages: currentMinPages,
                    maxPages: currentMaxPages
                });
            }
            performSearch(elements, true);
        });

        // 並び順の同期関数
        const syncSortBy = (val) => {
            if (elements.sortBySelect) elements.sortBySelect.value = val;
            if (elements.sortBySelectMobile) elements.sortBySelectMobile.value = val;
        };

        // 並び順の変更イベントリスナーを追加
        if (elements.sortBySelect) {
            elements.sortBySelect.addEventListener('change', () => {
                const val = elements.sortBySelect.value;
                syncSortBy(val);
                if (typeof MangaApp.saveSearchSettings === 'function') {
                    MangaApp.saveSearchSettings({ sortBy: val });
                }
                performSearch(elements, true);
            });
        }

        if (elements.sortBySelectMobile) {
            elements.sortBySelectMobile.addEventListener('change', () => {
                const val = elements.sortBySelectMobile.value;
                syncSortBy(val);
                if (typeof MangaApp.saveSearchSettings === 'function') {
                    MangaApp.saveSearchSettings({ sortBy: val });
                }
                performSearch(elements, true);
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
        }, DEBOUNCE_DELAY_MS));

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
            maxVal = null;
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
            const rawInput = elements.searchInput.value.trim();
            currentQuery = rawInput;

            // 入力が作品コード(RJ/BJ等)っぽかったり、コロンを含まない単一ワードなら
            // タイトル・コード検索(q)を優先的に検討する。
            // ただし MangaApp.resolveTagQueryString がタグとして解決した場合は両方検討したいが、
            // 現状のバックエンド仕様では q と tag は AND になるため、
            // 検索ボックスからの入力は基本的に q として送る。

            currentResolvedQuery = typeof MangaApp.resolveTagQueryString === 'function'
                ? MangaApp.resolveTagQueryString(currentQuery)
                : currentQuery;

            currentUnifiedQuery = currentQuery;

            // もし入力が完全にタグ形式(例: "artist:xxx" や "female:xxx")だけで構成されているなら
            // tagとして扱う方が精度的には高い。
            // しかし「どこでも検索」を優先するため、常にqをセットする。

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
        const sortBy = (elements.sortBySelectMobile && window.getComputedStyle(elements.sortBySelectMobile.parentElement).display !== 'none')
            ? elements.sortBySelectMobile.value
            : (elements.sortBySelect ? elements.sortBySelect.value : 'created_at');

        showLoading(elements, true);

        try {
            const data = await fetchSearchResults(
                currentResolvedQuery,
                currentQuery,
                currentPage,
                currentMinPages,
                currentMaxPages,
                sortBy,
                currentUnifiedQuery
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
            updateUrlWithPage(currentPage, currentQuery, currentUnifiedQuery);

            if (results.length) {
                const hiddenTags = MangaApp.getHiddenTags();
                const startItem = (currentPage - 1) * LIMIT + 1;
                const endItem = Math.min(currentPage * LIMIT, totalCount);
                elements.resultMeta.textContent = `${startItem}-${endItem} 件を表示中 / 合計 ${totalCount} 件${hiddenTags.length ? '（除外タグ反映）' : ''}`;

                // 次のページのサムネイルを先読み（即座に開始 + 全画像ロード後にも）
                if (currentPage < totalPages) {
                    // 即座に先読み開始（全画像ロードを待たない）
                    prefetchNextPageThumbnails(currentPage + 1, sortBy, currentUnifiedQuery);
                }
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
    async function fetchSearchResults(resolvedQuery, userQuery, page, minPages, maxPages, sortBy = 'created_at', unifiedQuery = '') {
        const cacheKey = `${resolvedQuery}|${userQuery}|${unifiedQuery}|${page}|${minPages ?? 0}|${maxPages ?? ''}|${sortBy}|${MangaApp.getHiddenTags().join(',')}`;
        if (searchCache.has(cacheKey)) {
            return searchCache.get(cacheKey);
        }

        const params = new URLSearchParams();
        params.append('limit', LIMIT.toString());
        params.append('page', page.toString());
        const queryForRequest = resolvedQuery || userQuery;
        if (unifiedQuery) {
            params.append('q', unifiedQuery);
        } else if (queryForRequest) {
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
        if (searchCache.size > SEARCH_CACHE_MAX_SIZE) {
            const firstKey = searchCache.keys().next().value;
            searchCache.delete(firstKey);
        }
        return result;
    }

    function renderResults(elements, results, reset) {
        const fragment = document.createDocumentFragment();
        results.forEach((gallery, index) => {
            const card = createGalleryCard(gallery, index);
            if (cardObserver && index >= 4) {
                if (isPageLoaded) {
                    cardObserver.observe(card);
                } else {
                    deferredToObserve.push(card);
                }
            }
            fragment.appendChild(card);
        });

        // ページネーション対応: ページ変更時も常にクリアして入れ替える
        elements.cardGrid.innerHTML = '';
        elements.cardGrid.appendChild(fragment);
    }

    function createGalleryCard(gallery, index = 100) {
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
        img.alt = gallery.japanese_title || '';
        const firstImage = resolveFirstImageUrl(gallery);

        if (firstImage) {
            const baseUrl = firstImage.startsWith('/proxy/') ? firstImage : `/proxy/${firstImage}`;
            const thumbUrl = `${baseUrl}?thumbnail=true&small=true`;
            const fullUrl = baseUrl;

            // 最初の数枚は即時読み込み（Lazy Loadを回避）して表示速度を向上
            const isPriority = index < PRIORITY_IMAGE_COUNT;

            // 環境に応じたターゲットURL
            let finalSrc = fullUrl;
            let lowResSrc = null;

            if (MangaApp.isMobile()) {
                finalSrc = thumbUrl;
            } else {
                finalSrc = fullUrl;
                lowResSrc = thumbUrl;
            }

            if (isPriority) {
                img.src = finalSrc;
                img.style.display = 'block';
                const ph = thumbnail.querySelector('.image-placeholder');
                if (ph) ph.remove();

                if (index === 0) {
                    img.setAttribute('fetchpriority', 'high');
                }
            } else {
                img.dataset.src = finalSrc;
                if (lowResSrc) {
                    img.dataset.lowRes = lowResSrc;
                }
            }

            if (!cardObserver && !isPriority) {
                setupFallbackImageLoad(img, thumbnail);
            }
        }
        thumbnail.appendChild(img);

        const likeButton = createLikeButton(gallery.gallery_id);

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
        tagArray.slice(0, VISIBLE_TAGS_COUNT).forEach((tag) => {
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
            if (MangaApp.isMobile()) {
                element.style.backgroundImage = `url(${url})`;
            } else {
                const bUrl = url.split('?')[0];
                const tUrl = `${bUrl}?thumbnail=true&small=true`;
                element.style.backgroundImage = `url(${tUrl})`;
                const originalImg = new Image();
                originalImg.src = bUrl;
                originalImg.onload = () => {
                    element.style.backgroundImage = `url(${originalImg.src})`;
                };
            }
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
                const baseUrl = fetchedUrl.startsWith('/proxy/') ? fetchedUrl : `/proxy/${fetchedUrl}`;
                const thumbUrl = `${baseUrl}?thumbnail=true&small=true`;

                if (MangaApp.isMobile()) {
                    element.style.backgroundImage = `url(${thumbUrl})`;
                } else {
                    element.style.backgroundImage = `url(${thumbUrl})`;
                    const originalImg = new Image();
                    originalImg.src = baseUrl;
                    originalImg.onload = () => {
                        element.style.backgroundImage = `url(${originalImg.src})`;
                    };
                }
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
        history.slice(0, HISTORY_DISPLAY_COUNT).forEach((item) => {
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

    // =========================
    // パーソナライズおすすめセクション
    // =========================

    async function loadRecommendations(elements) {
        // user_idを取得（TrackingからまたはlocalStorageから）
        let userId = null;
        if (window.Tracking && typeof window.Tracking.getUserId === 'function') {
            userId = window.Tracking.getUserId();
        } else {
            try {
                userId = localStorage.getItem('tracking_user_id');
            } catch (e) {
                // ignore
            }
        }

        if (!userId) {
            return;
        }

        try {
            const hiddenTags = MangaApp.getHiddenTags();
            const excludeTag = hiddenTags.length ? hiddenTags.join(',') : '';

            const params = new URLSearchParams();
            params.append('user_id', userId);
            params.append('limit', String(RECOMMENDATION_LIMIT));
            if (excludeTag) {
                params.append('exclude_tag', excludeTag);
            }

            const response = await fetch(`/api/recommendations/personal?${params.toString()}`);
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const data = await response.json();

            // おすすめがない場合または機能が無効の場合は何もしない
            if (!data.results || data.results.length === 0 || data.disabled) {
                return;
            }

            // データがある場合のみセクションを作成してDOMに追加
            const recommendSection = document.createElement('section');
            recommendSection.id = 'recommendSection';
            recommendSection.className = 'recommend-section';

            const sectionTitle = data.has_personalization
                ? '<i class="fas fa-magic"></i> あなたへのおすすめ'
                : '<i class="fas fa-star"></i> 人気作品';

            recommendSection.innerHTML = `
                <div class="section-header">
                    <h2>${sectionTitle}</h2>
                </div>
                <div id="recommendGrid" class="recommend-grid"></div>
            `;

            // 履歴セクションの後、または検索フォームの後に挿入
            const historySection = elements.historySection || document.getElementById('historySection');
            if (historySection) {
                historySection.parentNode.insertBefore(recommendSection, historySection.nextSibling);
            } else {
                const cardGrid = elements.cardGrid || document.getElementById('cardGrid');
                if (cardGrid) {
                    cardGrid.parentNode.insertBefore(recommendSection, cardGrid);
                }
            }

            const recommendGrid = document.getElementById('recommendGrid');
            if (!recommendGrid) return;

            // おすすめカードを表示
            const fragment = document.createDocumentFragment();
            data.results.forEach((gallery) => {
                const card = createRecommendCard(gallery);
                if (cardObserver) {
                    cardObserver.observe(card);
                }
                fragment.appendChild(card);
            });
            recommendGrid.appendChild(fragment);

            // インプレッション記録（おすすめ一覧に表示された）
            const mangaIds = data.results.map(r => r.gallery_id);
            const allTags = [];
            data.results.forEach(r => {
                const tags = safeParseArray(r.tags);
                tags.forEach(t => {
                    if (!allTags.includes(t)) allTags.push(t);
                });
            });

            if (window.Tracking && window.Tracking.logImpression) {
                window.Tracking.logImpression({
                    mangaIds: mangaIds,
                    tags: allTags.slice(0, TRACKING_MAX_TAGS)
                });
            }

        } catch (error) {
            console.error('おすすめ取得エラー:', error);
            recommendSection.style.display = 'none';
        }
    }

    function createRecommendCard(gallery) {
        const card = document.createElement('a');
        card.href = `/viewer?id=${gallery.gallery_id}`;
        card.className = 'card recommend-card';
        card.dataset.galleryId = gallery.gallery_id;

        // サムネイル
        const thumbnail = document.createElement('div');
        thumbnail.className = 'card-thumbnail';
        const placeholder = document.createElement('div');
        placeholder.className = 'image-placeholder';
        thumbnail.appendChild(placeholder);

        const img = document.createElement('img');
        img.alt = gallery.japanese_title || '';
        const firstImage = resolveFirstImageUrl(gallery);

        if (firstImage) {
            const baseUrl = firstImage.startsWith('/proxy/') ? firstImage : `/proxy/${firstImage}`;
            const thumbUrl = `${baseUrl}?thumbnail=true&small=true`;
            const fullUrl = baseUrl;

            if (MangaApp.isMobile()) {
                img.dataset.src = thumbUrl;
            } else {
                img.dataset.src = fullUrl;
                img.dataset.lowRes = thumbUrl;
            }

            if (!cardObserver) {
                setupFallbackImageLoad(img, thumbnail);
            }
        }
        thumbnail.appendChild(img);

        const likeButton = createLikeButton(gallery.gallery_id);

        // 情報
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

        info.appendChild(title);
        info.appendChild(meta);

        card.appendChild(thumbnail);
        card.appendChild(likeButton);
        card.appendChild(info);

        // クリック時の処理
        card.addEventListener('click', () => {
            MangaApp.addHistoryEntry(gallery);

            // クリック記録
            if (window.Tracking && window.Tracking.logImpression) {
                const tags = safeParseArray(gallery.tags);
                // Trackingにはクリック用のlogImpression的なメソッドがないが、後でviewer側で記録される
            }
        });

        return card;
    }

    function showLoading(elements, visible) {
        isLoading = visible;
        elements.loadingIndicator.classList.toggle('hidden', !visible);
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

        if (totalPages <= 1) {
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
            let startPage = Math.max(1, page - Math.floor(PAGINATION_MAX_BUTTONS / 2));
            let endPage = Math.min(totalPages, startPage + PAGINATION_MAX_BUTTONS - 1);

            if (endPage - startPage < PAGINATION_MAX_BUTTONS - 1) {
                startPage = Math.max(1, endPage - PAGINATION_MAX_BUTTONS + 1);
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

    // 指定したコンテナ内のすべての画像の読み込み完了を待つ
    function waitForAllImagesLoaded(container) {
        return new Promise((resolve) => {
            const images = container.querySelectorAll('img');
            if (images.length === 0) {
                resolve();
                return;
            }

            let loadedCount = 0;
            const totalImages = images.length;

            const checkComplete = () => {
                loadedCount++;
                if (loadedCount >= totalImages) {
                    resolve();
                }
            };

            images.forEach((img) => {
                if (img.complete && img.naturalWidth > 0) {
                    checkComplete();
                } else if (img.src || img.dataset.src) {
                    img.addEventListener('load', checkComplete, { once: true });
                    img.addEventListener('error', checkComplete, { once: true });
                    // まだsrcがセットされていない場合は少し待って再確認
                    if (!img.src && img.dataset.src) {
                        setTimeout(() => {
                            if (!img.src) {
                                checkComplete();
                            }
                        }, 5000);
                    }
                } else {
                    checkComplete();
                }
            });

            // 安全のため、10秒後にタイムアウト
            setTimeout(() => {
                resolve();
            }, 10000);
        });
    }

    // 次のページのサムネイルを先読みする
    let prefetchedPages = new Set();

    async function prefetchNextPageThumbnails(nextPage, sortBy, unifiedQuery = '') {
        // 既に先読み済みの場合はスキップ
        const cacheKey = `${currentResolvedQuery}|${unifiedQuery}|${nextPage}|${sortBy}`;
        if (prefetchedPages.has(cacheKey)) {
            return;
        }
        prefetchedPages.add(cacheKey);

        // 先読みページ数が多くなりすぎないよう制限
        if (prefetchedPages.size > 10) {
            const firstKey = prefetchedPages.values().next().value;
            prefetchedPages.delete(firstKey);
        }

        try {
            // 次のページのデータを取得
            const data = await fetchSearchResults(
                currentResolvedQuery,
                currentQuery,
                nextPage,
                currentMinPages,
                currentMaxPages,
                sortBy,
                unifiedQuery
            );

            if (!data.results || data.results.length === 0) {
                return;
            }

            // サムネイルURLを抽出して先読み
            const thumbnailUrls = [];
            data.results.forEach((gallery) => {
                let imageUrl = '';
                if (Array.isArray(gallery.image_urls) && gallery.image_urls.length > 0) {
                    imageUrl = gallery.image_urls[0];
                } else if (typeof gallery.image_urls === 'string' && gallery.image_urls) {
                    imageUrl = gallery.image_urls;
                } else if (gallery.thumbnail_url) {
                    imageUrl = gallery.thumbnail_url;
                }

                if (imageUrl) {
                    const baseUrl = imageUrl.startsWith('/proxy/') ? imageUrl : `/proxy/${imageUrl}`;
                    const resolved = MangaApp.isMobile() ? `${baseUrl}?thumbnail=true&small=true` : baseUrl;
                    thumbnailUrls.push(resolved);
                }
            });

            // 画像を先読み（link prefetchまたはImageオブジェクトを使用）
            thumbnailUrls.forEach((url) => {
                // Imageオブジェクトで先読み
                const img = new Image();
                img.src = url;
            });

        } catch (error) {
            console.error('Prefetch error:', error);
        }
    }


    // URLにページ番号を反映する関数
    function updateUrlWithPage(page, query, unifiedQuery = '') {
        const url = new URL(window.location);

        // ページ番号を設定（1ページ目の場合は削除してURLをきれいに）
        if (page > 1) {
            url.searchParams.set('page', page.toString());
        } else {
            url.searchParams.delete('page');
        }

        // タグを設定
        if (query && !unifiedQuery) {
            url.searchParams.set('tag', query);
        } else {
            url.searchParams.delete('tag');
        }

        // 統一検索クエリを設定
        if (unifiedQuery) {
            url.searchParams.set('q', unifiedQuery);
        } else {
            url.searchParams.delete('q');
        }

        // URLを更新（履歴に追加）
        window.history.replaceState({ page, query }, '', url.toString());
    }
})();
