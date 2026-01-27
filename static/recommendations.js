(function () {
    let grid;
    let loading;
    let errorMessage;
    let summaryCount;
    let hiddenSummary;
    let lastSessionIdUsed = null;

    document.addEventListener('DOMContentLoaded', () => {
        grid = document.getElementById('recommendationGrid');
        loading = document.getElementById('loadingIndicator');
        errorMessage = document.getElementById('errorMessage');
        summaryCount = document.getElementById('summaryCount');
        hiddenSummary = document.getElementById('hiddenTagsSummary');
        const themeToggle = document.getElementById('themeToggle');
        const refreshButton = document.getElementById('refreshButton');

        MangaApp.applyThemeToDocument(document);
        MangaApp.ensureTranslations().catch(() => { });
        updateThemeToggleIcon(MangaApp.getPreferredTheme());
        themeToggle.addEventListener('click', () => {
            const theme = MangaApp.toggleTheme();
            updateThemeToggleIcon(theme);
        });

        refreshButton.addEventListener('click', () => loadRecommendations(grid, loading, errorMessage, summaryCount, hiddenSummary));
        document.addEventListener('manga:hidden-tags-change', () => {
            updateHiddenSummary(hiddenSummary);
            loadRecommendations(grid, loading, errorMessage, summaryCount, hiddenSummary);
        });

        document.addEventListener('tracking:session-ready', (event) => {
            const sessionId = event && event.detail ? event.detail.sessionId : null;
            if (!sessionId || sessionId === lastSessionIdUsed) {
                return;
            }
            if (!grid) {
                return;
            }
            loadRecommendations(grid, loading, errorMessage, summaryCount, hiddenSummary);
        });

        updateHiddenSummary(hiddenSummary);
        loadRecommendations(grid, loading, errorMessage, summaryCount, hiddenSummary);
    });

    function updateThemeToggleIcon(theme) {
        const toggle = document.getElementById('themeToggle');
        if (!toggle) return;
        toggle.innerHTML = theme === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }

    function updateHiddenSummary(element) {
        const hidden = MangaApp.getHiddenTags();
        element.textContent = hidden.length ? `除外タグ: ${hidden.join(', ')}` : '除外タグなし';
    }

    function getTrackingSessionId() {
        if (window.Tracking && typeof window.Tracking.getSessionId === 'function') {
            return window.Tracking.getSessionId();
        }
        return null;
    }

    async function loadRecommendations(grid, loading, errorMessage, summaryCount, hiddenSummary) {
        showLoading(loading, true);
        errorMessage.style.display = 'none';
        try {
            const params = new URLSearchParams();
            params.append('limit', '24');
            const hidden = MangaApp.getHiddenTags();
            if (hidden.length) {
                params.append('exclude_tag', hidden.join(','));
            }
            const sessionId = getTrackingSessionId();
            if (sessionId) {
                params.append('session_id', sessionId);
            }
            const response = await fetch(`/api/recommendations?${params.toString()}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            renderRecommendations(grid, data.results || []);
            summaryCount.textContent = `${(data.results || []).length} 件を表示中`;
            lastSessionIdUsed = sessionId || null;
        } catch (error) {
            console.error('おすすめ取得エラー', error);
            errorMessage.textContent = 'おすすめを取得できませんでした。時間をおいて再度お試しください。';
            errorMessage.style.display = 'block';
            grid.innerHTML = '';
            summaryCount.textContent = '0 件を表示中';
            lastSessionIdUsed = null;
        } finally {
            showLoading(loading, false);
        }
    }

    function renderRecommendations(grid, items) {
        grid.innerHTML = '';
        const fragment = document.createDocumentFragment();
        items.forEach((item) => {
            const card = createRecommendationCard(item);
            fragment.appendChild(card);
        });
        grid.appendChild(fragment);
    }

    function createRecommendationCard(gallery) {
        const card = document.createElement('a');
        card.href = `/viewer?id=${gallery.gallery_id}`;
        card.className = 'card';
        card.dataset.galleryId = gallery.gallery_id;

        const thumbnail = document.createElement('div');
        thumbnail.className = 'card-thumbnail';
        const img = document.createElement('img');
        const firstImage = Array.isArray(gallery.image_urls) && gallery.image_urls.length > 0 ? gallery.image_urls[0] : '';
        if (firstImage) {
            const baseUrl = firstImage.startsWith('/proxy/') ? firstImage : `/proxy/${firstImage}`;
            const thumbUrl = `${baseUrl}?thumbnail=true&small=true`;

            if (MangaApp.isMobile()) {
                img.src = thumbUrl;
            } else {
                img.src = thumbUrl;
                const originalImg = new Image();
                originalImg.src = baseUrl;
                originalImg.onload = () => {
                    img.src = originalImg.src;
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
            const toggled = MangaApp.toggleLike(gallery.gallery_id);
            likeButton.classList.toggle('active', toggled);
            likeButton.innerHTML = toggled ? '<i class="fas fa-heart"></i>' : '<i class="far fa-heart"></i>';
        });

        const info = document.createElement('div');
        info.className = 'card-info';
        const title = document.createElement('h3');
        title.className = 'card-title';
        title.textContent = gallery.japanese_title || '無題';
        const meta = document.createElement('div');
        meta.className = 'card-meta';
        if (gallery.page_count) {
            meta.textContent = `${gallery.page_count}ページ`;
        }
        const tags = document.createElement('div');
        tags.className = 'card-tags';
        const tagArray = parseArray(gallery.tags);
        tagArray.slice(0, 4).forEach((tag) => {
            if (!tag) return;
            const chip = document.createElement('span');
            chip.className = 'tag-chip';
            chip.textContent = MangaApp.translateTag(tag);
            chip.title = tag;
            tags.appendChild(chip);
        });

        info.appendChild(title);
        info.appendChild(meta);
        info.appendChild(tags);

        card.appendChild(thumbnail);
        card.appendChild(likeButton);
        card.appendChild(info);

        card.addEventListener('click', () => {
            MangaApp.addHistoryEntry({
                gallery_id: gallery.gallery_id,
                japanese_title: gallery.japanese_title,
                image_urls: gallery.image_urls,
                page_count: gallery.page_count,
            });
        });

        return card;
    }

    function parseArray(value) {
        if (Array.isArray(value)) return value;
        if (typeof value === 'string') {
            try {
                const parsed = JSON.parse(value);
                return Array.isArray(parsed) ? parsed : value.split(/[\s,]+/).filter(Boolean);
            } catch (error) {
                return value.split(/[\s,]+/).filter(Boolean);
            }
        }
        return [];
    }

    function showLoading(element, visible) {
        element.style.display = visible ? 'block' : 'none';
    }
})();
