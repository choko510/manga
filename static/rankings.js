document.addEventListener('DOMContentLoaded', () => {
    const rankingsSection = document.getElementById('rankingsSection');
    const searchSection = document.getElementById('searchSection');
    const rankingTabs = document.querySelectorAll('.ranking-tab');
    const rankingGrid = document.getElementById('rankingGrid');
    const rankingLoadingIndicator = document.getElementById('rankingLoadingIndicator');
    const rankingLoadMoreContainer = document.getElementById('rankingLoadMoreContainer');
    const rankingLoadMoreButton = document.getElementById('rankingLoadMoreButton');
    
    let currentRankingType = 'daily';
    let currentRankingOffset = 0;
    let hasMoreRankings = true;
    let isLoadingRankings = false;

    // ランキングタブのクリックイベント
    rankingTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (isLoadingRankings) return;
            
            // アクティブタブを更新
            rankingTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            currentRankingType = tab.dataset.type;
            currentRankingOffset = 0;
            hasMoreRankings = true;
            loadRankings();
        });
    });

    // さらに表示ボタンのクリックイベント
    if (rankingLoadMoreButton) {
        rankingLoadMoreButton.addEventListener('click', loadMoreRankings);
    }

    // ランキングデータを読み込む関数
    async function loadRankings(reset = true) {
        if (isLoadingRankings) return;
        
        isLoadingRankings = true;
        rankingLoadingIndicator.style.display = 'block';
        
        if (rankingLoadMoreContainer) {
            rankingLoadMoreContainer.style.display = 'none';
        }
        
        try {
            const limit = 20; // 1回の表示数
            // 統合されたsearchエンドポイントを使用
            const params = new URLSearchParams();
            params.append('sort_by', currentRankingType);
            params.append('limit', limit.toString());
            params.append('offset', currentRankingOffset.toString());
            
            const response = await fetch(`/search?${params.toString()}`);
            const data = await response.json();
            
            if (reset) {
                rankingGrid.innerHTML = '';
            }
            
            // ランキングカードを生成
            data.results.forEach((gallery, index) => {
                const card = createRankingCard(gallery, currentRankingOffset + index + 1);
                rankingGrid.appendChild(card);
            });
            
            currentRankingOffset += data.results.length;
            hasMoreRankings = data.has_more;
            
            // さらに表示ボタンの表示/非表示
            if (rankingLoadMoreContainer) {
                if (hasMoreRankings) {
                    rankingLoadMoreContainer.style.display = 'block';
                } else {
                    rankingLoadMoreContainer.style.display = 'none';
                }
            }
        } catch (error) {
            console.error('ランキングの読み込みに失敗しました:', error);
        } finally {
            isLoadingRankings = false;
            rankingLoadingIndicator.style.display = 'none';
        }
    }

    // さらに表示関数
    function loadMoreRankings() {
        if (!hasMoreRankings || isLoadingRankings) return;
        loadRankings(false);
    }

    // ランキングカードを生成する関数
    function createRankingCard(gallery, rank) {
        const card = document.createElement('div');
        card.className = 'card';
        card.setAttribute('data-gallery-id', gallery.gallery_id);
        
        // ランク番号を追加
        const rankNumber = document.createElement('div');
        rankNumber.className = 'rank-number';
        rankNumber.textContent = rank;
        card.appendChild(rankNumber);
        
        // サムネイル
        const thumbnail = document.createElement('div');
        thumbnail.className = 'card-thumbnail';
        
        const placeholder = document.createElement('div');
        placeholder.className = 'image-placeholder';
        thumbnail.appendChild(placeholder);
        
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
            const img = document.createElement('img');
            const resolved = firstImage.startsWith('/proxy/') ? firstImage : `/proxy/${firstImage}`;
            img.src = resolved;
            img.alt = gallery.japanese_title || 'ギャラリー';
            img.loading = 'lazy';
            img.onload = () => {
                const placeholderElement = thumbnail.querySelector('.image-placeholder');
                if (placeholderElement) {
                    placeholderElement.remove();
                }
            };
            thumbnail.appendChild(img);
        }
        
        card.appendChild(thumbnail);
        
        // カード情報
        const cardInfo = document.createElement('div');
        cardInfo.className = 'card-info';
        
        // タイトル
        const title = document.createElement('h3');
        title.className = 'card-title';
        title.textContent = gallery.japanese_title || 'タイトルなし';
        cardInfo.appendChild(title);
        
        // メタ情報
        const cardMeta = document.createElement('div');
        cardMeta.className = 'card-meta';
        cardMeta.textContent = `${gallery.page_count || 0} ページ`;
        cardInfo.appendChild(cardMeta);
        
        // タグ
        if (gallery.tags) {
            try {
                const tags = JSON.parse(gallery.tags);
                if (tags && tags.length > 0) {
                    const tagList = document.createElement('ul');
                    tagList.className = 'card-tags';
                    
                    // 最大5タグまで表示
                    tags.slice(0, 5).forEach(tag => {
                        const tagItem = document.createElement('li');
                        tagItem.className = 'tag-chip';
                        
                        const tagSpan = document.createElement('span');
                        tagSpan.className = 'tag-jp';
                        tagSpan.textContent = tag;
                        
                        tagItem.appendChild(tagSpan);
                        tagList.appendChild(tagItem);
                    });
                    
                    cardInfo.appendChild(tagList);
                }
            } catch (e) {
                console.error('タグの解析に失敗しました:', e);
            }
        }
        
        card.appendChild(cardInfo);
        
        // カードクリックイベント
        card.addEventListener('click', () => {
            window.location.href = `/viewer?id=${gallery.gallery_id}`;
        });
        
        return card;
    }

    // ナビゲーションのランキングリンクをクリックした場合
    const rankingsNavLink = document.querySelector('a[href="/rankings"]');
    if (rankingsNavLink) {
        rankingsNavLink.addEventListener('click', (e) => {
            e.preventDefault();
            
            // ランキングセクションを表示
            rankingsSection.style.display = 'block';
            searchSection.style.display = 'none';
            
            // ナビゲーションのアクティブ状態を更新
            document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
            rankingsNavLink.classList.add('active');
            
            // 初回のランキングデータを読み込み
            if (rankingGrid.children.length === 0) {
                loadRankings();
            }
        });
    }
});