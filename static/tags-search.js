/**
 * 共通タグ検索・オートコンプリートユーティリティ
 * - data/tag-translations.json を読み込み
 * - Fuse.js であいまい検索
 * - 各ページ(tags/index/recommendations/rankings等)から再利用可能
 *
 * 期待するHTML側:
 * - 入力要素: input[data-tag-search-input]
 * - 結果コンテナ(任意): [data-tag-search-suggestions]
 *   存在しない場合は入力要素の直下に自動生成
 *
 * 使い方(各ページJS):
 * - window.TagSearch.initAutocomplete(inputElement, {
 *       onSelect(tag) { ...タグ決定時の処理... }
 *   });
 *
 * Fuse.js CDN例（各HTMLに追加想定）:
 * <script src="https://cdn.jsdelivr.net/npm/fuse.js@6.6.2/dist/fuse.min.js"></script>
 * <script src="/static/tags-search.js"></script>
 */
(function () {

    const DEFAULT_FUSE_OPTIONS = {
        // スコアが低いほどマッチ精度が高い。0.3〜0.5あたりが実用ライン
        threshold: 0.4,
        ignoreLocation: true,
        includeScore: true,
        minMatchCharLength: 1,
        keys: [
            // 英語タグ本体
            {
                name: 'tag',
                weight: 0.6,
            },
            // 翻訳(日本語)
            {
                name: 'translation',
                weight: 0.3,
            },
            // 説明・エイリアス(あいまい検索用)
            {
                name: 'description',
                weight: 0.05,
            },
            {
                name: 'aliases',
                weight: 0.05,
            },
        ],
    };

    const STATE = {
        loaded: false,
        loading: false,
        error: null,
        tags: [],
        fuse: null,
        // 取得元: /static/tag-translations.json または /api/tag-translations
        // まず静的JSONを試し、無ければAPIにフォールバックする戦略にしている
        source: 'auto',
    };

    function normalise(str) {
        return (str || '').toString().trim().toLowerCase();
    }

    async function fetchJson(url) {
        // キャッシュを完全に回避するため、タイムスタンプを付与しつつ cache: 'no-store' を指定
        const noCacheUrl = url + (url.includes('?') ? '&' : '?') + 'ts=' + Date.now();
        const res = await fetch(noCacheUrl, { cache: 'no-store' });
        if (!res.ok) {
            throw new Error('Failed to fetch ' + url + ' (' + res.status + ')');
        }
        return res.json();
    }

    function buildTagListFromApiPayload(payload) {
        // /api/tag-translations の形式に合わせて正規化
        // map: { [tag]: { translation, description, aliases } | string }
        if (!payload || typeof payload !== 'object') return [];
        const map = payload.translations || payload; // 柔軟に対応
        const result = [];
        for (const [tag, value] of Object.entries(map)) {
            if (!tag) continue;
            const entry = (value && typeof value === 'object') ? value : { translation: value };
            const translation = typeof entry.translation === 'string' ? entry.translation : '';
            const description = typeof entry.description === 'string' ? entry.description : '';
            const aliases = Array.isArray(entry.aliases) ? entry.aliases : [];
            result.push({
                tag,
                translation,
                description,
                aliases,
            });
        }
        return result;
    }

    async function ensureLoaded() {
        if (STATE.loaded || STATE.loading) {
            // loading中でも呼び出し元はawaitで待てる
            while (STATE.loading) {
                // 簡易スピン (数msスリープ)
                // 実運用上問題ない程度の軽さを想定
                // eslint-disable-next-line no-await-in-loop
                await new Promise((r) => setTimeout(r, 10));
            }
            if (!STATE.loaded && STATE.error) {
                throw STATE.error;
            }
            return;
        }

        STATE.loading = true;
        STATE.error = null;

        try {
            let tags = [];

            // 優先: 静的JSON (存在すればCDNキャッシュなども効きやすい)
            try {
                const staticData = await fetchJson('/static/tag-translations.json');
                tags = buildTagListFromApiPayload(staticData);
                STATE.source = 'static';
            } catch (e) {
                // フォールバック: API
                const apiData = await fetchJson('/api/tag-translations');
                tags = buildTagListFromApiPayload(apiData);
                STATE.source = 'api';
            }

            STATE.tags = tags;

            if (!window.Fuse) {
                throw new Error('Fuse.js が読み込まれていません。各HTMLに CDN スクリプトを追加してください。');
            }

            STATE.fuse = new window.Fuse(tags, DEFAULT_FUSE_OPTIONS);
            STATE.loaded = true;
        } catch (err) {
            STATE.error = err;
            console.error('[TagSearch] 初期化に失敗しました:', err);
            throw err;
        } finally {
            STATE.loading = false;
        }
    }

    /**
     * 生タグリスト取得 (必要ならページ側の高度な制御用)
     */
    async function getAllTags() {
        await ensureLoaded();
        return STATE.tags.slice();
    }

    /**
     * 日本語表示タグなどから英語タグへ逆変換するためのマップを構築
     * - translation完全一致、および aliases 完全一致で英語タグを引けるようにする
     * - 大文字小文字は無視
     */
    function buildReverseMap() {
        const map = new Map();
        for (const item of STATE.tags) {
            const tag = item.tag;
            if (!tag) continue;

            // 日本語訳
            if (item.translation) {
                const key = normalise(item.translation);
                if (key && !map.has(key)) {
                    map.set(key, tag);
                }
            }

            // エイリアス
            if (Array.isArray(item.aliases)) {
                for (const alias of item.aliases) {
                    const key = normalise(alias);
                    if (key && !map.has(key)) {
                        map.set(key, tag);
                    }
                }
            }
        }
        return map;
    }

    /**
     * 与えられた検索語から API 用の英語タグへ正規化
     * - 日本語訳/エイリアスに完全一致した場合は対応する英語タグを返す
     * - 一致しない場合は元の文字列をそのまま返す（既に英語タグ入力の場合など）
     * @param {string} input
     * @returns {Promise<string>}
     */
    async function normalizeQueryToEnglishTag(input) {
        const raw = (input || '').trim();
        if (!raw) return '';

        await ensureLoaded();

        const lower = normalise(raw);
        const reverseMap = buildReverseMap();

        const mapped = reverseMap.get(lower);
        if (mapped) {
            return mapped;
        }

        // 既に英語タグを直接入力している場合もあるので、そのまま返す
        return raw;
    }

    /**
     * Fuse.js を使った検索
     * @param {string} query
     * @param {object} [options]
     * @param {number} [options.limit=20]
     * @returns {Promise<Array<{tag, translation, description, aliases, score}>>}
     */
    async function search(query, options) {
        const q = normalise(query);
        const limit = (options && options.limit) || 20;
        if (!q) return [];

        await ensureLoaded();
        if (!STATE.fuse) return [];

        const results = STATE.fuse.search(q, { limit });
        return results.map((r) => ({
            ...r.item,
            score: r.score,
        }));
    }

    // 公開API（既存オブジェクトがあれば拡張、なければ新規作成）
    window.TagSearch = Object.assign(window.TagSearch || {}, {
        _state: STATE,
        ensureLoaded,
        getAllTags,
        search,
        initAutocomplete,
        normalizeQueryToEnglishTag,
    });

    /**
     * 入力ボックスにオートコンプリートを付与
     * @param {HTMLInputElement} input
     * @param {object} config
     * @param {(tag: string, meta: object) => void} config.onSelect タグ決定時のコールバック (必須)
     * @param {number} [config.minLength=1] 検索開始の最小文字数
     * @param {number} [config.limit=10] 表示候補数
     * @param {HTMLElement} [config.container] 候補表示先(省略時はinputの直下に生成)
     */
    function initAutocomplete(input, config) {
        const cfg = Object.assign(
            {
                minLength: 1,
                limit: 10,
            },
            config || {}
        );
        if (!input || typeof cfg.onSelect !== 'function') {
            console.error('[TagSearch] initAutocomplete: input または onSelect が不正です');
            return;
        }

        let listEl = cfg.container || null;
        let currentIndex = -1;
        let lastQuery = '';

        function ensureListEl() {
            if (listEl && listEl.parentNode) return listEl;
            listEl = document.createElement('div');
            listEl.className = 'tag-autocomplete-list';
            listEl.style.position = 'absolute';
            listEl.style.zIndex = '9999';
            listEl.style.background = 'var(--bg-card, #111827)';
            listEl.style.color = 'var(--text-primary, #f9fafb)';
            listEl.style.border = '1px solid rgba(148, 163, 184, 0.35)';
            listEl.style.boxShadow = '0 10px 25px rgba(15,23,42,0.35)';
            listEl.style.borderRadius = '10px';
            listEl.style.marginTop = '4px';
            listEl.style.padding = '4px 0';
            listEl.style.maxHeight = '260px';
            listEl.style.overflowY = 'auto';
            listEl.style.fontSize = '13px';

            // input の直下に相対配置
            const wrapper = document.createElement('div');
            wrapper.style.position = 'relative';
            input.parentNode.insertBefore(wrapper, input);
            wrapper.appendChild(input);
            wrapper.appendChild(listEl);
            return listEl;
        }

        function clearList() {
            if (!listEl) return;
            listEl.innerHTML = '';
            listEl.style.display = 'none';
            currentIndex = -1;
        }

        function renderResults(items) {
            const target = ensureListEl();
            target.innerHTML = '';
            currentIndex = -1;

            if (!items.length) {
                target.style.display = 'none';
                return;
            }

            items.forEach((item, index) => {
                const el = document.createElement('div');
                el.className = 'tag-autocomplete-item';
                el.style.padding = '6px 10px';
                el.style.cursor = 'pointer';
                el.style.display = 'flex';
                el.style.flexDirection = 'column';
                el.style.gap = '2px';

                const primary = document.createElement('div');
                primary.style.display = 'flex';
                primary.style.justifyContent = 'space-between';
                primary.style.gap = '8px';

                const left = document.createElement('div');
                left.style.fontWeight = '600';
                left.textContent = item.translation || item.tag;

                const right = document.createElement('div');
                right.style.fontSize = '11px';
                right.style.color = 'var(--text-secondary, #9ca3af)';
                right.textContent = item.tag;

                primary.appendChild(left);
                primary.appendChild(right);
                el.appendChild(primary);

                if (item.description) {
                    const desc = document.createElement('div');
                    desc.style.fontSize = '11px';
                    desc.style.color = 'var(--text-secondary, #9ca3af)';
                    desc.textContent = item.description;
                    el.appendChild(desc);
                }

                el.addEventListener('mouseenter', () => {
                    setActive(index);
                });

                el.addEventListener('mousedown', (ev) => {
                    ev.preventDefault(); // input blur 防止
                    selectItem(item);
                });

                target.appendChild(el);
            });

            target.style.display = 'block';
        }

        function setActive(index) {
            if (!listEl) return;
            const children = listEl.querySelectorAll('.tag-autocomplete-item');
            if (!children.length) return;
            children.forEach((c) => {
                c.style.background = 'transparent';
                c.style.color = 'inherit';
            });
            if (index < 0 || index >= children.length) {
                currentIndex = -1;
                return;
            }
            currentIndex = index;
            const active = children[index];
            active.style.background = 'rgba(37,99,235,0.16)';
            active.style.color = 'var(--accent, #60a5fa)';
            active.scrollIntoView({ block: 'nearest' });
        }

        function move(delta) {
            if (!listEl) return;
            const children = listEl.querySelectorAll('.tag-autocomplete-item');
            if (!children.length) return;
            let next = currentIndex + delta;
            if (next < 0) next = children.length - 1;
            if (next >= children.length) next = 0;
            setActive(next);
        }

        function selectActive() {
            if (!listEl) return;
            const children = listEl.querySelectorAll('.tag-autocomplete-item');
            if (!children.length) return;
            const index = currentIndex >= 0 ? currentIndex : 0;
            const el = children[index];
            const tag = el && el.__tagItem;
            if (!tag) return;
            selectItem(tag);
        }

        function selectItem(item) {
            clearList();
            lastQuery = '';
            cfg.onSelect(item.tag, item);
        }

        async function handleInput() {
            const value = input.value;
            const q = value.trim();
            lastQuery = value;

            if (!q || q.length < cfg.minLength) {
                clearList();
                return;
            }

            try {
                const items = await search(q, { limit: cfg.limit });
                // 入力中に別のクエリになっていたら破棄
                if (lastQuery !== value) return;
                renderResults(items);
                // 各DOM要素に逆参照を付与して Enter 選択で拾えるように
                if (listEl) {
                    const children = listEl.querySelectorAll('.tag-autocomplete-item');
                    items.forEach((item, idx) => {
                        if (children[idx]) {
                            children[idx].__tagItem = item;
                        }
                    });
                }
            } catch (e) {
                console.error('[TagSearch] 検索中にエラー:', e);
                clearList();
            }
        }

        function handleKeydown(ev) {
            if (!listEl || listEl.style.display === 'none') {
                if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
                    // 初回 ArrowDown で検索開始してもよいが、ここでは何もしない
                }
                if (ev.key === 'Enter') {
                    // 候補なしでEnterされた場合は input の値をそのまま使いたいケースもあるため
                    // ここでは何もしない（ページ側 onSelect の設計次第）
                }
                return;
            }

            if (ev.key === 'ArrowDown') {
                ev.preventDefault();
                move(1);
            } else if (ev.key === 'ArrowUp') {
                ev.preventDefault();
                move(-1);
            } else if (ev.key === 'Enter') {
                ev.preventDefault();
                selectActive();
            } else if (ev.key === 'Escape') {
                ev.preventDefault();
                clearList();
            }
        }

        function handleBlur() {
            // 少し遅延させて mousedown(on候補) を拾えるように
            setTimeout(() => {
                clearList();
            }, 150);
        }

        input.addEventListener('input', handleInput);
        input.addEventListener('keydown', handleKeydown);
        input.addEventListener('blur', handleBlur);
    }

    window.TagSearch = {
        ensureLoaded,
        getAllTags,
        search,
        initAutocomplete,
        _state: STATE,
    };
})();