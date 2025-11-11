// 新トラッキングスクリプト
// 要件:
// - FingerprintJS + localStorageで一貫した user_id を生成・保持
// - セッションIDを管理し、/api/tracking/* に対して送信
// - viewer.html と連携して、閲覧した漫画ID・ページ数・閲覧時間を記録
// - よく使う検索タグも記録
// - 旧tracking.js実装は破棄し、このファイルで統一的に扱う

(function () {
    "use strict";

    // =========================
    // 定数
    // =========================

    const CONFIG = {
        // APIエンドポイントのベース
        baseUrl: "",
        api: {
            identify: "/api/tracking/identify",
            session: "/api/tracking/session",
            mangaViewStart: "/api/tracking/manga-view/start",
            mangaViewUpdate: "/api/tracking/manga-view/update",
            mangaViewEnd: "/api/tracking/manga-view/end",
            search: "/api/tracking/search",
            events: "/api/tracking/events"
        },
        // FingerprintJS CDN
        fingerprintJsCdn: "https://cdn.jsdelivr.net/npm/@fingerprintjs/fingerprintjs@3/dist/fp.min.js",
        // localStorage keys
        storageKeys: {
            userId: "tracking_user_id",
            userIdSource: "tracking_user_id_source",
            sessionId: "tracking_session_id",
            lastSessionTouched: "tracking_session_last_touched",
            viewerState: "tracking_manga_viewer_state"
        },
        // セッション有効秒数（一定期間操作がなければ新セッション扱い）
        sessionTtlSeconds: 60 * 30, // 30分
        // ビューア更新送信間隔
        viewerUpdateIntervalSec: 15,
        // ページ離脱時は sendBeacon を優先
        useSendBeacon: true,
        // デバッグログ
        debug: false
    };

    // =========================
    // 内部ユーティリティ
    // =========================

    function nowIso() {
        return new Date().toISOString();
    }

    function logDebug(...args) {
        if (CONFIG.debug) {
            console.debug("[Tracking]", ...args);
        }
    }

    function safeJsonParse(str, fallback) {
        try {
            return JSON.parse(str);
        } catch {
            return fallback;
        }
    }

    function randomId(prefix) {
        return (
            prefix +
            "_" +
            Math.random().toString(36).slice(2) +
            "_" +
            Date.now().toString(36)
        );
    }

    // =========================
    // Fingerprint / user_id 管理
    // =========================

    let userId = null;
    let userIdSource = null;
    let userIdReady = false;
    let userIdReadyCallbacks = [];

    function getStoredUserId() {
        try {
            const id = localStorage.getItem(CONFIG.storageKeys.userId);
            const src = localStorage.getItem(CONFIG.storageKeys.userIdSource);
            if (id && typeof id === "string") {
                return { id, source: src || "unknown" };
            }
        } catch {
            // ignore
        }
        return null;
    }

    function storeUserId(id, source) {
        try {
            localStorage.setItem(CONFIG.storageKeys.userId, id);
            localStorage.setItem(CONFIG.storageKeys.userIdSource, source || "unknown");
        } catch {
            // ignore
        }
    }

    function resolveUserIdSyncFallback() {
        const existing = getStoredUserId();
        if (existing) {
            userId = existing.id;
            userIdSource = existing.source;
            return;
        }
        const generated = randomId("u");
        userId = generated;
        userIdSource = "generated";
        storeUserId(userId, userIdSource);
    }

    async function loadFingerprintJsIfNeeded() {
        if (typeof FingerprintJS !== "undefined") {
            return;
        }
        await new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = CONFIG.fingerprintJsCdn;
            script.async = true;
            script.onload = () => resolve();
            script.onerror = () => reject(new Error("FingerprintJS load error"));
            document.head.appendChild(script);
        });
    }

    async function initUserId() {
        // 既に完了していれば即返す
        if (userIdReady && userId) {
            return userId;
        }

        // 先にlocalStorage確認
        const existing = getStoredUserId();
        if (existing) {
            userId = existing.id;
            userIdSource = existing.source;
            userIdReady = true;
            notifyUserIdReady();
            // identify送信（非同期）
            identifyUser().catch(() => {});
            return userId;
        }

        // FingerprintJS試行
        try {
            await loadFingerprintJsIfNeeded();
            if (typeof FingerprintJS !== "undefined") {
                const fp = await FingerprintJS.load();
                const result = await fp.get();
                if (result && result.visitorId) {
                    userId = "fp_" + result.visitorId;
                    userIdSource = "fingerprint";
                    storeUserId(userId, userIdSource);
                }
            }
        } catch (e) {
            logDebug("Fingerprint init failed", e);
        }

        // Fingerprintで決まらなければフォールバック
        if (!userId) {
            resolveUserIdSyncFallback();
        }

        userIdReady = true;
        notifyUserIdReady();

        // identify送信（非同期）
        identifyUser().catch(() => {});

        return userId;
    }

    function onUserIdReady(cb) {
        if (userIdReady && userId) {
            cb(userId);
        } else {
            userIdReadyCallbacks.push(cb);
        }
    }

    function notifyUserIdReady() {
        userIdReadyCallbacks.splice(0).forEach((cb) => {
            try {
                cb(userId);
            } catch {
                // ignore
            }
        });
    }

    // =========================
    // セッション管理
    // =========================

    let sessionId = null;

    function getStoredSession() {
        try {
            const id = localStorage.getItem(CONFIG.storageKeys.sessionId);
            const last = parseInt(
                localStorage.getItem(CONFIG.storageKeys.lastSessionTouched) || "0",
                10
            );
            if (id && last > 0) {
                return { id, last };
            }
        } catch {
            // ignore
        }
        return null;
    }

    function touchSession(id) {
        try {
            localStorage.setItem(CONFIG.storageKeys.sessionId, id);
            localStorage.setItem(
                CONFIG.storageKeys.lastSessionTouched,
                String(Math.floor(Date.now() / 1000))
            );
        } catch {
            // ignore
        }
    }

    function ensureSessionId() {
        const nowSec = Math.floor(Date.now() / 1000);
        const stored = getStoredSession();
        if (stored) {
            const age = nowSec - stored.last;
            if (age <= CONFIG.sessionTtlSeconds) {
                sessionId = stored.id;
                touchSession(sessionId);
                return sessionId;
            }
        }
        sessionId = randomId("s");
        touchSession(sessionId);
        return sessionId;
    }

    async function ensureSessionRegistered() {
        const uid = await initUserId();
        const sid = ensureSessionId();
        const payload = {
            user_id: uid,
            session_id: sid,
            user_agent: navigator.userAgent || "",
            referrer: document.referrer || ""
        };
        return sendJson("POST", CONFIG.api.session, payload, { ignoreError: true });
    }

    // =========================
    // 通信ユーティリティ
    // =========================

    function buildUrl(path) {
        if (!path.startsWith("/")) return path;
        if (!CONFIG.baseUrl) return path;
        return CONFIG.baseUrl.replace(/\/+$/, "") + path;
    }

    async function sendJson(method, path, body, options) {
        const url = buildUrl(path);
        const opt = options || {};
        const payload = body != null ? JSON.stringify(body) : null;

        // ページ離脱時などに sendBeacon を使いたいケース
        if (
            opt.beacon &&
            typeof navigator !== "undefined" &&
            typeof navigator.sendBeacon === "function" &&
            payload &&
            method === "POST"
        ) {
            try {
                const success = navigator.sendBeacon(
                    url,
                    new Blob([payload], { type: "application/json" })
                );
                if (success) {
                    return { ok: true, beacon: true };
                }
            } catch {
                // フォールバックでfetch
            }
        }

        try {
            const res = await fetch(url, {
                method,
                headers: {
                    "Content-Type": "application/json"
                },
                body: payload
            });
            if (!res.ok) {
                if (!opt.ignoreError) {
                    logDebug("Tracking request failed", method, path, res.status);
                }
                return { ok: false, status: res.status };
            }
            const ct = res.headers.get("content-type") || "";
            if (ct.includes("application/json")) {
                return await res.json();
            }
            return { ok: true };
        } catch (e) {
            if (!opt.ignoreError) {
                logDebug("Tracking request error", method, path, e);
            }
            return { ok: false, error: String(e) };
        }
    }

    // =========================
    // /api/tracking 実装
    // =========================

    async function identifyUser() {
        if (!userId) {
            await initUserId();
        }
        const payload = {
            user_id: userId,
            fingerprint: userIdSource === "fingerprint" ? userId : null,
            user_agent: navigator.userAgent || ""
        };
        return sendJson("POST", CONFIG.api.identify, payload, { ignoreError: true });
    }

    // =========================
    // ビューアトラッキング
    // =========================

    let currentMangaView = {
        id: null, // manga_view_id
        mangaId: null,
        pageCount: null,
        startedAt: null,
        lastUpdateAt: null,
        maxPage: 0,
        lastPage: 0,
        accumSec: 0
    };

    let viewerUpdateTimer = null;
    let viewerLastTick = null;

    function loadViewerStateFromStorage() {
        try {
            const raw = localStorage.getItem(CONFIG.storageKeys.viewerState);
            if (!raw) return;
            const st = safeJsonParse(raw, null);
            if (!st || typeof st !== "object") return;
            currentMangaView.id = st.id || null;
            currentMangaView.mangaId = st.mangaId || null;
            currentMangaView.pageCount = st.pageCount || null;
            currentMangaView.startedAt = st.startedAt || null;
            currentMangaView.lastUpdateAt = st.lastUpdateAt || null;
            currentMangaView.maxPage = st.maxPage || 0;
            currentMangaView.lastPage = st.lastPage || 0;
            currentMangaView.accumSec = st.accumSec || 0;
        } catch {
            // ignore
        }
    }

    function saveViewerStateToStorage() {
        try {
            if (!currentMangaView.id) {
                localStorage.removeItem(CONFIG.storageKeys.viewerState);
                return;
            }
            const st = {
                id: currentMangaView.id,
                mangaId: currentMangaView.mangaId,
                pageCount: currentMangaView.pageCount,
                startedAt: currentMangaView.startedAt,
                lastUpdateAt: currentMangaView.lastUpdateAt,
                maxPage: currentMangaView.maxPage,
                lastPage: currentMangaView.lastPage,
                accumSec: currentMangaView.accumSec
            };
            localStorage.setItem(
                CONFIG.storageKeys.viewerState,
                JSON.stringify(st)
            );
        } catch {
            // ignore
        }
    }

    function resetViewerState() {
        currentMangaView = {
            id: null,
            mangaId: null,
            pageCount: null,
            startedAt: null,
            lastUpdateAt: null,
            maxPage: 0,
            lastPage: 0,
            accumSec: 0
        };
        if (viewerUpdateTimer) {
            clearInterval(viewerUpdateTimer);
            viewerUpdateTimer = null;
        }
        viewerLastTick = null;
        try {
            localStorage.removeItem(CONFIG.storageKeys.viewerState);
        } catch {
            // ignore
        }
    }

    function viewerTickAccum() {
        if (!currentMangaView.id) return;
        const now = Date.now();
        if (!viewerLastTick) {
            viewerLastTick = now;
            return;
        }
        const diffSec = Math.max(0, Math.floor((now - viewerLastTick) / 1000));
        if (diffSec > 0) {
            currentMangaView.accumSec += diffSec;
            viewerLastTick = now;
        }
    }

    async function startMangaView(params) {
        const mangaId = Number(params && params.mangaId);
        const pageCount = Number(params && params.pageCount) || null;
        if (!mangaId || Number.isNaN(mangaId)) {
            logDebug("startMangaView: invalid mangaId", params);
            return;
        }

        await ensureSessionRegistered();
        const uid = userId || (await initUserId());
        const sid = sessionId || ensureSessionId();

        resetViewerState();

        currentMangaView.mangaId = mangaId;
        currentMangaView.pageCount = pageCount;
        currentMangaView.startedAt = nowIso();
        currentMangaView.lastUpdateAt = currentMangaView.startedAt;
        currentMangaView.maxPage = 0;
        currentMangaView.lastPage = 0;
        currentMangaView.accumSec = 0;

        const payload = {
            user_id: uid,
            session_id: sid,
            manga_id: mangaId,
            page_count: pageCount,
            started_at: currentMangaView.startedAt
        };

        const res = await sendJson(
            "POST",
            CONFIG.api.mangaViewStart,
            payload,
            { ignoreError: true }
        );

        if (res && (res.manga_view_id || res.id)) {
            currentMangaView.id = res.manga_view_id || res.id;
            saveViewerStateToStorage();
            viewerLastTick = Date.now();
            if (!viewerUpdateTimer) {
                viewerUpdateTimer = setInterval(async () => {
                    try {
                        await updateMangaView({ reason: "interval" });
                    } catch {
                        // ignore
                    }
                }, CONFIG.viewerUpdateIntervalSec * 1000);
            }
            logDebug("manga-view started", currentMangaView);
        } else {
            // IDが取れない場合はサーバ側記録なしでローカルのみ
            logDebug("manga-view start failed or no id returned", res);
        }
    }

    async function updateMangaView(options) {
        if (!currentMangaView.id) return;

        viewerTickAccum();

        const now = nowIso();
        const increment =
            currentMangaView.accumSec > 0
                ? currentMangaView.accumSec
                : undefined;

        const payload = {
            manga_view_id: currentMangaView.id,
            current_page: currentMangaView.lastPage || 0,
            max_page: currentMangaView.maxPage || 0,
            now,
            increment_duration_sec: increment
        };

        // サーバ側で加算型更新を想定
        await sendJson(
            "POST",
            CONFIG.api.mangaViewUpdate,
            payload,
            { ignoreError: true }
        );

        currentMangaView.lastUpdateAt = now;
        currentMangaView.accumSec = 0;
        saveViewerStateToStorage();

        if (options && options.reason) {
            logDebug("manga-view update", options.reason, payload);
        }
    }

    async function endMangaView(options) {
        if (!currentMangaView.id) {
            resetViewerState();
            return;
        }

        viewerTickAccum();

        const endedAt = nowIso();
        const finalPage = currentMangaView.lastPage || currentMangaView.maxPage || 0;
        const totalDuration =
            (currentMangaView.accumSec || 0);

        const payload = {
            manga_view_id: currentMangaView.id,
            ended_at: endedAt,
            final_page: finalPage,
            total_duration_sec: totalDuration > 0 ? totalDuration : undefined
        };

        // ページ離脱に強い送信
        await sendJson(
            "POST",
            CONFIG.api.mangaViewEnd,
            payload,
            {
                ignoreError: true,
                beacon: CONFIG.useSendBeacon || (options && options.beacon)
            }
        );

        logDebug("manga-view end", payload);

        resetViewerState();
    }

    function setCurrentPage(pageIndex) {
        if (!currentMangaView.mangaId) {
            return;
        }
        const idx = Number(pageIndex);
        if (!Number.isFinite(idx) || idx < 0) {
            return;
        }
        currentMangaView.lastPage = idx;
        if (idx > currentMangaView.maxPage) {
            currentMangaView.maxPage = idx;
        }
        saveViewerStateToStorage();
    }

    // =========================
    // 検索クエリ記録
    // =========================

    async function logSearchQuery(params) {
        if (!params || (!params.query && !params.tags)) {
            return;
        }

        await ensureSessionRegistered();
        const uid = userId || (await initUserId());
        const sid = sessionId || ensureSessionId();

        const rawQuery = String(params.query || "").slice(0, 200);
        const tags = Array.isArray(params.tags)
            ? params.tags
                  .map((t) => (t == null ? "" : String(t)))
                  .filter((t) => t.trim().length > 0)
                  .slice(0, 100)
            : [];

        const payload = {
            user_id: uid,
            session_id: sid,
            query: rawQuery,
            tags,
            used_at: nowIso()
        };

        logDebug("search query", payload);

        return sendJson(
            "POST",
            CONFIG.api.search,
            payload,
            { ignoreError: true }
        );
    }

    // =========================
    // グローバル公開API
    // =========================

    const ViewerAPI = {
        // viewer.html 側から呼ぶ: 漫画閲覧開始
        start: function (opts) {
            // opts: { mangaId, pageCount }
            initUserId().then(() => {
                ensureSessionRegistered().then(() => {
                    startMangaView(opts);
                });
            });
        },
        // viewer.html 側から呼ぶ: ページ変更時
        updatePage: function (opts) {
            // opts: { pageIndex }
            if (!opts) return;
            const idx = Number(opts.pageIndex);
            if (!Number.isFinite(idx) || idx < 0) return;
            setCurrentPage(idx);
            // ページ更新間隔に任せるが、明示的にも更新キック可能
            if (currentMangaView.id) {
                updateMangaView({ reason: "page_change" }).catch(() => {});
            }
        },
        // viewer.html 側から呼ぶ: 閲覧終了（任意）。beforeunload等でも自動呼び出しされる想定。
        end: function () {
            endMangaView({ beacon: true }).catch(() => {});
        }
    };

    const SearchAPI = {
        // 検索実行時に呼ぶ
        logQuery: function (opts) {
            logSearchQuery(opts).catch(() => {});
        }
    };

    const Tracking = {
        getUserId: function () {
            return userId;
        },
        onUserIdReady,
        viewer: ViewerAPI,
        search: SearchAPI,
        // セッションID参照用（必要なら）
        getSessionId: function () {
            return sessionId || getStoredSession()?.id || null;
        }
    };

    // =========================
    // 自動初期化
    // =========================

    // user_id / session を起動時に準備しておく
    (function bootstrap() {
        // 既存状態読み込み（連続セッション対策）
        loadViewerStateFromStorage();
        // 非同期でユーザーとセッションを初期化
        initUserId()
            .then(() => ensureSessionRegistered())
            .catch(() => {});
    })();

    // ページ離脱時にビューアを終了（ビューア利用時のみ効果）
    if (typeof window !== "undefined") {
        window.addEventListener("beforeunload", function () {
            if (currentMangaView && currentMangaView.id) {
                // 同期的ベストエフォート
                endMangaView({ beacon: true }).catch(() => {});
            }
        });

        document.addEventListener("visibilitychange", function () {
            if (document.visibilityState === "hidden") {
                if (currentMangaView && currentMangaView.id) {
                    endMangaView({ beacon: true }).catch(() => {});
                }
            }
        });
    }

    // グローバル公開
    if (typeof window !== "undefined") {
        window.Tracking = Tracking;
    }
})();