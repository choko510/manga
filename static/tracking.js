// 新トラッキングスクリプト
// シンプルなユーザー行動ログ収集

(function () {
    "use strict";

    // =========================
    // 設定
    // =========================
    const CONFIG = {
        api: {
            view: "/api/logs/view",
            impression: "/api/logs/impression",
            click: "/api/logs/click",
            search: "/api/logs/search"
        },
        storageKeys: {
            userId: "tracking_user_id"
        },
        // ビューア更新送信間隔（秒）
        viewerUpdateIntervalSec: 30,
        // デバッグログ
        debug: false
    };

    // =========================
    // ユーティリティ
    // =========================

    function logDebug(...args) {
        if (CONFIG.debug) {
            console.debug("[Tracking]", ...args);
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
    // user_id 管理
    // =========================

    let userId = null;

    function getStoredUserId() {
        try {
            return localStorage.getItem(CONFIG.storageKeys.userId);
        } catch {
            return null;
        }
    }

    function storeUserId(id) {
        try {
            localStorage.setItem(CONFIG.storageKeys.userId, id);
        } catch {
            // ignore
        }
    }

    function ensureUserId() {
        if (userId) return userId;
        userId = getStoredUserId();
        if (!userId) {
            userId = randomId("u");
            storeUserId(userId);
        }
        return userId;
    }

    // =========================
    // 通信
    // =========================

    async function sendJson(path, body) {
        try {
            const res = await fetch(path, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });
            if (!res.ok) {
                logDebug("Request failed", path, res.status);
            }
            return res.ok;
        } catch (e) {
            logDebug("Request error", path, e);
            return false;
        }
    }

    function sendBeaconJson(path, body) {
        try {
            if (typeof navigator !== "undefined" && navigator.sendBeacon) {
                const blob = new Blob([JSON.stringify(body)], { type: "application/json" });
                return navigator.sendBeacon(path, blob);
            }
        } catch {
            // fallback
        }
        // フォールバック: 同期的にfetchを試みない（ページ離脱時は諦める）
        return false;
    }

    // =========================
    // ビューアトラッキング
    // =========================

    let viewerState = {
        mangaId: null,
        pageCount: null,
        startTime: null,
        maxPage: 0,
        tags: []
    };

    let viewerUpdateTimer = null;

    function resetViewerState() {
        viewerState = {
            mangaId: null,
            pageCount: null,
            startTime: null,
            maxPage: 0,
            tags: []
        };
        if (viewerUpdateTimer) {
            clearInterval(viewerUpdateTimer);
            viewerUpdateTimer = null;
        }
    }

    function startViewer(opts) {
        // opts: { mangaId, pageCount, tags }
        if (!opts || !opts.mangaId) return;

        resetViewerState();
        viewerState.mangaId = Number(opts.mangaId);
        viewerState.pageCount = Number(opts.pageCount) || 0;
        viewerState.startTime = Date.now();
        viewerState.maxPage = 0;
        viewerState.tags = Array.isArray(opts.tags) ? opts.tags : [];

        // クリック記録
        ensureUserId();
        sendJson(CONFIG.api.click, {
            user_id: userId,
            manga_id: viewerState.mangaId,
            tags: viewerState.tags
        });

        // 定期更新開始
        viewerUpdateTimer = setInterval(() => {
            sendViewLog();
        }, CONFIG.viewerUpdateIntervalSec * 1000);

        logDebug("Viewer started", viewerState);
    }

    function updatePage(pageIndex) {
        if (!viewerState.mangaId) return;
        const idx = Number(pageIndex);
        if (!Number.isFinite(idx) || idx < 0) return;
        if (idx > viewerState.maxPage) {
            viewerState.maxPage = idx;
        }
    }

    function sendViewLog() {
        if (!viewerState.mangaId || !viewerState.startTime) return;

        const duration = Math.floor((Date.now() - viewerState.startTime) / 1000);
        ensureUserId();

        const payload = {
            user_id: userId,
            manga_id: viewerState.mangaId,
            duration: duration,
            max_page: viewerState.maxPage + 1, // 0-indexed -> 1-indexed
            page_count: viewerState.pageCount || null
        };

        sendJson(CONFIG.api.view, payload);
        logDebug("View log sent", payload);
    }

    function endViewer() {
        if (!viewerState.mangaId) {
            resetViewerState();
            return;
        }

        // 最終ログ送信（sendBeacon）
        const duration = Math.floor((Date.now() - viewerState.startTime) / 1000);
        ensureUserId();

        const payload = {
            user_id: userId,
            manga_id: viewerState.mangaId,
            duration: duration,
            max_page: viewerState.maxPage + 1,
            page_count: viewerState.pageCount || null
        };

        sendBeaconJson(CONFIG.api.view, payload);
        logDebug("Viewer ended", payload);

        resetViewerState();
    }

    // =========================
    // インプレッション記録
    // =========================

    function logImpression(opts) {
        // opts: { mangaIds, tags }
        if (!opts) return;

        ensureUserId();
        const mangaIds = Array.isArray(opts.mangaIds) ? opts.mangaIds.map(Number).filter(n => !isNaN(n)) : [];
        const tags = Array.isArray(opts.tags) ? opts.tags.filter(t => typeof t === "string" && t.trim()) : [];

        if (mangaIds.length === 0 && tags.length === 0) return;

        sendJson(CONFIG.api.impression, {
            user_id: userId,
            manga_ids: mangaIds,
            tags: tags
        });

        logDebug("Impression logged", { mangaIds, tags });
    }

    // =========================
    // 検索記録
    // =========================

    function logSearch(opts) {
        // opts: { tags }
        if (!opts || !Array.isArray(opts.tags) || opts.tags.length === 0) return;

        ensureUserId();
        const tags = opts.tags.filter(t => typeof t === "string" && t.trim());
        if (tags.length === 0) return;

        sendJson(CONFIG.api.search, {
            user_id: userId,
            tags: tags
        });

        logDebug("Search logged", { tags });
    }

    // =========================
    // 公開API
    // =========================

    const Tracking = {
        getUserId: function () {
            return ensureUserId();
        },
        viewer: {
            start: startViewer,
            updatePage: function (opts) {
                if (opts && typeof opts.pageIndex !== "undefined") {
                    updatePage(opts.pageIndex);
                }
            },
            end: endViewer
        },
        logImpression: logImpression,
        search: {
            logQuery: logSearch
        }
    };

    // =========================
    // 自動初期化
    // =========================

    ensureUserId();

    // ページ離脱時にビューアを終了
    if (typeof window !== "undefined") {
        window.addEventListener("beforeunload", function () {
            if (viewerState.mangaId) {
                endViewer();
            }
        });

        window.addEventListener("visibilitychange", function () {
            if (document.visibilityState === "hidden" && viewerState.mangaId) {
                endViewer();
            }
        });
    }

    // グローバル公開
    if (typeof window !== "undefined") {
        window.Tracking = Tracking;
    }
})();