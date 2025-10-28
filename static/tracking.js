/**
 * ユーザー行動追跡スクリプト
 * ページビュー、滞在時間、スクロール、クリック、マウス移動を追跡
 * プライバシーに配慮した設計
 */

// グローバル変数
let sessionId = null;
let fingerprintHash = null;
let pageViewId = null;
let pageStartTime = null;
let lastScrollPosition = 0;
let lastScrollTime = Date.now();
let eventQueue = [];
let isTrackingEnabled = true;
let privacyConsent = null; // プライバシー同意状態
let config = {
    apiEndpoint: '/api/tracking',
    batchInterval: 10000, // 10秒ごとにバッチ送信
    maxQueueSize: 50, // 最大キューサイズ
    mouseTrackingInterval: 2000, // マウス位置追跡間隔（ミリ秒）
    scrollThrottleDelay: 100, // スクロールイベントの節流遅延（ミリ秒）
    enableMouseTracking: true,
    enableScrollTracking: false, // スクロールトラッキングを無効化
    enableClickTracking: true,
    anonymizeData: true, // データの匿名化
    dataRetentionDays: 90 // データ保持期間（日）
};


// データの匿名化
function anonymizeData(data) {
    if (!config.anonymizeData) return data;
    
    const anonymized = { ...data };
    
    // IPアドレスの匿名化（サーバー側で実装）
    if (anonymized.ip_address) {
        // IPv4の場合：最後のオクテットを0に
        // IPv6の場合：最後の64ビットを0に
        anonymized.ip_address = '[ANONYMIZED]';
    }
    
    // ユーザーエージェントの一部をマスク
    if (anonymized.user_agent) {
        const ua = anonymized.user_agent;
        anonymized.user_agent = ua.substring(0, Math.max(20, ua.length / 2)) + '[...]';
    }
    
    // テキストデータの匿名化
    if (anonymized.element_text && anonymized.element_text.length > 10) {
        anonymized.element_text = anonymized.element_text.substring(0, 5) + '[...]';
    }
    
    return anonymized;
}

// Fingerprint.jsの読み込みと初期化
async function initializeFingerprint() {
    try {
        // FingerprintJSライブラリを動的に読み込み
        if (typeof FingerprintJS === 'undefined') {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/@fingerprintjs/fingerprintjs@3/dist/fp.min.js';
            script.async = true;
            document.head.appendChild(script);
            
            // スクリプト読み込み完了を待つ
            await new Promise(resolve => {
                script.onload = resolve;
            });
        }
        
        // フィンガープリントを取得
        const fp = await FingerprintJS.load();
        const result = await fp.get();
        fingerprintHash = result.visitorId;
        
        // セッションIDの生成または取得
        sessionId = getSessionId();
        
        // セッション情報をサーバーに送信
        updateSession();
        
        return true;
    } catch (error) {
        console.error('フィンガープリント初期化エラー:', error);
        // フォールバック：ランダムなIDを生成
        fingerprintHash = 'fallback_' + Math.random().toString(36).substr(2, 9);
        sessionId = getSessionId();
        updateSession();
        return false;
    }
}

// セッションIDの取得または生成
function getSessionId() {
    let storedSessionId = localStorage.getItem('tracking_session_id');
    if (!storedSessionId) {
        storedSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('tracking_session_id', storedSessionId);
    }
    return storedSessionId;
}

// セッション情報の更新
async function updateSession() {
    if (!isTrackingEnabled || !sessionId || !fingerprintHash) return;
    
    try {
        const response = await fetch(`${config.apiEndpoint}/session`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(anonymizeData({
                session_id: sessionId,
                fingerprint_hash: fingerprintHash,
                user_agent: navigator.userAgent,
                ip_address: null // クライアント側では取得できないためサーバー側で設定
            }))
        });
        
        if (!response.ok) {
            console.error('セッション更新エラー:', response.status);
        }
    } catch (error) {
        console.error('セッション更新リクエストエラー:', error);
    }
}

// ページビューの記録
async function recordPageView() {
    if (!isTrackingEnabled || !sessionId) return;
    
    pageStartTime = Date.now();
    
    try {
        const response = await fetch(`${config.apiEndpoint}/page-view`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(anonymizeData({
                session_id: sessionId,
                page_url: window.location.href,
                page_title: document.title,
                referrer: document.referrer || null,
                view_start: new Date(pageStartTime).toISOString(),
                view_end: null,
                time_on_page: null,
                scroll_depth_max: null
            }))
        });
        
        if (response.ok) {
            const data = await response.json();
            pageViewId = data.page_view_id;
        } else {
            console.error('ページビュー記録エラー:', response.status);
        }
    } catch (error) {
        console.error('ページビュー記録リクエストエラー:', error);
    }
}

// ページ滞在時間の更新
async function updatePageView() {
    if (!isTrackingEnabled || !sessionId || !pageViewId) return;
    
    const timeOnPage = Math.floor((Date.now() - pageStartTime) / 1000);
    const scrollDepth = calculateScrollDepth();
    
    try {
        const response = await fetch(`${config.apiEndpoint}/page-view`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(anonymizeData({
                session_id: sessionId,
                page_url: window.location.href,
                page_title: document.title,
                referrer: document.referrer || null,
                view_start: new Date(pageStartTime).toISOString(),
                view_end: new Date().toISOString(),
                time_on_page: timeOnPage,
                scroll_depth_max: scrollDepth
            }))
        });
        
        if (!response.ok) {
            console.error('ページビュー更新エラー:', response.status);
        }
    } catch (error) {
        console.error('ページビュー更新リクエストエラー:', error);
    }
}

// スクロール深度の計算
function calculateScrollDepth() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const documentHeight = document.documentElement.scrollHeight - window.innerHeight;
    return documentHeight > 0 ? Math.round((scrollTop / documentHeight) * 100) : 0;
}

// イベントのキューに追加
function queueEvent(eventType, eventData) {
    if (!isTrackingEnabled) return;
    
    const event = {
        session_id: sessionId,
        page_view_id: pageViewId,
        event_type: eventType,
        timestamp: new Date().toISOString(),
        ...eventData
    };
    
    eventQueue.push(event);
    
    // キューが最大サイズに達したら即時送信
    if (eventQueue.length >= config.maxQueueSize) {
        sendEventBatch();
    }
}

// イベントのバッチ送信
async function sendEventBatch() {
    if (!isTrackingEnabled || eventQueue.length === 0) return;
    
    const eventsToSend = [...eventQueue];
    eventQueue = []; // キューをクリア
    
    try {
        const response = await fetch(`${config.apiEndpoint}/events`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                events: eventsToSend.map(event => anonymizeData(event))
            })
        });
        
        if (!response.ok) {
            console.error('イベントバッチ送信エラー:', response.status);
            // 失敗した場合はキューに戻す
            eventQueue = [...eventsToSend, ...eventQueue];
        }
    } catch (error) {
        console.error('イベントバッチ送信リクエストエラー:', error);
        // 失敗した場合はキューに戻す
        eventQueue = [...eventsToSend, ...eventQueue];
    }
}

// スクロールイベントの処理（節流付き）
let scrollThrottleTimer = null;
function handleScroll() {
    if (!config.enableScrollTracking || !isTrackingEnabled) return;
    
    if (scrollThrottleTimer) return;
    
    scrollThrottleTimer = setTimeout(() => {
        const currentScrollPosition = window.pageYOffset || document.documentElement.scrollTop;
        const currentTime = Date.now();
        const scrollDirection = currentScrollPosition > lastScrollPosition ? 'down' : 'up';
        const timeDiff = currentTime - lastScrollTime;
        const scrollDistance = Math.abs(currentScrollPosition - lastScrollPosition);
        const scrollSpeed = timeDiff > 0 ? Math.round(scrollDistance / timeDiff * 1000) : 0; // ピクセル/秒
        
        queueEvent('scroll', {
            x_position: null,
            y_position: currentScrollPosition,
            scroll_direction: scrollDirection,
            scroll_speed: scrollSpeed
        });
        
        lastScrollPosition = currentScrollPosition;
        lastScrollTime = currentTime;
        scrollThrottleTimer = null;
    }, config.scrollThrottleDelay);
}

// クリックイベントの処理
function handleClick(event) {
    if (!config.enableClickTracking || !isTrackingEnabled) return;
    
    const element = event.target;
    const elementSelector = getElementSelector(element);
    const elementText = getElementText(element);
    
    queueEvent('click', {
        element_selector: elementSelector,
        element_text: elementText,
        x_position: event.clientX,
        y_position: event.clientY
    });
}

// マウス移動イベントの処理
function handleMouseMove(event) {
    if (!config.enableMouseTracking || !isTrackingEnabled) return;
    
    queueEvent('mouse_move', {
        x_position: event.clientX,
        y_position: event.clientY
    });
}

// 要素のCSSセレクタを取得
function getElementSelector(element) {
    if (!element) return null;
    
    // IDがあれば優先
    if (element.id) {
        return `#${element.id}`;
    }
    
    // クラスがあれば使用
    if (element.className && typeof element.className === 'string') {
        const classes = element.className.split(' ').filter(c => c.trim());
        if (classes.length > 0) {
            return `${element.tagName.toLowerCase()}.${classes.join('.')}`;
        }
    }
    
    // タグ名とdata属性を組み合わせ
    let selector = element.tagName.toLowerCase();
    
    // data-track属性があれば使用
    if (element.dataset.track) {
        selector += `[data-track="${element.dataset.track}"]`;
    }
    
    return selector;
}

// 要素のテキストを取得（最大50文字）
function getElementText(element) {
    if (!element) return null;
    
    let text = element.textContent || element.innerText || '';
    text = text.trim();
    
    // 長すぎるテキストは切り詰める
    if (text.length > 50) {
        text = text.substring(0, 47) + '...';
    }
    
    return text || null;
}

// トラッキングの有効/無効切り替え
function setTrackingEnabled(enabled) {
    isTrackingEnabled = enabled;
    localStorage.setItem('tracking_enabled', enabled.toString());
}

// トラッキング設定の読み込み
function loadConfig() {
    const savedConfig = localStorage.getItem('tracking_config');
    if (savedConfig) {
        try {
            const parsed = JSON.parse(savedConfig);
            config = { ...config, ...parsed };
        } catch (error) {
            console.error('設定読み込みエラー:', error);
        }
    }
    
    const savedEnabled = localStorage.getItem('tracking_enabled');
    if (savedEnabled !== null) {
        isTrackingEnabled = savedEnabled === 'true';
    }
}

// トラッキング設定の保存
function saveConfig() {
    localStorage.setItem('tracking_config', JSON.stringify(config));
}

// 初期化処理
async function initializeTracking() {
    // 設定の読み込み
    loadConfig();
    
    // プライバシー同意の確認
    const hasConsent = checkPrivacyConsent();
    if (!hasConsent) {
        console.log('プライバシー同意待機中');
        return;
    }
    
    if (!isTrackingEnabled) {
        console.log('トラッキングは無効になっています');
        return;
    }
    
    // フィンガープリントの初期化
    await initializeFingerprint();
    
    // ページビューの記録
    recordPageView();
    
    // イベントリスナーの設定
    if (config.enableScrollTracking) {
        window.addEventListener('scroll', handleScroll, { passive: true });
    }
    
    if (config.enableClickTracking) {
        document.addEventListener('click', handleClick, true);
    }
    
    if (config.enableMouseTracking) {
        // マウス移動は間引いて追跡
        setInterval(() => {
            if (document.hasFocus()) {
                document.addEventListener('mousemove', handleMouseMove, { once: true });
            }
        }, config.mouseTrackingInterval);
    }
    
    // 定期的なバッチ送信
    setInterval(sendEventBatch, config.batchInterval);
    
    // ページ離脱時の処理
    window.addEventListener('beforeunload', () => {
        updatePageView();
        sendEventBatch();
    });
    
    // ページ非表示時の処理
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            updatePageView();
            sendEventBatch();
        } else if (document.visibilityState === 'visible') {
            // ページが再表示されたら新しいページビューとして記録
            recordPageView();
        }
    });
    
    console.log('トラッキングが初期化されました');
}

// グローバル関数として公開（外部からの制御用）
// 単一のイベントを記録するAPIエンドポイント
async function recordSingleEvent(request) {
    try {
        const response = await fetch(`${config.apiEndpoint}/single-event`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(anonymizeData({
                session_id: request.session_id,
                page_view_id: request.page_view_id,
                event_type: request.event_type,
                element_selector: request.element_selector,
                element_text: request.element_text,
                x_position: request.x_position,
                y_position: request.y_position,
                scroll_direction: request.scroll_direction,
                scroll_speed: request.scroll_speed,
                timestamp: request.timestamp || new Date().toISOString()
            }))
        });
        
        if (!response.ok) {
            console.error('単一イベント記録エラー:', response.status);
        }
        
        return await response.json();
    } catch (error) {
        console.error('単一イベント記録リクエストエラー:', error);
        return null;
    }
}

window.Tracking = {
    setTrackingEnabled,
    saveConfig,
    loadConfig,
    sendEventBatch,
    updatePageView,
    getSessionId: () => sessionId,
    getPageViewId: () => pageViewId,
    isTrackingEnabled: () => isTrackingEnabled,
    recordSingleEvent
};

// DOMの読み込み完了後に初期化
if (document.readyState === 'loading') {
    initializeTracking();
}