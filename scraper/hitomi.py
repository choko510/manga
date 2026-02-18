from __future__ import annotations
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# 定数
NOZOMI_DEFAULT_PER_PAGE = 25

# Hitomi.la の人気ランキング URL
AllURL = "https://ltn.gold-usergeneratedcontent.net/index-japanese.nozomi"
YearURL = "https://ltn.gold-usergeneratedcontent.net/popular/year-all.nozomi"
MonthURL = "https://ltn.gold-usergeneratedcontent.net/popular/month-all.nozomi"
WeekURL = "https://ltn.gold-usergeneratedcontent.net/popular/week-all.nozomi"
DayURL = "https://ltn.gold-usergeneratedcontent.net/popular/today-all.nozomi"


def _endian_fmt(big_endian: bool, unsigned: bool) -> str:
    """
    エンディアンと符号の情報から struct フォーマット文字列を生成する
    
    Args:
        big_endian: ビッグエンディアンの場合 True
        unsigned: 符号なしの場合 True
    
    Returns:
        struct フォーマット文字列
    """
    # > = big-endian, < = little-endian
    # I = unsigned 32-bit, i = signed 32-bit
    endian = ">" if big_endian else "<"
    ty = "I" if unsigned else "i"
    return endian + ty


def _is_url(source: str) -> bool:
    """
    文字列が URL かどうかを判定する
    
    Args:
        source: 判定する文字列
    
    Returns:
        URL の場合 True
    """
    try:
        parsed = urlparse(source)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def _read_remote_range(url: str, start: int, end: int) -> bytes:
    """
    リモートファイルの範囲指定読み込み
    
    Args:
        url: ファイルの URL
        start: 開始バイト位置
        end: 終了バイト位置
    
    Returns:
        読み込んだデータ
    
    Raises:
        HTTPError: HTTP エラーが発生した場合
        URLError: URL エラーが発生した場合
    """
    headers = {"Range": f"bytes={start}-{end}"}
    req = Request(url, headers=headers)
    with urlopen(req) as response:
        if response.status not in (200, 206):
            raise HTTPError(url, response.status, response.reason, response.headers, None)
        return response.read()


def _remote_total_bytes_via_head(url: str) -> Optional[int]:
    """
    HEAD リクエストでリモートファイルのサイズを取得
    
    Args:
        url: ファイルの URL
    
    Returns:
        ファイルサイズ（バイト）。取得できない場合 None
    """
    try:
        req = Request(url, method="HEAD")
        with urlopen(req) as response:
            if response.status == 200:
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    return int(content_length)
    except Exception:
        pass
    return None


def _remote_total_bytes_via_range_probe(url: str) -> Optional[int]:
    """
    Range リクエストプローブでリモートファイルのサイズを取得
    
    Args:
        url: ファイルの URL
    
    Returns:
        ファイルサイズ（バイト）。取得できない場合 None
    """
    try:
        # 最初の1バイトをリクエストして Content-Range を確認
        headers = {"Range": "bytes=0-0"}
        req = Request(url, headers=headers)
        with urlopen(req) as response:
            if response.status == 206:
                content_range = response.headers.get("Content-Range")
                if content_range:
                    # "bytes 0-0/12345" 形式から総サイズを抽出
                    parts = content_range.split("/")
                    if len(parts) == 2 and parts[1].isdigit():
                        return int(parts[1])
    except Exception:
        pass
    return None


@dataclass
class NozomiFileInfo:
    """
    Nozomi ファイル情報
    """
    source: str
    total_items: Optional[int] = None  # 4バイト整数の数（ID数）
    total_bytes: Optional[int] = None


class NozomiReader:
    """
    Hitomi.la の "nozomi" バイナリ索引（4バイト整数配列）を読むためのユーティリティ。

    - ローカルファイルと HTTP/HTTPS URL の両方に対応
    - ページ単位/範囲単位の読み出しをサポート (Range リクエスト)
    - 既定は **ビッグエンディアン + 符号なし32bit**

    例:
        reader = NozomiReader("path/to/file.nozomi")
        ids = reader.read_page(page=1, per_page=25)
        n = len(reader)  # 総件数 (取得できる場合)

        reader_url = NozomiReader("https://example.com/tag-lang.nozomi")
        ids2 = reader_url.read_range(start=0, count=25)
    """

    def __init__(
        self,
        source: str,
        *,
        big_endian: bool = True,
        unsigned: bool = True,
    ) -> None:
        """
        NozomiReader を初期化する

        Args:
            source: ファイルパスまたは URL
            big_endian: ビッグエンディアンの場合 True（既定）
            unsigned: 符号なしの場合 True（既定）
        """
        self.source = source
        self.big_endian = big_endian
        self.unsigned = unsigned
        self._fmt = _endian_fmt(big_endian, unsigned)

        self._local_fp = None  # type: object | None
        self._local_mm = None  # type: mmap.mmap | None

        self._info = self._probe_info()

    def __enter__(self):
        """コンテキストマネージャの開始"""
        if not _is_url(self.source):
            fp = open(self.source, "rb")
            self._local_fp = fp
            try:
                self._local_mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                fp.close()
                self._local_fp = None
                raise
        return self

    def __exit__(self, exc_type, exc, tb):
        """コンテキストマネージャの終了"""
        if self._local_mm is not None:
            self._local_mm.close()
            self._local_mm = None
        if self._local_fp is not None:
            self._local_fp.close()
            self._local_fp = None

    def info(self) -> NozomiFileInfo:
        """
        ファイル情報を返す

        Returns:
            NozomiFileInfo オブジェクト
        """
        return self._info

    def __len__(self) -> int:
        """
        総件数を返す

        Returns:
            総件数

        Raises:
            ValueError: 総件数が不明の場合
        """
        if self._info.total_items is None:
            raise ValueError("総件数が不明です（リモートで Content-Length/Range が得られませんでした）。")
        return self._info.total_items

    def read_all(self) -> List[int]:
        """
        ファイル全体（全ID）を読み出し（巨大ファイルではメモリ注意）

        Returns:
            ID のリスト

        Raises:
            ValueError: 総サイズが不明で読み出せない場合
        """
        if _is_url(self.source):
            if self._info.total_bytes is None:
                raise ValueError("総サイズが不明のため read_all は利用できません。")
            buf = _read_remote_range(self.source, 0, self._info.total_bytes - 1)
            return [x[0] for x in struct.iter_unpack(self._fmt, buf)]
        else:
            if self._local_mm is None:
                with self:
                    return self.read_all()
            mm = self._local_mm
            return [x[0] for x in struct.iter_unpack(self._fmt, mm[:])]

    def read_range(self, start: int, count: int) -> List[int]:
        """
        指定範囲の ID を読み出す

        Args:
            start: 0始まりのIDインデックス
            count: 読み出す件数

        Returns:
            ID のリスト

        Raises:
            ValueError: 引数が不正な場合
        """
        if start < 0 or count < 0:
            raise ValueError("start と count は非負で指定してください。")
        byte_start = start * 4
        byte_end_inclusive = byte_start + count * 4 - 1
        if count == 0:
            return []

        if _is_url(self.source):
            buf = _read_remote_range(self.source, byte_start, byte_end_inclusive)
            if len(buf) % 4 != 0:
                raise ValueError("取得したバッファ長が 4 の倍数ではありません。")
            return [x[0] for x in struct.iter_unpack(self._fmt, buf)]
        else:
            if self._local_mm is None:
                with self:
                    return self.read_range(start, count)
            mm = self._local_mm
            end = byte_end_inclusive + 1
            if end > len(mm):
                end = len(mm)
            buf = mm[byte_start:end]
            if len(buf) % 4 != 0:
                if len(buf) == 0:
                    return []
                buf = buf[: len(buf) - (len(buf) % 4)]
            return [x[0] for x in struct.iter_unpack(self._fmt, buf)]

    def read_page(self, page: int = 1, per_page: int = NOZOMI_DEFAULT_PER_PAGE) -> List[int]:
        """
        ページ単位で ID を読み出す

        Args:
            page: ページ番号（1始まり）
            per_page: 1ページの件数

        Returns:
            ID のリスト

        Raises:
            ValueError: ページ番号が不正な場合
        """
        if page <= 0:
            raise ValueError("page は 1 以上で指定してください。")
        start = (page - 1) * per_page
        return self.read_range(start, per_page)

    def _probe_info(self) -> NozomiFileInfo:
        """
        ファイル情報を取得する

        Returns:
            NozomiFileInfo オブジェクト
        """
        if _is_url(self.source):
            total_bytes = _remote_total_bytes_via_head(self.source)
            if total_bytes is None:
                total_bytes = _remote_total_bytes_via_range_probe(self.source)
            total_items = total_bytes // 4 if total_bytes is not None else None
            return NozomiFileInfo(self.source, total_items, total_bytes)
        else:
            try:
                total_bytes = os.path.getsize(self.source)
            except OSError:
                total_bytes = None
            total_items = total_bytes // 4 if total_bytes is not None else None
            return NozomiFileInfo(self.source, total_items, total_bytes)

def load_existing_galleries(galleries_file: str = "cache/galleries.txt") -> Set[int]:
    """
    既存のギャラリーIDをgalleries.txtから読み込む
    ファイルが存在しない場合はread_galleries.pyで生成する
    
    Args:
        galleries_file: ギャラリーIDが保存されているファイルパス
    
    Returns:
        既存のギャラリーIDのセット
    """
    existing_ids = set()

    # ファイルが存在しないか、10行以下の場合もread_galleries.pyで再生成
    if not os.path.exists(galleries_file) or sum(1 for line in open(galleries_file, 'r', encoding='utf-8')) <= 10:
        print(f"{galleries_file}が見つからないか10行以下です。read_galleries.pyで再生成します。")
        try:
            # read_galleries.pyの関数をインポートして実行
            import sys
            import importlib.util
            
            # read_galleries.pyのパスを取得
            read_galleries_path = os.path.join(os.path.dirname(__file__), "read_galleries.py")
            
            # モジュールを動的にインポート
            spec = importlib.util.spec_from_file_location("read_galleries", read_galleries_path)
            read_galleries_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(read_galleries_module)
            
            # read_all_galleries関数を実行
            read_galleries_module.read_all_galleries()
            print(f"{galleries_file}を再生成しました。")
        except Exception as e:
            print(f"{galleries_file}の再生成中にエラーが発生しました: {e}")
            return existing_ids
    
    # ファイルから既存のIDを読み込む
    try:
        with open(galleries_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    existing_ids.add(int(line))
        print(f"galleries.txtから {len(existing_ids)}件の既存IDを読み込みました")
    except FileNotFoundError:
        print(f"{galleries_file}が見つかりません。新規作成します。")
    except Exception as e:
        print(f"{galleries_file}の読み込み中にエラーが発生しました: {e}")
    
    return existing_ids

async def download_all_popular_files() -> None:
    """
    すべての人気ランキングファイル（年、月、週、日）をダウンロードして
    それぞれの ID リストをテキストファイルに出力する
    galleries.txtに既存のIDがある場合は重複を除外して出力する
    """
    # 既存のギャラリーIDを読み込む
    existing_ids = load_existing_galleries()
    
    # 人気ランキングファイルの情報
    popular_files = [
        {"name": "all", "url": AllURL, "output": "cache/all_ids.txt"},
        {"name": "year", "url": YearURL, "output": "cache/year_ids.txt"},
        {"name": "month", "url": MonthURL, "output": "cache/month_ids.txt"},
        {"name": "week", "url": WeekURL, "output": "cache/week_ids.txt"},
        {"name": "day", "url": DayURL, "output": "cache/day_ids.txt"},
    ]
    
    print("人気ランキングファイルのダウンロードを開始します...")
    
    for file_info in popular_files:
        try:
            print(f"{file_info['name']} ランキングをダウンロード中: {file_info['url']}")
            
            # NozomiReader を使用して ID を取得
            reader = NozomiReader(file_info["url"])
            with reader:
                ids = reader.read_all()
            
            if file_info["name"] == "all":
                # AllURLの場合はフィルタリングせずにすべて出力
                output_ids = ids
                print(f"✓ {file_info['name']} ランキング: {len(output_ids)}件のIDを {file_info['output']} に出力しました")
            else:
                # 既存IDとの重複のみを抽出
                output_ids = [id for id in ids if id in existing_ids]
                filtered_count = len(output_ids)
                print(f"✓ {file_info['name']} ランキング: {len(output_ids)}件の重複IDを {file_info['output']} に出力しました")
                if len(ids) - filtered_count > 0:
                    print(f"  (新規ID除外: {len(ids) - filtered_count}件)")
            
            # テキストファイルに出力
            with open(file_info["output"], "w", encoding="utf-8") as f:
                for id in output_ids:
                    f.write(f"{id}\n")
            
        except Exception as e:
            print(f"✗ {file_info['name']} ランキングのダウンロードに失敗しました: {e}")
    
    print("すべてのダウンロード処理が完了しました")

def _cli():
    """コマンドラインインターフェース"""
    import asyncio
    
    # 引数がない場合は人気ランキングファイルをすべてダウンロード
    if len(sys.argv) == 1:
        asyncio.run(download_all_popular_files())
        return

if __name__ == "__main__":
    _cli()