from sqlalchemy import create_engine, text
import os

# sa.db からすべてのギャラリーを読み込んで縦一列にtxtファイルに出力する
def read_all_galleries():
    # データベースに接続
    db_path = os.path.join("db", "sa.db")
    engine = create_engine(f"sqlite:///{db_path}")
    
    with engine.connect() as conn:
        # galleries テーブルからすべての ID を取得（ソートなし）
        result = conn.execute(text("SELECT gallery_id FROM galleries"))
        
        # 結果をtxtファイルに出力
        with open("galleries.txt", "w", encoding="utf-8") as f:
            for row in result:
                f.write(str(row.gallery_id) + "\n")

if __name__ == "__main__":
    read_all_galleries()