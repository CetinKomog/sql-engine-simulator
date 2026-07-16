"""
data_analyzer.py
------------------
Modul 3: Sorgu loglama, Pandas ile istatistiksel profilleme ve CSV disa aktarim.
 
Bu modul, database_executor.py'den gelen ham sorgu sonuclarini (list of dicts)
girdi olarak alir ve projenin analiz + I/O cikti katmanini olusturur.
 
Sorumluluklar (Single Responsibility Principle):
    1. Her sorgu calistirma islemini (basarili/basarisiz) sure bilgisiyle
       birlikte kalici bir log dosyasina kaydetmek.
    2. Sayisal sutunlar uzerinde istatistiksel profil (mean, std, min, max,
       null_count) cikarmak.
    3. Sorgu sonucunu guvenli bir sekilde CSV dosyasina yazmak.
 
Bu sinif SQL calistirmaz (bu is DatabaseExecutor'in gorevidir) ve SQL
dogrulamasi yapmaz (bu is SQLEngineParser'in gorevidir); sadece sonuc
verisi uzerinde calisir.
"""
 
from __future__ import annotations
 
import csv
import logging
from pathlib import Path
from typing import Any
 
import pandas as pd
 
 
class DataAnalyzer:
    """
    Sorgu loglama, istatistiksel profilleme ve CSV export islemlerinden
    sorumlu sinif.
    """
 
    LOG_FILENAME = "sql_engine_history.log"
 
    def __init__(self, log_filename: str | None = None) -> None:
        """
        Args:
            log_filename: Log dosyasinin adi/yolu. None birakilirsa
                varsayilan olarak 'sql_engine_history.log' kullanilir.
        """
        self.log_filename = log_filename or self.LOG_FILENAME
        self.logger = self._setup_logger()
 
    def _setup_logger(self) -> logging.Logger:
        """
        Standart `logging` kutuphanesini kullanarak dosyaya yazan,
        zaman damgali bir logger kurar.
 
        Not: Ayni logger'a birden fazla handler eklenmesini onlemek icin
        (ornegin DataAnalyzer birden fazla kez olusturulursa) logger adi
        modul yoluna gore sabitlenir ve handler tekrari kontrol edilir.
        """
        logger = logging.getLogger("SQLEngineHistoryLogger")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Kok logger'a (console) sizmasin.
 
        # Ayni handler'in tekrar tekrar eklenmesini engelle (coklu instance durumu).
        already_configured = any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None)
            == str(Path(self.log_filename).resolve())
            for handler in logger.handlers
        )
 
        if not already_configured:
            file_handler = logging.FileHandler(self.log_filename, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
 
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
 
        return logger
 
    def log_query(
        self,
        query: str,
        duration_ms: float,
        status: str,
        row_count: int,
    ) -> None:
        """
        Calistirilan bir SQL sorgusunun sonucunu log dosyasina kaydeder.
 
        Args:
            query: Calistirilan ham SQL metni.
            duration_ms: Sorgunun calisma suresi (milisaniye).
            status: "SUCCESS" veya "ERROR" gibi durum bilgisi.
            row_count: Sorgudan donen/etkilenen satir sayisi.
 
        Log formati:
            YYYY-MM-DD HH:MM:SS | LEVEL | Query: [query_text] | Duration: X.XXms | Status: [SUCCESS/ERROR] | Rows: Y

        """
        # Cok satirli sorgular log dosyasini bozmasin diye tek satira indirgenir.
        normalized_query = " ".join(query.strip().split())
 
        log_level = logging.INFO if status.upper() == "SUCCESS" else logging.ERROR
 
        message = (
            f"Query: [{normalized_query}] | "
            f"Duration: {duration_ms:.2f}ms | "
            f"Status: [{status.upper()}] | "
            f"Rows: {row_count}"
        )
 
        self.logger.log(log_level, message)
 
    def profile_data(self, data: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Verilen sorgu sonucundaki sayisal sutunlarin istatistiksel profilini
        cikarir.
 
        Args:
            data: database_executor.py'den donen list of dicts formatindaki
                sorgu sonucu.
 
        Returns:
            Her sayisal sutun icin {mean, std, min, max, null_count}
            degerlerini iceren bir dictionary. Veri bossa veya hicbir
            sayisal sutun yoksa None doner.
        """
        if not data:
            return None
 
        df = pd.DataFrame(data)
 
        # Sadece int/float turundeki sutunlari secer (bool haric tutulur,
        # cunku pandas bool'u da sayisal sayar ve bu profil acisindan yanlis
        # yorumlara yol acabilir).
        numeric_df = df.select_dtypes(include=["number"]).select_dtypes(exclude=["bool"])
 
        if numeric_df.empty or numeric_df.shape[1] == 0:
            return None
 
        profile: dict[str, Any] = {}
 
        for column_name in numeric_df.columns:
            column = numeric_df[column_name]
            non_null_count = column.count()
 
            # Tek satirlik (veya tamamen NaN olan) veri setlerinde std() NaN
            # dondurebilir; bu durumda 0.0 kullanarak NaN sizintisini onluyoruz.
            std_value = column.std()
            if pd.isna(std_value):
                std_value = 0.0
 
            mean_value = column.mean()
            min_value = column.min()
            max_value = column.max()
 
            profile[column_name] = {
                "mean": round(float(mean_value), 2) if pd.notna(mean_value) else None,
                "std": round(float(std_value), 2),
                "min": None if pd.isna(min_value) else (
                    float(min_value) if not float(min_value).is_integer() else int(min_value)
                ),
                "max": None if pd.isna(max_value) else (
                    float(max_value) if not float(max_value).is_integer() else int(max_value)
                ),
                "null_count": int(column.isna().sum()),
            }
 
        return profile
 
    def export_to_csv(
        self,
        data: list[dict[str, Any]],
        filename: str = "query_result.csv",
    ) -> str:
        """
        Sorgu sonucunu (list of dicts) bir CSV dosyasina guvenli sekilde yazar.
 
        Args:
            data: Yazilacak sorgu sonucu.
            filename: Cikti dosyasinin adi/yolu. Varsayilan: 'query_result.csv'.
 
        Returns:
            Dosyanin basariyla kaydedildigini belirten bir mesaj.
 
        Raises:
            ValueError: `data` bos ise (yazilacak bir sey olmadigi icin).
        """
        if not data:
            raise ValueError("Bos veri seti CSV olarak disa aktarilamaz.")
 
        output_path = Path(filename)
 
        # Tum satirlarda ayni sutunlarin bulunmasi garanti olmayabilecegi icin
        # (ornegin farkli SELECT'ler sonucu), header'i tum satirlardaki
        # anahtarlarin birlesimi (union) olacak sekilde, sirayi koruyarak olusturuyoruz.
        fieldnames: list[str] = []
        for row in data:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
 
        # newline="" : csv modulunun Windows'ta olusturdugu bos satir
        # sorununu (blank rows) engellemek icin sarttir.
        with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
 
        return f"Basarili: {len(data)} satir '{output_path.resolve()}' dosyasina kaydedildi."
 
 
if __name__ == "__main__":
    # --- Self-test blogu: mock bir SQL sonuc kumesi ile 3 ozelligi de dogrular ---
 
    analyzer = DataAnalyzer()
 
    mock_query_result: list[dict[str, Any]] = [
        {"id": 1, "ad_soyad": "Ahmet Yilmaz", "departman": "Yazilim", "maas": 45000.0, "yas": 29},
        {"id": 2, "ad_soyad": "Ayse Demir", "departman": "Insan Kaynaklari", "maas": 38000.0, "yas": 34},
        {"id": 3, "ad_soyad": "Mehmet Kaya", "departman": "Muhasebe", "maas": None, "yas": 41},
        {"id": 4, "ad_soyad": "Zeynep Sahin", "departman": "Yazilim", "maas": 52500.5, "yas": 26},
    ]
 
    print("=== 1) Loglama Testi ===")
    analyzer.log_query(
        query="SELECT * FROM Personel WHERE departman = 'Yazilim'",
        duration_ms=12.3456,
        status="SUCCESS",
        row_count=len(mock_query_result),
    )
    analyzer.log_query(
        query="SELECT * FROM OlmayanTablo",
        duration_ms=2.15,
        status="ERROR",
        row_count=0,
    )
    print(f"Log kayitlari '{analyzer.log_filename}' dosyasina yazildi.")
    with open(analyzer.log_filename, "r", encoding="utf-8") as log_file:
        print(log_file.read())
 
    print("=== 2) Istatistiksel Profil Testi (--profile) ===")
    profile_result = analyzer.profile_data(mock_query_result)
    if profile_result is None:
        print("Sayisal sutun bulunamadi, profil olusturulamadi.")
    else:
        for column_name, stats in profile_result.items():
            print(f"\nSutun: {column_name}")
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name}: {stat_value}")
 
    print("\n--- Bos veri / sayisal sutunsuz veri testi ---")
    print("Bos liste profili:", analyzer.profile_data([]))
    print(
        "Sadece metinsel veri profili:",
        analyzer.profile_data([{"ad": "Ali"}, {"ad": "Veli"}]),
    )
 
    print("\n--- Tek satirlik veri seti (std -> 0.0 kontrolu) ---")
    single_row_profile = analyzer.profile_data([{"puan": 87}])
    print(single_row_profile)
 
    print("\n=== 3) CSV Export Testi (--export) ===")
    export_message = analyzer.export_to_csv(mock_query_result, filename="mock_query_result.csv")
    print(export_message)
 
    print("\n--- Bos veri ile export deneme (ValueError beklenir) ---")
    try:
        analyzer.export_to_csv([], filename="bos_dosya.csv")
    except ValueError as err:
        print(f"Beklenen hata yakalandi: {err}")