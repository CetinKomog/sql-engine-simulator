"""
database_executor.py
---------------------
Modul 2: SQLite in-memory veritabani kurulumu, sorgu calistirma ve
Faker ile dinamik veri uretimi.

Bu modul, projenin tum veritabani etkilesim (I/O) katmanidir. Bellek
icinde (in-memory) bir SQLite veritabani kurar, ornek veri ile doldurur,
disaridan gelen SQL sorgularini calistirir ve gerektiginde '--scale'
parametresi ile buyuk hacimli sahte (fake) veri uretir.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from faker import Faker


class DatabaseExecutor:
    """
    SQLite ':memory:' veritabani uzerinde calisan sorgu yurutme sinifi.

    Sorumluluklar (Single Responsibility Principle):
        - Bellek ici veritabanini kurmak ve baslangic (seed) verisini eklemek.
        - Disaridan gelen SQL metnini calistirip sonucu uygun formatta dondurmek.
        - Faker kullanarak tablolara olceklenebilir (scale) sahte veri eklemek.

    Bu sinif SQL metninin GECERLI olup olmadigini kontrol etmez; bu is
    1. modul olan SQLEngineParser'in sorumlulugundadir. DatabaseExecutor
    sadece kendisine verilen sorguyu oldugu gibi calistirir.
    """

    def __init__(self) -> None:
        # check_same_thread=False: CLI arac tek thread calissa da, ileride
        # rich/progress gibi araclarla farkli thread'lerden erisim ihtimaline
        # karsi esneklik saglar.
        self.connection: sqlite3.Connection = sqlite3.connect(
            ":memory:", check_same_thread=False
        )
        # Sonuclari dict'e cevirmeyi kolaylastirmak icin row_factory ayarlaniyor.
        self.connection.row_factory = sqlite3.Row
        self.cursor: sqlite3.Cursor = self.connection.cursor()

        self._faker = Faker("tr_TR")

        self._initialize_db()

    def _initialize_db(self) -> None:
        """
        'Yemekler' ve 'Personel' tablolarini olusturur ve her birine
        ucer satirlik baslangic (seed) verisi ekler.

        Bu metod private (isim onunde tek alt cizgi ile isaretli) oldugu
        icin sinif disindan cagirilmasi beklenmez; sadece __init__ icinde
        veritabani kurulumunun bir parcasi olarak calistirilir.
        """
        self.cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS Personel (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ad_soyad TEXT NOT NULL,
                departman TEXT NOT NULL,
                maas REAL NOT NULL,
                ise_giris_tarihi TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS Yemekler (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                yemek_adi TEXT NOT NULL,
                kategori TEXT NOT NULL,
                fiyat REAL NOT NULL,
                kalori INTEGER NOT NULL
            );
            """
        )

        seed_personel = [
            ("Ahmet Yilmaz", "Yazilim", 45000.0, "2021-03-15"),
            ("Ayse Demir", "Insan Kaynaklari", 38000.0, "2019-07-01"),
            ("Mehmet Kaya", "Muhasebe", 32000.0, "2022-11-20"),
        ]
        self.cursor.executemany(
            """
            INSERT INTO Personel (ad_soyad, departman, maas, ise_giris_tarihi)
            VALUES (?, ?, ?, ?)
            """,
            seed_personel,
        )

        seed_yemekler = [
            ("Mercimek Corbasi", "Corba", 25.0, 180),
            ("Izgara Tavuk", "Ana Yemek", 85.0, 420),
            ("Kunefe", "Tatli", 60.0, 550),
        ]
        self.cursor.executemany(
            """
            INSERT INTO Yemekler (yemek_adi, kategori, fiyat, kalori)
            VALUES (?, ?, ?, ?)
            """,
            seed_yemekler,
        )

        self.connection.commit()

    def execute_query(self, sql_string: str) -> list[dict[str, Any]] | str:
        """
        Verilen SQL komutunu calistirir.

        Args:
            sql_string: Calistirilacak SQL metni (SELECT, INSERT, UPDATE,
                DELETE, DDL vb. herhangi bir gecerli SQLite komutu olabilir).

        Returns:
            - Sorgu bir SELECT ise: her satirin sutun adi -> deger eslemesi
              oldugu bir `list[dict]` dondurulur (sonuc bos ise bos liste).
            - Sorgu SELECT degilse (INSERT/UPDATE/DELETE/DDL vb.):
              islem basariyla tamamlandiginda "Basarili" metni dondurulur.

        Raises:
            sqlite3.Error: SQLite tarafinda calisma zamani hatasi olusursa
                (ornegin olmayan bir tabloya erisim), bu exception oldugu
                gibi ust katmana (Logger/CLI) fırlatilir ki loglama modulu
                basarisiz sorguyu kaydedebilsin.
        """
        self.cursor.execute(sql_string)

        # SELECT (ve PRAGMA gibi satir donduren komutlar) icin cursor.description
        # dolu olur; INSERT/UPDATE/DELETE gibi komutlarda None kalir.
        if self.cursor.description is not None:
            column_names = [description[0] for description in self.cursor.description]
            rows = self.cursor.fetchall()
            return [dict(zip(column_names, row)) for row in rows]

        self.connection.commit()
        return "Basarili"

    def scale_data(self, row_count: int) -> str:
        """
        Faker kutuphanesini kullanarak 'Personel' ve 'Yemekler' tablolarina
        belirtilen sayida mantikli/rastgele dummy veri ekler.

        Args:
            row_count: Her iki tabloya da eklenecek satir sayisi
                (ornegin --scale 1000 icin 1000).

        Returns:
            Islemin sonucunu ozetleyen bilgilendirme mesaji.

        Raises:
            ValueError: row_count 0 veya negatif ise.
        """
        if row_count <= 0:
            raise ValueError("row_count pozitif bir tam sayi olmalidir.")

        departmanlar = [
            "Yazilim", "Insan Kaynaklari", "Muhasebe",
            "Pazarlama", "Satis", "Lojistik", "Ar-Ge",
        ]
        yemek_kategorileri = ["Corba", "Ana Yemek", "Ara Sicak", "Tatli", "Icecek", "Salata"]

        personel_batch: list[tuple[str, str, float, str]] = []
        yemekler_batch: list[tuple[str, str, float, int]] = []

        for _ in range(row_count):
            ad_soyad = self._faker.name()
            departman = self._faker.random_element(departmanlar)
            maas = round(self._faker.random_int(min=18000, max=120000) / 1.0, 2)
            ise_giris_tarihi = self._faker.date_between(
                start_date="-10y", end_date="today"
            ).isoformat()
            personel_batch.append((ad_soyad, departman, maas, ise_giris_tarihi))

            yemek_adi = self._faker.word().capitalize() + " " + self._faker.word().capitalize()
            kategori = self._faker.random_element(yemek_kategorileri)
            fiyat = round(self._faker.random_int(min=15, max=250) + self._faker.random.random(), 2)
            kalori = self._faker.random_int(min=80, max=900)
            yemekler_batch.append((yemek_adi, kategori, fiyat, kalori))

        self.cursor.executemany(
            """
            INSERT INTO Personel (ad_soyad, departman, maas, ise_giris_tarihi)
            VALUES (?, ?, ?, ?)
            """,
            personel_batch,
        )

        self.cursor.executemany(
            """
            INSERT INTO Yemekler (yemek_adi, kategori, fiyat, kalori)
            VALUES (?, ?, ?, ?)
            """,
            yemekler_batch,
        )

        self.connection.commit()

        return (
            f"Basarili: 'Personel' ve 'Yemekler' tablolarina "
            f"{row_count} satir dummy veri eklendi."
        )

    def close(self) -> None:
        """Veritabani baglantisini kapatir. Program sonunda cagrilmasi tavsiye edilir."""
        self.connection.close()


if __name__ == "__main__":
    # Basit manuel test / kullanim ornegi.
    executor = DatabaseExecutor()

    print("--- Baslangic SELECT testi (Personel) ---")
    print(executor.execute_query("SELECT * FROM Personel"))

    print("\n--- Baslangic SELECT testi (Yemekler) ---")
    print(executor.execute_query("SELECT * FROM Yemekler"))

    print("\n--- INSERT testi ---")
    print(
        executor.execute_query(
            "INSERT INTO Personel (ad_soyad, departman, maas, ise_giris_tarihi) "
            "VALUES ('Test Kullanici', 'Test', 1000, '2024-01-01')"
        )
    )

    print("\n--- scale_data testi (10 satir) ---")
    print(executor.scale_data(10))
    print("Toplam Personel sayisi:", executor.execute_query("SELECT COUNT(*) as adet FROM Personel"))
    print("Toplam Yemekler sayisi:", executor.execute_query("SELECT COUNT(*) as adet FROM Yemekler"))

    executor.close()