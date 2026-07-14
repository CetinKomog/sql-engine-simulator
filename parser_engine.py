"""
parser_engine.py
-----------------
Modul 1: SQL dogrulama, cok dilli ceviri ve AST (Abstract Syntax Tree) analizi.
 
Bu modul, projenin tum SQL ayristirma (parsing) islemlerinden sorumludur.
`sqlglot` kutuphanesini kullanarak gelen metnin gecerli bir SQL ifadesi olup
olmadigini kontrol eder, gecerliyse farkli SQL lehcelerine cevirir ve
sorgunun soyut sozdizimi agacini (AST) uretir.
"""
 
from __future__ import annotations
 
import sqlglot
from sqlglot.errors import ParseError, TokenError
 
 
class NotASQLError(Exception):
    """
    Girdi metni gecerli bir SQL ifadesi olarak ayristirilamadiginda firlatilir.
 
    Bu exception, ust katmanlarin hatayi yakalayip
    kullaniciya ekrana kirmizi hata yigini (traceback) basmadan,
    sessizce (veya kontrollu bir sekilde) yanit verebilmesini saglar.
    """
    pass
 
 
class SQLEngineParser:
    """
    sqlglot tabanli SQL dogrulama, ceviri ve AST cikarim sinifi.
 
    Sorumluluklar (Single Responsibility Principle):
        - Gelen metnin gecerli SQL olup olmadigini dogrulamak.
        - Gecerli sorgulari farkli SQL lehcelerine (dialect) cevirmek.
        - Sorgunun AST temsilini okunabilir string olarak sunmak.
 
    Bu sinif herhangi bir veritabani baglantisi actmaz veya sorgu
    calistirmaz; sadece metin/AST seviyesinde analiz yapar. Sorgu
    calistirma islemi 3. modul olan Executor'un sorumlulugundadir.
    """
 
    # Ceviri yapilacak hedef lehceler ve raporlama icin kullanilacak
    # anahtar isimleri (dictionary key'leri) burada merkezi olarak tanimlanir.
    TARGET_DIALECTS: dict[str, str] = {
        "MySQL": "mysql",
        "PostgreSQL": "postgres",
        "T-SQL": "tsql",
    }
 
    def __init__(self, source_dialect: str | None = None) -> None:
        """
        Args:
            source_dialect: Girdi SQL'inin yazildigi varsayilan lehce.
                None birakilirsa sqlglot'un genel (generic) SQL
                yorumlayicisi kullanilir.
        """
        self.source_dialect = source_dialect
 
    def _parse(self, sql_text: str) -> sqlglot.expressions.Expression:
        """
        Girdi metnini sqlglot ile ayristirip Expression nesnesi dondurur.
 
        Bos/whitespace-only metinler ve sqlglot'un tokenize/parse asamasinda
        hata verdigi (yani SQL olmayan) metinler icin NotASQLError firlatilir.
 
        Raises:
            NotASQLError: Metin bos ise veya gecerli bir SQL degilse.
        """
        if sql_text is None or not sql_text.strip():
            raise NotASQLError("Bos veya None metin SQL olarak ayristirilamaz.")
 
        try:
            parsed = sqlglot.parse_one(sql_text, read=self.source_dialect)
        except (ParseError, TokenError) as exc:
            raise NotASQLError(
                f"Girdi gecerli bir SQL ifadesi degil: {exc}"
            ) from exc
        except Exception as exc:  # sqlglot bazen genel Exception da firlatabilir
            raise NotASQLError(
                f"SQL ayristirma sirasinda beklenmeyen bir hata olustu: {exc}"
            ) from exc
 
        if parsed is None:
            raise NotASQLError("Girdi hicbir SQL ifadesine karsilik gelmiyor.")
 
        return parsed
 
    def parse_query(self, sql_text: str) -> dict[str, str]:
        """
        Verilen SQL metnini dogrular ve MySQL, PostgreSQL, T-SQL lehcelerine cevirir.
 
        Args:
            sql_text: Kullanicidan alinan ham SQL metni.
 
        Returns:
            Anahtarlari lehce adi ("MySQL", "PostgreSQL", "T-SQL"),
            degerleri o lehceye cevrilmis SQL string'i olan bir dictionary.
            Ayrica orijinal (normalize edilmis) sorgu "Original" anahtari
            altinda da eklenir.
 
        Raises:
            NotASQLError: Metin gecerli bir SQL ifadesi degilse.
        """
        parsed_expression = self._parse(sql_text)
 
        translations: dict[str, str] = {
            "Original": parsed_expression.sql(dialect=self.source_dialect)
        }
 
        for label, dialect_key in self.TARGET_DIALECTS.items():
            try:
                translations[label] = parsed_expression.sql(
                    dialect=dialect_key, pretty=True
                )
            except Exception as exc:
                # Bir lehceye ceviri basarisiz olursa tum islemi durdurmak yerine
                # o lehce icin hata bilgisini dondurup digerlerine devam ediyoruz.
                translations[label] = f"[Ceviri hatasi: {exc}]"
 
        return translations
 
    def get_ast(self, sql_text: str) -> str:
        """
        Verilen SQL ifadesinin soyut sozdizimi agacini (AST) okunabilir
        string (agac/tree) formatinda dondurur.
 
        Args:
            sql_text: AST'i cikarilacak SQL metni.
 
        Returns:
            sqlglot'un `repr()` ile urettigi girintili, insan tarafindan
            okunabilir AST metni.
 
        Raises:
            NotASQLError: Metin gecerli bir SQL ifadesi degilse.
        """
        parsed_expression = self._parse(sql_text)
        return repr(parsed_expression)
 
    def is_valid_sql(self, sql_text: str) -> bool:
        """
        Yardimci metod: Metnin gecerli SQL olup olmadigini exception
        firlatmadan, sadece True/False olarak kontrol etmek icin kullanilir.
        """
        try:
            self._parse(sql_text)
            return True
        except NotASQLError:
            return False
 
 
if __name__ == "__main__":
    # Basit manuel test / kullanim ornegi.
    parser = SQLEngineParser()
 
    test_query = "SELECT id, name FROM users WHERE age > 18 ORDER BY name"
    print("--- Gecerli sorgu testi ---")
    try:
        result = parser.parse_query(test_query)
        for dialect_name, translated_sql in result.items():
            print(f"\n[{dialect_name}]\n{translated_sql}")
 
        print("\n--- AST ---")
        print(parser.get_ast(test_query))
    except NotASQLError as err:
        print(f"Hata: {err}")
 
    print("\n--- Gecersiz metin testi ---")
    invalid_text = "merhaba dunya nasilsin"
    try:
        parser.parse_query(invalid_text)
        print("Beklenmedik: hata firlatilmadi!")
    except NotASQLError as err:
        print(f"Beklenen hata yakalandi (sessiz kalinabilir): {err}")