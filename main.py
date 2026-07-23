"""
main.py
--------
Modul 5 (Orkestrasyon Katmani): SQL Engine & Data Profiler CLI uygulamasinin
ana giris noktasi.

Bu modul, daha once yazilan 4 bagimsiz modulu bir araya getirir:
    - parser_engine.py     : SQL dogrulama + coklu dil ceviri + AST cikarim.
    - database_executor.py : SQLite in-memory calistirma motoru + Faker ile veri olcekleme.
    - data_analyzer.py     : Sorgu loglama, istatistiksel profilleme, CSV export.
    - ai_translator.py     : Dogal dil -> SQL cevirisi (opsiyonel AI modu).

Akis (Standart Mod):
    Girdi SQL --> [Parser: dogrula + ceviri] --> [Executor: calistir]
                --> [Analyzer: logla + profil + CSV export]

Akis (AI Modu, --ai bayragi ile):
    Girdi Dogal Dil --> [AITranslator: SQL uret] --> [Parser: dogrula + ceviri]
                      --> [Executor: calistir] --> [Analyzer: logla + profil + CSV export]

Tasarim ilkesi: Bu dosya sadece ORKESTRASYON yapar; is mantiginin tamami
ilgili modullerin sorumlulugundadir (Single Responsibility Principle).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ai_translator import AITranslator, AITranslatorError
from data_analyzer import DataAnalyzer
from database_executor import DatabaseExecutor
from parser_engine import NotASQLError, SQLEngineParser

# CLI'da kullanicinin secebilecegi kisa lehce isimlerini, parser_engine
# icindeki okunabilir etiketlere esler.
CLI_DIALECT_TO_LABEL: dict[str, str] = {
    "mysql": "MySQL",
    "postgres": "PostgreSQL",
    "tsql": "T-SQL",
}

DEFAULT_EXPORT_FILENAME = "query_result.csv"

console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Uygulamanin komut satiri arayuzunu (CLI) tanimlayan argparse
    parser'ini olusturur.

    Returns:
        Yapilandirilmis argparse.ArgumentParser nesnesi.
    """
    parser = argparse.ArgumentParser(
        prog="sql-engine-profiler",
        description=(
            "Terminal tabanli SQL Engine & Data Profiler. "
            "Ham SQL sorgularini veya dogal dil isteklerini calistirir, "
            "coklu lehceye cevirir, istatistiksel profil cikarir ve "
            "sonucu CSV olarak disa aktarir."
        ),
        epilog=(
            "Ornekler:\n"
            "  python main.py --query \"SELECT * FROM Personel\"\n"
            "  python main.py --query \"maasi 50000den buyuk olanlari getir\" --ai\n"
            "  python main.py --query \"SELECT * FROM Personel\" --target-dialect postgres --ast\n"
            "  python main.py --query \"SELECT * FROM Personel\" --profile --export\n"
            "  python main.py --query \"SELECT 1\" --scale 500\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help=(
            "Calistirilacak ham SQL sorgusu (Standart Mod) veya "
            "dogal dil istegi (--ai ile birlikte kullanilirsa AI Modu)."
        ),
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Etkinlestirilirse, --query metni dogal dil olarak kabul edilip AITranslator ile SQL'e cevrilir.",
    )
    parser.add_argument(
        "--target-dialect",
        type=str,
        choices=sorted(CLI_DIALECT_TO_LABEL.keys()),
        default=None,
        help="Sorgunun ayrica gosterilecegi hedef SQL lehcesi (mysql, postgres, tsql).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Sonuc verisinin istatistiksel profilini (mean, std, min, max, null_count) cikartir.",
    )
    parser.add_argument(
        "--export",
        nargs="?",
        const=DEFAULT_EXPORT_FILENAME,
        default=None,
        metavar="DOSYA_ADI",
        help=(
            f"Sonucu CSV dosyasina aktarir. Dosya adi verilmezse "
            f"varsayilan olarak '{DEFAULT_EXPORT_FILENAME}' kullanilir."
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        metavar="N",
        help="Sorgu calistirilmadan once Faker ile Personel/Yemekler tablolarina N satir dummy veri ekler.",
    )
    parser.add_argument(
        "--ast",
        action="store_true",
        help="Sorgunun Soyut Sozdizimi Agacini (AST) terminalde gosterir.",
    )

    return parser


def print_stage_header(stage_number: int, title: str) -> None:
    """Her yurutme asamasi icin tutarli, renkli bir baslik basar."""
    console.print(
        Panel(
            Text(title, style="bold white"),
            title=f"Adim {stage_number}",
            title_align="left",
            border_style="cyan",
            expand=False,
        )
    )


def resolve_sql_query(args: argparse.Namespace) -> str | None:
    """
    Kullanicinin girdisini (dogal dil ya da ham SQL) calistirilabilir bir
    SQL string'ine cevirir.

    AI Modu (--ai) acikken:
        - AITranslator yuklenmeye calisilir.
        - Basarili olursa dogal dil SQL'e cevrilir.
        - Model kullanilamiyorsa (kutuphane eksik/yukleme hatasi) durum
          kullaniciya net bir uyari ile bildirilir ve girdi, ham SQL
          olarak kabul edilip Standart Mod'a DUSER (graceful fallback).

    Args:
        args: argparse ile ayristirilmis komut satiri argumanlari.

    Returns:
        Calistirilmaya hazir SQL string'i. Kritik bir durumda (ornegin
        AI cevirisi sirasinda beklenmeyen bir hata olusursa) None doner.
    """
    if not args.ai:
        console.print("[bold yellow]Mod:[/bold yellow] Standart (Ham SQL)")
        return args.query

    print_stage_header(1, "AI Cevirisi (Dogal Dil -> SQL)")
    console.print(f"[bold]Girdi (Dogal Dil):[/bold] {args.query}")

    console.print("[dim]Yerel AI modeli baslatiliyor, lutfen bekleyin...[/dim]")
    translator = AITranslator()

    if not translator.is_available:
        console.print(
            Panel(
                "AI modeli yuklenemedi, [bold]AI Modu devre disi birakildi[/bold].\n"
                "Girdi, ham SQL sorgusu olarak Standart Mod'da calistirilmaya calisilacak.",
                title="⚠ Uyari",
                border_style="yellow",
            )
        )
        return args.query

    try:
        translated_sql = translator.translate_text_to_sql(args.query)
    except AITranslatorError as exc:
        console.print(
            Panel(
                f"AI cevirisi basarisiz oldu: {exc}\n"
                "Girdi, ham SQL sorgusu olarak Standart Mod'da calistirilmaya calisilacak.",
                title="⚠ Uyari",
                border_style="yellow",
            )
        )
        return args.query

    console.print(f"[bold green]Uretilen SQL:[/bold green] {translated_sql}")
    return translated_sql


def run_parser_stage(parser_engine: SQLEngineParser, sql_query: str, args: argparse.Namespace) -> bool:
    """
    SQL dogrulama, coklu lehce ceviri ve (istenirse) AST gosterim asamasini
    yurutur.

    Args:
        parser_engine: Kullanilacak SQLEngineParser ornegi.
        sql_query: Dogrulanacak/cevrilecek SQL metni.
        args: Komut satiri argumanlari (--target-dialect, --ast icin).

    Returns:
        Sorgu gecerli bir SQL ise True, degilse False (bu durumda
        cagiran fonksiyon akisi guvenli sekilde sonlandirmalidir).
    """
    print_stage_header(2, "SQL Dogrulama & Lehce Cevirisi")

    try:
        translations = parser_engine.parse_query(sql_query)
    except NotASQLError as exc:
        # Spesifikasyon geregi: gecersiz SQL, cignenmeden/cokmeden,
        # sessiz ve kontrollu bir sekilde raporlanir.
        console.print(
            Panel(
                f"Girdi gecerli bir SQL ifadesi olarak taninmadi.\nDetay: {exc}",
                title="✖ Gecersiz SQL",
                border_style="red",
            )
        )
        return False

    console.print(f"[bold green]✔ Sorgu gecerli.[/bold green] Normalize edilmis hali:")
    console.print(f"  {translations['Original']}")

    if args.target_dialect:
        label = CLI_DIALECT_TO_LABEL[args.target_dialect]
        console.print(f"\n[bold]{label} cevirisi:[/bold]")
        console.print(f"  {translations[label]}")

    if args.ast:
        console.print("\n[bold]Soyut Sozdizimi Agaci (AST):[/bold]")
        ast_text = parser_engine.get_ast(sql_query)
        console.print(Panel(ast_text, border_style="magenta", expand=False))

    return True


def run_execution_stage(
    executor: DatabaseExecutor,
    analyzer: DataAnalyzer,
    sql_query: str,
) -> tuple[list[dict[str, Any]] | str | None, float, str, int]:
    """
    SQL sorgusunu DatabaseExecutor uzerinde calistirir, sureyi olcer ve
    sonucu DataAnalyzer'a loglar.

    Args:
        executor: Kullanilacak DatabaseExecutor ornegi.
        analyzer: Kullanilacak DataAnalyzer ornegi (loglama icin).
        sql_query: Calistirilacak SQL metni.

    Returns:
        (sonuc, sure_ms, durum, satir_sayisi) seklinde 4'lu bir tuple.
        Hata durumunda sonuc None, durum "ERROR" olur.
    """
    print_stage_header(3, "Sorgu Calistirma (SQLite In-Memory)")

    start_time = time.perf_counter()
    try:
        result = executor.execute_query(sql_query)
        duration_ms = (time.perf_counter() - start_time) * 1000
        status = "SUCCESS"
        row_count = len(result) if isinstance(result, list) else 0

        analyzer.log_query(
            query=sql_query, duration_ms=duration_ms, status=status, row_count=row_count
        )

        console.print(f"[bold green]✔ Basariyla calistirildi.[/bold green]")
        console.print(f"  Sure     : {duration_ms:.2f}ms")
        console.print(f"  Sonuc    : {'SELECT sonucu (' + str(row_count) + ' satir)' if isinstance(result, list) else result}")

        return result, duration_ms, status, row_count

    except sqlite3.Error as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        status = "ERROR"

        analyzer.log_query(
            query=sql_query, duration_ms=duration_ms, status=status, row_count=0
        )

        console.print(
            Panel(
                f"Sorgu calistirilirken bir veritabani hatasi olustu:\n{exc}",
                title="✖ Calistirma Hatasi",
                border_style="red",
            )
        )
        return None, duration_ms, status, 0


def render_result_table(result: list[dict[str, Any]]) -> None:
    """Sorgu sonucunu (list of dicts) rich Table olarak terminalde gosterir."""
    if not result:
        console.print("[dim]Sorgu 0 satir dondurdu.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    for column_name in result[0].keys():
        table.add_column(str(column_name))

    # Terminali doldurmamak icin en fazla ilk 20 satir gosterilir.
    max_rows_to_display = 20
    for row in result[:max_rows_to_display]:
        table.add_row(*[str(value) for value in row.values()])

    console.print(table)

    if len(result) > max_rows_to_display:
        console.print(
            f"[dim]... ve {len(result) - max_rows_to_display} satir daha "
            f"(toplam {len(result)} satir).[/dim]"
        )


def run_analysis_stage(
    analyzer: DataAnalyzer,
    result: list[dict[str, Any]] | str | None,
    args: argparse.Namespace,
) -> None:
    """
    Sonuc verisi uzerinde (istenirse) istatistiksel profil cikarir ve
    (istenirse) CSV'ye disa aktarir.

    Args:
        analyzer: Kullanilacak DataAnalyzer ornegi.
        result: run_execution_stage'den donen ham sonuc (list of dicts,
            "Basarili" string'i veya None).
        args: Komut satiri argumanlari (--profile, --export icin).
    """
    print_stage_header(4, "Analiz & Raporlama")

    if not isinstance(result, list):
        console.print(
            "[dim]Bu sorgu satir tabanli bir sonuc dondurmedi "
            "(SELECT olmayan komut); profil/export atlaniyor.[/dim]"
        )
        return

    render_result_table(result)

    if args.profile:
        console.print("\n[bold]Istatistiksel Profil:[/bold]")
        profile = analyzer.profile_data(result)
        if profile is None:
            console.print("[dim]Sayisal sutun bulunamadi, profil olusturulamadi.[/dim]")
        else:
            profile_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
            profile_table.add_column("Sutun")
            profile_table.add_column("mean")
            profile_table.add_column("std")
            profile_table.add_column("min")
            profile_table.add_column("max")
            profile_table.add_column("null_count")

            for column_name, stats in profile.items():
                profile_table.add_row(
                    str(column_name),
                    str(stats["mean"]),
                    str(stats["std"]),
                    str(stats["min"]),
                    str(stats["max"]),
                    str(stats["null_count"]),
                )
            console.print(profile_table)

    if args.export is not None:
        try:
            export_message = analyzer.export_to_csv(result, filename=args.export)
            console.print(f"\n[bold green]✔ {export_message}[/bold green]")
        except ValueError as exc:
            console.print(f"\n[bold red]✖ CSV export basarisiz: {exc}[/bold red]")


def main() -> None:
    """
    Uygulamanin ana giris noktasi. Tum modulleri baslatir, CLI argumanlarini
    okur ve yurutme akisini (AI Modu / Standart Mod -> Parser -> Executor
    -> Analyzer) sirayla calistirir.
    """
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    console.print(
        Panel(
            "[bold]SQL Engine & Data Profiler[/bold]\n"
            "Terminal tabanli SQL calistirma, profilleme ve raporlama araci.",
            border_style="blue",
        )
    )
    console.print(f"[bold]Ham Girdi:[/bold] {args.query}\n")

    parser_engine = SQLEngineParser()
    executor = DatabaseExecutor()
    analyzer = DataAnalyzer()

    try:
        if args.scale is not None:
            print_stage_header(0, f"Veri Olcekleme (Faker ile {args.scale} satir)")
            try:
                scale_message = executor.scale_data(args.scale)
                console.print(f"[bold green]✔ {scale_message}[/bold green]\n")
            except ValueError as exc:
                console.print(f"[bold red]✖ Olcekleme basarisiz: {exc}[/bold red]\n")

        # --- Asama: Girdiyi SQL'e cozumle (AI Modu veya Standart Mod) ---
        sql_query = resolve_sql_query(args)
        if not sql_query:
            console.print("[bold red]✖ Calistirilacak bir SQL sorgusu elde edilemedi.[/bold red]")
            sys.exit(1)

        # --- Asama: SQL dogrulama + ceviri + (istege bagli) AST ---
        is_valid = run_parser_stage(parser_engine, sql_query, args)
        if not is_valid:
            sys.exit(1)

        # --- Asama: Sorguyu calistir ---
        result, duration_ms, status, row_count = run_execution_stage(executor, analyzer, sql_query)

        # --- Asama: Analiz + Loglama + CSV export ---
        run_analysis_stage(analyzer, result, args)

        console.print(
            Panel(
                f"[bold]Ozet[/bold]\n"
                f"Girdi        : {args.query}\n"
                f"Calisan SQL  : {sql_query}\n"
                f"Durum        : {status}\n"
                f"Sure         : {duration_ms:.2f}ms\n"
                f"Satir Sayisi : {row_count}",
                title="Islem Tamamlandi",
                border_style="green" if status == "SUCCESS" else "red",
            )
        )

    finally:
        executor.close()


if __name__ == "__main__":
    main()