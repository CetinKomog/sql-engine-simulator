"""
ai_translator.py
------------------
Modul 4: Dogal dil (Turkce/Ingilizce) girdisini SQL sorgusuna ceviren
"Text-to-SQL" modulu.
 
Bu modul, kullanicinin "maasi 50000'den buyuk olanlari getir" gibi dogal
dilde yazdigi bir istegi, yerel (local) olarak calisan hafif bir
Hugging Face Seq2Seq modeli (varsayilan: mrm8488/t5-base-finetuned-wikiSQL)
kullanarak calistirilabilir bir SQL ifadesine cevirir.
 
Sorumluluklar (Single Responsibility Principle):
    - Modeli ilk calistirmada indirip yerel onbellege (cache) almak ve
      sonraki calistirmalarda onbellekten yuklemek.
    - Dogal dil girdisini modelin bekledigi prompt formatina donusturmek.
    - Modelin urettigi ham metni temizleyip calistirilabilir, tek satirlik
      bir SQL string'ine cevirmek.
    - `transformers` / `torch` gibi agir bagimliliklar kurulu degilse veya
      model indirilemiyorsa (ornegin internet baglantisi yoksa) programin
      COKMESINI ENGELLEMEK; bunun yerine anlamli bir hata/durum bildirmek.
 
Bu sinif SQL'in GECERLI olup olmadigini dogrulamaz; uretilen SQL, dogrulama
icin 1. modul olan SQLEngineParser'a gonderilmelidir.
"""
 
from __future__ import annotations
 
import re
import sys
from typing import Any
 
# --------------------------------------------------------------------------
# Agir bagimliliklar (torch / transformers) opsiyonel olarak import edilir.
# Bu kutuphaneler kurulu degilse ya da import sirasinda beklenmedik bir
# hata olusursa, program COKMEMELI; sadece AI ceviri ozelligi devre disi
# kalmali ve diger 3 modul (Parser, Executor, Analyzer) calismaya devam
# edebilmelidir.
# --------------------------------------------------------------------------
try:
    from transformers import pipeline, Pipeline  # type: ignore
 
    _TRANSFORMERS_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ImportError, ModuleNotFoundError veya baska sorunlar
    _TRANSFORMERS_AVAILABLE = False
    _IMPORT_ERROR = exc
    Pipeline = Any  # type: ignore  # Tip belirtimi bozulmasin diye fallback.
 
 
class AITranslatorError(Exception):
    """
    AITranslator kullanilamaz oldugunda veya ceviri sirasinda bir hata
    olustugunda firlatilir.
 
    Bu exception, main.py gibi ust katmanlarin AI ozelligi devre disi
    kaldiginda programi cokertmeden kullaniciya bilgi verebilmesini saglar.
    """
    pass
 
 
class AITranslator:
    """
    Yerel olarak calisan bir Hugging Face Seq2Seq modeli araciligiyla
    dogal dil -> SQL cevirisi yapan sinif.
 
    Varsayilan model "mrm8488/t5-base-finetuned-wikiSQL" olup, WikiSQL
    veri kumesi uzerinde egitilmis, CPU uzerinde makul hizda calisabilen
    hafif bir T5 modelidir. Farkli bir model kullanmak istenirse
    `model_name` parametresi ile degistirilebilir.
    """
 
    DEFAULT_MODEL_NAME = "mrm8488/t5-base-finetuned-wikiSQL"
 
    def __init__(self, model_name: str | None = None, max_new_tokens: int = 128) -> None:
        """
        Args:
            model_name: Kullanilacak Hugging Face model adi/yolu. None
                birakilirsa DEFAULT_MODEL_NAME kullanilir.
            max_new_tokens: Modelin uretecegi maksimum token sayisi.
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.max_new_tokens = max_new_tokens
 
        self._pipeline: "Pipeline | None" = None
        self.is_available: bool = False
        self._unavailable_reason: str | None = None
 
        self._initialize_pipeline()
 
    def _initialize_pipeline(self) -> None:
        """
        Text-to-SQL pipeline'ini yukler.
 
        Model ilk calistirmada Hugging Face Hub'dan indirilip yerel
        onbellege (varsayilan olarak `~/.cache/huggingface`) kaydedilir;
        sonraki calistirmalarda dogrudan onbellekten yuklenir.
 
        Bu metod hicbir sekilde exception firlatmaz; basarisizlik durumunda
        `self.is_available = False` olarak isaretlenir ve nedeni
        `self._unavailable_reason` icinde saklanir. Boylece programin geri
        kalani (Parser, Executor, Analyzer) etkilenmeden calismaya devam eder.
        """
        if not _TRANSFORMERS_AVAILABLE:
            self._unavailable_reason = (
                "'transformers' ve/veya 'torch' kutuphaneleri kurulu degil. "
                f"(Detay: {_IMPORT_ERROR})"
            )
            print(
                "[AITranslator] UYARI: Gerekli kutuphaneler bulunamadi, "
                "AI destekli Text-to-SQL ozelligi devre disi birakildi.",
                file=sys.stderr,
            )
            print(
                "[AITranslator] Cozum icin: pip install torch transformers",
                file=sys.stderr,
            )
            return
 
        print(f"[AITranslator] Yerel AI modeli yukleniyor: '{self.model_name}' ...")
        print(
            "[AITranslator] Ilk calistirmada model internetten indirilip "
            "onbellege alinacaktir, bu islem birkac dakika surebilir."
        )
 
        try:
            self._pipeline = pipeline(
                task="text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,  # -1: CPU uzerinde calistir (GPU zorunlulugu yok).
            )
            self.is_available = True
            print(f"[AITranslator] Model basariyla yuklendi. Hazir. ✔")
        except Exception as exc:
            # Internet baglantisi olmamasi, model adinin hatali olmasi,
            # disk alani sorunlari vb. tum senaryolar burada yakalanir.
            self.is_available = False
            self._unavailable_reason = f"Model yuklenemedi: {exc}"
            print(
                f"[AITranslator] HATA: Model yuklenemedi, AI ceviri ozelligi "
                f"devre disi birakildi. Detay: {exc}",
                file=sys.stderr,
            )
 
    def _build_prompt(self, natural_language_text: str) -> str:
        """
        Modelin dogru formatta SQL uretmesi icin girdi metnini yapilandirilmis
        bir prompt haline getirir.
 
        Not: Secilen varsayilan model (t5-base-finetuned-wikiSQL) esasen
        Ingilizce dogal dil girdileri uzerinde egitilmistir; bu nedenle
        Turkce girdilerde ceviri kalitesi Ingilizce girdilere gore daha
        dusuk olabilir. Yine de prompt formati her iki dil icin de
        modelin bekledigi "translate English to SQL:" onekiyle olusturulur.
        """
        cleaned_text = natural_language_text.strip()
        return f"translate English to SQL: {cleaned_text}"
 
    def _post_process_sql(self, raw_output: str) -> str:
        """
        Modelin urettigi ham metni temizleyip calistirilabilir, tek satirlik
        bir SQL string'ine donusturur.
 
        Islemler:
            - Bas/son bosluklarin temizlenmesi.
            - Fazladan bosluklarin tek bosluga indirgenmesi.
            - Modelin bazen sonuca ekledigi prompt kaliplarinin ("translate
              English to SQL:" gibi) temizlenmesi.
            - Sorgu sonunda eksikse noktali virgul eklenmesi.
        """
        cleaned = raw_output.strip()
 
        # Model bazen prompt'un bir kismini ciktiya yanlislikla ekleyebilir;
        # bu tarz kaliplari guvenli sekilde temizliyoruz.
        prompt_leak_patterns = [
            r"^translate English to SQL:\s*",
            r"^translate to SQL:\s*",
        ]
        for pattern in prompt_leak_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
 
        # Fazladan bosluk/satir sonlarini tek bosluga indirger.
        cleaned = " ".join(cleaned.split())
 
        if not cleaned:
            return cleaned
 
        if not cleaned.endswith(";"):
            cleaned += ";"
 
        return cleaned
 
    def translate_text_to_sql(self, natural_language_text: str) -> str:
        """
        Dogal dilde yazilmis bir istegi calistirilabilir bir SQL ifadesine cevirir.
 
        Args:
            natural_language_text: Kullanicinin Turkce veya Ingilizce olarak
                yazdigi dogal dil istegi.
                Ornek: "maasi 50000'den buyuk olanlari getir"
 
        Returns:
            Temizlenmis, calistirilabilir SQL string'i.
 
        Raises:
            AITranslatorError: Model kullanilabilir degilse (kutuphane eksikse
                veya yukleme basarisiz olduysa) ya da girdi bos ise.
        """
        if not natural_language_text or not natural_language_text.strip():
            raise AITranslatorError("Cevrilecek metin bos olamaz.")
 
        if not self.is_available or self._pipeline is None:
            raise AITranslatorError(
                "AI ceviri ozelligi su anda kullanilamiyor. "
                f"Neden: {self._unavailable_reason}"
            )
 
        prompt = self._build_prompt(natural_language_text)
 
        try:
            generation_result = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        except Exception as exc:
            raise AITranslatorError(f"Ceviri sirasinda bir hata olustu: {exc}") from exc
 
        if not generation_result:
            raise AITranslatorError("Model bos bir sonuc dondurdu.")
 
        raw_text = generation_result[0].get("generated_text", "")
        sql_query = self._post_process_sql(raw_text)
 
        if not sql_query:
            raise AITranslatorError("Model gecerli bir SQL cikisi uretemedi.")
 
        return sql_query
 
 
if __name__ == "__main__":
    # --- Self-test blogu: dogal dilden SQL uretimini yerel olarak dogrular ---
 
    print("=== AITranslator Self-Test ===\n")
 
    translator = AITranslator()
 
    test_inputs = [
        "show all employees where salary is greater than 50000",
        "maasi 50000'den buyuk olanlari getir",
    ]
 
    for test_input in test_inputs:
        print(f"\nGirdi (Dogal Dil): {test_input}")
        try:
            generated_sql = translator.translate_text_to_sql(test_input)
            print(f"Uretilen SQL     : {generated_sql}")
        except AITranslatorError as err:
            # Model kurulu degilse veya yuklenemediyse program cokmez;
            # sadece bilgilendirici bir mesaj basilir.
            print(f"AI ceviri kullanilamiyor: {err}")
 
    print("\n=== Self-Test Tamamlandi ===")