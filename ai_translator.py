"""
ai_translator.py
------------------
Modul 4: Dogal dil (Turkce/Ingilizce) girdisini SQL sorgusuna ceviren
"Text-to-SQL" modulu.

Bu modul, kullanicinin "maasi 50000'den buyuk olanlari getir" gibi dogal
dilde yazdigi bir istegi, yerel (local) olarak calisan hafif bir
Hugging Face Seq2Seq modeli (varsayilan: mrm8488/t5-base-finetuned-wikiSQL)
kullanarak calistirilabilir bir SQL ifadesine cevirir.

ONEMLI NOT (Duzeltme):
    Onceki surumde `transformers.pipeline(task="text2text-generation", ...)`
    kullaniliyordu. Bazi transformers versiyonlarinda/ortamlarinda bu task
    adi kayitli olmayabiliyor ve "Unknown task text2text-generation" hatasi
    firlatiliyordu. Bu kirilgan pipeline soyutlamasindan kacinmak icin bu
    surumde model ve tokenizer DOGRUDAN `AutoTokenizer` ve
    `AutoModelForSeq2SeqLM` ile yuklenip `model.generate()` manuel olarak
    cagriliyor. Bu yaklasim hem daha guvenilir hem de transformers
    versiyonlarindan bagimsizdir.

Sorumluluklar (Single Responsibility Principle):
    - Modeli ve tokenizer'i ilk calistirmada indirip yerel onbellege
      (cache) almak ve sonraki calistirmalarda onbellekten yuklemek.
    - Dogal dil girdisini modelin bekledigi prompt formatina donusturmek.
    - Tokenizer ile girdiyi tensore cevirip model.generate() ile SQL
      token dizisi uretmek.
    - Uretilen token dizisini decode edip temizleyerek calistirilabilir,
      tek satirlik bir SQL string'ine cevirmek.
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
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ImportError, ModuleNotFoundError veya baska sorunlar
    _TRANSFORMERS_AVAILABLE = False
    _IMPORT_ERROR = exc


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

    Model ve tokenizer, kirilgan `pipeline()` soyutlamasi yerine dogrudan
    `AutoTokenizer` ve `AutoModelForSeq2SeqLM` ile yuklenir; ceviri
    `model.generate()` metodu manuel olarak cagrilarak yapilir.
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

        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self.is_available: bool = False
        self._unavailable_reason: str | None = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Tokenizer'i ve Seq2Seq modelini dogrudan `AutoTokenizer` /
        `AutoModelForSeq2SeqLM` ile yukler.

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
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            # Modeli degerlendirme (inference) moduna alir; dropout gibi
            # egitim-ozel katmanlar devre disi kalir, sonuclar tutarli olur.
            self._model.eval()

            self.is_available = True
            print(f"[AITranslator] Model basariyla yuklendi. Hazir. ✔")
        except Exception as exc:
            # Internet baglantisi olmamasi, model adinin hatali olmasi,
            # disk alani sorunlari, versiyon uyumsuzluklari vb. tum
            # senaryolar burada yakalanir.
            self.is_available = False
            self._tokenizer = None
            self._model = None
            self._unavailable_reason = f"Model/tokenizer yuklenemedi: {exc}"
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

        Bu metod, `pipeline()` soyutlamasi yerine tokenizer ve modeli
        dogrudan kullanir:
            1. Prompt olusturulur.
            2. Tokenizer ile prompt tensore (input_ids) cevirilir.
            3. `model.generate()` ile cikti token dizisi uretilir.
            4. Tokenizer ile cikti tekrar okunabilir metne (decode) cevirilir.
            5. Metin temizlenip calistirilabilir SQL formatina getirilir.

        Args:
            natural_language_text: Kullanicinin Turkce veya Ingilizce olarak
                yazdigi dogal dil istegi.
                Ornek: "maasi 50000'den buyuk olanlari getir"

        Returns:
            Temizlenmis, calistirilabilir SQL string'i.

        Raises:
            AITranslatorError: Model kullanilabilir degilse (kutuphane eksikse
                veya yukleme basarisiz olduysa) ya da girdi bos ise ya da
                uretim/decode sirasinda bir hata olusursa.
        """
        if not natural_language_text or not natural_language_text.strip():
            raise AITranslatorError("Cevrilecek metin bos olamaz.")

        if not self.is_available or self._model is None or self._tokenizer is None:
            raise AITranslatorError(
                "AI ceviri ozelligi su anda kullanilamiyor. "
                f"Neden: {self._unavailable_reason}"
            )

        prompt = self._build_prompt(natural_language_text)

        try:
            # Tokenizer ile prompt'u model girdisine (tensor) cevirir.
            input_ids = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).input_ids

            # Gradyan hesaplamasini kapatarak (inference-only) bellek ve
            # hiz acisindan daha verimli calistiriyoruz.
            with torch.no_grad():
                output_token_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                )

            raw_text = self._tokenizer.decode(
                output_token_ids[0], skip_special_tokens=True
            )
        except Exception as exc:
            raise AITranslatorError(f"Ceviri sirasinda bir hata olustu: {exc}") from exc

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