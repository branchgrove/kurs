"""Generate reference embeddings and tokenizations with the official Python package.

    pip install model2vec tokenizers
    python scripts/reference.py [model_path_or_name]

Writes model2vec/testdata/reference.json, which `go test ./...` compares against.
"""
import json
import os
import sys

from tokenizers import Tokenizer

from model2vec import StaticModel

# Use local ../model if it exists, otherwise fall back to command line or hub name
if len(sys.argv) > 1:
    model_name_or_path = sys.argv[1]
elif os.path.exists("../model/model.safetensors"):
    model_name_or_path = "../model"
elif os.path.exists("model/model.safetensors"):
    model_name_or_path = "model"
else:
    model_name_or_path = "minishlab/potion-base-8M"

print(f"Loading model from {model_name_or_path}...")
model = StaticModel.from_pretrained(model_name_or_path)

# Load tokenizer directly for exact token ID reference
tok_path = os.path.join(model_name_or_path, "tokenizer.json")
if os.path.exists(tok_path):
    tokenizer = Tokenizer.from_file(tok_path)
else:
    tokenizer = model.tokenizer

test_cases = [
    # --------------------------------------------------------------------------
    # 1. Basic sentences & minimal inputs
    # --------------------------------------------------------------------------
    ("basic", "hello_world", "Hello, world!"),
    ("basic", "reset_password", "How do I reset my password?"),
    ("basic", "quick_brown_fox", "The quick brown fox jumps over the lazy dog."),
    ("basic", "supercali", "supercalifragilisticexpialidocious"),
    ("basic", "single_char_a", "a"),
    ("basic", "single_word", "embedding"),
    ("basic", "two_words", "static vectors"),
    ("basic", "short_phrase", "natural language processing"),

    # --------------------------------------------------------------------------
    # 2. Empty string, whitespace, and formatting
    # --------------------------------------------------------------------------
    ("whitespace", "empty_string", ""),
    ("whitespace", "single_space", " "),
    ("whitespace", "multiple_spaces", "    "),
    ("whitespace", "single_tab", "\t"),
    ("whitespace", "single_newline", "\n"),
    ("whitespace", "crlf", "\r\n"),
    ("whitespace", "mixed_whitespace_only", "\t\t\n\r\n  \t"),
    ("whitespace", "leading_trailing", "   leading and   irregular   whitespace\t\n"),
    ("whitespace", "multiple_internal_spaces", "words   with   varying     gaps"),
    ("whitespace", "tabs_and_newlines_text", "line1\nline2\r\nline3\rline4\tcolumn1\tcolumn2"),
    ("whitespace", "non_breaking_space", "100\u00a0km and 50\u00a0miles"),
    ("whitespace", "narrow_no_break_space", "thin\u202fspace\u202ftest"),
    ("whitespace", "unicode_spaces_quads", "\u2000quad \u2001em \u2002en \u2003space \u2004three \u2005four \u2006six"),
    ("whitespace", "cjk_ideographic_space", "word1\u3000word2 with ideographic space"),
    ("whitespace", "form_feed_vertical_tab", "page1\fpage2\vvertical break"),

    # --------------------------------------------------------------------------
    # 3. Control characters, invisible characters, and formatting markers
    # --------------------------------------------------------------------------
    ("control_chars", "null_byte_in_text", "null\x00byte inside string"),
    ("control_chars", "ascii_controls", "bell\x07 backspace\x08 escape\x1b test"),
    ("control_chars", "delete_char", "delete\x7fcharacter"),
    ("control_chars", "ansi_escape_seq", "\x1b[31;1mred text\x1b[0m reset"),
    ("control_chars", "zero_width_space", "zero\u200bwidth\u200bspace\u200bmarker"),
    ("control_chars", "zero_width_joiners", "zero\u200cjoiner\u200dtest"),
    ("control_chars", "byte_order_mark", "\ufeffBOM at start and inside\ufeffword"),
    ("control_chars", "soft_hyphen", "soft\u00adhyphen\u00adinside"),
    ("control_chars", "replacement_character", "replacement\ufffdcharacter in data"),
    ("control_chars", "only_null_byte", "\x00"),
    ("control_chars", "only_bom", "\ufeff"),
    ("control_chars", "only_zero_width_space", "\u200b"),

    # --------------------------------------------------------------------------
    # 4. Accents, diacritics, and normalization
    # --------------------------------------------------------------------------
    ("accents", "french_cafe", "Café crème brûlée"),
    ("accents", "french_naive", "naïve résumé of Zoë"),
    ("accents", "french_facade_chateau", "façade and château in München"),
    ("accents", "nordic_smorgasbord", "smörgåsbord in København, Tromsø and Reykjavík with Ångström"),
    ("accents", "german_eszett", "Straße, süß, große und STRAẞE"),
    ("accents", "polish_pangram", "Zażółć gęślą jaźń"),
    ("accents", "czech_pangram", "Příliš žluťoučký kůň úpěl ďábelské ódy"),
    ("accents", "spanish_sentences", "¿Cómo estás? ¡Muy bien, señor! El niño come piña en el año 2024."),
    ("accents", "portuguese_nasal", "São Paulo, não, coração e maçã com açúcar"),
    ("accents", "romanian_accents", "România, oraș, țară, pâine"),
    ("accents", "turkish_dotted_dotless", "İstanbul, Diyarbakır, ılık çay ve ağaç"),
    ("accents", "vietnamese_tones", "tiếng Việt, phở bò Hà Nội, Nguyễn Huệ"),
    ("accents", "nfd_decomposed_input", "cafe\u0301 and Zoe\u0308"),
    ("accents", "combining_multiple_accents", "a\u0300\u0301b testing stacked diacritics"),
    ("accents", "non_decomposing_latin", "ø and Ø, ł and Ł, đ and Đ, þ and Þ, ð and Ð, æ and œ"),
    ("accents", "smart_quotes_dashes", "Ünïcödé ﬁligree and “smart quotes” — dashes"),

    # --------------------------------------------------------------------------
    # 5. Punctuation, dashes, quotes, and symbols
    # --------------------------------------------------------------------------
    ("punctuation", "dashes_variety", "hyphen - en–dash — em—dash ― horizontal ‒ figure ‑ nonbreaking"),
    ("punctuation", "curly_quotes", "“smart double quotes” and ‘smart single quotes’"),
    ("punctuation", "european_quotes", "«french guillemets» and »german guillemets« and „low quotes“ and ‚single low‘"),
    ("punctuation", "brackets_parentheses", "(parentheses) [brackets] {braces} <angles>"),
    ("punctuation", "ellipsis_variations", "waiting... three dots vs waiting… unicode ellipsis"),
    ("punctuation", "repeated_punctuation", "stop!!!!! really????? no..... way------"),
    ("punctuation", "all_ascii_punctuation", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
    ("punctuation", "contractions_ascii_quotes", "don't, can't, it's, you're, they've, we'll, I'm, o'clock"),
    ("punctuation", "contractions_curly_quotes", "don’t, can’t, it’s, you’re with curly apostrophe"),
    ("punctuation", "code_compound_operators", "::scope ->arrow =>fat_arrow :=walrus"),
    ("punctuation", "boolean_comparison_operators", "&&and ||or ==equal !=notequal <=lte >=gte"),
    ("punctuation", "dashes_dots_runs", "foo--bar---baz and ...three....four..two"),
    ("punctuation", "mixed_typographic_symbols", "bullet • middle · star ★ check ✓ cross ✗ section § degree 25°C"),
    ("punctuation", "legal_trademark_symbols", "© 2024 Acme Corp. All rights reserved. Brand™ and Registered®"),
    ("punctuation", "arrows_symbols", "arrow → left ← up ↑ down ↓ bidirectional ↔"),

    # --------------------------------------------------------------------------
    # 6. Math, numbers, dates, and currency
    # --------------------------------------------------------------------------
    ("math_currency", "basic_arithmetic", "2 + 2 = 4; 10 * 5 = 50; 100 / 10 = 10; 3 - 1 = 2"),
    ("math_currency", "pi_and_powers", "3.14159 is roughly pi; 2^10 = 1024"),
    ("math_currency", "math_comparisons", "x < y, a > b, x <= y, x >= y, a != b, a == b"),
    ("math_currency", "advanced_math_symbols", "10 ± 2, 5 × 4 = 20, 10 ÷ 2 = 5, x² + y² = z², π ≈ 3.14, √16 = 4, ∞"),
    ("math_currency", "ascii_vs_unicode_math_symbols", "a+b vs a±b, x*y vs x×y, a/b vs a÷b, x=y vs x≠y, x<=y vs x≤y"),
    ("math_currency", "signed_numbers", "-1 and +42 and -3.14e-10 and +0.001"),
    ("math_currency", "formatted_numbers", "1,000,000.50 and 100,000 and 0.0001"),
    ("math_currency", "hex_binary_scientific", "0xDEADBEEF, 0x1234, 0b101010, 1e-6, 3.5E+10"),
    ("math_currency", "ordinals", "1st, 2nd, 3rd, 4th, 21st, 100th"),
    ("math_currency", "number_indicators", "#1 ranked and No. 1 choice and № 1 winner"),
    ("math_currency", "phone_numbers", "+1 (800) 555-0199 and +44 20 7946 0958"),
    ("math_currency", "dates_and_timestamps", "2024-03-25T14:30:00Z and 12/31/1999 and 25.12.2024"),
    ("math_currency", "dollar_currency", "$100 or 100$ or $ 100 or $19.99"),
    ("math_currency", "euro_currency", "€50 or 50€ or € 50 or €99.95"),
    ("math_currency", "pound_yen_currency", "£25 or £ 25 or £1,000 and ¥10000 or ¥ 10000"),
    ("math_currency", "mixed_world_currency", "₹500, ₿0.05, ₩50000, ¢50, price: $10/mo, 50% discount"),

    # --------------------------------------------------------------------------
    # 7. CJK (Chinese, Japanese, Korean)
    # --------------------------------------------------------------------------
    ("cjk", "japanese_mixed", "日本語と English mixed 中文"),
    ("cjk", "japanese_kanji", "自然言語処理と機械学習の最新動向"),
    ("cjk", "japanese_hiragana", "こんにちは、お元気ですか？ありがとう！"),
    ("cjk", "japanese_katakana", "ニューラルネットワークとトランスフォーマー"),
    ("cjk", "japanese_brackets_quotes", "「AI技術」は日々進化しています！【重要】"),
    ("cjk", "japanese_halfwidth_katakana", "ﾊﾝｶｸｶﾀｶﾅﾃｽﾄ"),
    ("cjk", "chinese_simplified", "人工智能和深度学习正在改变世界。"),
    ("cjk", "chinese_traditional", "繁體中文與機器學習模型的評測。"),
    ("cjk", "chinese_classical", "子曰：學而時習之，不亦說乎？"),
    ("cjk", "chinese_fullwidth_punct", "你好，世界！这是测试：是否能正确处理标点符号？"),
    ("cjk", "cjk_quotes_brackets", "「引用」『二重引用』【隅付き括弧】《二重山括弧》"),
    ("cjk", "cjk_touching_ascii", "中A文 and 中文123 and 中#文"),
    ("cjk", "cjk_touching_brackets", "中文（测试）和中文【强调】"),
    ("cjk", "cjk_repeated_dashes_ellipsis", "中文——破折号与中文……省略号"),
    ("cjk", "mixed_cjk_latin_numbers", "iPhone 16 Pro 发布售价 7999 元，搭载 A18 Pro 芯片。"),
    ("cjk", "korean_hangul_syllables", "안녕하세요 자연어 처리 모델 테스트입니다."),
    ("cjk", "korean_mixed_latin", "BERT 모델 학습과 한국어 NLP 평가"),
    ("cjk", "korean_decomposed_jamo", "한글"),

    # --------------------------------------------------------------------------
    # 8. Other non-Latin scripts
    # --------------------------------------------------------------------------
    ("scripts", "cyrillic_russian_pangram", "Быстрая коричневая лиса прыгает через ленивую собаку."),
    ("scripts", "cyrillic_russian_short", "Привет, мир! Как твои дела?"),
    ("scripts", "cyrillic_ukrainian", "Київ — столиця України. Їжак у лісі."),
    ("scripts", "greek_pangram", "Ο ταχύς καστανός αλώπηξ πηδάει πάνω από τον νωθρό σκύλο."),
    ("scripts", "greek_conversational", "Γειά σου κόσμε! 123"),
    ("scripts", "arabic_sentence", "الذكاء الاصطناعي يغير العالم المعاصر."),
    ("scripts", "arabic_tashkeel", "اللُّغَةُ العَرَبِيَّةُ جَمِيلَةٌ"),
    ("scripts", "hebrew_sentence", "שלום עולם, מה שלומך היום?"),
    ("scripts", "hindi_devanagari", "नमस्ते दुनिया, कृत्रिम बुद्धिमत्ता"),
    ("scripts", "thai_sentence", "สวัสดีชาวโลก ภาษาไทย"),

    # --------------------------------------------------------------------------
    # 9. Emojis and multibyte unicode
    # --------------------------------------------------------------------------
    ("emojis", "inline_emojis", "Hello! 👋 I'm excited 🎉 to test this 🚀"),
    ("emojis", "emojis_only", "😀😃😄😁😆😅🤣"),
    ("emojis", "skin_tone_modifiers", "thumbs up: 👍 👍🏻 👍🏼 👍🏽 👍🏾 👍🏿"),
    ("emojis", "flag_sequences", "flags: 🇺🇸 🇬🇧 🇩🇪 🇫🇷 🇯🇵 🇨🇳"),
    ("emojis", "zwj_sequences", "family: 👨‍👩‍👧‍👦 and rainbow: 🏳️‍🌈"),
    ("emojis", "tech_space_emojis", "robots 🤖 and aliens 👽 in space 🛸"),

    # --------------------------------------------------------------------------
    # 10. Casing, acronyms, and identifier conventions
    # --------------------------------------------------------------------------
    ("identifiers", "all_uppercase", "THIS IS A FULLY CAPITALIZED SENTENCE WITH ACRONYMS LIKE NASA, FBI, HTML5, JSON, REST"),
    ("identifiers", "all_lowercase", "this is an all lowercase sentence without any capitalization"),
    ("identifiers", "title_case", "This Is Title Case For Every Word In The Sentence"),
    ("identifiers", "fullwidth_latin_and_digits", "ＡＢＣ ａｂｃ １２３ ４５６"),
    ("identifiers", "camel_case", "camelCaseIdentifier and PascalCaseClass and mixedCase123"),
    ("identifiers", "snake_case", "snake_case_variable and SCREAMING_SNAKE_CASE_CONSTANT"),
    ("identifiers", "kebab_case", "kebab-case-url-slug and TRAIN-CASE-HEADER"),
    ("identifiers", "tech_brands", "iPhone, iPad, macOS, eBay, YouTube, GitHub, OpenAI"),

    # --------------------------------------------------------------------------
    # 11. Code, URLs, paths, and technical syntax
    # --------------------------------------------------------------------------
    ("code", "full_url", "https://huggingface.co/minishlab/potion-base-8M?param=value#target"),
    ("code", "complex_auth_url", "https://user:pass@192.168.1.1:8080/api/v2/items?filter[name]=foo&sort=-date#section-2"),
    ("code", "json_stringified", "{\"status\": 200, \"message\": \"ok\", \"items\": [1, 2, 3], \"active\": true}"),
    ("code", "email_addresses", "e-mail: foo.bar@example.com (urgent!!!) contact@sub.domain.org"),
    ("code", "unix_paths", "/usr/local/bin/python3 and /var/log/nginx/access.log"),
    ("code", "windows_paths", "C:\\Program Files\\Go\\bin\\go.exe and D:\\data\\file.csv"),
    ("code", "go_syntax", "if err != nil { return fmt.Errorf(\"failed: %w\", err) }"),
    ("code", "python_syntax", "def encode(self, texts: list[str]) -> np.ndarray:"),
    ("code", "sql_query", "SELECT users.id, users.name FROM users WHERE active = 1 ORDER BY id DESC;"),
    ("code", "html_snippet", "<div class=\"main\" id=\"app\">Hello &amp; welcome!</div>"),
    ("code", "docker_command", "docker run -it --rm -v /host:/container:ro alpine:latest sh"),
    ("code", "git_command", "git commit -m \"feat(tokenizer): add support for BertNormalizer\""),

    # --------------------------------------------------------------------------
    # 12. WordPiece subword boundaries and length limits
    # --------------------------------------------------------------------------
    ("wordpiece", "antidisestablishment", "antidisestablishmentarianism and pseudopseudohypoparathyroidism"),
    ("wordpiece", "word_len_99", "a" * 99),
    ("wordpiece", "word_len_100", "a" * 100),
    ("wordpiece", "word_len_101", "a" * 101),
    ("wordpiece", "word_len_150", "b" * 150),
    ("wordpiece", "long_word_surrounded", "start " + "x" * 120 + " end"),
    ("wordpiece", "raw_subword_prefix", "##prefix and middle##sub and ##"),
    ("wordpiece", "alphabet_lowercase", "a b c d e f g h i j k l m n o p q r s t u v w x y z"),
    ("wordpiece", "alphabet_uppercase", "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"),
    ("wordpiece", "digits_zero_to_nine", "0 1 2 3 4 5 6 7 8 9"),
    ("wordpiece", "repeated_letters", "aaaaabbbbbcccccdddddeeeee"),
    ("wordpiece", "unmatchable_wordpiece", "zzzzzzzzzzzzzzzz"),
    ("wordpiece", "unknown_codepoints", "test \ufffe \uffff unk"),

    # --------------------------------------------------------------------------
    # 13. Sequence truncation stress test (> 512 tokens)
    # --------------------------------------------------------------------------
    ("truncation", "long_sequence_600_words", " ".join(["The quick brown fox jumps over the lazy dog repeatedly."] * 75)),
]

print(f"Tokenizing and embedding {len(test_cases)} test cases...")

out_refs = []
batch_sentences = [tc[2] for tc in test_cases]
batch_embeddings = model.encode(batch_sentences)

for (category, label, text), emb in zip(test_cases, batch_embeddings):
    # Get exact token IDs from Hugging Face Tokenizer
    enc = tokenizer.encode(text, add_special_tokens=False)
    out_refs.append({
        "category": category,
        "label": label,
        "text": text,
        "tokens": enc.ids,
        "embedding": emb.tolist(),
    })

# Output directory: testdata
out_dir = os.path.join(os.path.dirname(__file__), "..", "testdata")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "reference.json")

with open(out_file, "w") as f:
    json.dump(out_refs, f, indent=2)

print(f"Wrote {len(out_refs)} reference test cases to {out_file}")
print(f"Embedding dimension: {len(out_refs[0]['embedding'])}")
