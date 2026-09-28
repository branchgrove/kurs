package kurs

import (
	"os"
	"strings"
	"testing"
)

func newTestTokenizer(t *testing.T) *tokenizer {
	t.Helper()
	b, err := loadTokenizerBytes()
	if err != nil {
		t.Fatalf("required tokenizer file missing: %v", err)
	}
	tok, err := newTokenizer(b)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func loadTokenizerBytes() ([]byte, error) {
	b, err := os.ReadFile("model/tokenizer.json")
	if err != nil {
		b, err = os.ReadFile("../model/tokenizer.json")
	}
	return b, err
}

// TestTokenizerNormalization verifies BERT normalizer components in isolation.
func TestTokenizerNormalization(t *testing.T) {
	tok := newTestTokenizer(t)

	tests := []struct {
		name  string
		input string
		want  string
	}{
		// Lowercase & basic
		{"lowercase_ascii", "HELLO WORLD", "hello world"},
		{"lowercase_mixed", "CamelCase and UPPER", "camelcase and upper"},

		// Accents removal (NFD + Mn stripping)
		{"strip_accents_french", "café crème brûlée", "cafe creme brulee"},
		{"strip_accents_diaeresis", "naïve Zoë", "naive zoe"},
		{"strip_accents_cedilla", "façade", "facade"},
		{"strip_accents_spanish", "el niño come piña", "el nino come pina"},
		{"strip_accents_umlaut", "München und schön", "munchen und schon"},
		{"strip_accents_combining_nfd", "cafe\u0301 and Zoe\u0308", "cafe and zoe"},

		// Non-decomposing characters in NFD stay intact (while umlauts on vowels get stripped)
		{"non_decomposing_german_eszett", "Straße und süß", "straße und suß"},
		{"non_decomposing_capital_sharp_s", "STRAẞE", "straße"},
		{"non_decomposing_ligatures", "æ and œ", "æ and œ"},
		{"non_decomposing_nordic_slash_o", "København and Tromsø", "københavn and tromsø"},

		// Whitespace handling in BertNormalizer
		{"whitespace_tabs_newlines", "line1\tline2\nline3\rline4", "line1 line2 line3 line4"},
		{"whitespace_multiple_spaces", "word1    word2", "word1    word2"},
		{"whitespace_non_breaking", "100\u00a0km", "100 km"},
		{"whitespace_cjk_ideographic_space", "word1\u3000word2", "word1 word2"},
		{"whitespace_quad_spaces", "\u2000quad \u2001em", " quad  em"},

		// Control characters deleted by BertNormalizer (cleanText)
		{"control_null_byte", "null\x00byte", "nullbyte"},
		{"control_ascii_controls", "bell\x07 backspace\x08 escape\x1b", "bell backspace escape"},
		{"control_form_feed_vertical_tab", "page1\fpage2\vbreak", "page1page2break"},
		{"control_del_char", "delete\x7fchar", "deletechar"},
		{"control_replacement_char", "unknown\ufffdchar", "unknownchar"},
		{"control_zero_width_space", "zero\u200bwidth\u200bspace", "zerowidthspace"},
		{"control_zero_width_joiners", "zero\u200cjoiner\u200dtest", "zerojoinertest"},
		{"control_byte_order_mark", "\ufeffBOM text\ufeff", "bom text"},
		{"control_soft_hyphen", "soft\u00adhyphen", "softhyphen"},
		{"control_private_use", "apple\uf8fflogo", "applelogo"},

		// CJK Ideograph padding (spaces around each CJK ideograph)
		{"cjk_chinese_chars", "中文测试", " 中  文  测  试 "},
		{"cjk_touching_ascii", "中A文123", " 中 a 文 123"},
		{"cjk_touching_punct", "中#文", " 中 # 文 "},
		{"cjk_japanese_kanji", "日本語", " 日  本  語 "},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.normalize(tt.input)
			if got != tt.want {
				t.Errorf("normalize(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// TestWordPieceLimits tests the maxWordChars (100) boundary conditions.
func TestWordPieceLimits(t *testing.T) {
	tok := newTestTokenizer(t)

	t.Run("word_length_99", func(t *testing.T) {
		w99 := strings.Repeat("a", 99)
		tokens := tok.encode(w99, nil)
		if len(tokens) == 1 && tokens[0] == tok.unk {
			t.Errorf("99-char word should not be [UNK]")
		}
	})

	t.Run("word_length_100", func(t *testing.T) {
		w100 := strings.Repeat("a", 100)
		tokens := tok.encode(w100, nil)
		if len(tokens) == 1 && tokens[0] == tok.unk {
			t.Errorf("100-char word should not be [UNK]")
		}
	})

	t.Run("word_length_101", func(t *testing.T) {
		w101 := strings.Repeat("a", 101)
		tokens := tok.encode(w101, nil)
		if len(tokens) != 1 || tokens[0] != tok.unk {
			t.Errorf("101-char word should be [UNK], got tokens: %v", tokens)
		}
	})

	t.Run("word_length_150", func(t *testing.T) {
		w150 := strings.Repeat("b", 150)
		tokens := tok.encode(w150, nil)
		if len(tokens) != 1 || tokens[0] != tok.unk {
			t.Errorf("150-char word should be [UNK], got tokens: %v", tokens)
		}
	})

	t.Run("long_word_surrounded_by_valid_words", func(t *testing.T) {
		text := "start " + strings.Repeat("x", 120) + " end"
		tokens := tok.encode(text, nil)
		// Should have tokens for "start", UNK for 120 x's, and tokens for "end"
		if len(tokens) < 3 {
			t.Fatalf("expected at least 3 tokens, got %d: %v", len(tokens), tokens)
		}
		// Second word must be UNK
		if tokens[1] != tok.unk {
			t.Errorf("expected middle token to be [UNK], got %d", tokens[1])
		}
		// First and last words must NOT be UNK
		if tokens[0] == tok.unk || tokens[len(tokens)-1] == tok.unk {
			t.Errorf("boundary words should not be [UNK], got %v", tokens)
		}
	})

	t.Run("punctuation_splits_preventing_long_word", func(t *testing.T) {
		// 50 chars + '-' + 50 chars = 101 chars total, but '-' splits them
		// into two 50-char words, so neither is UNK!
		text := strings.Repeat("a", 50) + "-" + strings.Repeat("b", 50)
		tokens := tok.encode(text, nil)
		for _, tokId := range tokens {
			if tokId == tok.unk {
				t.Errorf("hyphen-split words should not be [UNK], got tokens: %v", tokens)
				break
			}
		}
	})
}

// TestIsPunct verifies ASCII and Unicode punctuation categorization.
func TestIsPunct(t *testing.T) {
	// All 32 ASCII punctuation characters must be true
	asciiPunct := "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
	for _, r := range asciiPunct {
		if !isPunct(r) {
			t.Errorf("isPunct(%q, U+%04X) = false, want true", r, r)
		}
	}

	// ASCII letters and digits must be false
	asciiAlphaNum := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	for _, r := range asciiAlphaNum {
		if isPunct(r) {
			t.Errorf("isPunct(%q, U+%04X) = true, want false", r, r)
		}
	}

	// Unicode punctuation must be true
	unicodePunct := []rune{'“', '”', '‘', '’', '«', '»', '—', '–', '…', '·', '•', '¿', '¡', '、', '。', '！', '？'}
	for _, r := range unicodePunct {
		if !isPunct(r) {
			t.Errorf("isPunct(%q, U+%04X) = false, want true", r, r)
		}
	}

	// Non-punctuation Unicode symbols (Sm, Sc, So) must be false
	unicodeSymbols := []rune{
		'±', '×', '÷', '≠', '≤', '≥', '≈', '∞', '√', // Math (Sm)
		'€', '£', '¥', '₹', '₿', '¢', '₩', // Currency (Sc)
		'©', '®', '™', '°', '★', '✓', '✗', // Other symbols (So)
	}
	for _, r := range unicodeSymbols {
		if isPunct(r) {
			t.Errorf("isPunct(%q, U+%04X) = true, want false", r, r)
		}
	}
}

// TestTokenizerRejections verifies that unsupported configurations are rejected.
func TestTokenizerRejections(t *testing.T) {
	validBPE := []byte(`{
		"model": {"type": "BPE", "vocab": {}}
	}`)
	if _, err := newTokenizer(validBPE); err == nil {
		t.Error("expected error for BPE model type, got nil")
	}

	unsupportedPreTok := []byte(`{
		"model": {"type": "WordPiece", "vocab": {}},
		"pre_tokenizer": {"type": "Whitespace"}
	}`)
	if _, err := newTokenizer(unsupportedPreTok); err == nil {
		t.Error("expected error for Whitespace pre_tokenizer, got nil")
	}

	unsupportedNorm := []byte(`{
		"model": {"type": "WordPiece", "vocab": {}},
		"normalizer": {"type": "NFKC"}
	}`)
	if _, err := newTokenizer(unsupportedNorm); err == nil {
		t.Error("expected error for NFKC normalizer, got nil")
	}

	unsupportedSeqNorm := []byte(`{
		"model": {"type": "WordPiece", "vocab": {"[UNK]": 0}},
		"normalizer": {
			"type": "Sequence",
			"normalizers": [
				{"type": "BertNormalizer"},
				{"type": "NFKC"}
			]
		}
	}`)
	if _, err := newTokenizer(unsupportedSeqNorm); err == nil {
		t.Error("expected error for sequence containing unsupported normalizer, got nil")
	}

	unsupportedSeqPreTok := []byte(`{
		"model": {"type": "WordPiece", "vocab": {"[UNK]": 0}},
		"pre_tokenizer": {
			"type": "Sequence",
			"pre_tokenizers": [
				{"type": "BertPreTokenizer"},
				{"type": "Whitespace"}
			]
		}
	}`)
	if _, err := newTokenizer(unsupportedSeqPreTok); err == nil {
		t.Error("expected error for sequence containing unsupported pre-tokenizer, got nil")
	}

	malformedJSON := []byte(`{ invalid json `)
	if _, err := newTokenizer(malformedJSON); err == nil {
		t.Error("expected error for malformed JSON, got nil")
	}
}
