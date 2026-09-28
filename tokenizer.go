package kurs

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// A BERT-style WordPiece tokenizer that reads a HuggingFace tokenizer.json.
// It supports the BertNormalizer + BertPreTokenizer + WordPiece setup used by
// the English potion-* models. Anything else is rejected loudly rather than
// silently producing different tokens than the Python reference.

type tokenizer struct {
	vocab        map[string]int32
	unk          int32 // -1 if the model has no unk token
	prefix       string
	maxWordChars int

	cleanText, chinese, stripAccents, lowercase bool
}

type normalizerConfig struct {
	Type               string             `json:"type"`
	CleanText          *bool              `json:"clean_text"`
	HandleChineseChars *bool              `json:"handle_chinese_chars"`
	StripAccents       *bool              `json:"strip_accents"`
	Lowercase          *bool              `json:"lowercase"`
	Normalizers        []normalizerConfig `json:"normalizers"`
}

type preTokenizerConfig struct {
	Type          string               `json:"type"`
	PreTokenizers []preTokenizerConfig `json:"pre_tokenizers"`
}

type tokenizerJSON struct {
	Normalizer   *normalizerConfig   `json:"normalizer"`
	PreTokenizer *preTokenizerConfig `json:"pre_tokenizer"`
	Model        struct {
		Type     string           `json:"type"`
		Vocab    map[string]int32 `json:"vocab"`
		UnkToken string           `json:"unk_token"`
		Prefix   *string          `json:"continuing_subword_prefix"`
		MaxChars int              `json:"max_input_chars_per_word"`
	} `json:"model"`
}

func findBertNormalizer(n *normalizerConfig) (*normalizerConfig, error) {
	if n == nil {
		return nil, nil
	}
	if n.Type == "BertNormalizer" {
		return n, nil
	}
	if n.Type == "Sequence" {
		var bertNorm *normalizerConfig
		for i := range n.Normalizers {
			found, err := findBertNormalizer(&n.Normalizers[i])
			if err != nil {
				return nil, err
			}
			if found != nil {
				if bertNorm != nil {
					return nil, fmt.Errorf("tokenizer: multiple BertNormalizers in sequence")
				}
				bertNorm = found
			}
		}
		if bertNorm == nil {
			return nil, fmt.Errorf("tokenizer: no BertNormalizer found in sequence")
		}
		return bertNorm, nil
	}
	return nil, fmt.Errorf("tokenizer: unsupported normalizer %q (only BertNormalizer)", n.Type)
}

func checkBertPreTokenizer(p *preTokenizerConfig) error {
	if p == nil {
		return nil
	}
	if p.Type == "BertPreTokenizer" {
		return nil
	}
	if p.Type == "Sequence" {
		hasBert := false
		for i := range p.PreTokenizers {
			if err := checkBertPreTokenizer(&p.PreTokenizers[i]); err != nil {
				return err
			}
			hasBert = true
		}
		if !hasBert {
			return fmt.Errorf("tokenizer: empty pre_tokenizer sequence")
		}
		return nil
	}
	return fmt.Errorf("tokenizer: unsupported pre_tokenizer %q (only BertPreTokenizer)", p.Type)
}

func newTokenizer(data []byte) (*tokenizer, error) {
	var j tokenizerJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	if j.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q (only WordPiece)", j.Model.Type)
	}
	if len(j.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: vocabulary is empty or missing")
	}
	if err := checkBertPreTokenizer(j.PreTokenizer); err != nil {
		return nil, err
	}
	t := &tokenizer{
		vocab:        j.Model.Vocab,
		unk:          -1,
		prefix:       "##",
		maxWordChars: 100,
	}
	if j.Model.Prefix != nil {
		t.prefix = *j.Model.Prefix
	}
	if j.Model.MaxChars > 0 {
		t.maxWordChars = j.Model.MaxChars
	}
	if id, ok := t.vocab[j.Model.UnkToken]; ok {
		t.unk = id
	}
	normCfg, err := findBertNormalizer(j.Normalizer)
	if err != nil {
		return nil, err
	}
	if normCfg != nil {
		def := func(p *bool, d bool) bool {
			if p == nil {
				return d
			}
			return *p
		}
		t.cleanText = def(normCfg.CleanText, true)
		t.chinese = def(normCfg.HandleChineseChars, true)
		t.lowercase = def(normCfg.Lowercase, true)
		t.stripAccents = def(normCfg.StripAccents, t.lowercase) // HF: null => follows lowercase
	}
	return t, nil
}

// encode appends the token ids for text to out. Unknown words yield t.unk.
func (t *tokenizer) encode(text string, out []int32) []int32 {
	text = t.normalize(text)
	wordStart := -1
	for i, r := range text {
		switch {
		case unicode.IsSpace(r):
			if wordStart >= 0 {
				out = t.wordpiece(text[wordStart:i], out)
				wordStart = -1
			}
		case isPunct(r):
			if wordStart >= 0 {
				out = t.wordpiece(text[wordStart:i], out)
				wordStart = -1
			}
			rlen := utf8.RuneLen(r)
			out = t.wordpiece(text[i:i+rlen], out)
		default:
			if wordStart < 0 {
				wordStart = i
			}
		}
	}
	if wordStart >= 0 {
		out = t.wordpiece(text[wordStart:], out)
	}
	return out
}

func (t *tokenizer) wordpiece(w string, out []int32) []int32 {
	// BERT limits word length to maxWordChars (default 100).
	// If rune count exceeds maxWordChars, it is [UNK].
	var stackOffsets [101]uint32
	var runeOffsets []uint32
	if t.maxWordChars <= 100 {
		runeOffsets = stackOffsets[:t.maxWordChars+1]
	} else {
		runeOffsets = make([]uint32, t.maxWordChars+1)
	}

	nRune := 0
	for idx := range w {
		if nRune >= t.maxWordChars {
			return append(out, t.unk)
		}
		runeOffsets[nRune] = uint32(idx)
		nRune++
	}
	if nRune == 0 {
		return out
	}
	runeOffsets[nRune] = uint32(len(w))

	start := len(out)
	var scratch [128]byte
	prefixLen := copy(scratch[:], t.prefix)

	for s := 0; s < nRune; {
		e := nRune
		found := int32(-1)
		for ; e > s; e-- {
			sub := w[runeOffsets[s]:runeOffsets[e]]
			if s == 0 {
				if id, ok := t.vocab[sub]; ok {
					found = id
					break
				}
			} else {
				subLen := len(sub)
				totalLen := prefixLen + subLen
				if totalLen <= len(scratch) {
					copy(scratch[prefixLen:], sub)
					if id, ok := t.vocab[string(scratch[:totalLen])]; ok {
						found = id
						break
					}
				} else {
					if id, ok := t.vocab[t.prefix+sub]; ok {
						found = id
						break
					}
				}
			}
		}
		if found < 0 {
			return append(out[:start], t.unk)
		}
		out = append(out, found)
		s = e
	}
	return out
}

func isPureASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

func cleanASCII(s string) string {
	needsCleaning := false
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b == '\t' || b == '\n' || b == '\r' || b < 32 || b == 127 {
			needsCleaning = true
			break
		}
	}
	if !needsCleaning {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch == '\t' || ch == '\n' || ch == '\r' {
			b.WriteByte(' ')
		} else if ch < 32 || ch == 127 {
			continue
		} else {
			b.WriteByte(ch)
		}
	}
	return b.String()
}

func (t *tokenizer) normalize(s string) string {
	if isPureASCII(s) {
		if t.cleanText {
			s = cleanASCII(s)
		}
		if t.lowercase {
			s = strings.ToLower(s)
		}
		return s
	}

	if t.cleanText || t.chinese {
		var b strings.Builder
		b.Grow(len(s))
		for _, r := range s {
			if t.cleanText {
				if isBertWhitespace(r) {
					r = ' '
				} else if isBertControl(r) {
					continue
				}
			}
			if t.chinese && isCJK(r) {
				b.WriteByte(' ')
				b.WriteRune(r)
				b.WriteByte(' ')
				continue
			}
			b.WriteRune(r)
		}
		s = b.String()
	}
	if t.stripAccents {
		d := norm.NFD.String(s)
		var b strings.Builder
		b.Grow(len(d))
		for _, r := range d {
			if !unicode.Is(unicode.Mn, r) {
				b.WriteRune(r)
			}
		}
		s = b.String()
	}
	if t.lowercase {
		s = strings.ToLower(s)
	}
	return s
}

// BERT treats ASCII space, \t, \n, \r and Unicode category Z as whitespace.
func isBertWhitespace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || unicode.Is(unicode.Z, r)
}

// BERT treats Unicode categories Cc, Cf, Co, plus NUL and U+FFFD as control characters
// (excluding \t, \n, \r which are whitespace).
func isBertControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return r == 0 || r == 0xFFFD || unicode.Is(unicode.Cc, r) || unicode.Is(unicode.Cf, r) || unicode.Is(unicode.Co, r)
}

// BERT treats all ASCII non-alphanumerics as punctuation, plus Unicode P*.
func isPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || (r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) || (r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) || (r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) || (r >= 0x2F800 && r <= 0x2FA1F)
}
