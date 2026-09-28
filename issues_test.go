package kurs

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func buildSafetensors(tensors map[string]tensorInfo, rawData []byte) []byte {
	headerBytes, _ := json.Marshal(tensors)
	var buf bytes.Buffer
	var headerLen [8]byte
	binary.LittleEndian.PutUint64(headerLen[:], uint64(len(headerBytes)))
	buf.Write(headerLen[:])
	buf.Write(headerBytes)
	buf.Write(rawData)
	return buf.Bytes()
}

func loadRealTokAndConfig(t *testing.T) ([]byte, []byte) {
	t.Helper()
	read := func(p string) []byte {
		b, err := os.ReadFile(p)
		if err != nil {
			b, err = os.ReadFile(filepath.Join("..", p))
		}
		if err != nil {
			t.Fatalf("failed reading required file %s: %v", p, err)
		}
		return b
	}
	return read("model/tokenizer.json"), read("model/config.json")
}

// TestIssue_EmbeddingsTensorShapeMismatch verifies that kurs.New rejects
// a safetensors file where the tensor byte length is shorter than the
// shape declares ([10, 4] requires 40 floats, but only 2 floats are provided).
func TestIssue_EmbeddingsTensorShapeMismatch(t *testing.T) {
	tokBytes, cfgBytes := loadRealTokAndConfig(t)

	// 2 floats provided (8 bytes), but shape claims [10, 4] (40 floats = 160 bytes)
	raw := make([]byte, 8)
	tensors := map[string]tensorInfo{
		"embeddings": {
			DType:   "F32",
			Shape:   []int{10, 4},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	_, err := New(stBytes, tokBytes, cfgBytes)
	if err == nil {
		t.Fatalf("ISSUE: New() accepted truncated embeddings tensor (declared shape [10, 4] = 40 floats, but only 2 floats provided in buffer)")
	}
}

// TestIssue_SafetensorsUnalignedByteLength verifies that float32s rejects
// raw byte lengths that are not multiples of 4 (e.g. 7 bytes).
func TestIssue_SafetensorsUnalignedByteLength(t *testing.T) {
	raw := []byte{1, 2, 3, 4, 5, 6, 7} // 7 bytes: not a multiple of 4
	tensors := map[string]tensorInfo{
		"tensor": {
			DType:   "F32",
			Shape:   []int{1},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	st, err := parseSafetensors(stBytes)
	if err != nil {
		t.Fatalf("parseSafetensors failed: %v", err)
	}

	_, floats, err := st.float32s("tensor")
	if err == nil {
		t.Fatalf("ISSUE: float32s() silently dropped 3 trailing bytes for 7-byte F32 tensor (returned %d floats, expected error)", len(floats))
	}
}

// TestIssue_CosineVectorLengthMismatch_Panics verifies that Cosine() returns 0
// without panicking when the first vector is longer than the second.
func TestIssue_CosineVectorLengthMismatch_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ISSUE: Cosine() panicked on vectors of different lengths (len %d vs len %d): %v", 3, 1, r)
		}
	}()

	a := []float32{1.0, 0.0, 0.0}
	b := []float32{1.0}
	c := Cosine(a, b)
	if c != 0 {
		t.Fatalf("expected Cosine of mismatched vector lengths to return 0, got %f", c)
	}
}

// TestIssue_CosineSilentElementTruncation verifies that Cosine() does not silently
// truncate or ignore extra elements when vector lengths differ.
func TestIssue_CosineSilentElementTruncation(t *testing.T) {
	a := []float32{1.0, 0.0}
	b := []float32{1.0, 0.0, 100.0} // predominantly along dimension 2

	c := Cosine(a, b)
	if c == 1.0 {
		t.Fatalf("ISSUE: Cosine() silently truncated vector b from length 3 to length 2 and returned 1.0, ignoring orthogonal elements")
	}
}

// TestIssue_NegativeShapeDimensions verifies that New() rejects negative shape
// dimensions (e.g. [-1, 4]) with an error.
func TestIssue_NegativeShapeDimensions(t *testing.T) {
	tokBytes, cfgBytes := loadRealTokAndConfig(t)

	raw := make([]byte, 16)
	tensors := map[string]tensorInfo{
		"embeddings": {
			DType:   "F32",
			Shape:   []int{-1, 4},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	_, err := New(stBytes, tokBytes, cfgBytes)
	if err == nil {
		t.Fatalf("ISSUE: New() accepted negative shape dimension [-1, 4] without returning an error")
	}
}

// TestIssue_TokenizerSequenceWrapper verifies that newTokenizer supports
// normalizers or pre-tokenizers wrapped in HuggingFace Sequence containers.
func TestIssue_TokenizerSequenceWrapper(t *testing.T) {
	tokJSON := []byte(`{
		"normalizer": {
			"type": "Sequence",
			"normalizers": [
				{
					"type": "BertNormalizer",
					"clean_text": true,
					"handle_chinese_chars": true,
					"strip_accents": true,
					"lowercase": true
				}
			]
		},
		"pre_tokenizer": {
			"type": "BertPreTokenizer"
		},
		"model": {
			"type": "WordPiece",
			"unk_token": "[UNK]",
			"vocab": {"[UNK]": 0, "hello": 1}
		}
	}`)

	_, err := newTokenizer(tokJSON)
	if err != nil {
		t.Fatalf("ISSUE: newTokenizer rejected normalizer wrapped in Sequence: %v", err)
	}
}

// TestIssue_TokenizerEmptyVocab verifies that newTokenizer rejects an
// empty vocabulary with an error.
func TestIssue_TokenizerEmptyVocab(t *testing.T) {
	tokJSON := []byte(`{
		"normalizer": {"type": "BertNormalizer"},
		"pre_tokenizer": {"type": "BertPreTokenizer"},
		"model": {
			"type": "WordPiece",
			"unk_token": "[UNK]",
			"vocab": {}
		}
	}`)

	_, err := newTokenizer(tokJSON)
	if err == nil {
		t.Fatalf("ISSUE: newTokenizer accepted an empty vocabulary without returning an error")
	}
}

// TestIssue_SafetensorsI64Overflow verifies that int32s rejects
// 64-bit integers exceeding MaxInt32 with an overflow error.
func TestIssue_SafetensorsI64Overflow(t *testing.T) {
	raw := make([]byte, 8)
	binary.LittleEndian.PutUint64(raw, math.MaxInt64) // overflows int32

	tensors := map[string]tensorInfo{
		"mapping": {
			DType:   "I64",
			Shape:   []int{1},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	st, err := parseSafetensors(stBytes)
	if err != nil {
		t.Fatalf("parseSafetensors failed: %v", err)
	}

	vals, err := st.int32s("mapping")
	if err == nil {
		t.Fatalf("ISSUE: int32s() silently truncated int64 (%d) to int32 (%d) without error", uint64(math.MaxInt64), vals[0])
	}
}

// TestIssue_TokenizerMaxWordCharsAbove100_NoPanic tests that tokenizer.json
// with max_input_chars_per_word > 100 handles words longer than 100 chars
// without panicking on fixed [101]uint32 runeOffsets array.
func TestIssue_TokenizerMaxWordCharsAbove100_NoPanic(t *testing.T) {
	tokJSON := []byte(`{
	"normalizer": {"type": "BertNormalizer"},
	"pre_tokenizer": {"type": "BertPreTokenizer"},
	"model": {
		"type": "WordPiece",
		"unk_token": "[UNK]",
		"max_input_chars_per_word": 150,
		"vocab": {"[UNK]": 0, "a": 1, "##a": 2}
	}
}`)

	tok, err := newTokenizer(tokJSON)
	if err != nil {
		t.Fatalf("failed to create tokenizer: %v", err)
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("ISSUE: wordpiece panicked on word of length 120 with max_input_chars_per_word=150: %v", r)
		}
	}()

	word120 := string(bytes.Repeat([]byte("a"), 120))
	tokens := tok.encode(word120, nil)
	if len(tokens) == 0 {
		t.Fatalf("expected tokens for 120-char word, got 0")
	}
}

// TestIssue_SafetensorsI64NegativeValues verifies that int32s accepts valid
// negative int64 values (like -1 for unmapped tokens in pruned vocabularies)
// without falsely reporting an int32 overflow.
func TestIssue_SafetensorsI64NegativeValues(t *testing.T) {
	raw := make([]byte, 16)
	v1 := int64(-1)
	v2 := int64(-100)
	binary.LittleEndian.PutUint64(raw[0:8], uint64(v1))
	binary.LittleEndian.PutUint64(raw[8:16], uint64(v2))

	tensors := map[string]tensorInfo{
		"mapping": {
			DType:   "I64",
			Shape:   []int{2},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	st, err := parseSafetensors(stBytes)
	if err != nil {
		t.Fatalf("parseSafetensors failed: %v", err)
	}

	vals, err := st.int32s("mapping")
	if err != nil {
		t.Fatalf("ISSUE: int32s rejected negative int64 values: %v", err)
	}
	if len(vals) != 2 || vals[0] != -1 || vals[1] != -100 {
		t.Fatalf("unexpected values: got %v, want [-1, -100]", vals)
	}
}

// TestIssue_SafetensorsShapeIntegerOverflow verifies that tensorElementCount
// rejects shapes whose dimensions cause integer overflow when multiplied.
func TestIssue_SafetensorsShapeIntegerOverflow(t *testing.T) {
	_, err := tensorElementCount("tensor", []int{math.MaxInt/2 + 1, 2})
	if err == nil {
		t.Fatalf("ISSUE: tensorElementCount returned nil error for overflowing shape dimensions")
	}
}

// TestIssue_OptionPrecedenceOverConfig verifies that user-supplied functional options
// take precedence over values loaded from config.json.
func TestIssue_OptionPrecedenceOverConfig(t *testing.T) {
	tokBytes, _ := loadRealTokAndConfig(t)
	raw := make([]byte, 16)
	tensors := map[string]tensorInfo{
		"embeddings": {
			DType:   "F32",
			Shape:   []int{2, 2},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	// config.json explicitly sets normalize: true
	cfgJSON := []byte(`{"normalize": true}`)

	// Caller explicitly requests WithNormalize(false)
	m, err := New(stBytes, tokBytes, cfgJSON, WithNormalize(false))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if m.normalize {
		t.Fatalf("ISSUE: functional option WithNormalize(false) was overwritten by config.json")
	}
}

// TestIssue_WeightedPoolingWithoutNormalization verifies that when weights are present
// and normalize is false, mean-pooling divides by sum of weights rather than token count n.
func TestIssue_WeightedPoolingWithoutNormalization(t *testing.T) {
	tokJSON := []byte(`{
	"normalizer": {"type": "BertNormalizer"},
	"pre_tokenizer": {"type": "BertPreTokenizer"},
	"model": {
		"type": "WordPiece",
		"unk_token": "[UNK]",
		"vocab": {"[UNK]": 0, "alpha": 1, "beta": 2}
	}
}`)
	cfgJSON := []byte(`{"normalize": false}`)

	embFloats := []float32{
		0.0, 0.0,
		2.0, 4.0,
		6.0, 8.0,
	}
	embRaw := make([]byte, len(embFloats)*4)
	for i, f := range embFloats {
		binary.LittleEndian.PutUint32(embRaw[i*4:], math.Float32bits(f))
	}

	weightFloats := []float32{0.0, 1.0, 3.0}
	weightRaw := make([]byte, len(weightFloats)*4)
	for i, f := range weightFloats {
		binary.LittleEndian.PutUint32(weightRaw[i*4:], math.Float32bits(f))
	}

	tensors := map[string]tensorInfo{
		"embeddings": {
			DType:   "F32",
			Shape:   []int{3, 2},
			Offsets: [2]int{0, len(embRaw)},
		},
		"weights": {
			DType:   "F32",
			Shape:   []int{3},
			Offsets: [2]int{len(embRaw), len(embRaw) + len(weightRaw)},
		},
	}
	var rawData []byte
	rawData = append(rawData, embRaw...)
	rawData = append(rawData, weightRaw...)

	stBytes := buildSafetensors(tensors, rawData)

	m, err := New(stBytes, tokJSON, cfgJSON)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	res := m.Encode("alpha beta")
	if len(res) != 2 {
		t.Fatalf("expected 2 dims, got %d", len(res))
	}
	if math.Abs(float64(res[0]-5.0)) > 1e-4 || math.Abs(float64(res[1]-7.0)) > 1e-4 {
		t.Fatalf("ISSUE: unnormalized weighted pooling gave %v, want [5.0, 7.0]", res)
	}
}

// TestIssue_MappingNon1DShape verifies that New rejects mapping tensors with shape != 1-D.
func TestIssue_MappingNon1DShape(t *testing.T) {
	tokBytes, cfgBytes := loadRealTokAndConfig(t)
	raw := make([]byte, 16)
	tensors := map[string]tensorInfo{
		"embeddings": {
			DType:   "F32",
			Shape:   []int{2, 2},
			Offsets: [2]int{0, len(raw)},
		},
		"mapping": {
			DType:   "I32",
			Shape:   []int{2, 2},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	_, err := New(stBytes, tokBytes, cfgBytes)
	if err == nil {
		t.Fatalf("ISSUE: New accepted 2-D mapping tensor without returning an error")
	}
}

// TestIssue_CosineNearZeroAndNonFinite verifies that Cosine does not return NaN or Inf
// when vectors contain NaNs, Infs, or have degenerate magnitudes.
func TestIssue_CosineNearZeroAndNonFinite(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
	}{
		{
			name: "nan_in_a",
			a:    []float32{float32(math.NaN()), 1.0},
			b:    []float32{1.0, 1.0},
		},
		{
			name: "nan_in_b",
			a:    []float32{1.0, 1.0},
			b:    []float32{float32(math.NaN()), 1.0},
		},
		{
			name: "inf_in_both",
			a:    []float32{float32(math.Inf(1)), 1.0},
			b:    []float32{float32(math.Inf(1)), 1.0},
		},
		{
			name: "negative_inf",
			a:    []float32{float32(math.Inf(-1)), 1.0},
			b:    []float32{1.0, 1.0},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Cosine(tc.a, tc.b)
			if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) {
				t.Fatalf("ISSUE: Cosine returned non-finite value %v for case %s", got, tc.name)
			}
			if got != 0 {
				t.Fatalf("expected 0 for non-finite vector inputs, got %v", got)
			}
		})
	}
}

// TestIssue_ConfigSeqLength verifies that seq_length in config.json sets MaxTokens,
// and that functional options still take precedence.
func TestIssue_ConfigSeqLength(t *testing.T) {
	tokBytes, _ := loadRealTokAndConfig(t)
	raw := make([]byte, 16)
	tensors := map[string]tensorInfo{
		"embeddings": {
			DType:   "F32",
			Shape:   []int{2, 2},
			Offsets: [2]int{0, len(raw)},
		},
	}
	stBytes := buildSafetensors(tensors, raw)

	cfgJSON := []byte(`{"seq_length": 128}`)
	m, err := New(stBytes, tokBytes, cfgJSON)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if m.MaxTokens() != 128 {
		t.Fatalf("expected MaxTokens() == 128 from config.json, got %d", m.MaxTokens())
	}

	// Option override
	mOverride, err := New(stBytes, tokBytes, cfgJSON, WithMaxTokens(64))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	if mOverride.MaxTokens() != 64 {
		t.Fatalf("expected MaxTokens() == 64 with option override, got %d", mOverride.MaxTokens())
	}
}
