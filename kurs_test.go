package kurs

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func load(t *testing.T, opts ...Option) *Model {
	t.Helper()
	read := func(p string) []byte {
		b, err := os.ReadFile(p)
		if err != nil {
			b, err = os.ReadFile(filepath.Join("..", p))
		}
		if err != nil {
			t.Fatalf("required model file missing: %v (download the model into ./model first)", err)
		}
		return b
	}
	m, err := New(read("model/model.safetensors"), read("model/tokenizer.json"), read("model/config.json"), opts...)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

type referenceCase struct {
	Category  string    `json:"category"`
	Label     string    `json:"label"`
	Text      string    `json:"text"`
	Tokens    []int32   `json:"tokens"`
	Embedding []float32 `json:"embedding"`
}

func loadReferences(t *testing.T) []referenceCase {
	t.Helper()
	b, err := os.ReadFile("testdata/reference.json")
	if err != nil {
		b, err = os.ReadFile(filepath.Join("..", "testdata", "reference.json"))
	}
	if err != nil {
		t.Fatalf("required reference file missing: %v (run scripts/reference.py to create testdata/reference.json)", err)
	}
	var refs []referenceCase
	if err := json.Unmarshal(b, &refs); err != nil {
		t.Fatal(err)
	}
	return refs
}

// Compares against embeddings produced by scripts/reference.py.
func TestParityWithPython(t *testing.T) {
	m := load(t)
	refs := loadReferences(t)

	for _, r := range refs {
		testName := fmt.Sprintf("%s/%s", r.Category, r.Label)
		t.Run(testName, func(t *testing.T) {
			got := m.Encode(r.Text)
			zero := isZeroVector(r.Embedding)
			if zero {
				// Empty or all-unknown input must yield an all-zero vector in Go too
				if !isZeroVector(got) {
					t.Errorf("%q: expected zero vector, got %v", r.Text, got[:min(len(got), 8)])
				}
				return
			}
			c := Cosine(got, r.Embedding)
			if c < 0.9999 {
				t.Errorf("%q: cosine vs python = %.6f (tokenizer or pooling mismatch?)", r.Text, c)
			}
		})
	}
}

// Compares exact token IDs produced by Go's WordPiece tokenizer against
// the HuggingFace tokenizer output in testdata/reference.json.
func TestTokenizerParityWithPython(t *testing.T) {
	m := load(t)
	refs := loadReferences(t)

	for _, r := range refs {
		testName := fmt.Sprintf("%s/%s", r.Category, r.Label)
		t.Run(testName, func(t *testing.T) {
			got := m.tok.encode(r.Text, nil)
			if len(got) != len(r.Tokens) {
				t.Fatalf("token count mismatch: got %d tokens %v, want %d tokens %v", len(got), got, len(r.Tokens), r.Tokens)
			}
			for i := range got {
				if got[i] != r.Tokens[i] {
					t.Errorf("token[%d] = %d, want %d (all got: %v, all want: %v)", i, got[i], r.Tokens[i], got, r.Tokens)
					break
				}
			}
		})
	}
}

// TestBatchParity ensures EncodeBatch produces identical embeddings to Encode.
func TestBatchParity(t *testing.T) {
	m := load(t)
	refs := loadReferences(t)

	texts := make([]string, len(refs))
	for i, r := range refs {
		texts[i] = r.Text
	}

	batchEmbs := m.EncodeBatch(texts)
	if len(batchEmbs) != len(texts) {
		t.Fatalf("batch size mismatch: got %d, want %d", len(batchEmbs), len(texts))
	}

	for i, text := range texts {
		single := m.Encode(text)
		batch := batchEmbs[i]
		if len(single) != len(batch) {
			t.Fatalf("text %d (%q): dim mismatch: single %d, batch %d", i, text, len(single), len(batch))
		}
		for j := range single {
			if single[j] != batch[j] {
				t.Fatalf("text %d (%q): element %d mismatch: single %f, batch %f", i, text, j, single[j], batch[j])
			}
		}
	}
}

// TestZeroVectorHandling verifies that edge cases yielding 0 known tokens
// safely produce an all-zero vector with no NaN, Inf, or panics.
func TestZeroVectorHandling(t *testing.T) {
	m := load(t)
	zeroInputs := []string{
		"",
		" ",
		"   ",
		"\t",
		"\n",
		"\r\n",
		"\t\n\r  ",
		"\x00",
		"\x00\x00\x00",
		"\ufeff",
		"\u200b\u200c\u200d",
		"\ufffe\uffff", // unassigned / non-characters
	}

	for _, input := range zeroInputs {
		t.Run(fmt.Sprintf("%q", input), func(t *testing.T) {
			got := m.Encode(input)
			if len(got) != m.Dim() {
				t.Fatalf("dim mismatch: got %d, want %d", len(got), m.Dim())
			}
			for i, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("invalid float at index %d: %v", i, v)
				}
				if v != 0 {
					t.Fatalf("expected 0 at index %d, got %v", i, v)
				}
			}
		})
	}
}

// TestTruncation tests the WithMaxTokens functional option.
func TestTruncation(t *testing.T) {
	mDefault := load(t)
	longText := "The quick brown fox jumps over the lazy dog repeatedly and continuously forever."

	fullTokens := mDefault.tok.encode(longText, nil)
	if len(fullTokens) < 10 {
		t.Fatalf("expected long text to have at least 10 tokens, got %d", len(fullTokens))
	}

	// Default MaxTokens is 512
	defaultEmb := mDefault.Encode(longText)

	// Truncate to 5 tokens with functional option
	mTruncated := load(t, WithMaxTokens(5))
	if mTruncated.MaxTokens() != 5 {
		t.Fatalf("expected MaxTokens() == 5, got %d", mTruncated.MaxTokens())
	}
	truncatedEmb := mTruncated.Encode(longText)

	if isZeroVector(defaultEmb) || isZeroVector(truncatedEmb) {
		t.Fatal("embeddings should not be zero")
	}

	// Truncated embedding should differ from full embedding
	c := Cosine(defaultEmb, truncatedEmb)
	if c > 0.9999 {
		t.Errorf("expected truncated embedding to differ from full embedding, got cosine %.6f", c)
	}
}

// TestCosineProperties tests mathematical properties of the Cosine function.
func TestCosineProperties(t *testing.T) {
	v1 := []float32{1, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0}
	v3 := []float32{-1, 0, 0, 0}
	zero := []float32{0, 0, 0, 0}

	// Self-similarity of non-zero vector must equal 1.0
	if c := Cosine(v1, v1); math.Abs(float64(c-1.0)) > 1e-6 {
		t.Errorf("Cosine(v1, v1) = %f, want 1.0", c)
	}

	// Orthogonal vectors must equal 0.0
	if c := Cosine(v1, v2); math.Abs(float64(c)) > 1e-6 {
		t.Errorf("Cosine(v1, v2) = %f, want 0.0", c)
	}

	// Opposite vectors must equal -1.0
	if c := Cosine(v1, v3); math.Abs(float64(c+1.0)) > 1e-6 {
		t.Errorf("Cosine(v1, v3) = %f, want -1.0", c)
	}

	// Zero vector must return 0.0 without NaN
	if c := Cosine(v1, zero); c != 0 || math.IsNaN(float64(c)) {
		t.Errorf("Cosine(v1, zero) = %f, want 0.0", c)
	}
	if c := Cosine(zero, zero); c != 0 || math.IsNaN(float64(c)) {
		t.Errorf("Cosine(zero, zero) = %f, want 0.0", c)
	}
}

// TestConcurrency verifies that Model.Encode is safe for concurrent use.
func TestConcurrency(t *testing.T) {
	m := load(t)
	queries := []string{
		"Hello, world!",
		"How do I reset my password?",
		"Café crème brûlée",
		"日本語と English mixed 中文",
		"supercalifragilisticexpialidocious",
	}

	expected := make([][]float32, len(queries))
	for i, q := range queries {
		expected[i] = m.Encode(q)
	}

	var wg sync.WaitGroup
	const goroutines = 20
	const iterations = 50

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			for it := 0; it < iterations; it++ {
				idx := (gid + it) % len(queries)
				got := m.Encode(queries[idx])
				exp := expected[idx]
				for k := range got {
					if got[k] != exp[k] {
						t.Errorf("concurrent mismatch for %q at [%d]: got %f, want %f", queries[idx], k, got[k], exp[k])
						return
					}
				}
			}
		}(g)
	}
	wg.Wait()
}

func isZeroVector(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
