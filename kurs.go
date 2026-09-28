package kurs

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Model is safe for concurrent use by multiple goroutines.
type Model struct {
	maxTokens int
	emb       []float32 // rows*dim, row-major
	rows, dim int
	weights   []float32 // optional per-token weights (indexed by token id)
	mapping   []int32   // optional token id -> row remapping (pruned vocabs)
	normalize bool
	tok       *tokenizer
}

// Option configures a Model during instantiation.
type Option func(*Model)

// WithMaxTokens sets the maximum token sequence truncation limit prior to pooling
// (default 512, matching the Python reference library). Values <= 0 disable truncation.
func WithMaxTokens(n int) Option {
	return func(m *Model) {
		m.maxTokens = n
	}
}

// WithNormalize configures whether output embeddings are L2-normalized.
// Defaults to true (or the value specified in config.json).
func WithNormalize(normalize bool) Option {
	return func(m *Model) {
		m.normalize = normalize
	}
}

// MaxTokens returns the configured maximum token limit (default 512).
func (m *Model) MaxTokens() int { return m.maxTokens }

// New builds a Model from the contents of model.safetensors, tokenizer.json
// and (optionally, may be nil) config.json. Optional configuration can be passed
// using [Option] helpers like [WithMaxTokens].
//
// Memory Note: On little-endian hosts, F32 tensors are referenced directly from
// safetensorsData without copying (zero-copy). Callers must not mutate or reuse
// the safetensorsData slice while Model is in use.
func New(safetensorsData, tokenizerData, configData []byte, opts ...Option) (*Model, error) {
	st, err := parseSafetensors(safetensorsData)
	if err != nil {
		return nil, err
	}
	tok, err := newTokenizer(tokenizerData)
	if err != nil {
		return nil, err
	}
	name := "embeddings"
	if !st.has(name) {
		if st.has("embedding") {
			name = "embedding"
		} else {
			return nil, fmt.Errorf("model2vec: no embeddings tensor; tensors: %v", st.names())
		}
	}
	info, emb, err := st.float32s(name)
	if err != nil {
		return nil, err
	}
	if len(info.Shape) != 2 {
		return nil, fmt.Errorf("model2vec: embeddings must be 2-D, got shape %v", info.Shape)
	}
	if info.Shape[0] <= 0 || info.Shape[1] <= 0 {
		return nil, fmt.Errorf("model2vec: embedding dimensions must be positive, got shape %v", info.Shape)
	}
	m := &Model{
		maxTokens: 512,
		emb:       emb,
		rows:      info.Shape[0],
		dim:       info.Shape[1],
		normalize: true,
		tok:       tok,
	}
	if st.has("weights") {
		var winfo tensorInfo
		if winfo, m.weights, err = st.float32s("weights"); err != nil {
			return nil, err
		}
		if len(winfo.Shape) != 1 {
			return nil, fmt.Errorf("model2vec: weights must be 1-D, got shape %v", winfo.Shape)
		}
	}
	if st.has("mapping") {
		minfo, _, rerr := st.raw("mapping")
		if rerr != nil {
			return nil, rerr
		}
		if len(minfo.Shape) != 1 {
			return nil, fmt.Errorf("model2vec: mapping must be 1-D, got shape %v", minfo.Shape)
		}
		if m.mapping, err = st.int32s("mapping"); err != nil {
			return nil, err
		}
	}
	if len(configData) > 0 {
		var cfg struct {
			Normalize    *bool `json:"normalize"`
			SeqLength    *int  `json:"seq_length"`
			MaxSeqLength *int  `json:"max_seq_length"`
			MaxTokens    *int  `json:"max_tokens"`
		}
		if err := json.Unmarshal(configData, &cfg); err != nil {
			return nil, fmt.Errorf("config.json: %w", err)
		}
		if cfg.Normalize != nil {
			m.normalize = *cfg.Normalize
		}
		if cfg.MaxTokens != nil {
			m.maxTokens = *cfg.MaxTokens
		} else if cfg.MaxSeqLength != nil {
			m.maxTokens = *cfg.MaxSeqLength
		} else if cfg.SeqLength != nil {
			m.maxTokens = *cfg.SeqLength
		}
	}
	for _, opt := range opts {
		opt(m)
	}
	return m, nil
}

// Dim returns the embedding dimensionality.
func (m *Model) Dim() int { return m.dim }

// Encode embeds one piece of text. Text with no known tokens yields a zero vector.
func (m *Model) Encode(text string) []float32 {
	out := make([]float32, m.dim)
	m.encodeInto(text, out)
	return out
}

func (m *Model) encodeInto(text string, out []float32) {
	var stackIDs [32]int32
	ids := m.tok.encode(text, stackIDs[:0])
	if m.maxTokens > 0 && len(ids) > m.maxTokens {
		ids = ids[:m.maxTokens]
	}

	n := 0
	var totalWeight float32
	for _, id := range ids {
		if id < 0 || id == m.tok.unk { // the Python lib drops unk tokens too
			continue
		}
		row := int(id)
		if m.mapping != nil {
			if row >= len(m.mapping) {
				continue
			}
			row = int(m.mapping[row])
		}
		if row < 0 || row >= m.rows {
			continue
		}
		w := float32(1)
		if m.weights != nil && int(id) < len(m.weights) {
			w = m.weights[id]
		}
		vec := m.emb[row*m.dim : (row+1)*m.dim]
		if w == 1 {
			j := 0
			for ; j+4 <= len(vec); j += 4 {
				out[j] += vec[j]
				out[j+1] += vec[j+1]
				out[j+2] += vec[j+2]
				out[j+3] += vec[j+3]
			}
			for ; j < len(vec); j++ {
				out[j] += vec[j]
			}
		} else {
			j := 0
			for ; j+4 <= len(vec); j += 4 {
				out[j] += w * vec[j]
				out[j+1] += w * vec[j+1]
				out[j+2] += w * vec[j+2]
				out[j+3] += w * vec[j+3]
			}
			for ; j < len(vec); j++ {
				out[j] += w * vec[j]
			}
		}
		n++
		totalWeight += w
	}
	if n == 0 {
		return
	}

	inv := float32(0)
	if m.weights != nil && totalWeight > 0 {
		inv = 1 / totalWeight
	} else {
		inv = 1 / float32(n)
	}
	var sumsq float64
	for i := range out {
		out[i] *= inv
		sumsq += float64(out[i]) * float64(out[i])
	}
	if m.normalize && sumsq > 0 {
		s := float32(1 / math.Sqrt(sumsq))
		for i := range out {
			out[i] *= s
		}
	}
}

// EncodeBatch embeds several texts concurrently across available CPU cores.
func (m *Model) EncodeBatch(texts []string) [][]float32 {
	n := len(texts)
	out := make([][]float32, n)
	if n == 0 {
		return out
	}

	arena := make([]float32, n*m.dim)
	for i := 0; i < n; i++ {
		out[i] = arena[i*m.dim : (i+1)*m.dim]
	}

	workers := runtime.GOMAXPROCS(0)
	if workers <= 1 || n <= 4 {
		for i := 0; i < n; i++ {
			m.encodeInto(texts[i], out[i])
		}
		return out
	}

	if workers > n {
		workers = n
	}
	chunkSize := (n + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunkSize
		if start >= n {
			break
		}
		end := start + chunkSize
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				m.encodeInto(texts[i], out[i])
			}
		}(start, end)
	}
	wg.Wait()
	return out
}

// Cosine returns the cosine similarity of a and b in [-1.0, 1.0].
//
// For normalized embeddings, this equals [Dot]. Prefer [Dot] for better performance
// when embeddings are known to be normalized. Use Cosine when vectors may not be
// unit-length (e.g. if unnormalized or from external sources).
//
// The score represents semantic similarity:
//   - 1.0: Identical orientation (maximum similarity / exact match).
//   - High positive values (e.g. 0.7 to 0.9+): Strong semantic relatedness.
//   - 0.0: Orthogonal / uncorrelated vectors (no shared direction).
//   - Negative values (down to -1.0): Diametrically opposed directions.
//
// If vector lengths do not match or either vector has zero magnitude, 0 is returned.
func Cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		a0, a1, a2, a3 := float64(a[i]), float64(a[i+1]), float64(a[i+2]), float64(a[i+3])
		b0, b1, b2, b3 := float64(b[i]), float64(b[i+1]), float64(b[i+2]), float64(b[i+3])
		dot += a0*b0 + a1*b1 + a2*b2 + a3*b3
		na += a0*a0 + a1*a1 + a2*a2 + a3*a3
		nb += b0*b0 + b1*b1 + b2*b2 + b3*b3
	}
	for ; i < len(a); i++ {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		na += ai * ai
		nb += bi * bi
	}
	if na == 0 || nb == 0 {
		return 0
	}
	denom := math.Sqrt(na) * math.Sqrt(nb)
	if denom <= 0 || math.IsNaN(denom) || math.IsInf(denom, 0) {
		return 0
	}
	res := float32(dot / denom)
	if math.IsNaN(float64(res)) || math.IsInf(float64(res), 0) {
		return 0
	}
	if res > 1 {
		return 1
	} else if res < -1 {
		return -1
	}
	return res
}

// Dot returns the dot product of two float32 vectors.
//
// For normalized embeddings (the default), Dot equals Cosine and is significantly
// faster because it skips norm calculations. Use Dot by default unless embeddings
// are unnormalized. Returns 0 if lengths mismatch.
func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		sum += float64(a[i])*float64(b[i]) +
			float64(a[i+1])*float64(b[i+1]) +
			float64(a[i+2])*float64(b[i+2]) +
			float64(a[i+3])*float64(b[i+3])
	}
	for ; i < len(a); i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}
