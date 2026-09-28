// Package kurs is an implementation of Model2Vec (https://github.com/MinishLab/model2vec)
// in Go.
//
// A [*Model] is safe for concurrent use by multiple goroutines.
//
// # Memory
//
// On little-endian architectures with 4-byte aligned data, F32 embedding tensors
// are used zero-copy directly from the input byte slice. Callers must not mutate
// the provided model bytes while the Model is in use.
//
// # Loading a Model
//
// Models are instantiated using [New] with raw bytes from model.safetensors,
// tokenizer.json, and (optionally) config.json. Models must use SafeTensors
// ("embeddings" tensor in F32/F16/BF16) and a WordPiece tokenizer.json
// (BertNormalizer + BertPreTokenizer). Other tokenizers (BPE, SentencePiece)
// and legacy formats (.vec, .bin) are not supported.
//
//	//go:embed model/model.safetensors
//	var modelBytes []byte
//
//	//go:embed model/tokenizer.json
//	var tokenizerBytes []byte
//
//	//go:embed model/config.json
//	var configBytes []byte
//
//	model, err := kurs.New(modelBytes, tokenizerBytes, configBytes)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
// # Embedding Text
//
// Use [Model.Encode] to embed a single string:
//
//	vec := model.Encode("How do I reset my password?")
//
// Text with no known vocabulary tokens yields an all-zero vector.
//
// Multiple texts can be encoded with [Model.EncodeBatch]:
//
//	batch := model.EncodeBatch([]string{"first document", "second document"})
//
// # Truncation and Options
//
// Inputs default to a 512-token limit (or the value in config.json).
// The package does not split text; chunking longer documents prior to
// encoding is recommended and must be done by the caller.
//
// [WithMaxTokens] sets the token sequence truncation limit prior to pooling (default 512):
//
//	model, err := kurs.New(modelBytes, tokenizerBytes, configBytes, kurs.WithMaxTokens(128))
//
// [WithNormalize] overrides whether output vectors are L2-normalized (defaults to true or config.json setting):
//
//	model, err := kurs.New(modelBytes, tokenizerBytes, configBytes, kurs.WithNormalize(false))
//
// # Similarity: Dot vs. Cosine
//
// For normalized embeddings (the default), prefer [Dot]. It equals [Cosine] but
// is significantly faster by omitting norm and square root calculations:
//
//	score := kurs.Dot(vecA, vecB)
//
// Use [Cosine] when vectors may be unnormalized (e.g. via [WithNormalize](false)
// or from external sources), ensuring values strictly in [-1.0, 1.0]:
//
//	score := kurs.Cosine(vecA, vecB)
//
// Higher values indicate greater semantic similarity:
//   - 1.0 represents identical vector directions (maximum similarity / exact match).
//   - High positive scores (e.g. 0.7 to 0.9+) indicate strong semantic relatedness.
//   - 0.0 indicates orthogonal/uncorrelated vectors (or empty / zero vectors).
//   - Negative values (down to -1.0) indicate opposing directions (rare in dense text embeddings).
package kurs
