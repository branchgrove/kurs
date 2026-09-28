package kurs

import (
	"os"
	"path/filepath"
	"testing"
)

func loadBenchmarkModel(b *testing.B) *Model {
	b.Helper()
	read := func(p string) []byte {
		data, err := os.ReadFile(p)
		if err != nil {
			data, err = os.ReadFile(filepath.Join("..", p))
		}
		if err != nil {
			b.Fatalf("failed to load model file %s: %v", p, err)
		}
		return data
	}
	m, err := New(read("model/model.safetensors"), read("model/tokenizer.json"), read("model/config.json"))
	if err != nil {
		b.Fatal(err)
	}
	return m
}

func loadBenchmarkTokenizer(b *testing.B) *tokenizer {
	b.Helper()
	data, err := os.ReadFile("model/tokenizer.json")
	if err != nil {
		data, err = os.ReadFile(filepath.Join("..", "model", "tokenizer.json"))
	}
	if err != nil {
		b.Fatalf("failed to load tokenizer: %v", err)
	}
	tok, err := newTokenizer(data)
	if err != nil {
		b.Fatal(err)
	}
	return tok
}

func BenchmarkEncode_Short(b *testing.B) {
	m := loadBenchmarkModel(b)
	query := "How do I reset my password?"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.Encode(query)
	}
}

func BenchmarkEncode_Paragraph(b *testing.B) {
	m := loadBenchmarkModel(b)
	text := "The quick brown fox jumps over the lazy dog repeatedly and continuously forever. " +
		"Natural language processing with static embeddings provides high performance with zero dependencies."
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.Encode(text)
	}
}

func BenchmarkEncodeBatch_4(b *testing.B) {
	m := loadBenchmarkModel(b)
	queries := []string{
		"How do I reset my password?",
		"Steps to recover your account credentials",
		"Can I use this offline in pure Go?",
		"Zero-copy embeddings with safetensors",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.EncodeBatch(queries)
	}
}

func BenchmarkEncodeBatch_10(b *testing.B) {
	m := loadBenchmarkModel(b)
	queries := []string{
		"How do I reset my password?",
		"Steps to recover your account credentials",
		"Can I use this offline in pure Go?",
		"Zero-copy embeddings with safetensors",
		"BERT wordpiece tokenizer benchmarking",
		"Natural language processing in Go",
		"Fast similarity search with cosine distance",
		"Model2vec potion base 8M inference",
		"Multithreaded batch encoding performance",
		"Handling edge cases like empty strings and accents",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.EncodeBatch(queries)
	}
}

func BenchmarkEncodeBatch_100(b *testing.B) {
	m := loadBenchmarkModel(b)
	queries := make([]string, 100)
	base := []string{
		"How do I reset my password?",
		"Steps to recover your account credentials",
		"Can I use this offline in pure Go?",
		"Zero-copy embeddings with safetensors",
		"BERT wordpiece tokenizer benchmarking",
	}
	for i := range queries {
		queries[i] = base[i%len(base)]
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = m.EncodeBatch(queries)
	}
}

func BenchmarkTokenizer_Normalize_ASCIILower(b *testing.B) {
	tok := loadBenchmarkTokenizer(b)
	text := "how do i reset my password with simple lowercase ascii"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.normalize(text)
	}
}

func BenchmarkTokenizer_Normalize_ASCIIWhitespace(b *testing.B) {
	tok := loadBenchmarkTokenizer(b)
	text := "how do i reset\tmy password\nwith tabs and newlines\r\nand spaces"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.normalize(text)
	}
}

func BenchmarkTokenizer_Normalize_Accents(b *testing.B) {
	tok := loadBenchmarkTokenizer(b)
	text := "Café crème brûlée with façade and naïve résumé in München"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.normalize(text)
	}
}

func BenchmarkTokenizer_Normalize_CJK(b *testing.B) {
	tok := loadBenchmarkTokenizer(b)
	text := "自然言語処理と機械学習の最新動向 人工智能和深度学习"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.normalize(text)
	}
}

func BenchmarkTokenizer_Encode(b *testing.B) {
	tok := loadBenchmarkTokenizer(b)
	text := "How do I reset my password? Steps to recover your account credentials."
	out := make([]int32, 0, 32)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.encode(text, out[:0])
	}
}

func BenchmarkTokenizer_PunctuationHeavy(b *testing.B) {
	tok := loadBenchmarkTokenizer(b)
	text := "hello! (world) [foo] {bar} <baz>: test; 'quote' \"double\" - dash / slash"
	out := make([]int32, 0, 32)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.encode(text, out[:0])
	}
}

func BenchmarkWordPiece_LongWord(b *testing.B) {
	tok := loadBenchmarkTokenizer(b)
	word := "antidisestablishmentarianism"
	out := make([]int32, 0, 16)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.wordpiece(word, out[:0])
	}
}

func BenchmarkCosine(b *testing.B) {
	m := loadBenchmarkModel(b)
	v1 := m.Encode("How do I reset my password?")
	v2 := m.Encode("Steps to recover your account credentials")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Cosine(v1, v2)
	}
}
