# kurs

[![Go Reference](https://pkg.go.dev/badge/github.com/branchgrove/kurs.svg)](https://pkg.go.dev/github.com/branchgrove/kurs)

> ***kurs*** — Swedish for *vector* (as in aviation: heading or course).

`kurs` runs [Model2Vec](https://github.com/MinishLab/model2vec) static embedding models (such as `minishlab/potion-base-8M`) in pure Go.

## Installation

```bash
go get github.com/branchgrove/kurs
```

## Quickstart

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/branchgrove/kurs"
)

func main() {
	modelBytes, _ := os.ReadFile("model/model.safetensors")
	tokBytes, _ := os.ReadFile("model/tokenizer.json")
	cfgBytes, _ := os.ReadFile("model/config.json")

	m, err := kurs.New(modelBytes, tokBytes, cfgBytes)
	if err != nil {
		log.Fatal(err)
	}

	a := m.Encode("How do I reset my password?")
	b := m.Encode("Steps to recover your account credentials")

	fmt.Printf("Similarity: %.3f\n", kurs.Dot(a, b)) // Or kurs.Cosine; Dot is faster for normalized vectors
}
```

> **Tip**: Since embeddings are normalized by default, `kurs.Dot` produces the exact same score as `kurs.Cosine` while running significantly faster. Use `kurs.Cosine` only if vectors may be unnormalized.

Models can also be bundled into single-binary distributions using Go's `//go:embed`.

## Document Splitting & Truncation

Inputs default to a 512-token limit (or the value in `config.json`). `kurs` does not split text automatically; users must chunk longer documents themselves before encoding. Splitting is recommended to avoid truncation and semantic dilution.

## Model Compatibility

`kurs` supports any static embedding model distributed in Hugging Face format that satisfies:
- **Weights**: `model.safetensors` containing an `embeddings` or `embedding` 2-D tensor (`F32`, `F16`, or `BF16`).
- **Tokenizer**: `tokenizer.json` configured for BERT-style `WordPiece` (`BertNormalizer` + `BertPreTokenizer`).

BPE, Byte-Level BPE, SentencePiece, and legacy formats (`.vec`, `.bin`) are not supported.

## Downloading a Model

Download `model.safetensors`, `tokenizer.json`, and `config.json` from any WordPiece-based Model2Vec HuggingFace repo (e.g. [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M)):

```bash
mkdir -p model
curl -sSL -o model/model.safetensors https://huggingface.co/minishlab/potion-base-8M/resolve/main/model.safetensors
curl -sSL -o model/tokenizer.json https://huggingface.co/minishlab/potion-base-8M/resolve/main/tokenizer.json
curl -sSL -o model/config.json https://huggingface.co/minishlab/potion-base-8M/resolve/main/config.json
```

## Running Tests

Generate the reference test data:

```bash
uv run --with model2vec --with tokenizers python scripts/reference.py
```

Then run the tests:

```bash
go test ./...
```
