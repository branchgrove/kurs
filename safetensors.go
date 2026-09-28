package kurs

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"unsafe"
)

// Minimal safetensors reader: 8-byte little-endian header length, a JSON
// header describing each tensor (dtype, shape, byte offsets), then raw data.

type tensorInfo struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

type safetensors struct {
	tensors map[string]tensorInfo
	data    []byte // everything after the header
}

var littleEndian = binary.NativeEndian.Uint16([]byte{1, 0}) == 1

func parseSafetensors(b []byte) (*safetensors, error) {
	if len(b) < 8 {
		return nil, fmt.Errorf("safetensors: file too short")
	}
	n := binary.LittleEndian.Uint64(b[:8])
	if n == 0 || n > 100*1024*1024 || n > uint64(len(b)-8) {
		return nil, fmt.Errorf("safetensors: bad header length %d", n)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &raw); err != nil {
		return nil, fmt.Errorf("safetensors: header: %w", err)
	}
	st := &safetensors{tensors: map[string]tensorInfo{}, data: b[8+n:]}
	for name, msg := range raw {
		if name == "__metadata__" {
			continue
		}
		var t tensorInfo
		if err := json.Unmarshal(msg, &t); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		if t.Offsets[0] < 0 || t.Offsets[1] < t.Offsets[0] || t.Offsets[1] > len(st.data) {
			return nil, fmt.Errorf("safetensors: tensor %q has bad offsets", name)
		}
		st.tensors[name] = t
	}
	return st, nil
}

func (s *safetensors) has(name string) bool {
	_, ok := s.tensors[name]
	return ok
}

func (s *safetensors) names() []string {
	out := make([]string, 0, len(s.tensors))
	for k := range s.tensors {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func (s *safetensors) raw(name string) (tensorInfo, []byte, error) {
	t, ok := s.tensors[name]
	if !ok {
		return t, nil, fmt.Errorf("safetensors: no tensor %q (have %v)", name, s.names())
	}
	return t, s.data[t.Offsets[0]:t.Offsets[1]], nil
}

func tensorElementCount(name string, shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("safetensors: tensor %q has empty shape", name)
	}
	total := 1
	for _, dim := range shape {
		if dim <= 0 {
			return 0, fmt.Errorf("safetensors: tensor %q has invalid dimension %d", name, dim)
		}
		if dim > math.MaxInt/total {
			return 0, fmt.Errorf("safetensors: tensor %q shape %v exceeds integer capacity", name, shape)
		}
		total *= dim
	}
	return total, nil
}

// float32s returns the tensor as float32. F32 data is used zero-copy when the
// bytes are 4-byte aligned on a little-endian host (so the OS can page an
// embedded or mmapped model in lazily); F16/BF16 are converted.
func (s *safetensors) float32s(name string) (tensorInfo, []float32, error) {
	t, raw, err := s.raw(name)
	if err != nil {
		return t, nil, err
	}
	exp, err := tensorElementCount(name, t.Shape)
	if err != nil {
		return t, nil, err
	}
	switch t.DType {
	case "F32":
		if len(raw)%4 != 0 {
			return t, nil, fmt.Errorf("safetensors: tensor %q length %d is not a multiple of 4", name, len(raw))
		}
		n := len(raw) / 4
		if n != exp {
			return t, nil, fmt.Errorf("safetensors: tensor %q data length (%d floats) does not match shape %v (%d floats)", name, n, t.Shape, exp)
		}
		if n == 0 {
			return t, nil, nil
		}
		if littleEndian && uintptr(unsafe.Pointer(&raw[0]))%4 == 0 {
			return t, unsafe.Slice((*float32)(unsafe.Pointer(&raw[0])), n), nil
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return t, out, nil
	case "F16":
		if len(raw)%2 != 0 {
			return t, nil, fmt.Errorf("safetensors: tensor %q length %d is not a multiple of 2", name, len(raw))
		}
		n := len(raw) / 2
		if n != exp {
			return t, nil, fmt.Errorf("safetensors: tensor %q data length (%d floats) does not match shape %v (%d floats)", name, n, t.Shape, exp)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = f16to32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return t, out, nil
	case "BF16":
		if len(raw)%2 != 0 {
			return t, nil, fmt.Errorf("safetensors: tensor %q length %d is not a multiple of 2", name, len(raw))
		}
		n := len(raw) / 2
		if n != exp {
			return t, nil, fmt.Errorf("safetensors: tensor %q data length (%d floats) does not match shape %v (%d floats)", name, n, t.Shape, exp)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return t, out, nil
	}
	return t, nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %s", name, t.DType)
}

func (s *safetensors) int32s(name string) ([]int32, error) {
	t, raw, err := s.raw(name)
	if err != nil {
		return nil, err
	}
	exp, err := tensorElementCount(name, t.Shape)
	if err != nil {
		return nil, err
	}
	switch t.DType {
	case "I32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("safetensors: tensor %q length %d is not a multiple of 4", name, len(raw))
		}
		n := len(raw) / 4
		if n != exp {
			return nil, fmt.Errorf("safetensors: tensor %q data length (%d ints) does not match shape %v (%d ints)", name, n, t.Shape, exp)
		}
		out := make([]int32, n)
		for i := range out {
			out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case "I64":
		if len(raw)%8 != 0 {
			return nil, fmt.Errorf("safetensors: tensor %q length %d is not a multiple of 8", name, len(raw))
		}
		n := len(raw) / 8
		if n != exp {
			return nil, fmt.Errorf("safetensors: tensor %q data length (%d ints) does not match shape %v (%d ints)", name, n, t.Shape, exp)
		}
		out := make([]int32, n)
		for i := range out {
			val := int64(binary.LittleEndian.Uint64(raw[i*8:]))
			if val > math.MaxInt32 || val < math.MinInt32 {
				return nil, fmt.Errorf("safetensors: tensor %q value %d overflows int32", name, val)
			}
			out[i] = int32(val)
		}
		return out, nil
	}
	return nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %s", name, t.DType)
}

func f16to32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0:
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		e := uint32(127 - 15 + 1) // subnormal: renormalize
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3ff
		return math.Float32frombits(sign<<31 | e<<23 | frac<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign<<31 | 0xff<<23 | frac<<13)
	}
	return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | frac<<13)
}
