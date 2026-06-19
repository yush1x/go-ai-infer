package runner

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"math"
	"net/http"
	"strconv"
	"testing"
	"time"

	"go-ai-infer/board"
	"go-ai-infer/inference"
	"go-ai-infer/selfplay"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestHTTPStorageClientEncodesProtocol(t *testing.T) {
	game := testGame(2)
	client, err := NewHTTPStorageClient("http://storage.test/selfplay/game", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	client.client.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost {
			t.Errorf("method=%s, want POST", r.Method)
		}
		if r.Header.Get("Content-Type") != "application/octet-stream" {
			t.Errorf("unexpected content type")
		}
		if r.Header.Get("X-Sample-Count") != "2" {
			t.Errorf("sample count=%q, want 2", r.Header.Get("X-Sample-Count"))
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		if len(body) != 2*BytesPerSample {
			t.Fatalf("body bytes=%d, want %d", len(body), 2*BytesPerSample)
		}

		gotFeature := math.Float32frombits(binary.LittleEndian.Uint32(body[:4]))
		if gotFeature != 10 {
			t.Errorf("first feature=%v, want 10", gotFeature)
		}
		policyOffset := 2 * inference.FeatureSize * 4
		gotPolicy := math.Float32frombits(binary.LittleEndian.Uint32(body[policyOffset : policyOffset+4]))
		if gotPolicy != 20 {
			t.Errorf("first policy=%v, want 20", gotPolicy)
		}
		valueOffset := policyOffset + 2*inference.PolicySize*4
		gotValue := math.Float32frombits(binary.LittleEndian.Uint32(body[valueOffset : valueOffset+4]))
		if gotValue != 1 {
			t.Errorf("first value=%v, want 1", gotValue)
		}
		ownershipOffset := valueOffset + 2*4 + 2*4
		if got := int8(body[ownershipOffset]); got != -1 {
			t.Errorf("first ownership=%d, want -1", got)
		}

		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(bytes.NewBufferString(`{"samples":2,"status":"written"}`)),
		}, nil
	})
	if err := client.SaveGame(context.Background(), game); err != nil {
		t.Fatal(err)
	}
}

func TestHTTPStorageClientRejectsFailure(t *testing.T) {
	client, err := NewHTTPStorageClient("http://storage.test/selfplay/game", time.Second)
	if err != nil {
		t.Fatal(err)
	}
	client.client.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusInternalServerError,
			Header:     make(http.Header),
			Body:       io.NopCloser(bytes.NewBufferString("save failed")),
		}, nil
	})
	if err := client.SaveGame(context.Background(), testGame(1)); err == nil {
		t.Fatal("expected storage error")
	}
}

func testGame(n int) *selfplay.Game {
	samples := make([]selfplay.Sample, n)
	for i := range samples {
		samples[i].Features[0] = float32(10 + i)
		samples[i].Policy[0] = float32(20 + i)
		samples[i].Value = 1
		samples[i].Score = float32(3 + i)
		for p := range samples[i].Ownership {
			samples[i].Ownership[p] = board.OwnershipUnknown
		}
	}
	return &selfplay.Game{
		Samples:    samples,
		TotalMoves: n,
		Actions:    make([]int, n),
		BlackLead:  3.5,
		Winner:     board.Black,
		Final:      board.FinalResult{},
	}
}

func TestBytesPerSampleMatchesProtocol(t *testing.T) {
	if BytesPerSample != 14813 {
		t.Fatalf("BytesPerSample=%s, want 14813", strconv.Itoa(BytesPerSample))
	}
}
