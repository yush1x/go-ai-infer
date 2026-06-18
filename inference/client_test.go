package inference

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestHTTPClientPredict(t *testing.T) {
	const batchSize = 2

	client, err := NewHTTPClient("http://python.test/predict", time.Second)
	if err != nil {
		t.Fatalf("NewHTTPClient() error = %v", err)
	}

	client.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		// 1. 验证 Go 发出的 Header 和连续 float32 请求体。
		if got := req.Header.Get("X-Batch-Size"); got != strconv.Itoa(batchSize) {
			t.Errorf("X-Batch-Size = %q, want %d", got, batchSize)
		}

		body, readErr := io.ReadAll(req.Body)
		if readErr != nil {
			t.Fatalf("read request body: %v", readErr)
		}
		if got, want := len(body), batchSize*FeatureSize*4; got != want {
			t.Fatalf("request bytes = %d, want %d", got, want)
		}
		if got := float32At(body, 0); got != 11 {
			t.Errorf("first feature = %v, want 11", got)
		}
		if got := float32At(body, FeatureSize); got != 22 {
			t.Errorf("second sample feature = %v, want 22", got)
		}

		// 2. 构造协议规定的 head-major 响应。
		response := makeHeadMajorResponse(batchSize)
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(bytes.NewReader(response)),
			Header:     make(http.Header),
		}, nil
	})

	features := make([]Features, batchSize)
	features[0][0] = 11
	features[1][0] = 22

	results, err := client.Predict(context.Background(), features)
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	// 3. 验证不同输出头正确归入各自样本。
	if got := results[1].Policy[3]; got != 1003 {
		t.Errorf("sample 1 policy[3] = %v, want 1003", got)
	}
	if got := results[0].Value; got != 2000 {
		t.Errorf("sample 0 value = %v, want 2000", got)
	}
	if got := results[1].Score; got != 3001 {
		t.Errorf("sample 1 score = %v, want 3001", got)
	}
	if got := results[1].Ownership[5]; got != 5005 {
		t.Errorf("sample 1 ownership[5] = %v, want 5005", got)
	}
}

func TestHTTPClientRejectsInvalidResponseSize(t *testing.T) {
	client, err := NewHTTPClient("http://python.test/predict", time.Second)
	if err != nil {
		t.Fatalf("NewHTTPClient() error = %v", err)
	}

	client.client.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader("short")),
			Header:     make(http.Header),
		}, nil
	})

	_, err = client.Predict(context.Background(), []Features{{}})
	if err == nil || !strings.Contains(err.Error(), "invalid response size") {
		t.Fatalf("Predict() error = %v, want invalid response size", err)
	}
}

func makeHeadMajorResponse(batchSize int) []byte {
	values := make([]float32, batchSize*OutputSize)
	valueOffset := batchSize * PolicySize
	scoreOffset := valueOffset + batchSize
	ownershipOffset := scoreOffset + batchSize

	for sample := 0; sample < batchSize; sample++ {
		for i := 0; i < PolicySize; i++ {
			values[sample*PolicySize+i] = float32(sample*1000 + i)
		}
		values[valueOffset+sample] = float32(2000 + sample)
		values[scoreOffset+sample] = float32(3000 + sample)
		for i := 0; i < OwnershipSize; i++ {
			values[ownershipOffset+sample*OwnershipSize+i] = float32(4000 + sample*1000 + i)
		}
	}

	body := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(body[i*4:(i+1)*4], math.Float32bits(value))
	}
	return body
}
