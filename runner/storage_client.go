package runner

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"

	"go-ai-infer/board"
	"go-ai-infer/inference"
	"go-ai-infer/selfplay"
)

const (
	DefaultStorageURL     = "http://127.0.0.1:8000/selfplay/game"
	DefaultStorageTimeout = 30 * time.Second
	BytesPerSample        = inference.FeatureSize*4 +
		inference.PolicySize*4 +
		4 +
		4 +
		board.Points
)

type HTTPStorageClient struct {
	url    string
	client *http.Client
}

func NewHTTPStorageClient(url string, timeout time.Duration) (*HTTPStorageClient, error) {
	if url == "" {
		return nil, errors.New("runner: storage URL is empty")
	}
	if timeout <= 0 {
		return nil, errors.New("runner: storage timeout must be positive")
	}
	return &HTTPStorageClient{
		url: url,
		client: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

type storageResponse struct {
	Samples int    `json:"samples"`
	Status  string `json:"status"`
}

func (c *HTTPStorageClient) SaveGame(ctx context.Context, game *selfplay.Game) error {
	if ctx == nil {
		return errors.New("runner: storage context is nil")
	}
	if game == nil {
		return errors.New("runner: game is nil")
	}
	n := len(game.Samples)
	if n <= 0 {
		return fmt.Errorf("runner: invalid sample count %d", n)
	}

	payload := encodeGame(game)
	expectedBytes := n * BytesPerSample
	if len(payload) != expectedBytes {
		return fmt.Errorf("runner: encoded game size=%d expected=%d", len(payload), expectedBytes)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("runner: create storage request: %w", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("X-Sample-Count", strconv.Itoa(n))

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("runner: request storage: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
	if err != nil {
		return fmt.Errorf("runner: read storage response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("runner: storage status %d: %s", resp.StatusCode, string(body))
	}

	var result storageResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("runner: decode storage response: %w", err)
	}
	if result.Status != "written" {
		return fmt.Errorf("runner: unexpected storage status %q", result.Status)
	}
	if result.Samples != n {
		return fmt.Errorf("runner: storage samples=%d expected=%d", result.Samples, n)
	}
	return nil
}

func encodeGame(game *selfplay.Game) []byte {
	n := len(game.Samples)
	payload := make([]byte, n*BytesPerSample)
	offset := 0

	writeFloat32 := func(value float32) {
		binary.LittleEndian.PutUint32(payload[offset:offset+4], math.Float32bits(value))
		offset += 4
	}

	for i := range game.Samples {
		for _, value := range game.Samples[i].Features {
			writeFloat32(value)
		}
	}
	for i := range game.Samples {
		for _, value := range game.Samples[i].Policy {
			writeFloat32(value)
		}
	}
	for i := range game.Samples {
		writeFloat32(game.Samples[i].Value)
	}
	for i := range game.Samples {
		writeFloat32(game.Samples[i].Score)
	}
	for i := range game.Samples {
		for _, value := range game.Samples[i].Ownership {
			payload[offset] = byte(value)
			offset++
		}
	}
	return payload
}
