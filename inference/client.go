package inference

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"
)

// HTTP 推理相关参数集中放在文件开头，方便后续根据部署环境调整。
const (
	DefaultPredictURL  = "http://127.0.0.1:8000/predict"
	DefaultHTTPTimeout = 30 * time.Second
)

var (
	ErrEmptyBatch = errors.New("inference: empty batch")
)

// HTTPClient 按 docs/inference_protocol.md 中的二进制协议调用 Python 服务。
type HTTPClient struct {
	url    string
	client *http.Client
}

func NewHTTPClient(url string, timeout time.Duration) (*HTTPClient, error) {
	if url == "" {
		return nil, errors.New("inference: predict URL is empty")
	}
	if timeout <= 0 {
		return nil, errors.New("inference: HTTP timeout must be positive")
	}

	return &HTTPClient{
		url: url,
		client: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

func (c *HTTPClient) Predict(ctx context.Context, features []Features) ([]Evaluation, error) {
	if len(features) == 0 {
		return nil, ErrEmptyBatch
	}

	// 1. 将 [B, 9, 19, 19] 的 float32 连续编码成 little-endian 字节流。
	payload := encodeFeatures(features)

	// 2. 构造请求，batch 大小通过 Header 传递给 Python 服务。
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("inference: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("X-Batch-Size", strconv.Itoa(len(features)))

	// 3. 发起请求并严格检查状态码与响应长度。
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("inference: request predict: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		if readErr != nil {
			return nil, fmt.Errorf("inference: predict status %d; read error body: %w", resp.StatusCode, readErr)
		}
		return nil, fmt.Errorf("inference: predict status %d: %s", resp.StatusCode, string(body))
	}

	expectedBytes := len(features) * OutputSize * 4
	body, err := io.ReadAll(io.LimitReader(resp.Body, int64(expectedBytes+1)))
	if err != nil {
		return nil, fmt.Errorf("inference: read response: %w", err)
	}
	if len(body) != expectedBytes {
		return nil, fmt.Errorf(
			"inference: invalid response size: got=%d expected=%d",
			len(body),
			expectedBytes,
		)
	}

	// 4. 按 head-major 布局拆分为每个样本自己的 Evaluation。
	return decodeEvaluations(body, len(features)), nil
}

func encodeFeatures(batch []Features) []byte {
	payload := make([]byte, len(batch)*FeatureSize*4)
	offset := 0

	for i := range batch {
		for _, value := range batch[i] {
			binary.LittleEndian.PutUint32(payload[offset:offset+4], math.Float32bits(value))
			offset += 4
		}
	}

	return payload
}

func decodeEvaluations(body []byte, batchSize int) []Evaluation {
	// 1. 响应顺序是全部 policy、全部 value、全部 score、全部 ownership。
	policyFloatOffset := 0
	valueFloatOffset := batchSize * PolicySize
	scoreFloatOffset := valueFloatOffset + batchSize
	ownershipFloatOffset := scoreFloatOffset + batchSize

	results := make([]Evaluation, batchSize)
	for sample := 0; sample < batchSize; sample++ {
		// 1.1 读取当前样本的 policy。
		for i := 0; i < PolicySize; i++ {
			results[sample].Policy[i] = float32At(body, policyFloatOffset+sample*PolicySize+i)
		}

		// 1.2 读取当前行动方视角的 value 和 score。
		results[sample].Value = float32At(body, valueFloatOffset+sample)
		results[sample].Score = float32At(body, scoreFloatOffset+sample)

		// 1.3 ownership 始终使用固定的黑白通道顺序。
		for i := 0; i < OwnershipSize; i++ {
			results[sample].Ownership[i] = float32At(
				body,
				ownershipFloatOffset+sample*OwnershipSize+i,
			)
		}
	}

	return results
}

func float32At(data []byte, floatIndex int) float32 {
	byteIndex := floatIndex * 4
	return math.Float32frombits(binary.LittleEndian.Uint32(data[byteIndex : byteIndex+4]))
}
