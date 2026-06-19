package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go-ai-infer/inference"
	"go-ai-infer/mcts"
	"go-ai-infer/runner"
)

const (
	totalGames      = 100 // 本次运行需要生成的总棋局数
	gameConcurrency = 8   // 同时进行自博弈的棋局数

	numSimulations = 200           // 每一步 MCTS 搜索执行的模拟次数
	cPuct          = float32(1.5)  // MCTS 在利用和探索之间的平衡系数
	dirichletAlpha = float32(0.03) // 自博弈根节点 Dirichlet 噪声分布参数
	dirichletEps   = float32(0.25) // 自博弈根节点混入随机噪声的比例
	passBonus      = float32(0.0)  // 热门落点均为己方眼时给予 pass 的额外分数

	inferenceBatchSize = 32                   // Python 单次批量推理的最大局面数
	inferenceMaxWait   = 5 * time.Millisecond // 推理 batch 未满时的最大等待时间
	inferenceQueueSize = 128                  // 等待批量推理的最大请求数量
	enableBatchLog     = true                 // 是否输出每次推理 batch 的发送和完成日志

	predictURL = "http://127.0.0.1:8000/predict"       // Python 模型推理接口地址
	storageURL = "http://127.0.0.1:8000/selfplay/game" // Python 训练数据保存接口地址

	inferenceTimeout = 30 * time.Second // 单次 Python 推理请求的超时时间
	storageTimeout   = 30 * time.Second // 单盘训练数据保存请求的超时时间
)

func main() {
	if err := run(); err != nil {
		log.Printf("selfplay stopped: %v", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	predictor, err := inference.NewHTTPClient(predictURL, inferenceTimeout)
	if err != nil {
		return fmt.Errorf("create inference client: %w", err)
	}

	inference.EnableBatchLog = enableBatchLog
	batchConfig := inference.BatcherConfig{
		BatchSize: inferenceBatchSize,
		MaxWait:   inferenceMaxWait,
		QueueSize: inferenceQueueSize,
	}
	batcher, err := inference.NewBatcher(predictor, batchConfig)
	if err != nil {
		return fmt.Errorf("create inference batcher: %w", err)
	}
	defer func() {
		if closeErr := batcher.Close(); closeErr != nil {
			log.Printf("close inference batcher: %v", closeErr)
		}
	}()

	mctsConfig := mcts.Config{
		NumSimulations: numSimulations,
		CPuct:          cPuct,
		SelfPlay:       true,
		DirichletAlpha: dirichletAlpha,
		DirichletEps:   dirichletEps,
		PassBonus:      passBonus,
	}
	searcher := mcts.NewSearcher(batcher, mctsConfig)

	storage, err := runner.NewHTTPStorageClient(storageURL, storageTimeout)
	if err != nil {
		return fmt.Errorf("create storage client: %w", err)
	}

	runConfig := runner.Config{
		Games:       totalGames,
		Concurrency: gameConcurrency,
	}
	selfplayRunner, err := runner.New(searcher, storage, runConfig)
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	log.Printf(
		"start selfplay: games=%d concurrency=%d simulations=%d batch_size=%d",
		totalGames,
		gameConcurrency,
		numSimulations,
		inferenceBatchSize,
	)

	stats, runErr := selfplayRunner.Run(ctx)
	log.Printf(
		"selfplay finished: requested=%d started=%d completed=%d saved=%d save_failed=%d "+
			"max_moves=%d search_failed=%d illegal_action=%d canceled=%d samples=%d duration=%s",
		stats.Requested,
		stats.Started,
		stats.Completed,
		stats.Saved,
		stats.SaveFailed,
		stats.MaxMoves,
		stats.SearchFailed,
		stats.IllegalAction,
		stats.Canceled,
		stats.Samples,
		stats.Duration,
	)
	if runErr != nil {
		return fmt.Errorf("run selfplay: %w", runErr)
	}
	return nil
}
