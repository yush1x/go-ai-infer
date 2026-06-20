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
	"go-ai-infer/match"
	"go-ai-infer/mcts"
)

// =============================================================================
// Match 配置：通常只需要修改这里
// =============================================================================
const (
	modelAName = "model-a"
	modelBName = "model-b"
	modelAURL  = "http://127.0.0.1:8000/predict/a"
	modelBURL  = "http://127.0.0.1:8000/predict/b"

	totalGames      = 20  // 建议使用偶数，保证双方执黑次数相同
	gameConcurrency = 8   // 同时运行的棋局数
	maxMoves        = 450 // 达到上限后按当前盘面结算

	numSimulations = 100
	cPuct          = float32(1.5)

	// true 会加入根节点噪声，并在前 30 手按访问概率采样，使多盘对局不完全重复。
	// 若需要完全确定性的单局比较可改成 false。
	enableMatchExploration = true
	dirichletAlpha         = float32(0.03)
	dirichletEps           = float32(0.05)
	passPolicyFloor        = float32(0.0)
	passBonus              = float32(0.8)

	inferenceBatchSize = 8
	inferenceMaxWait   = 5 * time.Millisecond
	inferenceQueueSize = 128
	inferenceTimeout   = 30 * time.Second

	outputFile = "matchgame/results.json"
)

func main() {
	if err := run(); err != nil {
		log.Printf("match stopped: %v", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	batcherA, err := newBatcher(modelAURL)
	if err != nil {
		return fmt.Errorf("create %s inference: %w", modelAName, err)
	}
	defer batcherA.Close()

	batcherB, err := newBatcher(modelBURL)
	if err != nil {
		return fmt.Errorf("create %s inference: %w", modelBName, err)
	}
	defer batcherB.Close()

	searchConfig := mcts.Config{
		NumSimulations: numSimulations, CPuct: cPuct,
		SelfPlay: enableMatchExploration, DirichletAlpha: dirichletAlpha,
		DirichletEps: dirichletEps, PassPolicyFloor: passPolicyFloor,
		PassBonus: passBonus,
	}
	searcherA := mcts.NewSearcher(batcherA, searchConfig)
	searcherB := mcts.NewSearcher(batcherB, searchConfig)

	config := match.Config{
		Games: totalGames, Concurrency: gameConcurrency, MaxMoves: maxMoves,
		OnGameEvent: logGameEvent,
	}
	runner, err := match.New(
		match.Player{Name: modelAName, Searcher: searcherA},
		match.Player{Name: modelBName, Searcher: searcherB},
		config,
	)
	if err != nil {
		return err
	}

	inference.EnableBatchLog = false
	fmt.Printf(
		"Match started: %s vs %s games=%d concurrency=%d simulations=%d max_moves=%d\n",
		modelAName, modelBName, totalGames, gameConcurrency, numSimulations, maxMoves,
	)
	report, stats, runErr := runner.Run(ctx)
	report.Config.Simulations = numSimulations
	report.Config.CPuct = cPuct
	report.Config.Exploration = enableMatchExploration
	report.Config.DirichletAlpha = dirichletAlpha
	report.Config.DirichletEps = dirichletEps
	report.Config.PassPolicyFloor = passPolicyFloor
	report.Config.PassBonus = passBonus
	report.Config.InferenceBatchSize = inferenceBatchSize
	report.Config.InferenceMaxWaitMillis = inferenceMaxWait.Milliseconds()
	if stats.Started > 0 {
		path, writeErr := match.WriteReport(outputFile, report)
		if writeErr != nil {
			return writeErr
		}
		fmt.Printf("Report: %s\nViewer: matchgame/viewer.html\n", path)
	}
	printSummary(stats)
	if runErr != nil {
		return runErr
	}
	return nil
}

func newBatcher(url string) (*inference.Batcher, error) {
	client, err := inference.NewHTTPClient(url, inferenceTimeout)
	if err != nil {
		return nil, err
	}
	return inference.NewBatcher(client, inference.BatcherConfig{
		BatchSize: inferenceBatchSize,
		MaxWait:   inferenceMaxWait,
		QueueSize: inferenceQueueSize,
	})
}

func logGameEvent(event match.GameEvent) {
	switch event.Status {
	case "completed", "max_moves":
		fmt.Printf(
			"Game #%03d finished: %s(black) vs %s(white), winner=%s, black_lead=%.1f, moves=%d, status=%s\n",
			event.Game, event.BlackModel, event.WhiteModel, event.Winner,
			event.BlackLead, event.Moves, event.Status,
		)
	case "search_failed", "illegal_action", "canceled":
		fmt.Printf(
			"Game #%03d failed: %s(black) vs %s(white), moves=%d, status=%s, error=%v\n",
			event.Game, event.BlackModel, event.WhiteModel, event.Moves, event.Status, event.Err,
		)
	}
}

func printSummary(stats match.Stats) {
	valid := stats.ValidGames()
	var rateA, rateB float64
	if valid > 0 {
		rateA = 100 * float64(stats.ModelAWins) / float64(valid)
		rateB = 100 * float64(stats.ModelBWins) / float64(valid)
	}
	fmt.Printf(
		"Match finished: valid=%d/%d completed=%d max_moves=%d failed=%d duration=%s\n",
		valid, stats.Requested, stats.Completed, stats.MaxMoves, stats.FailedGames(), stats.Duration.Round(time.Second),
	)
	fmt.Printf("%s: wins=%d win_rate=%.2f%%\n", modelAName, stats.ModelAWins, rateA)
	fmt.Printf("%s: wins=%d win_rate=%.2f%%\n", modelBName, stats.ModelBWins, rateB)
}
