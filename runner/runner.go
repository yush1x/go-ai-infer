package runner

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"go-ai-infer/selfplay"
)

const (
	DefaultGames       = 100
	DefaultConcurrency = 8
)

type Config struct {
	Games       int
	Concurrency int
	MaxMoves    int
	OnGameEvent func(GameEvent)
}

func DefaultConfig() Config {
	return Config{
		Games:       DefaultGames,
		Concurrency: DefaultConcurrency,
		MaxMoves:    selfplay.DefaultMaxMoves,
	}
}

type GameEvent struct {
	Game   int
	Status string
	Moves  int
	Err    error
}

// GameSaver 负责把一盘正常结束的棋提交给 Python 保存服务。
type GameSaver interface {
	SaveGame(ctx context.Context, game *selfplay.Game) error
}

type Stats struct {
	Requested int
	Started   int

	Completed     int
	Saved         int
	SaveFailed    int
	MaxMoves      int
	SearchFailed  int
	IllegalAction int
	Canceled      int

	Samples  int
	Duration time.Duration
}

type Runner struct {
	searcher selfplay.Searcher
	saver    GameSaver
	config   Config
	logger   *log.Logger
}

func New(searcher selfplay.Searcher, saver GameSaver, config Config) (*Runner, error) {
	if searcher == nil {
		return nil, errors.New("runner: searcher is nil")
	}
	if saver == nil {
		return nil, errors.New("runner: saver is nil")
	}
	if config.Games <= 0 {
		return nil, errors.New("runner: games must be positive")
	}
	if config.Concurrency <= 0 {
		return nil, errors.New("runner: concurrency must be positive")
	}
	if config.MaxMoves == 0 {
		config.MaxMoves = selfplay.DefaultMaxMoves
	}
	if config.MaxMoves < 0 {
		return nil, errors.New("runner: max moves must be positive")
	}
	if config.Concurrency > config.Games {
		config.Concurrency = config.Games
	}
	return &Runner{
		searcher: searcher,
		saver:    saver,
		config:   config,
		logger:   log.Default(),
	}, nil
}

// SetLogger 替换 Runner 的日志输出；传 nil 可关闭日志。
func (r *Runner) SetLogger(logger *log.Logger) {
	r.logger = logger
}

type gameResult struct {
	index  int
	result selfplay.Result
}

// Run 并发生成指定数量的棋局，并将正常结束的棋局逐盘提交给 Python。
// 保存请求串行执行；保存失败只记录并统计，不重试。
func (r *Runner) Run(ctx context.Context) (Stats, error) {
	if ctx == nil {
		return Stats{}, errors.New("runner: context is nil")
	}

	startedAt := time.Now()
	stats := Stats{Requested: r.config.Games}
	jobs := make(chan int)
	results := make(chan gameResult, r.config.Concurrency)

	var workers sync.WaitGroup
	workers.Add(r.config.Concurrency)
	for i := 0; i < r.config.Concurrency; i++ {
		go func() {
			defer workers.Done()
			for index := range jobs {
				r.emit(GameEvent{Game: index, Status: "running"})
				result := selfplay.PlayWithConfig(ctx, r.searcher, selfplay.PlayConfig{
					MaxMoves: r.config.MaxMoves,
					OnMove: func(move int) {
						r.emit(GameEvent{Game: index, Status: "running", Moves: move})
					},
				})
				results <- gameResult{index: index, result: result}
			}
		}()
	}

	go func() {
		defer close(jobs)
		for index := 1; index <= r.config.Games; index++ {
			select {
			case jobs <- index:
			case <-ctx.Done():
				return
			}
		}
	}()

	go func() {
		workers.Wait()
		close(results)
	}()

	for item := range results {
		stats.Started++
		result := item.result
		switch result.Status {
		case selfplay.StatusCompleted:
			stats.Completed++
			if result.Game == nil {
				stats.SaveFailed++
				r.emit(GameEvent{Game: item.index, Status: "save_failed", Moves: result.Moves})
				r.logf("runner: game=%d completed with nil game", item.index)
				continue
			}
			r.emit(GameEvent{Game: item.index, Status: "saving", Moves: result.Moves})
			if err := r.saver.SaveGame(ctx, result.Game); err != nil {
				stats.SaveFailed++
				r.emit(GameEvent{
					Game: item.index, Status: "save_failed", Moves: result.Moves, Err: err,
				})
				r.logf("runner: game=%d save failed: %v", item.index, err)
				continue
			}
			stats.Saved++
			stats.Samples += len(result.Game.Samples)
			r.emit(GameEvent{Game: item.index, Status: "completed", Moves: result.Moves})

		case selfplay.StatusMaxMoves:
			stats.MaxMoves++
			if result.Game == nil {
				stats.SaveFailed++
				r.emit(GameEvent{Game: item.index, Status: "save_failed", Moves: result.Moves})
				r.logf("runner: game=%d reached max moves with nil game", item.index)
				continue
			}
			r.emit(GameEvent{Game: item.index, Status: "saving", Moves: result.Moves})
			if err := r.saver.SaveGame(ctx, result.Game); err != nil {
				stats.SaveFailed++
				r.emit(GameEvent{
					Game: item.index, Status: "save_failed", Moves: result.Moves, Err: err,
				})
				r.logf("runner: game=%d max-moves save failed: %v", item.index, err)
				continue
			}
			stats.Saved++
			stats.Samples += len(result.Game.Samples)
			r.emit(GameEvent{Game: item.index, Status: "completed", Moves: result.Moves})
		case selfplay.StatusSearchFailed:
			stats.SearchFailed++
			r.emitResult(item.index, result)
			r.logFailure(item.index, result)
		case selfplay.StatusIllegalAction:
			stats.IllegalAction++
			r.emitResult(item.index, result)
			r.logFailure(item.index, result)
		case selfplay.StatusCanceled:
			stats.Canceled++
			r.emitResult(item.index, result)
			r.logFailure(item.index, result)
		default:
			stats.SearchFailed++
			r.emit(GameEvent{
				Game: item.index, Status: "failed", Moves: result.Moves, Err: result.Err,
			})
			r.logf("runner: game=%d unknown status=%q moves=%d err=%v",
				item.index, result.Status, result.Moves, result.Err)
		}
	}

	stats.Duration = time.Since(startedAt)
	if err := ctx.Err(); err != nil {
		return stats, err
	}
	if stats.Started != stats.Requested {
		return stats, fmt.Errorf(
			"runner: started %d of %d games without context cancellation",
			stats.Started,
			stats.Requested,
		)
	}
	return stats, nil
}

func (r *Runner) emitResult(index int, result selfplay.Result) {
	r.emit(GameEvent{
		Game: index, Status: string(result.Status), Moves: result.Moves, Err: result.Err,
	})
}

func (r *Runner) emit(event GameEvent) {
	if r.config.OnGameEvent != nil {
		r.config.OnGameEvent(event)
	}
}

func (r *Runner) logFailure(index int, result selfplay.Result) {
	r.logf("runner: game=%d status=%s moves=%d last_action=%d err=%v",
		index, result.Status, result.Moves, result.LastAction, result.Err)
}

func (r *Runner) logf(format string, args ...any) {
	if r.logger != nil {
		r.logger.Printf(format, args...)
	}
}
