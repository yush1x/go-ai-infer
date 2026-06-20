package match

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"go-ai-infer/board"
	"go-ai-infer/mcts"
	"go-ai-infer/selfplay"
)

const ReportVersion = 1

type Player struct {
	Name     string
	Searcher selfplay.Searcher
}

type Config struct {
	Games       int
	Concurrency int
	MaxMoves    int
	OnGameEvent func(GameEvent)
}

type GameEvent struct {
	Game       int
	Status     string
	Moves      int
	BlackModel string
	WhiteModel string
	Winner     string
	BlackLead  float32
	Err        error
}

type Stats struct {
	Requested int
	Started   int

	Completed     int
	MaxMoves      int
	SearchFailed  int
	IllegalAction int
	Canceled      int

	ModelAWins int
	ModelBWins int
	Duration   time.Duration
}

func (s Stats) ValidGames() int {
	return s.Completed + s.MaxMoves
}

func (s Stats) FailedGames() int {
	return s.SearchFailed + s.IllegalAction + s.Canceled
}

type Runner struct {
	modelA Player
	modelB Player
	config Config
}

func New(modelA, modelB Player, config Config) (*Runner, error) {
	if modelA.Name == "" || modelB.Name == "" {
		return nil, errors.New("match: model names must not be empty")
	}
	if modelA.Name == modelB.Name {
		return nil, errors.New("match: model names must be different")
	}
	if modelA.Searcher == nil || modelB.Searcher == nil {
		return nil, errors.New("match: model searchers must not be nil")
	}
	if config.Games <= 0 {
		return nil, errors.New("match: games must be positive")
	}
	if config.Concurrency <= 0 {
		return nil, errors.New("match: concurrency must be positive")
	}
	if config.MaxMoves == 0 {
		config.MaxMoves = selfplay.DefaultMaxMoves
	}
	if config.MaxMoves < 0 {
		return nil, errors.New("match: max moves must be positive")
	}
	if config.Concurrency > config.Games {
		config.Concurrency = config.Games
	}

	return &Runner{modelA: modelA, modelB: modelB, config: config}, nil
}

type gameJob struct {
	index int
	black Player
	white Player
}

type gameResult struct {
	job    gameJob
	result selfplay.Result
}

func (r *Runner) Run(ctx context.Context) (Report, Stats, error) {
	if ctx == nil {
		return Report{}, Stats{}, errors.New("match: context is nil")
	}

	startedAt := time.Now()
	stats := Stats{Requested: r.config.Games}
	records := make([]Game, r.config.Games)
	jobs := make(chan gameJob)
	results := make(chan gameResult, r.config.Concurrency)

	var workers sync.WaitGroup
	workers.Add(r.config.Concurrency)
	for range r.config.Concurrency {
		go func() {
			defer workers.Done()
			for job := range jobs {
				r.emit(GameEvent{
					Game: job.index, Status: "running",
					BlackModel: job.black.Name, WhiteModel: job.white.Name,
				})
				searcher := colorSearcher{
					black: job.black.Searcher,
					white: job.white.Searcher,
				}
				result := selfplay.PlayWithConfig(ctx, searcher, selfplay.PlayConfig{
					MaxMoves: r.config.MaxMoves,
					OnMove: func(move int) {
						r.emit(GameEvent{
							Game: job.index, Status: "running", Moves: move,
							BlackModel: job.black.Name, WhiteModel: job.white.Name,
						})
					},
				})
				results <- gameResult{job: job, result: result}
			}
		}()
	}

	go func() {
		defer close(jobs)
		for index := 1; index <= r.config.Games; index++ {
			black, white := r.modelA, r.modelB
			if index%2 == 0 {
				black, white = white, black
			}
			select {
			case jobs <- gameJob{index: index, black: black, white: white}:
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
		record, err := buildGame(item.job, item.result)
		if err != nil {
			return Report{}, stats, err
		}
		records[item.job.index-1] = record
		r.updateStats(&stats, record)
		r.emit(GameEvent{
			Game: item.job.index, Status: record.Status, Moves: record.TotalMoves,
			BlackModel: record.BlackModel, WhiteModel: record.WhiteModel,
			Winner: record.WinnerModel, BlackLead: record.BlackLead, Err: item.result.Err,
		})
	}

	stats.Duration = time.Since(startedAt)
	finishedRecords := make([]Game, 0, stats.Started)
	for _, record := range records {
		if record.ID != 0 {
			finishedRecords = append(finishedRecords, record)
		}
	}
	report := Report{
		Version:     ReportVersion,
		GeneratedAt: time.Now().Format(time.RFC3339),
		BoardSize:   board.Size,
		Komi:        selfplay.Komi,
		Models:      []string{r.modelA.Name, r.modelB.Name},
		Config: ReportConfig{
			Games: r.config.Games, Concurrency: r.config.Concurrency, MaxMoves: r.config.MaxMoves,
		},
		Summary: summaryFromStats(stats),
		Games:   finishedRecords,
	}

	if err := ctx.Err(); err != nil {
		return report, stats, err
	}
	if stats.Started != stats.Requested {
		return report, stats, fmt.Errorf(
			"match: started %d of %d games without context cancellation",
			stats.Started, stats.Requested,
		)
	}
	return report, stats, nil
}

type colorSearcher struct {
	black selfplay.Searcher
	white selfplay.Searcher
}

func (s colorSearcher) Search(ctx context.Context, b *board.Board) (*mcts.SearchResult, error) {
	if b.Round()%2 == 0 {
		return s.black.Search(ctx, b)
	}
	return s.white.Search(ctx, b)
}

func (r *Runner) updateStats(stats *Stats, game Game) {
	switch selfplay.Status(game.Status) {
	case selfplay.StatusCompleted:
		stats.Completed++
	case selfplay.StatusMaxMoves:
		stats.MaxMoves++
	case selfplay.StatusSearchFailed:
		stats.SearchFailed++
	case selfplay.StatusIllegalAction:
		stats.IllegalAction++
	case selfplay.StatusCanceled:
		stats.Canceled++
	default:
		stats.SearchFailed++
	}

	switch game.WinnerModel {
	case r.modelA.Name:
		stats.ModelAWins++
	case r.modelB.Name:
		stats.ModelBWins++
	}
}

func (r *Runner) emit(event GameEvent) {
	if r.config.OnGameEvent != nil {
		r.config.OnGameEvent(event)
	}
}
