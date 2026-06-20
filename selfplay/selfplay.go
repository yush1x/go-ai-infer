package selfplay

import (
	"context"
	"errors"
	"fmt"

	"go-ai-infer/board"
	"go-ai-infer/inference"
	"go-ai-infer/mcts"
)

// DefaultMaxMoves 是单盘自博弈默认允许执行的最大手数。
const DefaultMaxMoves = 400

// MaxMoves 保留为兼容旧调用；新代码应通过 PlayConfig.MaxMoves 配置。
const MaxMoves = DefaultMaxMoves

// Komi 与当前 MCTS 使用的固定规则一致。
const Komi float32 = 7.5

// Searcher 是 SelfPlay 对 MCTS 的最小依赖。
type Searcher interface {
	Search(ctx context.Context, b *board.Board) (*mcts.SearchResult, error)
}

type PlayConfig struct {
	MaxMoves int
	OnMove   func(move int)
}

func DefaultPlayConfig() PlayConfig {
	return PlayConfig{MaxMoves: DefaultMaxMoves}
}

type Status string

const (
	StatusCompleted     Status = "completed"
	StatusMaxMoves      Status = "max_moves"
	StatusSearchFailed  Status = "search_failed"
	StatusIllegalAction Status = "illegal_action"
	StatusCanceled      Status = "canceled"
)

// Sample 是一个训练样本。Features 和 Policy 在该手落子前记录，
// Value、Score 和 Ownership 在整盘正常结束后统一回填。
type Sample struct {
	Features  inference.Features
	Policy    [inference.PolicySize]float32
	Value     float32
	Score     float32
	Ownership [board.Points]int8

	Player int
	Action int
}

// Game 是一盘正常结束、可以进入训练的数据。
type Game struct {
	Samples    []Sample
	Actions    []int
	Final      board.FinalResult
	BlackLead  float32
	Winner     int
	TotalMoves int
}

// Result 同时表达成功结果和失败诊断。只有 StatusCompleted 时 Game 非 nil。
type Result struct {
	Status     Status
	Game       *Game
	Moves      int
	LastAction int
	Err        error
}

// Play 从空棋盘开始生成一盘自博弈棋局。
// 它不负责并发调度、日志或 HDF5 写入。
func Play(ctx context.Context, searcher Searcher) Result {
	return PlayWithConfig(ctx, searcher, DefaultPlayConfig())
}

// PlayWithConfig 从空棋盘开始生成一盘可配置的自博弈棋局。
func PlayWithConfig(ctx context.Context, searcher Searcher, config PlayConfig) Result {
	if ctx == nil {
		return failure(StatusSearchFailed, 0, -1, errors.New("selfplay: context is nil"))
	}
	if searcher == nil {
		return failure(StatusSearchFailed, 0, -1, errors.New("selfplay: searcher is nil"))
	}
	if config.MaxMoves <= 0 {
		return failure(StatusSearchFailed, 0, -1, errors.New("selfplay: max moves must be positive"))
	}

	b := board.New()
	samples := make([]Sample, 0, config.MaxMoves)
	actions := make([]int, 0, config.MaxMoves)
	lastAction := -1

	for move := 0; move < config.MaxMoves; move++ {
		if err := ctx.Err(); err != nil {
			return failure(StatusCanceled, move, lastAction, err)
		}

		player := board.Black
		if b.Round()%2 == 1 {
			player = board.White
		}

		searchResult, err := searcher.Search(ctx, b)
		if err != nil {
			status := StatusSearchFailed
			if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				status = StatusCanceled
			}
			return failure(status, move, lastAction, fmt.Errorf("selfplay: search move %d: %w", move, err))
		}
		if searchResult == nil {
			return failure(StatusSearchFailed, move, lastAction, fmt.Errorf("selfplay: search move %d returned nil result", move))
		}

		action := searchResult.Action
		sample := Sample{
			Features: b.Tensor(),
			Policy:   searchResult.VisitProbs,
			Player:   player,
			Action:   action,
		}

		if !applyAction(b, action) {
			return failure(
				StatusIllegalAction,
				move,
				action,
				fmt.Errorf("selfplay: search returned illegal action %d at move %d", action, move),
			)
		}

		samples = append(samples, sample)
		actions = append(actions, action)
		lastAction = action
		if config.OnMove != nil {
			config.OnMove(len(actions))
		}

		if b.IsFinish() == 1 {
			game := finishGame(samples, actions, b.FinalResult())
			return Result{
				Status:     StatusCompleted,
				Game:       game,
				Moves:      len(actions),
				LastAction: lastAction,
			}
		}
	}

	return Result{
		Status:     StatusMaxMoves,
		Game:       finishGame(samples, actions, b.FinalResult()),
		Moves:      config.MaxMoves,
		LastAction: lastAction,
		Err:        fmt.Errorf("selfplay: game reached move limit %d", config.MaxMoves),
	}
}

func applyAction(b *board.Board, action int) bool {
	if action == mcts.PassAction {
		return b.Move(-1, -1) == 0
	}
	if action < 0 || action >= board.Points {
		return false
	}
	return b.Move(action/board.Size, action%board.Size) == 0
}

func finishGame(samples []Sample, actions []int, final board.FinalResult) *Game {
	blackLead := float32(final.Black-final.White) - Komi
	winner := board.Black
	if blackLead < 0 {
		winner = board.White
	}

	for i := range samples {
		score := blackLead
		if samples[i].Player == board.White {
			score = -score
		}
		samples[i].Score = score
		switch {
		case score > 0:
			samples[i].Value = 1
		case score < 0:
			samples[i].Value = -1
		default:
			samples[i].Value = 0
		}
		samples[i].Ownership = final.Ownership
	}

	return &Game{
		Samples:    samples,
		Actions:    actions,
		Final:      final,
		BlackLead:  blackLead,
		Winner:     winner,
		TotalMoves: len(actions),
	}
}

func failure(status Status, moves int, lastAction int, err error) Result {
	return Result{
		Status:     status,
		Moves:      moves,
		LastAction: lastAction,
		Err:        err,
	}
}
