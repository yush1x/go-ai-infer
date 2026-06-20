package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"syscall"
	"time"

	"go-ai-infer/board"
	"go-ai-infer/inference"
	"go-ai-infer/mcts"
)

type config struct {
	PredictURL      string  `json:"predict_url"`
	OutputFile      string  `json:"-"`
	Simulations     int     `json:"simulations"`
	MaxMoves        int     `json:"max_moves"`
	CPuct           float32 `json:"c_puct"`
	DirichletAlpha  float32 `json:"dirichlet_alpha"`
	DirichletEps    float32 `json:"dirichlet_eps"`
	PassPolicyFloor float32 `json:"pass_policy_floor"`
	PassBonus       float32 `json:"pass_bonus"`
}

type topAction struct {
	Action      int     `json:"action"`
	Coordinate  string  `json:"coordinate"`
	Probability float32 `json:"probability"`
}

type moveTrace struct {
	Move             int         `json:"move"`
	Player           string      `json:"player"`
	Action           int         `json:"action"`
	Coordinate       string      `json:"coordinate"`
	Pass             bool        `json:"pass"`
	Board            []int       `json:"board"`
	RootValue        float32     `json:"root_value"`
	RawPassPolicy    float32     `json:"raw_pass_policy"`
	PassPrior        float32     `json:"pass_prior"`
	PassVisits       int32       `json:"pass_visits"`
	TotalVisits      int32       `json:"total_visits"`
	PassBonusApplied bool        `json:"pass_bonus_applied"`
	TopActions       []topAction `json:"top_actions"`
}

type gameTrace struct {
	GeneratedAt  string      `json:"generated_at"`
	Status       string      `json:"status"`
	Error        string      `json:"error,omitempty"`
	Config       config      `json:"config"`
	Moves        []moveTrace `json:"moves"`
	FinalBlack   int         `json:"final_black"`
	FinalWhite   int         `json:"final_white"`
	FinalNeutral int         `json:"final_neutral"`
}

func main() {
	cfg := parseFlags()
	if err := run(cfg); err != nil {
		fmt.Fprintln(os.Stderr, "debug game failed:", err)
		os.Exit(1)
	}
}

func parseFlags() config {
	cfg := config{
		PredictURL:      defaultPredictURL,
		OutputFile:      defaultOutputFile,
		Simulations:     defaultSimulations,
		MaxMoves:        defaultMaxMoves,
		CPuct:           defaultCPuct,
		DirichletAlpha:  defaultDirichletAlpha,
		DirichletEps:    defaultDirichletEps,
		PassPolicyFloor: defaultPassPolicyFloor,
		PassBonus:       defaultPassBonus,
	}
	flag.StringVar(&cfg.PredictURL, "predict", cfg.PredictURL, "Python predict endpoint")
	flag.StringVar(&cfg.OutputFile, "output", cfg.OutputFile, "JSON output file")
	flag.IntVar(&cfg.Simulations, "simulations", cfg.Simulations, "MCTS simulations per move")
	flag.IntVar(&cfg.MaxMoves, "max-moves", cfg.MaxMoves, "maximum moves")
	flag.Var((*float32Flag)(&cfg.CPuct), "c-puct", "PUCT exploration constant")
	flag.Var((*float32Flag)(&cfg.DirichletAlpha), "dirichlet-alpha", "Dirichlet alpha")
	flag.Var((*float32Flag)(&cfg.DirichletEps), "dirichlet-eps", "Dirichlet noise ratio")
	flag.Var((*float32Flag)(&cfg.PassPolicyFloor), "pass-floor", "minimum pass policy before normalization")
	flag.Var((*float32Flag)(&cfg.PassBonus), "pass-bonus", "pass PUCT bonus when top moves are eyes")
	flag.Parse()
	return cfg
}

type float32Flag float32

func (f *float32Flag) String() string {
	return fmt.Sprintf("%g", *f)
}

func (f *float32Flag) Set(value string) error {
	var parsed float64
	if _, err := fmt.Sscan(value, &parsed); err != nil {
		return err
	}
	*f = float32Flag(parsed)
	return nil
}

func run(cfg config) error {
	if cfg.Simulations <= 0 || cfg.MaxMoves <= 0 {
		return fmt.Errorf("simulations and max-moves must be positive")
	}
	if cfg.OutputFile == "" {
		return fmt.Errorf("output file must not be empty")
	}
	if cfg.PassPolicyFloor < 0 || cfg.PassBonus < 0 {
		return fmt.Errorf("pass-floor and pass-bonus must be non-negative")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	client, err := inference.NewHTTPClient(cfg.PredictURL, 30*time.Second)
	if err != nil {
		return err
	}
	inference.EnableBatchLog = false
	batcher, err := inference.NewBatcher(client, inference.BatcherConfig{
		BatchSize: 1,
		MaxWait:   time.Millisecond,
		QueueSize: 4,
	})
	if err != nil {
		return err
	}
	defer batcher.Close()

	searcher := mcts.NewSearcher(batcher, mcts.Config{
		NumSimulations:  cfg.Simulations,
		CPuct:           cfg.CPuct,
		SelfPlay:        true,
		DirichletAlpha:  cfg.DirichletAlpha,
		DirichletEps:    cfg.DirichletEps,
		PassPolicyFloor: cfg.PassPolicyFloor,
		PassBonus:       cfg.PassBonus,
	})

	trace := gameTrace{
		GeneratedAt: time.Now().Format(time.RFC3339),
		Status:      "running",
		Config:      cfg,
		Moves: []moveTrace{{
			Move:       0,
			Player:     "black",
			Action:     -1,
			Coordinate: "start",
			Board:      make([]int, board.Points),
		}},
	}

	b := board.New()
	lastStatus := "max_moves"
	for move := 1; move <= cfg.MaxMoves; move++ {
		if err := ctx.Err(); err != nil {
			trace.Status = "canceled"
			trace.Error = err.Error()
			lastStatus = trace.Status
			break
		}

		player := "black"
		if b.Round()%2 == 1 {
			player = "white"
		}
		result, err := searcher.Search(ctx, b)
		if err != nil {
			trace.Status = "search_failed"
			trace.Error = err.Error()
			lastStatus = trace.Status
			break
		}
		if !applyAction(b, result.Action) {
			trace.Status = "illegal_action"
			trace.Error = fmt.Sprintf("illegal action %d at move %d", result.Action, move)
			lastStatus = trace.Status
			break
		}

		trace.Moves = append(trace.Moves, moveTrace{
			Move:             move,
			Player:           player,
			Action:           result.Action,
			Coordinate:       coordinate(result.Action),
			Pass:             result.Action == mcts.PassAction,
			Board:            snapshotBoard(b),
			RootValue:        result.RootValue,
			RawPassPolicy:    result.RawPassPolicy,
			PassPrior:        result.PassPrior,
			PassVisits:       result.PassVisits,
			TotalVisits:      result.TotalVisits,
			PassBonusApplied: result.PassBonusApplied,
			TopActions:       topActions(result.VisitProbs, 10),
		})
		fmt.Fprintf(os.Stderr, "\rmove %d/%d action=%s pass_prior=%.5f pass_visits=%d/%d",
			move, cfg.MaxMoves, coordinate(result.Action), result.PassPrior, result.PassVisits, result.TotalVisits)

		if b.IsFinish() == 1 {
			trace.Status = "completed"
			lastStatus = trace.Status
			break
		}
	}
	fmt.Fprintln(os.Stderr)
	if trace.Status == "running" {
		trace.Status = lastStatus
	}
	final := b.FinalResult()
	trace.FinalBlack = final.Black
	trace.FinalWhite = final.White
	trace.FinalNeutral = final.Neutral

	path, err := writeTrace(cfg.OutputFile, trace)
	if err != nil {
		return err
	}
	fmt.Printf("status: %s\njson: %s\nviewer: debuggame/viewer.html\n", trace.Status, path)
	return nil
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

func snapshotBoard(b *board.Board) []int {
	features := b.Tensor()
	cells := make([]int, board.Points)
	for p := 0; p < board.Points; p++ {
		switch {
		case features[p] > 0:
			cells[p] = board.Black
		case features[board.Points+p] > 0:
			cells[p] = board.White
		}
	}
	return cells
}

func topActions(policy [mcts.PolicySize]float32, limit int) []topAction {
	actions := make([]topAction, 0, len(policy))
	for action, probability := range policy {
		if probability <= 0 {
			continue
		}
		actions = append(actions, topAction{
			Action: action, Coordinate: coordinate(action), Probability: probability,
		})
	}
	sort.Slice(actions, func(i, j int) bool {
		return actions[i].Probability > actions[j].Probability
	})
	if len(actions) > limit {
		actions = actions[:limit]
	}
	return actions
}

func coordinate(action int) string {
	if action == mcts.PassAction {
		return "pass"
	}
	if action < 0 || action >= board.Points {
		return "-"
	}
	return fmt.Sprintf("(%d,%d)", action/board.Size, action%board.Size)
}

func writeTrace(path string, trace gameTrace) (string, error) {
	data, err := json.MarshalIndent(trace, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return path, nil
	}
	return absolute, nil
}
