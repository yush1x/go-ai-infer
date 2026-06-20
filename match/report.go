package match

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"go-ai-infer/board"
	"go-ai-infer/mcts"
	"go-ai-infer/selfplay"
)

type Report struct {
	Version     int          `json:"version"`
	GeneratedAt string       `json:"generated_at"`
	BoardSize   int          `json:"board_size"`
	Komi        float32      `json:"komi"`
	Models      []string     `json:"models"`
	Config      ReportConfig `json:"config"`
	Summary     Summary      `json:"summary"`
	Games       []Game       `json:"games"`
}

type ReportConfig struct {
	Games                  int     `json:"games"`
	Concurrency            int     `json:"concurrency"`
	MaxMoves               int     `json:"max_moves"`
	Simulations            int     `json:"simulations,omitempty"`
	CPuct                  float32 `json:"c_puct,omitempty"`
	Exploration            bool    `json:"exploration"`
	DirichletAlpha         float32 `json:"dirichlet_alpha,omitempty"`
	DirichletEps           float32 `json:"dirichlet_eps,omitempty"`
	PassPolicyFloor        float32 `json:"pass_policy_floor,omitempty"`
	PassBonus              float32 `json:"pass_bonus,omitempty"`
	InferenceBatchSize     int     `json:"inference_batch_size,omitempty"`
	InferenceMaxWaitMillis int64   `json:"inference_max_wait_ms,omitempty"`
}

type Summary struct {
	ModelAWins    int     `json:"model_a_wins"`
	ModelBWins    int     `json:"model_b_wins"`
	ModelAWinRate float64 `json:"model_a_win_rate"`
	ModelBWinRate float64 `json:"model_b_win_rate"`
	ValidGames    int     `json:"valid_games"`
	Completed     int     `json:"completed"`
	MaxMoves      int     `json:"max_moves"`
	Failed        int     `json:"failed"`
}

type Game struct {
	ID           int     `json:"id"`
	Status       string  `json:"status"`
	Error        string  `json:"error,omitempty"`
	BlackModel   string  `json:"black_model"`
	WhiteModel   string  `json:"white_model"`
	WinnerModel  string  `json:"winner_model,omitempty"`
	WinnerColor  string  `json:"winner_color,omitempty"`
	BlackLead    float32 `json:"black_lead"`
	FinalBlack   int     `json:"final_black"`
	FinalWhite   int     `json:"final_white"`
	FinalNeutral int     `json:"final_neutral"`
	TotalMoves   int     `json:"total_moves"`
	Frames       []Frame `json:"frames,omitempty"`
}

type Frame struct {
	Move       int    `json:"move"`
	Player     string `json:"player"`
	Model      string `json:"model"`
	Action     int    `json:"action"`
	Coordinate string `json:"coordinate"`
	Pass       bool   `json:"pass"`
	Board      []int  `json:"board"`
}

func WriteReport(path string, report Report) (string, error) {
	if path == "" {
		return "", fmt.Errorf("match: output path must not be empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", fmt.Errorf("match: create output directory: %w", err)
	}
	file, err := os.Create(path)
	if err != nil {
		return "", fmt.Errorf("match: create report: %w", err)
	}
	encoder := json.NewEncoder(file)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(report); err != nil {
		_ = file.Close()
		return "", fmt.Errorf("match: encode report: %w", err)
	}
	if err := file.Close(); err != nil {
		return "", fmt.Errorf("match: close report: %w", err)
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return path, nil
	}
	return absolute, nil
}

func buildGame(job gameJob, result selfplay.Result) (Game, error) {
	record := Game{
		ID: job.index, Status: string(result.Status), BlackModel: job.black.Name,
		WhiteModel: job.white.Name, TotalMoves: result.Moves,
	}
	if result.Err != nil {
		record.Error = result.Err.Error()
	}
	if result.Game == nil {
		return record, nil
	}

	game := result.Game
	record.BlackLead = game.BlackLead
	record.FinalBlack = game.Final.Black
	record.FinalWhite = game.Final.White
	record.FinalNeutral = game.Final.Neutral
	record.WinnerColor = colorName(game.Winner)
	if game.Winner == board.Black {
		record.WinnerModel = job.black.Name
	} else {
		record.WinnerModel = job.white.Name
	}

	frames, err := buildFrames(game.Actions, job.black.Name, job.white.Name)
	if err != nil {
		return Game{}, fmt.Errorf("match: rebuild game %d: %w", job.index, err)
	}
	record.Frames = frames
	return record, nil
}

func buildFrames(actions []int, blackModel, whiteModel string) ([]Frame, error) {
	b := board.New()
	frames := make([]Frame, 0, len(actions)+1)
	frames = append(frames, Frame{
		Move: 0, Player: "black", Model: blackModel, Action: -1,
		Coordinate: "start", Board: snapshotBoard(b),
	})

	for index, action := range actions {
		playerName := "black"
		model := blackModel
		if b.Round()%2 == 1 {
			playerName = "white"
			model = whiteModel
		}
		if !applyAction(b, action) {
			return nil, fmt.Errorf("illegal recorded action %d at move %d", action, index+1)
		}
		frames = append(frames, Frame{
			Move: index + 1, Player: playerName, Model: model, Action: action,
			Coordinate: coordinate(action), Pass: action == mcts.PassAction,
			Board: snapshotBoard(b),
		})
	}
	return frames, nil
}

func snapshotBoard(b *board.Board) []int {
	features := b.Tensor()
	cells := make([]int, board.Points)
	for point := 0; point < board.Points; point++ {
		switch {
		case features[point] > 0:
			cells[point] = board.Black
		case features[board.Points+point] > 0:
			cells[point] = board.White
		}
	}
	return cells
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

func coordinate(action int) string {
	if action == mcts.PassAction {
		return "pass"
	}
	if action < 0 || action >= board.Points {
		return "-"
	}
	return fmt.Sprintf("(%d,%d)", action/board.Size, action%board.Size)
}

func colorName(color int) string {
	if color == board.Black {
		return "black"
	}
	return "white"
}

func summaryFromStats(stats Stats) Summary {
	valid := stats.ValidGames()
	var rateA, rateB float64
	if valid > 0 {
		rateA = float64(stats.ModelAWins) / float64(valid)
		rateB = float64(stats.ModelBWins) / float64(valid)
	}
	return Summary{
		ModelAWins: stats.ModelAWins, ModelBWins: stats.ModelBWins,
		ModelAWinRate: rateA, ModelBWinRate: rateB, ValidGames: valid,
		Completed: stats.Completed, MaxMoves: stats.MaxMoves, Failed: stats.FailedGames(),
	}
}
