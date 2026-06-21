package main

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"go-ai-infer/inference"
	"go-ai-infer/match"
)

const (
	matchLatencyWindowSize    = 2048
	matchGameCellWidth        = 42
	matchDefaultTerminalWidth = 80
)

type matchGameDisplay struct {
	status     string
	moves      int
	blackModel string
	whiteModel string
	winner     string
	blackLead  float32
}

type matchBatchDisplay struct {
	capacity   int
	count      int64
	sizeSum    int64
	full       int64
	errors     int64
	latencySum time.Duration
	latencies  []time.Duration
	next       int
}

type matchDashboard struct {
	mu sync.Mutex

	out         io.Writer
	interactive bool
	modelA      string
	modelB      string
	games       []matchGameDisplay
	startedAt   time.Time
	batchA      matchBatchDisplay
	batchB      matchBatchDisplay
	rendered    int

	stop chan struct{}
	done chan struct{}
}

func newMatchDashboard(out *os.File, games int, modelA, modelB string, batchCapacity int) *matchDashboard {
	info, err := out.Stat()
	interactive := err == nil && info.Mode()&os.ModeCharDevice != 0
	gameStates := make([]matchGameDisplay, games)
	for i := range gameStates {
		gameStates[i].status = "waiting"
	}
	return &matchDashboard{
		out: out, interactive: interactive, modelA: modelA, modelB: modelB,
		games: gameStates, startedAt: time.Now(),
		batchA: newMatchBatchDisplay(batchCapacity),
		batchB: newMatchBatchDisplay(batchCapacity),
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
	}
}

func newMatchBatchDisplay(capacity int) matchBatchDisplay {
	return matchBatchDisplay{
		capacity:  capacity,
		latencies: make([]time.Duration, 0, matchLatencyWindowSize),
	}
}

func (d *matchDashboard) Start() {
	go d.run()
}

func (d *matchDashboard) Stop() {
	close(d.stop)
	<-d.done

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.interactive {
		d.clearLocked()
		d.renderLocked(true)
	}
}

func (d *matchDashboard) OnGameEvent(event match.GameEvent) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if event.Game < 1 || event.Game > len(d.games) {
		return
	}

	d.games[event.Game-1] = matchGameDisplay{
		status: event.Status, moves: event.Moves,
		blackModel: event.BlackModel, whiteModel: event.WhiteModel,
		winner: event.Winner, blackLead: event.BlackLead,
	}

	if !d.interactive && isFinishedMatchStatus(event.Status) {
		if event.Winner != "" {
			fmt.Fprintf(
				d.out,
				"Game #%03d finished: %s(black) vs %s(white), winner=%s, black_lead=%.1f, moves=%d, status=%s\n",
				event.Game, event.BlackModel, event.WhiteModel, event.Winner,
				event.BlackLead, event.Moves, event.Status,
			)
		} else {
			fmt.Fprintf(
				d.out,
				"Game #%03d failed: %s(black) vs %s(white), moves=%d, status=%s, error=%v\n",
				event.Game, event.BlackModel, event.WhiteModel, event.Moves, event.Status, event.Err,
			)
		}
	}
}

func (d *matchDashboard) OnBatchA(event inference.BatchEvent) {
	d.onBatch(&d.batchA, event)
}

func (d *matchDashboard) OnBatchB(event inference.BatchEvent) {
	d.onBatch(&d.batchB, event)
}

func (d *matchDashboard) onBatch(batch *matchBatchDisplay, event inference.BatchEvent) {
	d.mu.Lock()
	defer d.mu.Unlock()
	batch.count++
	batch.sizeSum += int64(event.Size)
	if event.Size == event.Capacity {
		batch.full++
	}
	if event.Err != nil {
		batch.errors++
	}
	batch.latencySum += event.Duration
	if len(batch.latencies) < matchLatencyWindowSize {
		batch.latencies = append(batch.latencies, event.Duration)
		return
	}
	batch.latencies[batch.next] = event.Duration
	batch.next = (batch.next + 1) % matchLatencyWindowSize
}

func (d *matchDashboard) run() {
	defer close(d.done)
	interval := dashboardTextSummaryInterval
	if d.interactive {
		interval = dashboardRefreshInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	if d.interactive {
		d.mu.Lock()
		d.renderLocked(false)
		d.mu.Unlock()
	}

	for {
		select {
		case <-ticker.C:
			d.mu.Lock()
			if d.interactive {
				d.clearLocked()
				d.renderLocked(false)
			} else {
				fmt.Fprintln(d.out, d.matchSummaryLocked())
				fmt.Fprintln(d.out, d.batchSummaryLocked(d.modelA, &d.batchA))
				fmt.Fprintln(d.out, d.batchSummaryLocked(d.modelB, &d.batchB))
			}
			d.mu.Unlock()
		case <-d.stop:
			return
		}
	}
}

func (d *matchDashboard) renderLocked(final bool) {
	lines := []string{
		d.matchSummaryLocked(),
		d.batchSummaryLocked(d.modelA, &d.batchA),
		d.batchSummaryLocked(d.modelB, &d.batchB),
		"",
	}
	lines = append(lines, d.gameLinesLocked(matchTerminalWidth(d.out))...)
	for _, line := range lines {
		fmt.Fprintf(d.out, "%s\x1b[K\n", line)
	}
	d.rendered = len(lines)
	if !final {
		fmt.Fprint(d.out, "\x1b[K")
	}
}

func (d *matchDashboard) clearLocked() {
	if d.rendered == 0 {
		return
	}
	fmt.Fprintf(d.out, "\x1b[%dA", d.rendered)
	for i := 0; i < d.rendered; i++ {
		fmt.Fprint(d.out, "\r\x1b[2K")
		if i < d.rendered-1 {
			fmt.Fprint(d.out, "\x1b[1B")
		}
	}
	if d.rendered > 1 {
		fmt.Fprintf(d.out, "\x1b[%dA", d.rendered-1)
	}
	fmt.Fprint(d.out, "\r")
	d.rendered = 0
}

func (d *matchDashboard) matchSummaryLocked() string {
	var running, finished, failed, winsA, winsB int
	for _, game := range d.games {
		switch game.status {
		case "running":
			running++
		case "completed", "max_moves":
			finished++
			if game.winner == d.modelA {
				winsA++
			} else if game.winner == d.modelB {
				winsB++
			}
		case "waiting":
		default:
			finished++
			failed++
		}
	}
	return fmt.Sprintf(
		"Match  %d/%d | 运行 %d | %s胜 %d | %s胜 %d | 失败 %d | 耗时 %s",
		finished, len(d.games), running, d.modelA, winsA, d.modelB, winsB,
		failed, matchFormatDuration(time.Since(d.startedAt)),
	)
}

func (d *matchDashboard) gameLinesLocked(width int) []string {
	columns := width / matchGameCellWidth
	if columns < 1 {
		columns = 1
	}
	lines := make([]string, 0, (len(d.games)+columns-1)/columns)
	for start := 0; start < len(d.games); start += columns {
		end := start + columns
		if end > len(d.games) {
			end = len(d.games)
		}
		cells := make([]string, 0, end-start)
		for i := start; i < end; i++ {
			game := d.games[i]
			cell := fmt.Sprintf(
				"#%03d %s %d步 %s黑/%s白",
				i+1, matchDisplayStatus(game.status), game.moves,
				shortModelName(game.blackModel), shortModelName(game.whiteModel),
			)
			if game.winner != "" {
				cell += fmt.Sprintf(" %s胜", shortModelName(game.winner))
			}
			if i < end-1 {
				cell = matchPadDisplayWidth(cell, matchGameCellWidth-3)
			}
			cells = append(cells, cell)
		}
		lines = append(lines, strings.Join(cells, " | "))
	}
	return lines
}

func (d *matchDashboard) batchSummaryLocked(model string, batch *matchBatchDisplay) string {
	if batch.count == 0 {
		return fmt.Sprintf(
			"Batch %-10s 平均 --/%d | 满批 -- | 推理 avg -- p95 -- | 错误 0",
			model, batch.capacity,
		)
	}
	averageSize := float64(batch.sizeSum) / float64(batch.count)
	fullRate := 100 * float64(batch.full) / float64(batch.count)
	averageLatency := batch.latencySum / time.Duration(batch.count)
	return fmt.Sprintf(
		"Batch %-10s 平均 %.1f/%d | 满批 %.1f%% | 推理 avg %s p95 %s | 错误 %d",
		model, averageSize, batch.capacity, fullRate,
		matchFormatLatency(averageLatency), matchFormatLatency(matchPercentile95(batch.latencies)),
		batch.errors,
	)
}

func isFinishedMatchStatus(status string) bool {
	switch status {
	case "completed", "max_moves", "search_failed", "illegal_action", "canceled":
		return true
	default:
		return false
	}
}

func matchDisplayStatus(status string) string {
	switch status {
	case "waiting":
		return "等待"
	case "running":
		return "运行中"
	case "completed":
		return "已结束"
	case "max_moves":
		return "步数上限"
	case "search_failed":
		return "搜索失败"
	case "illegal_action":
		return "非法动作"
	case "canceled":
		return "已取消"
	default:
		return strings.ReplaceAll(status, "_", " ")
	}
}

func shortModelName(name string) string {
	if name == "" {
		return "-"
	}
	const limit = 12
	runes := []rune(name)
	if len(runes) <= limit {
		return name
	}
	return string(runes[:limit-1]) + "…"
}

func matchPercentile95(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := (95*len(sorted) + 99) / 100
	return sorted[index-1]
}

func matchFormatDuration(duration time.Duration) string {
	duration = duration.Round(time.Second)
	if duration < time.Minute {
		return duration.String()
	}
	hours := int(duration / time.Hour)
	minutes := int(duration % time.Hour / time.Minute)
	seconds := int(duration % time.Minute / time.Second)
	if hours > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", hours, minutes, seconds)
	}
	return fmt.Sprintf("%dm%02ds", minutes, seconds)
}

func matchFormatLatency(duration time.Duration) string {
	if duration < time.Millisecond {
		return duration.Round(time.Microsecond).String()
	}
	return duration.Round(100 * time.Microsecond).String()
}

func matchPadDisplayWidth(value string, width int) string {
	padding := width - matchDisplayWidth(value)
	if padding <= 0 {
		return value
	}
	return value + strings.Repeat(" ", padding)
}

func matchDisplayWidth(value string) int {
	width := 0
	for _, r := range value {
		if r >= 0x2E80 {
			width += 2
		} else {
			width++
		}
	}
	return width
}

func matchTerminalWidth(out io.Writer) int {
	file, ok := out.(*os.File)
	if !ok {
		return matchDefaultTerminalWidth
	}
	type winsize struct {
		rows    uint16
		columns uint16
		x       uint16
		y       uint16
	}
	size := winsize{}
	_, _, errno := syscall.Syscall(
		syscall.SYS_IOCTL,
		file.Fd(),
		uintptr(syscall.TIOCGWINSZ),
		uintptr(unsafe.Pointer(&size)),
	)
	if errno != 0 || size.columns == 0 {
		return matchDefaultTerminalWidth
	}
	return int(size.columns)
}
