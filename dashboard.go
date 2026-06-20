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
	"go-ai-infer/runner"
)

const (
	dashboardRefreshInterval = 500 * time.Millisecond
	textSummaryInterval      = 10 * time.Second
	latencyWindowSize        = 2048
	gameCellWidth            = 24
	defaultTerminalWidth     = 80
)

type gameDisplay struct {
	status string
	moves  int
}

type dashboard struct {
	mu sync.Mutex

	out         io.Writer
	interactive bool
	games       []gameDisplay
	startedAt   time.Time

	batchCapacity int
	batchCount    int64
	batchSizeSum  int64
	fullBatches   int64
	batchErrors   int64
	latencySum    time.Duration
	latencies     []time.Duration
	latencyNext   int
	renderedLines int

	stop chan struct{}
	done chan struct{}
}

func newDashboard(out *os.File, games, batchCapacity int) *dashboard {
	info, err := out.Stat()
	interactive := err == nil && info.Mode()&os.ModeCharDevice != 0

	gameStates := make([]gameDisplay, games)
	for i := range gameStates {
		gameStates[i].status = "waiting"
	}

	return &dashboard{
		out:           out,
		interactive:   interactive,
		games:         gameStates,
		startedAt:     time.Now(),
		batchCapacity: batchCapacity,
		latencies:     make([]time.Duration, 0, latencyWindowSize),
		stop:          make(chan struct{}),
		done:          make(chan struct{}),
	}
}

func (d *dashboard) Start() {
	go d.run()
}

func (d *dashboard) Stop(stats runner.Stats) {
	close(d.stop)
	<-d.done

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.interactive {
		d.clearLocked()
		d.renderLocked(true)
	}
	fmt.Fprintf(
		d.out,
		"Finished  requested=%d saved=%d failed=%d samples=%d duration=%s\n",
		stats.Requested,
		stats.Saved,
		stats.SaveFailed+stats.MaxMoves+stats.SearchFailed+stats.IllegalAction+stats.Canceled,
		stats.Samples,
		formatDuration(stats.Duration),
	)
	fmt.Fprintln(d.out, d.batchSummaryLocked())
}

func (d *dashboard) OnGameEvent(event runner.GameEvent) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if event.Game < 1 || event.Game > len(d.games) {
		return
	}
	d.games[event.Game-1] = gameDisplay{status: event.Status, moves: event.Moves}
}

func (d *dashboard) OnBatchEvent(event inference.BatchEvent) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.batchCount++
	d.batchSizeSum += int64(event.Size)
	if event.Size == event.Capacity {
		d.fullBatches++
	}
	if event.Err != nil {
		d.batchErrors++
	}
	d.latencySum += event.Duration

	if len(d.latencies) < latencyWindowSize {
		d.latencies = append(d.latencies, event.Duration)
		return
	}
	d.latencies[d.latencyNext] = event.Duration
	d.latencyNext = (d.latencyNext + 1) % latencyWindowSize
}

func (d *dashboard) run() {
	defer close(d.done)

	interval := textSummaryInterval
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
				fmt.Fprintf(d.out, "%s | %s\n", d.selfplaySummaryLocked(), d.batchSummaryLocked())
			}
			d.mu.Unlock()
		case <-d.stop:
			return
		}
	}
}

func (d *dashboard) renderLocked(final bool) {
	lines := []string{
		d.selfplaySummaryLocked(),
		d.batchSummaryLocked(),
		"",
	}
	lines = append(lines, d.gameLinesLocked(terminalWidth(d.out))...)

	for _, line := range lines {
		fmt.Fprintf(d.out, "%s\x1b[K\n", line)
	}
	d.renderedLines = len(lines)
	if !final {
		fmt.Fprint(d.out, "\x1b[K")
	}
}

func (d *dashboard) clearLocked() {
	if d.renderedLines == 0 {
		return
	}
	fmt.Fprintf(d.out, "\x1b[%dA", d.renderedLines)
	for i := 0; i < d.renderedLines; i++ {
		fmt.Fprint(d.out, "\r\x1b[2K")
		if i < d.renderedLines-1 {
			fmt.Fprint(d.out, "\x1b[1B")
		}
	}
	if d.renderedLines > 1 {
		fmt.Fprintf(d.out, "\x1b[%dA", d.renderedLines-1)
	}
	fmt.Fprint(d.out, "\r")
	d.renderedLines = 0
}

func (d *dashboard) gameLinesLocked(width int) []string {
	columns := width / gameCellWidth
	if columns < 1 {
		columns = 1
	}

	rows := (len(d.games) + columns - 1) / columns
	lines := make([]string, 0, rows)
	for start := 0; start < len(d.games); start += columns {
		end := start + columns
		if end > len(d.games) {
			end = len(d.games)
		}

		cells := make([]string, 0, end-start)
		for i := start; i < end; i++ {
			game := d.games[i]
			cell := fmt.Sprintf("#%03d %s %d步", i+1, displayStatus(game.status), game.moves)
			if i < end-1 {
				cell = padDisplayWidth(cell, gameCellWidth-3)
			}
			cells = append(cells, cell)
		}
		lines = append(lines, strings.Join(cells, " | "))
	}
	return lines
}

func (d *dashboard) selfplaySummaryLocked() string {
	var running, saved, failed int
	for _, game := range d.games {
		switch game.status {
		case "running", "saving":
			running++
		case "completed":
			saved++
		case "waiting":
		default:
			failed++
		}
	}
	finished := saved + failed
	return fmt.Sprintf(
		"Selfplay  %d/%d | 运行 %d | 已保存 %d | 失败 %d | 耗时 %s",
		finished,
		len(d.games),
		running,
		saved,
		failed,
		formatDuration(time.Since(d.startedAt)),
	)
}

func (d *dashboard) batchSummaryLocked() string {
	if d.batchCount == 0 {
		return fmt.Sprintf(
			"Batch     平均 --/%d | 满批 -- | 推理 avg -- p95 -- | 错误 0",
			d.batchCapacity,
		)
	}

	averageSize := float64(d.batchSizeSum) / float64(d.batchCount)
	fullRate := 100 * float64(d.fullBatches) / float64(d.batchCount)
	averageLatency := d.latencySum / time.Duration(d.batchCount)
	return fmt.Sprintf(
		"Batch     平均 %.1f/%d | 满批 %.1f%% | 推理 avg %s p95 %s | 错误 %d",
		averageSize,
		d.batchCapacity,
		fullRate,
		formatLatency(averageLatency),
		formatLatency(percentile95(d.latencies)),
		d.batchErrors,
	)
}

func percentile95(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := (95*len(sorted) + 99) / 100
	return sorted[index-1]
}

func displayStatus(status string) string {
	switch status {
	case "waiting":
		return "等待"
	case "running":
		return "运行中"
	case "saving":
		return "保存中"
	case "completed":
		return "已保存"
	case "save_failed":
		return "保存失败"
	case "max_moves":
		return "达到步数上限"
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

func formatDuration(duration time.Duration) string {
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

func formatLatency(duration time.Duration) string {
	if duration < time.Millisecond {
		return duration.Round(time.Microsecond).String()
	}
	return duration.Round(100 * time.Microsecond).String()
}

func padDisplayWidth(value string, width int) string {
	padding := width - displayWidth(value)
	if padding <= 0 {
		return value
	}
	return value + strings.Repeat(" ", padding)
}

func displayWidth(value string) int {
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

func terminalWidth(out io.Writer) int {
	file, ok := out.(*os.File)
	if !ok {
		return defaultTerminalWidth
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
		return defaultTerminalWidth
	}
	return int(size.columns)
}
