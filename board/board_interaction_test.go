package board

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
)

// printBoard 打印当前棋盘。
func printBoard(b *Board) {
	fmt.Print("   ")
	for y := range Size {
		fmt.Printf("%2d", y)
	}
	fmt.Println()

	for x := range Size {
		fmt.Printf("%2d ", x)
		for y := range Size {
			p := x*Size + y
			switch b.boards[0][p] {
			case Black:
				fmt.Print("● ")
			case White:
				fmt.Print("○ ")
			default:
				fmt.Print("· ")
			}
		}
		fmt.Printf("%2d\n", x)
	}

	fmt.Print("   ")
	for y := range Size {
		fmt.Printf("%2d", y)
	}
	fmt.Println()
}

// currentPlayer 返回当前行动方名称。
func currentPlayer(b *Board) string {
	if b.round%2 == 0 {
		return "黑棋 (●)"
	}
	return "白棋 (○)"
}

// TestInteractivePlay 双方轮流下棋的交互式程序。
// 运行方式: go test -v -run TestInteractivePlay -timeout 0
// 输入格式:
//
//	落子: x y   (例如 "3 4" 表示第 3 行第 4 列)
//	停着: pass 或 -1 -1
//	退出: quit 或 exit
func TestInteractivePlay(t *testing.T) {
	b := New()

	// go test 默认不把 stdin 连到终端，显式打开 /dev/tty 读取键盘输入
	tty, err := os.Open("/dev/tty")
	if err != nil {
		t.Fatalf("无法打开终端: %v", err)
	}
	defer tty.Close()
	scanner := bufio.NewScanner(tty)

	fmt.Println("=== 围棋交互式对局 ===")
	fmt.Println("输入格式: 行 列 (例如 3 4), 或 pass 停着, 或 quit 退出")
	fmt.Println()

	printBoard(b)

	for b.IsFinish() == 0 {
		fmt.Printf("\n第 %d 手, %s 请走棋: ", b.round+1, currentPlayer(b))

		if !scanner.Scan() {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// 处理退出
		if line == "quit" || line == "exit" {
			fmt.Println("对局结束。")
			return
		}

		// 处理 pass
		if line == "pass" || line == "-1 -1" || line == "-1,-1" {
			ret := b.Move(-1, -1)
			if ret != 0 {
				fmt.Println("❌ 停着失败！")
				continue
			}
			justPassed := currentColor(b.round - 1)
			justPassedName := "黑棋 (●)"
			if justPassed == White {
				justPassedName = "白棋 (○)"
			}
			fmt.Printf("✓ %s 停着\n", justPassedName)
			printBoard(b)
			continue
		}

		// 解析坐标
		parts := strings.Fields(line)
		if len(parts) != 2 {
			fmt.Println("❌ 格式错误，请输入两个数字: 行 列")
			continue
		}

		x, err1 := strconv.Atoi(parts[0])
		y, err2 := strconv.Atoi(parts[1])
		if err1 != nil || err2 != nil {
			fmt.Println("❌ 坐标无效，请输入整数")
			continue
		}

		// 执行落子
		ret := b.Move(x, y)
		if ret != 0 {
			fmt.Printf("❌ (%d, %d) 非法落子！\n", x, y)
			continue
		}

		justMoved := currentColor(b.round - 1)
		justMovedName := "黑棋 (●)"
		if justMoved == White {
			justMovedName = "白棋 (○)"
		}
		fmt.Printf("✓ %s 下在 (%d, %d)\n", justMovedName, x, y)
		printBoard(b)
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "读取输入出错: %v\n", err)
	}

	if b.IsFinish() == 1 {
		fmt.Println("\n=== 对局结束（连续两手停着）===")
		score := b.Score()
		fmt.Printf("黑棋: %d 目 | 白棋: %d 目 | 中立: %d 目\n", score[0], score[1], score[2])
	}
}
