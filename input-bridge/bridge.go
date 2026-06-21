//go:build windows

package main

import (
	"bufio"
	"context"
	_ "embed"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/Alia5/VIIPER/device/keyboard"
	"github.com/Alia5/VIIPER/device/mouse"
	"github.com/Alia5/VIIPER/viiperclient"
	"golang.org/x/sys/windows"
)

//go:embed embed/viiper.exe
var viiperBin []byte

const (
	viiperAPIAddr    = "localhost:3242"
	serverWaitTime   = 30 * time.Second
	serverPollPeriod = 200 * time.Millisecond
)

var (
	serverMu      sync.Mutex
	serverCmd     *exec.Cmd
	serverStarted bool
	serverPID     int
	viiperTempDir string
)

func ensureViiperServer() error {
	serverMu.Lock()
	defer serverMu.Unlock()

	api := viiperclient.New(viiperAPIAddr)
	if _, err := api.PingCtx(context.Background()); err == nil {
		return nil
	}

	path, dir, err := extractViiper()
	if err != nil {
		return err
	}
	viiperTempDir = dir

	cmd := exec.Command(path, "server")
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NO_WINDOW}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start server: %w", err)
	}
	serverPID = cmd.Process.Pid

	go discardOutput(stdout)
	go discardOutput(stderr)

	if err := waitForViiperServer(viiperAPIAddr, serverWaitTime); err != nil {
		killProcessTree(serverPID)
		_, _ = cmd.Process.Wait()
		serverPID = 0
		removeViiperTempDir()
		return err
	}

	serverCmd = cmd
	serverStarted = true
	return nil
}

func stopViiperServer() {
	serverMu.Lock()
	pid := serverPID
	started := serverStarted
	cmd := serverCmd
	serverPID = 0
	serverStarted = false
	serverCmd = nil
	dir := viiperTempDir
	viiperTempDir = ""
	serverMu.Unlock()

	if !started || pid <= 0 {
		return
	}

	killProcessTree(pid)
	if cmd != nil && cmd.Process != nil {
		_, _ = cmd.Process.Wait()
	}
	removeViiperTempDirPath(dir)
}

func discardOutput(r io.Reader) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
	}
}

func killProcessTree(pid int) {
	if pid <= 0 {
		return
	}
	_ = exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F").Run()
}

func removeViiperTempDir() {
	serverMu.Lock()
	dir := viiperTempDir
	viiperTempDir = ""
	serverMu.Unlock()
	removeViiperTempDirPath(dir)
}

func removeViiperTempDirPath(dir string) {
	if dir == "" {
		return
	}
	_ = os.RemoveAll(dir)
}

func extractViiper() (string, string, error) {
	dir, err := os.MkdirTemp("", "viiper-hexbots-*")
	if err != nil {
		return "", "", fmt.Errorf("create temp dir: %w", err)
	}
	path := filepath.Join(dir, "viiper.exe")
	if err := os.WriteFile(path, viiperBin, 0o755); err != nil {
		_ = os.RemoveAll(dir)
		return "", "", fmt.Errorf("write viiper.exe: %w", err)
	}
	return path, dir, nil
}

func waitForViiperServer(addr string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	api := viiperclient.New(addr)

	for time.Now().Before(deadline) {
		if _, err := api.PingCtx(context.Background()); err == nil {
			return nil
		}
		time.Sleep(serverPollPeriod)
	}
	return fmt.Errorf("VIIPER server ping timed out after %s", timeout)
}

type inputBridge struct {
	mu sync.Mutex

	api         *viiperclient.Client
	keyStream   *viiperclient.DeviceStream
	mouseStream *viiperclient.DeviceStream
	busID       uint32
	createdBus  bool

	modifiers  uint8
	mouseState mouse.InputState
}

func newInputBridge() *inputBridge {
	return &inputBridge{}
}

func (b *inputBridge) ready() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.keyStream != nil && b.mouseStream != nil
}

func (b *inputBridge) init(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.keyStream != nil && b.mouseStream != nil {
		return nil
	}

	if err := ensureViiperServer(); err != nil {
		return err
	}

	api := viiperclient.New(viiperAPIAddr)
	busID, createdBus, err := ensureBus(ctx, api)
	if err != nil {
		return err
	}

	keyStream, _, err := api.AddDeviceAndConnect(ctx, busID, "keyboard", nil)
	if err != nil {
		if createdBus {
			_, _ = api.BusRemoveCtx(ctx, busID)
		}
		return fmt.Errorf("keyboard setup: %w", err)
	}

	mouseStream, _, err := api.AddDeviceAndConnect(ctx, busID, "mouse", nil)
	if err != nil {
		_ = keyStream.Close()
		if createdBus {
			_, _ = api.BusRemoveCtx(ctx, busID)
		}
		return fmt.Errorf("mouse setup: %w", err)
	}

	b.api = api
	b.keyStream = keyStream
	b.mouseStream = mouseStream
	b.busID = busID
	b.createdBus = createdBus
	return nil
}

func (b *inputBridge) shutdown(ctx context.Context) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.keyStream != nil {
		_ = b.keyStream.Close()
		b.keyStream = nil
	}
	if b.mouseStream != nil {
		_ = b.mouseStream.Close()
		b.mouseStream = nil
	}
	if b.api != nil && b.createdBus {
		_, _ = b.api.BusRemoveCtx(ctx, b.busID)
	}
	b.api = nil
	b.modifiers = 0
	b.mouseState = mouse.InputState{}
}

func (b *inputBridge) sendKey(scanCode uint16, down bool) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.keyStream == nil {
		return fmt.Errorf("input bridge not initialized")
	}

	vk := scanCodeToVK(scanCode)
	if mod, ok := vkToModifier(vk); ok {
		if down {
			b.modifiers |= mod
		} else {
			b.modifiers &^= mod
		}
		state := keyboard.InputState{Modifiers: b.modifiers}
		return b.keyStream.WriteBinary(&state)
	}

	hid, ok := vkToHID(vk)
	if !ok {
		return fmt.Errorf("unsupported scan code %d (vk 0x%02X)", scanCode, vk)
	}

	if down {
		press := keyboard.PressKeyWithMod(b.modifiers, hid)
		return b.keyStream.WriteBinary(&press)
	}

	state := keyboard.InputState{Modifiers: b.modifiers}
	return b.keyStream.WriteBinary(&state)
}

func (b *inputBridge) sendMouseButton(button int, down bool) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.mouseStream == nil {
		return fmt.Errorf("input bridge not initialized")
	}

	if button == 5 {
		if !down {
			return nil
		}
		state := b.mouseState
		state.Wheel = -1
		err := b.mouseStream.WriteBinary(&state)
		state.Wheel = 0
		return err
	}

	flag, ok := ahiButtonToMouseFlag(button)
	if !ok {
		return fmt.Errorf("unsupported mouse button %d", button)
	}

	if down {
		b.mouseState.Buttons |= flag
	} else {
		b.mouseState.Buttons &^= flag
	}

	return b.mouseStream.WriteBinary(&b.mouseState)
}

func ahiButtonToMouseFlag(button int) (uint8, bool) {
	switch button {
	case 0:
		return mouse.BtnLeft, true
	case 1:
		return mouse.BtnRight, true
	case 2:
		return mouse.BtnMiddle, true
	case 3:
		return mouse.BtnBack, true
	case 4:
		return mouse.BtnForward, true
	default:
		return 0, false
	}
}

func ensureBus(ctx context.Context, api *viiperclient.Client) (uint32, bool, error) {
	busesResp, err := api.BusListCtx(ctx)
	if err != nil {
		return 0, false, err
	}

	if len(busesResp.Buses) > 0 {
		busID := busesResp.Buses[0]
		for _, bus := range busesResp.Buses[1:] {
			if bus < busID {
				busID = bus
			}
		}

		devices, err := api.DevicesListCtx(ctx, busID)
		if err == nil {
			for _, dev := range devices.Devices {
				_, _ = api.DeviceRemoveCtx(ctx, busID, dev.DevID)
			}
		}

		return busID, false, nil
	}

	resp, err := api.BusCreateCtx(ctx, 0)
	if err != nil {
		return 0, false, err
	}
	return resp.BusID, true, nil
}
