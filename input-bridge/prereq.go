//go:build windows

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func inputDriverReady() bool {
	if usbipServiceRunning() {
		return true
	}

	sysDrivers := filepath.Join(os.Getenv("SystemRoot"), "System32", "drivers", "usbip2_ude.sys")
	if _, err := os.Stat(sysDrivers); err == nil {
		return true
	}

	pattern := filepath.Join(os.Getenv("SystemRoot"), "System32", "DriverStore", "FileRepository", "usbip2_ude.inf_*", "usbip2_ude.sys")
	matches, err := filepath.Glob(pattern)
	return err == nil && len(matches) > 0
}

func usbipServiceRunning() bool {
	out, err := exec.Command("sc", "query", "usbip2_ude").CombinedOutput()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), "RUNNING")
}
