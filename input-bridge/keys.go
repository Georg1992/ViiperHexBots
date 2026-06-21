//go:build windows

package main

import (
	"github.com/Alia5/VIIPER/device/keyboard"
	"golang.org/x/sys/windows"
)

const (
	mapVKVscToVK   = 1
	mapVKVscToVKEx = 3
)

var procMapVirtualKey = windows.NewLazySystemDLL("user32.dll").NewProc("MapVirtualKeyW")

func scanCodeToVK(scanCode uint16) uint16 {
	if vk := mapScanCode(scanCode, mapVKVscToVK); vk != 0 {
		return vk
	}

	param := uintptr(scanCode) << 16
	if scanCode >= 0x100 {
		param |= 1 << 24
	}
	ret, _, _ := procMapVirtualKey.Call(param, uintptr(mapVKVscToVKEx))
	return uint16(ret)
}

func mapScanCode(scanCode uint16, mapType uint32) uint16 {
	ret, _, _ := procMapVirtualKey.Call(uintptr(scanCode), uintptr(mapType))
	return uint16(ret)
}

func vkToHID(vk uint16) (uint8, bool) {
	if hid, ok := vkToHIDTable[vk]; ok {
		return hid, true
	}
	return 0, false
}

func vkToModifier(vk uint16) (uint8, bool) {
	switch vk {
	case 0xA0:
		return keyboard.ModLeftShift, true
	case 0xA1:
		return keyboard.ModRightShift, true
	case 0xA2:
		return keyboard.ModLeftCtrl, true
	case 0xA3:
		return keyboard.ModRightCtrl, true
	case 0xA4:
		return keyboard.ModLeftAlt, true
	case 0xA5:
		return keyboard.ModRightAlt, true
	case 0x5B, 0x5C:
		return keyboard.ModLeftGUI, true
	default:
		return 0, false
	}
}

var vkToHIDTable = map[uint16]uint8{
	0x41: keyboard.KeyA, 0x42: keyboard.KeyB, 0x43: keyboard.KeyC, 0x44: keyboard.KeyD,
	0x45: keyboard.KeyE, 0x46: keyboard.KeyF, 0x47: keyboard.KeyG, 0x48: keyboard.KeyH,
	0x49: keyboard.KeyI, 0x4A: keyboard.KeyJ, 0x4B: keyboard.KeyK, 0x4C: keyboard.KeyL,
	0x4D: keyboard.KeyM, 0x4E: keyboard.KeyN, 0x4F: keyboard.KeyO, 0x50: keyboard.KeyP,
	0x51: keyboard.KeyQ, 0x52: keyboard.KeyR, 0x53: keyboard.KeyS, 0x54: keyboard.KeyT,
	0x55: keyboard.KeyU, 0x56: keyboard.KeyV, 0x57: keyboard.KeyW, 0x58: keyboard.KeyX,
	0x59: keyboard.KeyY, 0x5A: keyboard.KeyZ,
	0x30: keyboard.Key0, 0x31: keyboard.Key1, 0x32: keyboard.Key2, 0x33: keyboard.Key3,
	0x34: keyboard.Key4, 0x35: keyboard.Key5, 0x36: keyboard.Key6, 0x37: keyboard.Key7,
	0x38: keyboard.Key8, 0x39: keyboard.Key9,
	0x20: keyboard.KeySpace,
	0x0D: keyboard.KeyEnter,
	0x08: keyboard.KeyBackspace,
	0x09: keyboard.KeyTab,
	0x1B: keyboard.KeyEscape,
	0x25: keyboard.KeyLeft, 0x26: keyboard.KeyUp,
	0x27: keyboard.KeyRight, 0x28: keyboard.KeyDown,
	0x2D: keyboard.KeyInsert, 0x2E: keyboard.KeyDelete,
	0x24: keyboard.KeyHome, 0x23: keyboard.KeyEnd,
	0x21: keyboard.KeyPageUp, 0x22: keyboard.KeyPageDown,
	0x70: keyboard.KeyF1, 0x71: keyboard.KeyF2, 0x72: keyboard.KeyF3, 0x73: keyboard.KeyF4,
	0x74: keyboard.KeyF5, 0x75: keyboard.KeyF6, 0x76: keyboard.KeyF7, 0x77: keyboard.KeyF8,
	0x78: keyboard.KeyF9, 0x79: keyboard.KeyF10, 0x7A: keyboard.KeyF11, 0x7B: keyboard.KeyF12,
	0xBA: keyboard.KeySemicolon, 0xBB: keyboard.KeyEqual, 0xBC: keyboard.KeyComma,
	0xBD: keyboard.KeyMinus, 0xBE: keyboard.KeyPeriod, 0xBF: keyboard.KeySlash,
	0xC0: keyboard.KeyGrave, 0xDB: keyboard.KeyLeftBrace, 0xDC: keyboard.KeyBackslash,
	0xDD: keyboard.KeyRightBrace, 0xDE: keyboard.KeyApostrophe,
	0x60: keyboard.KeyKp0, 0x61: keyboard.KeyKp1, 0x62: keyboard.KeyKp2, 0x63: keyboard.KeyKp3,
	0x64: keyboard.KeyKp4, 0x65: keyboard.KeyKp5, 0x66: keyboard.KeyKp6, 0x67: keyboard.KeyKp7,
	0x68: keyboard.KeyKp8, 0x69: keyboard.KeyKp9,
	0x6A: keyboard.KeyKpAsterisk, 0x6B: keyboard.KeyKpPlus, 0x6D: keyboard.KeyKpMinus,
	0x6E: keyboard.KeyKpDot, 0x6F: keyboard.KeyKpSlash,
}
