#Persistent
#SingleInstance Force
#KeyHistory 0
#InstallKeybdHook
#InstallMouseHook
#include Lib\ViiperInput.ahk
#include MobData.ahk
#include BotLogic.ahk
#include MemoryOperations.ahk
#include utilityFunctions.ahk
#MaxThreadsPerHotkey 2

SetBatchLines, -1
ListLines, Off
SendMode Input
SetKeyDelay, 30, 30
SetMouseDelay, 30

; Global coordinator modes (affects all included files)
CoordMode, Mouse, Screen
CoordMode, Pixel, Screen

; ====== CORE BOT STATES ======
global botRunning := false
global botPaused := false

; ====== WINDOW MANAGEMENT ======
global gameWindowID := 0
global gameWindowTitle := ""
global gameProcess := ""
global windowIDs := {}
global titleToIndex := {}

; ====== CONFIG DEFAULTS ======
; Window Settings
global warperCoordsSet := false
global warperX := ""
global warperY := ""
global warperLocation := 0

; Sliders
global SearchRange := 16 ; Default search range (9-16)
global TimeOnLocation := 20 ; Default time in seconds
global Iterations := 0 ; Default iterations before Kafra
global WeightModifier := 49

; Checkboxes
global TakeFlyWings := 0 ; Default checked (true)
global wingsTaken := 100
global DetectCaptcha := 0 ; Default unchecked (false)

; Keybindings
global SkillButtonKey := "" ; Default attack key
global SkillDelay := 300 ; Default 500ms delay
global TeleportButtonKey := "" ; Default teleport key
global SavePointButtonKey := "" ; Default save point key
global SPButtonKey := "" ; Default SP item key
global OpenStorageButtonKey := "" ; Default storage key
global SkillTimerButtonKey := "" ; Default skill timer key
global SkillTimerInterval := 20 ; Default 20 seconds

; Monster Selection
global selectedMonsterIndex := 1 ; Default to first monster
global targetColor = MobColors[selectedMonsterIndex]
global SelectedMonster1, SelectedMonster2, SelectedMonster3, SelectedMonster4, SelectedMonster5, SelectedMonster6

; ====== VIIPER INPUT ======
global AHI := "" ; ViiperInput instance (AHI-compatible API)
global mouseId := 1
global keyboardId := 1

SetDefaultKeyboardLayout("00000409") ; English - US

if (!FileExist("config.ini")) {
    FileAppend, , config.ini ; Create empty file
}

global AHI := new ViiperInput()

; -------------------------------
; Load Config Values First
; -------------------------------
if (FileExist("config.ini")) {
    Gosub, LoadConfig
}

; --------------------------
; GUI Setup (With Loaded Values)
; --------------------------
Gui, Font, s10, Segoe UI

; Title
Gui, Add, Text, x10 y15 w700 h30 Center, ViiperHex Bot
Gui, Add, Text, x10 y50 w700 h2 0x10 ; Divider

; Window Selection
Gui, Add, Text, x20 y70 w280 h30, Select Game Window:
Gui, Add, DropDownList, x20 y100 w200 h25 r10 vSelectedWindow gOnWindowSelect, || 
Gui, Add, Button, x230 y100 w70 h25 gRefreshWindows, Refresh
Gui, Add, Text, x20 y130 w280 h25 vWindowInfo, % (gameWindowTitle ? gameWindowTitle : "No window selected")

; Bot Status
Gui, Add, Text, x600 y70 w150 h30 vBotStatus, Status: Off
Gui, Add, Progress, x600 y+3 w150 h12 cRed vStatusLight, 100

; Input backend
Gui, Add, Text, x600 y170 w150 h40 vInputStatus, Input: VIIPER
Gui, Add, Text, x600 y+15 w150 h40 vInputHint, Virtual HID devices

; Warper Coordinates Group - GUI Definition (unchanged)
Gui, Add, GroupBox, x600 y300 w150 h100, Warper Coordinates
Gui, Add, Button, x610 y320 w130 h30 gSetWarperCoords vSetWarperCoordsBtn, Set Warper Position
Gui, Add, Text, x610 y360 w130 h20 vWarperCoordsText, % warperCoordsSet ? "X: " warperX " Y: " warperY : "Not set"
Gui, Add, Button, x610 y390 w130 h25 gResetWarperCoords vResetWarperBtn, Reset Coordinates
GuiControl, % warperCoordsSet ? "Enable" : "Disable", ResetWarperBtn

; Time on Location Controls
Gui, Add, Text, x620 y450 w120 h50 vTimeOnLocationTextLabel, Time on Location (s):
Gui, Add, Slider, x600 y485 w150 h25 vTimeOnLocation gUpdateSliderValues Range20-240, %TimeOnLocation%
Gui, Add, Text, x760 y485 w30 h25 vTimeOnLocationValueText Center, %TimeOnLocation%
GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocationTextLabel
GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocation
GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocationValueText

; Set initial visibility based on warperCoordsSet
if (!warperCoordsSet) {
    GuiControl, Hide, TimeOnLocationTextLabel
    GuiControl, Hide, TimeOnLocation
    GuiControl, Hide, TimeOnLocationValueText
}

; Monster Type
monsterStartY := 170
Gui, Add, Text, x20 y%monsterStartY% w120 h30, Monster Type:
yPos := monsterStartY + 30

Loop % MobNames.MaxIndex() {
    ; Use the pre-loaded selectedMonsterIndex from config
    isChecked := (A_Index == selectedMonsterIndex) ? "Checked" : ""
        Gui, Add, Radio, x20 y%yPos% %isChecked% vSelectedMonster%A_Index% gUpdateGlobalsFromUI, % MobNames[A_Index]
        yPos += 35
    }

    ; Keybindings
    Gui, Add, Text, x160 y200 w130 h25, Attack Skill Button:
    Gui, Add, Hotkey, x300 yp w55 vSkillButtonKey gUpdateGlobalsFromUI, %SkillButtonKey%

    Gui, Add, Text, x360 yp+0 w100 h25, delay
    Gui, Add, Edit, x410 yp+0 w40 h25 vSkillDelay Number gUpdateGlobalsFromUI, %SkillDelay%
    Gui, Add, Text, x455 yp+0 w30 h25, (ms)

    Gui, Add, Text, x160 y+10 w130 h25, Teleport Button:
    Gui, Add, Hotkey, x300 yp w55 vTeleportButtonKey gUpdateGlobalsFromUI, %TeleportButtonKey%

    Gui, Add, Text, x160 y+10 w130 h25, To Save Point Button:
    Gui, Add, Hotkey, x300 yp w55 vSavePointButtonKey gUpdateGlobalsFromUI, %SavePointButtonKey%

    Gui, Add, Text, x160 y+10 w130 h25, Open Storage Button:
    Gui, Add, Hotkey, x300 yp w55 vOpenStorageButtonKey gUpdateGlobalsFromUI, %OpenStorageButtonKey%

    Gui, Add, Text, x160 y+10 w130 h25, Skill Timer Button:
    Gui, Add, Hotkey, x300 yp w55 vSkillTimerButtonKey gUpdateGlobalsFromUI, %SkillTimerButtonKey%

    Gui, Add, Text, x360 yp+0 w100 h25, every
    Gui, Add, Edit, x410 yp+0 w40 h25 vSkillTimerInterval Number gUpdateGlobalsFromUI, %SkillTimerInterval%
    Gui, Add, Text, x455 yp+0 w30 h25, (s)

    Gui, Add, Text, x160 y+10 w130 h25, SP Item Button:
    Gui, Add, Hotkey, x300 yp w55 vSPButtonKey gUpdateGlobalsFromUI, %SPButtonKey%

    ; Sliders
    inputStartY := yPos + 30

    ; Sliders with g-Labels
    Gui, Add, Text, x20 y%inputStartY% w120 h40, Search Range (9-16 Cells):
    Gui, Add, Slider, x150 yp w200 h25 vSearchRange gUpdateSliderValues Range9-16 TickInterval1 ToolTip, %SearchRange%
    Gui, Add, Text, x+5 yp w30 h25 vSearchRangeText Center, %SearchRange%

    Gui, Add, Text, x20 y+20 w120 h40, Items To Kafra when weight is:
    Gui, Add, Slider, x150 yp w200 h25 vWeightModifier gUpdateSliderValues Range49-90 TickInterval1 ToolTip, %WeightModifier%
    Gui, Add, Text, x+5 yp w30 h25 vWeightModifierText Center, % (WeightModifier = 49 ? "Off" : WeightModifier)
    Gui, Add, Text, x+1 yp w30 h25, `%
    ; Checkboxes
    Gui, Add, CheckBox, x20 y600 vTakeFlyWings gUpdateTakeFlyWings Checked%TakeFlyWings%, Take Fly Wings
    Gui, Add, Edit, x+10 yp-3 w50 vFlyWingsAmount Number Limit3 -WantReturn, %FlyWingsAmount%
    Gui, Add, UpDown, Range1-500, %FlyWingsAmount% ; This adds spin controls
    GuiControl,, FlyWingsAmount, % (FlyWingsAmount ? FlyWingsAmount : 100) ; Default to 15 if empty
    GuiControl, % (TakeFlyWings ? "Enable" : "Disable"), FlyWingsAmount

    Gui, Add, CheckBox, x20 y+30 vDetectCaptcha gUpdateGlobalsFromUI Checked%DetectCaptcha%, Detect Captcha (HoneyRO)

    ; Control Buttons
    buttonStartY := inputStartY + 300
    reminderY := buttonStartY - 40
    Gui, Add, Text, x220 y%reminderY% w280 h30 Center, Press F12 to quickly toggle bot

    Gui, Add, Button, x220 y%buttonStartY% w120 h40 gExitBot, Exit
    Gui, Add, Button, x360 y%buttonStartY% w120 h40 gMainBotButton vBotButton, Start Bot
    Gui, Add, Button, x480 y%buttonStartY% w120 h40 gContinueBot vContinueButton Hidden, Continue

    Gui, Show, w800 h900, Hex Bot
    GuiControl,, StatusLight, 100
    Gosub, RefreshWindows
    return

    MainBotButton:
        if (!botRunning) {
            ; Start the bot
            Gosub, StartBotProcedure
        } else {
            ; Stop the bot
            Gosub, StopBotProcedure
        }
    return

    ContinueBot:
        Gosub, UnpauseBotProcedure
    return

    StartBotProcedure:
        Critical
        Gui, Submit, NoHide

        ; STRICT verification
        if (!gameWindowID || gameWindowID = 0) {
            MsgBox, 16, Error, Please select a valid game window first!
            return
        }

        ; ENHANCED existence check
        if !WinExist("ahk_id " gameWindowID) {
            MsgBox, 16, Error, The game window doesn't exist!`nPlease refresh and select again.
            Gosub, RefreshWindows
            return
        }
        botRunning := true
        botPaused := false

        ; Update buttons
        GuiControl,, BotButton, Stop Bot
        GuiControl, Hide, ContinueButton
        GuiControl,, BotStatus, Status: ON
        GuiControl, +cGreen, StatusLight
        GuiControl,, StatusLight, 100
        Gosub, LockGUI

        RestoreWindow()
        ; Auto-pause when tabbing out
        SetTimer, CheckWindowFocus, 300 ; Checks every 500ms

        ; Read hotkey inputs
        GuiControlGet, skillKey,, SkillButtonKey
        GuiControlGet, teleportKey,, TeleportButtonKey
        GuiControlGet, savePointKey,, SavePointButtonKey
        GuiControlGet, spKey,, SPButtonKey
        GuiControlGet, storageKey,, OpenStorageButtonKey
        GuiControlGet, skillTimerKey,, SkillTimerButtonKey

        Gosub, SaveSettings

        SetTimer, StartBotWrapper, -1
    return

    StopBotProcedure:
        botRunning := false
        botPaused := false
        GuiControl,, BotButton, Start Bot
        GuiControl, Hide, ContinueButton 
        GuiControl,, BotStatus, Status: Off
        GuiControl, +cRed, StatusLight
        GuiControl,, StatusLight, 100
        Gosub, UnlockGUI
        ; ... any other cleanup ...
    return

    PauseBotProcedure:
        botPaused := true

        ; Update buttons
        GuiControl,, BotButton, Stop Bot ; Still shows Stop when paused
        GuiControl, Show, ContinueButton ; Show Continue option
        GuiControl,, BotStatus, Status: PAUSED
        GuiControl, +cYellow, StatusLight
        ToolTip, BOT PAUSED, % A_ScreenWidth//2-100, 10
        SetTimer, RemoveToolTip, -2000
    return

    UnpauseBotProcedure:
        botPaused := false
        GuiControl, Hide, ContinueButton
        GuiControl,, BotStatus, Status: ONLINE
        GuiControl, +cGreen, StatusLight
        RestoreWindow() ; Your existing function
        ToolTip, BOT RESUMED, % A_ScreenWidth//2-100, 10
        SetTimer, RemoveToolTip, -2000
    return

    ContinueFromPause:
        botPaused := false
        GuiControl,, BotButton, Stop Bot
        GuiControl,, BotStatus, Status: ONLINE
        GuiControl, +cGreen, StatusLight
        ToolTip, BOT CONTINUED, % A_ScreenWidth//2-100, 10
        SetTimer, RemoveToolTip, -2000
        RestoreWindow() ; Your existing function
    return

    ; Wrapper to start bot in separate thread
    StartBotWrapper:
        StartBot()
    return

    ; --------------------------
    ; SAVE SETTINGS FUNCTION
    ; --------------------------
    SaveSettings:
        ; Submit current GUI values
        Gui, Submit, NoHide

        ; Clear existing file and build with formatting
        FileDelete, config.ini

        ; ====== [LastSession] ======
        FileAppend, `n`n[LastSession]`n, config.ini
        IniWrite, %gameProcess%, config.ini, LastSession, GameProcess
        IniWrite, %gameWindowTitle%, config.ini, LastSession, GameTitle

        ; ====== [Window] ======
        FileAppend, `n`n[Window]`n, config.ini
        if (gameWindowID && gameWindowTitle && gameProcess) {
            IniWrite, %gameWindowID%, config.ini, Window, ID
            IniWrite, %gameWindowTitle%, config.ini, Window, Title
            IniWrite, %gameProcess%, config.ini, Window, Process
        }

        ; ====== [MonsterSettings] ======
        FileAppend, `n`n[MonsterSettings]`n, config.ini
        Loop % MobNames.MaxIndex() {
            if (SelectedMonster%A_Index%) {
                IniWrite, %A_Index%, config.ini, MonsterSettings, SelectedMonster
                IniWrite, %targetColor%, config.ini, MonsterSettings, targetColor
                break
            }
        }

        ; ====== [Settings] ======
        FileAppend, `n`n[Settings]`n, config.ini
        IniWrite, %SearchRange%, config.ini, Settings, SearchRange
        IniWrite, %TimeOnLocation%, config.ini, Settings, TimeOnLocation
        IniWrite, %Iterations%, config.ini, Settings, Iterations
        IniWrite, %WeightModifier%, config.ini, Settings, WeightModifier
        IniWrite, %TakeFlyWings%, config.ini, Settings, TakeFlyWings
        IniWrite, %DetectCaptcha%, config.ini, Settings, DetectCaptcha

        ; ====== [Warper] ======
        FileAppend, `n`n[Warper]`n, config.ini
        if (warperX && warperY) {
            IniWrite, %warperX%, config.ini, Warper, X
            IniWrite, %warperY%, config.ini, Warper, Y
            IniWrite, %warperLocation%, config.ini, Warper, warperLocation
        } else {
            IniDelete, config.ini, Warper, X
            IniDelete, config.ini, Warper, Y
            IniDelete, config.ini, Warper, warperLocation
        }

        ; ====== [Keybindings] ======
        FileAppend, `n`n[Keybindings]`n, config.ini
        GuiControlGet, SkillButtonKey,, SkillButtonKey
        GuiControlGet, TeleportButtonKey,, TeleportButtonKey
        GuiControlGet, SavePointButtonKey,, SavePointButtonKey
        GuiControlGet, SPButtonKey,, SPButtonKey
        GuiControlGet, OpenStorageButtonKey,, OpenStorageButtonKey
        GuiControlGet, SkillTimerButtonKey,, SkillTimerButtonKey

        IniWrite, %SkillButtonKey%, config.ini, Keybindings, SkillButton
        IniWrite, %SkillDelay%, config.ini, Keybindings, SkillDelay
        IniWrite, %TeleportButtonKey%, config.ini, Keybindings, TeleportButton
        IniWrite, %SavePointButtonKey%, config.ini, Keybindings, SavePointButton
        IniWrite, %SPButtonKey%, config.ini, Keybindings, SPButton
        IniWrite, %OpenStorageButtonKey%, config.ini, Keybindings, OpenStorageButton
        IniWrite, %SkillTimerButtonKey%, config.ini, Keybindings, SkillTimerButton
        IniWrite, %SkillTimerInterval%, config.ini, Keybindings, SkillTimerInterval

    return

    LoadConfig:
        ; Window
        IniRead, gameWindowID, config.ini, Window, ID, %gameWindowID%
        IniRead, gameWindowTitle, config.ini, Window, Title, %gameWindowTitle%
        IniRead, gameProcess, config.ini, Window, Process, %gameProcess%

        ; Sliders
        IniRead, SearchRange, config.ini, Settings, SearchRange, %SearchRange%
        IniRead, TimeOnLocation, config.ini, Settings, TimeOnLocation, %TimeOnLocation%
        IniRead, WeightModifier, config.ini, Settings, WeightModifier, %WeightModifier%

        ; Checkboxes
        IniRead, TakeFlyWings, config.ini, Settings, TakeFlyWings, %TakeFlyWings%
        IniRead, DetectCaptcha, config.ini, Settings, DetectCaptcha, %DetectCaptcha%

        ; Warper
        IniRead, warperX, config.ini, Warper, X
        IniRead, warperY, config.ini, Warper, Y
        IniRead, warperLocation, config.ini, Warper, warperLocation

        ; Set the flag and clean up if invalid
        warperCoordsSet := (warperX != "" && warperY != "" && warperX != "ERROR" && warperY != "ERROR")

        ; Only reset if coordinates are INVALID
        if (!warperCoordsSet) {
            warperX := ""
            warperY := ""
            warperLocation := 0
        }

        ; Keybindings
        IniRead, SkillButtonKey, config.ini, Keybindings, SkillButton, %SkillButtonKey%
        IniRead, SkillDelay, config.ini, Keybindings, SkillDelay, %SkillDelay%
        IniRead, TeleportButtonKey, config.ini, Keybindings, TeleportButton, %TeleportButtonKey%
        IniRead, SavePointButtonKey, config.ini, Keybindings, SavePointButton, %SavePointButtonKey%
        IniRead, SPButtonKey, config.ini, Keybindings, SPButton, %SPButtonKey%
        IniRead, OpenStorageButtonKey, config.ini, Keybindings, OpenStorageButton, %OpenStorageButtonKey%
        IniRead, SkillTimerButtonKey, config.ini, Keybindings, SkillTimerButton, %SkillTimerButtonKey%
        IniRead, SkillTimerInterval, config.ini, Keybindings, SkillTimerInterval, %SkillTimerInterval%

        ; Monster
        IniRead, selectedMonsterIndex, config.ini, MonsterSettings, SelectedMonster, 1
        targetColor := MobColors[selectedMonsterIndex]
    return

    ; --------------------------
    ; UPDATE SLIDER VALUES FUNCTION
    ; --------------------------
    UpdateSliderValues:
        GuiControlGet, focusedControl, FocusV

        if (focusedControl = "SearchRange") {
            GuiControlGet, SearchRange,, SearchRange
            GuiControl,, SearchRangeText, %SearchRange%
        }
        else if (focusedControl = "WeightModifier") {
            GuiControlGet, WeightModifier,, WeightModifier
            GuiControl,, WeightModifierText, % (WeightModifier = 49 ? "Off" : WeightModifier)
        }
        else if (focusedControl = "TimeOnLocation") {
            GuiControlGet, TimeOnLocation,, TimeOnLocation
            GuiControl,, TimeOnLocationValueText, %TimeOnLocation%
        }
        Gosub, UpdateGlobalsFromUI
    return

    RestoreWindow(){
        ; RESTORE MINIMIZED WINDOW
        WinGet, minMaxStatus, MinMax, ahk_id %gameWindowID%
        if (minMaxStatus = -1) {
            WinRestore, ahk_id %gameWindowID%
            Sleep, 1000 ; Give time to restore
        }

        ; ACTIVATE WINDOW 
        WinActivate, ahk_id %gameWindowID%
        WinWaitActive, ahk_id %gameWindowID%, , 2

        ; FINAL VERIFICATION
        WinGet, activeID, ID, A
        if (activeID != gameWindowID) {
            ; LAST RESORT - remove disabled style
            WinSet, Style, -0x8000000, ahk_id %gameWindowID%
            WinActivate, ahk_id %gameWindowID%
            WinWaitActive, ahk_id %gameWindowID%, , 3
            if ErrorLevel {
                MsgBox, 16, Error, Failed to activate game window!`nTry running as Administrator.
                return
            }
        }
    }

    LockGUI:
        ; Disable Window Selection controls
        GuiControl, Disable, SelectedWindow
        GuiControl, Disable, RefreshWindows

        ; Disable Warper Coordinates controls
        GuiControl, Disable, SetWarperCoordsBtn
        GuiControl, Disable, ResetWarperBtn

        ; Disable Time on Location controls
        GuiControl, Disable, TimeOnLocation
        GuiControl, Disable, TimeOnLocationTextLabel
        GuiControl, Disable, TimeOnLocationValueText

        ; Disable Monster Type radio buttons
        Loop % MobNames.MaxIndex() {
            GuiControl, Disable, SelectedMonster%A_Index%
        }

        ; Disable Keybinding controls
        GuiControl, Disable, SkillButtonKey
        GuiControl, Disable, SkillDelay
        GuiControl, Disable, TeleportButtonKey
        GuiControl, Disable, SavePointButtonKey
        GuiControl, Disable, SPButtonKey
        GuiControl, Disable, OpenStorageButtonKey
        GuiControl, Disable, SkillTimerButtonKey
        GuiControl, Disable, SkillTimerInterval

        ; Disable Sliders
        GuiControl, Disable, SearchRange
        GuiControl, Disable, SearchRangeText
        GuiControl, Disable, WeightModifier
        GuiControl, Disable, WeightModifierText

        ; Disable Checkboxes
        GuiControl, Disable, TakeFlyWings
        GuiControl, Disable, DetectCaptcha

        ; Visual feedback
        Gui, Font, cGray
        GuiControl, Font, WindowInfo
        GuiControl, Font, WarperCoordsText
        GuiControl, Font, MouseStatus
        GuiControl, Font, KeyboardStatus
    return

    UnlockGUI:
        ; Enable Window Selection controls
        GuiControl, Enable, SelectedWindow
        GuiControl, Enable, RefreshWindows

        ; Enable Warper Coordinates controls (with conditional enable for Reset button)
        GuiControl, Enable, SetWarperCoordsBtn
        GuiControl, % warperCoordsSet ? "Enable" : "Disable", ResetWarperBtn

        ; Enable Time on Location controls (with conditional visibility)
        GuiControl, % warperCoordsSet ? "Enable" : "Disable", TimeOnLocation
        GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocationTextLabel
        GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocationValueText

        ; Enable Monster Type radio buttons
        Loop % MobNames.MaxIndex() {
            GuiControl, Enable, SelectedMonster%A_Index%
        }

        ; Enable Keybinding controls
        GuiControl, Enable, SkillButtonKey
        GuiControl, Enable, SkillDelay
        GuiControl, Enable, TeleportButtonKey
        GuiControl, Enable, SavePointButtonKey
        GuiControl, Enable, SPButtonKey
        GuiControl, Enable, OpenStorageButtonKey
        GuiControl, Enable, SkillTimerButtonKey
        GuiControl, Enable, SkillTimerInterval

        ; Enable Sliders
        GuiControl, Enable, SearchRange
        GuiControl, Enable, SearchRangeText
        GuiControl, Enable, WeightModifier
        GuiControl, Enable, WeightModifierText

        ; Enable Checkboxes
        GuiControl, Enable, TakeFlyWings
        GuiControl, Enable, DetectCaptcha

        ; Restore visual style
        Gui, Font, cBlack
        GuiControl, Font, WindowInfo
        GuiControl, Font, WarperCoordsText
        GuiControl, Font, MouseStatus
        GuiControl, Font, KeyboardStatus
    return

    RefreshWindows:
        GuiControl,, SelectedWindow, |
        windowList := ""
        windowIDs := {}
        titleToIndex := {}

        DetectHiddenWindows, On
        WinGet, windows, List
        DetectHiddenWindows, Off

        itemCount := 0
        Loop, %windows%
        {
            id := windows%A_Index%
            WinGetTitle, title, ahk_id %id%
            WinGet, process, ProcessName, ahk_id %id%

            if (title = "" || process = "")
                continue

            WinGet, guiID, ID, Monster Hunter Bot ahk_class AutoHotkeyGUI
            if (id = guiID)
                continue

            itemCount += 1
            WinGet, minMaxStatus, MinMax, ahk_id %id%
            stateSymbol := (minMaxStatus = -1) ? "[MIN] " : ""
                displayText := stateSymbol title " (" process ")"

                windowList .= "|" displayText
                windowIDs[itemCount] := id
                titleToIndex[displayText] := itemCount
            }

            GuiControl,, SelectedWindow, %windowList%

            ; Attempt to auto-select previously used window
            IniRead, lastTitle, config.ini, LastSession, GameTitle, ERROR
            IniRead, lastProcess, config.ini, LastSession, GameProcess, ERROR

            if (lastTitle != "ERROR" && lastProcess != "ERROR") {
                for index, id in windowIDs {
                    WinGetTitle, tTitle, ahk_id %id%
                    WinGet, tProcess, ProcessName, ahk_id %id%
                    if (tTitle = lastTitle && tProcess = lastProcess) {
                        WinGet, minMaxStatus, MinMax, ahk_id %id%
                        stateSymbol := (minMaxStatus = -1) ? "[MIN] " : ""
                            selectedDisplay := stateSymbol tTitle " (" tProcess ")"
                            GuiControl, ChooseString, SelectedWindow, %selectedDisplay%
                            Gosub, OnWindowSelect
                        return
                    }
                }
            }

            ; If nothing selected, pick the first one (if any)
            if (windowList != "") {
                GuiControl, Choose, SelectedWindow, 1
                Gosub, OnWindowSelect
            }
            return

            OnWindowSelect:
                GuiControlGet, selectedText,, SelectedWindow

                if (!selectedText || !titleToIndex.HasKey(selectedText)) {
                    GuiControl,, WindowInfo, No window selected
                    gameWindowID := 0
                    return
                }

                selectedIndex := titleToIndex[selectedText]
                gameWindowID := windowIDs[selectedIndex]

                if (!gameWindowID || !WinExist("ahk_id " gameWindowID)) {
                    GuiControl,, WindowInfo, Window not found! Refresh list.
                    gameWindowID := 0
                    return
                }

                WinGetTitle, gameWindowTitle, ahk_id %gameWindowID%
                WinGet, gameProcess, ProcessName, ahk_id %gameWindowID%
                GuiControl,, WindowInfo, % "SELECTED: " gameProcess

                WinSet, Transparent, 150, ahk_id %gameWindowID%
                Sleep, 300
                WinSet, Transparent, Off, ahk_id %gameWindowID%
            return

            RemoveToolTip:
                ToolTip ; This clears any active tooltip
                SetTimer, RemoveToolTip, Off ; Turn off the timer
            return

            CheckWindowFocus:
                if (botRunning && !botPaused && !WinActive(gameWindowTitle)) {
                    Gosub, PauseBotProcedure
                    GuiControl,, BotStatus, Status: PAUSED (TAB)
                }
            return

            SetWarperCoords:
                ; Visual feedback
                GuiControl, Disable, SetWarperCoordsBtn
                GuiControl,, SetWarperCoordsBtn, Setting...

                ; Create always-on-top tooltip
                CoordMode, ToolTip, Screen
                SetTimer, UpdateWarperToolTip, 50

                ; Wait for W key + mouse click
                KeyWait, w, D
                KeyWait, LButton, D

                ; Get coordinates
                MouseGetPos, warperX, warperY
                warperLocation := ReadMemoryUInt(gameProcess,currentLocationAddress)

                ; Clean up
                SetTimer, UpdateWarperToolTip, Off
                ToolTip

                ; Update UI
                warperCoordsSet := true
                GuiControl,, WarperCoordsText, X: %warperX% Y: %warperY%
                GuiControl, Enable, ResetWarperBtn
                GuiControl,, SetWarperCoordsBtn, Set Warper Position

                ; Show time controls
                GuiControl, Show, TimeOnLocationTextLabel
                GuiControl, Show, TimeOnLocation
                GuiControl, Show, TimeOnLocationValueText

                ; Feedback
                ToolTip, Coordinates set!
                SetTimer, RemoveToolTip, -2000
            return

            UpdateWarperToolTip:
                MouseGetPos, mX, mY
                ToolTip, Hold W and click the warper NPC, mX+20, mY+20
                WinSet, AlwaysOnTop, On, ahk_class tooltips_class32
            return

            ResetWarperCoords:
                ; Reset values
                warperX := ""
                warperY := ""
                warperCoordsSet := false

                ; Update UI
                GuiControl,, WarperCoordsText, Not set
                GuiControl, Disable, ResetWarperBtn
                GuiControl, Enable, SetWarperCoordsBtn ; <--- THIS IS THE CRUCIAL LINE

                ; Hide time controls
                GuiControl, Hide, TimeOnLocationTextLabel
                GuiControl, Hide, TimeOnLocation
                GuiControl, Hide, TimeOnLocationValueText

                ; Feedback
                ToolTip, Coordinates reset!
                SetTimer, RemoveToolTip, 2000
            return

            UpdateTakeFlyWings:
                Gui, Submit, NoHide
                GuiControl, % (TakeFlyWings ? "Enable" : "Disable"), FlyWingsAmount
            return

            UpdateGlobalsFromUI:
                Gui, Submit, NoHide

                ; Update monster selection
                Loop % MobNames.MaxIndex() {
                    if (SelectedMonster%A_Index%) {
                        selectedMonsterIndex := A_Index
                        targetColor := MobColors[selectedMonsterIndex]
                        break
                    }
                }

                ; Update slider values
                SearchRange := SearchRange
                TimeOnLocation := TimeOnLocation
                WeightModifier := WeightModifier

                ; Update checkbox states
                TakeFlyWings := TakeFlyWings
                DetectCaptcha := DetectCaptcha

                ; Update keybindings
                SkillButtonKey := SkillButtonKey
                TeleportButtonKey := TeleportButtonKey
                SavePointButtonKey := SavePointButtonKey
                SPButtonKey := SPButtonKey
                OpenStorageButtonKey := OpenStorageButtonKey
                SkillTimerButtonKey := SkillTimerButtonKey
                SkillDelay := SkillDelay
                SkillTimerInterval := SkillTimerInterval

                ; Update your global variable
                wingsTaken := FlyWingsAmount
            return

            ExitBot:
            ExitApp
            return

            GuiClose:
            ExitApp
