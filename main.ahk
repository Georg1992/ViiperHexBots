#Persistent
#SingleInstance Force
#KeyHistory 0
#InstallKeybdHook
#InstallMouseHook
#include Lib\ViiperInput.ahk
#include Lib\SessionLog.ahk
#include Lib\BotSession.ahk
#include Lib\MobData.ahk
#include Lib\ClientProfile.ahk
#include Lib\HuntTracks.ahk
#include Lib\HuntPolicy.ahk
#include Lib\MobRecognition.ahk
#include Lib\MobStateRecognition.ahk
#include Lib\MemoryOperations.ahk
#include Lib\BotLogic.ahk
#include Lib\utilityFunctions.ahk
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
global botStopRequested := false

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
global SearchRange := 16 ; Default search range (9-16 cells)
global cellSize := 50 ; Pixels per RO viewport cell
global TimeOnLocation := 20 ; Default time in seconds
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

; ====== VIIPER INPUT ======
global Input := ""
global inputReady := false
global inputShutdownDone := false
global viperShutdownRequested := false
global exitCleanupDone := false
OnExit("MainOnExit")
global LogBoxHwnd := 0

SetDefaultKeyboardLayout("00000409") ; English - US

if (!FileExist("config.ini")) {
    FileAppend, , config.ini ; Create empty file
}

; -------------------------------
; Load Config Values First
; -------------------------------
if (FileExist("config.ini")) {
    Gosub, LoadConfig
}

LoadClientProfile(clientProfileName)
if (!clientSupportsMemory)
    memoryReadingEnabled := false

clientOptions := ""
for index, profileName in ListClientProfiles()
    clientOptions .= (clientOptions = "" ? "" : "|") . profileName

; --------------------------
; GUI Setup (With Loaded Values)
; --------------------------
Gui, Font, s10, Segoe UI

; Title
Gui, Add, Text, x10 y15 w900 h30 Center, ViiperHex Bot
Gui, Add, Text, x10 y50 w900 h2 0x10 ; Divider

; Window Selection
Gui, Add, GroupBox, x15 y65 w570 h95, Game Window
Gui, Add, Text, x30 y90 w150 h20, Select game window:
Gui, Add, DropDownList, x30 y115 w450 h25 r10 vSelectedWindow gOnWindowSelect, ||
Gui, Add, Button, x490 y115 w75 h25 gRefreshWindows, Refresh
Gui, Add, Text, x30 y140 w535 h20 vWindowInfo, % (gameWindowTitle ? gameWindowTitle : "No window selected")

; Bot Status
Gui, Add, GroupBox, x610 y65 w280 h95, Status
Gui, Add, Text, x630 y90 w120 h25 vBotStatus, Status: Off
Gui, Add, Progress, x630 y120 w230 h14 cRed vStatusLight, 100

; Input backend
Gui, Add, GroupBox, x610 y170 w280 h105, Input
Gui, Add, Text, x630 y195 w230 h25 vInputStatus, Input: Starting...
Gui, Add, Text, x630 y225 w230 h35 vInputHint, Launch the game after VIIPER is ready

; Client and mob setup
Gui, Add, GroupBox, x15 y175 w270 h195, Setup
Gui, Add, Text, x30 y205 w100 h20, Client Profile:
Gui, Add, DropDownList, x135 y202 w130 h25 r10 vSelectedClientProfile gOnClientProfileChange, %clientOptions%
Gui, Add, CheckBox, x30 y235 w220 h25 vUseMemoryReading gUpdateMemoryMode Checked%memoryReadingEnabled%, Use memory reading

; Descriptor mob selection
Gui, Add, Text, x30 y280 w220 h20, Descriptor Mob:
mobY := 305

if (MobNames.MaxIndex() < 1) {
    MsgBox, 16, ViiperHexBots, No mob descriptors found.`n`nCreate one with:`n.\scripts\build-mob-descriptor.ps1 -Mob horn
    ExitApp
}
if (selectedMonsterIndex < 1 || selectedMonsterIndex > MobNames.MaxIndex())
    selectedMonsterIndex := 1

Loop % MobNames.MaxIndex() {
    ; Use the pre-loaded selectedMonsterIndex from config
    isChecked := (A_Index == selectedMonsterIndex) ? "Checked" : ""
    Gui, Add, Radio, x35 y%mobY% %isChecked% vSelectedMonster%A_Index% gUpdateGlobalsFromUI, % MobNames[A_Index]
    mobY += 25
}

Gui, Add, Text, x30 y390 w230 h20, Mob detection: simple descriptor

; Keybindings
Gui, Add, GroupBox, x305 y175 w280 h245, Keybindings
Gui, Add, Text, x325 y205 w125 h22, Attack Skill:
Gui, Add, Hotkey, x455 y202 w65 h25 vSkillButtonKey gUpdateGlobalsFromUI, %SkillButtonKey%
Gui, Add, Text, x325 y235 w125 h22, Attack delay:
Gui, Add, Edit, x455 y232 w45 h25 vSkillDelay Number gUpdateGlobalsFromUI, %SkillDelay%
Gui, Add, Text, x505 y235 w35 h22, ms

Gui, Add, Text, x325 y270 w125 h22, Teleport:
Gui, Add, Hotkey, x455 y267 w65 h25 vTeleportButtonKey gUpdateGlobalsFromUI, %TeleportButtonKey%
Gui, Add, Text, x325 y300 w125 h22, Save Point:
Gui, Add, Hotkey, x455 y297 w65 h25 vSavePointButtonKey gUpdateGlobalsFromUI, %SavePointButtonKey%
Gui, Add, Text, x325 y330 w125 h22, Open Storage:
Gui, Add, Hotkey, x455 y327 w65 h25 vOpenStorageButtonKey gUpdateGlobalsFromUI, %OpenStorageButtonKey%
Gui, Add, Text, x325 y360 w125 h22, Skill Timer:
Gui, Add, Hotkey, x455 y357 w65 h25 vSkillTimerButtonKey gUpdateGlobalsFromUI, %SkillTimerButtonKey%
Gui, Add, Text, x325 y390 w125 h22, SP Item:
Gui, Add, Hotkey, x455 y387 w65 h25 vSPButtonKey gUpdateGlobalsFromUI, %SPButtonKey%
Gui, Add, Text, x525 y360 w35 h22, every
Gui, Add, Edit, x560 y357 w35 h25 vSkillTimerInterval Number gUpdateGlobalsFromUI, %SkillTimerInterval%
Gui, Add, Text, x600 y360 w20 h22, s

; Hunt settings
Gui, Add, GroupBox, x15 y430 w570 h130, Hunt Settings
Gui, Add, Text, x35 y460 w135 h30, Search Range (9-16 Cells):
Gui, Add, Slider, x175 y457 w260 h25 vSearchRange gUpdateSliderValues Range9-16 TickInterval1 ToolTip, %SearchRange%
Gui, Add, Text, x445 y457 w35 h25 vSearchRangeText Center, %SearchRange%

Gui, Add, Text, x35 y500 w135 h30 vWeightSliderLabel, Items To Kafra when weight is:
Gui, Add, Slider, x175 y497 w260 h25 vWeightModifier gUpdateSliderValues Range49-90 TickInterval1 ToolTip, %WeightModifier%
Gui, Add, Text, x445 y497 w45 h25 vWeightModifierText Center, % (WeightModifier = 49 ? "Off" : WeightModifier)
Gui, Add, Text, x490 y497 w25 h25, `%

Gui, Add, CheckBox, x35 y535 vTakeFlyWings gUpdateTakeFlyWings Checked%TakeFlyWings%, Take Fly Wings
Gui, Add, Edit, x145 y532 w50 vFlyWingsAmount Number Limit3 -WantReturn, %FlyWingsAmount%
Gui, Add, UpDown, Range1-500, %FlyWingsAmount% ; This adds spin controls
GuiControl,, FlyWingsAmount, % (FlyWingsAmount ? FlyWingsAmount : 100)
GuiControl, % (TakeFlyWings ? "Enable" : "Disable"), FlyWingsAmount
Gui, Add, CheckBox, x260 y535 vDetectCaptcha gUpdateGlobalsFromUI Checked%DetectCaptcha%, Detect Captcha

; Warper Coordinates
Gui, Add, GroupBox, x610 y290 w280 h125, Warper Coordinates
Gui, Add, Button, x630 y315 w130 h28 gSetWarperCoords vSetWarperCoordsBtn, Set Position
Gui, Add, Button, x770 y315 w95 h28 gResetWarperCoords vResetWarperBtn, Reset
Gui, Add, Text, x630 y352 w230 h20 vWarperCoordsText, % warperCoordsSet ? "X: " warperX " Y: " warperY : "Not set"
GuiControl, % warperCoordsSet ? "Enable" : "Disable", ResetWarperBtn

; Time on Location Controls
Gui, Add, Text, x630 y378 w130 h22 vTimeOnLocationTextLabel, Time on location:
Gui, Add, Slider, x630 y397 w185 h25 vTimeOnLocation gUpdateSliderValues Range20-240, %TimeOnLocation%
Gui, Add, Text, x820 y397 w45 h25 vTimeOnLocationValueText Center, %TimeOnLocation%
GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocationTextLabel
GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocation
GuiControl, % warperCoordsSet ? "Show" : "Hide", TimeOnLocationValueText

; Set initial visibility based on warperCoordsSet
if (!warperCoordsSet) {
    GuiControl, Hide, TimeOnLocationTextLabel
    GuiControl, Hide, TimeOnLocation
    GuiControl, Hide, TimeOnLocationValueText
}

; Log panel
Gui, Add, GroupBox, x610 y430 w280 h225, Log
Gui, Add, Edit, x625 y455 w250 h185 ReadOnly -WantReturn +VScroll vLogBox gLogBoxFocus

GuiControl, ChooseString, SelectedClientProfile, %clientProfileName%
Gosub, ApplyMemoryDependentUI

; Control Buttons
Gui, Add, Text, x230 y605 w300 h25 Center, Press F12 to quickly toggle bot
Gui, Add, Button, x255 y640 w120 h40 gExitBot, Exit
Gui, Add, Button, x400 y640 w120 h40 gMainBotButton vBotButton, Start Bot
Gui, Add, Button, x545 y640 w120 h40 gContinueBot vContinueButton Hidden, Continue

Gui, Show, w920 h710, Hex Bot
    Menu, Tray, NoStandard
    Menu, Tray, Add, Open, TrayOpen
    Menu, Tray, Add, Reload Script, TrayReload
    Menu, Tray, Add
    Menu, Tray, Add, Exit, TrayExit
    Menu, Tray, Default, Open
    Menu, Tray, Tip, ViiperHexBots
    GuiControlGet, LogBoxHwnd, Hwnd, LogBox
    GuiControl,, StatusLight, 100
    UpdateSearchRangeLabel()

    GuiControl, Disable, SelectedWindow
    GuiControl, Disable, RefreshWindows
    GuiControl, Disable, BotButton

    if (A_Args.Length() > 0 && A_Args[1] = "--validate") {
        ExitApp
        return
    }

    SessionLogStart()
    AppendLog("ViiperHexBots started")
    AppendLog("Starting VIIPER before game launch...")
    SetTimer, InitViiperInput, -1
    return

    InitViiperInput:
        global Input, inputReady
        Input := new ViiperInput()
        inputReady := true
        SetInputStatus("Input: Ready", "Virtual keyboard and mouse active — launch the game now")
        GuiControl, Enable, SelectedWindow
        GuiControl, Enable, RefreshWindows
        GuiControl, Enable, BotButton
        AppendLog("All set — select or launch the game window")
        SessionLogWriteRuntimeContext()
        Gosub, RefreshWindows
    return

    LogBoxFocus:
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

        if (!inputReady) {
            MsgBox, 16, Error, VIIPER is not ready yet.`nPlease wait for initialization to finish.
            return
        }

        Gosub, OnWindowSelect

        ; STRICT verification
        if (!gameWindowID || gameWindowID = 0) {
            MsgBox, 16, Error, Please select a valid game window first!`nChoose the game in the dropdown (with its .exe name) and click Refresh if needed.
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
        botStopRequested := false

        ; Update buttons
        GuiControl,, BotButton, Stop Bot
        GuiControl, Hide, ContinueButton
        GuiControl,, BotStatus, Status: ON
        GuiControl, +cGreen, StatusLight
        GuiControl,, StatusLight, 100
        Gosub, LockGUI
        if (MemoryFeaturesActive())
            AppendLog("Bot started (memory reading on)")
        else
            AppendLog("Bot started (memory reading off)")

        RestoreWindow()
        SyncSearchRangeFromUI()
        if (!GetHuntSearchRegion(searchXs, searchYs, searchWs, searchHs)) {
            AppendLog("ERROR: Game window not selected — select window and try again")
            Gosub, StopBotProcedure
            return
        }
        if IsFunc("AppendLog")
            AppendLog("Search box: " . searchWs . "x" . searchHs . " px (" . SearchRange . " cells) at " . searchXs . "," . searchYs)
        ShowSearchRegionOverlay(searchXs, searchYs, searchWs, searchHs, 2500)
        HuntSessionReset(true)
        BotSessionStart(MobTemplateFolderName())
        SessionLogRegisterBotRun()
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
        botStopRequested := true
        SetTimer, CheckWindowFocus, Off
        ReleaseBotInputs()
        BotSessionStop("stopped")
        AppendLog("Bot stopped (VIIPER still running)")
        GuiControl,, BotButton, Start Bot
        GuiControl, Hide, ContinueButton 
        GuiControl,, BotStatus, Status: Off
        GuiControl, +cRed, StatusLight
        GuiControl,, StatusLight, 100
        Gosub, UnlockGUI
    return

    PauseBotProcedure:
        botPaused := true
        WinGet, focusLostActiveId, ID, A
        SessionLogFocusChange("paused (focus lost)", focusLostActiveId)
        AppendLog("Bot paused (focus lost)")

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
        SessionLogFocusChange("resumed")
        AppendLog("Bot resumed")
        GuiControl, Hide, ContinueButton
        GuiControl,, BotStatus, Status: ONLINE
        GuiControl, +cGreen, StatusLight
        RestoreWindow()
        ToolTip, BOT RESUMED, % A_ScreenWidth//2-100, 10
        SetTimer, RemoveToolTip, -2000
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

        ; ====== [Client] ======
        FileAppend, `n`n[Client]`n, config.ini
        IniWrite, %SelectedClientProfile%, config.ini, Client, Profile
        IniWrite, %UseMemoryReading%, config.ini, Client, UseMemoryReading

        ; ====== [MonsterSettings] ======
        FileAppend, `n`n[MonsterSettings]`n, config.ini
        Loop % MobNames.MaxIndex() {
            if (SelectedMonster%A_Index%) {
                IniWrite, %A_Index%, config.ini, MonsterSettings, SelectedMonster
                break
            }
        }

        ; ====== [MobRecognition] ======
        FileAppend, `n`n[MobRecognition]`n, config.ini
        IniWrite, %mobRecognitionDebug%, config.ini, MobRecognition, Debug

        ; ====== [Settings] ======
        FileAppend, `n`n[Settings]`n, config.ini
        IniWrite, %SearchRange%, config.ini, Settings, SearchRange
        IniWrite, %TimeOnLocation%, config.ini, Settings, TimeOnLocation
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
        ; Client
        IniRead, clientProfileName, config.ini, Client, Profile, HoneyRO
        IniRead, memoryReadingEnabled, config.ini, Client, UseMemoryReading, 1

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
        if (selectedMonsterIndex < 1 || selectedMonsterIndex > MobNames.MaxIndex())
            selectedMonsterIndex := 1

        ; Mob recognition
        IniRead, mobRecognitionDebug, config.ini, MobRecognition, Debug, 0
    return

    ; --------------------------
    ; UPDATE SLIDER VALUES FUNCTION
    ; --------------------------
    ApplyMemoryDependentUI:
        ApplyMemoryDependentUI()
    return

    OnClientProfileChange:
        GuiControlGet, SelectedClientProfile,, SelectedClientProfile
        GuiControlGet, UseMemoryReading,, UseMemoryReading
        LoadClientProfile(SelectedClientProfile)
        if (!clientSupportsMemory) {
            memoryReadingEnabled := false
            UseMemoryReading := 0
        } else {
            memoryReadingEnabled := UseMemoryReading
        }
        Gosub, ApplyMemoryDependentUI
        AppendLog("Client profile: " . clientProfileName)
    return

    UpdateMemoryMode:
        Gui, Submit, NoHide
        memoryReadingEnabled := UseMemoryReading
        Gosub, ApplyMemoryDependentUI
    return

    UpdateSliderValues:
        GuiControlGet, SearchRange,, SearchRange
        UpdateSearchRangeLabel()

        GuiControlGet, focusedControl, FocusV
        if (focusedControl = "WeightModifier") {
            GuiControlGet, WeightModifier,, WeightModifier
            GuiControl,, WeightModifierText, % (WeightModifier = 49 ? "Off" : WeightModifier)
        }
        else if (focusedControl = "TimeOnLocation") {
            GuiControlGet, TimeOnLocation,, TimeOnLocation
            GuiControl,, TimeOnLocationValueText, %TimeOnLocation%
        }
        Gosub, UpdateGlobalsFromUI
    return

    LockGUI:
        ; Disable Window Selection controls
        GuiControl, Disable, SelectedWindow
        GuiControl, Disable, RefreshWindows

        GuiControl, Disable, SelectedClientProfile
        GuiControl, Disable, UseMemoryReading

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
        GuiControl, Disable, WeightSliderLabel

        ; Disable Checkboxes
        GuiControl, Disable, TakeFlyWings
        GuiControl, Disable, DetectCaptcha

        ; Visual feedback
        Gui, Font, cGray
        GuiControl, Font, WindowInfo
        GuiControl, Font, WarperCoordsText
    return

    UnlockGUI:
        ; Enable Window Selection controls
        GuiControl, Enable, SelectedWindow
        GuiControl, Enable, RefreshWindows

        GuiControl, Enable, SelectedClientProfile
        GuiControl, Enable, UseMemoryReading

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
        GuiControl, Enable, WeightSliderLabel

        ; Enable Checkboxes
        GuiControl, Enable, TakeFlyWings
        GuiControl, Enable, DetectCaptcha

        Gosub, ApplyMemoryDependentUI

        ; Restore visual style
        Gui, Font, cBlack
        GuiControl, Font, WindowInfo
        GuiControl, Font, WarperCoordsText
    return

    RefreshWindows:
        global windowIDs, titleToIndex, gameWindowID, gameWindowTitle, gameProcess

        previousWindowID := gameWindowID
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
            if (id = A_ScriptHwnd)
                continue

            WinGetTitle, title, ahk_id %id%
            WinGet, process, ProcessName, ahk_id %id%

            if (title = "" || process = "")
                continue

            if (process ~= "i)^explorer\.exe$")
                continue

            itemCount += 1
            WinGet, minMaxStatus, MinMax, ahk_id %id%
            displayText := FormatWindowListEntry(title, process, minMaxStatus)

            windowList .= "|" . displayText
            windowIDs[itemCount] := id
            titleToIndex[displayText] := itemCount
        }

        GuiControl,, SelectedWindow, % (windowList ? windowList : "|")

        if (previousWindowID) {
            for index, id in windowIDs {
                if (id = previousWindowID) {
                    WinGetTitle, tTitle, ahk_id %id%
                    WinGet, tProcess, ProcessName, ahk_id %id%
                    WinGet, minMaxStatus, MinMax, ahk_id %id%
                    selectedDisplay := FormatWindowListEntry(tTitle, tProcess, minMaxStatus)
                    GuiControl, ChooseString, SelectedWindow, %selectedDisplay%
                    Gosub, OnWindowSelect
                    return
                }
            }
        }

        IniRead, lastTitle, config.ini, LastSession, GameTitle, ERROR
        IniRead, lastProcess, config.ini, LastSession, GameProcess, ERROR

        if (lastTitle != "ERROR" && lastProcess != "ERROR") {
            for index, id in windowIDs {
                WinGetTitle, tTitle, ahk_id %id%
                WinGet, tProcess, ProcessName, ahk_id %id%
                if (tTitle = lastTitle && tProcess = lastProcess) {
                    WinGet, minMaxStatus, MinMax, ahk_id %id%
                    selectedDisplay := FormatWindowListEntry(tTitle, tProcess, minMaxStatus)
                    GuiControl, ChooseString, SelectedWindow, %selectedDisplay%
                    Gosub, OnWindowSelect
                    return
                }
            }
        }

        if (itemCount = 0) {
            GuiControl,, WindowInfo, No windows found — launch game and Refresh
            AppendLog("No windows in list (try Run as administrator)")
        } else {
            GuiControl,, WindowInfo, Select a game window
        }
        gameWindowID := 0
    return

    OnWindowSelect:
        global windowIDs, titleToIndex, gameWindowID, gameWindowTitle, gameProcess

        GuiControlGet, selectedText,, SelectedWindow

        if (!selectedText || !titleToIndex.HasKey(selectedText)) {
            GuiControl,, WindowInfo, No window selected
            gameWindowID := 0
            if (selectedText)
                AppendLog("Window lookup failed — click Refresh and select again")
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
        GuiControl,, WindowInfo, % "SELECTED: " . selectedText
        AppendLog("Game window selected: " . gameProcess)
        SessionLogWriteRuntimeContext()

        WinSet, Transparent, 150, ahk_id %gameWindowID%
        Sleep, 300
        WinSet, Transparent, Off, ahk_id %gameWindowID%
    return

    RemoveToolTip:
        ToolTip
        SetTimer, RemoveToolTip, Off
    return

    CheckWindowFocus:
        if (botRunning && !botPaused && gameWindowID) {
            if (!WinActive("ahk_id " . gameWindowID)) {
                Gosub, PauseBotProcedure
                GuiControl,, BotStatus, Status: PAUSED (TAB)
            }
        }
    return

    SetWarperCoords:
                if (!MemoryFeaturesActive()) {
                    MsgBox, 48, Warper, Warper coordinates require memory reading.
                    return
                }

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
                memoryReadingEnabled := UseMemoryReading

                ; Update monster selection
                Loop % MobNames.MaxIndex() {
                    if (SelectedMonster%A_Index%) {
                        selectedMonsterIndex := A_Index
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
                Gosub, ExitApplication
            return

            GuiClose:
                Gosub, ExitApplication
            return

            TrayOpen:
                Gui, Show
                Gui, Restore
            return

            TrayReload:
                Reload
            return

            TrayExit:
                Gosub, ExitApplication
            return

            ExitApplication:
                global exitCleanupDone
                global botStopRequested
                if (exitCleanupDone) {
                    ExitApp
                    return
                }
                exitCleanupDone := true
                botStopRequested := true
                ReleaseBotInputs()
                SetTimer, CheckWindowFocus, Off
                SetTimer, InitViiperInput, Off
                if (botRunning)
                    Gosub, StopBotProcedure
                viperShutdownRequested := true
                AppendLog("Closing bot and stopping VIIPER...")
                MobRecognitionExitCleanup()
                ShutdownInput()
                SessionLogEnd("user exit")
                ExitApp
            return

MainOnExit(ExitReason, ExitCode) {
    if (ExitReason = "Reload")
        return
    global exitCleanupDone
    global botStopRequested
    if (exitCleanupDone)
        return
    exitCleanupDone := true
    botStopRequested := true
    ReleaseBotInputs()

    SetTimer, CheckWindowFocus, Off
    SetTimer, InitViiperInput, Off
    global botRunning
    if (botRunning)
        Gosub, StopBotProcedure
    global viperShutdownRequested
    viperShutdownRequested := true
    MobRecognitionExitCleanup()
    ShutdownInput()
    global sessionLogPath
    if (sessionLogPath != "")
        SessionLogEnd("exit: " . ExitReason)
}

ApplyMemoryDependentUI() {
    global memoryReadingEnabled, clientSupportsMemory, captchaEnabled, captchaLabel
    global warperCoordsSet, TakeFlyWings, DetectCaptcha

    if (!clientSupportsMemory) {
        memoryReadingEnabled := false
        GuiControl,, UseMemoryReading, 0
        GuiControl, Disable, UseMemoryReading
    } else {
        GuiControl, Enable, UseMemoryReading
        checked := memoryReadingEnabled ? 1 : 0
        GuiControl,, UseMemoryReading, %checked%
    }

    memActive := MemoryFeaturesActive()
    ctlState := memActive ? "Enable" : "Disable"

    GuiControl, %ctlState%, SetWarperCoordsBtn
    GuiControl, % (memActive && warperCoordsSet) ? "Enable" : "Disable", ResetWarperBtn
    GuiControl, %ctlState%, WeightModifier
    GuiControl, %ctlState%, WeightModifierText
    GuiControl, %ctlState%, WeightSliderLabel
    GuiControl, %ctlState%, TakeFlyWings
    GuiControl, % (memActive && TakeFlyWings) ? "Enable" : "Disable", FlyWingsAmount

    if (captchaEnabled && memActive) {
        GuiControl, Show, DetectCaptcha
        GuiControl, Enable, DetectCaptcha
        GuiControl,, DetectCaptcha, Detect Captcha (%captchaLabel%)
    } else {
        GuiControl,, DetectCaptcha, 0
        DetectCaptcha := 0
        GuiControl, Hide, DetectCaptcha
    }

    if (memActive && warperCoordsSet) {
        GuiControl, Show, TimeOnLocationTextLabel
        GuiControl, Show, TimeOnLocation
        GuiControl, Show, TimeOnLocationValueText
        GuiControl, Enable, TimeOnLocation
    } else {
        GuiControl, Hide, TimeOnLocationTextLabel
        GuiControl, Hide, TimeOnLocation
        GuiControl, Hide, TimeOnLocationValueText
        GuiControl, Disable, TimeOnLocation
    }
}

F12::
    if (botRunning)
        Gosub, StopBotProcedure
    else
        Gosub, StartBotProcedure
return
