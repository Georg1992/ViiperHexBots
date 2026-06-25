AppendLog(message) {
    static logLines := []

    if IsFunc("BehaviorLogWrite") {
        fn := "BehaviorLogWrite"
        %fn%(message)
    }

    FormatTime, timestamp,, HH:mm:ss
    line := "[" . timestamp . "] " . message

    logLines.Push(line)
    if (logLines.MaxIndex() > 200)
        logLines.RemoveAt(1)

    newText := ""
    for index, entry in logLines
        newText .= (newText = "" ? "" : "`r`n") . entry

    GuiControl,, LogBox, %newText%
    if (LogBoxHwnd)
        SendMessage, 0x115, 7, 0,, ahk_id %LogBoxHwnd%
    if IsFunc("HuntLogOverlay_OnLog")
        HuntLogOverlay_OnLog(line, message)
}

SetInputStatus(statusText, hintText := "") {
    GuiControl,, InputStatus, %statusText%
    if (hintText != "")
        GuiControl,, InputHint, %hintText%
}

FormatWindowListEntry(title, process, minMaxStatus := 0) {
    safeTitle := StrReplace(title, "|", "-")
    stateSymbol := (minMaxStatus = -1) ? "[MIN] " : ""
    return stateSymbol . safeTitle . " (" . process . ")"
}

BotShouldStop() {
    global botStopRequested
    return botStopRequested
}

RequestBotStop(reason := "stopped") {
    global botRunning, botPaused, botStopRequested

    botStopRequested := true
    botPaused := false
    ReleaseBotInputs()
    if IsFunc("AppendLog")
        AppendLog("Bot stop requested: " . reason)

    SetTimer, StopBotProcedure, -1
    return false
}

BotSleep(ms) {
    deadline := A_TickCount + ms
    while (A_TickCount < deadline) {
        if (BotShouldStop()) {
            ReleaseBotInputs()
            return false
        }
        remaining := deadline - A_TickCount
        chunk := remaining < 50 ? remaining : 50
        Sleep %chunk%
    }
    return true
}

SetDefaultKeyboardLayout(layout)
{
    DllCall("LoadKeyboardLayout", Str, layout, UInt, 1)
    DllCall("ActivateKeyboardLayout", UInt, DllCall("LoadKeyboardLayout", Str, layout, UInt, 1), UInt, 0)
}

MoveMouseTo(x, y) {
    if (BotShouldStop())
        return false
    DllCall("SetCursorPos", "Int", x, "Int", y)
    BotSleep(5)
    return true
}

GetSearchBoxSizePx() {
    global SearchRange, cellSize
    return SearchRange * cellSize
}

SyncSearchRangeFromUI() {
    global SearchRange, botRunning

    ; Disabled sliders return empty/stale values (same issue as hotkeys during bot run).
    if (botRunning)
        return

    GuiControlGet, sliderValue,, SearchRange
    SearchRange := sliderValue + 0
    if (SearchRange < 9)
        SearchRange := 9
    if (SearchRange > 16)
        SearchRange := 16
    UpdateSearchRangeLabel()
}

UpdateSearchRangeLabel() {
    global SearchRange, cellSize
    sizePx := GetSearchBoxSizePx()
    GuiControl,, SearchRangeText, %SearchRange% " (" . sizePx . "px)"
}

GetHuntSearchRegion(ByRef xs, ByRef ys, ByRef ws, ByRef hs) {
    global SearchRange, cellSize, gameWindowID

    xs := 0
    ys := 0
    ws := 0
    hs := 0

    if (!gameWindowID || !WinExist("ahk_id " gameWindowID))
        return false

    searchSize := GetSearchBoxSizePx()
    WinGetPos, wx, wy, , , ahk_id %gameWindowID%
    ControlGetPos, cx, cy, cw, ch, , ahk_id %gameWindowID%
    clientLeft := wx + cx
    clientTop := wy + cy

    ws := searchSize
    hs := searchSize
    xs := clientLeft + (cw // 2) - (ws // 2)
    ys := clientTop + (ch // 2) - (hs // 2)

    if (xs < clientLeft)
        xs := clientLeft
    if (ys < clientTop)
        ys := clientTop
    if (xs + ws > clientLeft + cw)
        xs := clientLeft + cw - ws
    if (ys + hs > clientTop + ch)
        ys := clientTop + ch - hs

    return (ws > 0 && hs > 0)
}

RestoreWindow() {
    global gameWindowID

    if (!gameWindowID || !WinExist("ahk_id " gameWindowID))
        return

    WinGet, minMaxStatus, MinMax, ahk_id %gameWindowID%
    if (minMaxStatus = -1) {
        WinRestore, ahk_id %gameWindowID%
        Sleep, 1000
    }

    WinActivate, ahk_id %gameWindowID%
    WinWaitActive, ahk_id %gameWindowID%, , 2

    WinGet, activeID, ID, A
    if (activeID != gameWindowID) {
        WinSet, Style, -0x8000000, ahk_id %gameWindowID%
        WinActivate, ahk_id %gameWindowID%
        WinWaitActive, ahk_id %gameWindowID%, , 3
        if ErrorLevel {
            MsgBox, 16, Error, Failed to activate game window!`nTry running as Administrator.
            return
        }
    }
}

ShowSearchRegionOverlay(x, y, w, h, durationMs := 3000) {
    HideSearchRegionOverlay()
    border := 3
    bottomY := y + h - border
    rightX := x + w - border

    Gui, SFTop:Destroy
    Gui, SFTop:+AlwaysOnTop -Caption +ToolWindow +E0x20
    Gui, SFTop:Color, Red
    Gui, SFTop:Show, x%x% y%y% w%w% h%border% NA

    Gui, SFBot:Destroy
    Gui, SFBot:+AlwaysOnTop -Caption +ToolWindow +E0x20
    Gui, SFBot:Color, Red
    Gui, SFBot:Show, x%x% y%bottomY% w%w% h%border% NA

    Gui, SFLeft:Destroy
    Gui, SFLeft:+AlwaysOnTop -Caption +ToolWindow +E0x20
    Gui, SFLeft:Color, Red
    Gui, SFLeft:Show, x%x% y%y% w%border% h%h% NA

    Gui, SFRight:Destroy
    Gui, SFRight:+AlwaysOnTop -Caption +ToolWindow +E0x20
    Gui, SFRight:Color, Red
    Gui, SFRight:Show, x%rightX% y%y% w%border% h%h% NA

    SetTimer, HideSearchRegionOverlay, -%durationMs%
}

HideSearchRegionOverlay() {
    Gui, SFTop:Destroy
    Gui, SFBot:Destroy
    Gui, SFLeft:Destroy
    Gui, SFRight:Destroy
}

InputClick(){
    if (BotShouldStop())
        return false
    Input.SendMouseButton(0, 1)
    if (!BotSleep(50)) {
        Input.SendMouseButton(0, 0)
        return false
    }
    Input.SendMouseButton(0, 0)
    return true
}

AltClicks(times){
    if (BotShouldStop())
        return false
    Input.SendKey(56, 1)
    if (!BotSleep(50)) {
        Input.SendKey(56, 0)
        return false
    }
    Loop, %times%{
        if (BotShouldStop())
            break
        Input.SendMouseButton(1, 1)
        if (!BotSleep(50))
            break
        Input.SendMouseButton(1, 0)
        if (!BotSleep(50))
            break
    }
    Input.SendKey(56, 0)
    return !BotShouldStop()
}

HuntSkillClick(KeySC) {
    if (BotShouldStop())
        return false
    if (!KeySC) {
        if IsFunc("AppendLog")
            AppendLog("[HUNT] skill key not set")
        return false
    }
    Input.SendKey(KeySC, 1)
    if (!BotSleep(20)) {
        Input.SendKey(KeySC, 0)
        Input.SendMouseButton(0, 0)
        return false
    }
    Input.SendMouseButton(0, 1)
    if (!BotSleep(20)) {
        Input.SendKey(KeySC, 0)
        Input.SendMouseButton(0, 0)
        return false
    }
    Input.SendMouseButton(0, 0)
    Input.SendKey(KeySC, 0)
    return true
}

ReleaseBotInputs() {
    global SkillButtonKey, TeleportButtonKey, SavePointButtonKey, SkillTimerButtonKey
    keys := [SkillButtonKey, TeleportButtonKey, SavePointButtonKey, SkillTimerButtonKey]
    for index, keyName in keys {
        if (keyName = "")
            continue
        keySC := GetKeySC(keyName) + 0
        if (keySC > 0)
            Input.SendKey(keySC, 0)
    }
    Input.SendMouseButton(0, 0)
    Input.SendMouseButton(1, 0)
}

CheckImageOnScreen(image){
    ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, %image%
    if (ErrorLevel = 0) {
        return true
    }
    return false
}

MoveCursorToImage(image, xOffset := 0, yOffset := 0){
    if (BotShouldStop())
        return false
    ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, %image%
    if (ErrorLevel = 0) {
        ; Image found, move the cursor
        return MoveMouseTo(FoundX + xOffset, FoundY + yOffset)
    } else if (ErrorLevel = 1) {
        if IsFunc("AppendLog")
            AppendLog("Image not found: " . image)
    }
    BotSleep(200)
    return false
}

ZoomOut(){
    global zoomWheelDelta, zoomSteps, zoomDelayMs, gameWindowID, gameWindowTitle

    if (BotShouldStop())
        return false

    if (gameWindowID) {
        WinGet, activeID, ID, A
        if (activeID != gameWindowID) {
            WinActivate, ahk_id %gameWindowID%
            WinWaitActive, ahk_id %gameWindowID%, , 2
            if (!BotSleep(200))
                return false
        }
    }

    if (IsFunc("AppendLog"))
        AppendLog("Zooming out")

    Loop %zoomSteps% {
        if (BotShouldStop())
            return false
        if (zoomWheelDelta > 0)
            Click, WheelUp
        else
            Click, WheelDown
        if (!BotSleep(zoomDelayMs))
            return false
    }
    return true
}

ParseKeyCombo(keyCombo, ByRef scKey, ByRef scModifier) {
    scKey := 0
    scModifier := 0
    keyCombo := Trim(StrReplace(keyCombo, " ", ""))

    ; Check for modifiers
    if (InStr(keyCombo, "!") && InStr(keyCombo, "+")) { ; Alt+Shift+Key
        scModifier := GetKeySC("LAlt") || GetKeySC("LShift")
        keyCombo := RegExReplace(keyCombo, "[!\+]")
    } else if (InStr(keyCombo, "!")) { ; Alt+Key
        scModifier := GetKeySC("LAlt")
        keyCombo := StrReplace(keyCombo, "!")
    } else if (InStr(keyCombo, "+")) { ; Shift+Key
        scModifier := GetKeySC("LShift")
        keyCombo := StrReplace(keyCombo, "+")
    } else if (InStr(keyCombo, "^")) { ; Ctrl+Key
        scModifier := GetKeySC("LCtrl")
        keyCombo := StrReplace(keyCombo, "^")
    }

    ; Get SC for the main key
    if (keyCombo != "") {
        scKey := GetKeySC(keyCombo) + 0
    }

    return (scKey != 0)
}

SendKeyCombo(rawKeyCombo, pressDuration := 50, holdDuration := 300) {
    static keyParser := Func("ParseKeyCombo") ; Cache the function reference

    if (BotShouldStop())
        return false

    ; Parse the key combination
    scKey := 0, scModifier := 0
    if !%keyParser%(rawKeyCombo, scKey, scModifier) {
        if IsFunc("AppendLog")
            AppendLog("Failed to parse key combination: " . rawKeyCombo)
        return false
    }

    ; Clear any stuck modifiers first
    ClearModifiers()

    ; Press modifiers if needed
    if (scModifier) {
        if (scModifier & GetKeySC("LAlt"))
            Input.SendKey(GetKeySC("LAlt"), 1)
        if (scModifier & GetKeySC("LShift"))
            Input.SendKey(GetKeySC("LShift"), 1)
        if (scModifier & GetKeySC("LCtrl"))
            Input.SendKey(GetKeySC("LCtrl"), 1)
        if (!BotSleep(holdDuration)) {
            ClearModifiers()
            return false
        }
    }

    ; Press and release main key
    if (BotShouldStop()) {
        ClearModifiers()
        return false
    }
    Input.SendKey(scKey, 1)
    if (!BotSleep(pressDuration)) {
        Input.SendKey(scKey, 0)
        ClearModifiers()
        return false
    }
    Input.SendKey(scKey, 0)
    if (!BotSleep(holdDuration)) {
        ClearModifiers()
        return false
    }

    ; Release modifiers
    ClearModifiers()
    return true
}

ClearModifiers() {
    static modifiers := [GetKeySC("LAlt"), GetKeySC("LShift"), GetKeySC("LCtrl")]
    for each, modSC in modifiers {
        Input.SendKey(modSC, 0)
    }
    Sleep 30
}
