LogToFile(message) {
    FormatTime, timestamp,, yyyy-MM-dd HH:mm:ss
    FileAppend, [%timestamp%] %message%`n, bot_log.txt
}

AppendLog(message) {
    static logLines := []

    if IsFunc("SessionLogWrite") {
        fn := "SessionLogWrite"
        %fn%("INFO", "app", message)
    }

    FormatTime, timestamp,, HH:mm:ss
    line := "[" . timestamp . "] " . message
    LogToFile(message)

    logLines.Push(line)
    if (logLines.MaxIndex() > 200)
        logLines.RemoveAt(1)

    newText := ""
    for index, entry in logLines
        newText .= (newText = "" ? "" : "`r`n") . entry

    GuiControl,, LogBox, %newText%
    if (LogBoxHwnd)
        SendMessage, 0x115, 7, 0,, ahk_id %LogBoxHwnd%
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

SetDefaultKeyboardLayout(layout)
{
    DllCall("LoadKeyboardLayout", Str, layout, UInt, 1)
    DllCall("ActivateKeyboardLayout", UInt, DllCall("LoadKeyboardLayout", Str, layout, UInt, 1), UInt, 0)
}

MoveMouseTo(x, y) {
    DllCall("SetCursorPos", "Int", x, "Int", y)
    Sleep, 5
}

GetSearchBoxSizePx() {
    global SearchRange, cellSize
    return SearchRange * cellSize
}

SyncSearchRangeFromUI() {
    global SearchRange

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

    searchSize := GetSearchBoxSizePx()

    if (gameWindowID && WinExist("ahk_id " gameWindowID)) {
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
    } else {
        ws := searchSize
        hs := searchSize
        xs := A_ScreenWidth // 2 - ws // 2
        ys := A_ScreenHeight // 2 - hs // 2
    }
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

ColorToHex(color) {
    SetFormat, IntegerFast, Hex
    hex := SubStr(color + 0x1000000, 2, 6)
    SetFormat, IntegerFast, D
    return hex
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

TestMonsterSearch() {
    global gameWindowID, gameWindowTitle, gameProcess, SearchRange
    global cellSize, MobNames, selectedMonsterIndex

    SyncSearchRangeFromUI()
    AppendLog("--- Monster search test ---")

    if (!gameWindowID || !WinExist("ahk_id " gameWindowID)) {
        MsgBox, 48, Test Search, No game window selected.`n`nSelect your client.exe window and click Refresh first.
        AppendLog("TEST FAIL: No valid game window (select one and Refresh)")
        return
    }

    RestoreWindow()
    Sleep 300

    WinGetTitle, wTitle, ahk_id %gameWindowID%
    WinGet, wProcess, ProcessName, ahk_id %gameWindowID%
    WinGetPos, wx, wy, ww, wh, ahk_id %gameWindowID%
    ControlGetPos, cx, cy, cw, ch, , ahk_id %gameWindowID%

    GetHuntSearchRegion(xs, ys, ws, hs)
    x2 := xs + ws
    y2 := ys + hs

    mobName := MobNames[selectedMonsterIndex]
    mobFolder := MobTemplateFolderName()

    AppendLog("TEST: hwnd=" . gameWindowID . " process=" . wProcess)
    AppendLog("TEST: title=" . wTitle)
    AppendLog("TEST: window rect " . wx . "," . wy . " size " . ww . "x" . wh)
    AppendLog("TEST: client rect " . cx . "," . cy . " size " . cw . "x" . ch)
    AppendLog("TEST: search box " . xs . "," . ys . " to " . x2 . "," . y2 . " (" . ws . "x" . hs . ")")
    AppendLog("TEST: monster=" . mobName . " templates=" . mobFolder)

    searchCenterX := xs + (ws // 2)
    searchCenterY := ys + (hs // 2)
    clientCenterX := cx + (cw // 2)
    clientCenterY := cy + (ch // 2)

    regionWarning := ""
    if (searchCenterX < cx || searchCenterX > cx + cw || searchCenterY < cy || searchCenterY > cy + ch) {
        regionWarning .= "`n- Search center is outside the game area (title bar offset)"
        AppendLog("TEST: WARNING — search center is outside the client (game) area")
    }
    if (x2 < cx || xs > cx + cw || y2 < cy || ys > cy + ch) {
        regionWarning .= "`n- Search box does not overlap the game area"
        AppendLog("TEST: WARNING — search box does not overlap the client area")
    }

    foundX := 0
    foundY := 0
    found := FindMobTarget(foundX, foundY, xs, ys, ws, hs)
    if (!found) {
        matchResult := "No OpenCV match for " . mobName
        AppendLog("TEST: OpenCV — no match")
    } else {
        matchResult := "Match at " . foundX . "," . foundY
        AppendLog("TEST: OpenCV — match at " . foundX . "," . foundY)
    }

    AppendLog("TEST: red border shows where the bot searches for 3 seconds")
    ShowSearchRegionOverlay(xs, ys, ws, hs)

    summary := "Monster: " . mobName
    summary .= "`nDescriptor: generated_descriptors\" . mobFolder . "\simple"
    summary .= "`nSearch range: " . SearchRange . " cells = " . ws . "x" . hs . " px"
    summary .= "`nSearch box: " . xs . "," . ys . " to " . x2 . "," . y2
    summary .= "`n`nResult: " . matchResult
    if (regionWarning != "")
        summary .= "`n`nRegion warnings:" . regionWarning
    summary .= "`n`nRed border = OpenCV search region. Full details in the Log panel."

    MsgBox, 64, Test Search, %summary%
}

InputClick(){
    Input.SendMouseButton(0, 1)
    sleep 50
    Input.SendMouseButton(0, 0)
}

AltClicks(times){
    Input.SendKey(56, 1)
    sleep 50
    Loop, %times%{
        Input.SendMouseButton(1, 1)
        sleep 50
        Input.SendMouseButton(1, 0)
        sleep 50
    }
    Input.SendKey(56, 0)
}

SkillClick(KeySC){
    sleep 50
    Input.SendKey(KeySC, 1)
    sleep 50
    Input.SendKey(KeySC, 0)
    Input.SendMouseButton(0, 1)
    sleep 50
    Input.SendMouseButton(0, 0)
}

HuntSkillClick(KeySC) {
    Input.SendKey(KeySC, 1)
    Sleep 20
    Input.SendKey(KeySC, 0)
    Input.SendMouseButton(0, 1)
    Sleep 20
    Input.SendMouseButton(0, 0)
}

CheckImageOnScreen(image){
    ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, %image%
    if (ErrorLevel = 0) {
        return true
    }
    return false
}

MoveCursorToImage(image, xOffset := 0, yOffset := 0){
    ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, %image%
    if (ErrorLevel = 0) {
        ; Image found, move the cursor
        MoveMouseTo(FoundX + xOffset, FoundY + yOffset)
    } else if (ErrorLevel = 1) {
        MsgBox, Image was not found or an error occurred.
        Pause, On
    }
    sleep 200
}

ZoomOut(){
    global zoomWheelDelta, zoomSteps, zoomDelayMs, gameWindowID, gameWindowTitle

    if (gameWindowID) {
        WinGet, activeID, ID, A
        if (activeID != gameWindowID) {
            WinActivate, ahk_id %gameWindowID%
            WinWaitActive, ahk_id %gameWindowID%, , 2
            Sleep 200
        }
    }

    if (IsFunc("AppendLog"))
        AppendLog("Zooming out (" . (zoomWheelDelta > 0 ? "scroll up" : "scroll down") . ")")

    Loop %zoomSteps% {
        if (zoomWheelDelta > 0)
            Click, WheelUp
        else
            Click, WheelDown
        Sleep %zoomDelayMs%
    }
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

    ; Parse the key combination
    scKey := 0, scModifier := 0
    if !%keyParser%(rawKeyCombo, scKey, scModifier) {
        MsgBox, Failed to parse key combination: %rawKeyCombo%
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
        Sleep holdDuration
    }

    ; Press and release main key
    Input.SendKey(scKey, 1)
    Sleep pressDuration
    Input.SendKey(scKey, 0)
    Sleep holdDuration

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
