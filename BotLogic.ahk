SendMode Input
CoordMode, Mouse, Screen

global etc_img := "images\etc_img.bmp"
global eqp_img := "images\eqp_img.bmp"
global use_img := "images\use_img.bmp"
global close_img := "images\close_img.bmp"
global cell1_img := "images\cell1_img.bmp"
global flywing_img := "images\wing_img.bmp"
global ok_img := "images\ok_img.bmp"
global empty_cell_img := "images\empty_cell_img.bmp"

global cellSize = 50
global wingcount := 0

; Game variables
global maxSp := 0
global currentSp := 0
global currentWeight := 0
global totalWeight := 0
global currentLocation := 0

StartBot(){
    if (MemoryFeaturesActive()) {
        totalWeight := ReadMemoryUInt(gameProcess, totalWeightAddress)
        UpdateGameStats()

        if (currentLocation == 0 || currentWeight == 0 || maxSp == 0) {
            MsgBox % "Failed to initialize game variables!`nCheck the client profile and memory addresses."
            return false
        }
    }

    ZoomOut()
    skillSC := GetKeySC(SkillButtonKey) + 0
    teleportSC := GetKeySC(TeleportButtonKey) + 0
    if IsFunc("AppendLog") {
        AppendLog("Bot hunt loop started")
        if (teleportSC = 0)
            AppendLog("WARNING: Teleport hotkey is not set")
        if (skillSC = 0)
            AppendLog("WARNING: Attack hotkey is not set")
    }
    while(botRunning) {
        if (!botRunning)
            break
        while (botPaused && botRunning)
            Sleep 100
        if (!botRunning)
            break
        if(SkillTimerButtonKey != ""){
            SendKeyCombo(SkillTimerButtonKey)
        }

        if (MemoryFeaturesActive()) {
            if(warperCoordsSet && (currentLocation == warperLocation)){ 
                if IsFunc("AppendLog")
                    AppendLog("At warper — moving to hunt map")
                MoveToTheMap(warperX, warperY)
            } 

            if(currentLocation != warperLocation){
                Hunt(skillSC, teleportSC) 
            } else if (warperCoordsSet && IsFunc("AppendLog")) {
                AppendLog("Still at warper — update location or clear warper coords to hunt")
                Sleep 1000
            }
        } else {
            Hunt(skillSC, teleportSC)
        }
        iterations++
    }
}

Hunt(skillSC, teleportSC) {
    static lastWarpTime := 0
    static lastSkillTime := 0
    static attackedXs := []
    static attackedYs := []
    static attackCounts := []
    static unreachableXs := []
    static unreachableYs := []
    static emptyScans := 0
    static fastIdle := false

    attackedRadiusPx := 72
    attacksBeforeUnreachable := 3
    postAttackSleepMs := 50
    emptyScanSleepMs := 25
    fastIdleScanSleepMs := 15
    killWaitScanSleepMs := 8

    SyncSearchRangeFromUI()

    if (teleportSC = 0) {
        if IsFunc("AppendLog")
            AppendLog("Hunt: Teleport key not set — bot cannot move")
        return
    }

    if (lastWarpTime == 0) {
        lastWarpTime := A_TickCount
        lastSkillTime := A_TickCount
    }

    while(botRunning && !botPaused) {
        if (MemoryFeaturesActive())
            UpdateGameStats()

        if (MemoryFeaturesActive() && DetectCaptcha && captchaEnabled && DetectCAPTCHA()){
            botRunning := false
            break
        }

        if (SkillTimerButtonKey != "" && (A_TickCount - lastSkillTime) >= (SkillTimerInterval * 1000)) {
            SendKeyCombo(SkillTimerButtonKey)
            lastSkillTime := A_TickCount
            Sleep 300
        }

        if (MemoryFeaturesActive() && warperCoordsSet && SavePointButtonKey != "" && (A_TickCount - lastWarpTime) >= (TimeOnLocation * 1000)) {
            WarpToSavePoint()
            lastWarpTime := A_TickCount
            Sleep 1000
            break
        }

        if (MemoryFeaturesActive() && WeightModifier >= 50 && currentWeight >= (totalWeight * WeightModifier / 100)) {
            ItemsToStorage()
            currentWeight := ReadMemoryUInt(gameProcess,currentWeightAddress)
        }

        if (MemoryFeaturesActive() && wingcount <= 0 && TakeFlyWings){
            GetFlyWings()
        }

        GetHuntSearchRegion(xs, ys, ws, hs)
        if (ws <= 0 || hs <= 0) {
            if IsFunc("AppendLog")
                AppendLog("Hunt: invalid search region — select game window and refresh")
            Sleep 500
            continue
        }

        mobName := MobTemplateFolderName()
        if IsFunc("SessionLogHuntScan") {
            fn := "SessionLogHuntScan"
            %fn%(mobName, xs, ys, ws, hs)
        }

        jsonText := MobRecognitionHuntScan(mobName, xs, ys, ws, hs, attackedXs, attackedYs, unreachableXs, unreachableYs, emptyScans, attackCounts, fastIdle, false)
        if (jsonText = "") {
            Sleep 100
            continue
        }

        livingInRange := 0
        canTeleport := false
        attackX := 0
        attackY := 0
        attackConf := 0
        huntStatus := ""
        engagementsResolved := true
        teleportScansRequired := 6
        if (!MobRecognitionParseHuntPlan(jsonText, livingInRange, canTeleport, attackX, attackY, attackConf, huntStatus, engagementsResolved, teleportScansRequired)) {
            if IsFunc("AppendLog")
                AppendLog("Hunt: stale detect server — restarting")
            MobRecognitionShutdownServer()
            Sleep 200
            continue
        }

        GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)
        MobRecognitionApplyHuntMarkUnreachable(jsonText, unreachableXs, unreachableYs, attackedRadiusPx)
        MobRecognitionUpdateUnreachableFromScan(jsonText, attackedXs, attackedYs, attackCounts, unreachableXs, unreachableYs, ignoreX, ignoreY, ignoreW, ignoreH, attackedRadiusPx, attacksBeforeUnreachable)

        if (attackX != 0 && attackY != 0) {
            emptyScans := 0
            fastIdle := false
            if IsFunc("AppendLog")
                AppendLog("Hunt [" . huntStatus . "]: attack @" . attackX . "," . attackY . " conf=" . attackConf . " living=" . livingInRange)
            MoveMouseTo(attackX, attackY)
            HuntSkillClick(skillSC)
            MobRecognitionRecordAttackSlot(attackX, attackY, attackedXs, attackedYs, attackCounts, attackedRadiusPx)
            Sleep %postAttackSleepMs%
            continue
        }

        scanSleepMs := emptyScanSleepMs
        if (fastIdle && attackedXs.MaxIndex() = 0)
            scanSleepMs := fastIdleScanSleepMs
        else if (attackedXs.MaxIndex() = 0)
            scanSleepMs := fastIdleScanSleepMs

        if (livingInRange > 0) {
            if IsFunc("AppendLog")
                AppendLog("Hunt [" . huntStatus . "]: " . livingInRange . " living mob(s) — keep hunting")
            emptyScans := 0
            fastIdle := false
            Sleep %scanSleepMs%
            continue
        }

        if (!engagementsResolved) {
            emptyScans++
            if IsFunc("AppendLog")
                AppendLog("Hunt [" . huntStatus . "]: waiting for kill confirmation (" . emptyScans . ")")
            Sleep %killWaitScanSleepMs%
            continue
        }

        emptyScans++
        if IsFunc("AppendLog")
            AppendLog("Hunt [" . huntStatus . "]: clear scan " . emptyScans . "/" . teleportScansRequired)

        if (canTeleport) {
            if IsFunc("AppendLog")
                AppendLog("Hunt: area clear — teleporting")
            Teleport(teleportSC)
            attackedXs := []
            attackedYs := []
            attackCounts := []
            unreachableXs := []
            unreachableYs := []
            emptyScans := 0
            fastIdle := true
            Sleep 40
            continue
        }

        Sleep %scanSleepMs%
    }
}

Teleport(teleportSC){
    Input.SendKey(teleportSC, 1)
    sleep 50
    Input.SendKey(teleportSC, 0)
    sleep 400
    if(TakeFlyWings && MemoryFeaturesActive()){
        wingcount--
    }
}

MoveToTheMap(posX, posY) {
    MoveMouseTo(posX, posY)
    Sleep 500
    Input.SendMouseButton(0, 1)
    sleep 50
    Input.SendMouseButton(0, 0)
    Sleep 500
    enterSC := GetKeySC("Enter") + 0
    Input.SendKey(enterSC, 1)
    sleep 50
    Input.SendKey(enterSC, 0)
    Sleep 2000
    UpdateGameStats()
}

WarpToSavePoint() {
    SendKeyCombo(SavePointButtonKey)
    Sleep 2000
    UpdateGameStats()
}

GetFlyWings() {
    sleep 100
    ManageInventoryWindow()
    MoveCursorToImage(cell1_img,0,40)
    if !SendKeyCombo(OpenStorageButtonKey) {
        return false
    }
    Sleep 800
    if(CheckInventoryCell(flywing_img)){
        AltClicks(1)

    }
    sleep 500
    MoveCursorToImage(flywing_img)
    sleep 100
    Input.SendMouseButton(0, 1)
    sleep 100
    MoveCursorToImage(etc_img,100,20)
    Input.SendMouseButton(0, 0)
    sleep 200
    send %wingsTaken%
    sleep 200
    enterSC := GetKeySC("Enter") + 0
    Input.SendKey(enterSC, 1)
    sleep 50
    Input.SendKey(enterSC, 0)
    ManageInventoryWindow()
    MoveCursorToImage(close_img)
    sleep 200
    InputClick()
    wingcount := wingsTaken
    sleep 200
}

ManageInventoryWindow(){
    action := "open"
    if(action = "close"){
        ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, etc_img

    }
    Input.SendKey(56, 1)
    sleep 50
    Input.SendKey(18, 1)
    sleep 50
    Input.SendKey(18, 0)
    sleep 50
    Input.SendKey(56, 0)
    sleep 500
}

DetectCAPTCHA() {
    global xs, ys, ws, hs, captchaColor
    PixelSearch, x, y, xs, ys, xs + ws, ys + hs, %captchaColor%, 1, Fast RGB 
    if (ErrorLevel = 0) {
        Loop,8{
            SoundBeep, 750, 1000
            sleep 500
        }
        Pause, On
        return true
    }
    return false
}

CheckInventoryCell(image, ignoreWing := true) {
    MouseGetPos, currentX, currentY

    cellSize := 40
    searchLeft := currentX - cellSize//2
    searchTop := currentY - cellSize//2
    searchRight := currentX + cellSize//2
    searchBottom := currentY + cellSize//2

    ImageSearch, FoundX, FoundY, searchLeft, searchTop, searchRight, searchBottom, %image%

    if (ErrorLevel = 0) {
        if(image == flywing_img && ignoreWing == false){
            nextCellX := currentX + cellSize
            nextCellY := currentY

            maxRight := A_ScreenWidth - cellSize//2
            if (nextCellX > maxRight) {
                nextCellX := cellSize//2
                nextCellY += cellSize
            }
            MoveMouseTo(nextCellX, nextCellY)
        }
        return true
    }

    return false
}

ItemsToStorage(){
    Sleep 500
    ManageInventoryWindow()
    sleep 500
    MoveCursorToImage(use_img)
    Sleep 100
    InputClick()
    SendKeyCombo(OpenStorageButtonKey)
    MoveCursorToImage(cell1_img,0,40)
    while(!CheckInventoryCell(empty_cell_img)){
        CheckInventoryCell(flywing_img, false)
        AltClicks(1)
        sleep 50
    }
    sleep 100
    MoveCursorToImage(eqp_img)
    sleep 100
    InputClick()
    sleep 50
    MoveCursorToImage(cell1_img,0,40)
    while(!CheckInventoryCell(empty_cell_img)){
        AltClicks(1)
        sleep 50
    }

    MoveCursorToImage(etc_img)
    sleep 100
    InputClick()
    sleep 100
    MoveCursorToImage(cell1_img,0,40)
    while(!CheckInventoryCell(empty_cell_img)){
        sleep 50
        if(CheckImageOnScreen(ok_img)){
            Input.SendKey(284, 1)
            sleep 50
            Input.SendKey(284, 0)
            MouseGetPos, currentX, currentY
            MoveMouseTo(currentX + 40, currentY)
        }
        AltClicks(1)
    }
    sleep 100
    MoveCursorToImage(close_img,10,10)
    sleep 100
    InputClick()
    ManageInventoryWindow()
    sleep 500
}
