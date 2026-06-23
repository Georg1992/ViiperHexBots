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
global huntLastWarpTime := 0
global huntLastSkillTime := 0
global huntAttackedXs := []
global huntAttackedYs := []
global huntAttackCounts := []
global huntUnreachableXs := []
global huntUnreachableYs := []
global huntEmptyScans := 0
global huntFastIdle := false

HuntStateReset(resetWarpTimer := true) {
    global huntLastWarpTime, huntLastSkillTime
    global huntAttackedXs, huntAttackedYs, huntAttackCounts
    global huntUnreachableXs, huntUnreachableYs, huntEmptyScans, huntFastIdle

    if (resetWarpTimer)
        huntLastWarpTime := 0
    huntLastSkillTime := 0
    huntAttackedXs := []
    huntAttackedYs := []
    huntAttackCounts := []
    huntUnreachableXs := []
    huntUnreachableYs := []
    huntEmptyScans := 0
    huntFastIdle := false
}

StartBot(){
    global botRunning, botPaused, botStopRequested
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
    while(botRunning && !botStopRequested) {
        if (!botRunning || botStopRequested)
            break
        while (botPaused && botRunning && !botStopRequested)
            BotSleep(100)
        if (!botRunning || botStopRequested)
            break
        if(SkillTimerButtonKey != ""){
            if (!SendKeyCombo(SkillTimerButtonKey))
                break
        }

        if (MemoryFeaturesActive()) {
            if(warperCoordsSet && (currentLocation == warperLocation)){ 
                if IsFunc("AppendLog")
                    AppendLog("At warper — moving to hunt map")
                MoveToTheMap(warperX, warperY)
            } 

            if (!botRunning || botStopRequested)
                break

            if(currentLocation != warperLocation){
                Hunt(skillSC, teleportSC) 
            } else if (warperCoordsSet && IsFunc("AppendLog")) {
                AppendLog("Still at warper — update location or clear warper coords to hunt")
                BotSleep(1000)
            }
        } else {
            Hunt(skillSC, teleportSC)
        }
        iterations++
    }
    BotSessionStop("loop ended")
}

Hunt(skillSC, teleportSC) {
    global botRunning, botPaused, botStopRequested
    global huntLastWarpTime, huntLastSkillTime
    global huntAttackedXs, huntAttackedYs, huntAttackCounts
    global huntUnreachableXs, huntUnreachableYs, huntEmptyScans, huntFastIdle

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

    if (huntLastWarpTime == 0) {
        huntLastWarpTime := A_TickCount
        huntLastSkillTime := A_TickCount
    }

    while(botRunning && !botPaused && !botStopRequested) {
        if (MemoryFeaturesActive())
            UpdateGameStats()

        if (SkillTimerButtonKey != "" && (A_TickCount - huntLastSkillTime) >= (SkillTimerInterval * 1000)) {
            if (!SendKeyCombo(SkillTimerButtonKey))
                break
            huntLastSkillTime := A_TickCount
            BotSleep(300)
        }

        if (MemoryFeaturesActive() && warperCoordsSet && SavePointButtonKey != "" && (A_TickCount - huntLastWarpTime) >= (TimeOnLocation * 1000)) {
            WarpToSavePoint()
            huntLastWarpTime := A_TickCount
            BotSleep(1000)
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
        if (!ws || !hs) {
            if IsFunc("AppendLog")
                AppendLog("Hunt: invalid search region — select game window and refresh")
            BotSleep(500)
            continue
        }

        if (MemoryFeaturesActive() && DetectCaptcha && captchaEnabled && DetectCAPTCHA(xs, ys, ws, hs)) {
            RequestBotStop("captcha detected")
            break
        }

        mobName := MobTemplateFolderName()
        if IsFunc("BotSessionHuntScan") {
            fn := "BotSessionHuntScan"
            %fn%(mobName, xs, ys, ws, hs)
        }

        jsonText := MobRecognitionHuntScan(mobName, xs, ys, ws, hs, huntAttackedXs, huntAttackedYs, huntUnreachableXs, huntUnreachableYs, huntEmptyScans, huntAttackCounts, huntFastIdle, false)
        if (!botRunning || botStopRequested)
            break
        if (jsonText = "") {
            BotSleep(100)
            continue
        }

        livingInRange := 0
        totalLivingInRange := 0
        canTeleport := false
        attackX := 0
        attackY := 0
        attackConf := 0
        huntStatus := ""
        engagementsResolved := true
        teleportScansRequired := 6
        if (!MobRecognitionParseHuntPlan(jsonText, livingInRange, canTeleport, attackX, attackY, attackConf, huntStatus, engagementsResolved, teleportScansRequired)) {
            if IsFunc("AppendLog")
                AppendLog("Hunt: invalid detect response — retrying")
            BotSleep(200)
            continue
        }

        if (RegExMatch(jsonText, "i)""totalLivingInRange"":(\d+)", match))
            totalLivingInRange := match1 + 0

        GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)
        MobRecognitionApplyHuntMarkUnreachable(jsonText, huntUnreachableXs, huntUnreachableYs, attackedRadiusPx)
        MobRecognitionUpdateUnreachableFromScan(jsonText, huntAttackedXs, huntAttackedYs, huntAttackCounts, huntUnreachableXs, huntUnreachableYs, ignoreX, ignoreY, ignoreW, ignoreH, attackedRadiusPx, attacksBeforeUnreachable)

        if (attackX != 0 && attackY != 0) {
            if (!botRunning || botStopRequested)
                break
            huntEmptyScans := 0
            huntFastIdle := false
            if IsFunc("AppendLog")
                AppendLog("Hunt [" . huntStatus . "]: attack @" . attackX . "," . attackY . " conf=" . attackConf . " targets=" . livingInRange . " living=" . totalLivingInRange)
            MoveMouseTo(attackX, attackY)
            HuntSkillClick(skillSC)
            MobRecognitionRecordAttackSlot(attackX, attackY, huntAttackedXs, huntAttackedYs, huntAttackCounts, attackedRadiusPx)
            BotSessionRecordAttack(attackX, attackY, attackConf)
            BotSleep(postAttackSleepMs)
            continue
        }

        scanSleepMs := emptyScanSleepMs
        if (huntFastIdle && huntAttackedXs.MaxIndex() = 0)
            scanSleepMs := fastIdleScanSleepMs
        else if (huntAttackedXs.MaxIndex() = 0)
            scanSleepMs := fastIdleScanSleepMs

        if (totalLivingInRange > 0) {
            if IsFunc("AppendLog")
                AppendLog("Hunt [" . huntStatus . "]: " . totalLivingInRange . " living mob(s) in range — keep hunting")
            huntEmptyScans := 0
            huntFastIdle := false
            BotSleep(scanSleepMs)
            continue
        }

        if (!engagementsResolved) {
            huntEmptyScans++
            if IsFunc("AppendLog")
                AppendLog("Hunt [" . huntStatus . "]: waiting for kill confirmation (" . huntEmptyScans . ")")
            BotSleep(killWaitScanSleepMs)
            continue
        }

        huntEmptyScans++
        if IsFunc("AppendLog")
            AppendLog("Hunt [" . huntStatus . "]: clear scan " . huntEmptyScans . "/" . teleportScansRequired)

        if (canTeleport) {
            if (!botRunning || botStopRequested)
                break
            if IsFunc("AppendLog")
                AppendLog("Hunt: area clear — teleporting")
            Teleport(teleportSC)
            HuntStateReset(false)
            huntFastIdle := true
            BotSleep(40)
            continue
        }

        BotSleep(scanSleepMs)
    }
}

Teleport(teleportSC){
    global botRunning, botStopRequested
    if (!botRunning || botStopRequested)
        return
    Input.SendKey(teleportSC, 1)
    if (!BotSleep(50)) {
        Input.SendKey(teleportSC, 0)
        return
    }
    if (!botRunning || botStopRequested) {
        Input.SendKey(teleportSC, 0)
        return
    }
    Input.SendKey(teleportSC, 0)
    BotSleep(400)
    if(TakeFlyWings && MemoryFeaturesActive()){
        wingcount--
    }
}

MoveToTheMap(posX, posY) {
    if (BotShouldStop())
        return false
    MoveMouseTo(posX, posY)
    if (!BotSleep(500))
        return false
    Input.SendMouseButton(0, 1)
    if (!BotSleep(50)) {
        Input.SendMouseButton(0, 0)
        return false
    }
    Input.SendMouseButton(0, 0)
    if (!BotSleep(500))
        return false
    enterSC := GetKeySC("Enter") + 0
    Input.SendKey(enterSC, 1)
    if (!BotSleep(50)) {
        Input.SendKey(enterSC, 0)
        return false
    }
    Input.SendKey(enterSC, 0)
    if (!BotSleep(2000))
        return false
    UpdateGameStats()
    return true
}

WarpToSavePoint() {
    if (!SendKeyCombo(SavePointButtonKey))
        return false
    if (!BotSleep(2000))
        return false
    UpdateGameStats()
    return true
}

GetFlyWings() {
    if (BotShouldStop())
        return false
    if (!BotSleep(100))
        return false
    ManageInventoryWindow()
    if (BotShouldStop())
        return false
    MoveCursorToImage(cell1_img,0,40)
    if !SendKeyCombo(OpenStorageButtonKey) {
        return false
    }
    if (!BotSleep(800))
        return false
    if(CheckInventoryCell(flywing_img)){
        AltClicks(1)

    }
    if (!BotSleep(500))
        return false
    MoveCursorToImage(flywing_img)
    if (!BotSleep(100))
        return false
    Input.SendMouseButton(0, 1)
    if (!BotSleep(100)) {
        Input.SendMouseButton(0, 0)
        return false
    }
    MoveCursorToImage(etc_img,100,20)
    Input.SendMouseButton(0, 0)
    if (!BotSleep(200))
        return false
    send %wingsTaken%
    if (!BotSleep(200))
        return false
    enterSC := GetKeySC("Enter") + 0
    Input.SendKey(enterSC, 1)
    if (!BotSleep(50)) {
        Input.SendKey(enterSC, 0)
        return false
    }
    Input.SendKey(enterSC, 0)
    ManageInventoryWindow()
    MoveCursorToImage(close_img)
    if (!BotSleep(200))
        return false
    InputClick()
    wingcount := wingsTaken
    return BotSleep(200)
}

ManageInventoryWindow(){
    if (BotShouldStop())
        return false
    action := "open"
    if(action = "close"){
        ImageSearch, FoundX, FoundY, 0, 0, A_ScreenWidth, A_ScreenHeight, etc_img

    }
    Input.SendKey(56, 1)
    if (!BotSleep(50)) {
        Input.SendKey(56, 0)
        return false
    }
    Input.SendKey(18, 1)
    if (!BotSleep(50)) {
        Input.SendKey(18, 0)
        Input.SendKey(56, 0)
        return false
    }
    Input.SendKey(18, 0)
    if (!BotSleep(50)) {
        Input.SendKey(56, 0)
        return false
    }
    Input.SendKey(56, 0)
    return BotSleep(500)
}

DetectCAPTCHA(xs, ys, ws, hs) {
    global captchaColor
    PixelSearch, x, y, xs, ys, xs + ws, ys + hs, %captchaColor%, 1, Fast RGB 
    if (ErrorLevel = 0) {
        Loop,8{
            if (BotShouldStop())
                break
            SoundBeep, 750, 1000
            if (!BotSleep(500))
                break
        }
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
    if (BotShouldStop())
        return false
    if (!BotSleep(500))
        return false
    ManageInventoryWindow()
    if (!BotSleep(500))
        return false
    MoveCursorToImage(use_img)
    if (!BotSleep(100))
        return false
    InputClick()
    SendKeyCombo(OpenStorageButtonKey)
    MoveCursorToImage(cell1_img,0,40)
    while(!BotShouldStop() && !CheckInventoryCell(empty_cell_img)){
        CheckInventoryCell(flywing_img, false)
        AltClicks(1)
        if (!BotSleep(50))
            return false
    }
    if (BotShouldStop() || !BotSleep(100))
        return false
    MoveCursorToImage(eqp_img)
    if (!BotSleep(100))
        return false
    InputClick()
    if (!BotSleep(50))
        return false
    MoveCursorToImage(cell1_img,0,40)
    while(!BotShouldStop() && !CheckInventoryCell(empty_cell_img)){
        AltClicks(1)
        if (!BotSleep(50))
            return false
    }

    MoveCursorToImage(etc_img)
    if (!BotSleep(100))
        return false
    InputClick()
    if (!BotSleep(100))
        return false
    MoveCursorToImage(cell1_img,0,40)
    while(!BotShouldStop() && !CheckInventoryCell(empty_cell_img)){
        if (!BotSleep(50))
            return false
        if(CheckImageOnScreen(ok_img)){
            Input.SendKey(284, 1)
            if (!BotSleep(50)) {
                Input.SendKey(284, 0)
                return false
            }
            Input.SendKey(284, 0)
            MouseGetPos, currentX, currentY
            MoveMouseTo(currentX + 40, currentY)
        }
        AltClicks(1)
    }
    if (BotShouldStop() || !BotSleep(100))
        return false
    MoveCursorToImage(close_img,10,10)
    if (!BotSleep(100))
        return false
    InputClick()
    ManageInventoryWindow()
    return BotSleep(500)
}
