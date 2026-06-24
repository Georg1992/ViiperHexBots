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
global HUNT_STATE_INTERVAL_MS := 150
global HUNT_DISCOVERY_INTERVAL_MS := 1000
global HUNT_POST_ATTACK_STATE_DELAY_MS := 120
global huntServerBusy := false
global huntScanTimersActive := false
global huntSkillSC := 0
global huntTeleportSC := 0
global huntPendingDirectState := ""

HuntClearPendingDirectState(reason := "") {
    global huntPendingDirectState
    if (!IsObject(huntPendingDirectState))
        return
    if (reason != "" && IsFunc("AppendLog"))
        AppendLog("[DIRECT] clear reason=" . reason)
    huntPendingDirectState := ""
}

HuntDropPendingDirectState(reason) {
    global huntPendingDirectState
    if (!IsObject(huntPendingDirectState))
        return
    if IsFunc("AppendLog")
        AppendLog("[DIRECT] drop reason=" . reason . " id=" . huntPendingDirectState.trackId . " epoch=" . huntPendingDirectState.areaEpoch)
    huntPendingDirectState := ""
}

HuntValidatePendingDirectState(ByRef dropReason) {
    global huntPendingDirectState, HUNT_AREA_EPOCH
    dropReason := ""
    if (!IsObject(huntPendingDirectState))
        return false
    if (huntPendingDirectState.areaEpoch != HUNT_AREA_EPOCH) {
        dropReason := "epoch_changed"
        return false
    }
    if (!IsObject(HuntTracks_GetTrackById(huntPendingDirectState.trackId))) {
        dropReason := "track_missing"
        return false
    }
    GetHuntSearchRegion(xs, ys, ws, hs)
    if (!ws || !hs) {
        dropReason := "invalid_roi"
        return false
    }
    return true
}

HuntPendingDirectStateReadyToRun() {
    global huntPendingDirectState
    dropReason := ""
    if (!IsObject(huntPendingDirectState))
        return false
    if (!HuntValidatePendingDirectState(dropReason)) {
        HuntDropPendingDirectState(dropReason)
        return false
    }
    return (A_TickCount >= huntPendingDirectState.readyTick)
}

HuntShouldBlockDiscoveryForDirect() {
    global huntPendingDirectState
    if (!HuntPendingDirectStateReadyToRun())
        return false
    if IsFunc("AppendLog")
        AppendLog("[DISCOVERY] skip reason=direct_ready id=" . huntPendingDirectState.trackId . " epoch=" . huntPendingDirectState.areaEpoch)
    return true
}

HuntScheduleDirectStateCheck(trackId, screenX, screenY) {
    global huntPendingDirectState, HUNT_POST_ATTACK_STATE_DELAY_MS, HUNT_AREA_EPOCH

    huntPendingDirectState := {}
    huntPendingDirectState.trackId := trackId
    huntPendingDirectState.x := screenX
    huntPendingDirectState.y := screenY
    huntPendingDirectState.areaEpoch := HUNT_AREA_EPOCH
    huntPendingDirectState.readyTick := A_TickCount + HUNT_POST_ATTACK_STATE_DELAY_MS

    if IsFunc("AppendLog")
        AppendLog("[DIRECT] queue id=" . trackId . " epoch=" . HUNT_AREA_EPOCH . " ready=" . huntPendingDirectState.readyTick)
}

HuntTryRunDirectStateCheck() {
    global huntPendingDirectState, huntServerBusy, botRunning, botPaused, botStopRequested

    if (!IsObject(huntPendingDirectState))
        return false

    dropReason := ""
    if (!HuntValidatePendingDirectState(dropReason)) {
        HuntDropPendingDirectState(dropReason)
        return false
    }

    if (A_TickCount < huntPendingDirectState.readyTick)
        return false
    if (huntServerBusy)
        return false
    if (!botRunning || botPaused || botStopRequested)
        return false

    trackId := huntPendingDirectState.trackId
    screenX := huntPendingDirectState.x
    screenY := huntPendingDirectState.y
    epoch := huntPendingDirectState.areaEpoch

    GetHuntSearchRegion(xs, ys, ws, hs)
    if (!ws || !hs) {
        HuntDropPendingDirectState("invalid_roi")
        return false
    }

    mobName := MobTemplateFolderName()
    if IsFunc("AppendLog")
        AppendLog("[DIRECT] run id=" . trackId . " epoch=" . epoch)

    huntServerBusy := true
    MobStateRecognizeDirectAndApply(mobName, xs, ys, ws, hs, trackId, screenX, screenY)
    huntServerBusy := false
    huntPendingDirectState := ""
    return true
}

HuntAttackTrack(skillSC, track) {
    global SkillDelay, botRunning, botStopRequested

    if (!IsObject(track))
        return false

    remainingDelay := SkillDelay - (A_TickCount - track.lastAttackTick)
    if (remainingDelay > 0) {
        BotSleep(remainingDelay)
        if (!botRunning || botStopRequested)
            return false
    }

    if IsFunc("AppendLog")
        AppendLog("Hunt [engage]: attack track id=" . track.id . " @" . Round(track.x) . "," . Round(track.y) . " conf=" . Round(track.confidence, 2))
    MoveMouseTo(Round(track.x), Round(track.y))
    if (!HuntSkillClick(skillSC))
        return false
    HuntTracks_ApplyAttackEvent(track.id)
    HuntScheduleDirectStateCheck(track.id, Round(track.x), Round(track.y))
    BotSessionRecordAttack(Round(track.x), Round(track.y), track.confidence)
    return true
}

HuntFilterCandidates(candidates, xs, ys, ws, hs, ByRef filteredCandidates) {
    filteredCandidates := []
    GetMobSearchPlayerIgnore(xs, ys, ws, hs, ignoreX, ignoreY, ignoreW, ignoreH)
    for index, candidate in candidates {
        if (!candidate.living)
            continue
        if (MobPointInsideIgnore(candidate.x, candidate.y, ignoreX, ignoreY, ignoreW, ignoreH))
            continue
        filteredCandidates.Push(candidate)
    }
}

HuntTryAreaClear(discoveryLivingCount) {
    global botRunning, botPaused, botStopRequested, huntTeleportSC

    if (!botRunning || botPaused || botStopRequested)
        return
    if (!HuntPolicy_ShouldTeleport(discoveryLivingCount))
        return

    if IsFunc("AppendLog")
        AppendLog("Hunt: area clear — teleporting (alive=" . HuntTracks_GetAliveCount() . " scanLiving=" . discoveryLivingCount . ")")
    Teleport(huntTeleportSC)
    HuntAreaReset()
}

HuntStartScanTimers() {
    global huntScanTimersActive, HUNT_STATE_INTERVAL_MS, HUNT_DISCOVERY_INTERVAL_MS, huntServerBusy
    if (huntScanTimersActive)
        return
    SetTimer, HuntStateTick, %HUNT_STATE_INTERVAL_MS%
    SetTimer, HuntDiscoveryTick, %HUNT_DISCOVERY_INTERVAL_MS%
    huntScanTimersActive := true
    if (!huntServerBusy)
        HuntDiscoveryTick()
}

HuntStopScanTimers() {
    global huntScanTimersActive, huntServerBusy
    SetTimer, HuntStateTick, Off
    SetTimer, HuntDiscoveryTick, Off
    huntScanTimersActive := false
    huntServerBusy := false
    HuntClearPendingDirectState("stop")
}

HuntStateTick() {
    global botRunning, botPaused, botStopRequested, huntServerBusy

    if (!botRunning || botPaused || botStopRequested)
        return

    if (HuntTryRunDirectStateCheck())
        return

    if (huntServerBusy)
        return

    GetHuntSearchRegion(xs, ys, ws, hs)
    if (!ws || !hs)
        return

    stateRequests := []
    HuntTracks_CollectStateRequests(stateRequests)
    if (!stateRequests.MaxIndex())
        return

    mobName := MobTemplateFolderName()
    huntServerBusy := true
    MobStateRecognizeAndApply(mobName, xs, ys, ws, hs, stateRequests)
    huntServerBusy := false
    HuntTryRunDirectStateCheck()
}

HuntDiscoveryTick() {
    global botRunning, botPaused, botStopRequested, huntServerBusy

    if (!botRunning || botPaused || botStopRequested)
        return

    if (HuntTryRunDirectStateCheck())
        return

    if (HuntShouldBlockDiscoveryForDirect())
        return

    if (huntServerBusy)
        return

    GetHuntSearchRegion(xs, ys, ws, hs)
    if (!ws || !hs)
        return

    HuntTracks_SetRoiCenter(xs + (ws // 2), ys + (hs // 2))

    mobName := MobTemplateFolderName()
    if IsFunc("BotSessionHuntScan") {
        fn := "BotSessionHuntScan"
        %fn%(mobName, xs, ys, ws, hs)
    }

    huntServerBusy := true
    jsonText := MobRecognitionDiscoveryDetect(mobName, xs, ys, ws, hs, false)

    if (!botRunning || botStopRequested) {
        huntServerBusy := false
        return
    }
    if (jsonText = "" || !MobJsonIsOk(jsonText)) {
        huntServerBusy := false
        if (jsonText != "" && IsFunc("AppendLog"))
            AppendLog("Hunt: discovery failed — retrying")
        return
    }

    candidates := []
    MobRecognitionParseCandidates(jsonText, candidates)
    filteredCandidates := []
    HuntFilterCandidates(candidates, xs, ys, ws, hs, filteredCandidates)
    discoveryLivingCount := HuntTracks_CountLivingDetections(filteredCandidates)
    HuntTracks_ApplyDetections(filteredCandidates)
    HuntTryAreaClear(discoveryLivingCount)
    huntServerBusy := false
    HuntTryRunDirectStateCheck()
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
    global huntSkillSC, huntTeleportSC
    global SkillDelay

    huntSkillSC := skillSC
    huntTeleportSC := teleportSC

    postAttackSleepMs := 50
    emptyScanSleepMs := 25

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

    HuntStartScanTimers()

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
            HuntStopScanTimers()
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
            HuntStopScanTimers()
            RequestBotStop("captcha detected")
            break
        }

        HuntTryRunDirectStateCheck()

        targetId := HuntPolicy_SelectTarget()
        if (targetId) {
            if (!botRunning || botStopRequested)
                break
            targetTrack := HuntTracks_GetTrackById(targetId)
            if IsFunc("AppendLog")
                AppendLog("Hunt [target]: track id=" . targetId . " @" . Round(targetTrack.x) . "," . Round(targetTrack.y) . " attacks=" . targetTrack.attackCount . " alive=" . HuntTracks_GetAliveCount())
            if (HuntAttackTrack(skillSC, targetTrack))
                BotSleep(postAttackSleepMs)
            continue
        }

        BotSleep(emptyScanSleepMs)
    }

    HuntStopScanTimers()
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
    if IsFunc("BotSessionRecordTeleport")
        BotSessionRecordTeleport()
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
