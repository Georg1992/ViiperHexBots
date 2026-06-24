#Requires AutoHotkey v1.1.33+

; Track identity and state machine (single source of truth for mob tracks).

global HUNT_TRACK_MATCH_RADIUS := 45
global HUNT_ATTACK_RESULT_WINDOW_MS := 1800
global HUNT_AREA_EPOCH := 0
global HUNT_TRACK_DEBUG := false

global huntTracks := []
global huntTrackNextId := 1
global huntTrackScanId := 0
global huntTrackRoiCenterX := 0
global huntTrackRoiCenterY := 0

HuntTracks_Log(prefix, message) {
    if IsFunc("AppendLog")
        AppendLog("[" . prefix . "] " . message)
    if (HUNT_TRACK_DEBUG && IsFunc("SessionLogWrite"))
        SessionLogWrite("DEBUG", "tracks", "[" . prefix . "] " . message)
}

HuntTracks_Reset() {
    global huntTracks, huntTrackNextId, huntTrackScanId
    huntTracks := []
    huntTrackNextId := 1
    huntTrackScanId := 0
}

HuntTracks_CountLivingDetections(detections) {
    count := 0
    if (!IsObject(detections))
        return 0
    for index, detection in detections {
        if (detection.living)
            count++
    }
    return count
}

HuntTracks_SetRoiCenter(centerX, centerY) {
    global huntTrackRoiCenterX, huntTrackRoiCenterY
    huntTrackRoiCenterX := centerX
    huntTrackRoiCenterY := centerY
}

HuntTracks_GetAliveCount() {
    global huntTracks
    return huntTracks.MaxIndex() ? huntTracks.MaxIndex() : 0
}

HuntTracks_GetTrackById(id) {
    global huntTracks
    if (id = "" || id = 0)
        return ""
    for index, track in huntTracks {
        if (track.id = id)
            return track
    }
    return ""
}

HuntTracks_DistanceSqFromRoiCenter(x, y) {
    global huntTrackRoiCenterX, huntTrackRoiCenterY
    dx := x - huntTrackRoiCenterX
    dy := y - huntTrackRoiCenterY
    return (dx * dx) + (dy * dy)
}

HuntTracks_IsResultPending(track) {
    if (!IsObject(track))
        return false
    return (track.pendingResultUntilTick > A_TickCount)
}

HuntTracks_ClearPendingResult(track, reason := "") {
    if (!IsObject(track))
        return
    track.pendingResultUntilTick := 0
    track.pendingResultReason := ""
}

HuntTracks_CollectStateRequests(ByRef outRequests) {
    global huntTracks
    outRequests := []
    for index, track in huntTracks {
        req := {}
        req.id := track.id
        req.x := Round(track.x)
        req.y := Round(track.y)
        outRequests.Push(req)
    }
}

HuntTracks_RemoveTrackById(id) {
    global huntTracks
    kept := []
    for index, track in huntTracks {
        if (track.id != id)
            kept.Push(track)
    }
    huntTracks := kept
}

HuntTracks_CreateTrack(x, y, confidence) {
    global huntTracks, huntTrackNextId, huntTrackScanId
    track := {}
    track.id := huntTrackNextId
    huntTrackNextId++
    track.x := x
    track.y := y
    track.lastSeenScan := huntTrackScanId
    track.attackCount := 0
    track.confidence := confidence
    track.lastAttackTick := 0
    track.lastStateCheckTick := 0
    track.pendingResultUntilTick := 0
    track.pendingResultReason := ""
    track.createdScan := huntTrackScanId
    track.updatedTick := A_TickCount
    huntTracks.Push(track)
    HuntTracks_Log("TRACK", "new id=" . track.id . " x=" . Round(x) . " y=" . Round(y) . " conf=" . Round(confidence, 2))
    return track
}

HuntTracks_FindNearestAliveTrack(x, y, matchRadiusSq, matchedTrackIds) {
    global huntTracks
    bestTrackIndex := 0
    bestDistSq := matchRadiusSq + 1
    for trackIndex, track in huntTracks {
        if (matchedTrackIds.HasKey(track.id))
            continue
        dx := x - track.x
        dy := y - track.y
        distSq := (dx * dx) + (dy * dy)
        if (distSq <= matchRadiusSq && distSq < bestDistSq) {
            bestDistSq := distSq
            bestTrackIndex := trackIndex
        }
    }
    return bestTrackIndex
}

HuntTracks_ApplyLivingDetectionMatch(track, detection) {
    global huntTrackScanId
    track.x := detection.x
    track.y := detection.y
    track.lastSeenScan := huntTrackScanId
    track.confidence := detection.confidence
    track.updatedTick := A_TickCount
    if (HUNT_TRACK_DEBUG)
        HuntTracks_Log("TRACK", "detection-match id=" . track.id . " x=" . Round(track.x) . " y=" . Round(track.y))
}

HuntTracks_ApplyDetections(detections) {
    global huntTracks, huntTrackScanId, HUNT_TRACK_MATCH_RADIUS
    huntTrackScanId++

    matchedTrackIds := {}
    matchRadiusSq := HUNT_TRACK_MATCH_RADIUS * HUNT_TRACK_MATCH_RADIUS
    livingDetections := []

    if (IsObject(detections)) {
        for index, detection in detections {
            if (detection.living)
                livingDetections.Push(detection)
        }
    }

    for detIndex, detection in livingDetections {
        bestTrackIndex := HuntTracks_FindNearestAliveTrack(detection.x, detection.y, matchRadiusSq, matchedTrackIds)
        if (bestTrackIndex > 0) {
            track := huntTracks[bestTrackIndex]
            matchedTrackIds[track.id] := true
            HuntTracks_ApplyLivingDetectionMatch(track, detection)
        } else {
            HuntTracks_CreateTrack(detection.x, detection.y, detection.confidence)
        }
    }
}

HuntTracks_ApplyStateUpdates(updates) {
    global huntTracks
    if (!IsObject(updates) || !updates.MaxIndex())
        return

    for index, update in updates {
        track := HuntTracks_GetTrackById(update.id)
        if (!IsObject(track))
            continue

        track.lastStateCheckTick := A_TickCount

        if (update.state = "dead") {
            if IsFunc("AppendLog")
                AppendLog("[TRACK] state dead id=" . track.id . " remove")
            if IsFunc("BotSessionRecordKill")
                BotSessionRecordKill(track.id)
            HuntTracks_RemoveTrackById(track.id)
        } else if (update.state = "gone") {
            if IsFunc("AppendLog")
                AppendLog("[TRACK] state gone id=" . track.id . " remove")
            HuntTracks_RemoveTrackById(track.id)
        } else if (update.state = "alive") {
            track.x := update.x
            track.y := update.y
            track.confidence := update.confidence
            track.updatedTick := A_TickCount
            HuntTracks_ClearPendingResult(track)
            if IsFunc("AppendLog")
                AppendLog("[TRACK] state alive id=" . track.id . " clearPending")
            if (HUNT_TRACK_DEBUG)
                HuntTracks_Log("TRACK", "apply-state id=" . track.id . " alive @" . Round(update.x) . "," . Round(update.y))
        }
    }
}

HuntTracks_ApplyAttackEvent(trackId) {
    global HUNT_ATTACK_RESULT_WINDOW_MS
    track := HuntTracks_GetTrackById(trackId)
    if (!IsObject(track))
        return false
    track.attackCount++
    track.lastAttackTick := A_TickCount
    track.pendingResultUntilTick := A_TickCount + HUNT_ATTACK_RESULT_WINDOW_MS
    track.pendingResultReason := "attack"
    if IsFunc("AppendLog")
        AppendLog("[HUNT] attack id=" . trackId . " pendingUntil=" . track.pendingResultUntilTick)
    return true
}

HuntAreaReset() {
    global HUNT_AREA_EPOCH
    if IsFunc("HuntClearPendingDirectState")
        HuntClearPendingDirectState("area_reset")
    HUNT_AREA_EPOCH++
    HuntTracks_Reset()
}

HuntSessionReset(resetWarpTimer := true) {
    global huntLastWarpTime, huntLastSkillTime
    if (resetWarpTimer)
        huntLastWarpTime := 0
    huntLastSkillTime := 0
    HuntAreaReset()
}
