#Requires AutoHotkey v1.1.33+

; HuntTracks: sole owner of MobTrack instances (single source of truth).
; Discovery, state, and attack only propose events; this module applies them.

global HUNT_TRACK_MATCH_RADIUS := 90
global HUNT_DETECTION_CLUSTER_RADIUS := 55
global HUNT_MOVEMENT_SLACK_PX_PER_STATE_TICK := 30
global HUNT_DISCOVERY_MATCH_SLACK_CAP_PX := 150
global HUNT_ATTACK_RESULT_WINDOW_MS := 1800
global HUNT_ATTACK_MAX_COORD_AGE_MS := 1200
global HUNT_TRACK_MISS_LIMIT := 3
global HUNT_AREA_EPOCH := 0
global HUNT_NEW_TRACK_STATE_GRACE_MS := 1000
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
        if (detection.living && !detection.dead)
            count++
    }
    return count
}

HuntTracks_SetRoiCenter(centerX, centerY) {
    global huntTrackRoiCenterX, huntTrackRoiCenterY
    huntTrackRoiCenterX := centerX
    huntTrackRoiCenterY := centerY
}

HuntTracks_GetTrackCount() {
    global huntTracks
    return huntTracks.MaxIndex() ? huntTracks.MaxIndex() : 0
}

HuntTracks_GetAttackableCount() {
    global huntTracks
    count := 0
    for index, track in huntTracks {
        if (MobTrack_IsAttackable(track))
            count++
    }
    return count
}

HuntTracks_HasAttackableTracks() {
    return (HuntTracks_GetAttackableCount() > 0)
}

MobTrack_IsKnownTarget(track) {
    return MobTrack_IsAlive(track)
}

HuntTracks_GetKnownTargetCount() {
    global huntTracks
    count := 0
    for index, track in huntTracks {
        if (MobTrack_IsKnownTarget(track))
            count++
    }
    return count
}

HuntTracks_HasKnownTargets() {
    return (HuntTracks_GetKnownTargetCount() > 0)
}

; Legacy alias — total tracks in store (includes pending).
HuntTracks_GetAliveCount() {
    return HuntTracks_GetTrackCount()
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

HuntTracks_CollectStateRequests(ByRef outRequests) {
    global huntTracks
    outRequests := []
    pendingFirst := []
    rest := []
    for index, track in huntTracks {
        req := {}
        req.id := track.id
        req.x := Round(track.x)
        req.y := Round(track.y)
        if (track.state = "pending")
            pendingFirst.Push(req)
        else
            rest.Push(req)
    }
    for index, req in pendingFirst
        outRequests.Push(req)
    for index, req in rest
        outRequests.Push(req)
}

MobTrack_WasAttacked(track) {
    if (!IsObject(track))
        return false
    if (track.attackCount > 0)
        return true
    if (track.state = "pending")
        return true
    return false
}

MobTrack_IsAlive(track) {
    if (!IsObject(track))
        return false
    return (track.state = "alive" || track.state = "pending")
}

MobTrack_GetCoordAgeMs(track) {
    if (!IsObject(track))
        return 999999
    return A_TickCount - track.updatedTick
}

MobTrack_IsCoordFresh(track) {
    global HUNT_ATTACK_MAX_COORD_AGE_MS
    if (!IsObject(track))
        return false
    return (MobTrack_GetCoordAgeMs(track) <= HUNT_ATTACK_MAX_COORD_AGE_MS)
}

MobTrack_NotePendingTimeout(track) {
    if (!IsObject(track))
        return
    if (track.pendingTimeoutLogged)
        return
    track.pendingTimeoutLogged := true
    if IsFunc("AppendLog")
        AppendLog("[TRACK] pending timeout id=" . track.id . " allow retry")
}

MobTrack_IsPending(track) {
    global HUNT_ATTACK_RESULT_WINDOW_MS
    if (!IsObject(track))
        return false
    if (track.state != "pending" || track.pendingResultResolved)
        return false
    if (track.pendingResultUntilTick > A_TickCount)
        return true
    MobTrack_NotePendingTimeout(track)
    ; After window: stay blocked until state or discovery refreshes coordinates.
    if (track.updatedTick > track.lastAttackTick)
        return false
    return true
}

MobTrack_IsExpired(track) {
    if (!IsObject(track))
        return false
    if (track.state != "pending")
        return false
    return (track.pendingResultUntilTick <= A_TickCount)
}

MobTrack_IsAttackable(track) {
    if (!MobTrack_IsAlive(track))
        return false
    if (MobTrack_IsPending(track))
        return false
    if (!MobTrack_IsCoordFresh(track))
        return false
    return true
}

MobTrack_GetDiscoveryMatchRadiusSq(track) {
    global HUNT_TRACK_MATCH_RADIUS, HUNT_STATE_INTERVAL_MS, HUNT_MOVEMENT_SLACK_PX_PER_STATE_TICK
    global HUNT_DISCOVERY_MATCH_SLACK_CAP_PX
    ageMs := MobTrack_GetCoordAgeMs(track)
    if (HUNT_STATE_INTERVAL_MS <= 0)
        HUNT_STATE_INTERVAL_MS := 100
    movementSlack := (ageMs / HUNT_STATE_INTERVAL_MS) * HUNT_MOVEMENT_SLACK_PX_PER_STATE_TICK
    if (movementSlack > HUNT_DISCOVERY_MATCH_SLACK_CAP_PX)
        movementSlack := HUNT_DISCOVERY_MATCH_SLACK_CAP_PX
    radius := HUNT_TRACK_MATCH_RADIUS + movementSlack
    if (track.state = "pending")
        radius := radius * 1.5
    return radius * radius
}

HuntTracks_AnyTrackNeedsCoordRefresh() {
    global huntTracks, HUNT_STATE_INTERVAL_MS
    threshold := HUNT_STATE_INTERVAL_MS * 3
    for index, track in huntTracks {
        if (!MobTrack_IsAlive(track))
            continue
        if (MobTrack_GetCoordAgeMs(track) > threshold)
            return true
    }
    return false
}

MobTrack_FindNearestForDetection(x, y, matchedTrackIds) {
    global huntTracks
    bestTrackIndex := 0
    bestDistSq := 999999999
    for trackIndex, track in huntTracks {
        if (!MobTrack_IsAlive(track))
            continue
        if (matchedTrackIds.HasKey(track.id))
            continue
        dx := x - track.x
        dy := y - track.y
        distSq := (dx * dx) + (dy * dy)
        trackRadiusSq := MobTrack_GetDiscoveryMatchRadiusSq(track)
        if (distSq <= trackRadiusSq && distSq < bestDistSq) {
            bestDistSq := distSq
            bestTrackIndex := trackIndex
        }
    }
    return bestTrackIndex
}

HuntTracks_ClusterLivingDetections(detections, clusterRadius) {
    if (!IsObject(detections))
        return []

    clusterRadiusSq := clusterRadius * clusterRadius
    sorted := []
    for index, detection in detections {
        if (!detection.living || detection.dead)
            continue
        sorted.Push(detection)
    }
    if (!sorted.MaxIndex())
        return []

    count := sorted.MaxIndex()
    Loop % count - 1 {
        i := A_Index
        best := i
        Loop % count - i {
            j := i + A_Index
            if (sorted[j].confidence > sorted[best].confidence)
                best := j
        }
        if (best != i) {
            tmp := sorted[i]
            sorted[i] := sorted[best]
            sorted[best] := tmp
        }
    }

    clusters := []
    for index, detection in sorted {
        merged := false
        for ci, cluster in clusters {
            dx := detection.x - cluster.x
            dy := detection.y - cluster.y
            if ((dx * dx + dy * dy) <= clusterRadiusSq) {
                merged := true
                break
            }
        }
        if (!merged) {
            cluster := {}
            cluster.x := detection.x
            cluster.y := detection.y
            cluster.confidence := detection.confidence
            cluster.living := true
            cluster.dead := false
            clusters.Push(cluster)
        }
    }
    return clusters
}

MobTrack_Create(mobName, x, y, confidence) {
    global huntTracks, huntTrackNextId, HUNT_AREA_EPOCH
    track := {}
    track.id := huntTrackNextId
    huntTrackNextId++
    track.mobName := mobName
    track.x := x
    track.y := y
    track.confidence := confidence
    track.attackCount := 0
    track.lastDiscoveryTick := A_TickCount
    track.lastStateTick := 0
    track.lastAttackTick := 0
    track.pendingResultUntilTick := 0
    track.pendingResultResolved := false
    track.pendingTimeoutLogged := false
    track.suspiciousDeadCount := 0
    track.stateGoneCount := 0
    track.discoveryMissCount := 0
    track.state := "alive"
    track.areaEpoch := HUNT_AREA_EPOCH
    track.createdTick := A_TickCount
    track.updatedTick := A_TickCount
    huntTracks.Push(track)
    HuntTracks_Log("TRACK", "create id=" . track.id . " @" . Round(x) . "," . Round(y) . " conf=" . Round(confidence, 2))
    return track
}

MobTrack_Remove(id) {
    global huntTracks
    kept := []
    for index, track in huntTracks {
        if (track.id != id)
            kept.Push(track)
    }
    huntTracks := kept
}

MobTrack_ApplyDiscoveryDetection(detection, mobName, ByRef matchedTrackIds) {
    global huntTracks

    if (!detection.living || detection.dead)
        return 0

    bestTrackIndex := MobTrack_FindNearestForDetection(detection.x, detection.y, matchedTrackIds)
    if (bestTrackIndex > 0) {
        track := huntTracks[bestTrackIndex]
        matchedTrackIds[track.id] := true
        return 0
    }

    newTrack := MobTrack_Create(mobName, detection.x, detection.y, detection.confidence)
    matchedTrackIds[newTrack.id] := true
    return 1
}

HuntTracks_ClusterDetectionsForApply(detections) {
    global HUNT_DETECTION_CLUSTER_RADIUS
    return HuntTracks_ClusterLivingDetections(detections, HUNT_DETECTION_CLUSTER_RADIUS)
}

HuntTracks_FinalizeDiscoveryScan(matchedTrackIds) {
    global huntTracks, HUNT_TRACK_MISS_LIMIT
    if (!IsObject(matchedTrackIds))
        matchedTrackIds := {}

    removeIds := []
    for index, track in huntTracks {
        if (MobTrack_WasAttacked(track))
            continue
        if (matchedTrackIds.HasKey(track.id)) {
            track.discoveryMissCount := 0
            continue
        }
        track.discoveryMissCount++
        if (track.discoveryMissCount >= HUNT_TRACK_MISS_LIMIT)
            removeIds.Push(track.id)
    }

    for index, trackId in removeIds {
        track := HuntTracks_GetTrackById(trackId)
        missCount := IsObject(track) ? track.discoveryMissCount : HUNT_TRACK_MISS_LIMIT
        if IsFunc("AppendLog")
            AppendLog("[TRACK] remove id=" . trackId . " reason=discovery_missed_unattacked miss=" . missCount)
        if IsFunc("BotSessionRecordDiscoveryMissRemoval")
            BotSessionRecordDiscoveryMissRemoval(trackId)
        MobTrack_Remove(trackId)
    }
}

HuntTracks_ReceiveDiscoveryDetections(detections, mobName) {
    global huntTrackScanId
    huntTrackScanId++

    if (!IsObject(detections))
        return 0

    clustered := HuntTracks_ClusterDetectionsForApply(detections)
    added := 0
    matchedTrackIds := {}
    for index, detection in clustered
        added += MobTrack_ApplyDiscoveryDetection(detection, mobName, matchedTrackIds)
    HuntTracks_FinalizeDiscoveryScan(matchedTrackIds)
    return added
}

MobTrack_ApplyStateObservation(observation) {
    track := HuntTracks_GetTrackById(observation.id)
    if (!IsObject(track))
        return

    track.lastStateTick := A_TickCount
    wasAttacked := MobTrack_WasAttacked(track)

    if (observation.state = "dead") {
        if (!wasAttacked) {
            track.suspiciousDeadCount++
            if IsFunc("BotSessionRecordIgnoredUnattackedDead")
                BotSessionRecordIgnoredUnattackedDead(track.id)
            if IsFunc("AppendLog")
                AppendLog("[TRACK] state dead ignored id=" . track.id . " reason=unattacked count=" . track.suspiciousDeadCount)
            return
        }
        if IsFunc("AppendLog")
            AppendLog("[TRACK] kill confirmed id=" . track.id . " attackCount=" . track.attackCount)
        if IsFunc("BotSessionRecordConfirmedKill")
            BotSessionRecordConfirmedKill(track.id)
        if IsFunc("AppendLog")
            AppendLog("[TRACK] remove id=" . track.id . " reason=dead_confirmed attackCount=" . track.attackCount)
        MobTrack_Remove(track.id)
        return
    }

    if (observation.state = "gone") {
        if (!wasAttacked) {
            track.stateGoneCount++
            if IsFunc("BotSessionRecordIgnoredUnattackedGone")
                BotSessionRecordIgnoredUnattackedGone(track.id)
            if IsFunc("AppendLog")
                AppendLog("[TRACK] state gone ignored id=" . track.id . " reason=unattacked count=" . track.stateGoneCount)
            return
        }
        if IsFunc("BotSessionRecordGoneRemoval")
            BotSessionRecordGoneRemoval(track.id)
        if IsFunc("AppendLog")
            AppendLog("[TRACK] remove id=" . track.id . " reason=gone attackCount=" . track.attackCount)
        MobTrack_Remove(track.id)
        return
    }

    if (observation.state = "unknown") {
        if (HUNT_TRACK_DEBUG)
            HuntTracks_Log("TRACK", "state unknown id=" . track.id . " attackCount=" . track.attackCount)
        return
    }

    if (observation.state = "alive") {
        track.suspiciousDeadCount := 0
        track.stateGoneCount := 0
        track.x := observation.x
        track.y := observation.y
        track.confidence := observation.confidence
        track.state := "alive"
        track.pendingResultUntilTick := 0
        track.pendingResultResolved := true
        track.pendingTimeoutLogged := false
        track.updatedTick := A_TickCount
        if (HUNT_TRACK_DEBUG)
            HuntTracks_Log("TRACK", "state id=" . track.id . " alive @" . Round(observation.x) . "," . Round(observation.y))
    }
}

HuntTracks_ReceiveStateObservations(observations) {
    if (!IsObject(observations) || !observations.MaxIndex())
        return
    for index, observation in observations
        MobTrack_ApplyStateObservation(observation)
}

MobTrack_ApplyAttack(trackId) {
    global HUNT_ATTACK_RESULT_WINDOW_MS
    track := HuntTracks_GetTrackById(trackId)
    if (!IsObject(track))
        return false
    track.attackCount++
    track.lastAttackTick := A_TickCount
    track.pendingResultUntilTick := A_TickCount + HUNT_ATTACK_RESULT_WINDOW_MS
    track.pendingResultResolved := false
    track.pendingTimeoutLogged := false
    track.state := "pending"
    track.updatedTick := A_TickCount
    if IsFunc("AppendLog")
        AppendLog("[HUNT] attack id=" . trackId . " pendingUntil=" . track.pendingResultUntilTick)
    return true
}

; --- Public entry points (event receivers) ---

HuntTracks_ApplyDiscoveryDetections(detections) {
    mobName := ""
    if IsFunc("MobTemplateFolderName")
        mobName := MobTemplateFolderName()
    return HuntTracks_ReceiveDiscoveryDetections(detections, mobName)
}

HuntTracks_ApplyDetections(detections) {
    return HuntTracks_ApplyDiscoveryDetections(detections)
}

HuntTracks_ApplyStateUpdates(updates) {
    HuntTracks_ReceiveStateObservations(updates)
}

HuntTracks_ApplyAttackEvent(trackId) {
    return MobTrack_ApplyAttack(trackId)
}

; --- Legacy aliases (delegate to MobTrack model) ---

HuntTracks_IsResultPending(track) {
    return MobTrack_IsPending(track)
}

HuntTracks_ClearPendingResult(track, reason := "") {
    if (!IsObject(track))
        return
    track.pendingResultUntilTick := 0
    if (track.state = "pending")
        track.state := "alive"
}

HuntAreaReset() {
    global HUNT_AREA_EPOCH
    if IsFunc("HuntClearPendingDirectState")
        HuntClearPendingDirectState("area_reset")
    HUNT_AREA_EPOCH++
    HuntTracks_Reset()
    if IsFunc("HuntMode_OnAreaReset")
        HuntMode_OnAreaReset()
}

HuntSessionReset(resetWarpTimer := true) {
    global huntLastWarpTime, huntLastSkillTime
    if (resetWarpTimer)
        huntLastWarpTime := 0
    huntLastSkillTime := 0
    HuntAreaReset()
}
