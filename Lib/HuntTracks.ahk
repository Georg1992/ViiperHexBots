#Requires AutoHotkey v1.1.33+

; Mob identity and state for the current hunt area (reset via HuntAreaReset).

global HUNT_TRACK_MATCH_RADIUS := 45
global HUNT_TRACK_MISS_LIMIT := 4
global HUNT_TRACK_REQUIRE_CLEAR_SCAN := true
global HUNT_TRACK_UNREACHABLE_ATTACKS := 3
global HUNT_TRACK_DEBUG := false

global huntTracks := []
global huntTrackNextId := 1
global huntTrackScanId := 0
global huntTrackClearScans := 0
global huntTrackRoiCenterX := 0
global huntTrackRoiCenterY := 0

global CurrentTargetTrackId := ""

HuntTracks_Log(prefix, message) {
    if IsFunc("AppendLog")
        AppendLog("[" . prefix . "] " . message)
    if (HUNT_TRACK_DEBUG && IsFunc("SessionLogWrite"))
        SessionLogWrite("DEBUG", "tracks", "[" . prefix . "] " . message)
}

HuntTracks_Reset() {
    global huntTracks, huntTrackNextId, huntTrackScanId, huntTrackClearScans
    huntTracks := []
    huntTrackNextId := 1
    huntTrackScanId := 0
    huntTrackClearScans := 0
}

HuntTracks_SetRoiCenter(centerX, centerY) {
    global huntTrackRoiCenterX, huntTrackRoiCenterY
    huntTrackRoiCenterX := centerX
    huntTrackRoiCenterY := centerY
}

HuntTracks_GetAliveCount() {
    global huntTracks
    count := 0
    for index, track in huntTracks {
        if (track.state = "alive" && !track.unreachable)
            count++
    }
    return count
}

HuntTracks_CountsForAreaClear(track) {
    global huntTrackScanId
    if (track.state != "alive" || track.unreachable)
        return false
    if (track.lastSeenScan = huntTrackScanId)
        return true
    return (track.attackCount < 1)
}

HuntTracks_GetAreaClearAliveCount() {
    global huntTracks
    count := 0
    for index, track in huntTracks {
        if (HuntTracks_CountsForAreaClear(track))
            count++
    }
    return count
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

HuntTracks_DistanceFromRoiCenter(x, y) {
    global huntTrackRoiCenterX, huntTrackRoiCenterY
    if (!huntTrackRoiCenterX && !huntTrackRoiCenterY)
        return 0
    dx := x - huntTrackRoiCenterX
    dy := y - huntTrackRoiCenterY
    return Sqrt(dx * dx + dy * dy)
}

HuntTracks_CreateTrack(x, y, confidence) {
    global huntTracks, huntTrackNextId, huntTrackScanId
    track := {}
    track.id := huntTrackNextId
    huntTrackNextId++
    track.x := x
    track.y := y
    track.state := "alive"
    track.lastSeenScan := huntTrackScanId
    track.missCount := 0
    track.attackCount := 0
    track.confidence := confidence
    track.lastAttackTick := 0
    track.unreachable := false
    track.createdScan := huntTrackScanId
    track.updatedTick := A_TickCount
    huntTracks.Push(track)
    HuntTracks_Log("TRACK", "new id=" . track.id . " x=" . Round(x) . " y=" . Round(y) . " conf=" . Round(confidence, 2))
    return track
}

HuntTracks_ApplyMatch(track, candidate) {
    global huntTrackScanId
    track.x := candidate.x
    track.y := candidate.y
    track.lastSeenScan := huntTrackScanId
    track.missCount := 0
    track.confidence := candidate.confidence
    track.updatedTick := A_TickCount
    if (candidate.dead) {
        if (track.state != "dead") {
            track.state := "dead"
            HuntTracks_Log("TRACK", "dead id=" . track.id)
        }
    } else if (candidate.living && track.state = "alive") {
        track.state := "alive"
    }
    if (HUNT_TRACK_DEBUG)
        HuntTracks_Log("TRACK", "match id=" . track.id . " x=" . Round(track.x) . " y=" . Round(track.y) . " state=" . track.state . " miss=0")
}

HuntTracks_CollectAttackProbes(ByRef outXs, ByRef outYs) {
    global huntTracks
    outXs := []
    outYs := []
    for index, track in huntTracks {
        if (track.state != "alive" || track.attackCount < 1)
            continue
        outXs.Push(Round(track.x))
        outYs.Push(Round(track.y))
    }
}

HuntTracks_CandidateNearDeadTrack(x, y, radius) {
    global huntTracks
    matchRadiusSq := radius * radius
    for index, track in huntTracks {
        if (track.state != "dead")
            continue
        dx := x - track.x
        dy := y - track.y
        if ((dx * dx) + (dy * dy) <= matchRadiusSq)
            return true
    }
    return false
}

HuntTracks_FindNearestAttackedTrack(x, y, matchRadiusSq, matchedTrackIds) {
    global huntTracks
    bestTrackIndex := 0
    bestDistSq := matchRadiusSq + 1
    for trackIndex, track in huntTracks {
        if (track.state = "gone" || track.attackCount < 1)
            continue
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

HuntTracks_AllTracksDead() {
    global huntTracks
    if (!huntTracks.MaxIndex())
        return false
    return (HuntTracks_GetAliveCount() = 0)
}

HuntTracks_ApplyDeadCandidate(candidate, ByRef matchedTrackIds) {
    global huntTracks, HUNT_TRACK_MATCH_RADIUS
    matchRadiusSq := HUNT_TRACK_MATCH_RADIUS * HUNT_TRACK_MATCH_RADIUS
    bestTrackIndex := HuntTracks_FindNearestTrack(candidate.x, candidate.y, matchRadiusSq, matchedTrackIds, true)
    if (!bestTrackIndex)
        bestTrackIndex := HuntTracks_FindNearestTrack(candidate.x, candidate.y, matchRadiusSq, matchedTrackIds, false)
    if (!bestTrackIndex)
        bestTrackIndex := HuntTracks_FindNearestAttackedTrack(candidate.x, candidate.y, matchRadiusSq, matchedTrackIds)
    if (bestTrackIndex > 0) {
        track := huntTracks[bestTrackIndex]
        matchedTrackIds[track.id] := true
        HuntTracks_ApplyMatch(track, candidate)
    }
}

HuntTracks_FindNearestTrack(x, y, matchRadiusSq, matchedTrackIds, aliveOnly := false) {
    global huntTracks
    bestTrackIndex := 0
    bestDistSq := matchRadiusSq + 1
    for trackIndex, track in huntTracks {
        if (track.state = "gone")
            continue
        if (aliveOnly && track.state != "alive")
            continue
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

HuntTracks_Update(candidates) {
    global huntTracks, huntTrackScanId, huntTrackClearScans, HUNT_TRACK_MATCH_RADIUS, HUNT_TRACK_MISS_LIMIT
    huntTrackScanId++

    matchedTrackIds := {}
    matchRadiusSq := HUNT_TRACK_MATCH_RADIUS * HUNT_TRACK_MATCH_RADIUS
    deadCandidates := []
    livingCandidates := []

    if (IsObject(candidates)) {
        for index, candidate in candidates {
            if (candidate.dead)
                deadCandidates.Push(candidate)
            else if (candidate.living)
                livingCandidates.Push(candidate)
        }
    }

    for candIndex, candidate in deadCandidates
        HuntTracks_ApplyDeadCandidate(candidate, matchedTrackIds)

    if (livingCandidates.MaxIndex() > 1)
        HuntTracks_SortCandidatesByConfidence(livingCandidates)

    for candIndex, candidate in livingCandidates {
        if (HuntTracks_CandidateNearDeadTrack(candidate.x, candidate.y, HUNT_TRACK_MATCH_RADIUS))
            continue

        bestTrackIndex := HuntTracks_FindNearestTrack(candidate.x, candidate.y, matchRadiusSq, matchedTrackIds, true)
        if (bestTrackIndex > 0) {
            track := huntTracks[bestTrackIndex]
            matchedTrackIds[track.id] := true
            HuntTracks_ApplyMatch(track, candidate)
        } else if (!HuntTracks_CandidateNearDeadTrack(candidate.x, candidate.y, HUNT_TRACK_MATCH_RADIUS)) {
            HuntTracks_CreateTrack(candidate.x, candidate.y, candidate.confidence)
        }
    }

    for index, track in huntTracks {
        if (matchedTrackIds.HasKey(track.id))
            continue
        if (track.state != "alive")
            continue
        track.missCount++
        if (track.missCount > HUNT_TRACK_MISS_LIMIT) {
            track.state := "gone"
            HuntTracks_Log("TRACK", "gone id=" . track.id . " miss=" . track.missCount)
        }
    }

    if (HuntTracks_GetAreaClearAliveCount() > 0)
        huntTrackClearScans := 0
}

HuntTracks_SortCandidatesByConfidence(ByRef candidates) {
    count := candidates.MaxIndex()
    if (!count || count < 2)
        return
    Loop % count - 1 {
        outer := A_Index
        Loop % count - outer {
            inner := outer + A_Index
            if (candidates[inner].confidence > candidates[outer].confidence) {
                tmp := candidates[outer]
                candidates[outer] := candidates[inner]
                candidates[inner] := tmp
            }
        }
    }
}

HuntTracks_IsFresh(track) {
    global huntTrackScanId
    return (IsObject(track) && track.lastSeenScan = huntTrackScanId)
}

HuntTracks_GetActionableAliveCount() {
    global huntTracks, huntTrackScanId
    count := 0
    for index, track in huntTracks {
        if (track.state != "alive" || track.unreachable)
            continue
        if (track.lastSeenScan != huntTrackScanId)
            continue
        count++
    }
    return count
}

HuntTracks_SelectTarget() {
    global huntTracks, huntTrackScanId, CurrentTargetTrackId
    bestId := 0
    bestScore := -1.0e9
    for index, track in huntTracks {
        if (track.state != "alive")
            continue
        if (track.unreachable)
            continue
        if (track.lastSeenScan != huntTrackScanId)
            continue
        if (CurrentTargetTrackId != "" && track.id = CurrentTargetTrackId)
            continue
        dist := HuntTracks_DistanceFromRoiCenter(track.x, track.y)
        score := (track.confidence * 100.0) - (track.attackCount * 20.0) - (dist * 0.1)
        if (score > bestScore) {
            bestScore := score
            bestId := track.id
        }
    }
    return bestId
}

HuntTracks_MarkAttack(id) {
    track := HuntTracks_GetTrackById(id)
    if (!IsObject(track))
        return false
    track.attackCount++
    track.lastAttackTick := A_TickCount
    HuntTracks_Log("HUNT", "target id=" . id . " x=" . Round(track.x) . " y=" . Round(track.y) . " attacks=" . track.attackCount)
    return true
}

HuntTracks_MarkDead(id) {
    track := HuntTracks_GetTrackById(id)
    if (!IsObject(track))
        return false
    if (track.state != "dead")
        HuntTracks_Log("TRACK", "dead id=" . id)
    track.state := "dead"
    return true
}

HuntTracks_MarkGone(id) {
    track := HuntTracks_GetTrackById(id)
    if (!IsObject(track))
        return false
    track.state := "gone"
    HuntTracks_Log("TRACK", "gone id=" . id)
    return true
}

HuntTracks_MarkUnreachable(id) {
    track := HuntTracks_GetTrackById(id)
    if (!IsObject(track))
        return false
    track.unreachable := true
    HuntTracks_Log("TRACK", "unreachable id=" . id)
    return true
}

HuntTracks_AllCleared(requiredClearScans := 1) {
    global huntTrackClearScans, HUNT_TRACK_REQUIRE_CLEAR_SCAN
    if (HuntTracks_GetAreaClearAliveCount() > 0) {
        huntTrackClearScans := 0
        return false
    }
    if (!HUNT_TRACK_REQUIRE_CLEAR_SCAN)
        return true
    huntTrackClearScans++
    if (huntTrackClearScans >= requiredClearScans) {
        HuntTracks_Log("HUNT", "cleared alive=0 currentTarget=none")
        return true
    }
    return false
}

HuntTracks_DebugDump() {
    global huntTracks, huntTrackScanId, CurrentTargetTrackId, huntTrackClearScans
    HuntTracks_Log("TRACK", "dump scan=" . huntTrackScanId . " tracks=" . huntTracks.MaxIndex() . " alive=" . HuntTracks_GetAliveCount() . " clearScans=" . huntTrackClearScans . " current=" . CurrentTargetTrackId)
    for index, track in huntTracks {
        HuntTracks_Log("TRACK", "  id=" . track.id . " " . track.state . " x=" . Round(track.x) . " y=" . Round(track.y) . " miss=" . track.missCount . " atk=" . track.attackCount . " conf=" . Round(track.confidence, 2) . (track.unreachable ? " UNR" : ""))
    }
}

HuntAreaReset() {
    global CurrentTargetTrackId
    HuntTracks_Reset()
    CurrentTargetTrackId := ""
}

HuntSessionReset(resetWarpTimer := true) {
    global huntLastWarpTime, huntLastSkillTime, huntFastIdle
    if (resetWarpTimer)
        huntLastWarpTime := 0
    huntLastSkillTime := 0
    huntFastIdle := false
    HuntAreaReset()
}
