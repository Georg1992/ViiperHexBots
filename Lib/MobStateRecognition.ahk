#Requires AutoHotkey v1.1.33+

; Mob state recognition for known tracks (not discovery).

global MOB_STATE_DEBUG := false

MobState_Log(message) {
    if IsFunc("AppendLog")
        AppendLog("[STATE] " . message)
}

MobStateBuildRequestJson(mobName, roiX, roiY, roiW, roiH, requests) {
    sessionId := ""
    if IsFunc("BotSessionGetId") {
        fn := "BotSessionGetId"
        sessionId := %fn%()
    }
    scaleJson := ""
    if IsFunc("BotSessionScaleRangeJson") {
        fn := "BotSessionScaleRangeJson"
        scaleJson := %fn%(mobName)
    }

    tracksJson := "["
    count := requests.MaxIndex()
    Loop % count {
        req := requests[A_Index]
        if (A_Index > 1)
            tracksJson .= ","
        localX := req.x - roiX
        localY := req.y - roiY
        tracksJson .= "{""trackId"":" . req.id . ",""x"":" . localX . ",""y"":" . localY . "}"
    }
    tracksJson .= "]"

    return "{""cmd"":""state"",""mob"":""" . mobName . """,""roi"":[" . roiX . "," . roiY . "," . roiW . "," . roiH . "],""tracks"":" . tracksJson . ",""sessionId"":""" . sessionId . """" . scaleJson . "}"
}

MobStateBuildDirectRequestJson(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY) {
    sessionId := ""
    if IsFunc("BotSessionGetId") {
        fn := "BotSessionGetId"
        sessionId := %fn%()
    }
    scaleJson := ""
    if IsFunc("BotSessionScaleRangeJson") {
        fn := "BotSessionScaleRangeJson"
        scaleJson := %fn%(mobName)
    }

    localX := screenX - roiX
    localY := screenY - roiY
    tracksJson := "[{""trackId"":" . trackId . ",""x"":" . localX . ",""y"":" . localY . "}]"

    return "{""cmd"":""state"",""mode"":""direct"",""mob"":""" . mobName . """,""roi"":[" . roiX . "," . roiY . "," . roiW . "," . roiH . "],""tracks"":" . tracksJson . ",""sessionId"":""" . sessionId . """" . scaleJson . "}"
}

MobStateFindJsonArrayBounds(jsonText, key) {
    marker := """" . key . """:["
    markerPos := InStr(jsonText, marker)
    if (!markerPos)
        return {innerStart: 0, innerEnd: 0}

    arrayStart := markerPos + StrLen(marker) - 1
    depth := 0
    index := arrayStart
    length := StrLen(jsonText)
    while (index <= length) {
        ch := SubStr(jsonText, index, 1)
        if (ch = "[")
            depth++
        else if (ch = "]") {
            depth--
            if (depth = 0)
                return {innerStart: arrayStart + 1, innerEnd: index - 1}
        }
        index++
    }
    return {innerStart: 0, innerEnd: 0}
}

MobStateExtractUpdatesSection(jsonText) {
    bounds := MobStateFindJsonArrayBounds(jsonText, "trackUpdates")
    if (bounds.innerStart = 0 || bounds.innerEnd < bounds.innerStart)
        return ""
    return SubStr(jsonText, bounds.innerStart, bounds.innerEnd - bounds.innerStart + 1)
}

MobStateParseUpdateBlock(block, ByRef outUpdate) {
    outUpdate := ""

    trackId := 0
    if (!RegExMatch(block, "i)""trackId"":(\d+)", match))
        return false
    trackId := match1 + 0

    state := ""
    if (!RegExMatch(block, "i)""state"":""([^""]+)""", match))
        return false
    state := match1

    confidence := 0.0
    if (RegExMatch(block, "i)""confidence"":([0-9.]+)", match))
        confidence := match1 + 0.0

    x := 0
    y := 0
    if (RegExMatch(block, "i)""x"":(\d+)", match))
        x := match1 + 0
    if (RegExMatch(block, "i)""y"":(\d+)", match))
        y := match1 + 0

    update := {}
    update.id := trackId
    update.state := state
    update.confidence := confidence
    update.x := x
    update.y := y
    outUpdate := update
    return true
}

MobStateParseUpdates(jsonText, ByRef outUpdates) {
    outUpdates := []
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return 0

    section := MobStateExtractUpdatesSection(jsonText)
    if (section = "")
        return 0

    pos := 1
    while (pos := RegExMatch(section, "i)\{[^{}]+\}", block, pos)) {
        update := ""
        if (MobStateParseUpdateBlock(block, update))
            outUpdates.Push(update)
        pos += StrLen(block)
    }
    return outUpdates.MaxIndex() ? outUpdates.MaxIndex() : 0
}

MobStateRecognize(mobName, roiX, roiY, roiW, roiH, requests) {
    if (roiW <= 0 || roiH <= 0)
        return ""
    if (!IsObject(requests) || !requests.MaxIndex())
        return ""

    requestJson := MobStateBuildRequestJson(mobName, roiX, roiY, roiW, roiH, requests)
    count := requests.MaxIndex()
    MobState_Log("req tracks=" . count)

    jsonText := MobRecognitionSendServerRequest(requestJson, 3000)
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return ""

    updates := []
    MobStateParseUpdates(jsonText, updates)
    updateCount := updates.MaxIndex()
    if (updateCount) {
        Loop % updateCount {
            u := updates[A_Index]
            MobState_Log("update id=" . u.id . " state=" . u.state . " conf=" . Round(u.confidence, 2))
        }
    }
    return jsonText
}

MobStateRecognizeAndApply(mobName, roiX, roiY, roiW, roiH, requests) {
    jsonText := MobStateRecognize(mobName, roiX, roiY, roiW, roiH, requests)
    if (jsonText = "")
        return false

    updates := []
    if (!MobStateParseUpdates(jsonText, updates))
        return false

    HuntTracks_ApplyStateUpdates(updates)
    return true
}

MobStateRecognizeDirect(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY) {
    if (roiW <= 0 || roiH <= 0)
        return ""
    if (trackId = "" || trackId = 0)
        return ""

    requestJson := MobStateBuildDirectRequestJson(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY)
    MobState_Log("direct req id=" . trackId)

    jsonText := MobRecognitionSendServerRequest(requestJson, 3000)
    if (jsonText = "" || !MobJsonIsOk(jsonText))
        return ""

    return jsonText
}

MobStateRecognizeDirectAndApply(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY) {
    jsonText := MobStateRecognizeDirect(mobName, roiX, roiY, roiW, roiH, trackId, screenX, screenY)
    if (jsonText = "")
        return false

    updates := []
    if (!MobStateParseUpdates(jsonText, updates))
        return false

    update := updates[1]
    if (update.state = "unknown") {
        if IsFunc("AppendLog")
            AppendLog("[STATE] direct unknown id=" . trackId)
        track := HuntTracks_GetTrackById(trackId)
        if (IsObject(track))
            track.lastStateCheckTick := A_TickCount
        return true
    }

    if IsFunc("AppendLog")
        AppendLog("[STATE] direct result id=" . trackId . " state=" . update.state . " conf=" . Round(update.confidence, 2))

    HuntTracks_ApplyStateUpdates(updates)
    return true
}
