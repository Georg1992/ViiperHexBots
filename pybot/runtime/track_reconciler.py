"""Discovery reconciliation — matches detections to existing tracks; lists
new-mob candidates for tracking to create.

Discovery finds NEW mobs and publishes candidate positions. Tracking owns
all track creation — it ingests candidates on the next fresh frame, runs a
local-follow search to get exact coordinates, and creates the track there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pybot.recognition.rules import (
    DiscoveryDetection,
    cluster_living_detections,
    detection_matches_existing,
)
from pybot.recognition.detector.detector import load_detector_config


@dataclass
class DiscoveryReconcileResult:
    """Output of discovery reconciliation."""

    new_candidates: list[DiscoveryDetection]
    matched: list[tuple[int, DiscoveryDetection]]
    removed_ids: list[int]
    matched_count: int


class TrackReconciler:
    """Stateless service that matches detections and lists absences."""

    @staticmethod
    def match_and_absent(
        detections: list[DiscoveryDetection],
        existing_positions: list[tuple[int, int]],
        existing_track_positions: list[tuple],
        *,
        detector_config: dict | None = None,
    ) -> DiscoveryReconcileResult:
        """Match detections to existing tracks; return new candidates.

        Moving tracks get a bounded grace radius. Assignment is a global,
        one-to-one optimization: maximize the number of matches first, then
        minimize the total distance. This prevents two nearby kiting mobs from
        greedily stealing one another's identity.

        ``vel_x``/``vel_y`` are the latest observed displacement hint, not a
        velocity in pixels/second. They are therefore applied for at most one
        predicted step and never multiplied by an arbitrary scan interval.
        """
        config = detector_config or load_detector_config()
        cluster_radius = int(config["discoveryClusterRadiusPx"])
        dedup_radius = int(config["trackDedupRadiusPx"])
        moving_radius = int(config["trackDedupMovingRadiusPx"])
        radius_sq = dedup_radius * dedup_radius

        clustered = cluster_living_detections(
            detections,
            cluster_radius=cluster_radius,
        )
        unmatched_ids = {entry[0] for entry in existing_track_positions}
        assignments = TrackReconciler._assign_track_ids(
            clustered,
            existing_track_positions,
            unmatched_ids,
            radius_sq=radius_sq,
            moving_radius=moving_radius,
        )

        candidate_positions: list[tuple[int, int]] = []
        matched: list[tuple[int, DiscoveryDetection]] = []
        matched_count = 0
        new_candidates: list[DiscoveryDetection] = []
        for index, detection in enumerate(clustered):
            matched_tid = assignments.get(index)
            if matched_tid is not None:
                unmatched_ids.discard(matched_tid)
                matched.append((matched_tid, detection))
                matched_count += 1
                # Keep dedup anchored to the capture-time positions of live
                # tracks. A matched detection is not a new known object, and
                # using its (possibly moved) center here would make the wider
                # track radius swallow a nearby distinct candidate.
                continue

            if TrackReconciler._matches_existing_track(
                detection,
                existing_positions,
                existing_track_positions,
                dedup_radius=dedup_radius,
            ) or detection_matches_existing(
                detection.x,
                detection.y,
                candidate_positions,
                dedup_radius=cluster_radius,
            ):
                # The first check is against already-known live tracks. When
                # discovery has a valid bbox for both objects, proximity alone
                # is not enough: non-overlapping close sprites are distinct.
                # The second check is only against other *new* detections from
                # this discovery frame, preserving the narrower cluster
                # boundary rather than the wider existing-track radius.
                matched_count += 1
                continue

            new_candidates.append(detection)
            candidate_positions.append((detection.x, detection.y))

        return DiscoveryReconcileResult(
            new_candidates=new_candidates,
            matched=matched,
            removed_ids=sorted(unmatched_ids),
            matched_count=matched_count,
        )

    @staticmethod
    def _assign_track_ids(
        detections: list[DiscoveryDetection],
        track_positions: list[tuple],
        unmatched_ids: set[int],
        *,
        radius_sq: int,
        moving_radius: int,
    ) -> dict[int, int]:
        """Return a minimum-cost one-to-one detection-index → track-id map.

        The number of simultaneously tracked mobs is small, so a memoized
        bounded search is clearer and safer than a greedy edge walk. Every
        detection may remain unmatched; matched-count is the primary objective
        so a farther valid mob is not accidentally treated as a new candidate.
        """
        base_radius = max(0, int(radius_sq ** 0.5))
        track_by_id = {
            int(entry[0]): entry
            for entry in track_positions
            if int(entry[0]) in unmatched_ids
        }
        track_ids = tuple(sorted(track_by_id))
        if not detections or not track_ids:
            return {}

        candidates: list[list[tuple[int, int]]] = []
        for detection in detections:
            edges: list[tuple[int, int]] = []
            for track_id in track_ids:
                entry = track_by_id[track_id]
                px, py = int(entry[1]), int(entry[2])
                dx = detection.x - px
                dy = detection.y - py
                current_dist_sq = (dx * dx) + (dy * dy)
                vel_x = float(entry[3]) if len(entry) > 3 else 0.0
                vel_y = float(entry[4]) if len(entry) > 4 else 0.0
                speed = (vel_x * vel_x + vel_y * vel_y) ** 0.5
                lost_count = int(entry[5]) if len(entry) > 5 else 0
                if lost_count > 0:
                    # Once local tracking has reported a miss, Discovery gets
                    # one bounded recovery radius to find the same mob again.
                    # This is deliberately unavailable to live tracks, so
                    # ordinary discovery cannot pull identities across the
                    # scene or fight high-frequency tracking.
                    allowed_radius = int(moving_radius)
                else:
                    # Motion earns only one bounded displacement's worth of
                    # grace. This is independent of discovery cadence.
                    motion_credit = min(
                        max(0, int(moving_radius) - base_radius),
                        max(0, int(round(speed))),
                    )
                    allowed_radius = base_radius + motion_credit
                allowed_sq = allowed_radius * allowed_radius

                predicted_dx = detection.x - (px + vel_x)
                predicted_dy = detection.y - (py + vel_y)
                predicted_dist_sq = (
                    predicted_dx * predicted_dx
                    + predicted_dy * predicted_dy
                )
                if current_dist_sq > allowed_sq and predicted_dist_sq > allowed_sq:
                    continue

                # Prefer a current-center match over a prediction-only match,
                # while still allowing bounded recovery to bridge a lost
                # local-follow anchor.
                score = min(current_dist_sq, predicted_dist_sq)
                edges.append((int(score), track_id))
            candidates.append(sorted(edges))

        # Min-cost max-flow on this small bipartite graph gives maximum
        # cardinality first and minimum total distance second, without a
        # scene-size-dependent greedy fallback. Bellman-Ford is intentional:
        # residual reverse edges have negative costs, and discovery cadence is
        # slow enough that this bounded matching is not on the capture hot path.
        detection_count = len(candidates)
        track_count = len(track_ids)
        source = 0
        detection_offset = 1
        track_offset = detection_offset + detection_count
        sink = track_offset + track_count
        graph: list[list[list[int]]] = [
            [] for _ in range(sink + 1)
        ]

        def add_edge(start: int, end: int, cost: int) -> None:
            graph[start].append([end, len(graph[end]), 1, cost])
            graph[end].append([start, len(graph[start]) - 1, 0, -cost])

        for det_index in range(detection_count):
            add_edge(source, detection_offset + det_index, 0)
            for edge_cost, track_id in candidates[det_index]:
                track_index = track_ids.index(track_id)
                add_edge(
                    detection_offset + det_index,
                    track_offset + track_index,
                    edge_cost,
                )
        for track_index in range(track_count):
            add_edge(track_offset + track_index, sink, 0)

        node_count = sink + 1
        while True:
            distances: list[int | None] = [None] * node_count
            previous: list[tuple[int, int] | None] = [None] * node_count
            distances[source] = 0
            for _ in range(node_count - 1):
                changed = False
                for node in range(node_count):
                    base_cost = distances[node]
                    if base_cost is None:
                        continue
                    for edge_index, edge in enumerate(graph[node]):
                        if edge[2] <= 0:
                            continue
                        next_node, _reverse, _capacity, edge_cost = edge
                        candidate_cost = base_cost + edge_cost
                        if (
                            distances[next_node] is None
                            or candidate_cost < distances[next_node]
                        ):
                            distances[next_node] = candidate_cost
                            previous[next_node] = (node, edge_index)
                            changed = True
                if not changed:
                    break
            if distances[sink] is None:
                break

            node = sink
            while node != source:
                prior = previous[node]
                if prior is None:  # defensive; sink was reachable above
                    break
                prior_node, edge_index = prior
                edge = graph[prior_node][edge_index]
                edge[2] -= 1
                graph[node][edge[1]][2] += 1
                node = prior_node
            else:
                continue
            break

        assignments: dict[int, int] = {}
        track_index_by_node = {
            track_offset + index: track_id
            for index, track_id in enumerate(track_ids)
        }
        for det_index in range(detection_count):
            node = detection_offset + det_index
            for edge in graph[node]:
                track_id = track_index_by_node.get(edge[0])
                if track_id is not None and edge[2] == 0:
                    assignments[det_index] = track_id
                    break
        return assignments

    @staticmethod
    def _matches_existing_track(
        detection: DiscoveryDetection,
        existing_positions: list[tuple[int, int]],
        existing_track_positions: list[tuple],
        *,
        dedup_radius: int,
    ) -> bool:
        """Return whether an unmatched detection is an existing object.

        A center-radius check is the fallback for older callers that do not
        carry discovery boxes. Runtime tracks do carry the last discovery bbox;
        when both boxes are valid, require overlap as well as proximity. This
        keeps a distinct sprite beside a tracked sprite from being swallowed
        by the wider same-object radius.
        """
        radius_sq = dedup_radius * dedup_radius
        entries = list(existing_track_positions)
        if not entries:
            entries = [(index, x, y) for index, (x, y) in enumerate(existing_positions)]
        for entry in entries:
            if len(entry) < 3:
                continue
            px, py = int(entry[1]), int(entry[2])
            dx = detection.x - px
            dy = detection.y - py
            if dx * dx + dy * dy > radius_sq:
                continue
            track_bbox = entry[6] if len(entry) > 6 else (0, 0, 0, 0)
            det_bbox = detection.bbox
            if (
                len(track_bbox) == 4
                and track_bbox[2] > 0
                and track_bbox[3] > 0
                and len(det_bbox) == 4
                and det_bbox[2] > 0
                and det_bbox[3] > 0
            ):
                tx, ty, tw, th = (int(value) for value in track_bbox)
                dx0, dy0, dw, dh = (int(value) for value in det_bbox)
                overlaps = (
                    tx < dx0 + dw
                    and dx0 < tx + tw
                    and ty < dy0 + dh
                    and dy0 < ty + th
                )
                if not overlaps:
                    continue
            return True
        return False

