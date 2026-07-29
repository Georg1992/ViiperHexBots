import time
from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame
import cv2

detector = MobDetector(PROJECT_ROOT, load_detector_config())
for suite in MOB_FIXTURE_SUITES:
    descriptor = detector._get_descriptor(suite.mob_name)
    w = descriptor.avg_width
    h = descriptor.avg_height
    min_side = min(w, h)
    downscale = 2
    if min_side / downscale < 24.0:
        downscale = 1
    print(f"{suite.folder}: size={w}x{h}, min_side={min_side}, downscale={downscale}")
