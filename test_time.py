import time
from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame
import cv2

detector = MobDetector(PROJECT_ROOT, load_detector_config())
for suite in MOB_FIXTURE_SUITES:
    print(suite.folder)
    for image in suite.images()[:1]:
            frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
            frame = fixture_search_frame(frame)
            
            t0 = time.time()
            result = detector.detect(frame, suite.mob_name)
            t1 = time.time()
            print(f"{image.file_name}: {t1-t0:.2f}s (found {len(result.accepted)})")
