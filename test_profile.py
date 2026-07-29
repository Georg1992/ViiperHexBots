import time
import cProfile
import pstats
import io
from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame
import cv2

detector = MobDetector(PROJECT_ROOT, load_detector_config())
for suite in MOB_FIXTURE_SUITES:
    if suite.folder == "Noxious":
        for image in suite.images()[:1]:
            frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
            frame = fixture_search_frame(frame)
            
            pr = cProfile.Profile()
            pr.enable()
            result = detector.detect(frame, suite.mob_name)
            pr.disable()
            
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(30)
            print(s.getvalue())
