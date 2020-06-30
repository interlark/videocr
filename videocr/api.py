from . import utils
from .video import Video
import psutil

def get_subtitles(
        video_path: str, lang='eng', time_start='0:00', time_end='',
        conf_threshold=65, sim_threshold=90, use_fullframe=False, 
        download_best=False, img_hook=None, take_every_nth_frame=1, processes=psutil.cpu_count(logical=False)) -> str:
    utils.download_lang_data(lang, download_best)

    v = Video(video_path)
    v.run_ocr(lang, time_start, time_end, conf_threshold, use_fullframe, img_hook, take_every_nth_frame, processes)
    return v.get_subtitles(sim_threshold)


def save_subtitles_to_file(
        video_path: str, file_path='subtitle.srt', lang='eng',
        time_start='0:00', time_end='', conf_threshold=65, sim_threshold=90,
        use_fullframe=False) -> None:
    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(get_subtitles(
            video_path, lang, time_start, time_end, conf_threshold,
            sim_threshold, use_fullframe))
