#!/usr/bin/env python3
"""Compose one labelled local-inference comparison video from a showcase catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1920, 1080, 30
BACKGROUND = (12, 18, 28)
WHITE = (236, 241, 248)
MUTED = (166, 180, 198)
ACCENT = (106, 204, 242)
ORDER = ['JAX_004', 'JAX_068', 'JAX_214', 'JAX_260', 'JAX_168', 'JAX_175']


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe(path: Path) -> dict:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,nb_frames,avg_frame_rate:format=duration', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    if not data.get('streams') or float(data.get('format', {}).get('duration', 0)) <= 0:
        raise RuntimeError(f'Invalid video: {path}')
    return data


def asset(root: Path, value: str) -> Path:
    if not value:
        raise ValueError('Required video asset is absent')
    relative = value.removeprefix('/media/') if value.startswith('/media/') else value
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Invalid or missing catalog asset: {value}')
    return path


def encoder() -> list[str]:
    return ['-an', '-r', str(FPS), '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level:v', '4.2', '-threads', '4',
            '-movflags', '+faststart']


def font(path: Path, size: int):
    return ImageFont.truetype(str(path), size=size)


def card(path: Path, font_path: Path, title: str, lines: list[str], footer: str) -> None:
    image = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 138, 220, 146), fill=ACCENT)
    draw.text((100, 190), title, font=font(font_path, 56), fill=WHITE)
    for i, line in enumerate(lines):
        draw.text((100, 340 + i * 94), line, font=font(font_path, 34), fill=WHITE if i == 0 else MUTED)
    draw.text((100, 960), footer, font=font(font_path, 25), fill=MUTED)
    image.save(path)


def overlay(path: Path, font_path: Path, title: str, labels: list[str], notes: str,
            chapter: int) -> None:
    image = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 111), fill=BACKGROUND + (255,))
    draw.rectangle((0, 1008, WIDTH, HEIGHT), fill=BACKGROUND + (255,))
    draw.text((48, 16), title, font=font(font_path, 34), fill=WHITE)
    draw.text((48, 73), labels[0], font=font(font_path, 24), fill=WHITE)
    draw.text((976, 73), labels[1], font=font(font_path, 24), fill=ACCENT)
    for x in (48, 976):
        draw.rectangle((x - 1, 111, x + 896, 1008), outline=MUTED, width=1)
    draw.text((48, 1020), notes, font=font(font_path, 20), fill=MUTED)
    for i in range(len(ORDER)):
        step = (WIDTH - 96) / len(ORDER)
        x = 48 + round(i * step)
        draw.rectangle((x, 1060, x + round(step) - 20, 1065), fill=ACCENT if i == chapter else (48, 61, 80))
    image.save(path)


def make_card_clip(png: Path, output: Path, seconds: float) -> None:
    run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', str(FPS), '-i', str(png),
         '-t', str(seconds), *encoder(), str(output)])


def make_scene_clip(inputs: list[Path], png: Path, output: Path, seconds: float) -> list[dict]:
    input_meta = [probe(path) for path in inputs]
    command = ['ffmpeg', '-v', 'error', '-y', '-filter_complex_threads', '2']
    for path in inputs:
        command += ['-i', str(path)]
    command += ['-loop', '1', '-framerate', str(FPS), '-i', str(png)]
    filters = [f'color=c=0x0c121c:s={WIDTH}x{HEIGHT}:r={FPS}:d={seconds}[bg]']
    for i, meta in enumerate(input_meta):
        duration = float(meta['format']['duration'])
        filters.append(
            f'[{i}:v]setpts=(PTS-STARTPTS)*{seconds / duration:.12f},'
            f'scale=896:896:flags=lanczos,setsar=1,fps={FPS},'
            f'tpad=stop_mode=clone:stop_duration=1,trim=duration={seconds}[v{i}]')
    filters += ['[bg][v0]overlay=48:112:shortest=1[left]',
                '[left][v1]overlay=976:112:shortest=1[base]']
    filters.append(f'[base][{len(inputs)}:v]overlay=0:0:shortest=1,format=yuv420p[out]')
    command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-t', str(seconds),
                *encoder(), str(output)]
    run(command)
    return input_meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    args = parser.parse_args()
    root = args.catalog.resolve().parent
    catalog = json.loads(args.catalog.read_text())
    scenes = {scene['id']: scene for scene in catalog['scenes'] if scene['id'] in ORDER}
    if set(scenes) != set(ORDER) or any(
            {stage['id'] for stage in scene['stages']} != {'stage1', 'stage2'}
            for scene in scenes.values()):
        raise RuntimeError('Expected both complete stages for all six selected scenes')
    if not args.font.is_file():
        raise FileNotFoundError(args.font)
    work = args.output.resolve().parent / 'composition_work'
    work.mkdir(parents=True, exist_ok=True)
    segments = []
    chapters = []
    elapsed = 0.0

    def append_segment(path: Path, seconds: float, title: str, details: dict) -> None:
        nonlocal elapsed
        measured = probe(path)
        if abs(float(measured['format']['duration']) - seconds) > 1 / FPS + 0.01:
            raise RuntimeError(f'Unexpected segment duration: {path}')
        segments.append({'file': str(path), 'start': elapsed, 'duration': seconds, 'title': title, **details})
        elapsed += seconds

    intro_png, intro_mp4 = work / 'intro.png', work / '000_intro.mp4'
    card(intro_png, args.font, '三维场景本地推理与原版效果对照',
         ['GaussianZoom Stage2  /  Skyfall-GS',
          '6 个场景 · 12 个完整阶段模型 · RTX 4090 本地 CUDA 推理',
          '先看本次 Stage1 → Stage2，再看可用的原版官方视频对照',
          '原版参考覆盖 4 个场景；缺失参考会明确标注'],
         '固定公开相机轨迹；原版与本次的外观、光照和视频压缩可能不同。')
    make_card_clip(intro_png, intro_mp4, 4)
    append_segment(intro_mp4, 4, '说明', {'kind': 'title'})
    chapters.append({'title': '说明', 'start': 0, 'end': elapsed})

    for index, scene_id in enumerate(ORDER):
        scene = scenes[scene_id]
        stages = {stage['id']: stage for stage in scene['stages']}
        chapter_start = elapsed
        title = f'{index + 1:02d} / {len(ORDER):02d}   {scene_id}'
        reference = scene.get('reference') or {}
        has_reference = bool(reference.get('video')) and 'stage2' in stages
        trajectory = stages[scene['default_stage']]['trajectory']
        seconds = 5 if has_reference else 10
        png = work / f'{scene_id}_internal.png'
        clip = work / f'{len(segments):03d}_{scene_id}_internal.mp4'
        note = ('本次模型内部对照，不是原版对照。' if has_reference else
                '本地参考包没有该场景的原版同轨迹视频；这里只比较本次两个阶段。')
        overlay(png, args.font, title + '  |  本次阶段对照',
                ['本次 Stage1 · 30,000 步', '本次 GaussianZoom Stage2 · 80,000 步'],
                f'{trajectory}  ·  {note}', index)
        inputs = [asset(root, stages['stage1']['video']), asset(root, stages['stage2']['video'])]
        metadata = make_scene_clip(inputs, png, clip, seconds)
        append_segment(clip, seconds, scene_id + ' 本次阶段对照',
                       {'scene': scene_id, 'kind': 'internal_stage_comparison', 'inputs': [str(p) for p in inputs], 'source_video_metadata': metadata})
        if has_reference:
            png = work / f'{scene_id}_official.png'
            clip = work / f'{len(segments):03d}_{scene_id}_official.mp4'
            overlay(png, args.font, title + '  |  与原版对照',
                    ['原版 Skyfall-GS · 官方发布视频', '本次 GaussianZoom Stage2 · 本地推理'],
                    f'{trajectory}  ·  按同名轨迹进度同步；外观/光照/压缩可能不同，不作逐像素真值对齐声明。', index)
            inputs = [asset(root, reference['video']), asset(root, stages['stage2']['video'])]
            metadata = make_scene_clip(inputs, png, clip, 10)
            append_segment(clip, 10, scene_id + ' 原版对照',
                           {'scene': scene_id, 'kind': 'official_skyfall_comparison', 'inputs': [str(p) for p in inputs], 'source_video_metadata': metadata})
        chapters.append({'title': scene_id, 'start': chapter_start, 'end': elapsed})
        print('COMPOSED_SCENE ' + scene_id, flush=True)


    concat = work / 'segments.ffconcat'
    concat.write_text('ffconcat version 1.0\n' + ''.join("file '" + entry['file'].replace("'", "'\\''") + "'\n" for entry in segments))
    metadata = work / 'chapters.ffmeta'
    lines = [';FFMETADATA1', 'title=本地三维场景推理与 Skyfall-GS 对照', 'comment=Local CUDA inference; official released reference videos; camera/appearance scope documented.']
    for chapter in chapters:
        lines += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={round(chapter["start"] * 1000)}',
                  f'END={round(chapter["end"] * 1000)}', 'title=' + chapter['title']]
    metadata.write_text('\n'.join(lines) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat),
         '-f', 'ffmetadata', '-i', str(metadata), '-map', '0:v:0', '-map_metadata', '1', '-map_chapters', '1',
         '-c', 'copy', '-movflags', '+faststart', str(args.output)])
    final = probe(args.output)
    if abs(float(final['format']['duration']) - elapsed) > 0.1:
        raise RuntimeError('Final video duration differs from composed segment timeline')
    result = {'catalog': str(args.catalog.resolve()), 'catalog_sha256': hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              'video': str(args.output.resolve()), 'width': WIDTH, 'height': HEIGHT, 'fps': FPS,
              'duration_seconds': elapsed, 'segments': segments, 'chapters': chapters,
              'scope': 'One video; no webpage. Fresh local model renders compared to labelled official released videos where available.',
              'final_probe': final}
    args.output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('COMPARISON_VIDEO_COMPLETE ' + json.dumps({'path': str(args.output), 'duration': elapsed, 'bytes': args.output.stat().st_size}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
