#!/usr/bin/env python3
"""Render a safe, repeatable talking-head or short-video edit with FFmpeg."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def run(command: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def probe(path: Path) -> Dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(run(command).stdout)


def stream_info(metadata: Dict[str, Any], stream_type: str) -> Optional[Dict[str, Any]]:
    return next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == stream_type), None)


def parse_edl(path: Path, duration: float) -> List[Tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = []
    for index, item in enumerate(payload.get("segments", []), start=1):
        try:
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"EDL 第 {index} 段无效：{exc}") from exc
        if end - start < 0.08:
            raise ValueError(f"EDL 第 {index} 段短于 0.08 秒")
        segments.append((start, end))
    if not segments:
        raise ValueError("EDL 没有有效 segments")
    return segments


def detect_silence(
    input_path: Path, threshold: float, minimum_duration: float, duration: float
) -> List[Tuple[float, float]]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-af",
        f"silencedetect=noise={threshold}dB:d={minimum_duration}",
        "-f",
        "null",
        "-",
    ]
    completed = run(command, check=False)
    text = completed.stderr
    starts = re.finditer(r"silence_start:\s*([0-9.]+)", text)
    ends = re.finditer(r"silence_end:\s*([0-9.]+)", text)
    events = [(match.start(), "start", float(match.group(1))) for match in starts]
    events += [(match.start(), "end", float(match.group(1))) for match in ends]
    events.sort(key=lambda item: item[0])
    ranges = []
    current = None
    for _, kind, timestamp in events:
        if kind == "start":
            current = timestamp
        elif current is not None:
            ranges.append((current, timestamp))
            current = None
    if current is not None:
        ranges.append((current, duration))
    return ranges


def subtract_silence(
    source_segments: List[Tuple[float, float]],
    silence_ranges: List[Tuple[float, float]],
    padding: float,
) -> List[Tuple[float, float]]:
    output: List[Tuple[float, float]] = []
    for segment_start, segment_end in source_segments:
        cursor = segment_start
        for silence_start, silence_end in silence_ranges:
            if silence_end <= segment_start or silence_start >= segment_end:
                continue
            cut_start = max(segment_start, silence_start + padding)
            cut_end = min(segment_end, silence_end - padding)
            if cut_start > cursor + 0.08:
                output.append((cursor, cut_start))
            cursor = max(cursor, cut_end)
        if segment_end > cursor + 0.08:
            output.append((cursor, segment_end))
    return output


def escape_subtitle_path(path: Path) -> str:
    value = str(path.resolve())
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def ffmpeg_has_filter(name: str) -> bool:
    completed = run(["ffmpeg", "-hide_banner", "-filters"], check=False)
    return bool(re.search(rf"\b{re.escape(name)}\b", completed.stdout + completed.stderr))


def parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d+):(\d+)[,.](\d+)", value.strip())
    if not match:
        raise ValueError(f"无效 SRT 时间码：{value}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / (10 ** len(match.group(4)))


def parse_srt(path: Path) -> List[Dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip())
    cues = []
    for block_number, block in enumerate(blocks, start=1):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        timing = lines[timing_index].split("-->")
        if len(timing) != 2:
            raise ValueError(f"SRT 第 {block_number} 块时间行无效")
        start = parse_srt_timestamp(timing[0])
        end = parse_srt_timestamp(timing[1].split()[0])
        text = "\n".join(lines[timing_index + 1 :]).strip()
        emphasis_match = re.match(r"^(?:【重点】|\[重点\]|重点[:：])\s*", text)
        emphasis = emphasis_match is not None
        if emphasis_match:
            text = text[emphasis_match.end() :].strip()
        if text and end > start:
            cues.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "emphasis": emphasis,
                }
            )
    if not cues:
        raise ValueError("SRT 没有有效字幕")
    return cues


def find_caption_font() -> Path:
    candidates = [
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError("找不到可用于字幕栅格化的系统字体")


def wrap_caption(text: str, font: Any, max_width: int, draw: Any) -> List[str]:
    lines: List[str] = []
    for paragraph in text.splitlines():
        current = ""
        for character in paragraph:
            candidate = current + character
            width = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)[2]
            if current and width > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines or [text]


def rasterize_captions(
    subtitle: Path,
    width: int,
    height: int,
    output_dir: Path,
    cues: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ValueError("FFmpeg 无 subtitles 滤镜，且 Python 缺少 Pillow，无法烧录字幕") from exc
    cues = cues or parse_srt(subtitle)
    base_font_size = max(24, round(width * 0.052))
    font_path = find_caption_font()
    output = []
    for index, cue in enumerate(cues):
        emphasis = bool(cue.get("emphasis"))
        font_size = round(base_font_size * 1.12) if emphasis else base_font_size
        font = ImageFont.truetype(str(font_path), font_size)
        stroke_width = 4 if emphasis else 3
        fill_color = (255, 213, 79, 255) if emphasis else (255, 255, 255, 255)
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        lines = wrap_caption(cue["text"], font, round(width * 0.84), draw)
        spacing = max(6, round(font_size * 0.18))
        boxes = [
            draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            for line in lines
        ]
        text_width = max(box[2] - box[0] for box in boxes)
        line_height = max(box[3] - box[1] for box in boxes)
        text_height = len(lines) * line_height + max(0, len(lines) - 1) * spacing
        pad_x = round(font_size * 0.45)
        pad_y = round(font_size * 0.30)
        left = round((width - text_width) / 2)
        top = height - round(height * 0.09) - text_height
        draw.rounded_rectangle(
            (left - pad_x, top - pad_y, left + text_width + pad_x, top + text_height + pad_y),
            radius=max(8, round(font_size * 0.25)),
            fill=(0, 0, 0, 155),
        )
        cursor = top
        for line, box in zip(lines, boxes):
            line_width = box[2] - box[0]
            x = round((width - line_width) / 2)
            draw.text(
                (x, cursor),
                line,
                font=font,
                fill=fill_color,
                stroke_width=stroke_width,
                stroke_fill=(0, 0, 0, 255),
            )
            cursor += line_height + spacing
        path = output_dir / f"caption-{index:04d}.png"
        image.save(path)
        output.append({**cue, "path": path})
    return output


def video_chain(
    label: str,
    aspect: str,
    subtitle: Optional[Path],
    rasterized_captions: List[Dict[str, Any]],
) -> Tuple[str, str]:
    filters = []
    output = label
    if aspect != "keep":
        width, height = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080),
        }[aspect]
        filters.append(
            f"[{output}]split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:2[bg];"
            f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[framed]"
        )
        output = "framed"
    if subtitle and not rasterized_captions:
        style = (
            "FontName=PingFang SC,FontSize=18,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,"
            "Alignment=2,MarginV=80"
        )
        filters.append(
            f"[{output}]subtitles=filename='{escape_subtitle_path(subtitle)}':"
            f"force_style='{style}'[subbed]"
        )
        output = "subbed"
    for index, cue in enumerate(rasterized_captions):
        input_index = cue["input_index"]
        filters.append(f"[{input_index}:v]format=rgba[caption{index}]")
        filters.append(
            f"[{output}][caption{index}]overlay=0:0:"
            f"enable='between(t,{cue['start']:.6f},{cue['end']:.6f})':"
            f"shortest=1[subbed{index}]"
        )
        output = f"subbed{index}"
    filters.append(f"[{output}]setsar=1,format=yuv420p[vout]")
    return ";".join(filters), "vout"


def build_filter(
    segments: List[Tuple[float, float]],
    has_audio: bool,
    aspect: str,
    subtitle: Optional[Path],
    rasterized_captions: List[Dict[str, Any]],
    music: Optional[Path],
    music_volume: float,
) -> Tuple[str, bool]:
    chains = []
    concat_inputs = []
    for index, (start, end) in enumerate(segments):
        chains.append(f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{index}]")
        concat_inputs.append(f"[v{index}]")
        if has_audio:
            chains.append(f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{index}]")
            concat_inputs.append(f"[a{index}]")
    if has_audio:
        chains.append("".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=1[vcat][acat]")
    else:
        chains.append("".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=0[vcat]")
    video_filters, _ = video_chain("vcat", aspect, subtitle, rasterized_captions)
    chains.append(video_filters)
    output_has_audio = has_audio or music is not None
    planned_duration = sum(end - start for start, end in segments)
    if has_audio:
        chains.append("[acat]loudnorm=I=-16:TP=-1.5:LRA=11[speech]")
    if music:
        chains.append(
            f"[1:a]atrim=0:{planned_duration:.6f},asetpts=PTS-STARTPTS,"
            f"volume={music_volume:.4f},afade=t=out:st={max(0.0, planned_duration - 1.0):.6f}:d=1[music]"
        )
        if has_audio:
            chains.append("[speech][music]amix=inputs=2:duration=first:dropout_transition=2[aout]")
        else:
            chains.append("[music]anull[aout]")
    elif has_audio:
        chains.append("[speech]anull[aout]")
    return ";".join(chains), output_has_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--aspect", choices=("keep", "9:16", "16:9", "1:1"), default="keep")
    parser.add_argument("--subtitle")
    parser.add_argument("--edl-json")
    parser.add_argument("--trim-start", type=float, default=0.0)
    parser.add_argument("--trim-end", type=float, default=0.0, help="从结尾裁掉的秒数")
    parser.add_argument("--remove-silence", action="store_true")
    parser.add_argument("--silence-threshold", type=float, default=-38.0)
    parser.add_argument("--silence-duration", type=float, default=0.55)
    parser.add_argument("--silence-padding", type=float, default=0.10)
    parser.add_argument("--music")
    parser.add_argument("--music-volume", type=float, default=0.08)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--plan-json")
    parser.add_argument("--qc-json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise SystemExit(f"缺少依赖：{executable}")
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_path.is_file():
        raise SystemExit(f"输入视频不存在：{input_path}")
    if input_path == output_path:
        raise SystemExit("输出路径不能与原视频相同")
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"输出已存在；如确认覆盖新产物，请加 --overwrite：{output_path}")
    subtitle = Path(args.subtitle).resolve() if args.subtitle else None
    music = Path(args.music).resolve() if args.music else None
    edl = Path(args.edl_json).resolve() if args.edl_json else None
    for label, path in (("字幕", subtitle), ("音乐", music), ("EDL", edl)):
        if path and not path.is_file():
            raise SystemExit(f"{label}文件不存在：{path}")
    if not 0 <= args.music_volume <= 1:
        raise SystemExit("--music-volume 必须在 0–1 之间")
    metadata = probe(input_path)
    duration = float(metadata["format"]["duration"])
    video = stream_info(metadata, "video")
    audio = stream_info(metadata, "audio")
    if not video:
        raise SystemExit("输入文件没有视频流")
    clip_end = duration - max(0.0, args.trim_end)
    clip_start = max(0.0, args.trim_start)
    if clip_end - clip_start < 0.1:
        raise SystemExit("裁切后的时长小于 0.1 秒")
    segments = parse_edl(edl, duration) if edl else [(clip_start, clip_end)]
    silence_ranges: List[Tuple[float, float]] = []
    if args.remove_silence:
        if not audio:
            raise SystemExit("输入没有音频流，不能执行静音压缩")
        silence_ranges = detect_silence(input_path, args.silence_threshold, args.silence_duration, duration)
        segments = subtract_silence(segments, silence_ranges, args.silence_padding)
        if not segments:
            raise SystemExit("静音压缩后没有可保留片段")
    target_width, target_height = (
        {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080),
        }.get(args.aspect)
        or (int(video["width"]), int(video["height"]))
    )
    caption_temp: Optional[tempfile.TemporaryDirectory] = None
    rasterized_captions: List[Dict[str, Any]] = []
    parsed_subtitle_cues: List[Dict[str, Any]] = []
    subtitle_mode = None
    if subtitle:
        parsed_subtitle_cues = parse_srt(subtitle)
        has_emphasis = any(cue.get("emphasis") for cue in parsed_subtitle_cues)
        if ffmpeg_has_filter("subtitles") and not has_emphasis:
            subtitle_mode = "ffmpeg-subtitles"
        else:
            subtitle_mode = "pillow-raster-overlay"
            caption_temp = tempfile.TemporaryDirectory(prefix="self-media-captions-")
            rasterized_captions = rasterize_captions(
                subtitle,
                target_width,
                target_height,
                Path(caption_temp.name),
                cues=parsed_subtitle_cues,
            )
    next_input_index = 1 + (1 if music else 0)
    for cue in rasterized_captions:
        cue["input_index"] = next_input_index
        next_input_index += 1
    filter_complex, output_has_audio = build_filter(
        segments,
        audio is not None,
        args.aspect,
        subtitle,
        rasterized_captions,
        music,
        args.music_volume,
    )
    command = ["ffmpeg", "-hide_banner", "-y" if args.overwrite else "-n", "-i", str(input_path)]
    if music:
        command += ["-stream_loop", "-1", "-i", str(music)]
    for cue in rasterized_captions:
        command += ["-loop", "1", "-framerate", "1", "-i", str(cue["path"])]
    command += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-movflags",
        "+faststart",
    ]
    if output_has_audio:
        command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-an"]
    command += [str(output_path)]
    plan = {
        "input": str(input_path),
        "output": str(output_path),
        "source": {
            "duration": duration,
            "video": video,
            "audio": audio,
        },
        "settings": {
            "aspect": args.aspect,
            "subtitle": str(subtitle) if subtitle else None,
            "subtitle_mode": subtitle_mode,
            "music": str(music) if music else None,
            "remove_silence": args.remove_silence,
        },
        "silence_ranges_detected": silence_ranges,
        "subtitle_cues": [
            {
                "start": cue["start"],
                "end": cue["end"],
                "text": cue["text"],
                "emphasis": bool(cue.get("emphasis")),
            }
            for cue in parsed_subtitle_cues
        ],
        "segments": [{"start": start, "end": end} for start, end in segments],
        "planned_duration": round(sum(end - start for start, end in segments), 3),
        "command": command,
    }
    if args.plan_json:
        plan_path = Path(args.plan_json)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        if caption_temp:
            caption_temp.cleanup()
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = run(command, check=False)
    if completed.returncode != 0:
        if caption_temp:
            caption_temp.cleanup()
        error_tail = "\n".join(completed.stderr.splitlines()[-25:])
        raise SystemExit(f"FFmpeg 渲染失败：\n{error_tail}")
    output_metadata = probe(output_path)
    output_duration = float(output_metadata["format"]["duration"])
    qc = {
        "ok": output_path.is_file() and output_path.stat().st_size > 0 and output_duration > 0,
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "duration": output_duration,
        "streams": output_metadata.get("streams", []),
        "duration_delta_from_plan": round(output_duration - plan["planned_duration"], 3),
    }
    if args.qc_json:
        qc_path = Path(args.qc_json)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if caption_temp:
        caption_temp.cleanup()
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    return 0 if qc["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
