import asyncio
import base64
import binascii
import io
import ipaddress
import json
import math
import mimetypes
import os
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi.responses import JSONResponse


SUPPORTED_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".aac",
    ".m4a",
    ".webm",
    ".mp4",
}
MIME_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
}


class CompatAPIError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class MaterializedAudio:
    path: str
    mime_type: str
    file_size: int


def openai_error(message: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error"}},
    )


def parse_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def is_upload(value) -> bool:
    return hasattr(value, "read") and hasattr(value, "filename")


async def request_payload(request) -> dict:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        return dict(await request.form())
    if content_type.startswith("application/json") or not content_type:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise CompatAPIError("request body must be a JSON object")
        return payload
    raise CompatAPIError("Content-Type must be application/json or multipart/form-data", 415)


def parse_extra_params(payload: dict) -> dict:
    extra_params = payload.get("extra_params") or {}
    if isinstance(extra_params, str):
        try:
            extra_params = json.loads(extra_params)
        except json.JSONDecodeError as ex:
            raise CompatAPIError("extra_params must be valid JSON") from ex
    if not isinstance(extra_params, dict):
        raise CompatAPIError("extra_params must be an object")
    return extra_params


def first_value(payload: dict, extra_params: dict, *names, default=None):
    for name in names:
        if name in extra_params and extra_params[name] is not None:
            return extra_params[name]
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def parse_emotion_vector(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as ex:
            raise CompatAPIError("emo_vector must be valid JSON") from ex
    if (
        not isinstance(value, list)
        or len(value) != 8
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        raise CompatAPIError("emo_vector must contain exactly 8 numbers")
    vector = [float(item) for item in value]
    if any(item < 0 or item > 1.2 for item in vector):
        raise CompatAPIError("each emo_vector value must be between 0 and 1.2")
    if sum(vector) > 1.5:
        raise CompatAPIError("the sum of emo_vector must not exceed 1.5")
    return vector


def parse_speed(value) -> float:
    try:
        speed = float(value) if value not in (None, "") else 1.0
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(speed) or speed < 0.25 or speed > 4.0:
        return 1.0
    return speed


def _atempo_filter(speed: float) -> str:
    factors = []
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    factors.append(speed)
    return ",".join(f"atempo={factor:.10g}" for factor in factors)


def encode_audio(wav, sample_rate: int, response_format: str, speed: float = 1.0) -> tuple[bytes, str]:
    formats = {
        "wav": ("WAV", "PCM_16", "audio/wav"),
        "flac": ("FLAC", "PCM_16", "audio/flac"),
        "pcm": ("RAW", "PCM_16", "audio/pcm"),
    }
    if response_format not in formats:
        raise CompatAPIError("response_format must be wav, flac, or pcm for the IndexTTS2 backend")
    container_format, subtype, media_type = formats[response_format]

    if speed != 1.0:
        with io.BytesIO() as input_buffer:
            sf.write(input_buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
            input_bytes = input_buffer.getvalue()

        channels = 1 if getattr(wav, "ndim", 1) == 1 else wav.shape[1]
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-filter:a",
            _atempo_filter(speed),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as ex:
            stderr_output = getattr(ex, "stderr", None) or b""
            if isinstance(stderr_output, bytes):
                stderr_output = stderr_output.decode("utf-8", errors="replace")
            stderr = stderr_output.strip()
            raise RuntimeError(f"ffmpeg audio speed adjustment failed: {stderr}") from ex

        adjusted_wav = np.frombuffer(result.stdout, dtype="<i2")
        if channels > 1:
            adjusted_wav = adjusted_wav.reshape(-1, channels)
        wav = adjusted_wav

    with io.BytesIO() as audio_buffer:
        sf.write(audio_buffer, wav, sample_rate, format=container_format, subtype=subtype)
        return audio_buffer.getvalue(), media_type


async def close_uploads(*values):
    closed = set()
    for value in values:
        if is_upload(value) and id(value) not in closed:
            closed.add(id(value))
            await value.close()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def max_audio_bytes() -> int:
    return int(os.getenv("MAX_VOICE_AUDIO_BYTES", str(10 * 1024 * 1024)))


def audio_suffix(filename: Optional[str], mime_type: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in SUPPORTED_AUDIO_SUFFIXES:
        return suffix
    normalized_mime = (mime_type or "").split(";", 1)[0].strip().lower()
    return MIME_SUFFIXES.get(normalized_mime, ".wav")


def audio_mime_type(suffix: str, provided: Optional[str] = None) -> str:
    normalized = (provided or "").split(";", 1)[0].strip().lower()
    if normalized.startswith("audio/") or normalized == "video/mp4":
        return normalized
    return mimetypes.types_map.get(suffix, "application/octet-stream")


async def save_upload_file(upload, output_dir: str, stem: str) -> MaterializedAudio:
    suffix = audio_suffix(getattr(upload, "filename", None), getattr(upload, "content_type", None))
    mime_type = audio_mime_type(suffix, getattr(upload, "content_type", None))
    output_path = os.path.join(output_dir, stem + suffix)
    total = 0
    try:
        with open(output_path, "wb") as output_file:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_audio_bytes():
                    raise CompatAPIError("audio file exceeds the 10 MB upload limit", 413)
                output_file.write(chunk)
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
    if total == 0:
        raise CompatAPIError("audio file is empty")
    return MaterializedAudio(output_path, mime_type, total)


def _validate_remote_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CompatAPIError("audio URL must use http or https")
    if parsed.username or parsed.password:
        raise CompatAPIError("audio URL must not contain credentials")
    if _env_bool("ALLOW_PRIVATE_AUDIO_URLS"):
        return
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or default_port, type=socket.SOCK_STREAM)
    except socket.gaierror as ex:
        raise CompatAPIError(f"unable to resolve audio URL host: {ex}") from ex
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise CompatAPIError(
                "audio URL resolves to a private or non-public address; "
                "set ALLOW_PRIVATE_AUDIO_URLS=1 only for trusted internal URLs"
            )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _download_http_audio(url: str) -> tuple[bytes, str, str]:
    opener = urllib.request.build_opener(_NoRedirectHandler())
    current_url = url
    timeout = float(os.getenv("AUDIO_URL_TIMEOUT_SECONDS", "15"))
    for _ in range(6):
        _validate_remote_url(current_url)
        request = urllib.request.Request(current_url, headers={"User-Agent": "IndexTTS2/voice-fetch"})
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as ex:
            if ex.code in {301, 302, 303, 307, 308}:
                location = ex.headers.get("Location")
                if not location:
                    raise CompatAPIError("audio URL redirect has no Location header") from ex
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            raise CompatAPIError(f"audio URL returned HTTP {ex.code}", 400) from ex
        except (urllib.error.URLError, TimeoutError, OSError) as ex:
            raise CompatAPIError(f"failed to download audio URL: {ex}") from ex

        with response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_audio_bytes():
                raise CompatAPIError("remote audio exceeds the 10 MB limit", 413)
            chunks = []
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > max_audio_bytes():
                    raise CompatAPIError("remote audio exceeds the 10 MB limit", 413)
                chunks.append(chunk)
            if total == 0:
                raise CompatAPIError("remote audio is empty")
            mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            suffix = audio_suffix(urllib.parse.urlsplit(current_url).path, mime_type)
            return b"".join(chunks), suffix, audio_mime_type(suffix, mime_type)
    raise CompatAPIError("audio URL has too many redirects")


def _decode_audio_base64(source: str) -> tuple[bytes, str, str]:
    source = source.strip()
    mime_type = "audio/wav"
    suffix = ".wav"
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise CompatAPIError("audio data URL must use base64 encoding")
        mime_type = header[5:].split(";", 1)[0].lower() or "audio/wav"
        suffix = audio_suffix(None, mime_type)
    else:
        payload = source[7:] if source.startswith("base64:") else source
        payload = "".join(payload.split())
        if len(payload) < 128:
            raise CompatAPIError("voice was not found and is not a supported audio URL or base64 value", 404)
    if len(payload) > (max_audio_bytes() * 4 // 3) + 8:
        raise CompatAPIError("base64 audio exceeds the 10 MB limit", 413)
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as ex:
        raise CompatAPIError("invalid base64 audio data") from ex
    if not data:
        raise CompatAPIError("base64 audio is empty")
    if len(data) > max_audio_bytes():
        raise CompatAPIError("base64 audio exceeds the 10 MB limit", 413)
    return data, suffix, audio_mime_type(suffix, mime_type)


async def materialize_audio_source(
    source: str,
    output_dir: str,
    stem: str,
) -> MaterializedAudio:
    if not isinstance(source, str) or not source.strip():
        raise CompatAPIError("audio source must be a non-empty string")
    source = source.strip()

    if source.startswith(("http://", "https://")):
        data, suffix, mime_type = await asyncio.to_thread(_download_http_audio, source)
    elif source.startswith(("data:", "base64:")) or len(source) >= 128:
        data, suffix, mime_type = _decode_audio_base64(source)
    else:
        raise CompatAPIError("voice was not found and is not a supported HTTP URL or base64 value", 404)

    output_path = os.path.join(output_dir, stem + suffix)
    with open(output_path, "wb") as output_file:
        output_file.write(data)
    return MaterializedAudio(output_path, mime_type, len(data))
