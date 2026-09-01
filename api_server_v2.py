import os
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import tempfile
import time
import soundfile as sf

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.api_compat import (
    CompatAPIError,
    close_uploads,
    encode_audio,
    first_value,
    is_upload,
    materialize_audio_source,
    openai_error,
    parse_bool,
    parse_emotion_vector,
    parse_extra_params,
    parse_speed,
    request_payload,
    save_upload_file,
)
from indextts.infer_vllm_v2 import IndexTTS2

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        # logger.info(f"Emo control mode:{emo_control_method}, vec:{vec}")
        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    speaker_upload = None
    emotion_upload = None
    voice_value = None
    ref_audio_value = None
    try:
        if tts is None:
            raise CompatAPIError("TTS model is not initialized", 503)
        payload = await request_payload(request)
        extra_params = parse_extra_params(payload)

        text = payload.get("input", payload.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise CompatAPIError("input is required and must be a non-empty string")

        response_format = str(payload.get("response_format", "wav")).lower()
        speed = parse_speed(first_value(payload, extra_params, "speed", default=1.0))
        if parse_bool(payload.get("stream"), False):
            raise CompatAPIError("streaming is not supported by this IndexTTS2 compatibility server")

        speaker_upload = payload.get("spk_audio")
        ref_audio_value = payload.get("ref_audio")
        voice_value = payload.get("voice")
        if isinstance(ref_audio_value, list):
            if len(ref_audio_value) != 1:
                raise CompatAPIError("IndexTTS2 accepts exactly one ref_audio value")
            ref_audio_value = ref_audio_value[0]

        emotion_source = first_value(payload, extra_params, "emo_audio")
        emotion_upload = emotion_source if is_upload(emotion_source) else None
        emo_text = first_value(payload, extra_params, "emo_text")
        use_emo_text = parse_bool(
            first_value(payload, extra_params, "use_emo_text"),
            bool(emo_text),
        )
        emo_vector = parse_emotion_vector(
            first_value(payload, extra_params, "emo_vector", "emo_vec")
        )
        emo_alpha = float(first_value(payload, extra_params, "emo_alpha", "emo_weight", default=1.0))
        if emo_alpha < 0 or emo_alpha > 1:
            raise CompatAPIError("emo_alpha must be between 0 and 1")
        use_random = parse_bool(first_value(payload, extra_params, "use_random", "emo_random"), False)
        max_text_tokens = int(
            first_value(
                payload,
                extra_params,
                "max_text_tokens_per_sentence",
                "max_text_tokens_per_segment",
                default=120,
            )
        )
        if max_text_tokens < 1:
            raise CompatAPIError("max_text_tokens_per_sentence must be positive")

        legacy_method = payload.get("emo_control_method")
        if legacy_method is not None:
            legacy_method = int(legacy_method)
            if legacy_method not in {0, 1, 2, 3}:
                raise CompatAPIError("emo_control_method must be 0, 1, 2, or 3")
            use_emo_text = legacy_method == 3
            if legacy_method != 1:
                emotion_source = None
            if legacy_method != 2:
                emo_vector = None

        if use_emo_text:
            emotion_source = None
            emo_vector = None
        elif emo_vector is not None:
            emotion_source = None
            emo_vector = [round(item * emo_alpha, 4) for item in emo_vector]

        with tempfile.TemporaryDirectory(prefix="indextts2-request-") as temp_dir:
            if is_upload(speaker_upload):
                speaker_audio = await save_upload_file(speaker_upload, temp_dir, "speaker")
            elif is_upload(ref_audio_value):
                speaker_audio = await save_upload_file(ref_audio_value, temp_dir, "speaker")
            elif ref_audio_value:
                speaker_audio = await materialize_audio_source(ref_audio_value, temp_dir, "speaker")
            elif is_upload(voice_value):
                speaker_audio = await save_upload_file(voice_value, temp_dir, "speaker")
            elif voice_value:
                speaker_audio = await materialize_audio_source(
                    voice_value,
                    temp_dir,
                    "speaker",
                )
            else:
                raise CompatAPIError("voice, ref_audio, or spk_audio is required")

            emotion_path = None
            if not use_emo_text and emo_vector is None and emotion_source:
                if is_upload(emotion_source):
                    emotion_audio = await save_upload_file(emotion_source, temp_dir, "emotion")
                else:
                    emotion_audio = await materialize_audio_source(
                        emotion_source,
                        temp_dir,
                        "emotion",
                    )
                emotion_path = emotion_audio.path

            sample_rate, wav = await tts.infer(
                spk_audio_prompt=speaker_audio.path,
                text=text,
                output_path=None,
                emo_audio_prompt=emotion_path,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
                use_random=use_random,
                max_text_tokens_per_sentence=max_text_tokens,
            )

        audio_bytes, media_type = await asyncio.to_thread(
            encode_audio,
            wav,
            sample_rate,
            response_format,
            speed,
        )
        return Response(content=audio_bytes, media_type=media_type)
    except CompatAPIError as ex:
        return openai_error(str(ex), ex.status_code)
    except (TypeError, ValueError) as ex:
        return openai_error(str(ex))
    except Exception:
        logger.exception("/v1/audio/speech failed")
        return openai_error("internal server error", 500)
    finally:
        await close_uploads(speaker_upload, ref_audio_value, voice_value, emotion_upload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)
