# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    ref_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models (CustomVoice/VoiceDesign/Base) that provides:
      - from_pretrained() initialization via AutoModel/AutoProcessor
      - generation APIs for:
          * CustomVoice: generate_custom_voice()
          * VoiceDesign: generate_voice_design()
          * Base: generate_voice_clone() + create_voice_clone_prompt()
      - consistent output: (wavs: List[np.ndarray], sample_rate: int)

    Notes:
      - This wrapper expects the underlying model class to be `Qwen3TTSForConditionalGeneration`
      - Language / speaker validation is done via model methods:
          model.get_supported_languages(), model.get_supported_speakers()
    """

    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: Optional[Dict[str, Any]] = None):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3TTSModel":
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Qwen3TTSModel:
                Wrapper instance containing `model`, `processor`, and generation defaults.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)

        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        """
        Validate that requested languages are supported by the model.

        Args:
            languages (List[str]): Language names for each sample.

        Raises:
            ValueError: If any language is not supported.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        """
        Validate that requested speakers are supported by the Instruct model.

        Args:
            speakers (List[Optional[str]]): Speaker names for each sample.

        Raises:
            ValueError: If any speaker is not supported.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                a[0] = np.mean(a[0], axis=-1).astype(np.float32)
                out[i] = (a[0], a[1])
        return out

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        input_ids = []
        for text in texts:
            input = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = input["input_ids"].to(self.device)
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from generate_config.json if present.
          - Otherwise, fall back to the hard defaults.

        Args:
            do_sample, top_k, top_p, temperature, repetition_penalty,
            subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens:
                Common generation parameters.
            **kwargs:
                Other arguments forwarded to model.generate().

        Returns:
            Dict[str, Any]: Final kwargs to pass into model.generate().
        """
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    # voice clone model
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.

        Modes:
          - x_vector_only_mode=True:
              Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
              This is mutually exclusive with ICL.
          - x_vector_only_mode=False:
              ICL mode is enabled automatically (icl_mode=True). In this case ref_text is required,
              because the model continues/conditions on the reference text + reference speech codes.

        Batch behavior:
          - ref_audio can be a single item or a list.
          - ref_text and x_vector_only_mode can be scalars or lists.
          - If any of them are lists with length > 1, lengths must match.

        Audio input:
          - str: local wav path / URL / base64
          - (np.ndarray, sr): waveform + sampling rate

        Args:
            ref_audio:
                Reference audio(s) used to extract:
                  - ref_code via `model.speech_tokenizer.encode(...)`
                  - ref_spk_embedding via `model.extract_speaker_embedding(...)` (resampled to 24k)
            ref_text:
                Reference transcript(s). Required when x_vector_only_mode=False (ICL mode).
            x_vector_only_mode:
                Whether to use speaker embedding only. If False, ICL mode will be used.

        Returns:
            List[VoiceClonePromptItem]:
                List of prompt items that can be converted into `voice_clone_prompt` dict.

        Raises:
            ValueError:
                - If x_vector_only_mode=False but ref_text is missing.
                - If batch lengths mismatch.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support create_voice_clone_prompt, Please check Model Card or Readme for more details."
            )
        
        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        xvec_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}"
            )

        normalized = self._normalize_audio_inputs(ref_audio_list)

        ref_wavs_for_code: List[np.ndarray] = []
        ref_sr_for_code: List[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        if len(set(ref_sr_for_code)) == 1:
            enc = self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

        items: List[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(y=wav_resample.astype(np.float32), 
                                           orig_sr=int(sr), 
                                           target_sr=self.model.speaker_encoder_sample_rate)

            spk_emb = self.model.extract_speaker_embedding(audio=wav_resample,
                                                           sr=self.model.speaker_encoder_sample_rate)

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        return items

    def _prompt_items_to_voice_clone_prompt(self, items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    # voice clone model
    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Voice clone speech using the Base model.

        You can provide either:
          - (ref_audio, ref_text, x_vector_only_mode) and let this method build the prompt, OR
          - `VoiceClonePromptItem` returned by `create_voice_clone_prompt`, OR
          - a list of `VoiceClonePromptItem` returned by `create_voice_clone_prompt`.
        
        `ref_audio` Supported forms:
        - str: wav path / URL / base64 audio string
        - (np.ndarray, sr): waveform + sampling rate
        - list of the above

        Input flexibility:
          - text/language can be scalar or list.
          - prompt can be single or batch.
          - If batch mode (len(text)>1), lengths must match.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            ref_audio:
                Reference audio(s) for prompt building. Required if voice_clone_prompt is not provided.
            ref_text:
                Reference text(s) used for ICL mode (required when x_vector_only_mode=False).
            x_vector_only_mode:
                If True, only speaker embedding is used (ignores ref_text/ref_code).
                If False, ICL mode is used automatically.
            voice_clone_prompt:
                list[VoiceClonePromptItem] from `create_voice_clone_prompt`.
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If batch sizes mismatch or required prompt inputs are missing.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )
        
        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for i, rt in enumerate(ref_texts_for_ids):
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

        wavs_out: List[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    # voice design model
    @torch.no_grad()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )
        
        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        instructs = self._ensure_list(instruct)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}")

        self._validate_languages(languages)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    # custom voice model
    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id, optionally controlled by instruction text.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            speaker:
                Speaker name(s). Will be validated against `model.get_supported_speakers()` (case-insensitive).
            instruct:
                Optional instruction(s). If None, treated as empty (no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If any speaker/language is unsupported or batch sizes mismatch.
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        speakers = self._ensure_list(speaker)
        if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
            instruct = None
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs


    # =========================================================================
    # STREAMING GENERATION — private helpers + public methods
    # =========================================================================

    def _resolve_backend(self, backend: str) -> str:
        """Resolve backend string with auto-detection for CUDA graphs."""
        if backend == "auto":
            if torch.cuda.is_available():
                return "faster"
            return "dynamic"
        return backend

    def _init_cuda_graphs(self):
        """Lazy-initialize PredictorGraph and TalkerGraph on first streaming call."""
        if hasattr(self, "_predictor_graph") and self._predictor_graph is not None:
            return

        from .predictor_graph import PredictorGraph
        from .talker_graph import TalkerGraph

        m = self.model
        talker = m.talker
        talker_config = m.config.talker_config
        predictor = talker.code_predictor
        pred_config = predictor.model.config
        talker_hidden = talker_config.hidden_size

        device = str(self.device).split(":")[0] if isinstance(self.device, torch.device) else str(self.device)
        # Use the model's actual parameter dtype (the config lookup never had a "dtype" key)
        dtype = next(talker.parameters()).dtype

        self._predictor_graph = PredictorGraph(
            predictor, pred_config, talker_hidden,
            device=device, dtype=dtype,
            do_sample=True, top_k=50, temperature=0.9,
        )
        self._talker_graph = TalkerGraph(
            talker.model, talker_config,
            device=device, dtype=dtype, max_seq_len=2048,
        )
        self._graphs_initialized = True

    def _warmup_cuda_graphs(self, prefill_len: int = 100):
        """Warm up CUDA graphs (capture on first real run)."""
        if not getattr(self, "_graphs_initialized", False):
            return
        if getattr(self, "_graphs_warmed_up", False):
            return
        self._predictor_graph.capture(num_warmup=3)
        self._talker_graph.capture(prefill_len=prefill_len, num_warmup=3)
        self._graphs_warmed_up = True

    def _prepare_generation_custom(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
    ):
        """Prepare generation inputs for custom-voice model (mirrors FasterQwen3TTS)."""
        input_texts = [self._build_assistant_text(text)]
        input_ids = self._tokenize_texts(input_texts)

        instruct_ids = []
        if instruct is None or instruct == "":
            instruct_ids.append(None)
        else:
            instruct_ids.append(self._tokenize_texts([self._build_instruct_text(instruct)])[0])

        m = self.model
        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=[None],
            voice_clone_prompt=None,
            languages=[language] if language is not None else ["Auto"],
            speakers=[speaker],
            non_streaming_mode=non_streaming_mode,
            instruct_ids=instruct_ids,
        )

        return m, tie, tam, tth, tpe

    def _build_talker_inputs_local(
        self,
        m,
        input_ids,
        ref_ids,
        voice_clone_prompt,
        languages,
        speakers,
        non_streaming_mode: bool,
        instruct_ids=None,
    ):
        """Local copy of upstream talker input building for qwen-tts main repo."""
        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = m.generate_speaker_prompt(voice_clone_prompt)

        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        m.talker.text_projection(m.talker.get_text_embeddings()(instruct_id))
                    )

        if speakers is None:
            speakers = [None] * len(input_ids)

        trailing_text_hiddens = []
        tts_pad_embed = None

        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:
                    speaker_embed = None
                else:
                    if speaker.lower() not in m.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    spk_id = m.config.talker_config.spk_id[speaker.lower()]
                    speaker_embed = m.talker.get_input_embeddings()(
                        torch.tensor(spk_id, device=m.talker.device, dtype=input_id.dtype)
                    )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None
            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in m.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                language_id = m.config.talker_config.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker not in ("", None)
                and m.config.talker_config.spk_is_dialect[speaker.lower()]
            ):
                dialect = m.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = m.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = m.talker.text_projection(
                m.talker.get_text_embeddings()(
                    torch.tensor(
                        [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]],
                        device=m.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)

            if language_id is None:
                codec_prefill_list = [[
                    m.config.talker_config.codec_nothink_id,
                    m.config.talker_config.codec_think_bos_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]
            else:
                codec_prefill_list = [[
                    m.config.talker_config.codec_think_id,
                    m.config.talker_config.codec_think_bos_id,
                    language_id,
                    m.config.talker_config.codec_think_eos_id,
                ]]

            codec_input_emebdding_0 = m.talker.get_input_embeddings()(
                torch.tensor(codec_prefill_list, device=m.talker.device, dtype=input_id.dtype)
            )
            codec_input_emebdding_1 = m.talker.get_input_embeddings()(
                torch.tensor(
                    [[m.config.talker_config.codec_pad_id, m.config.talker_config.codec_bos_id]],
                    device=m.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1)

            _talker_input_embed_role = m.talker.text_projection(
                m.talker.get_text_embeddings()(input_id[:, :3])
            )
            _talker_input_embed = torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            ) + codec_input_emebdding[:, :-1]

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt.get("ref_code", None) is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = m.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(m.talker.device).clone(),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        m.talker.text_projection(
                            m.talker.get_text_embeddings()(input_id[:, 3:4])
                        )
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    m.talker.text_projection(
                                        m.talker.get_text_embeddings()(input_id[:, 3:-5])
                                    ),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + m.talker.get_input_embeddings()(
                                torch.tensor(
                                    [[m.config.talker_config.codec_bos_id]],
                                    device=m.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (
                            m.talker.text_projection(
                                m.talker.get_text_embeddings()(input_id[:, 4:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )

            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0,
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])

        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0,
        )
        arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
            len(trailing_text_original_lengths), -1
        )
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed

    def _get_suppress_mask(self, eos_id: int, vocab_size: int, device):
        """Boolean mask suppressing the top-1024 codec ids (except EOS).

        Built once per (device, vocab, eos) and cached — previously rebuilt on every
        generation call via ~1024 individual GPU element writes.
        """
        key = (str(device), int(vocab_size), int(eos_id))
        cache = getattr(self, "_suppress_mask_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        start = max(0, vocab_size - 1024)
        if start < vocab_size:
            idx = torch.arange(start, vocab_size, device=device)
            mask[idx] = True
            if start <= eos_id < vocab_size:
                mask[eos_id] = False
        self._suppress_mask_cache = (key, mask)
        return mask

    @torch.inference_mode()
    def _fast_generate_streaming(
        self,
        talker,
        tie, tam, tth, tpe,
        config,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ):
        """Streaming generation using CUDA graphs (predictor + talker)."""
        from .sampling import apply_repetition_penalty, sample_logits

        eos_id = config.codec_eos_token_id
        device = tie.device

        suppress_mask = self._get_suppress_mask(eos_id, config.vocab_size, device)

        predictor = talker.code_predictor
        talker_codec_embed = talker.get_input_embeddings()
        talker_codec_head = talker.codec_head
        predictor_codec_embeds = predictor.get_input_embeddings()
        num_code_groups = config.num_code_groups

        t_start = time.time()

        out = talker.forward(
            inputs_embeds=tie,
            attention_mask=tam,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=tth,
            tts_pad_embed=tpe,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )

        talker_past_kv = out.past_key_values
        past_hidden = out.past_hidden
        gen_step = out.generation_step

        logits = out.logits[:, -1, :]
        suppress_eos = min_new_tokens > 0
        token = sample_logits(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )

        prefill_len = self._talker_graph.prefill_kv(talker_past_kv)
        rope_deltas = getattr(talker, "rope_deltas", None)
        self._talker_graph.set_generation_state(tam, rope_deltas)

        torch.cuda.synchronize()
        t_prefill = time.time() - t_start

        chunk_buffer = []
        # Unique-first-token buffer for the repetition penalty: O(1) per step instead of
        # re-stacking the whole history every step (O(T^2) allocations/copies).
        uniq_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
        n_uniq = 0
        seen_set = set()
        total_steps = 0
        chunk_count = 0
        chunk_start = time.time()

        for step_idx in range(max_new_tokens):
            tok_val = token.item()
            if tok_val == eos_id:
                break
            n_emitted = step_idx + 1
            if tok_val not in seen_set:
                seen_set.add(tok_val)
                uniq_buf[n_uniq] = tok_val
                n_uniq += 1

            last_id_hidden = talker_codec_embed(token.unsqueeze(1))
            pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            codebook_token_ids = self._predictor_graph.run(pred_input)

            all_cb = torch.cat([token.view(1), codebook_token_ids])
            chunk_buffer.append(all_cb.detach())

            codec_hiddens = [last_id_hidden]
            for i in range(num_code_groups - 1):
                codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)

            if gen_step < tth.shape[1]:
                inputs_embeds = inputs_embeds + tth[:, gen_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tpe

            current_pos = prefill_len + step_idx
            if current_pos >= self._talker_graph.max_seq_len - 1:
                break

            hidden_states = self._talker_graph.run(inputs_embeds, position=current_pos)
            logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

            if repetition_penalty != 1.0 and n_uniq > 0:
                # uniq_buf[:n_uniq] holds each distinct first token exactly once
                logits = apply_repetition_penalty(logits, uniq_buf[:n_uniq], repetition_penalty)

            suppress_eos = n_emitted < min_new_tokens
            token = sample_logits(
                logits.squeeze(0), temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, suppress_mask=suppress_mask,
                suppress_tokens=[eos_id] if suppress_eos else None,
            )
            past_hidden = hidden_states[:, -1:, :].clone()
            gen_step += 1

            if len(chunk_buffer) >= chunk_size:
                torch.cuda.synchronize()
                chunk_decode_time = time.time() - chunk_start
                total_steps += len(chunk_buffer)

                yield torch.stack(chunk_buffer), {
                    "chunk_index": chunk_count,
                    "chunk_steps": len(chunk_buffer),
                    "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                    "decode_ms": chunk_decode_time * 1000,
                    "total_steps_so_far": total_steps,
                    "is_final": False,
                }

                chunk_buffer = []
                chunk_count += 1
                chunk_start = time.time()
                # EOS termination is caught at the top of the loop, before any work.

        if chunk_buffer:
            torch.cuda.synchronize()
            chunk_decode_time = time.time() - chunk_start
            total_steps += len(chunk_buffer)

            yield torch.stack(chunk_buffer), {
                "chunk_index": chunk_count,
                "chunk_steps": len(chunk_buffer),
                "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                "decode_ms": chunk_decode_time * 1000,
                "total_steps_so_far": total_steps,
                "is_final": True,
            }

    @torch.inference_mode()
    def _parity_generate_streaming(
        self,
        talker,
        tie, tam, tth, tpe,
        config,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
    ):
        """Streaming generation using dynamic cache (no CUDA graphs)."""
        from .sampling import apply_repetition_penalty, sample_logits

        eos_id = config.codec_eos_token_id
        device = tie.device

        suppress_mask = self._get_suppress_mask(eos_id, config.vocab_size, device)

        t_start = time.time()

        out = talker.forward(
            inputs_embeds=tie,
            attention_mask=tam,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=tth,
            tts_pad_embed=tpe,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )

        talker_past_kv = out.past_key_values
        past_hidden = out.past_hidden
        gen_step = out.generation_step

        logits = out.logits[:, -1, :]
        suppress_eos = min_new_tokens > 0
        token = sample_logits(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )

        if tam is not None:
            tam = tam.clone()

        torch.cuda.synchronize()
        t_prefill = time.time() - t_start

        chunk_buffer = []
        # Unique-first-token buffer for the repetition penalty (O(1) per step).
        uniq_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
        n_uniq = 0
        seen_set = set()
        total_steps = 0
        chunk_count = 0
        chunk_start = time.time()

        for step_idx in range(max_new_tokens):
            tok_val = token.item()
            if tok_val == eos_id:
                break
            n_emitted = step_idx + 1
            if tok_val not in seen_set:
                seen_set.add(tok_val)
                uniq_buf[n_uniq] = tok_val
                n_uniq += 1

            cache_position = None
            if tam is not None:
                tam = torch.cat(
                    [tam, tam.new_ones((tam.shape[0], 1))], dim=1
                )
                cache_position = torch.tensor([tam.shape[1] - 1], device=tam.device)

            out = talker.forward(
                input_ids=token.view(1, 1),
                attention_mask=tam,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                trailing_text_hidden=tth,
                tts_pad_embed=tpe,
                generation_step=gen_step,
                past_hidden=past_hidden,
                past_key_values=talker_past_kv,
                cache_position=cache_position,
            )

            codec_ids = out.hidden_states[1]
            if codec_ids is None:
                break

            chunk_buffer.append(codec_ids.squeeze(0).detach())

            logits = out.logits[:, -1, :]
            if repetition_penalty != 1.0 and n_uniq > 0:
                # uniq_buf[:n_uniq] holds each distinct first token exactly once
                logits = apply_repetition_penalty(logits, uniq_buf[:n_uniq], repetition_penalty)

            suppress_eos = n_emitted < min_new_tokens
            token = sample_logits(
                logits, temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, suppress_mask=suppress_mask,
                suppress_tokens=[eos_id] if suppress_eos else None,
            )

            talker_past_kv = out.past_key_values
            past_hidden = out.past_hidden
            gen_step = out.generation_step

            if len(chunk_buffer) >= chunk_size:
                torch.cuda.synchronize()
                chunk_decode_time = time.time() - chunk_start
                total_steps += len(chunk_buffer)

                yield torch.stack(chunk_buffer), {
                    "chunk_index": chunk_count,
                    "chunk_steps": len(chunk_buffer),
                    "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                    "decode_ms": chunk_decode_time * 1000,
                    "total_steps_so_far": total_steps,
                    "is_final": False,
                }

                chunk_buffer = []
                chunk_count += 1
                chunk_start = time.time()
                # EOS termination is caught at the top of the loop, before any work.

        if chunk_buffer:
            torch.cuda.synchronize()
            chunk_decode_time = time.time() - chunk_start
            total_steps += len(chunk_buffer)

            yield torch.stack(chunk_buffer), {
                "chunk_index": chunk_count,
                "chunk_steps": len(chunk_buffer),
                "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                "decode_ms": chunk_decode_time * 1000,
                "total_steps_so_far": total_steps,
                "is_final": True,
            }

    def _decode_chunk_to_audio(self, speech_tokenizer, all_codes, ref_codes, context_frames=25):
        """Hybrid decode: accumulated early, sliding window after calibration."""
        n_new = all_codes[-1].shape[0] if all_codes else 0
        all_flat = torch.cat(all_codes, dim=0)
        n_total = all_flat.shape[0]

        samples_per_frame = getattr(self, "_stream_samples_per_frame", None)
        prev_audio_len = getattr(self, "_stream_prev_audio_len", 0)

        if samples_per_frame is None:
            if ref_codes is not None:
                codes_input = torch.cat([ref_codes.to(all_flat.device), all_flat], dim=0)
            else:
                codes_input = all_flat
            audio_list, sr = speech_tokenizer.decode({"audio_codes": codes_input.unsqueeze(0)})
            audio = audio_list[0]
            if hasattr(audio, "cpu"):
                audio = audio.flatten().cpu().numpy()
            else:
                audio = audio.flatten() if hasattr(audio, "flatten") else audio

            if ref_codes is not None:
                ref_len = ref_codes.shape[0]
                total_len = codes_input.shape[0]
                ref_audio_cut = int(ref_len / max(total_len, 1) * len(audio))
                gen_audio = audio[ref_audio_cut:]
            else:
                gen_audio = audio

            new_audio = gen_audio[prev_audio_len:]
            self._stream_prev_audio_len = len(gen_audio)

            if n_total >= context_frames:
                self._stream_samples_per_frame = len(gen_audio) / n_total
                samples_per_frame = self._stream_samples_per_frame
        else:
            ctx_start = max(0, n_total - n_new - context_frames)
            window = all_flat[ctx_start:]
            n_ctx = window.shape[0] - n_new

            audio_list, sr = speech_tokenizer.decode({"audio_codes": window.unsqueeze(0)})
            audio = audio_list[0]
            if hasattr(audio, "cpu"):
                audio = audio.flatten().cpu().numpy()
            else:
                audio = audio.flatten() if hasattr(audio, "flatten") else audio

            if n_ctx > 0:
                ctx_samples = int(round(n_ctx * samples_per_frame))
                new_audio = audio[ctx_samples:]
            else:
                new_audio = audio

        return new_audio, sr

    # =========================================================================
    # PUBLIC STREAMING METHODS
    # =========================================================================

    @torch.inference_mode()
    def generate_custom_voice_streaming(
        self,
        text: str,
        speaker: str,
        language: str,
        instruct: Optional[str] = None,
        non_streaming_mode: Optional[bool] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        backend: str = "auto",
    ):
        """
        Stream custom-voice speech generation, yielding audio chunks.

        Args:
            text: Text to synthesize
            speaker: Speaker name (validated against model's supported speakers)
            language: Target language
            instruct: Optional instruction string
            non_streaming_mode: When None, uses upstream default (True for custom_voice).
                Set False for step-by-step text feeding during decode.
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before EOS is allowed
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample
            repetition_penalty: Repetition penalty
            chunk_size: Codec steps per chunk (12 = ~1 second of audio)
            backend: "auto" (CUDA graphs if available, else dynamic cache),
                     "faster" (requires CUDA + PredictorGraph/TalkerGraph),
                     "dynamic" (always dynamic cache, no graphs).

        Yields:
            Tuple of (audio_chunk_numpy, sample_rate, timing_dict)
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} does not support "
                "generate_custom_voice_streaming. Please check Model Card or Readme."
            )

        self._validate_languages([language])
        self._validate_speakers([speaker])

        if self.model.tts_model_size in "0b6":
            instruct = None

        backend = self._resolve_backend(backend)
        use_cuda_graphs = backend == "faster"

        # Reset streaming state
        self._stream_samples_per_frame = None
        self._stream_prev_audio_len = 0

        m, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text, language=language, speaker=speaker,
            instruct=instruct, non_streaming_mode=non_streaming_mode if non_streaming_mode is not None else False,
        )

        if use_cuda_graphs:
            if not torch.cuda.is_available():
                raise ValueError("backend='faster' requires CUDA. Use backend='auto' or 'dynamic'.")
            self._init_cuda_graphs()
            self._warmup_cuda_graphs(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None
        speech_tokenizer = m.speech_tokenizer

        if use_cuda_graphs:
            stream_fn = self._fast_generate_streaming
            stream_kwargs = dict(
                talker=talker, tie=tie, tam=tam, tth=tth, tpe=tpe, config=config,
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
            )
        else:
            stream_fn = self._parity_generate_streaming
            stream_kwargs = dict(
                talker=talker, tie=tie, tam=tam, tth=tth, tpe=tpe, config=config,
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
            )

        all_codes = []
        for codec_chunk, timing in stream_fn(**stream_kwargs):
            all_codes.append(codec_chunk)
            new_audio, sr = self._decode_chunk_to_audio(
                speech_tokenizer, all_codes, ref_codes=None, context_frames=25,
            )
            yield new_audio, sr, timing

    @torch.inference_mode()
    def generate_voice_clone_streaming(
        self,
        text: str,
        language: str,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        backend: str = "auto",
    ):
        """
        Stream voice-clone speech generation, yielding audio chunks.

        Same as generate_voice_clone() but yields (audio_chunk, sample_rate, timing)
        tuples every chunk_size codec steps (~chunk_size/12 seconds of audio).

        Args:
            text: Text to synthesize
            language: Target language
            ref_audio: Reference audio for voice cloning.
            ref_text: Transcription of reference audio (required in ICL mode).
            x_vector_only_mode: If True, use only speaker embedding.
            voice_clone_prompt: Precomputed prompt from create_voice_clone_prompt().
            non_streaming_mode: When None, uses upstream default (False).
            max_new_tokens, min_new_tokens, temperature, top_k, top_p, do_sample,
                repetition_penalty: Standard generation parameters.
            chunk_size: Codec steps per chunk (12 = ~1 second of audio).
            backend: "auto", "faster", or "dynamic" (see generate_custom_voice_streaming).

        Yields:
            Tuple of (audio_chunk_numpy, sample_rate, timing_dict)
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} does not support "
                "generate_voice_clone_streaming. Please check Model Card or Readme."
            )

        texts = [text]
        languages = [language]
        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode,
            )
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
                ref_texts_for_ids = [it.ref_text for it in voice_clone_prompt]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        input_ids = self._tokenize_texts([self._build_assistant_text(text)])

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for rt in ref_texts_for_ids:
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        # Reset streaming state
        self._stream_samples_per_frame = None
        self._stream_prev_audio_len = 0

        m, tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=self.model,
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            speakers=[None],
            non_streaming_mode=non_streaming_mode or False,
            instruct_ids=[None],
        )

        backend = self._resolve_backend(backend)
        use_cuda_graphs = backend == "faster"

        if use_cuda_graphs:
            if not torch.cuda.is_available():
                raise ValueError("backend='faster' requires CUDA. Use backend='auto' or 'dynamic'.")
            self._init_cuda_graphs()
            self._warmup_cuda_graphs(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None
        speech_tokenizer = m.speech_tokenizer

        # In ICL mode: prepend reference codes before decoding so the codec decoder
        # has acoustic context from the reference audio (matches official implementation).
        ref_codes = None
        using_icl_mode = False
        if voice_clone_prompt_dict is not None:
            using_icl_mode = any(voice_clone_prompt_dict.get("icl_mode", []))
            if using_icl_mode and voice_clone_prompt_dict.get("ref_code") and voice_clone_prompt_dict["ref_code"][0] is not None:
                ref_codes = voice_clone_prompt_dict["ref_code"][0]

        if use_cuda_graphs:
            stream_fn = self._fast_generate_streaming
        else:
            stream_fn = self._parity_generate_streaming

        stream_kwargs = dict(
            talker=talker, tie=tie, tam=tam, tth=tth, tpe=tpe, config=config,
            max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        )

        all_codes = []
        for codec_chunk, timing in stream_fn(**stream_kwargs):
            all_codes.append(codec_chunk)
            new_audio, sr = self._decode_chunk_to_audio(
                speech_tokenizer, all_codes, ref_codes=ref_codes, context_frames=25,
            )
            yield new_audio, sr, timing

    @torch.inference_mode()
    def generate_voice_design_streaming(
        self,
        text: str,
        instruct: str,
        language: Optional[str] = None,
        non_streaming_mode: bool = True,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        chunk_size: int = 12,
        backend: str = "auto",
    ):
        """
        Stream voice-design speech generation, yielding audio chunks.

        Same as generate_voice_design() but yields (audio_chunk, sample_rate, timing)
        tuples every chunk_size codec steps (~chunk_size/12 seconds of audio).

        Args:
            text: Text to synthesize
            instruct: Instruction describing desired voice/style.
            language: Target language.
            non_streaming_mode: When None, uses upstream default (True).
            max_new_tokens, min_new_tokens, temperature, top_k, top_p, do_sample,
                repetition_penalty: Standard generation parameters.
            chunk_size: Codec steps per chunk (12 = ~1 second of audio).
            backend: "auto", "faster", or "dynamic" (see generate_custom_voice_streaming).

        Yields:
            Tuple of (audio_chunk_numpy, sample_rate, timing_dict)
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with tts_model_type={self.model.tts_model_type} does not support "
                "generate_voice_design_streaming. Please check Model Card or Readme."
            )

        self._validate_languages([language] if language is not None else ["Auto"])

        # Reset streaming state
        self._stream_samples_per_frame = None
        self._stream_prev_audio_len = 0

        m, tie, tam, tth, tpe = self._prepare_generation_custom(
            text=text, language=language or "Auto", speaker=None,
            instruct=instruct, non_streaming_mode=non_streaming_mode or True,
        )

        backend = self._resolve_backend(backend)
        use_cuda_graphs = backend == "faster"

        if use_cuda_graphs:
            if not torch.cuda.is_available():
                raise ValueError("backend='faster' requires CUDA. Use backend='auto' or 'dynamic'.")
            self._init_cuda_graphs()
            self._warmup_cuda_graphs(tie.shape[1])

        talker = m.talker
        config = m.config.talker_config
        talker.rope_deltas = None
        speech_tokenizer = m.speech_tokenizer

        if use_cuda_graphs:
            stream_fn = self._fast_generate_streaming
        else:
            stream_fn = self._parity_generate_streaming

        stream_kwargs = dict(
            talker=talker, tie=tie, tam=tam, tth=tth, tpe=tpe, config=config,
            max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            do_sample=do_sample, repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        )

        all_codes = []
        for codec_chunk, timing in stream_fn(**stream_kwargs):
            all_codes.append(codec_chunk)
            new_audio, sr = self._decode_chunk_to_audio(
                speech_tokenizer, all_codes, ref_codes=None, context_frames=25,
            )
            yield new_audio, sr, timing

    def get_supported_speakers(self) -> Optional[List[str]]:
        """
        List supported speaker names for the current model.

        This is a convenience wrapper around `model.get_supported_speakers()`.
        If the underlying model does not expose speaker constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported speaker names (lowercased), if available.
                - None if the model does not provide supported speakers.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)


    def get_supported_languages(self) -> Optional[List[str]]:
        """
        List supported language names for the current model.

        This is a convenience wrapper around `model.get_supported_languages()`.
        If the underlying model does not expose language constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported language names (lowercased), if available.
                - None if the model does not provide supported languages.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
