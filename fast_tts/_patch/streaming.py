"""Streaming + CUDA-graph methods for ``Qwen3TTSModel``.

Methods the stock PyPI ``qwen-tts`` wheel does not ship: streaming
generation, CUDA-graph acceleration (predictor/talker capture) and token
sampling helpers. They are attached to the stock ``qwen_tts.Qwen3TTSModel``
at import time by :func:`fast_tts._patch.apply_patch`.

Provenance: the CUDA-graph capture strategy follows the approach published in
faster-qwen3-tts (MIT) and is implemented here independently; the talker input
construction follows the official Qwen3-TTS algorithm (Apache-2.0),
restructured with hoisted constants and per-sample helper dispatch.

The lazy ``from .predictor_graph / .talker_graph / .sampling`` imports inside
the methods resolve against this package's own copies of those modules.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def _speaker_embed_for(m, speaker, clone_embeds, index):
    """Resolve the per-sample speaker embedding (explicit dispatch)."""
    if clone_embeds is None:
        if not speaker:
            return None
        spk_table = m.config.talker_config.spk_id
        key = speaker.lower()
        if key not in spk_table:
            raise NotImplementedError(f"Speaker {speaker} not implemented")
        spk_id = torch.tensor(spk_table[key], device=m.talker.device, dtype=torch.long)
        return m.talker.get_input_embeddings()(spk_id)
    if clone_embeds["x_vector_only_mode"][index] or clone_embeds["icl_mode"][index]:
        return clone_embeds[index]
    return None


def _language_code_for(talker_cfg, language):
    """Map a language name to its codec id; 'auto' yields None (nothink path)."""
    key = language.lower()
    if key == "auto":
        return None
    table = talker_cfg.codec_language_id
    if key not in table:
        raise NotImplementedError(f"Language {language} not implemented")
    return table[key]


def _dialect_code_for(talker_cfg, speaker):
    """Dialect codec language forced by a speaker (None for non-dialect speakers)."""
    if not speaker:
        return None
    return talker_cfg.spk_is_dialect.get(speaker.lower())


def _codec_prefill_ids(talker_cfg, language_id):
    """Codec-side prefill token ids for one sample."""
    if language_id is None:
        return [talker_cfg.codec_nothink_id, talker_cfg.codec_think_bos_id, talker_cfg.codec_think_eos_id]
    return [talker_cfg.codec_think_id, talker_cfg.codec_think_bos_id, language_id, talker_cfg.codec_think_eos_id]


class Qwen3TTSStreamingMixin:
    """Streaming and CUDA-graph methods attached onto Qwen3TTSModel at import time.

    See the module docstring for provenance of each block.
    """

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
        # One shared memory pool for both graphs: consolidated VRAM reservation.
        self._graph_pool = torch.cuda.graphs.graph_pool_handle()
        self._predictor_graph.capture(num_warmup=3, pool=self._graph_pool)
        self._talker_graph.capture(prefill_len=prefill_len, num_warmup=3, pool=self._graph_pool)
        self._graphs_warmed_up = True

    def _prepare_generation_custom(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
    ):
        """Prepare generation inputs for the custom-voice model."""
        input_ids = self._tokenize_texts([self._build_assistant_text(text)])

        if not instruct:
            instruct_ids = [None]
        else:
            instruct_ids = [self._tokenize_texts([self._build_instruct_text(instruct)])[0]]

        m = self.model
        tie, tam, tth, tpe = self._build_talker_inputs_local(
            m=m,
            input_ids=input_ids,
            ref_ids=[None],
            voice_clone_prompt=None,
            languages=[language if language is not None else "Auto"],
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
        """Assemble talker input embeddings for streaming generation.

        Builds the per-sample embedding sequence (instruct prefix, role
        tokens, codec prefill block, text tail) and right-pads the batch by
        flipping, zero-padding and flipping back; returns
        (embeds [B, L, H], attention_mask, trailing_text_hiddens, tts_pad_embed).

        The algorithm follows the official Qwen3-TTS talker input
        construction (Apache-2.0), restructured here with per-sample helper
        dispatch and constants hoisted out of the batch loop.
        """
        n = len(input_ids)
        if speakers is None:
            speakers = [None] * n

        clone_embeds = m.generate_speaker_prompt(voice_clone_prompt) if voice_clone_prompt is not None else None

        # Constants shared by every sample — computed once (the upstream
        # version recomputed these per sample).
        tts_ids = torch.tensor(
            [[m.config.tts_bos_token_id, m.config.tts_eos_token_id, m.config.tts_pad_token_id]],
            device=m.talker.device, dtype=torch.long,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            m.talker.text_projection(m.talker.get_text_embeddings()(tts_ids)).chunk(3, dim=1)
        )
        codec_tail_embed = m.talker.get_input_embeddings()(
            torch.tensor(
                [[m.config.talker_config.codec_pad_id, m.config.talker_config.codec_bos_id]],
                device=m.talker.device, dtype=torch.long,
            )
        )

        talker_cfg = m.config.talker_config
        sample_parts = []      # per-sample embedding segments (instruct prefix + body)
        trailing_hiddens = []

        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            spk_embed = _speaker_embed_for(m, speaker, clone_embeds, index)

            assert language is not None
            lang_key = language.lower()
            language_id = _language_code_for(talker_cfg, language)
            if lang_key in ("chinese", "auto") and speaker:
                dialect = _dialect_code_for(talker_cfg, speaker)
                if dialect:
                    language_id = talker_cfg.codec_language_id[dialect]

            codec_prefix_embeds = m.talker.get_input_embeddings()(
                torch.tensor([_codec_prefill_ids(talker_cfg, language_id)], device=m.talker.device, dtype=torch.long)
            )
            if spk_embed is None:
                codec_block = torch.cat([codec_prefix_embeds, codec_tail_embed], dim=1)
            else:
                codec_block = torch.cat([codec_prefix_embeds, spk_embed.view(1, 1, -1), codec_tail_embed], dim=1)

            role_embeds = m.talker.text_projection(m.talker.get_text_embeddings()(input_id[:, :3]))
            body = (
                torch.cat((tts_pad_embed.expand(-1, codec_block.shape[1] - 2, -1), tts_bos_embed), dim=1)
                + codec_block[:, :-1]
            )
            sample_embeds = torch.cat([role_embeds, body], dim=1)

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
                sample_embeds = torch.cat([sample_embeds, icl_input_embed], dim=1)
            else:
                tail_text = m.talker.text_projection(m.talker.get_text_embeddings()(input_id[:, 3:4]))
                sample_embeds = torch.cat([sample_embeds, tail_text + codec_block[:, -1:]], dim=1)

                if non_streaming_mode:
                    text_tail = torch.cat(
                        (m.talker.text_projection(m.talker.get_text_embeddings()(input_id[:, 3:-5])), tts_eos_embed),
                        dim=1,
                    )
                    pad_block = m.talker.get_input_embeddings()(
                        torch.tensor(
                            [[talker_cfg.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                            device=m.talker.device, dtype=torch.long,
                        )
                    )
                    bos_block = tts_pad_embed + m.talker.get_input_embeddings()(
                        torch.tensor([[talker_cfg.codec_bos_id]], device=m.talker.device, dtype=torch.long)
                    )
                    sample_embeds = torch.cat([sample_embeds[:, :-1], text_tail + pad_block, bos_block], dim=1)
                    trailing_text_hidden = tts_pad_embed
                else:
                    trailing_text_hidden = torch.cat(
                        (m.talker.text_projection(m.talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed),
                        dim=1,
                    )

            parts = []
            if instruct_ids is not None and instruct_ids[index] is not None:
                parts.append(m.talker.text_projection(m.talker.get_text_embeddings()(instruct_ids[index])))
            parts.append(sample_embeds)
            sample_parts.append(parts)
            trailing_hiddens.append(trailing_text_hidden)

        full_embeds = [torch.cat(parts, dim=1) for parts in sample_parts]

        # Right-pad the batch: flip each sequence, zero-pad to equal length, flip back.
        lengths = torch.tensor([e.shape[1] for e in full_embeds])
        flipped = [e.squeeze(0).flip(0) for e in full_embeds]
        padded = torch.nn.utils.rnn.pad_sequence(flipped, batch_first=True, padding_value=0.0).flip(1)

        bsz, max_len = padded.shape[0], padded.shape[1]
        pos_grid = torch.arange(max_len).expand(bsz, -1)
        talker_attention_mask = (pos_grid >= (max_len - lengths).unsqueeze(1)).long().to(padded.device)

        tail_seqs = [t.squeeze(0) for t in trailing_hiddens]
        padded_tails = torch.nn.utils.rnn.pad_sequence(tail_seqs, batch_first=True, padding_value=0.0)
        tail_grid = torch.arange(padded_tails.shape[1], device=padded_tails.device).expand(len(tail_seqs), -1)
        pad_where = tail_grid >= torch.tensor([s.shape[0] for s in tail_seqs], device=padded_tails.device).unsqueeze(1)
        padded_tails[pad_where] = tts_pad_embed.squeeze()

        return padded, talker_attention_mask, padded_tails, tts_pad_embed

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

