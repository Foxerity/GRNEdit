import copy
import glob
import json
import os.path as osp
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import tqdm

from grn.dataset.official_t2iv.dataset_joint_vi import JointViDataset
from grn.official_edit_stage15.text_conditioning import add_stage15_t2v_prefix, stage15_combined_text_len

def cache_scope_from_meta_file(meta_file: str) -> str:
    stem = osp.splitext(osp.basename(str(meta_file or 'unknown_meta')))[0] or 'unknown_meta'
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in stem)


def list_official_edit_jsonls(meta_folder: str) -> List[str]:
    direct = glob.glob(osp.join(meta_folder, '*.jsonl'))
    nested = glob.glob(osp.join(meta_folder, '*', '*.jsonl'))
    return sorted(set(direct + nested))


class EditPairJointViDataset(JointViDataset):
    """Paired source-target video editing dataset."""

    def get_video_caption(self, meta, mapped_duration):
        caption_type = str(getattr(self, 'video_caption_type', '') or '').strip()
        if caption_type:
            if caption_type == 'caption':
                caption_values = meta.get('caption')
                if not isinstance(caption_values, list) or not caption_values:
                    raise ValueError('video_caption_type=caption requires non-empty meta["caption"].')
                first_caption = caption_values[0]
                if isinstance(first_caption, dict):
                    caption = str(first_caption.get('content') or '').strip()
                else:
                    caption = str(first_caption or '').strip()
            else:
                caption = str(meta.get(caption_type) or '').strip()
                if not caption:
                    raise ValueError(f'video_caption_type={caption_type!r} requires non-empty meta[{caption_type!r}].')
        else:
            caption = super().get_video_caption(meta, mapped_duration)

        if caption_type:
            if self.enable_dynamic_length_prompt and (self.epoch_rank_generator.random() < self.other_args.short_cap_prob):
                caption = self.random_drop_sentences(caption, min_sentences=2)
            if 'quality_prompt' in meta:
                caption = caption + ' ' + meta['quality_prompt']
            if meta['first_frame_condition']:
                caption = '<I2V>' + caption
        assert caption
        return add_stage15_t2v_prefix(caption)

    @staticmethod
    def _reprompt_from_meta(meta: Dict[str, Any]) -> str:
        reprompt = meta.get('reprompt')
        if isinstance(reprompt, str):
            reprompt = reprompt.strip()
        else:
            reprompt = ''
        if reprompt.lower() in {'null', 'none', 'n/a', 'na'}:
            reprompt = ''
        return reprompt

    def append_text_tokens(self, metas, skip_count_text_token=False, bucket_size=100):
        t1 = time.time()
        pbar = tqdm.tqdm(total=len(metas) // bucket_size + 1, desc='append text tokens')
        valid_metas = []
        use_reprompt_text = bool(getattr(self.other_args, 'stage15_use_reprompt_text', 0))
        for bucket_id in range(len(metas) // bucket_size + 1):
            pbar.update(1)
            start = bucket_id * bucket_size
            end = min(start + bucket_size, len(metas))
            if start >= end:
                break
            captions = []
            reprompt_captions = []
            caps_per_meta = []
            for i in range(start, end):
                if use_reprompt_text:
                    captions.extend(metas[i]['caption'])
                    reprompt_captions.extend([metas[i]['reprompt'] for _ in metas[i]['caption']])
                else:
                    captions.extend(metas[i]['caption'])
                caps_per_meta.append(len(metas[i]['caption']))
            assert len(captions), f'{len(captions)=}'
            t5_cap = int(getattr(self.other_args, 't5_max_tokens', self.max_text_len) or self.max_text_len)
            if skip_count_text_token:
                lens = [0 for _ in range(len(captions))]
                reprompt_lens = [0 for _ in range(len(captions))]
            else:
                if use_reprompt_text:
                    all_texts = captions + reprompt_captions
                    all_lens = self.get_captions_lens(all_texts)
                    lens = all_lens[:len(captions)]
                    reprompt_lens = all_lens[len(captions):]
                else:
                    lens = self.get_captions_lens(captions)
                    reprompt_lens = [0 for _ in range(len(captions))]
                lens = np.clip(np.array(lens), a_min=0, a_max=t5_cap)
                reprompt_lens = np.clip(np.array(reprompt_lens), a_min=0, a_max=t5_cap)
            ptr = 0
            for i in range(start, end):
                caption_count = caps_per_meta[i - start]
                if use_reprompt_text:
                    text_tokens = sum(
                        stage15_combined_text_len(lens[ptr + j], reprompt_lens[ptr + j], t5_cap)
                        for j in range(caption_count)
                    )
                else:
                    text_tokens = sum(lens[ptr:ptr + caption_count])
                ptr += caption_count
                metas[i]['text_tokens'] = int(text_tokens)
                metas[i]['cum_text_visual_tokens'] = metas[i]['cum_text_visual_tokens'] + metas[i]['text_tokens']
                metas[i]['text_visual_tokens'] = metas[i]['cum_text_visual_tokens'][-1]
                if metas[i]['text_visual_tokens'] <= self.train_max_token_len * self.other_args.dense_ratio4seqpack:
                    valid_metas.append(metas[i])
        t2 = time.time()
        print(f'append text tokens: {t2 - t1:.1f}s')
        return valid_metas

    def _target_path(self, meta: Dict[str, Any]) -> str:
        video_path = meta.get('video_path')
        gt_path = meta.get('gt_path')
        if not isinstance(video_path, str) or not video_path:
            video_path = gt_path
        if not isinstance(video_path, str) or not video_path:
            raise ValueError('edit pair meta requires video_path or gt_path for target video.')
        if isinstance(gt_path, str) and gt_path and osp.abspath(gt_path) != osp.abspath(video_path):
            raise ValueError(
                'edit pair meta has conflicting target paths: '
                f'video_path={video_path}, gt_path={gt_path}'
            )
        return video_path

    def _source_path(self, meta: Dict[str, Any]) -> str:
        source_path = meta.get('source_path')
        if not isinstance(source_path, str) or not source_path:
            raise ValueError('edit pair official jsonl requires source_path.')
        return source_path

    def _count_jsonl_lines(self, part_filepaths):
        line_counts = []
        total_rows = 0
        for file in part_filepaths:
            with open(file, encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            line_counts.append((file, line_count, total_rows))
            total_rows += line_count
        return line_counts, total_rows

    def _build_row_level_entries_for_folder(self, meta_folder, meta_folder_repeat, meta_folder_identifier):
        meta_folder_repeat = float(meta_folder_repeat)
        if meta_folder_repeat <= 0:
            raise ValueError(f'meta_folder_repeat must be > 0, got {meta_folder_repeat} for {meta_folder}')

        part_filepaths = list_official_edit_jsonls(meta_folder)
        if not part_filepaths:
            raise FileNotFoundError(f'official edit meta folder has no jsonl files: {meta_folder}')
        total_part_files = len(part_filepaths)
        self.epoch_generator.shuffle(part_filepaths)
        line_counts, total_rows = self._count_jsonl_lines(part_filepaths)

        full_repeats = int(np.floor(meta_folder_repeat))
        fractional_repeat = meta_folder_repeat - full_repeats
        selected_arrays = []

        if total_rows > 0 and full_repeats > 0:
            row_ids = np.arange(total_rows, dtype=np.int64)
            if full_repeats == 1:
                selected_arrays.append(row_ids.copy())
            else:
                selected_arrays.append(np.tile(row_ids, full_repeats))

        fractional_rows = 0
        if total_rows > 0 and fractional_repeat > 0:
            fractional_rows = int(total_rows * fractional_repeat)
            if fractional_rows > 0:
                row_ids = np.arange(total_rows, dtype=np.int64)
                self.epoch_generator.shuffle(row_ids)
                selected_arrays.append(row_ids[:fractional_rows])

        if selected_arrays:
            selected_rows = selected_arrays[0] if len(selected_arrays) == 1 else np.concatenate(selected_arrays)
            self.epoch_generator.shuffle(selected_rows)
            rank_rows = selected_rows[self.rank::self.num_replicas]
        else:
            rank_rows = np.array([], dtype=np.int64)

        entries = []
        for file, line_count, start_offset in line_counts:
            if line_count <= 0:
                continue
            end_offset = start_offset + line_count
            mask = (rank_rows >= start_offset) & (rank_rows < end_offset)
            local_line_indices = rank_rows[mask] - start_offset
            if local_line_indices.size:
                unique_lines, repeat_counts = np.unique(local_line_indices, return_counts=True)
                entries.append(
                    {
                        'path': file,
                        'identifier': meta_folder_identifier,
                        'selected_line_repeats': {
                            int(line): int(count)
                            for line, count in zip(unique_lines.tolist(), repeat_counts.tolist())
                        },
                    }
                )

        logical_rows = int(total_rows * full_repeats + fractional_rows)
        folder_summary = (
            f'{osp.basename(osp.normpath(meta_folder))}: repeat={meta_folder_repeat:g}, '
            f'jsonl={total_part_files}->{len(entries)}, rows={total_rows}->{logical_rows}, '
            f'rank_rows={len(rank_rows)}'
        )
        return entries, folder_summary

    def get_mapped_duration2metas(self):
        part_entries = []
        folder_summaries = []
        for meta_folder, meta_folder_repeat, meta_folder_identifier in zip(
            self.meta_folders,
            self.meta_folder_repeats,
            self.meta_folder_identifiers,
        ):
            folder_entries, folder_summary = self._build_row_level_entries_for_folder(
                meta_folder,
                meta_folder_repeat,
                meta_folder_identifier,
            )
            part_entries.extend(folder_entries)
            folder_summaries.append(folder_summary)
        self.epoch_generator.shuffle(part_entries)
        self.print(f'{self.rank=} edit-pair folder selection: {folder_summaries}')
        self.print(f'{self.rank=} edit-pair jsonls sample: {[entry["path"] for entry in part_entries[:4]]}')

        mapped_duration2metas = {}
        pbar = tqdm.tqdm(total=len(part_entries))
        total, selected_total, corrupt, missing_reprompt = 0, 0, 0, 0
        stop_read = False
        rough_h_div_w = self.h_div_w_templates[np.argmin(np.abs((9 / 16 - self.h_div_w_templates)))]
        use_reprompt_text = bool(getattr(self.other_args, 'stage15_use_reprompt_text', 0))
        for part_entry in part_entries:
            part_filepath = part_entry['path']
            file_quality_prompt = part_entry['identifier']
            selected_line_repeats = part_entry['selected_line_repeats']
            if stop_read:
                break
            pbar.update(1)
            try:
                f = open(part_filepath, encoding='utf-8')
            except Exception as e:
                print(f'{part_filepath=} Error: {e}')
                continue
            with f:
                line_iter = enumerate(f)
                for zero_based_line_no, line in line_iter:
                    line_no = zero_based_line_no + 1
                    total += 1
                    repeat_count = selected_line_repeats.get(zero_based_line_no, 0)
                    if repeat_count <= 0:
                        continue
                    for _ in range(repeat_count):
                        selected_total += 1
                        try:
                            meta = json.loads(line)
                            if file_quality_prompt:
                                meta['quality_prompt'] = file_quality_prompt
                            target_path = self._target_path(meta)
                            source_path = self._source_path(meta)
                            meta['video_path'] = target_path
                            meta['gt_path'] = target_path
                            meta['source_path'] = source_path
                            meta['meta_file'] = part_filepath
                            meta['meta_line_no'] = line_no
                            if use_reprompt_text:
                                reprompt = self._reprompt_from_meta(meta)
                                if not reprompt:
                                    missing_reprompt += 1
                                    if missing_reprompt <= 10:
                                        print(
                                            '[official edit reprompt warning] skip row without reprompt: '
                                            f'{part_filepath}:{line_no}',
                                            flush=True,
                                        )
                                    continue
                                meta['reprompt'] = reprompt
                        except Exception as e:
                            corrupt += 1
                            print(e, corrupt, total, corrupt / max(total, 1))
                            continue

                        if ('height' in meta) and ('width' in meta):
                            cur_h_div_w_template = self.h_div_w_templates[
                                np.argmin(np.abs((meta['height'] / meta['width'] - self.h_div_w_templates)))
                            ]
                        else:
                            cur_h_div_w_template = rough_h_div_w
                        if 'h_div_w' in meta:
                            del meta['h_div_w']
                        meta['first_frame_condition'] = False
                        meta['pn'] = self.sample_pn(meta, self.pn_list, self.pn_probs)

                        if self.epoch_rank_generator.random() < self.other_args.i2v_ratio:
                            meta['first_frame_condition'] = True
                        begin_frame_id, end_frame_id, fps = meta['begin_frame_id'], meta['end_frame_id'], meta['fps']
                        real_duration = (end_frame_id - begin_frame_id) / fps
                        mapped_duration = int(np.round(real_duration / self.duration_resolution)) * self.duration_resolution
                        if mapped_duration < self.min_training_duration:
                            continue
                        if mapped_duration > self.max_training_duration:
                            if self.drop_long_video:
                                continue
                            mapped_duration = self.max_training_duration
                        if self.other_args.use_clipwise_caption:
                            meta['caption'] = [
                                meta['caption-InternVL2.0'],
                                self.get_video_caption(meta, mapped_duration),
                            ]
                        else:
                            meta['caption'] = [self.get_video_caption(meta, mapped_duration)]
                        sample_frames = int(mapped_duration * self.video_fps + 1)
                        pt = (sample_frames - 1) // self.temporal_compress_rate + 1
                        scale_schedule = self.dynamic_resolution_h_w[cur_h_div_w_template][meta['pn']]['pt2scale_schedule'][pt]
                        meta['sample_frames'] = sample_frames

                        if mapped_duration not in mapped_duration2metas:
                            mapped_duration2metas[mapped_duration] = []

                        cum_visual_tokens = []
                        preserve_scale_inds = {}
                        assert len(scale_schedule) == len(self.other_args.video_scale_probs), (
                            f'{len(scale_schedule)=} {len(self.other_args.video_scale_probs)=}'
                        )
                        for scale_ind, scale in enumerate(scale_schedule):
                            if self.epoch_rank_generator.random() < self.other_args.video_scale_probs[scale_ind]:
                                preserve_scale_inds[scale_ind] = True
                                tokens_this_scale = np.array(scale).prod(-1) + self.other_args.add_scale_token
                                cum_visual_tokens.append(tokens_this_scale)
                        cum_visual_tokens = np.array(cum_visual_tokens).cumsum()
                        meta['cum_text_visual_tokens'] = cum_visual_tokens
                        meta['preserve_scale_inds'] = preserve_scale_inds

                        if self.other_args.cache_check_mode == 1:
                            if self.exists_cache_file(meta):
                                mapped_duration2metas[mapped_duration].append(meta)
                        elif self.other_args.cache_check_mode == -1:
                            if not self.exists_cache_file(meta):
                                mapped_duration2metas[mapped_duration].append(meta)
                        else:
                            mapped_duration2metas[mapped_duration].append(meta)

                        total_metas = sum([len(item) for item in mapped_duration2metas.values()])
                        if (self.other_args.restrict_data_size > 0) and (
                            total_metas > self.other_args.restrict_data_size / self.num_replicas
                        ):
                            stop_read = True
                            break
                    if stop_read:
                        break

        mapped_duration2freqs = {}
        for mapped_duration in sorted(mapped_duration2metas.keys()):
            mapped_duration2freqs[mapped_duration] = len(mapped_duration2metas[mapped_duration])

        for mapped_duration in list(mapped_duration2metas.keys()):
            freqs = mapped_duration2freqs[mapped_duration]
            assert len(mapped_duration2metas[mapped_duration]) >= freqs
            self.epoch_rank_generator.shuffle(mapped_duration2metas[mapped_duration])
            mapped_duration2metas[mapped_duration] = mapped_duration2metas[mapped_duration][:freqs]
            skip_count_text_token = self.other_args.skip_count_text_token or self.other_args.add_class_token > 0
            mapped_duration2metas[mapped_duration] = self.append_text_tokens(
                mapped_duration2metas[mapped_duration],
                skip_count_text_token=skip_count_text_token,
            )
            if not mapped_duration2metas[mapped_duration]:
                del mapped_duration2metas[mapped_duration]

        total_metas = sum([len(item) for item in mapped_duration2metas.values()])
        if total_metas <= 0:
            raise ValueError(
                'No valid official edit-pair metas after filtering. '
                f'scanned_rows={total}, selected_rows={selected_total}, corrupt={corrupt}, '
                f'missing_reprompt={missing_reprompt}, '
                f'train_max_token_len={self.train_max_token_len}, '
                f'dense_ratio4seqpack={getattr(self.other_args, "dense_ratio4seqpack", None)}, '
                f'video_frames={self.other_args.video_frames}, video_fps={self.video_fps}. '
                'Increase TRAIN_MAX_TOKEN_LEN/DENSE_RATIO4SEQPACK or check edit-pair metadata duration/resolution.'
            )
        for mapped_duration in sorted(mapped_duration2metas.keys()):
            freq = len(mapped_duration2metas[mapped_duration])
            mapped_duration2freqs[mapped_duration] = freq
            proportion = freq / total_metas * 100
            print(f'{mapped_duration=}, {freq=}, {proportion=:.1f}%')
        return mapped_duration2metas, mapped_duration2freqs

    def _role_meta(self, meta: Dict[str, Any], role: str) -> Dict[str, Any]:
        item = copy.deepcopy(meta)
        if role == 'source':
            item['video_path'] = item['source_path']
        elif role == 'target':
            item['video_path'] = item['gt_path']
        else:
            raise ValueError(f'unknown edit pair role: {role}')
        item['edit_pair_role'] = role
        return item

    def _cache_exists_for_meta(self, meta: Dict[str, Any]) -> bool:
        cache_file = self.get_video_cache_file(
            meta['video_path'],
            meta['begin_frame_id'],
            meta['end_frame_id'],
            self.video_fps,
            meta['pn'],
        )
        return osp.exists(cache_file)

    def exists_cache_file(self, meta):
        return self._cache_exists_for_meta(self._role_meta(meta, 'target')) and self._cache_exists_for_meta(
            self._role_meta(meta, 'source')
        )

    def _try_prepare_edit_pair(self, meta: Dict[str, Any]) -> Tuple[bool, Any, Any, bool, bool]:
        target_flag, target_item = self.prepare_video_input(self._role_meta(meta, 'target'))
        source_flag, source_item = self.prepare_video_input(self._role_meta(meta, 'source'))
        return bool(target_flag and source_flag), target_item, source_item, bool(target_flag), bool(source_flag)

    def _find_valid_edit_pair(self, mapped_duration, example_ind: int):
        mapped_duration_metas = self.mapped_duration2metas[mapped_duration]
        start_example_ind = int(example_ind) % len(mapped_duration_metas)
        original_meta = mapped_duration_metas[start_example_ind]
        max_replacement_tokens = int(original_meta.get('text_visual_tokens', self.train_max_token_len))
        skipped_examples = []

        def scan_bucket(candidate_duration, start_ind: int):
            candidate_metas = self.mapped_duration2metas[candidate_duration]
            for offset in range(len(candidate_metas)):
                candidate_ind = (start_ind + offset) % len(candidate_metas)
                candidate_meta = candidate_metas[candidate_ind]
                candidate_tokens = int(candidate_meta.get('text_visual_tokens', self.train_max_token_len + 1))
                if candidate_tokens > max_replacement_tokens:
                    continue
                ok, target_item, source_item, target_flag, source_flag = self._try_prepare_edit_pair(candidate_meta)
                if ok:
                    return candidate_duration, candidate_meta, target_item, source_item
                skipped_examples.append(
                    f'duration={candidate_duration} line={candidate_meta.get("meta_line_no")} '
                    f'target_flag={target_flag} source_flag={source_flag}'
                )
                if len(skipped_examples) <= 3:
                    self.print(
                        'Skip dirty official edit pair: '
                        f'mapped_duration={candidate_duration}, meta_file={candidate_meta.get("meta_file")}, '
                        f'meta_line_no={candidate_meta.get("meta_line_no")}, '
                        f'target={candidate_meta.get("gt_path")}, source={candidate_meta.get("source_path")}, '
                        f'target_flag={target_flag}, source_flag={source_flag}'
                    )
            return None

        found = scan_bucket(mapped_duration, start_example_ind)
        if found is not None:
            return found[1:]

        for fallback_duration in sorted(self.mapped_duration2metas.keys()):
            if fallback_duration == mapped_duration:
                continue
            fallback_metas = self.mapped_duration2metas[fallback_duration]
            if not fallback_metas:
                continue
            found = scan_bucket(
                fallback_duration,
                int(example_ind) % len(fallback_metas),
            )
            if found is not None:
                fallback_duration, meta, target_item, source_item = found
                self.print(
                    'Use replacement official edit pair from another duration bucket: '
                    f'from_duration={mapped_duration}, to_duration={fallback_duration}, '
                    f'original_line={original_meta.get("meta_line_no")}, replacement_line={meta.get("meta_line_no")}'
                )
                return meta, target_item, source_item

        self.print(
            'Drop dirty official edit pair from this batch: '
            f'mapped_duration={mapped_duration}, example_ind={example_ind}, '
            f'max_replacement_tokens={max_replacement_tokens}, first_failures={skipped_examples[:3]}'
        )
        return None

    def __getitem__(self, batch_ind_ptr):
        for batch_offset in range(len(self.batches)):
            batch_info = self.batches[(batch_ind_ptr + batch_offset) % len(self.batches)]
            source_items, target_items, meta_list = [], [], []
            for mapped_duration, example_ind in batch_info:
                found = self._find_valid_edit_pair(mapped_duration, example_ind)
                if found is None:
                    continue
                meta, target_item, source_item = found
                if (
                    target_item['raw_features_cthw'] is not None
                    and source_item['raw_features_cthw'] is not None
                    and tuple(target_item['raw_features_cthw'].shape)
                    != tuple(source_item['raw_features_cthw'].shape)
                ):
                    raise RuntimeError(
                        'source/target cached feature shape mismatch: '
                        f'target={tuple(target_item["raw_features_cthw"].shape)}, '
                        f'source={tuple(source_item["raw_features_cthw"].shape)}'
                    )
                if (
                    target_item['img_T3HW'] is not None
                    and source_item['img_T3HW'] is not None
                    and tuple(target_item['img_T3HW'].shape) != tuple(source_item['img_T3HW'].shape)
                ):
                    raise RuntimeError(
                        'source/target decoded image shape mismatch: '
                        f'target={tuple(target_item["img_T3HW"].shape)}, '
                        f'source={tuple(source_item["img_T3HW"].shape)}'
                    )
                source_items.append(source_item)
                target_items.append(target_item)
                meta_list.append(meta)

            if not source_items:
                self.print(
                    'Skip empty dirty official edit batch: '
                    f'batch_ind_ptr={batch_ind_ptr}, batch_offset={batch_offset}, batch_size={len(batch_info)}'
                )
                continue

            captions = [item['text_input'] for item in target_items]
            reprompts = [
                [self._reprompt_from_meta(meta)] * len(caption_group)
                for meta, caption_group in zip(meta_list, captions)
            ]
            data = {
                'captions': captions,
                'reprompts': reprompts,
                'source_items': source_items,
                'target_items': target_items,
                'text_cache_scopes': [cache_scope_from_meta_file(meta.get('meta_file', '')) for meta in meta_list],
                'meta_list': meta_list,
                'media': 'edit_videos',
            }
            return data

        raise RuntimeError(
            'failed to prepare any non-empty official edit-pair batch after scanning all batches; '
            f'total_batches={len(self.batches)}'
        )
