#!/usr/bin/env python3
"""Download paper-split CTs and masks from their original Hugging Face repos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil

REX_REPO = 'rajpurkarlab/ReXGroundingCT'
CT_REPO = 'ibrahimhamamci/CT-RATE'
REPO_ROOT = Path(__file__).resolve().parents[1]


def ct_remote_path(name: str) -> str:
    match = re.fullmatch(r'(train|valid)_(\d+)_([a-zA-Z0-9]+)_(\d+)\.nii\.gz', name)
    if match is None:
        raise ValueError(f'Unexpected CT-RATE filename: {name!r}')
    split, patient, scan, _ = match.groups()
    # CT-RATE v2 includes corrected HU intensities and spatial metadata.
    return f'dataset/{split}_fixed/{split}_{patient}/{split}_{patient}_{scan}/{name}'


def select_names(splits: dict, selected: list[str], metadata: dict, limit: int | None) -> list[str]:
    available = {
        row['name'] for rows in metadata.values() if isinstance(rows, list)
        for row in rows if isinstance(row, dict) and 'name' in row
    }
    names = set()
    for split in selected:
        entries = splits[split]
        if not isinstance(entries, list) or not all(isinstance(x, str) for x in entries):
            raise ValueError(f'Invalid split: {split}')
        names.update(entries)
    missing = names - available
    if missing:
        raise ValueError(f'{len(missing)} paper-split CTs are absent from dataset.json: {sorted(missing)[:5]}')
    names = sorted(names)
    if limit is not None:
        names = names[:limit]
    if not names:
        raise ValueError('No CTs selected.')
    for name in names:
        ct_remote_path(name)
    return names


def build_plan(names: list[str], root: Path, skip_cts: bool, skip_masks: bool) -> list[dict]:
    plan = []
    for name in names:
        if not skip_cts:
            plan.append({'repo_id': CT_REPO, 'filename': ct_remote_path(name),
                         'destination': str(root / 'cts' / name)})
        if not skip_masks:
            plan.append({'repo_id': REX_REPO, 'filename': f'segmentations/{name}',
                         'destination': str(root / 'segmentations' / name)})
    return plan


def download_one(item: dict, revision: str, root: Path, downloader) -> str:
    dest = Path(item['destination'])
    if dest.is_file() and dest.stat().st_size > 0:
        return 'existing'
    # Hub uses its own incomplete files for interrupted transfers. Flatten only
    # completed downloads into the layout expected by the preprocessing scripts.
    staging = root / '.downloads' / item['repo_id'].replace('/', '--')
    path = Path(downloader(repo_id=item['repo_id'], repo_type='dataset',
        filename=item['filename'], revision=revision, local_dir=str(staging)))
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError('Download produced no data.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, dest)
    return 'downloaded'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_root', type=Path, default=Path('data/ReXGroundingCT'))
    parser.add_argument('--splits_json', type=Path, default=REPO_ROOT/'splits/paper_splits.json')
    parser.add_argument('--split', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    parser.add_argument('--limit', type=int, help='Download only the first N selected CTs for a smoke test.')
    parser.add_argument('--env_file', type=Path, help='Optional local .env containing HF_TOKEN; never commit it.')
    parser.add_argument('--metadata_json', type=Path, help='Use an existing original ReX dataset.json.')
    parser.add_argument('--rex_revision', default='main', help='ReX commit, branch or tag; resolved to a commit for downloads.')
    parser.add_argument('--ct_revision', default='main', help='CT-RATE commit, branch or tag; resolved to a commit for downloads.')
    parser.add_argument('--skip_cts', action='store_true')
    parser.add_argument('--skip_segmentations', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='Plan only; may fetch dataset.json, never CTs or masks.')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.skip_cts and args.skip_segmentations:
        parser.error('Cannot skip both CTs and segmentations')
    if args.env_file:
        if not args.env_file.is_file():
            parser.error('--env_file does not exist')
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    # Import after loading .env so HF_TOKEN is available during Hub initialization.
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    splits = json.loads(args.splits_json.read_text(encoding='utf-8'))
    try:
        api = HfApi()
        revisions = {REX_REPO: args.rex_revision, CT_REPO: args.ct_revision}
        if not args.dry_run or not args.metadata_json:
            revisions[REX_REPO] = api.dataset_info(REX_REPO, revision=args.rex_revision).sha
        if not args.dry_run and not args.skip_cts:
            revisions[CT_REPO] = api.dataset_info(CT_REPO, revision=args.ct_revision).sha
        metadata_path = args.metadata_json
        if metadata_path is None:
            metadata_path = Path(hf_hub_download(repo_id=REX_REPO, repo_type='dataset',
                filename='dataset.json', revision=revisions[REX_REPO],
                local_dir=str(root/'.downloads'/'metadata')))
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        names = select_names(splits, args.split, metadata, args.limit)
        plan = build_plan(names, root, args.skip_cts, args.skip_segmentations)
        print(f'Selected {len(names)} CTs, {len(plan)} volume/mask files. Existing nonempty files are reused.')
        print('CT downloads can require substantial disk space; this is not the complete CT-RATE dataset.')
        record = {'selected_splits':args.split, 'limit':args.limit, 'revisions':revisions, 'files':plan}
        if args.dry_run:
            record_path = root/'download_plan.json'
            record_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
            print(f'Dry run: no volumes downloaded. Plan: {record_path}')
            return
        target_metadata = root/'dataset.json'
        if metadata_path.resolve() != target_metadata.resolve():
            if target_metadata.exists() and target_metadata.read_bytes() != metadata_path.read_bytes():
                raise ValueError('Existing dataset.json differs; use another --out_root or matching --metadata_json.')
            shutil.copy2(metadata_path, target_metadata)
        record_path = root/'download_manifest.json'
        if record_path.exists():
            previous = json.loads(record_path.read_text(encoding='utf-8'))
            previous_files = {item['destination']: item for item in previous['files']}
            for item in plan:
                old = previous_files.get(item['destination'])
                if old and (old['repo_id'], old['filename']) != (item['repo_id'], item['filename']):
                    raise ValueError('Download source changed (for example CT-RATE v1 to v2); '
                                     'use another --out_root to avoid reusing uncorrected volumes.')
            for repo in {item['repo_id'] for item in plan}:
                if previous['revisions'][repo] != revisions[repo]:
                    raise ValueError('Dataset revision changed; pass the recorded revision or use another --out_root.')
        record_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        for i, item in enumerate(plan, 1):
            state = download_one(item, revisions[item['repo_id']], root, hf_hub_download)
            print(f'[{i}/{len(plan)}] {Path(item["destination"]).name}: {state}')
        print(f'Done. CTs: {root/"cts"}; masks: {root/"segmentations"}; metadata: {target_metadata}')
        print('Next: generate chest bounding boxes and projections as described in data/README.md.')
    except GatedRepoError:
        parser.exit(1, 'Access denied. Accept the terms on BOTH dataset pages, then run hf auth login '
            'with a read-access token, or set HF_TOKEN.\n')
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        parser.exit(1, f'Hub request failed (HTTP {status}). Check access to both repositories, '
            'your connection, and the requested revisions. No token is printed.\n')


if __name__ == '__main__':
    main()
