# Paste this entire file into one Kaggle Python cell. Set T4 and Internet On.
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
COMMIT = '1b8acb6ebae7d7e4ecb962fbaeee48c5e7895676'
FILES = {84: ('best_metric_model.pth', 'c0626e82080a204d851c0d5893eab173ad9bf0ae13376fcd1bf31813554f99c3'),
         100: ('checkpoint_epoch_100.pth', '6a14beb942ecea8f1e873f73be725ff68e75014bd1bbf286af62640a847cefbe')}
REFERENCE = {84: [(0.7250, 1989), (0.0000, 4541)], 100: [(0.5034, 6304), (0.0926, 26724)]}
FULL_REFERENCE = {84: (0.6858, 1030397), 100: (0.4382, 2502962)}
PACKAGES = {'torch': '2.11.0+cu130', 'monai': '1.6.1', 'nibabel': '5.4.2', 'numpy': '2.1.3',
            'scipy': '1.16.3', 'pandas': '2.3.3', 'scikit-learn': '1.6.1', 'PyYAML': '6.0.3',
            'connected-components-3d': '4.1.0'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_checkpoint(name, expected_hash):
    matches = sorted(INPUT.rglob(name))
    for path in matches:
        if path.is_file() and sha256(path) == expected_hash:
            return path
    raise RuntimeError(f'Cannot find the exact D1 file {name}; filename candidates: {matches}')


def find_dataset(extract_path):
    def roots(parent):
        return sorted({p.parent for p in parent.rglob('sub-strokecase0001')
                       if (p.parent / 'sub-strokecase0250').is_dir() and (p.parent / 'derivatives').is_dir()})

    candidates = roots(INPUT)
    if not candidates:
        archives = sorted(p for p in INPUT.rglob('ISLES-2022-250cases.*') if p.is_file() and zipfile.is_zipfile(p))
        require(len(archives) == 1, f'Expected one ISLES-2022-250cases ZIP/SIP; found {archives}')
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(extract_path)
        candidates = roots(extract_path)
    require(len(candidates) == 1, f'Expected one extracted 250-case tree; found {candidates}')
    return candidates[0]


def verify_scores(epoch, score, positives, expected, label):
    ref_score, ref_pos = expected
    dice_ok = abs(score - ref_score) <= 0.005
    volume_ok = abs(positives - ref_pos) <= max(5, ref_pos * 0.02)
    require(dice_ok and volume_ok, f'{label}, ep{epoch}: Dice={score:.5f}, positive={positives}; '
            f'D1 reference Dice={ref_score:.4f}, positive={ref_pos}')


def main():
    require(sys.version_info[:2] == (3, 13), f'Expected Python 3.13; got {sys.version}')
    paths = {epoch: select_checkpoint(*entry) for epoch, entry in FILES.items()}
    print('D1 CHECKPOINT SHA256 VERIFIED:', paths, flush=True)

    def installed(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    managed = {'monai', 'connected-components-3d'}
    for name, version in PACKAGES.items():
        if name not in managed:
            require(installed(name) == version, f'Environment mismatch: {name}={installed(name)}, expected {version}')
    install = [f'{name}=={PACKAGES[name]}' for name in sorted(managed) if installed(name) != PACKAGES[name]]
    if install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', *install], check=True)
    require(all(installed(n) == v for n, v in PACKAGES.items()), 'Package lock verification failed')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from monai.data import Dataset
    from monai.inferers import sliding_window_inference
    from monai.utils import set_determinism

    require(torch.cuda.is_available() and torch.version.cuda == '13.0', 'Expected CUDA 13.0 and a GPU')
    set_determinism(seed=42)
    gpu = torch.cuda.get_device_name(0)
    print(f'ENVIRONMENT VERIFIED: {gpu} / CUDA {torch.version.cuda}', flush=True)
    with tempfile.TemporaryDirectory(prefix='d1_pair_', dir=WORK) as temp_dir:
        scratch = Path(temp_dir)
        repo = scratch / 'PBL6'
        subprocess.run(['git', 'clone', '--quiet', 'https://github.com/UwUswain/PBL6.git', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--quiet', '--detach', COMMIT], check=True)
        sys.path.insert(0, str(repo))
        from src.dataset.parser import get_kfold_splits, parse_isles_dataset, verify_isles_dataset
        from src.dataset.transforms import get_val_transforms
        from src.metrics.dice import compute_dice
        from src.metrics.lesion import analyze_gt_components
        from src.models.builder import get_model

        root = find_dataset(scratch / 'dataset')
        cases = parse_isles_dataset(root, require_mask=True)
        info = verify_isles_dataset(cases, expected_num_cases=250, require_exact_isles250_ids=True,
                                    required_modalities=['dwi', 'adc'])
        require(info['ids_sha256'] == 'd9e5eb71b2aae852', '250-case ID fingerprint mismatch')
        train_cases, val_cases = get_kfold_splits(cases, n_splits=5, seed=42)[0]
        ids = {key: [c['case_id'] for c in group] for key, group in [('train_ids', train_cases), ('val_ids', val_cases)]}
        require(len(ids['val_ids']) == 50 and len(ids['train_ids']) == 200, 'Wrong Fold 0 split size')

        models = {}
        for epoch, path in paths.items():
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            require(ckpt['epoch'] == epoch, f'Wrong epoch in {path}')
            manifest = ckpt['run_manifest']
            require(manifest['code_version']['git_commit'] == COMMIT, 'D1 commit mismatch')
            for name, digest in manifest['code_version']['core_files_sha256'].items():
                if name.startswith('src/'):
                    require(sha256(repo / name) == digest, f'Source hash mismatch: {name}')
            require(manifest['split']['fold'] == 0 and all(manifest['split'][k] == v for k, v in ids.items()),
                    'Ordered train/validation IDs differ from D1')
            cfg = manifest['config']['resolved']
            expected = {'val_roi_size': [96, 96, 64], 'amp': True, 'sw_overlap': 0.25,
                        'postprocess_min_size': 0, 'modalities': ['dwi', 'adc'], 'use_edwi': False}
            require(all(cfg[k] == v for k, v in expected.items()), f'Validation config mismatch: ep{epoch}')
            model = get_model(model_name=cfg['model_name'], in_channels=2, out_channels=1,
                              init_filters=cfg['init_filters'], fusion_mode=cfg['fusion_mode'],
                              deep_supervision=cfg['deep_supervision'])
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            models[epoch] = model.cuda().eval()
            del ckpt
        print('VERIFIED 250 CASES / FOLD 0 ORDER / BOTH MODEL STATES', flush=True)

        ds = Dataset(val_cases, get_val_transforms(modalities=['dwi', 'adc']))
        rows, lesions, slices = [], [], {}
        for i in range(len(ds)):
            item = ds[i]
            case_id = ids['val_ids'][i]
            image = item['image'].unsqueeze(0).cuda()
            gt = item['mask'][0].numpy() > 0
            preds = {}
            row = {'case_id': case_id, 'gt_voxels': int(gt.sum()), 'gt_ml': float(gt.sum()) / 1000}
            for epoch, model in models.items():
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    logits = sliding_window_inference(image, (96, 96, 64), 1, model, overlap=0.25, mode='gaussian')
                pred = (torch.sigmoid(logits[0, 0]) > 0.5).cpu().numpy()
                del logits
                preds[epoch] = pred
                n = int(pred.sum())
                tp = int(np.count_nonzero(pred & gt))
                dice = compute_dice(pred, gt)
                if i < 2:
                    verify_scores(epoch, dice, n, REFERENCE[epoch][i], case_id)
                row.update({f'ep{epoch}_{k}': v for k, v in {'dice': dice, 'pred_voxels': n, 'pred_ml': n / 1000,
                           'tp': tp, 'fp': n - tp, 'fn': row['gt_voxels'] - tp}.items()})
                lesions.extend({'case_id': case_id, 'epoch': epoch, **r} for r in analyze_gt_components(pred, gt))
            row['dice_delta_100_minus_84'] = row['ep100_dice'] - row['ep84_dice']
            rows.append(row)
            z_gt = int(np.argmax(gt.sum(axis=(0, 1))))
            error = gt ^ preds[100]
            z_err = int(np.argmax(error.sum(axis=(0, 1))))
            slices[case_id] = [(z, item['image'][0, :, :, z].numpy().copy(), gt[:, :, z].copy(),
                                *(preds[e][:, :, z].copy() for e in FILES)) for z in (z_gt, z_err)]
            print(f'{i + 1:02d}/50 {case_id}: {row["ep84_dice"]:.4f} vs {row["ep100_dice"]:.4f}', flush=True)
            if i == 1:
                print('SMOKE 2 CASES x 2 CHECKPOINTS PASSED; continuing', flush=True)

        df = pd.DataFrame(rows)
        out = Path(tempfile.mkdtemp(prefix='d1_pair_results_', dir=WORK))
        df.to_csv(out / 'cases.csv', index=False)
        summary = {'gpu': gpu, 'commit': COMMIT, 'reference_tolerance_dice': 0.005, 'epochs': {}}
        for epoch in FILES:
            sums = {k: int(df[f'ep{epoch}_{k}'].sum()) for k in ('tp', 'fp', 'fn', 'pred_voxels')}
            mean = float(df[f'ep{epoch}_dice'].mean())
            summary['epochs'][epoch] = {'mean_dice': mean, **sums}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        parity_errors = []
        for epoch, actual in summary['epochs'].items():
            try:
                verify_scores(epoch, actual['mean_dice'], actual['pred_voxels'], FULL_REFERENCE[epoch], 'Full Fold 0')
            except RuntimeError as error:
                parity_errors.append(str(error))
        comp = pd.DataFrame(lesions)
        require(not comp.empty, 'No GT lesions found across Fold 0; check the dataset')
        comp.to_csv(out / 'lesions.csv', index=False)
        comp['size_band_ml'] = pd.cut(comp['comp_voxels'], bins=[0, 100, 1000, np.inf],
                                      labels=['<0.1', '0.1–<1', '>=1'], right=False)
        comp.groupby(['epoch', 'size_band_ml'], observed=False).agg(
            n_gt=('comp_id', 'size'), n_detected=('detected_iou20', 'sum'),
            recall_iou20=('detected_iou20', 'mean'), recall_any_hit=('touched_any', 'mean'),
        ).to_csv(out / 'lesion_recall.csv')
        selected = {'largest_drop': df.nsmallest(5, 'dice_delta_100_minus_84')['case_id'].tolist(),
                    'lowest_ep84': df.nsmallest(5, 'ep84_dice')['case_id'].tolist()}
        summary['overlay_groups'] = selected
        for case_id in dict.fromkeys(selected['largest_drop'] + selected['lowest_ep84']):
            fig, axes = plt.subplots(2, 3, figsize=(10, 7))
            for (z, dwi, *masks), ax_row in zip(slices[case_id], axes):
                for ax, title, mask in zip(ax_row, ('GT', 'Epoch 84', 'Epoch 100'), masks):
                    ax.imshow(dwi.T, cmap='gray', origin='lower')
                    ax.imshow(np.ma.masked_where(~mask.T, mask.T), cmap='autumn', alpha=0.45,
                              origin='lower', vmin=0, vmax=1)
                    ax.set_title(f'{title} / z={z}')
                    ax.axis('off')
            fig.suptitle(case_id)
            fig.tight_layout()
            fig.savefig(out / f'{case_id}.png', dpi=120)
            plt.close(fig)
        summary['full_parity_passed'] = not parity_errors
        summary['parity_errors'] = parity_errors
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        archive = shutil.make_archive(str(out), 'zip', root_dir=out)
        if parity_errors:
            print(f'FULL FOLD 0 PARITY FAILED. Diagnostic files: {archive}', flush=True)
            for error in parity_errors:
                print(error, flush=True)
        else:
            print(f'FULL FOLD 0 PARITY PASSED. Download {archive}', flush=True)


if __name__ == '__main__':
    main()
