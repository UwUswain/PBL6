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
         95: ('checkpoint_epoch_95.pth', '0cfe3f601dd5715a3f63e204f513cdef61ec2d31ae2d2db9a7562cc431978beb')}
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
        scales = {}
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
            models[epoch] = model.cpu()
            require(not list(model.named_buffers()), "Model buffers require explicit restoration")
            scales[epoch] = float(ckpt["scaler_state_dict"]["scale"])
            del ckpt
        print('VERIFIED 250 CASES / FOLD 0 ORDER / BOTH MODEL STATES', flush=True)


        from src.dataset.transforms import get_train_transforms
        from src.losses.builder import get_loss_function
        loss_fn = get_loss_function(loss_type='dice_focal', include_background=False,
                                    sigmoid=True, squared_pred=False, gamma=2.0, alpha=None)
        out = Path(tempfile.mkdtemp(prefix='d1_gradient_probe_', dir=WORK))
        reports = []
        transform = get_train_transforms(patch_size=(96, 96, 80), modalities=['dwi', 'adc'])
        transform.set_random_state(seed=42)
        # Three batches of two subjects, two patches per subject.
        for batch_index in range(3):
            cases_batch = train_cases[batch_index*2:batch_index*2+2]
            patches = [patch for case in cases_batch for patch in transform(case)]
            images = torch.stack([p['image'] for p in patches]).cuda()
            labels = torch.stack([p['mask'] for p in patches]).cuda()
            for epoch in (84, 95):
                model = models[epoch].cuda().train()
                amp_scale = scales[epoch]
                baseline_gradients = {}
                for mode in ('original_amp_loss', 'fp32_loss'):
                    set_determinism(seed=4200 + batch_index)
                    model.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats()
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                    targets = [labels] + [torch.nn.functional.interpolate(
                        labels, size=x.shape[2:], mode='nearest') for x in outputs[1:]]
                    values = []
                    components = []
                    with torch.amp.autocast('cuda', enabled=mode == 'original_amp_loss'):
                        for logits, target in zip(outputs, targets):
                            if mode == 'fp32_loss':
                                logits, target = logits.float(), target.float()
                            values.append(loss_fn(logits, target))
                            with torch.no_grad():
                                components.append({
                                    'dice': float(loss_fn.dice(logits, target)),
                                    'focal': float(loss_fn.focal(logits, target))})
                        total = values[0] + 0.5*values[1] + 0.25*values[2]
                    # Same restored scale for both branches. No optimizer update.
                    (total * amp_scale).backward()
                    groups = {}
                    dot = norm_a = norm_b = diff = 0.0
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None:
                            continue
                        grad = parameter.grad.detach().float().cpu() / amp_scale
                        group = ('main_head' if name.startswith('seg_dwi.conv_final') else
                                 'aux_heads' if name.startswith('aux_head') else 'remaining_layers')
                        stats = groups.setdefault(group, {'squared_norm': 0.0, 'nonfinite': 0})
                        stats['nonfinite'] += int((~torch.isfinite(grad)).sum())
                        stats['squared_norm'] += float((grad.double()**2).sum())
                        if mode == 'original_amp_loss':
                            baseline_gradients[name] = grad
                        else:
                            a = baseline_gradients[name].double()
                            b = grad.double()
                            dot += float((a*b).sum())
                            norm_a += float((a*a).sum())
                            norm_b += float((b*b).sum())
                            diff += float(((a-b)**2).sum())
                    report = {'epoch': epoch, 'batch': batch_index, 'mode': mode, 'case_ids': [c['case_id'] for c in cases_batch],
                              'patch_shape': list(images.shape), 'scale': amp_scale,
                              'loss': float(total.detach()), 'head_losses': [float(x.detach()) for x in values],
                              'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                              'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                              'components': components, 'gradient_groups': groups,
                              'foreground_per_patch': [int((t>0).sum()) for t in labels]}
                    if mode == 'fp32_loss':
                        report['gradient_cosine'] = dot / max((norm_a*norm_b)**0.5, 1e-30)
                        report['relative_gradient_difference'] = (diff/max(norm_a,1e-30))**0.5
                    reports.append(report)
                    print(json.dumps(report), flush=True)
                    (out / 'probe.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
                    del outputs, targets, values, total, logits, target
                model.zero_grad(set_to_none=True)
                model.cpu()
                del baseline_gradients
                torch.cuda.empty_cache()
            del images, labels
        (out / 'protocol.txt').write_text(
            'Epochs 84 and 95; identical batches across epochs; first 6 ordered train subjects; 3 batches of 4 patches; seed 42; '
            'matched model weights and forward RNG; per-checkpoint restored AMP scale; telemetry matches loss context; gradients before clipping; '
            'no optimizer updates. Screening only; cannot establish collapse causality. '
            'Environment and source verified by runner. Auxiliary weights 0.5, 0.25.',
            encoding='utf-8')
        archive = shutil.make_archive(str(out), 'zip', root_dir=out)
        print(f'GRADIENT PROBE COMPLETE. Download {archive}', flush=True)


if __name__ == '__main__':
    main()
