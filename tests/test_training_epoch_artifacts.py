"""Exercise the actual production epoch loop without MRI data or GPU work."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from scripts import train
import torch


class TestEpochArtifacts(unittest.TestCase):
    def test_resume_loop_writes_telemetry_and_checkpoints(self):
        # Execute the actual loop AST, not a copied implementation. Replace only
        # expensive training/validation with fixed results; use real torch.save.
        tree = ast.parse(Path(train.__file__).read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        loop = next(n for n in main.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "epoch")
        program = compile(ast.Module(body=[loop], type_ignores=[]), train.__file__, "exec")
        stateful = Mock()
        stateful.state_dict.return_value = {"test_tensor": torch.tensor([1.0])}
        optimizer = Mock()
        optimizer.param_groups = [{"lr": 1.104528e-4}]
        optimizer.state_dict.return_value = {"param_groups": optimizer.param_groups}
        scheduler = Mock(T_max=150)
        scheduler.state_dict.return_value = {"T_max": 150}
        fake_train = Mock(return_value=(0.58, {
            "train_loss": 0.58, "grad_norm": 1.2,
            "empty_patch_ratio": 0.0, "mean_fg_aux1": 0.0,
        }))
        fake_validate = Mock(return_value=(0.69, 50, 123))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            env = dict(vars(train))
            env.update(
                start_epoch=71, total_epochs=100, optimizer=optimizer,
                scheduler=scheduler, model=stateful, scaler=stateful,
                train_epoch=fake_train, validate=fake_validate,
                train_loader=None, val_loader=None, loss_fn=None,
                device=torch.device("cpu"), use_amp=False,
                val_interval=2, val_roi_size=(96, 96, 64),
                args=SimpleNamespace(sw_overlap=0.25), postprocess_min_size=0,
                best_dice=0.6816, val_dice=0.0, run_dir=folder,
                run_manifest={"test": True}, is_resumed=True,
                resumed_source="source_epoch70.pth", resumed_epoch=70,
                telemetry_file=folder / "telemetry_history.jsonl",
            )
            # Reproduce the Kaggle failure if the production import is removed.
            with self.assertRaisesRegex(NameError, "json"):
                exec(program, {key: value for key, value in env.items() if key != "json"})
            fake_train.reset_mock()
            scheduler.reset_mock()
            exec(program, env)
            records = [json.loads(line) for line in env["telemetry_file"].read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], list(range(71, 101)))
            self.assertFalse(records[0]["val_ran"])
            self.assertIsNone(records[0]["val_dice"])
            self.assertEqual(records[0]["mean_fg_aux1"], 0.0)
            self.assertTrue(records[1]["val_ran"])
            self.assertEqual(fake_validate.call_count, 15)
            self.assertEqual(fake_train.call_count, 30)
            self.assertEqual(scheduler.step.call_count, 30)
            for name, epoch in [("best_metric_model.pth", 72), ("last_checkpoint.pth", 100)] + [
                (f"checkpoint_epoch_{e}.pth", e) for e in (80, 85, 90, 95, 100)
            ]:
                checkpoint = torch.load(folder / name, map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["epoch"], epoch)
                self.assertEqual(checkpoint["resume_provenance"]["resumed_epoch"], 70)


if __name__ == "__main__":
    unittest.main()
