from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from medical_rag import outputs as outputs_module
from medical_rag.outputs import (
    BACKUP_PREFIX,
    STAGING_PREFIX,
    OutputCommitError,
    _replace_with_retry,
    blocked_outputs,
    chapter_file_name,
    is_internal_output,
    staged_output_dir,
)


class ReplaceRetryTests(unittest.TestCase):
    def test_retries_transient_permission_error_on_windows(self) -> None:
        source = Path("source")
        destination = Path("destination")
        # Use a patched function so no filesystem entry is needed for this
        # unit test; the third attempt succeeds.
        with mock.patch("medical_rag.outputs.os.replace", side_effect=[PermissionError(5, "拒绝访问"), PermissionError(5, "拒绝访问"), None]) as replace:
            with mock.patch("medical_rag.outputs.time.sleep") as sleep:
                _replace_with_retry(source, destination, attempts=3)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_persistent_permission_error_has_actionable_message(self) -> None:
        with mock.patch("medical_rag.outputs.os.replace", side_effect=PermissionError(5, "拒绝访问")):
            with mock.patch("medical_rag.outputs.time.sleep"):
                with self.assertRaisesRegex(PermissionError, "关闭正在打开 processed_v3"):
                    _replace_with_retry(Path("source"), Path("destination"), attempts=2)


class StagedOutputDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_outputs_"))
        self.target = self.tmp / "processed_v3"

    def _write(self, staging: Path, chapters: list[str], pages: int) -> None:
        cleaned = staging / "cleaned"
        structured = staging / "structured"
        cleaned.mkdir(parents=True, exist_ok=True)
        structured.mkdir(parents=True, exist_ok=True)
        for number in range(1, pages + 1):
            (cleaned / f"page-{number:04d}.md").write_text(f"page {number}", encoding="utf-8")
        for name in chapters:
            (structured / name).write_text(f"# {name}", encoding="utf-8")
        (staging / "quality.json").write_text("{}", encoding="utf-8")

    def _residue(self) -> list[str]:
        return sorted(
            path.name
            for path in self.target.iterdir()
            if path.name.startswith((STAGING_PREFIX, BACKUP_PREFIX))
        )

    def test_replaces_and_removes_stale_outputs(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md", "02-乙.md"], pages=3)
        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")),
            ["01-甲.md", "02-乙.md"],
        )
        self.assertEqual(len(list((self.target / "cleaned").glob("*.md"))), 3)

        # 第二次只产出 1 章 1 页：旧的 02-乙.md 与多余页必须消失
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md"], pages=1)
        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")), ["01-甲.md"]
        )
        self.assertEqual(
            sorted(p.name for p in (self.target / "cleaned").glob("*.md")), ["page-0001.md"]
        )
        self.assertTrue((self.target / "quality.json").exists())

    def test_missing_managed_entry_is_deleted(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md"], pages=1)
        self.assertTrue((self.target / "quality.json").exists())
        # 第二次不产出 quality.json → 旧文件应被删除，而不是留下来冒充新结果
        with staged_output_dir(self.target) as staging:
            structured = staging / "structured"
            structured.mkdir(parents=True)
            (structured / "01-甲.md").write_text("x", encoding="utf-8")
        self.assertFalse((self.target / "quality.json").exists())

    def test_failure_keeps_previous_output_and_leaves_no_residue(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md", "02-乙.md"], pages=3)
        before = sorted(p.name for p in (self.target / "structured").glob("*.md"))

        with self.assertRaises(RuntimeError):
            with staged_output_dir(self.target) as staging:
                self._write(staging, ["01-新.md"], pages=1)
                raise RuntimeError("生成失败")

        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")), before
        )
        self.assertTrue((self.target / "quality.json").exists())
        self.assertEqual(self._residue(), [])

    def test_no_residue_after_success(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md"], pages=1)
        self.assertEqual(self._residue(), [])

    def test_staging_dir_is_created_inside_target(self) -> None:
        with staged_output_dir(self.target) as staging:
            self.assertEqual(staging.parent, self.target)
            self.assertTrue(staging.name.startswith(STAGING_PREFIX))


class RollbackFailureTests(unittest.TestCase):
    """回滚也失败时（产物被别的安全上下文写过）绝不能把备份删掉。

    旧实现先删新产物、再 ``except OSError: pass`` 放回备份、最后无条件
    ``rmtree(backup)``：一旦放不回，旧产物就被永久删除。这里锁死新行为：
    报错说清哪一项没恢复，备份目录保留下来。
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_rollback_"))
        self.target = self.tmp / "processed_v3"
        with staged_output_dir(self.target) as staging:
            (staging / "cleaned").mkdir(parents=True)
            (staging / "cleaned" / "page-0001.md").write_text("旧页", encoding="utf-8")
            (staging / "structured").mkdir(parents=True)
            (staging / "structured" / "01-旧章.md").write_text("旧章", encoding="utf-8")
            (staging / "quality.json").write_text('{"v": 1}', encoding="utf-8")

    def _write_new(self, staging: Path) -> None:
        (staging / "cleaned").mkdir(parents=True)
        (staging / "cleaned" / "page-0001.md").write_text("新页", encoding="utf-8")
        (staging / "structured").mkdir(parents=True)
        (staging / "structured" / "01-新章.md").write_text("新章", encoding="utf-8")

    def _backups(self) -> list[Path]:
        return [p for p in self.target.iterdir() if p.name.startswith(BACKUP_PREFIX)]

    def _failing_replace(self, source: Path, destination: Path, attempts: int = 8) -> None:
        # 新产物（来自 .staging-*）放不进去 → 触发回滚；从备份放回的不拦
        if Path(source).name == "structured" and STAGING_PREFIX in str(source):
            raise PermissionError(f"无法替换处理产物：{source} → {destination}")
        _replace_with_retry(source, destination, attempts=attempts)

    def test_unremovable_new_output_keeps_backup_and_raises(self) -> None:
        real_remove = outputs_module._remove

        def fake_remove(path: Path) -> bool:
            # 新 cleaned 删不掉：回滚到这里就卡住，不能硬放回备份
            if Path(path).name == "cleaned":
                return False
            return real_remove(path)

        with mock.patch(
            "medical_rag.outputs._replace_with_retry", side_effect=self._failing_replace
        ):
            with mock.patch("medical_rag.outputs._remove", side_effect=fake_remove):
                with self.assertRaises(OutputCommitError) as caught:
                    with staged_output_dir(self.target) as staging:
                        self._write_new(staging)

        message = str(caught.exception)
        self.assertIn("回滚失败", message)
        self.assertIn("cleaned", message)
        backups = self._backups()
        self.assertEqual(len(backups), 1, "备份目录必须保留，否则旧产物就没了")
        self.assertTrue((backups[0] / "cleaned" / "page-0001.md").exists())
        # 没被卡住的项照旧回滚到位
        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")), ["01-旧章.md"]
        )
        self.assertTrue((self.target / "quality.json").exists())

    def test_clean_rollback_removes_backup(self) -> None:
        with mock.patch(
            "medical_rag.outputs._replace_with_retry", side_effect=self._failing_replace
        ):
            with self.assertRaises(PermissionError) as caught:
                with staged_output_dir(self.target) as staging:
                    self._write_new(staging)

        self.assertNotIsInstance(caught.exception, OutputCommitError)
        self.assertEqual(self._backups(), [])
        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")), ["01-旧章.md"]
        )
        self.assertTrue((self.target / "quality.json").exists())


class BlockedOutputTests(unittest.TestCase):
    """`blocked_outputs` 要能认出“名字还在、但打不开”的产物。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_blocked_"))
        self.target = self.tmp / "processed_v3"
        (self.target / "cleaned").mkdir(parents=True)
        (self.target / "cleaned" / "page-0001.md").write_text("x", encoding="utf-8")
        (self.target / "structured").mkdir(parents=True)
        (self.target / "quality.json").write_text("{}", encoding="utf-8")

    def test_readable_outputs_are_not_reported(self) -> None:
        self.assertEqual(blocked_outputs(self.target), [])

    def test_missing_target_reports_nothing(self) -> None:
        self.assertEqual(blocked_outputs(self.tmp / "nope"), [])

    def test_access_denied_directories_are_reported(self) -> None:
        with mock.patch(
            "medical_rag.outputs.os.scandir", side_effect=PermissionError(5, "拒绝访问")
        ):
            self.assertEqual(blocked_outputs(self.target), ["cleaned", "structured"])

    def test_access_denied_file_is_reported(self) -> None:
        with mock.patch("builtins.open", side_effect=PermissionError(5, "拒绝访问")):
            self.assertEqual(blocked_outputs(self.target), ["quality.json"])


class NamingTests(unittest.TestCase):
    def test_chapter_file_name_sanitises_illegal_characters(self) -> None:
        self.assertEqual(chapter_file_name(2, "骨/关节:总论"), "02-骨_关节_总论.md")

    def test_chapter_file_name_falls_back_for_empty_title(self) -> None:
        self.assertEqual(chapter_file_name(3, "   "), "03-第3章.md")

    def test_is_internal_output(self) -> None:
        self.assertTrue(is_internal_output(Path("processed_v3/.staging-abc/structured/01.md")))
        self.assertTrue(is_internal_output(Path("processed_v3/.backup-xyz/01.md")))
        self.assertFalse(is_internal_output(Path("processed_v3/structured/01.md")))


if __name__ == "__main__":
    unittest.main()
