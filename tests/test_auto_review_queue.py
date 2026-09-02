import argparse
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from auto_review_queue import (  # noqa: E402
    Decision,
    WorkerError,
    acquire_worker_lock,
    apply_review,
    ci_state,
    created_at_on_or_before,
    create_review_worktree,
    decide,
    has_current_head_formal_review,
    load_cached_review,
    parse_args,
    parse_timestamp,
    pr_file_paths,
    queued_prs,
    submission_dir_from_files,
)
from generate_submissions_data import package_sha256  # noqa: E402


class AutoReviewQueueTests(unittest.TestCase):
    def test_merge_is_bound_to_reviewed_head_for_all_merge_modes(self) -> None:
        head_sha = "a" * 40
        live = {
            "headRefOid": head_sha,
            "state": "OPEN",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "statusCheckRollup": [
                {"name": "submission-validation", "conclusion": "SUCCESS"}
            ],
        }
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        comment_file = Path(tmp.name) / "pr-comment.md"
        comment_file.write_text(
            "# AI Agent 评审意见\n\n## 七维评分\n- **任务书对齐 4/5**：证据充分。\n",
            encoding="utf-8",
        )
        for admin_merge in (False, True):
            with self.subTest(admin_merge=admin_merge):
                with (
                    patch("auto_review_queue.pr_meta", return_value=live),
                    patch("auto_review_queue.has_current_head_formal_review", return_value=False),
                    patch("auto_review_queue.run") as run_mock,
                ):
                    apply_review(
                        "open-city-ai/haidian",
                        42,
                        head_sha,
                        Decision("accept", 90, "accepted"),
                        comment_file,
                        ROOT,
                        admin_merge=admin_merge,
                    )

                    review_call = next(
                        call
                        for call in run_mock.call_args_list
                        if call.args[0][:3] == ["gh", "pr", "review"]
                    )
                    review_body = review_call.args[0][review_call.args[0].index("--body") + 1]
                    self.assertIn("no rejection condition was triggered", review_body)
                    self.assertIn("review-readiness checks also passed", review_body)
                    self.assertIn("Accepted for repository intake only", review_body)
                    self.assertIn("## 七维评分", review_body)
                    self.assertIn("任务书对齐 4/5", review_body)
                    run_mock.assert_any_call(
                        [
                            "gh",
                            "pr",
                            "merge",
                            "42",
                            "--repo",
                            "open-city-ai/haidian",
                            "--merge",
                            "--match-head-commit",
                            head_sha,
                            *(["--admin"] if admin_merge else []),
                        ],
                        cwd=ROOT,
                    )
                    review_command = next(
                        call.args[0]
                        for call in run_mock.call_args_list
                        if call.args[0][:3] == ["gh", "pr", "review"]
                    )
                    self.assertIn("no rejection condition was triggered", review_command[-1])

    def test_default_image_budget_matches_bilingual_packet(self) -> None:
        with patch.object(sys, "argv", ["auto_review_queue"]):
            args = parse_args()
        self.assertEqual(18, args.max_images)

    def test_cutoff_is_inclusive_and_normalized_to_utc(self) -> None:
        cutoff = parse_timestamp("2026-08-31T23:59:59+08:00")
        self.assertEqual(datetime(2026, 8, 31, 15, 59, 59, tzinfo=timezone.utc), cutoff)
        self.assertTrue(
            created_at_on_or_before({"createdAt": "2026-08-31T15:59:59Z"}, cutoff)
        )
        self.assertFalse(
            created_at_on_or_before({"createdAt": "2026-08-31T16:00:00Z"}, cutoff)
        )

    def test_cutoff_requires_timezone(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_timestamp("2026-08-31T23:59:59")

    def test_formal_review_must_match_current_head(self) -> None:
        head_sha = "a" * 40
        reviews = [[
            {"commit_id": "b" * 40, "state": "APPROVED"},
            {"commit_id": head_sha, "state": "COMMENTED"},
            {"commit_id": head_sha, "state": "CHANGES_REQUESTED"},
        ]]
        completed = type("Completed", (), {"stdout": json.dumps(reviews)})()
        with patch("auto_review_queue.run", return_value=completed):
            self.assertTrue(
                has_current_head_formal_review("open-city-ai/haidian", 42, head_sha, ROOT)
            )

    def test_old_or_comment_only_reviews_do_not_count_as_formal(self) -> None:
        head_sha = "a" * 40
        reviews = [[
            {"commit_id": "b" * 40, "state": "APPROVED"},
            {"commit_id": head_sha, "state": "COMMENTED"},
        ]]
        completed = type("Completed", (), {"stdout": json.dumps(reviews)})()
        with patch("auto_review_queue.run", return_value=completed):
            self.assertFalse(
                has_current_head_formal_review("open-city-ai/haidian", 42, head_sha, ROOT)
            )

    def test_review_worktree_uses_sparse_checkout(self) -> None:
        worktree = ROOT / ".pr-worktree" / "test"
        with patch("auto_review_queue.run") as run_mock:
            create_review_worktree(
                ROOT,
                worktree,
                "refs/review/head",
                "submissions/alice/plan",
            )
        self.assertEqual(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                "--no-checkout",
                str(worktree),
                "refs/review/head",
            ],
            run_mock.call_args_list[0].args[0],
        )
        sparse_command = run_mock.call_args_list[1].args[0]
        self.assertEqual(["git", "sparse-checkout", "set", "--no-cone", "--"], sparse_command[:5])
        self.assertIn("submissions/alice/plan", sparse_command)
        self.assertIn("brief/site-package/agent_taskbook.json", sparse_command)

    def test_review_worktree_cleans_up_after_setup_failure(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worktree = Path(tmp.name) / "review-worktree"
        calls: list[list[str]] = []

        def fake_run(command: list[str], *, cwd: Path, capture: bool = True):
            calls.append(command)
            if command[1:3] == ["worktree", "add"]:
                worktree.mkdir()
                return type("Completed", (), {"stdout": ""})()
            if command[1:3] == ["sparse-checkout", "set"]:
                raise WorkerError("sparse checkout failed")
            return type("Completed", (), {"stdout": ""})()

        with patch("auto_review_queue.run", side_effect=fake_run):
            with self.assertRaisesRegex(WorkerError, "sparse checkout failed"):
                create_review_worktree(
                    ROOT,
                    worktree,
                    "refs/review/head",
                    "submissions/alice/plan",
                )

        self.assertEqual(
            ["git", "worktree", "remove", "--force", str(worktree)],
            calls[-1],
        )

    def test_review_worktree_cleans_up_after_checkout_failure(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worktree = Path(tmp.name) / "review-worktree"
        calls: list[list[str]] = []

        def fake_run(command: list[str], *, cwd: Path, capture: bool = True):
            calls.append(command)
            if command[1:3] == ["worktree", "add"]:
                worktree.mkdir()
                return type("Completed", (), {"stdout": ""})()
            if command[1:3] == ["checkout", "--detach"]:
                raise WorkerError("checkout failed")
            return type("Completed", (), {"stdout": ""})()

        with patch("auto_review_queue.run", side_effect=fake_run):
            with self.assertRaisesRegex(WorkerError, "checkout failed"):
                create_review_worktree(
                    ROOT,
                    worktree,
                    "refs/review/head",
                    "submissions/alice/plan",
                )

        self.assertEqual(
            ["git", "worktree", "remove", "--force", str(worktree)],
            calls[-1],
        )

    def test_review_worktree_does_not_remove_after_add_failure(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worktree = Path(tmp.name) / "review-worktree"
        calls: list[list[str]] = []

        def fake_run(command: list[str], *, cwd: Path, capture: bool = True):
            calls.append(command)
            if command[1:3] == ["worktree", "add"]:
                raise WorkerError("worktree add failed")
            return type("Completed", (), {"stdout": ""})()

        with patch("auto_review_queue.run", side_effect=fake_run):
            with self.assertRaisesRegex(WorkerError, "worktree add failed"):
                create_review_worktree(
                    ROOT,
                    worktree,
                    "refs/review/head",
                    "submissions/alice/plan",
                )

        self.assertEqual(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                "--no-checkout",
                str(worktree),
                "refs/review/head",
            ],
            calls[0],
        )
        self.assertEqual(1, len(calls))

    def test_accepts_score_at_threshold_when_intake_ready_even_if_not_publishable(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                name: {"status": "pass"}
                for name in [
                    "deterministic_validation",
                    "spatial_review",
                    "visual_review",
                    "professional_evidence_review",
                ]
            },
            "recommendation": "formal-review-ready",
            "can_enter_formal_review": True,
            "required_next_actions_zh": [],
        }
        decision = {
            "weighted_score_100": 60,
            "publication_recommendation": "do-not-publish",
        }
        self.assertEqual("accept", decide(review, decision, 60).action)

    def test_conditional_followups_do_not_block_intake(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                name: {"status": "pass"}
                for name in [
                    "deterministic_validation",
                    "spatial_review",
                    "visual_review",
                    "professional_evidence_review",
                ]
            },
            "recommendation": "formal-review-ready",
            "can_enter_formal_review": True,
            "required_next_actions_zh": [],
            "conditional_followups": [
                {
                    "action_zh": "正式边界发布后，从拓扑开始重算全部空间载体。",
                    "blocking_now": False,
                    "trigger": "official-data-available",
                    "owner": "participant",
                }
            ],
        }
        outcome = decide(review, {"weighted_score_100": 90}, 60)
        self.assertEqual("accept", outcome.action)

    def test_invalid_conditional_followup_fails_closed(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                name: {"status": "pass"}
                for name in [
                    "deterministic_validation",
                    "spatial_review",
                    "visual_review",
                    "professional_evidence_review",
                ]
            },
            "recommendation": "formal-review-ready",
            "can_enter_formal_review": True,
            "required_next_actions_zh": [],
            "conditional_followups": [
                {
                    "action_zh": "立即修复当前错误声明。",
                    "blocking_now": True,
                    "trigger": "now",
                    "owner": "participant",
                }
            ],
        }
        outcome = decide(review, {"weighted_score_100": 90}, 60)
        self.assertEqual("request-changes", outcome.action)
        self.assertIn("conditional_followups", outcome.reason)

    def test_non_formal_recommendation_blocks_intake(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                name: {"status": "pass"}
                for name in [
                    "deterministic_validation",
                    "spatial_review",
                    "visual_review",
                    "professional_evidence_review",
                ]
            },
            "recommendation": "request-changes",
            "can_enter_formal_review": True,
            "required_next_actions_zh": [],
        }
        outcome = decide(review, {"weighted_score_100": 90}, 60)
        self.assertEqual("request-changes", outcome.action)
        self.assertEqual("intake blocked by review fields: recommendation", outcome.reason)

    def test_false_formal_review_flag_blocks_intake(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                name: {"status": "pass"}
                for name in [
                    "deterministic_validation",
                    "spatial_review",
                    "visual_review",
                    "professional_evidence_review",
                ]
            },
            "recommendation": "formal-review-ready",
            "can_enter_formal_review": False,
            "required_next_actions_zh": [],
        }
        outcome = decide(review, {"weighted_score_100": 90}, 60)
        self.assertEqual("request-changes", outcome.action)
        self.assertEqual("intake blocked by review fields: can_enter_formal_review", outcome.reason)

    def test_required_participant_actions_block_intake(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                name: {"status": "pass"}
                for name in [
                    "deterministic_validation",
                    "spatial_review",
                    "visual_review",
                    "professional_evidence_review",
                ]
            },
            "recommendation": "formal-review-ready",
            "can_enter_formal_review": True,
            "required_next_actions_zh": ["补充权属证明。"],
        }
        outcome = decide(review, {"weighted_score_100": 90}, 60)
        self.assertEqual("request-changes", outcome.action)
        self.assertEqual("intake blocked by review fields: required_next_actions_zh", outcome.reason)

    def test_low_score_is_not_merged(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                name: {"status": "pass"}
                for name in [
                    "deterministic_validation",
                    "spatial_review",
                    "visual_review",
                    "professional_evidence_review",
                ]
            },
        }
        self.assertEqual("low-quality", decide(review, {"weighted_score_100": 59.9}, 60).action)

    def test_failed_gate_overrides_high_score(self) -> None:
        review = {
            "mandatory_rejection": {"result": "pass"},
            "gate_checks": {
                "deterministic_validation": {"status": "pass"},
                "spatial_review": {"status": "pass"},
                "visual_review": {"status": "fail"},
                "professional_evidence_review": {"status": "pass"},
            },
        }
        self.assertEqual("request-changes", decide(review, {"weighted_score_100": 95}, 60).action)

    def test_mandatory_rejection_overrides_score(self) -> None:
        review = {"mandatory_rejection": {"result": "fail"}, "gate_checks": {}}
        self.assertEqual("request-changes", decide(review, {"weighted_score_100": 95}, 60).action)

    def test_submission_scope_requires_one_author_directory(self) -> None:
        paths = ["submissions/Alice/plan/proposal.md", "submissions/Alice/plan/agent.json"]
        self.assertEqual("submissions/Alice/plan", submission_dir_from_files(paths, "alice"))
        with self.assertRaises(WorkerError):
            submission_dir_from_files(paths + ["README.md"], "alice")

    def test_worker_lock_rejects_second_holder_until_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / ".worker.lock"
            first = acquire_worker_lock(lock_path)
            try:
                with self.assertRaises(WorkerError):
                    acquire_worker_lock(lock_path)
            finally:
                first.close()
            third = acquire_worker_lock(lock_path)
            third.close()

    def test_pr_file_paths_preserve_unicode_from_paginated_json(self) -> None:
        payload = [[
            {"filename": "submissions/alice/plan/proposal.md"},
            {"filename": "submissions/alice/plan/visual/assets/01-总体方案图.png"},
        ]]
        with patch("auto_review_queue.run") as mocked_run:
            mocked_run.return_value.stdout = json.dumps(payload, ensure_ascii=False)
            paths = pr_file_paths("open-city-ai/haidian", 999, ROOT)

        self.assertEqual(payload[0][1]["filename"], paths[1])
        self.assertEqual("submissions/alice/plan", submission_dir_from_files(paths, "alice"))
        mocked_run.assert_called_once_with(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                "repos/open-city-ai/haidian/pulls/999/files",
            ],
            cwd=ROOT,
        )

    def test_queued_prs_filter_object_labels_without_search(self) -> None:
        open_prs = [
            {"number": 101, "labels": [{"name": "review/queued"}]},
            {"number": 102, "labels": [{"name": "review/ci-failed"}]},
            {"number": 103, "labels": []},
        ]
        with patch("auto_review_queue.gh_json", return_value=open_prs) as mocked_gh_json:
            self.assertEqual([open_prs[0]], queued_prs("open-city-ai/haidian", "review/queued", ROOT))

        args = mocked_gh_json.call_args.args[1]
        self.assertNotIn("--label", args)
        self.assertIn("labels", args[-1])

    def test_ci_state(self) -> None:
        self.assertEqual(
            "success",
            ci_state({"statusCheckRollup": [{"name": "submission-validation", "conclusion": "SUCCESS"}]}),
        )
        self.assertEqual(
            "failure",
            ci_state({"statusCheckRollup": [{"name": "submission-validation", "conclusion": "FAILURE"}]}),
        )
        self.assertEqual(
            "pending",
            ci_state({"statusCheckRollup": [{"name": "submission-validation", "conclusion": ""}]}),
        )
        self.assertEqual(
            "success",
            ci_state(
                {
                    "statusCheckRollup": [
                        {"name": "submission-validation", "conclusion": "SUCCESS"},
                        {"name": "unrelated", "conclusion": "FAILURE"},
                    ]
                }
            ),
        )
        self.assertEqual(
            "success",
            ci_state(
                {
                    "statusCheckRollup": [
                        {"name": "submission-validation", "conclusion": "SKIPPED"},
                        {"name": "submission-validation", "conclusion": "SUCCESS"},
                    ]
                }
            ),
        )

    def test_ci_state_uses_latest_validation_run(self) -> None:
        self.assertEqual(
            "success",
            ci_state(
                {
                    "statusCheckRollup": [
                        {
                            "name": "submission-validation",
                            "status": "COMPLETED",
                            "conclusion": "FAILURE",
                            "startedAt": "2026-08-08T15:00:00Z",
                        },
                        {
                            "name": "submission-validation",
                            "status": "COMPLETED",
                            "conclusion": "SUCCESS",
                            "startedAt": "2026-08-08T16:00:00Z",
                        },
                    ]
                }
            ),
        )
        self.assertEqual(
            "pending",
            ci_state(
                {
                    "statusCheckRollup": [
                        {
                            "name": "submission-validation",
                            "status": "COMPLETED",
                            "conclusion": "SUCCESS",
                            "startedAt": "2026-08-08T15:00:00Z",
                        },
                        {
                            "name": "submission-validation",
                            "status": "IN_PROGRESS",
                            "conclusion": "",
                            "startedAt": "2026-08-08T16:00:00Z",
                        },
                    ]
                }
            ),
        )

    def test_reuses_only_complete_matching_exact_head_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkout = Path(temp_dir) / "checkout"
            submission = checkout / "submissions" / "alice" / "plan"
            submission.mkdir(parents=True)
            schema_path = (
                checkout
                / "brief"
                / "site-package"
                / "schemas"
                / "advisory_review.schema.json"
            )
            schema_path.parent.mkdir(parents=True)
            schema_path.write_text(
                json.dumps({"properties": {"schema_version": {"const": "0.2.1"}}}),
                encoding="utf-8",
            )
            (submission / "proposal.md").write_text("proposal", encoding="utf-8")
            (submission / "manifest.json").write_text('{"files": []}', encoding="utf-8")
            digest = package_sha256(submission)
            audit = Path(temp_dir) / "audit"
            audit.mkdir()
            review = {
                "schema_version": "0.2.1",
                "submission_dir": "submissions/alice/plan",
                "mandatory_rejection": {"result": "pass"},
                "gate_checks": {
                    name: {"status": "pass"}
                    for name in [
                        "deterministic_validation",
                        "spatial_review",
                        "visual_review",
                        "professional_evidence_review",
                    ]
                },
                "recommendation": "formal-review-ready",
                "can_enter_formal_review": True,
                "required_next_actions_zh": [],
            }
            decision = {
                "submission_dir": "submissions/alice/plan",
                "reviewed_package_sha256": digest,
                "weighted_score_100": 61,
                "publication_recommendation": "publish-qualified",
                "dry_run": False,
                "model_output_schema_valid": True,
            }
            (audit / "ai-review.json").write_text(json.dumps(review), encoding="utf-8")
            (audit / "ai-decision.json").write_text(json.dumps(decision), encoding="utf-8")
            (audit / "pr-comment.md").write_text("review", encoding="utf-8")

            cached = load_cached_review(audit, "submissions/alice/plan", checkout, 60)
            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertEqual("accept", cached[2].action)

            review["schema_version"] = "0.2.0"
            (audit / "ai-review.json").write_text(json.dumps(review), encoding="utf-8")
            self.assertIsNone(load_cached_review(audit, "submissions/alice/plan", checkout, 60))

            review["schema_version"] = "0.2.1"
            (audit / "ai-review.json").write_text(json.dumps(review), encoding="utf-8")

            (submission / "proposal.md").write_text("updated", encoding="utf-8")
            self.assertIsNone(load_cached_review(audit, "submissions/alice/plan", checkout, 60))


if __name__ == "__main__":
    unittest.main()
