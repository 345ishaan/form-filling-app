"""Tests for post-run CMS filled PDF export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.cms_sim import (
    _cms_filled_pdf_share_links,
    _cms_render_filled_pdf,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BLANK_G28 = REPO_ROOT / "tasks" / "cms" / "blank" / "g-28.pdf"


@pytest.mark.skipif(not BLANK_G28.is_file(), reason="blank g-28.pdf not in repo")
def test_export_cms_filled_pdf_after_agent(tmp_path: Path) -> None:
    filled_export = {
        "forms": {
            "g28": {
                "fields": [
                    {"id": "Pt1Line2a_FamilyName", "value": "Gupta"},
                    {"id": "Pt1Line2b_GivenName", "value": "Ishan"},
                ]
            }
        }
    }
    result = _cms_render_filled_pdf(
        filled_export=filled_export,
        blank_pdf_bytes=BLANK_G28.read_bytes(),
        run_dir=tmp_path,
        blank_pdf_name="g-28.pdf",
        form_type="g28",
    )
    assert not result.get("error"), result
    out_pdf = tmp_path / "form_filled_g28.pdf"
    assert out_pdf.is_file()
    assert (tmp_path / "g-28.pdf").is_file()
    assert result["exports"][0]["fields_written"] >= 2


def test_build_share_links_without_deploy() -> None:
    result = _cms_filled_pdf_share_links("cms-test-run", ["form_filled_g28.pdf"])
    assert result["run_id"] == "cms-test-run"
    assert result["links"] == {}
    assert "deploy_note" in result


def test_build_share_links_with_mock_endpoint() -> None:
    mock_fn = MagicMock()
    mock_fn.get_web_url.return_value = "https://example--cms-filled.modal.run"
    with patch("modal.Function.from_name", return_value=mock_fn):
        result = _cms_filled_pdf_share_links("cms-abc", ["form_filled_g28.pdf"])
    assert result["links"]["form_filled_g28.pdf"].startswith("https://example--cms-filled.modal.run")
    assert "run_id=cms-abc" in result["links"]["form_filled_g28.pdf"]
