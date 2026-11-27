"""Security spec for startup weights.

These tests fail while model startup still downloads the Google Drive archive,
extracts it without comparing the bytes to ``utils.downloader.WEIGHTS_ZIP_SHA256``,
and loads the extracted ``.pt`` files through a pickle loader.

They are expected to pass once both of these are true:

- ``download_all_models`` compares the downloaded archive to
  ``utils.downloader.WEIGHTS_ZIP_SHA256`` (hex SHA-256, read when the download
  runs) and extracts only those verified bytes. A later read of the same path
  must not be extracted. No member is created before the digest matches.
- ``YOLOModule`` and the action, ball, court, and player wrappers refuse a
  checkpoint whose pickle runs code. Refusal must not run that code and must
  not be skipped for the files startup just extracted.

Run:

    python -m pytest tests/test_unverified_weight_archive.py -q --import-mode=importlib --confcutdir=tests
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable
from typing import Dict

import pytest
import requests
import torch
from requests import Response

os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml_manager.models.ActionDetectorModule import ActionDetectorModule
from ml_manager.models.BallDetectorModule import BallDetectorModule
from ml_manager.models.CourtSegmentationModule import CourtSegmentationModule
from ml_manager.models.PlayerDetectorModule import PlayerDetectorModule
from ml_manager.models.YoloModule import YOLOModule
from ml_manager.settings.weights_config import ModelWeightsConfig
from ml_manager.utils import downloader

ARCHIVE_MEMBERS = (
    "ball/weights/best.pt",
    "action/weights/best.pt",
    "court/weights/best.pt",
    "game_state/placeholder.txt",
)
BENIGN_MEMBER = b"benign-checkpoint-not-a-pickle"
MALICIOUS_ONLY_MEMBER = "game_state/untrusted-marker.txt"


def _record_execution(path: str) -> str:
    """Side effect a checkpoint payload must not be able to invoke."""
    Path(path).write_text("executed")
    return path


class _CheckpointPayload:
    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self):
        return (_record_execution, (self.path,))


def _pickle_checkpoint(path: Path, marker: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": _CheckpointPayload(str(marker))}, path)
    return path.read_bytes()


def _zip_bytes(members: Dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _benign_archive() -> bytes:
    members = {name: BENIGN_MEMBER for name in ARCHIVE_MEMBERS}
    return _zip_bytes(members)


def _malicious_archive(marker: Path) -> bytes:
    checkpoint = _pickle_checkpoint(marker.parent / "payload.pt", marker)
    members = {name: checkpoint for name in ARCHIVE_MEMBERS if name.endswith(".pt")}
    members["game_state/placeholder.txt"] = b"x"
    members[MALICIOUS_ONLY_MEMBER] = b"untrusted"
    return _zip_bytes(members)


class _ResponseBody:
    """File-like body used by ``requests.Response.iter_content``.

    ``gdown`` 6.5.2 reads the response with ``raw.read(size, decode_content=...)``
    (``gdown/download.py``, ``_iter_response_chunks``) and writes those chunks
    unchanged to the destination. This stand-in is only the response body; the
    real ``gdown.download`` parser and writer still run.
    """

    def __init__(self, data: bytes) -> None:
        self._buffer = BytesIO(data)

    def read(self, amt: int = -1, decode_content: bool = False) -> bytes:
        return self._buffer.read(amt)

    def stream(self, amt: int = 8192, decode_content: bool = False):
        while True:
            chunk = self._buffer.read(amt)
            if not chunk:
                break
            yield chunk

    def close(self) -> None:
        return None

    def release_conn(self) -> None:
        return None


def _drive_response(url: str, body: bytes) -> Response:
    response = Response()
    response.status_code = 200
    response.url = url
    response.headers["Content-Type"] = "application/zip"
    response.headers["Content-Disposition"] = 'attachment; filename="all_weights.zip"'
    response.headers["Content-Length"] = str(len(body))
    response.raw = _ResponseBody(body)
    return response


def _install_drive_body(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    """Serve ``body`` as the object ``gdown.download(id=...)`` saves.

    Product code calls ``gdown.download(id=WEIGHTS_ZIP_DRIVE_ID, output=...)``.
    gdown turns that id into ``https://drive.google.com/uc?id=`` and, when the
    response is not HTML and carries ``Content-Disposition``, writes the body
    to ``output`` (``gdown/download.py`` ``download``).
    """
    real_get = requests.Session.get

    def fake_get(self, url, *args, **kwargs):
        if "drive.google.com" in str(url) or "docs.google.com" in str(url):
            return _drive_response(str(url), body)
        return real_get(self, url, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "get", fake_get)


def _pin(monkeypatch: pytest.MonkeyPatch, digest: str) -> None:
    monkeypatch.setattr(downloader, "WEIGHTS_ZIP_SHA256", digest, raising=False)


def _block_unrelated_asset_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default pose-asset name from opening a network connection.

    Startup still calls ``YOLO`` on that name after the three archive
    checkpoints. ``attempt_download_asset`` returns a path that is already on
    disk and raises immediately otherwise, which is what the offline branch of
    that helper does after a failed fetch.
    """
    import ultralytics.utils.downloads as downloads

    def local_only(file, *args, **kwargs):
        candidate = Path(str(file))
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(file)

    monkeypatch.setattr(downloads, "attempt_download_asset", local_only)


def _record_extracted_creates(monkeypatch: pytest.MonkeyPatch, weights_dir: Path) -> list:
    """Record archive members created under ``weights_dir``.

    ``zipfile`` writes members with the builtin ``open(..., "wb")``, which goes
    through the C file object rather than ``os.open``. Deleting the member
    afterwards still leaves it in this list.
    """
    created = []
    real_open = open
    root = weights_dir.resolve()

    def tracking_open(file, mode="r", *args, **kwargs):
        mode_text = str(mode)
        if any(flag in mode_text for flag in ("w", "a", "x")):
            try:
                resolved = Path(os.fspath(file)).resolve()
                relative = resolved.relative_to(root)
            except (OSError, ValueError):
                relative = None
            if relative is not None and len(relative.parts) > 1:
                created.append(relative.as_posix())
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    return created


def _swap_archive_on_later_read(monkeypatch: pytest.MonkeyPatch, archive_path: Path, replacement: bytes) -> None:
    """After the archive has been read once, the next read sees ``replacement``.

    ``ZipFile`` opens the path with ``io.open`` and a hash check usually uses
    ``open`` or ``Path.read_bytes``. Those are different bindings of the same
    builtin, so both are wrapped, as is ``os.open``. A check that hashes
    ``archive_path`` and then opens that path again for extraction must not
    extract ``replacement``.
    """
    import io

    real_open = io.open
    real_os_open = os.open
    target = archive_path.resolve()
    state = {"reads": 0, "swapped": False}

    def _note_read() -> None:
        state["reads"] += 1
        if state["reads"] >= 2 and not state["swapped"]:
            state["swapped"] = True
            target.write_bytes(replacement)

    def tracking_open(file, mode="r", *args, **kwargs):
        path = Path(os.fspath(file))
        try:
            same = path.resolve() == target
        except OSError:
            same = False
        mode_text = str(mode)
        if same and "b" in mode_text and any(flag in mode_text for flag in ("r", "+")):
            _note_read()
        return real_open(file, mode, *args, **kwargs)

    def tracking_os_open(path, flags, mode=0o777, *, dir_fd=None):
        try:
            same = Path(path).resolve() == target
        except (OSError, ValueError):
            same = False
        if same and (flags & os.O_ACCMODE) in (os.O_RDONLY, os.O_RDWR):
            _note_read()
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(io, "open", tracking_open)
    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(os, "open", tracking_os_open)


def _assert_untrusted_archive_rejected(weights_dir: Path, marker: Path, created: list) -> None:
    assert not marker.exists()
    assert created == []
    extracted = [path for path in weights_dir.rglob("*") if path.is_file() and path.parent != weights_dir]
    assert extracted == []


def _run_startup(monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    monkeypatch.chdir(project)
    _block_unrelated_asset_downloads(monkeypatch)
    from ml_manager.ml_manager import MLManager

    MLManager(device="cpu")


def test_download_all_models_rejects_unpinned_archive_before_extract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "executed"
    body = _malicious_archive(marker)
    _pin(monkeypatch, "0" * 64)
    _install_drive_body(monkeypatch, body)
    weights_dir = tmp_path / "weights"
    created = _record_extracted_creates(monkeypatch, weights_dir)

    downloader.download_all_models(weights_dir=weights_dir, quiet=True)

    _assert_untrusted_archive_rejected(weights_dir, marker, created)
    assert not (weights_dir / MALICIOUS_ONLY_MEMBER).exists()


def test_download_missing_weights_rejects_unpinned_archive_before_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "executed"
    body = _malicious_archive(marker)
    _pin(monkeypatch, "0" * 64)
    _install_drive_body(monkeypatch, body)
    weights_dir = tmp_path / "weights"
    created = _record_extracted_creates(monkeypatch, weights_dir)

    ModelWeightsConfig.download_missing_weights(weights_dir=weights_dir, force=True)

    _assert_untrusted_archive_rejected(weights_dir, marker, created)


def test_matching_pin_extracts_that_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _benign_archive()
    _pin(monkeypatch, hashlib.sha256(body).hexdigest())
    _install_drive_body(monkeypatch, body)
    weights_dir = tmp_path / "weights"

    assert downloader.download_all_models(weights_dir=weights_dir, quiet=True) is True

    for name in ARCHIVE_MEMBERS:
        assert (weights_dir / name).read_bytes() == BENIGN_MEMBER


def test_manager_startup_does_not_run_untrusted_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "executed"
    body = _malicious_archive(marker)
    _pin(monkeypatch, "0" * 64)
    _install_drive_body(monkeypatch, body)
    weights_dir = tmp_path / "weights"
    created = _record_extracted_creates(monkeypatch, weights_dir)

    _run_startup(monkeypatch, tmp_path)

    _assert_untrusted_archive_rejected(weights_dir, marker, created)


def test_bytes_opened_for_extract_are_the_bytes_that_matched_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "executed"
    benign = _benign_archive()
    malicious = _malicious_archive(marker)
    _pin(monkeypatch, hashlib.sha256(benign).hexdigest())
    _install_drive_body(monkeypatch, benign)
    weights_dir = tmp_path / "weights"
    archive_path = weights_dir / "all_weights.zip"
    _swap_archive_on_later_read(monkeypatch, archive_path, malicious)

    _run_startup(monkeypatch, tmp_path)

    assert not marker.exists()
    assert not (weights_dir / MALICIOUS_ONLY_MEMBER).exists()
    for name in ARCHIVE_MEMBERS:
        extracted = weights_dir / name
        if extracted.is_file():
            assert extracted.read_bytes() == BENIGN_MEMBER


@pytest.mark.parametrize(
    "loader",
    [
        YOLOModule,
        ActionDetectorModule,
        BallDetectorModule,
        CourtSegmentationModule,
        PlayerDetectorModule,
    ],
)
def test_checkpoint_loader_does_not_run_pickle(
    tmp_path: Path, loader: Callable[..., object]
) -> None:
    marker = tmp_path / "executed"
    checkpoint = tmp_path / "best.pt"
    _pickle_checkpoint(checkpoint, marker)

    with pytest.raises(Exception):
        loader(model_path=str(checkpoint), device="cpu")

    assert not marker.exists()
