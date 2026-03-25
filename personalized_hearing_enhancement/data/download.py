from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

console = Console()

DATASETS = {
    "librispeech_train": {
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "sha256": "d4ddd1d5d3f09f94f5ba3dc7d3f109d968b1f0f0593675114220a7677f257306",
        "target_dir": "LibriSpeech/train-clean-100",
    },
    "librispeech_dev": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "sha256": "42e2234ba48799c1f50f24a7926300a7097f3132cdbb67df11c7f00411e031bf",
        "target_dir": "LibriSpeech/dev-clean",
    },
    "musan": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "sha256": "0c472d4fc0c5141eca47ad1ffeb2a7df2e1c4160153be67712ab6dbb8a364837",
        "target_dir": "musan",
    },
    "rir": {
        "url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "sha256": "e6f48e257286e05de56413b4779d8ff0ac50f0cfcfb5a124c6f63a1e24238f48",
        "target_dir": "RIRS_NOISES",
    },
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Progress(
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("download", name=out_path.name, total=0)

        def hook(block_count: int, block_size: int, total_size: int) -> None:
            progress.update(task, completed=block_count * block_size, total=total_size)

        urllib.request.urlretrieve(url, out_path.as_posix(), hook)  # noqa: S310


def _extract(archive_path: Path, dest_dir: Path) -> None:
    if archive_path.suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return

    mode = "r:gz" if archive_path.suffixes[-2:] == [".tar", ".gz"] else "r:*"
    with tarfile.open(archive_path, mode) as tf:
        tf.extractall(dest_dir)


def _verify_or_redownload(url: str, sha256: str, archive_path: Path) -> None:
    if archive_path.exists():
        existing = _sha256(archive_path)
        if existing == sha256:
            console.log(f"[green]Verified[/green] {archive_path.name}")
            return
        console.log(f"[yellow]Checksum mismatch; re-downloading {archive_path.name}[/yellow]")
        archive_path.unlink(missing_ok=True)

    _download(url, archive_path)
    actual = _sha256(archive_path)
    if actual != sha256:
        raise RuntimeError(f"Checksum mismatch for {archive_path.name}: {actual} != {sha256}")


def download_datasets(data_cache: str | Path = "./data_cache", selected: Iterable[str] | None = None) -> None:
    root = Path(data_cache)
    root.mkdir(parents=True, exist_ok=True)

    keys = list(DATASETS.keys()) if selected is None else list(selected)
    for key in keys:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset key: {key}")

        spec = DATASETS[key]
        target = root / spec["target_dir"]
        if target.exists() and (any(target.rglob("*.wav")) or any(target.rglob("*.flac"))):
            console.log(f"[green]Skipping[/green] {key}; found prepared files at {target}")
            continue

        url = spec["url"]
        archive_name = url.split("/")[-1]
        archive_path = root / archive_name
        console.log(f"Preparing dataset: [bold]{key}[/bold]")
        _verify_or_redownload(url, spec["sha256"], archive_path)
        _extract(archive_path, root)
        console.log(f"[green]Done[/green] {key}")


if __name__ == "__main__":
    download_datasets()
